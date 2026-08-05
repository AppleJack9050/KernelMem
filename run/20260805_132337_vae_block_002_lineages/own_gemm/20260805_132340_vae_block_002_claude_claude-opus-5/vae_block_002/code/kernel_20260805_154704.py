# =====================================================================================
# ModelNew — fused VAE residual block (Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +x)
#
# 1) GRANULARITY: (D) full forward rewrite. The vendor conv is OWNED: both 3x3
#    convolutions are my own tiled implicit-GEMM CUDA kernel built on TF32 tensor-core
#    mma.sync.m16n8k8 (the same instruction class cuDNN's sm80_xmma_fprop kernel issues
#    on this sm_120 part). No cuDNN / at::conv2d / at::cudnn_convolution anywhere.
#
# 2) OPS REPLACED: F.conv2d (x2), F.group_norm (x2, incl. RowwiseMoments), F.silu (x2),
#    the residual add, and the NCHW<->NHWC layout conversions cuDNN needed (deleted
#    outright: my kernel consumes NCHW directly, so those 4 nchwToNhwc passes vanish).
#
# 3) FUSION MAP:
#    - conv_gn_kernel  : implicit GEMM (M=pixels, N=out-channels, K=Cin*9) with a
#                        halo-tiled shared-memory input patch (8x32 output tile + 1 px
#                        halo, 64 out-channels, 8 in-channels/stage) and register
#                        software prefetch of the next K-stage. Its EPILOGUE computes
#                        the per-(batch,group) sum / sum-of-squares of the conv output
#                        while the tile is still in registers -> the GroupNorm reduction
#                        pass over the conv output (a full extra read of 67 MB) is gone.
#                        fp32->TF32 conversion happens in-register, so cuDNN's separate
#                        convertTensor read+write pass is gone too.
#    - gn_finalize     : deterministic tree reduction of the per-tile partials (double
#                        accumulate) -> per-(b,c) affine pair scale=gamma*rstd,
#                        shift=beta-mean*scale. GN normalize+affine collapses to one FMA.
#    - act_add_vec4/scalar : applies that FMA + SiLU (fused), and on the last call also
#                        the residual add, vectorized float4 when H*W%4==0.
#                        So GN+SiLU (+add) is ONE pass instead of moments/normalize/
#                        silu/add kernels.
#    - wtrans          : one-time 3x3 weight relayout to [rs][Cin][Cout] + TF32 rounding
#                        so the mainloop's B-fragment loads are fully coalesced/vectorized.
#    Kernel launches per forward: 2 wtrans + conv + finalize + act + conv + finalize + act.
#
# 4) WHAT STAYS IN PYTORCH: nothing on the compute path. Only tensor allocation
#    (torch::empty) and a PyTorch fallback path for configurations the kernel does not
#    support (non-fp32/CPU/channels not divisible as C/32==8), which the benchmark never
#    hits — it exists purely so an unexpected shape degrades instead of failing.
#
# PRECISION: fp32 storage everywhere, fp32 accumulation in the mma and fp32/double in all
# reductions; TF32 (cvt.rna, same rounding cuDNN uses) only for the tensor-core operands,
# matching the reference's own TF32 conv path. No narrower dtype is ever used.
# =====================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define TH 8
#define TW 32
#define BM (TH*TW)
#define BN 64
#define BC 8
#define ROWW (TW+2)
#define ROWH (TH+2)
#define APATCH (ROWH*ROWW)
#define APLANE 344      /* padded plane stride: (mod 32)==24 -> conflict-free A frags */
#define BNP 72          /* padded n stride:    (mod 32)==8  -> conflict-free B frags */
#define NTHREADS 256

__device__ __forceinline__ unsigned tf32b(float f){
  unsigned r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(f));
  return r;
}

/* weight relayout (K,C,3,3) -> [rs][C][K], rounded to TF32 once */
__global__ void wtrans_kernel(const float* __restrict__ w, unsigned* __restrict__ wt, int K, int C){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= K*C) return;
  int c = i / K;
  int k = i - c*K;
  const float* src = w + (size_t)(k*C + c)*9;
  #pragma unroll
  for (int rs = 0; rs < 9; ++rs)
    wt[(size_t)(rs*C + c)*K + k] = tf32b(src[rs]);
}

template<bool ACT, int OCC>
__global__ __launch_bounds__(NTHREADS, OCC)
void conv_gn_kernel(const float* __restrict__ inp,
                    const unsigned* __restrict__ wt,
                    float* __restrict__ out,
                    float* __restrict__ psum,
                    float* __restrict__ psumsq,
                    const float* __restrict__ scale,
                    const float* __restrict__ shift,
                    int C, int K, int H, int W, int TWt, int Ptiles)
{
  __shared__ unsigned sh_a[BC*APLANE];
  __shared__ __align__(16) unsigned sh_w[9*BC*BNP];
  __shared__ float red[8][4][2];
  __shared__ float sh_sc[256];
  __shared__ float sh_sh[256];

  const int ptile = blockIdx.x;
  const int ntile = blockIdx.y;
  const int b     = blockIdx.z;
  const int th = ptile / TWt;
  const int tw = ptile - th*TWt;
  const int h0 = th*TH;
  const int w0 = tw*TW;
  const int n0 = ntile*BN;

  const int t    = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const int gid  = lane >> 2;
  const int tid4 = lane & 3;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;

  float acc[4][4][4];
  #pragma unroll
  for (int mi=0; mi<4; ++mi)
    #pragma unroll
    for (int ni=0; ni<4; ++ni)
      #pragma unroll
      for (int q=0; q<4; ++q) acc[mi][ni][q] = 0.f;

  // A-fragment smem base offsets per m-tile (before r,s offsets)
  int abase[4];
  #pragma unroll
  for (int mi=0; mi<4; ++mi){
    int pb = warp_m*64 + mi*16;
    int y  = pb >> 5;
    int x  = (pb & 31) + gid;
    abase[mi] = tid4*APLANE + y*ROWW + x;
  }
  int bbase[4];
  #pragma unroll
  for (int ni=0; ni<4; ++ni)
    bbase[ni] = tid4*BNP + (warp_n*32 + ni*8 + gid);

  const float* inp_b = inp + (size_t)b*C*H*W;

  // ---- per-thread load mapping (constant across the k-loop) ----
  const int lrow = t >> 5;          // 0..7  : patch row group
  const int lcol = t & 31;          // 0..31 : patch column
  const int wu   = t >> 4;          // 0..15 : (r,s,c) pair group
  const int wnn  = (t & 15) * 4;    // weight n index (vec4)
  const int exr  = t >> 1;          // extra patch row (t < 160)
  const int exc  = 32 + (t & 1);    // extra patch column

  if (ACT) {
    for (int i = t; i < C; i += NTHREADS) {
      sh_sc[i] = scale[b*C + i];
      sh_sh[i] = shift[b*C + i];
    }
    __syncthreads();
  }

  float aReg[11];
  uint4 wReg[5];

#define LOAD_CHUNK(cbase)                                                            \
  {                                                                                  \
    const int _c0 = (cbase);                                                         \
    _Pragma("unroll")                                                                \
    for (int q = 0; q < 10; ++q) {                                                   \
      int row = q*8 + lrow;                                                          \
      int cc  = row / 10;                                                            \
      int i   = row - cc*10;                                                         \
      int ih  = h0 - 1 + i;                                                          \
      int iw  = w0 - 1 + lcol;                                                       \
      aReg[q] = (ih >= 0 && ih < H && iw >= 0 && iw < W)                             \
                ? inp_b[((size_t)(_c0+cc)*H + ih)*W + iw] : 0.f;                     \
    }                                                                                \
    if (t < 160) {                                                                   \
      int cc = exr / 10;                                                             \
      int i  = exr - cc*10;                                                          \
      int ih = h0 - 1 + i;                                                           \
      int iw = w0 - 1 + exc;                                                         \
      aReg[10] = (ih >= 0 && ih < H && iw >= 0 && iw < W)                            \
                 ? inp_b[((size_t)(_c0+cc)*H + ih)*W + iw] : 0.f;                    \
    }                                                                                \
    _Pragma("unroll")                                                                \
    for (int q = 0; q < 5; ++q) {                                                    \
      int v  = wu + 16*q;                                                            \
      if (v < 72) {                                                                  \
        int rs = v >> 3;                                                             \
        int cc = v & 7;                                                              \
        wReg[q] = *(const uint4*)(wt + (size_t)(rs*C + _c0 + cc)*K + n0 + wnn);      \
      }                                                                              \
    }                                                                                \
  }

#define STORE_CHUNK                                                                  \
  {                                                                                  \
    _Pragma("unroll")                                                                \
    for (int q = 0; q < 10; ++q) {                                                   \
      int row = q*8 + lrow;                                                          \
      int cc  = row / 10;                                                            \
      int i   = row - cc*10;                                                         \
      float v = aReg[q];                                                             \
      if (ACT) {                                                                     \
        int ih = h0 - 1 + i;                                                         \
        int iw = w0 - 1 + lcol;                                                      \
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {                                \
          float u = sh_sc[cur_c0 + cc]*v + sh_sh[cur_c0 + cc];                       \
          v = __fdividef(u, 1.f + __expf(-u));                                       \
        } else v = 0.f;                                                              \
      }                                                                              \
      sh_a[cc*APLANE + i*ROWW + lcol] = tf32b(v);                                    \
    }                                                                                \
    if (t < 160) {                                                                   \
      int cc = exr / 10;                                                             \
      int i  = exr - cc*10;                                                          \
      float v = aReg[10];                                                            \
      if (ACT) {                                                                     \
        int ih = h0 - 1 + i;                                                         \
        int iw = w0 - 1 + exc;                                                       \
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {                                \
          float u = sh_sc[cur_c0 + cc]*v + sh_sh[cur_c0 + cc];                       \
          v = __fdividef(u, 1.f + __expf(-u));                                       \
        } else v = 0.f;                                                              \
      }                                                                              \
      sh_a[cc*APLANE + i*ROWW + exc] = tf32b(v);                                     \
    }                                                                                \
    _Pragma("unroll")                                                                \
    for (int q = 0; q < 5; ++q) {                                                    \
      int v  = wu + 16*q;                                                            \
      if (v < 72) {                                                                  \
        int rs = v >> 3;                                                             \
        int cc = v & 7;                                                              \
        *(uint4*)(&sh_w[rs*(BC*BNP) + cc*BNP + wnn]) = wReg[q];                      \
      }                                                                              \
    }                                                                                \
  }

  LOAD_CHUNK(0)

  for (int cur_c0 = 0; cur_c0 < C; cur_c0 += BC) {
    __syncthreads();          // everyone finished reading the previous chunk
    STORE_CHUNK
    if (cur_c0 + BC < C) { LOAD_CHUNK(cur_c0 + BC) }   // prefetch; latency hidden by mma
    __syncthreads();

    #pragma unroll
    for (int rs = 0; rs < 9; ++rs) {
      const int r = rs / 3;
      const int s = rs - r*3;
      unsigned a[4][4];
      #pragma unroll
      for (int mi=0; mi<4; ++mi) {
        int o = abase[mi] + r*ROWW + s;
        a[mi][0] = sh_a[o];
        a[mi][1] = sh_a[o + 8];
        a[mi][2] = sh_a[o + 4*APLANE];
        a[mi][3] = sh_a[o + 4*APLANE + 8];
      }
      unsigned bfr[4][2];
      #pragma unroll
      for (int ni=0; ni<4; ++ni) {
        int o = rs*(BC*BNP) + bbase[ni];
        bfr[ni][0] = sh_w[o];
        bfr[ni][1] = sh_w[o + 4*BNP];
      }
      #pragma unroll
      for (int mi=0; mi<4; ++mi) {
        #pragma unroll
        for (int ni=0; ni<4; ++ni) {
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
              "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),
              "r"(bfr[ni][0]), "r"(bfr[ni][1]));
        }
      }
    }
  }

  // ---------------- epilogue: store + fused group moments ----------------
  float* out_b = out + (size_t)b*K*H*W;
  float gs[4], gq[4];
  #pragma unroll
  for (int ni=0; ni<4; ++ni) { gs[ni]=0.f; gq[ni]=0.f; }

  #pragma unroll
  for (int mi=0; mi<4; ++mi) {
    int pb = warp_m*64 + mi*16;
    int y  = pb >> 5;
    int x  = (pb & 31) + gid;
    int h  = h0 + y;
    int wA = w0 + x;
    int wB = w0 + x + 8;
    bool hv = (h < H);
    bool vA = hv && (wA < W);
    bool vB = hv && (wB < W);
    #pragma unroll
    for (int ni=0; ni<4; ++ni) {
      int nn = warp_n*32 + ni*8 + 2*tid4;
      size_t o0 = ((size_t)(n0+nn)*H + h)*W;
      float v0 = acc[mi][ni][0];
      float v1 = acc[mi][ni][1];
      float v2 = acc[mi][ni][2];
      float v3 = acc[mi][ni][3];
      if (vA) {
        out_b[o0 + wA]         = v0;
        out_b[o0 + H*W + wA]   = v1;
        gs[ni] += v0 + v1;
        gq[ni] += v0*v0 + v1*v1;
      }
      if (vB) {
        out_b[o0 + wB]         = v2;
        out_b[o0 + H*W + wB]   = v3;
        gs[ni] += v2 + v3;
        gq[ni] += v2*v2 + v3*v3;
      }
    }
  }

  #pragma unroll
  for (int ni=0; ni<4; ++ni) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      gs[ni] += __shfl_down_sync(0xffffffffu, gs[ni], off);
      gq[ni] += __shfl_down_sync(0xffffffffu, gq[ni], off);
    }
    if (lane == 0) { red[warp][ni][0] = gs[ni]; red[warp][ni][1] = gq[ni]; }
  }
  __syncthreads();
  if (t < 8) {                       // fixed-order (deterministic) cross-warp reduction
    int lg = t;                      // local group 0..7
    int wn = lg >> 2;
    int ni = lg & 3;
    float s = 0.f, q = 0.f;
    #pragma unroll
    for (int wm = 0; wm < 4; ++wm) {
      s += red[wm*2 + wn][ni][0];
      q += red[wm*2 + wn][ni][1];
    }
    int g = ntile*8 + lg;
    int G = K / 8;
    psum  [((size_t)b*G + g)*Ptiles + ptile] = s;
    psumsq[((size_t)b*G + g)*Ptiles + ptile] = q;
  }
}

__global__ void gn_finalize_kernel(const float* __restrict__ ps,
                                   const float* __restrict__ pss,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ scale,
                                   float* __restrict__ shift,
                                   int Ptiles, int G, int CPG, int C,
                                   float inv_count, float eps)
{
  __shared__ double sh[2][256];
  int idx = blockIdx.x;
  int b = idx / G;
  int g = idx - b*G;
  const float* p1 = ps  + (size_t)idx*Ptiles;
  const float* p2 = pss + (size_t)idx*Ptiles;
  double s = 0.0, q = 0.0;
  for (int i = threadIdx.x; i < Ptiles; i += blockDim.x) { s += p1[i]; q += p2[i]; }
  sh[0][threadIdx.x] = s;
  sh[1][threadIdx.x] = q;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (threadIdx.x < st) {
      sh[0][threadIdx.x] += sh[0][threadIdx.x + st];
      sh[1][threadIdx.x] += sh[1][threadIdx.x + st];
    }
    __syncthreads();
  }
  double mean = sh[0][0] * (double)inv_count;
  double var  = sh[1][0] * (double)inv_count - mean*mean;
  float rstd = rsqrtf((float)var + eps);
  float fmean = (float)mean;
  for (int j = threadIdx.x; j < CPG; j += blockDim.x) {
    int c = g*CPG + j;
    float gm = gamma[c];
    float sc = gm * rstd;
    scale[b*C + c] = sc;
    shift[b*C + c] = beta[c] - fmean * sc;
  }
}

template<bool ADD>
__global__ void act_add_vec4(const float* __restrict__ y, const float* __restrict__ x,
                             const float* __restrict__ scale, const float* __restrict__ shift,
                             float* __restrict__ o, int HW)
{
  int c = blockIdx.y, b = blockIdx.z, C = gridDim.y;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int n4 = HW >> 2;
  if (i >= n4) return;
  size_t base = ((size_t)b*C + c)*HW;
  float s = scale[b*C + c], t = shift[b*C + c];
  float4 yv = ((const float4*)(y + base))[i];
  float4 xv = make_float4(0.f,0.f,0.f,0.f);
  if (ADD) xv = ((const float4*)(x + base))[i];
  float4 ov;
  float u;
  u = s*yv.x + t; ov.x = __fdividef(u,1.f+__expf(-u)) + xv.x;
  u = s*yv.y + t; ov.y = __fdividef(u,1.f+__expf(-u)) + xv.y;
  u = s*yv.z + t; ov.z = __fdividef(u,1.f+__expf(-u)) + xv.z;
  u = s*yv.w + t; ov.w = __fdividef(u,1.f+__expf(-u)) + xv.w;
  ((float4*)(o + base))[i] = ov;
}

template<bool ADD>
__global__ void act_add_scalar(const float* __restrict__ y, const float* __restrict__ x,
                               const float* __restrict__ scale, const float* __restrict__ shift,
                               float* __restrict__ o, int HW)
{
  int c = blockIdx.y, b = blockIdx.z, C = gridDim.y;
  size_t base = ((size_t)b*C + c)*HW;
  float s = scale[b*C + c], t = shift[b*C + c];
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < HW; i += gridDim.x*blockDim.x) {
    float u = s*y[base+i] + t;
    float r = __fdividef(u,1.f+__expf(-u));
    if (ADD) r += x[base+i];
    o[base+i] = r;
  }
}

template<bool ADD>
static void launch_act(const float* y, const float* x, const float* sc, const float* sh,
                       float* o, int B, int C, int HW, cudaStream_t stream)
{
  if ((HW & 3) == 0) {
    int n4 = HW >> 2;
    dim3 grid((n4 + 255)/256, C, B);
    act_add_vec4<ADD><<<grid, 256, 0, stream>>>(y, x, sc, sh, o, HW);
  } else {
    dim3 grid(min((HW + 255)/256, 64), C, B);
    act_add_scalar<ADD><<<grid, 256, 0, stream>>>(y, x, sc, sh, o, HW);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_conv(const torch::Tensor& in, const torch::Tensor& wt, torch::Tensor& out,
                        torch::Tensor& ps, torch::Tensor& pss,
                        const float* scale, const float* shift,
                        int B, int C, int K, int H, int W, int TWt, int Ptiles, bool act,
                        cudaStream_t stream)
{
  dim3 grid(Ptiles, K/BN, B);
  int nblocks = Ptiles * (K/BN) * B;
  static int nsm = -1;
  if (nsm < 0) nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  bool occ2 = (nblocks >= 2*nsm);   // wide grid: 2 CTAs/SM; narrow grid: 1 CTA/SM (no spills)
#define LAUNCH(A, O) conv_gn_kernel<A, O><<<grid, NTHREADS, 0, stream>>>(                 \
      in.data_ptr<float>(), (const unsigned*)wt.data_ptr<int>(), out.data_ptr<float>(),   \
      ps.data_ptr<float>(), pss.data_ptr<float>(), scale, shift, C, K, H, W, TWt, Ptiles)
  (void)act;
  if (occ2) { LAUNCH(false, 2); } else { LAUNCH(false, 1); }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                             torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                             double eps)
{
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "x must be float32 cuda");
  auto xc  = x.is_contiguous()  ? x  : x.contiguous();
  auto w1c = w1.is_contiguous() ? w1 : w1.contiguous();
  auto w2c = w2.is_contiguous() ? w2 : w2.contiguous();
  auto g1c = g1.is_contiguous() ? g1 : g1.contiguous();
  auto b1c = b1.is_contiguous() ? b1 : b1.contiguous();
  auto g2c = g2.is_contiguous() ? g2 : g2.contiguous();
  auto b2c = b2.is_contiguous() ? b2 : b2.contiguous();

  int B = xc.size(0), C = xc.size(1), H = xc.size(2), W = xc.size(3);
  int K = w1c.size(0);
  TORCH_CHECK(K % BN == 0 && C % BC == 0, "channel counts unsupported");
  const int G = 32;
  const int CPG = C / G;
  TORCH_CHECK(CPG == BC, "channels per group must be 8");
  TORCH_CHECK(K == C, "in/out channels must match");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto fopt = xc.options();
  auto iopt = xc.options().dtype(torch::kInt32);

  int THt = (H + TH - 1) / TH;
  int TWt = (W + TW - 1) / TW;
  int Ptiles = THt * TWt;

  auto wt1 = torch::empty({9, C, K}, iopt);
  auto wt2 = torch::empty({9, C, K}, iopt);
  {
    int n = K*C;
    int thr = 256, blk = (n + thr - 1)/thr;
    wtrans_kernel<<<blk, thr, 0, stream>>>(w1c.data_ptr<float>(), (unsigned*)wt1.data_ptr<int>(), K, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wtrans_kernel<<<blk, thr, 0, stream>>>(w2c.data_ptr<float>(), (unsigned*)wt2.data_ptr<int>(), K, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto y1 = torch::empty({B, K, H, W}, fopt);
  auto y2 = torch::empty({B, K, H, W}, fopt);
  auto ps  = torch::empty({B, G, Ptiles}, fopt);
  auto pss = torch::empty({B, G, Ptiles}, fopt);
  auto sc1 = torch::empty({B, C}, fopt);
  auto sh1 = torch::empty({B, C}, fopt);
  auto sc2 = torch::empty({B, C}, fopt);
  auto sh2 = torch::empty({B, C}, fopt);

  float inv_count = 1.0f / (float)((double)H * (double)W * (double)CPG);

  launch_conv(xc, wt1, y1, ps, pss, nullptr, nullptr, B, C, K, H, W, TWt, Ptiles, false, stream);
  gn_finalize_kernel<<<B*G, 256, 0, stream>>>(ps.data_ptr<float>(), pss.data_ptr<float>(),
      g1c.data_ptr<float>(), b1c.data_ptr<float>(), sc1.data_ptr<float>(), sh1.data_ptr<float>(),
      Ptiles, G, CPG, C, inv_count, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  {
    auto y1a = torch::empty({B, K, H, W}, fopt);
    launch_act<false>(y1.data_ptr<float>(), nullptr, sc1.data_ptr<float>(), sh1.data_ptr<float>(),
                      y1a.data_ptr<float>(), B, C, H*W, stream);
    launch_conv(y1a, wt2, y2, ps, pss, nullptr, nullptr,
                B, C, K, H, W, TWt, Ptiles, false, stream);
  }
  gn_finalize_kernel<<<B*G, 256, 0, stream>>>(ps.data_ptr<float>(), pss.data_ptr<float>(),
      g2c.data_ptr<float>(), b2c.data_ptr<float>(), sc2.data_ptr<float>(), sh2.data_ptr<float>(),
      Ptiles, G, CPG, C, inv_count, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto out = torch::empty({B, K, H, W}, fopt);
  launch_act<true>(y2.data_ptr<float>(), xc.data_ptr<float>(), sc2.data_ptr<float>(),
                   sh2.data_ptr<float>(), out.data_ptr<float>(), B, C, H*W, stream);
  return out;
}
'''

cpp_src = r'''
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                             torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                             double eps);
'''

_ext = load_inline(
    name="fused_resblock_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_resblock"],
    verbose=True,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo",
                       "-gencode=arch=compute_120,code=sm_120"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def _supported(self, x, w1, w2):
        return (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == w1.size(0) == w1.size(1) == w2.size(0) == w2.size(1)
                and x.size(1) % 64 == 0 and x.size(1) // 32 == 8
                and w1.size(2) == 3 and w1.size(3) == 3
                and w2.size(2) == 3 and w2.size(3) == 3)

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if self._supported(x, conv1_weight, conv2_weight):
            return self.ext.fused_resblock(x, conv1_weight, norm1_weight, norm1_bias,
                                           conv2_weight, norm2_weight, norm2_bias, float(eps))
        # generic fallback (never taken by the benchmark shapes)
        o = F.conv2d(x, conv1_weight, None, 1, 1)
        o = F.group_norm(o, 32, norm1_weight, norm1_bias, eps)
        o = F.silu(o)
        o = F.conv2d(o, conv2_weight, None, 1, 1)
        o = F.group_norm(o, 32, norm2_weight, norm2_bias, eps)
        o = F.silu(o)
        return o + x
