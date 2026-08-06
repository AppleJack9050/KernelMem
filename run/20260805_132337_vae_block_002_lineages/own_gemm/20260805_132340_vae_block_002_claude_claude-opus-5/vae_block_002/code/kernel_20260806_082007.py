# =============================================================================
# 002_vae_conv3x3_groupnorm_silu_residual_fused  --  ModelNew
# Round 7: atomic_privatize of the GroupNorm moment reduction.
# gn_partial_kernel + gn_finalize_kernel are replaced by ONE grid-filling
# kernel that privatizes (sum, sumsq) per group in shared memory, atomically
# combines into per-(n,g) accumulators, and lets the last-arriving block of an
# image finalize mean/rstd in double precision.  Everything else (CUTLASS
# conv, nchw2nhwc, gn_silu_nhwc, gn_silu_add_nhwc2nchw, batch chunking, the
# two worker streams, events and joins) is byte-identical to the base kernel.
# =============================================================================
import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


def _find_cutlass():
    cands = [
        os.environ.get("CUTLASS_PATH", ""),
        os.environ.get("CUTLASS_DIR", ""),
        "/home/otter77/git_project/KernelMem/third_party/cutlass",
        "/home/elek/KernelMem/third_party/cutlass",
        os.path.expanduser("~/KernelMem/third_party/cutlass"),
    ]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "include", "cutlass")):
            return os.path.join(c, "include")
        if c and os.path.isdir(os.path.join(c, "cutlass")):
            return c
    return "/home/otter77/git_project/KernelMem/third_party/cutlass/include"


CUTLASS_INC = _find_cutlass()

cuda_src = r'''
#include <cudaTypedefs.h>
using PFN_cuTensorMapEncodeTiled  = PFN_cuTensorMapEncodeTiled_v12000;
using PFN_cuTensorMapEncodeIm2col = PFN_cuTensorMapEncodeIm2col_v12000;

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// ---------------------------------------------------------------------------
// CUTLASS TF32 implicit-GEMM 3x3 convolution (NHWC activations, KRSC filters).
// UNCHANGED from the base kernel: 128x128x16 threadblock, 64x64x16 warp tile,
// m16n8k8 tensor-core MMA, fp32 accumulators.
// ---------------------------------------------------------------------------
using ConvKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    float, cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<ConvKernel>;

// Range form: operates on a sub-range of images given already-offset pointers.
static void conv3x3_nhwc_range(const float* a, const float* w, float* d,
                               int Nsub, int H, int W, int C, int K,
                               const torch::TensorOptions& opts,
                               cudaStream_t stream) {
  const int R = 3, S = 3;
  cutlass::conv::Conv2dProblemSize ps(Nsub, H, W, C, K, R, S, H, W,
                                      1, 1, 1, 1, 1, 1,
                                      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC> ra(
      const_cast<float*>(a), cutlass::layout::TensorNHWC::packed({Nsub, H, W, C}));
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC> rb(
      const_cast<float*>(w), cutlass::layout::TensorNHWC::packed({K, R, S, C}));
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC> rd(
      d, cutlass::layout::TensorNHWC::packed({Nsub, H, W, K}));
  typename ImplicitGemm::Arguments args{ps, ra, rb, rd, rd, {1.0f, 0.0f}};
  ImplicitGemm op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
              "cutlass conv: unsupported problem");
  size_t ws = op.get_workspace_size(args);
  void* wsptr = nullptr;
  torch::Tensor wsbuf;
  if (ws > 0) {
    wsbuf = torch::empty({(long)ws}, opts.dtype(torch::kUInt8));
    wsptr = wsbuf.data_ptr();
  }
  TORCH_CHECK(op.initialize(args, wsptr, stream) == cutlass::Status::kSuccess,
              "cutlass conv: initialize failed");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "cutlass conv: launch failed");
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_f(float v) {
  return v / (1.0f + __expf(-v));
}

#define TT 64  // transpose tile (pixels x channels)

// NCHW -> NHWC ---------------------------------------------------------------
__global__ __launch_bounds__(256) void nchw2nhwc_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int HW, int C, int vec) {
  __shared__ float sm[TT][TT + 1];   // [c_local][p_local]
  const int p0 = blockIdx.x * TT;
  const int c0 = blockIdx.y * TT;
  const int n  = blockIdx.z;
  const float* base = src + (size_t)n * (size_t)C * HW;
  float*      obase = dst + (size_t)n * (size_t)HW * C;
  const int t = threadIdx.x;
  const int lc = t >> 4;
  const int lp = (t & 15) * 4;
  const int lp2 = t >> 4;
  const int lc2 = (t & 15) * 4;
  const bool full = (vec != 0) && (p0 + TT <= HW);

  if (full) {   // interior tile: no bounds checks at all
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = lc + r * 16;
      float4 v = *reinterpret_cast<const float4*>(base + (size_t)(c0 + c) * HW + p0 + lp);
      sm[c][lp] = v.x; sm[c][lp + 1] = v.y; sm[c][lp + 2] = v.z; sm[c][lp + 3] = v.w;
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp2 + r * 16;
      float4 v;
      v.x = sm[lc2][p]; v.y = sm[lc2 + 1][p]; v.z = sm[lc2 + 2][p]; v.w = sm[lc2 + 3][p];
      *reinterpret_cast<float4*>(obase + (size_t)(p0 + p) * C + c0 + lc2) = v;
    }
  } else {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = lc + r * 16;
      const float* s = base + (size_t)(c0 + c) * HW + p0 + lp;
      #pragma unroll
      for (int j = 0; j < 4; ++j) sm[c][lp + j] = (p0 + lp + j < HW) ? s[j] : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp2 + r * 16;
      if (p0 + p >= HW) continue;
      float4 v;
      v.x = sm[lc2][p]; v.y = sm[lc2 + 1][p]; v.z = sm[lc2 + 2][p]; v.w = sm[lc2 + 3][p];
      *reinterpret_cast<float4*>(obase + (size_t)(p0 + p) * C + c0 + lc2) = v;
    }
  }
}

// ---------------------------------------------------------------------------
// GroupNorm moments -- ONE grid-filling kernel (atomic_privatize).
// Block (bx, n) privatizes (sum, sumsq) per group in shared memory over the
// pixel range [bx*pixPerBlock, min(HW,(bx+1)*pixPerBlock)), atomically folds
// its G group partials into acc[(n*G+g)*2 .. +1]; the LAST arriving block of
// image n (tracked by ctr[n]) finalizes mean/rstd for all G groups with the
// same double-precision formula the old gn_finalize_kernel used.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void gn_moments_kernel(
    const float* __restrict__ y, float* acc, int* __restrict__ ctr,
    float* __restrict__ mean, float* __restrict__ rstd,
    int HW, int C, int G, int Cg, int pixPerBlock, int blocksPerImage,
    long cnt, float eps) {
  const int tpp = C >> 2;                 // float4 lanes per pixel
  const int ppp = 256 / tpp;              // pixels handled per pass
  const int c4  = threadIdx.x % tpp;
  const int pr  = threadIdx.x / tpp;
  const int n   = blockIdx.y;
  const int p0  = blockIdx.x * pixPerBlock;
  const int p1  = min(HW, p0 + pixPerBlock);
  const float4* yb = reinterpret_cast<const float4*>(y + (size_t)n * (size_t)HW * C);

  float s = 0.f, ss = 0.f;
  for (int p = p0 + pr; p < p1; p += ppp) {
    float4 v = yb[(size_t)p * tpp + c4];
    s  += v.x + v.y + v.z + v.w;
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  __shared__ float sms[256], smss[256];
  __shared__ int s_last;
  sms[threadIdx.x] = s; smss[threadIdx.x] = ss;
  if (threadIdx.x == 0) s_last = 0;
  __syncthreads();

  if ((int)threadIdx.x < G) {             // privatized per-group fold
    const int g = threadIdx.x;
    const int c4lo = (g * Cg) >> 2;
    const int c4hi = ((g + 1) * Cg) >> 2;
    float ts = 0.f, tss = 0.f;
    for (int i = 0; i < ppp; ++i)
      for (int c = c4lo; c < c4hi; ++c) {
        int idx = i * tpp + c;
        ts += sms[idx]; tss += smss[idx];
      }
    const size_t o = ((size_t)n * G + g) * 2;
    atomicAdd(&acc[o], ts);
    atomicAdd(&acc[o + 1], tss);
  }
  __threadfence();
  __syncthreads();

  if (threadIdx.x == 0) {                 // arrival counter -> last block wins
    int old = atomicAdd(&ctr[n], 1);
    s_last = (old == blocksPerImage - 1) ? 1 : 0;
  }
  __syncthreads();

  if (s_last && (int)threadIdx.x < G) {   // subsumes the old finalize kernel
    const int g = threadIdx.x;
    volatile float* va = acc;
    const size_t o = ((size_t)n * G + g) * 2;
    double sum = (double)va[o];
    double sq  = (double)va[o + 1];
    double m = sum / (double)cnt;
    double v = sq / (double)cnt - m * m;
    if (v < 0.0) v = 0.0;
    mean[n * G + g] = (float)m;
    rstd[n * G + g] = (float)(1.0 / sqrt(v + (double)eps));
  }
}

// GroupNorm + SiLU, NHWC -> NHWC ---------------------------------------------
__global__ __launch_bounds__(256) void gn_silu_nhwc_kernel(
    const float* __restrict__ y, float* __restrict__ out,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    int HW, int C, int G, int Cg) {
  const int tpp = C >> 2;
  const int n = blockIdx.y;
  const float4* yb = reinterpret_cast<const float4*>(y + (size_t)n * (size_t)HW * C);
  float4* ob = reinterpret_cast<float4*>(out + (size_t)n * (size_t)HW * C);
  const size_t total = (size_t)HW * tpp;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (size_t)gridDim.x * blockDim.x) {
    const int c4 = (int)(i % (size_t)tpp);
    const int c = c4 * 4;
    const int g = c / Cg;
    const float mu = mean[n * G + g], rs = rstd[n * G + g];
    float4 v = yb[i];
    float4 gm = reinterpret_cast<const float4*>(gamma)[c4];
    float4 bt = reinterpret_cast<const float4*>(beta)[c4];
    float4 o;
    o.x = silu_f((v.x - mu) * rs * gm.x + bt.x);
    o.y = silu_f((v.y - mu) * rs * gm.y + bt.y);
    o.z = silu_f((v.z - mu) * rs * gm.z + bt.z);
    o.w = silu_f((v.w - mu) * rs * gm.w + bt.w);
    ob[i] = o;
  }
}

// GroupNorm + SiLU + residual + NHWC -> NCHW ---------------------------------
__global__ __launch_bounds__(256) void gn_silu_add_nhwc2nchw_kernel(
    const float* __restrict__ y, const float* __restrict__ res, float* __restrict__ out,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    int HW, int C, int G, int Cg, int vec) {
  __shared__ float sm[TT][TT + 1];   // [c_local][p_local]
  const int p0 = blockIdx.x * TT;
  const int c0 = blockIdx.y * TT;
  const int n  = blockIdx.z;
  const float* yb = y + (size_t)n * (size_t)HW * C;
  const int t = threadIdx.x;

  // phase 1: read NHWC, normalize + affine + silu, stage transposed in shared
  const int lp = t >> 4;             // pixel within tile
  const int lc = (t & 15) * 4;       // channel within tile
  const int c  = c0 + lc;
  const int g  = c / Cg;
  const float mu = mean[n * G + g], rs = rstd[n * G + g];
  const float4 gm = reinterpret_cast<const float4*>(gamma)[c >> 2];
  const float4 bt = reinterpret_cast<const float4*>(beta)[c >> 2];

  const int lc2 = t >> 4;
  const int lp2 = (t & 15) * 4;
  const size_t nbase = (size_t)n * (size_t)C * HW;
  const bool full = (vec != 0) && (p0 + TT <= HW);

  if (full) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp + r * 16;
      float4 v = *reinterpret_cast<const float4*>(yb + (size_t)(p0 + p) * C + c);
      sm[lc + 0][p] = silu_f((v.x - mu) * rs * gm.x + bt.x);
      sm[lc + 1][p] = silu_f((v.y - mu) * rs * gm.y + bt.y);
      sm[lc + 2][p] = silu_f((v.z - mu) * rs * gm.z + bt.z);
      sm[lc + 3][p] = silu_f((v.w - mu) * rs * gm.w + bt.w);
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int cc = lc2 + r * 16;
      const size_t off = nbase + (size_t)(c0 + cc) * HW + p0 + lp2;
      float4 rv = *reinterpret_cast<const float4*>(res + off);
      float4 o;
      o.x = sm[cc][lp2 + 0] + rv.x;
      o.y = sm[cc][lp2 + 1] + rv.y;
      o.z = sm[cc][lp2 + 2] + rv.z;
      o.w = sm[cc][lp2 + 3] + rv.w;
      *reinterpret_cast<float4*>(out + off) = o;
    }
  } else {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp + r * 16;
      if (p0 + p >= HW) continue;
      float4 v = *reinterpret_cast<const float4*>(yb + (size_t)(p0 + p) * C + c);
      sm[lc + 0][p] = silu_f((v.x - mu) * rs * gm.x + bt.x);
      sm[lc + 1][p] = silu_f((v.y - mu) * rs * gm.y + bt.y);
      sm[lc + 2][p] = silu_f((v.z - mu) * rs * gm.z + bt.z);
      sm[lc + 3][p] = silu_f((v.w - mu) * rs * gm.w + bt.w);
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int cc = lc2 + r * 16;
      const size_t off = nbase + (size_t)(c0 + cc) * HW + p0 + lp2;
      if (vec && p0 + lp2 + 3 < HW) {
        float4 rv = *reinterpret_cast<const float4*>(res + off);
        float4 o;
        o.x = sm[cc][lp2 + 0] + rv.x;
        o.y = sm[cc][lp2 + 1] + rv.y;
        o.z = sm[cc][lp2 + 2] + rv.z;
        o.w = sm[cc][lp2 + 3] + rv.w;
        *reinterpret_cast<float4*>(out + off) = o;
      } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          if (p0 + lp2 + j < HW) out[off + j] = sm[cc][lp2 + j] + res[off + j];
      }
    }
  }
}

// KCRS -> KRSC filter permutation --------------------------------------------
__global__ __launch_bounds__(256) void kcrs2krsc_kernel(
    const float* __restrict__ w, float* __restrict__ o, int C, int RS) {
  extern __shared__ float sm[];
  const int k = blockIdx.x;
  const float* wb = w + (size_t)k * C * RS;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float* s = wb + (size_t)c * RS;
    for (int j = 0; j < RS; ++j) sm[j * C + c] = s[j];
  }
  __syncthreads();
  float* ob = o + (size_t)k * C * RS;
  for (int i = threadIdx.x; i < C * RS; i += blockDim.x) ob[i] = sm[i];
}

// ---------------------------------------------------------------------------
// host driver
// ---------------------------------------------------------------------------
static inline torch::Tensor to_nhwc_weight(const torch::Tensor& w, cudaStream_t stream) {
  const int K = (int)w.size(0), C = (int)w.size(1);
  const int R = (int)w.size(2), S = (int)w.size(3);
  const int RS = R * S;
  const size_t smem = (size_t)C * RS * sizeof(float);
  if (smem <= 48 * 1024) {
    auto o = torch::empty({K, R, S, C}, w.options());
    kcrs2krsc_kernel<<<K, 256, smem, stream>>>(w.data_ptr<float>(), o.data_ptr<float>(), C, RS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
  }
  return w.permute({0, 2, 3, 1}).contiguous();
}

// range form: pointers already offset to image n0; ONE launch per call now.
static void group_moments_range(const float* y, float* acc, int* ctr,
                                float* mean, float* rstd,
                                int nSub, int HW, int C, int G,
                                int blocksPerImage, int pixPerBlock,
                                float eps, cudaStream_t stream) {
  const int Cg = C / G;
  dim3 grid(blocksPerImage, nSub);
  gn_moments_kernel<<<grid, 256, 0, stream>>>(
      y, acc, ctr, mean, rstd, HW, C, G, Cg, pixPerBlock, blocksPerImage,
      (long)HW * Cg, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static inline void record_stream_(const torch::Tensor& t, c10::cuda::CUDAStream s) {
  c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), s);
}

torch::Tensor fused_resblock(torch::Tensor x, torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight, torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight, torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "x must be cuda float32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  auto xc = x.is_contiguous() ? x : x.contiguous();
  const int N = (int)xc.size(0), C = (int)xc.size(1);
  const int H = (int)xc.size(2), W = (int)xc.size(3);
  const int HW = H * W;
  const int G = 32;
  TORCH_CHECK(C % G == 0 && (C / G) % 4 == 0 && C % TT == 0 && (C / 4) <= 256 &&
                  256 % (C / 4) == 0,
              "unsupported channel count");

  auto main_stream = at::cuda::getCurrentCUDAStream();
  auto opts = xc.options();

  auto w1 = to_nhwc_weight(conv1_weight.is_contiguous() ? conv1_weight
                                                        : conv1_weight.contiguous(), main_stream);
  auto w2 = to_nhwc_weight(conv2_weight.is_contiguous() ? conv2_weight
                                                        : conv2_weight.contiguous(), main_stream);
  auto g1 = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
  auto b1 = norm1_bias.is_contiguous() ? norm1_bias : norm1_bias.contiguous();
  auto g2 = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
  auto b2 = norm2_bias.is_contiguous() ? norm2_bias : norm2_bias.contiguous();

  // ---- buffers, allocated ONCE for the full batch on the main stream -------
  auto xh   = torch::empty({N, H, W, C}, opts);   // NHWC staging of x, later z1
  auto y1   = torch::empty({N, H, W, C}, opts);   // conv output, later y2
  auto mean = torch::empty({N * G}, opts);
  auto rstd = torch::empty({N * G}, opts);
  auto out  = torch::empty({N, C, H, W}, opts);

  // ---- atomic_privatize scratch: acc (2 stages x N*G x {sum,sumsq}) plus the
  // per-(stage,image) arrival counters, in ONE allocation so a single
  // cudaMemsetAsync on the main stream zeroes both before the workers start.
  const long accFloats = (long)2 * N * G * 2;
  const long ctrInts   = (long)2 * N;
  auto scratch = torch::empty({accFloats + ctrInts}, opts);
  float* accp = scratch.data_ptr<float>();
  int*   ctrp = reinterpret_cast<int*>(accp + accFloats);
  C10_CUDA_CHECK(cudaMemsetAsync(accp, 0,
                                 (size_t)(accFloats + ctrInts) * sizeof(float),
                                 main_stream.stream()));

  const int vec = (HW % 4 == 0) ? 1 : 0;
  const int Cg = C / G;
  const float epsf = (float)eps;

  // ---- chunk sizing gate (anti-regression) --------------------------------
  // CTAs of the CUTLASS conv for a chunk of c images:
  //   ceil(c*HW/128) tiles along GEMM-M  x  ceil(C/128) tiles along GEMM-N
  const long tilesN = (C + 127) / 128;
  int chunkN = 0;
  for (int c = 1; c <= N; ++c) {
    long ctas = (((long)c * HW + 127) / 128) * tilesN;
    if (ctas >= 170) { chunkN = c; break; }
  }
  int chunks = (chunkN > 0) ? ((N + chunkN - 1) / chunkN) : 1;
  const bool pipelined = (chunkN > 0 && chunks >= 2);
  if (!pipelined) { chunkN = N; chunks = 1; }

  // ---- GN moment grid sizing: grid-filling, and NO empty block -------------
  const int tpp = C >> 2;
  const int ppp = 256 / tpp;               // pixels per pass per block
  const int nSubMax = (chunkN < N) ? chunkN : N;
  int bpi = (int)(((long)HW + (long)ppp * 4 - 1) / ((long)ppp * 4));
  const int bcap = (680 + max(nSubMax, 1) - 1) / max(nSubMax, 1);
  if (bpi > bcap) bpi = bcap;
  if (bpi < 1) bpi = 1;
  int pixPerBlock = (HW + bpi - 1) / bpi;
  if (pixPerBlock < 1) pixPerBlock = 1;
  const int blocksPerImage = (HW + pixPerBlock - 1) / pixPerBlock;

  // ---- worker streams (created once, cached) ------------------------------
  static c10::cuda::CUDAStream worker[2] = {at::cuda::getStreamFromPool(),
                                            at::cuda::getStreamFromPool()};

  if (pipelined) {
    // weights repacked + buffers allocated/zeroed on the main stream: both
    // workers must wait for that before touching them.
    at::cuda::CUDAEvent start_ev;
    start_ev.record(main_stream);
    start_ev.block(worker[0]);
    start_ev.block(worker[1]);
    for (int i = 0; i < 2; ++i) {
      record_stream_(xh, worker[i]);      record_stream_(y1, worker[i]);
      record_stream_(mean, worker[i]);    record_stream_(rstd, worker[i]);
      record_stream_(scratch, worker[i]); record_stream_(out, worker[i]);
      record_stream_(w1, worker[i]);      record_stream_(w2, worker[i]);
    }
  }

  const float* xcp = xc.data_ptr<float>();
  float* xhp = xh.data_ptr<float>();
  float* y1p = y1.data_ptr<float>();
  float* meanp = mean.data_ptr<float>();
  float* rstdp = rstd.data_ptr<float>();
  float* outp = out.data_ptr<float>();
  const float* w1p = w1.data_ptr<float>();
  const float* w2p = w2.data_ptr<float>();
  const float* g1p = g1.data_ptr<float>();
  const float* b1p = b1.data_ptr<float>();
  const float* g2p = g2.data_ptr<float>();
  const float* b2p = b2.data_ptr<float>();

  float* acc0 = accp;                       // stage-1 slot
  float* acc1 = accp + (long)N * G * 2;     // stage-2 slot
  int*   ctr0 = ctrp;
  int*   ctr1 = ctrp + N;

  for (int k = 0; k < chunks; ++k) {
    const int n0 = k * chunkN;
    const int nSub = min(chunkN, N - n0);
    if (nSub <= 0) break;

    c10::cuda::CUDAStream s = pipelined ? worker[k & 1] : main_stream;
    c10::cuda::CUDAStreamGuard guard(s);
    cudaStream_t cs = s.stream();

    const size_t offNHWC = (size_t)n0 * (size_t)HW * C;
    const size_t offNCHW = (size_t)n0 * (size_t)C * HW;
    const size_t offNG   = (size_t)n0 * G;

    dim3 tgrid((HW + TT - 1) / TT, C / TT, nSub);
    nchw2nhwc_kernel<<<tgrid, 256, 0, cs>>>(xcp + offNCHW, xhp + offNHWC, HW, C, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    conv3x3_nhwc_range(xhp + offNHWC, w1p, y1p + offNHWC, nSub, H, W, C, C, opts, cs);

    group_moments_range(y1p + offNHWC, acc0 + offNG * 2, ctr0 + n0,
                        meanp + offNG, rstdp + offNG,
                        nSub, HW, C, G, blocksPerImage, pixPerBlock, epsf, cs);

    {   // z1 == xh slice (staged NHWC copy of this chunk's x is dead by now)
      int blocks = (int)(((size_t)HW * (C / 4) + 255) / 256);
      if (blocks > 4096) blocks = 4096;
      dim3 gr(blocks, nSub);
      gn_silu_nhwc_kernel<<<gr, 256, 0, cs>>>(
          y1p + offNHWC, xhp + offNHWC, meanp + offNG, rstdp + offNG,
          g1p, b1p, HW, C, G, Cg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // y2 == y1 slice (conv1 output of this chunk is dead by now)
    conv3x3_nhwc_range(xhp + offNHWC, w2p, y1p + offNHWC, nSub, H, W, C, C, opts, cs);

    group_moments_range(y1p + offNHWC, acc1 + offNG * 2, ctr1 + n0,
                        meanp + offNG, rstdp + offNG,
                        nSub, HW, C, G, blocksPerImage, pixPerBlock, epsf, cs);

    gn_silu_add_nhwc2nchw_kernel<<<tgrid, 256, 0, cs>>>(
        y1p + offNHWC, xcp + offNCHW, outp + offNCHW,
        meanp + offNG, rstdp + offNG, g2p, b2p, HW, C, G, Cg, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (pipelined) {   // join back onto the caller's stream
    for (int i = 0; i < 2; ++i) {
      at::cuda::CUDAEvent done;
      done.record(worker[i]);
      done.block(main_stream);
    }
  }
  return out;
}
'''

cpp_src = r'''
torch::Tensor fused_resblock(torch::Tensor x, torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight, torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight, torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias, double eps);
'''

_ext = load_inline(
    name="vae_resblock_fused_atomicpriv",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_resblock"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo",
        "-gencode=arch=compute_120,code=sm_120",
        "-I" + CUTLASS_INC,
    ],
)


class ModelNew(nn.Module):
    # =====================================================================
    # GRANULARITY: (D) full forward rewrite -- the whole residual block runs
    # inside one extension call and the convolution is our own CUTLASS TF32
    # tiled implicit GEMM (no cuDNN / at::conv2d anywhere).  This round
    # atomic-privatizes the GroupNorm moment reduction (one grid-filling
    # kernel instead of partial+finalize) while keeping the batch-chunked
    # two-stream pipelining of the base kernel.
    # This module is stateless (the reference is stateless too): all weights
    # arrive as forward() arguments, so there is nothing to hold and no
    # parameter parity to break.
    # =====================================================================
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        return self.ext.fused_resblock(x, conv1_weight, norm1_weight, norm1_bias,
                                       conv2_weight, norm2_weight, norm2_bias, float(eps))
