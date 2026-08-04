# ==========================================================================
# ModelNew — SOL 002 VAE residual block (Conv3x3->GN->SiLU->Conv3x3->GN->SiLU->+x)
# HEADER:
# 1) GRANULARITY: (D) fully rewrite forward.
# 2) OPS REPLACED: both 3x3 convs (the vendor sm90_xmma implicit-GEMM kernels),
#    both group_norm, both silu, the residual add, and both cudnn nchw<->nhwc
#    layout kernels (deleted: we never leave NCHW).
# 3) FUSION MAP:
#    wt_kernel        : G g G^T Winograd F(2x2,3x3) filter transform -> U[16][C][K].
#    winograd_kernel  : THE conv. Winograd F(2x2,3x3), 16 mults/2x2 tile vs 36.
#                       B^T d B input transform computed in registers from a shared
#                       halo tile, kept in SHARED ONLY (never global); 16 transform
#                       -domain GEMMs on TF32 wmma tensor cores, one plane per warp
#                       (64x32 tile -> 8 acc fragments = 64 acc regs, the limit the
#                       16x accumulator multiplicity allows); A^T M A inverse
#                       transform fused in the epilogue out of shared memory.
#                       FUSED PROLOGUE (ACT, conv2): GroupNorm affine + SiLU of the
#                       previous conv output applied during halo load, so the
#                       normalized/activated intermediate never hits global memory.
#                       FUSED EPILOGUE (both): per-group sum/sumsq reduced in-kernel
#                       -> GroupNorm's statistics pass over the tensor is deleted.
#    gn_finalize      : deterministic tree reduce of partials -> per-(b,c)
#                       scale/shift (fp64 accumulation).
#    final_kernel     : GroupNorm2 affine + SiLU + residual add in one pass.
# 4) REMAINS IN PYTORCH: allocation + contiguity guard only. No cuDNN/at::conv2d.
# Precision: fp32 storage/accumulate everywhere; TF32 tensor cores only for the
# transform-domain products (the reference conv is TF32 too).
# ==========================================================================
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define TH_TILES 4
#define TW_TILES 8
#define P_TILE   32
#define K_TILE   64
#define CC       8
#define NTHREADS 512
#define IHH      (TH_TILES*2+2)
#define IWW      (TW_TILES*2+2)
#define IWP      20
#define CPG      8
#define NG       32

__global__ void wt_kernel(const float* __restrict__ w, float* __restrict__ U, int K, int C){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= K*C) return;
    int k = idx % K, c = idx / K;
    const float* g = w + ((size_t)k*C + c)*9;
    float R[4][3];
#pragma unroll
    for (int j=0;j<3;++j){ float a=g[j], b=g[3+j], d=g[6+j];
        R[0][j]=a; R[1][j]=0.5f*(a+b+d); R[2][j]=0.5f*(a-b+d); R[3][j]=d; }
#pragma unroll
    for (int i=0;i<4;++i){ float a=R[i][0], b=R[i][1], d=R[i][2];
        float u0=a, u1=0.5f*(a+b+d), u2=0.5f*(a-b+d), u3=d;
        size_t base = (size_t)c*K + k, pl = (size_t)C*K;
        U[(size_t)(i*4+0)*pl + base]=u0;
        U[(size_t)(i*4+1)*pl + base]=u1;
        U[(size_t)(i*4+2)*pl + base]=u2;
        U[(size_t)(i*4+3)*pl + base]=u3; }
}

__device__ __forceinline__ float silu_f(float v){ return v / (1.0f + __expf(-v)); }

template<bool ACT>
__global__ __launch_bounds__(NTHREADS) void winograd_kernel(
        const float* __restrict__ in, const float* __restrict__ U,
        const float* __restrict__ scale, const float* __restrict__ shift,
        float* __restrict__ out, float* __restrict__ psum, float* __restrict__ psq,
        int C, int H, int W, int nBW, int nSpatial){
    extern __shared__ float smem[];
    float* Us   = smem;
    float* Vs   = Us + 16*CC*K_TILE;
    float* Is   = Vs + 16*CC*P_TILE;
    float* Accs = smem;
    __shared__ float wsum[16];
    __shared__ float wsq[16];

    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
    const int KB = C / K_TILE;
    const int bz = blockIdx.z, kb = bz % KB, b = bz / KB;
    const int k0 = kb*K_TILE;
    const int h0 = (int)blockIdx.y*TH_TILES*2 - 1;
    const int w0 = (int)blockIdx.x*TW_TILES*2 - 1;
    const int spat = (int)blockIdx.y*nBW + (int)blockIdx.x;

    wmma::fragment<wmma::accumulator,16,16,8,float> acc[8];
#pragma unroll
    for (int i=0;i<8;++i) wmma::fill_fragment(acc[i], 0.0f);

    for (int c0=0; c0<C; c0+=CC){
        __syncthreads();
        for (int t=tid; t<CC*IHH*IWW; t+=NTHREADS){
            int ci = t/(IHH*IWW), r = (t%(IHH*IWW))/IWW, s = t%IWW;
            int hh = h0+r, ww = w0+s;
            float v = 0.0f;
            if (hh>=0 && hh<H && ww>=0 && ww<W){
                int c = c0+ci;
                v = in[(((size_t)b*C + c)*H + hh)*W + ww];
                if (ACT){ v = v*scale[b*C+c] + shift[b*C+c]; v = silu_f(v); }
            }
            Is[ci*IHH*IWP + r*IWP + s] = v;
        }
        __syncthreads();
        for (int t=tid; t<16*CC*K_TILE; t+=NTHREADS){
            int kk = t % K_TILE, rest = t / K_TILE, ci = rest % CC, xi = rest / CC;
            Us[(xi*CC+ci)*K_TILE + kk] = U[((size_t)xi*C + (c0+ci))*C + (k0+kk)];
        }
        for (int t=tid; t<CC*P_TILE; t+=NTHREADS){
            int ci = t / P_TILE, p = t % P_TILE;
            int ty = p / TW_TILES, tx = p % TW_TILES;
            const float* bp = Is + ci*IHH*IWP + (2*ty)*IWP + (2*tx);
            float d[4][4];
#pragma unroll
            for (int i=0;i<4;++i)
#pragma unroll
                for (int j=0;j<4;++j) d[i][j] = bp[i*IWP + j];
            float tc[4][4];
#pragma unroll
            for (int j=0;j<4;++j){
                tc[0][j] = d[0][j]-d[2][j];
                tc[1][j] = d[1][j]+d[2][j];
                tc[2][j] = d[2][j]-d[1][j];
                tc[3][j] = d[1][j]-d[3][j];
            }
            int off = ci*P_TILE + p;
#pragma unroll
            for (int i=0;i<4;++i){
                float a=tc[i][0], bb=tc[i][1], cc2=tc[i][2], dd=tc[i][3];
                Vs[(i*4+0)*CC*P_TILE + off] = a - cc2;
                Vs[(i*4+1)*CC*P_TILE + off] = bb + cc2;
                Vs[(i*4+2)*CC*P_TILE + off] = cc2 - bb;
                Vs[(i*4+3)*CC*P_TILE + off] = bb - dd;
            }
        }
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> af[4];
            wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bf[2];
            const float* up = Us + warp*CC*K_TILE;
            const float* vp = Vs + warp*CC*P_TILE;
#pragma unroll
            for (int m=0;m<4;++m){
                wmma::load_matrix_sync(af[m], up + m*16, K_TILE);
#pragma unroll
                for (int e=0;e<af[m].num_elements;++e) af[m].x[e] = wmma::__float_to_tf32(af[m].x[e]);
            }
#pragma unroll
            for (int n=0;n<2;++n){
                wmma::load_matrix_sync(bf[n], vp + n*16, P_TILE);
#pragma unroll
                for (int e=0;e<bf[n].num_elements;++e) bf[n].x[e] = wmma::__float_to_tf32(bf[n].x[e]);
            }
#pragma unroll
            for (int m=0;m<4;++m)
#pragma unroll
                for (int n=0;n<2;++n) wmma::mma_sync(acc[m*2+n], af[m], bf[n], acc[m*2+n]);
        }
    }

    // ---- epilogue: A^T M A inverse transform + GroupNorm partial statistics
    for (int q=0;q<4;++q){
        __syncthreads();
#pragma unroll
        for (int n=0;n<2;++n)
            wmma::store_matrix_sync(Accs + warp*16*P_TILE + n*16, acc[q*2+n], P_TILE, wmma::mem_row_major);
        __syncthreads();
        int kl = tid >> 5, p = tid & 31;
        float m[16];
#pragma unroll
        for (int xi=0; xi<16; ++xi) m[xi] = Accs[(xi*16 + kl)*P_TILE + p];
        float s0[4], s1[4];
#pragma unroll
        for (int j=0;j<4;++j){
            s0[j] = m[0*4+j] + m[1*4+j] + m[2*4+j];
            s1[j] = m[1*4+j] - m[2*4+j] - m[3*4+j];
        }
        float o[2][2];
        o[0][0] = s0[0]+s0[1]+s0[2];
        o[0][1] = s0[1]-s0[2]-s0[3];
        o[1][0] = s1[0]+s1[1]+s1[2];
        o[1][1] = s1[1]-s1[2]-s1[3];

        int kg = k0 + q*16 + kl;
        int ty = p / TW_TILES, tx = p % TW_TILES;
        int hb = ((int)blockIdx.y*TH_TILES + ty)*2;
        int wb = ((int)blockIdx.x*TW_TILES + tx)*2;
        float ls = 0.0f, lq = 0.0f;
#pragma unroll
        for (int i=0;i<2;++i){
            int hh = hb + i;
            if (hh >= H) continue;
#pragma unroll
            for (int j=0;j<2;++j){
                int ww = wb + j;
                if (ww >= W) continue;
                float v = o[i][j];
                out[(((size_t)b*C + kg)*H + hh)*W + ww] = v;
                ls += v; lq += v*v;
            }
        }
#pragma unroll
        for (int off=16; off>0; off>>=1){
            ls += __shfl_down_sync(0xffffffffu, ls, off);
            lq += __shfl_down_sync(0xffffffffu, lq, off);
        }
        if (lane == 0){ wsum[warp] = ls; wsq[warp] = lq; }
        __syncthreads();
        if (tid < 2){
            float as = 0.0f, aq = 0.0f;
            for (int wv = tid*8; wv < tid*8 + 8; ++wv){ as += wsum[wv]; aq += wsq[wv]; }
            int g = kb*(K_TILE/CPG) + q*2 + tid;
            psum[((size_t)b*NG + g)*nSpatial + spat] = as;
            psq [((size_t)b*NG + g)*nSpatial + spat] = aq;
        }
    }
}

__global__ __launch_bounds__(256) void gn_finalize(
        const float* __restrict__ psum, const float* __restrict__ psq,
        const float* __restrict__ gw, const float* __restrict__ gb,
        float* __restrict__ scale, float* __restrict__ shift,
        int nSpatial, int C, float eps, float cnt){
    __shared__ double rs[256];
    __shared__ double rq[256];
    int blk = blockIdx.x;
    int b = blk / NG, g = blk % NG;
    const float* ps = psum + (size_t)blk*nSpatial;
    const float* pq = psq  + (size_t)blk*nSpatial;
    double s = 0.0, q = 0.0;
    for (int i = threadIdx.x; i < nSpatial; i += 256){ s += (double)ps[i]; q += (double)pq[i]; }
    rs[threadIdx.x] = s; rq[threadIdx.x] = q;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1){
        if (threadIdx.x < off){ rs[threadIdx.x] += rs[threadIdx.x+off]; rq[threadIdx.x] += rq[threadIdx.x+off]; }
        __syncthreads();
    }
    double mean = rs[0] / (double)cnt;
    double var  = rq[0] / (double)cnt - mean*mean;
    if (var < 0.0) var = 0.0;
    float inv = (float)(1.0 / sqrt(var + (double)eps));
    if (threadIdx.x < CPG){
        int c = g*CPG + threadIdx.x;
        float w = gw[c], bb = gb[c];
        scale[b*C + c] = inv * w;
        shift[b*C + c] = bb - (float)mean * inv * w;
    }
}

__global__ void final_kernel(const float* __restrict__ y, const float* __restrict__ x,
                             const float* __restrict__ scale, const float* __restrict__ shift,
                             float* __restrict__ out, int C, int HW, long n){
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x*blockDim.x;
    for (; idx < n; idx += stride){
        int c = (int)((idx / HW) % C);
        int b = (int)(idx / ((long)HW*C));
        float v = y[idx]*scale[b*C+c] + shift[b*C+c];
        out[idx] = silu_f(v) + x[idx];
    }
}

static bool g_attr_set = false;
static void set_attrs(size_t sh){
    if (g_attr_set) return;
    cudaFuncSetAttribute((void*)winograd_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    cudaFuncSetAttribute((void*)winograd_kernel<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    g_attr_set = true;
}

torch::Tensor block_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                            torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b, double eps){
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "fp32 cuda input required");
    auto xc  = x.is_contiguous()  ? x  : x.contiguous();
    auto w1c = w1.is_contiguous() ? w1 : w1.contiguous();
    auto w2c = w2.is_contiguous() ? w2 : w2.contiguous();
    auto n1wc = n1w.is_contiguous()? n1w : n1w.contiguous();
    auto n1bc = n1b.is_contiguous()? n1b : n1b.contiguous();
    auto n2wc = n2w.is_contiguous()? n2w : n2w.contiguous();
    auto n2bc = n2b.is_contiguous()? n2b : n2b.contiguous();

    int B = xc.size(0), C = xc.size(1), H = xc.size(2), W = xc.size(3);
    TORCH_CHECK(C == 256, "kernel specialised for C=256 (32 groups x 8 ch)");
    int nTh = (H+1)/2, nTw = (W+1)/2;
    int nBH = (nTh + TH_TILES - 1)/TH_TILES;
    int nBW = (nTw + TW_TILES - 1)/TW_TILES;
    int nSpatial = nBH*nBW;
    int KB = C / K_TILE;

    auto opt = xc.options();
    auto U1 = torch::empty({16, (long)C, (long)C}, opt);
    auto U2 = torch::empty({16, (long)C, (long)C}, opt);
    auto y1 = torch::empty({B, C, H, W}, opt);
    auto y2 = torch::empty({B, C, H, W}, opt);
    auto out = torch::empty({B, C, H, W}, opt);
    auto ps  = torch::empty({(long)B*NG*nSpatial}, opt);
    auto pq  = torch::empty({(long)B*NG*nSpatial}, opt);
    auto sc1 = torch::empty({(long)B*C}, opt);
    auto sh1 = torch::empty({(long)B*C}, opt);
    auto sc2 = torch::empty({(long)B*C}, opt);
    auto sh2 = torch::empty({(long)B*C}, opt);

    auto stream = at::cuda::getDefaultCUDAStream();
    size_t shbytes = (size_t)(16*CC*K_TILE + 16*CC*P_TILE + CC*IHH*IWP) * sizeof(float);
    set_attrs(shbytes);

    int wt_threads = 256, wt_blocks = (C*C + wt_threads - 1)/wt_threads;
    wt_kernel<<<wt_blocks, wt_threads, 0, stream>>>(w1c.data_ptr<float>(), U1.data_ptr<float>(), C, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wt_kernel<<<wt_blocks, wt_threads, 0, stream>>>(w2c.data_ptr<float>(), U2.data_ptr<float>(), C, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid(nBW, nBH, B*KB), blk(NTHREADS);
    winograd_kernel<false><<<grid, blk, shbytes, stream>>>(
        xc.data_ptr<float>(), U1.data_ptr<float>(), nullptr, nullptr,
        y1.data_ptr<float>(), ps.data_ptr<float>(), pq.data_ptr<float>(),
        C, H, W, nBW, nSpatial);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float cnt = (float)((double)CPG * (double)H * (double)W);
    gn_finalize<<<B*NG, 256, 0, stream>>>(ps.data_ptr<float>(), pq.data_ptr<float>(),
        n1wc.data_ptr<float>(), n1bc.data_ptr<float>(), sc1.data_ptr<float>(), sh1.data_ptr<float>(),
        nSpatial, C, (float)eps, cnt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    winograd_kernel<true><<<grid, blk, shbytes, stream>>>(
        y1.data_ptr<float>(), U2.data_ptr<float>(), sc1.data_ptr<float>(), sh1.data_ptr<float>(),
        y2.data_ptr<float>(), ps.data_ptr<float>(), pq.data_ptr<float>(),
        C, H, W, nBW, nSpatial);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize<<<B*NG, 256, 0, stream>>>(ps.data_ptr<float>(), pq.data_ptr<float>(),
        n2wc.data_ptr<float>(), n2bc.data_ptr<float>(), sc2.data_ptr<float>(), sh2.data_ptr<float>(),
        nSpatial, C, (float)eps, cnt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    long n = (long)B*C*H*W;
    int fthreads = 256;
    int fblocks = (int)std::min((long)65535, (n + fthreads - 1)/fthreads);
    if (fblocks < 1) fblocks = 1;
    final_kernel<<<fblocks, fthreads, 0, stream>>>(y2.data_ptr<float>(), xc.data_ptr<float>(),
        sc2.data_ptr<float>(), sh2.data_ptr<float>(), out.data_ptr<float>(), C, H*W, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP = ("torch::Tensor block_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor n1w, "
        "torch::Tensor n1b, torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b, double eps);")

_ext = load_inline(
    name="winograd_vae_block",
    cpp_sources=_CPP,
    cuda_sources=_SRC,
    functions=["block_forward"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-gencode=arch=compute_90,code=sm_90",
    ],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps = float(eps.item())
        return self.ext.block_forward(x, conv1_weight, norm1_weight, norm1_bias,
                                      conv2_weight, norm2_weight, norm2_bias, float(eps))
