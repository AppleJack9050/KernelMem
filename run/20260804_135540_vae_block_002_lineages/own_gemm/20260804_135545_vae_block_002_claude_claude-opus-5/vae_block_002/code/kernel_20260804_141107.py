# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED HEADER (required)
# 1) GRANULARITY: (D) FULL FORWARD REWRITE.
#    The whole reference forward (conv3x3 -> GN -> SiLU -> conv3x3 -> GN ->
#    SiLU -> +residual) is executed by kernels written here.  No cuDNN /
#    at::conv2d / at::group_norm anywhere: the 3x3 convolutions are my own
#    tiled implicit-GEMM using TF32 tensor cores (wmma 16x16x8, fp32 accumulate,
#    which is exactly what the reference's sm90_..._tf32f32_f32 kernel does).
#
# 2) OPS REPLACED: F.conv2d (x2), F.group_norm (x2), F.silu (x2), residual add.
#    Also deleted entirely: cuDNN's nchwToNhwc / nhwcToNchw layout-conversion
#    kernels (24.7% of reference GPU time) — my implicit GEMM consumes NCHW
#    directly and never materialises a transposed copy.
#
# 3) FUSION MAP:
#    K1 conv3x3_gn_kernel<false> : implicit GEMM conv1 (M=out-ch, N=pixels,
#        K=Cin*9) -> writes y1 raw.
#    K2 gn_partial + K3 gn_finalize : GroupNorm(32 groups) statistics of y1 as
#        deterministic split reduction (fixed per-block slot, no atomics) and
#        folds (mean,rstd,gamma,beta) into per-(n,c) affine pair (a,b).
#    K
# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED HEADER (required)
# 1) GRANULARITY: (D) FULL FORWARD REWRITE. The vendor conv is OWNED here:
#    both 3x3 convolutions are my own tiled implicit GEMM on TF32 tensor cores
#    (wmma 16x16x8, fp32 accumulate = same math class as the reference's
#    sm90_..._f32f32_tf32f32_f32 kernel). No cuDNN / at::conv2d anywhere.
# 2) OPS REPLACED: F.conv2d x2, F.group_norm x2, F.silu x2, residual add,
#    plus cuDNN's nchwToNhwc/nhwcToNchw conversions (24.7% of ref time) are
#    deleted outright — the implicit GEMM consumes NCHW directly.
# 3) FUSION MAP:
#    K1 conv3x3_gn_kernel<false> : implicit-GEMM conv1 -> y1 (raw).
#    K2 gn_partial / K3 gn_finalize : GroupNorm stats of y1 via a deterministic
#       split reduction (each block owns one output slot, no atomics; slice
#       bounds are (total*i)/NS .. (total*(i+1))/NS so tails are always covered),
#       folded into per-(n,c) affine (a,b) = (rstd*gamma, beta-mean*rstd*gamma).
#    K4 conv3x3_gn_kernel<true> : implicit-GEMM conv2 whose PROLOGUE applies
#       norm1+SiLU to y1 while staging it into shared memory -> the normalized
#       activation NEVER round-trips through global memory (this is the point of
#       owning the conv). Zero padding is applied in the transformed domain.
#    K5/K6 gn stats of y2, K7 final_kernel : norm2 + SiLU + residual add in one
#       pass (reads y2 and x, writes out).
# 4) STILL IN PYTORCH: only tensor allocation (torch::empty) and shape logic.
#    Rationale: allocation is not GPU work; every FLOP/byte of the reference
#    forward is produced by the kernels above.
# PRECISION: fp32 storage everywhere, fp32 accumulate everywhere, TF32 only for
#    the conv multiplicands (permitted: the reference does the same).
# ==========================================================================
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define BM 64                 // output channels per block
#define TH 8                  // spatial tile rows
#define TW 32                 // spatial tile cols
#define BN (TH*TW)            // 256 pixels per block
#define BKC 8                 // input channels per K-chunk (wmma k = 8)
#define NTHREADS 256
#define NWARPS 8
#define WNW 4                 // warps along N
#define MPW 2                 // m-fragments per warp
#define NPW 4                 // n-fragments per warp
#define SBW (TW+2)
#define SBH (TH+2)
#define NSPLIT 32

__device__ __forceinline__ float silu_f(float v) {
    return v * (1.0f / (1.0f + __expf(-v)));
}

// ---------------------------------------------------------------------------
// Implicit-GEMM 3x3 convolution, pad=1, stride=1, NCHW in / NCHW out.
//   M = out channels (BM per block), N = pixels (BN per block), K = Cin*9.
//   Shared A : sA[rs][m][c]           (matrix_a, row_major, ld = BKC)
//   Shared B : sB[h][w][c]            (matrix_b, col_major, ld = BKC)
//     -> for tap (r,s) the fragment base is ((th+r)*SBW + tw + s)*BKC which is
//        always a multiple of 8 floats (32B aligned) for every r,s.
// APPLY_PRE fuses  silu(y*a + b)  into the shared-memory staging of the input.
// ---------------------------------------------------------------------------
template<bool APPLY_PRE>
__global__ __launch_bounds__(NTHREADS) void conv3x3_gn_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ wgt,
    float* __restrict__ outp,
    const float* __restrict__ pre_a,
    const float* __restrict__ pre_b,
    int C, int H, int W, int tilesW)
{
    __shared__ float sA[9 * BM * BKC];
    __shared__ float sB[SBH * SBW * BKC];
    __shared__ float sOut[NWARPS * 256];

    const int m0    = blockIdx.x * BM;
    const int tile  = blockIdx.y;
    const int nimg  = blockIdx.z;
    const int th0   = (tile / tilesW) * TH;
    const int tw0   = (tile % tilesW) * TW;
    const int HW    = H * W;
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int wm    = warp / WNW;
    const int wn    = warp % WNW;

    const float* inbase = inp + (size_t)nimg * C * HW;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[MPW][NPW];
#pragma unroll
    for (int i = 0; i < MPW; ++i)
#pragma unroll
        for (int j = 0; j < NPW; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int c0 = 0; c0 < C; c0 += BKC) {
        __syncthreads();

        // ---- stage weights: sA[rs][m][c] = w[m0+m][c0+c][r][s]
        for (int i = tid; i < BM * BKC; i += NTHREADS) {
            int mm = i / BKC;
            int cc = i - mm * BKC;
            const float* wp = wgt + ((size_t)(m0 + mm) * C + (c0 + cc)) * 9;
#pragma unroll
            for (int rs = 0; rs < 9; ++rs)
                sA[(rs * BM + mm) * BKC + cc] = wp[rs];
        }

        // ---- stage activations (with halo) : sB[h][w][c]
        for (int i = tid; i < SBH * SBW * BKC; i += NTHREADS) {
            int cc = i & (BKC - 1);
            int t  = i >> 3;
            int ww = t % SBW;
            int hh = t / SBW;
            int gh = th0 + hh - 1;
            int gw = tw0 + ww - 1;
            float v = 0.0f;
            if (gh >= 0 && gh < H && gw >= 0 && gw < W) {
                v = inbase[(size_t)(c0 + cc) * HW + (size_t)gh * W + gw];
                if (APPLY_PRE) {
                    int pidx = nimg * C + c0 + cc;
                    v = silu_f(v * pre_a[pidx] + pre_b[pidx]);
                }
            }
            sB[t * BKC + cc] = v;
        }
        __syncthreads();

#pragma unroll
        for (int rs = 0; rs < 9; ++rs) {
            const int r = rs / 3;
            const int s = rs - r * 3;

            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af[MPW];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> bf[NPW];

#pragma unroll
            for (int i = 0; i < MPW; ++i) {
                int mrow = wm * (MPW * 16) + i * 16;
                wmma::load_matrix_sync(af[i], &sA[(rs * BM + mrow) * BKC], BKC);
#pragma unroll
                for (int t = 0; t < af[i].num_elements; ++t)
                    af[i].x[t] = wmma::__float_to_tf32(af[i].x[t]);
            }
#pragma unroll
            for (int j = 0; j < NPW; ++j) {
                int nf   = wn * NPW + j;          // 16-pixel fragment id in tile
                int trow = nf / (TW / 16);
                int tcol = (nf % (TW / 16)) * 16;
                wmma::load_matrix_sync(bf[j], &sB[(((trow + r) * SBW) + (tcol + s)) * BKC], BKC);
#pragma unroll
                for (int t = 0; t < bf[j].num_elements; ++t)
                    bf[j].x[t] = wmma::__float_to_tf32(bf[j].x[t]);
            }
#pragma unroll
            for (int i = 0; i < MPW; ++i)
#pragma unroll
                for (int j = 0; j < NPW; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }

    // ---- epilogue: masked write of the raw conv result (NCHW)
#pragma unroll
    for (int i = 0; i < MPW; ++i) {
        int cbase = m0 + wm * (MPW * 16) + i * 16;
#pragma unroll
        for (int j = 0; j < NPW; ++j) {
            __syncwarp();
            wmma::store_matrix_sync(sOut + warp * 256, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            int nf   = wn * NPW + j;
            int trow = nf / (TW / 16);
            int tcol = (nf % (TW / 16)) * 16;
            int gh   = th0 + trow;
            if (gh < H) {
                for (int e = lane; e < 256; e += 32) {
                    int mm = e >> 4;
                    int nn = e & 15;
                    int gw = tw0 + tcol + nn;
                    if (gw < W)
                        outp[(size_t)(nimg * C + cbase + mm) * HW + (size_t)gh * W + gw] =
                            sOut[warp * 256 + e];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GroupNorm statistics, deterministic split reduction (no atomics).
// A group's channels are contiguous in NCHW, so each (n,g) slice is contiguous.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void gn_partial(
    const float* __restrict__ y, float* __restrict__ psum, float* __restrict__ psq,
    int C, int G, int CPG, int HW)
{
    __shared__ float rs[256];
    __shared__ float rq[256];

    const int bg = blockIdx.x;          // n*G + g
    const int sp = blockIdx.y;
    const int n  = bg / G;
    const int g  = bg - n * G;
    const long long total = (long long)CPG * HW;
    const long long beg = total * sp / NSPLIT;
    const long long end = total * (sp + 1) / NSPLIT;
    const size_t base = ((size_t)n * C + (size_t)g * CPG) * (size_t)HW;

    float s = 0.f, q = 0.f;
    for (long long i = beg + threadIdx.x; i < end; i += 256) {
        float v = y[base + (size_t)i];
        s += v;
        q += v * v;
    }
    rs[threadIdx.x] = s;
    rq[threadIdx.x] = q;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            rs[threadIdx.x] += rs[threadIdx.x + off];
            rq[threadIdx.x] += rq[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        psum[bg * NSPLIT + sp] = rs[0];
        psq[bg * NSPLIT + sp]  = rq[0];
    }
}

__global__ void gn_finalize(
    const float* __restrict__ psum, const float* __restrict__ psq,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ pa, float* __restrict__ pb,
    int B, int C, int G, int CPG, int HW, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * G) return;
    float s = 0.f, q = 0.f;
    for (int k = 0; k < NSPLIT; ++k) {
        s += psum[idx * NSPLIT + k];
        q += psq[idx * NSPLIT + k];
    }
    float tot  = (float)CPG * (float)HW;
    float mean = s / tot;
    float var  = q / tot - mean * mean;
    if (!(var > 0.f)) var = 0.f;
    float rstd = rsqrtf(var + eps);
    int n = idx / G;
    int g = idx - n * G;
    for (int i = 0; i < CPG; ++i) {
        int c = g * CPG + i;
        float gm = gamma[c];
        pa[n * C + c] = rstd * gm;
        pb[n * C + c] = beta[c] - mean * rstd * gm;
    }
}

// ---------------------------------------------------------------------------
// norm2 + SiLU + residual add, one pass.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void final_kernel(
    const float* __restrict__ y2, const float* __restrict__ x,
    const float* __restrict__ pa, const float* __restrict__ pb,
    float* __restrict__ out, int HW)
{
    const int plane = blockIdx.y;              // n*C + c
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= HW) return;
    const size_t off = (size_t)plane * HW + i;
    float v = y2[off] * pa[plane] + pb[plane];
    out[off] = silu_f(v) + x[off];
}

// ---------------------------------------------------------------------------
static void run_stats(const float* y, float* psum, float* psq,
                      const float* gamma, const float* beta,
                      float* pa, float* pb,
                      int B, int C, int G, int CPG, int HW, float eps,
                      cudaStream_t stream)
{
    dim3 gp(B * G, NSPLIT);
    gn_partial<<<gp, 256, 0, stream>>>(y, psum, psq, C, G, CPG, HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int nfin = (B * G + 127) / 128;
    gn_finalize<<<nfin, 128, 0, stream>>>(psum, psq, gamma, beta, pa, pb,
                                          B, C, G, CPG, HW, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                          torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                          double eps)
{
    TORCH_CHECK(x.is_cuda() && x.dim() == 4, "x must be a 4D CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "fp32 only");
    TORCH_CHECK(w1.scalar_type() == at::kFloat && w2.scalar_type() == at::kFloat, "fp32 only");

    auto xc  = x.is_contiguous()  ? x  : x.contiguous();
    auto w1c = w1.is_contiguous() ? w1 : w1.contiguous();
    auto w2c = w2.is_contiguous() ? w2 : w2.contiguous();
    auto g1c = g1.is_contiguous() ? g1 : g1.contiguous();
    auto b1c = b1.is_contiguous() ? b1 : b1.contiguous();
    auto g2c = g2.is_contiguous() ? g2 : g2.contiguous();
    auto b2c = b2.is_contiguous() ? b2 : b2.contiguous();

    const int B = (int)xc.size(0);
    const int C = (int)xc.size(1);
    const int H = (int)xc.size(2);
    const int W = (int)xc.size(3);
    const int G = 32;
    TORCH_CHECK(C % BM == 0 && C % G == 0, "C must be divisible by 64 and 32");
    TORCH_CHECK(w1c.size(2) == 3 && w1c.size(3) == 3, "3x3 kernels only");
    const int CPG = C / G;
    const int HW = H * W;

    auto opts = xc.options();
    auto y1 = torch::empty({B, C, H, W}, opts);
    auto y2 = torch::empty({B, C, H, W}, opts);
    auto out = torch::empty({B, C, H, W}, opts);
    auto psum = torch::empty({B * G * NSPLIT}, opts);
    auto psq  = torch::empty({B * G * NSPLIT}, opts);
    auto pa = torch::empty({B * C}, opts);
    auto pb = torch::empty({B * C}, opts);

    auto stream = at::cuda::getDefaultCUDAStream();

    const int tilesH = (H + TH - 1) / TH;
    const int tilesW = (W + TW - 1) / TW;
    dim3 gconv(C / BM, tilesH * tilesW, B);

    conv3x3_gn_kernel<false><<<gconv, NTHREADS, 0, stream>>>(
        xc.data_ptr<float>(), w1c.data_ptr<float>(), y1.data_ptr<float>(),
        nullptr, nullptr, C, H, W, tilesW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    run_stats(y1.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
              g1c.data_ptr<float>(), b1c.data_ptr<float>(),
              pa.data_ptr<float>(), pb.data_ptr<float>(),
              B, C, G, CPG, HW, (float)eps, stream);

    conv3x3_gn_kernel<true><<<gconv, NTHREADS, 0, stream>>>(
        y1.data_ptr<float>(), w2c.data_ptr<float>(), y2.data_ptr<float>(),
        pa.data_ptr<float>(), pb.data_ptr<float>(), C, H, W, tilesW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    run_stats(y2.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
              g2c.data_ptr<float>(), b2c.data_ptr<float>(),
              pa.data_ptr<float>(), pb.data_ptr<float>(),
              B, C, G, CPG, HW, (float)eps, stream);

    dim3 gf((HW + 255) / 256, B * C);
    final_kernel<<<gf, 256, 0, stream>>>(
        y2.data_ptr<float>(), xc.data_ptr<float>(),
        pa.data_ptr<float>(), pb.data_ptr<float>(),
        out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
'''

cpp_src = r'''
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                          torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                          double eps);
'''

_ext = load_inline(
    name="vae_resblock_tf32_implicit_gemm",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_block"],
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
    """Full-forward rewrite (granularity D) of the SOL VAE residual block.

    Stateless, like the reference: all weights arrive as forward arguments, so
    parameter parity is trivially preserved (no parameters are created here).
    """

    def __init__(self):
        super().__init__()
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps = float(eps.reshape(-1)[0].item()) if eps.numel() > 0 else 1e-6
        return self._ext.fused_block(x, conv1_weight, norm1_weight, norm1_bias,
                                     conv2_weight, norm2_weight, norm2_bias,
                                     float(eps))
