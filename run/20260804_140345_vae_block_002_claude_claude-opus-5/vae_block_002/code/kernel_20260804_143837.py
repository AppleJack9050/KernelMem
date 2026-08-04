# =============================================================================
# ModelNew — fused VAE residual block:
#   Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> +residual
#
# SEED GRANULARITY: (D) FULL FORWARD REWRITE.
#   1) Granularity: (D) — the whole forward is owned by this extension; the
#      vendor implicit-GEMM convolution (72.1% of reference GPU time) is
#      re-implemented here as a hand-written shared-memory tiled implicit GEMM
#      using wmma TF32 tensor cores with FP32 accumulate (same numeric class as
#      the reference, which runs cudnn with allow_tf32=True). No cuDNN/at::conv2d.
#   2) Replaced ops: F.conv2d (x2), F.group_norm (x2), F.silu (x2), residual add,
#      plus the NCHW<->NHWC conversions cudnn inserted (deleted: NCHW consumed
#      directly by the kernel).
#   3) Fusion map:
#      - wtrans_kernel          : one-off weight relayout (K,C,3,3)->(9,C,K) so
#                                 GEMM B-tile global loads are fully coalesced.
#      - conv_gn_kernel<false>  : conv1 implicit GEMM, 64(spatial)x128(channel)
#                                 tile, 4 warps, BK=16, wmma m16n8k8 TF32/f32acc.
#                                 EPILOGUE FUSED: per-group sum/sumsq computed
#                                 from accumulators still resident in shared mem,
#                                 so GroupNorm never re-reads the tensor.
#      - gn_finalize_kernel     : block partials -> mean/rstd -> per-(n,c) affine
#                                 (scale,bias) folded with GroupNorm gamma/beta.
#      - conv_gn_kernel<true>   : conv2, same GEMM, PROLOGUE FUSED: GroupNorm
#                                 affine + SiLU applied to the A-tile while it is
#                                 staged into shared memory, so the normalized+
#                                 SiLU activation never round-trips to global.
#      - final_kernel           : GroupNorm affine + SiLU + residual add in one
#                                 vectorized pass (single read of y2 and of x).
#   4) Remains in PyTorch: allocation + dispatch guard only. A pure-PyTorch
#      fallback exists solely for shapes/dtypes outside the compiled
#      specialization (C==256, fp32, CUDA); never used on the hot path.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define BM      64
#define BN      128
#define BK      16
#define APITCH  72
#define BPITCH  136
#define CPITCH  68

// ---------------------------------------------------------------------------
// Weight relayout: src (K, C, 3, 3) viewed as [K][C*9]  ->  dst[t][c][k]
// Shared-memory tiled transpose so both sides are coalesced.
// ---------------------------------------------------------------------------
__global__ void wtrans_kernel(const float* __restrict__ Wsrc,
                              float* __restrict__ Bt, int C, int K)
{
    __shared__ float tile[32][33];
    const int NC   = C * 9;
    const int col0 = blockIdx.x * 32;
    const int row0 = blockIdx.y * 32;
    const int cx   = threadIdx.x;
    const int ry   = threadIdx.y;

    for (int i = 0; i < 32; i += 8) {
        int r   = row0 + ry + i;
        int col = col0 + cx;
        if (r < K && col < NC) tile[ry + i][cx] = Wsrc[(size_t)r * NC + col];
    }
    __syncthreads();
    for (int i = 0; i < 32; i += 8) {
        int col = col0 + ry + i;
        int k   = row0 + cx;
        if (col < NC && k < K) {
            int c = col / 9;
            int t = col - c * 9;
            Bt[(size_t)t * C * K + (size_t)c * K + k] = tile[cx][ry + i];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused implicit-GEMM conv3x3 (pad 1, stride 1) + GroupNorm partial moments.
// PRE == true : input elements get  silu(x*scale + bias)  applied in prologue.
// C == K == 256, num_groups == 32 (group size 8).
// ---------------------------------------------------------------------------
template<bool PRE>
__global__ __launch_bounds__(128, 2)
void conv_gn_kernel(const float* __restrict__ X,
                    const float* __restrict__ Bt,
                    const float* __restrict__ pscale,
                    const float* __restrict__ pbias,
                    float* __restrict__ Y,
                    float* __restrict__ psum,
                    float* __restrict__ psumsq,
                    int H, int W, int WB, int NBS)
{
    constexpr int Cc = 256, Kc = 256;

    extern __shared__ __align__(16) float smem[];
    float* As = smem;
    float* Bs = smem + BK * APITCH;

    const int tid = threadIdx.x;
    const int wb  = blockIdx.x;
    const int h   = blockIdx.y;
    const int kb  = blockIdx.z & 1;
    const int n   = blockIdx.z >> 1;

    const int    w0 = wb * BM;
    const int    k0 = kb * BN;
    const int    HW = H * W;
    const size_t Xn = (size_t)n * Cc * (size_t)HW;
    const int    nC = n * Cc;

    const int warp   = tid >> 5;
    const int warp_m = warp & 1;
    const int warp_n = warp >> 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][4];

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int kh = 0; kh < 3; ++kh) {
        const int hh = h + kh - 1;
        if (hh < 0 || hh >= H) continue;          // whole tap is zero -> skip
        const size_t hbase = Xn + (size_t)hh * W;

        for (int kw = 0; kw < 3; ++kw) {
            const float* Bp = Bt + (size_t)(kh * 3 + kw) * Cc * Kc;

            for (int c0 = 0; c0 < Cc; c0 += BK) {
                __syncthreads();
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int idx = tid + r * 128;
                    int cc  = idx >> 6;
                    int wi  = idx & 63;
                    int ww  = w0 + wi + kw - 1;
                    float v = 0.0f;
                    if (ww >= 0 && ww < W) {
                        int ch = c0 + cc;
                        v = X[hbase + (size_t)ch * HW + ww];
                        if (PRE) {
                            float sc = pscale[nC + ch];
                            float bb = pbias[nC + ch];
                            v = v * sc + bb;
                            v = __fdividef(v, 1.0f + __expf(-v));
                        }
                    }
                    As[cc * APITCH + wi] = wmma::__float_to_tf32(v);
                }
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    int idx = tid + r * 128;
                    int cc  = idx >> 7;
                    int kk  = idx & 127;
                    Bs[cc * BPITCH + kk] =
                        wmma::__float_to_tf32(Bp[(size_t)(c0 + cc) * Kc + k0 + kk]);
                }
                __syncthreads();

#pragma unroll
                for (int kf = 0; kf < 2; ++kf) {
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                        wmma::load_matrix_sync(a_frag[i],
                            As + kf * 8 * APITCH + warp_m * 32 + i * 16, APITCH);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        wmma::load_matrix_sync(b_frag[j],
                            Bs + kf * 8 * BPITCH + warp_n * 64 + j * 16, BPITCH);
#pragma unroll
                    for (int i = 0; i < 2; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
    }

    // ---------------- fused epilogue: store + per-group moments -------------
    float* Cs = smem;                       // reuses As/Bs storage
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            wmma::store_matrix_sync(Cs + (warp_n * 64 + j * 16) * CPITCH + warp_m * 32 + i * 16,
                                    acc[i][j], CPITCH, wmma::mem_col_major);
    __syncthreads();

    __shared__ float sh_s[16][4];
    __shared__ float sh_ss[16][4];

    const int  wi = tid & 63;
    const int  nl = tid >> 6;
    const int  ww = w0 + wi;
    const bool wv = (ww < W);
    const size_t ybase = (size_t)n * Kc * (size_t)HW + (size_t)h * W + ww;

    for (int g = 0; g < 16; ++g) {
        float s = 0.0f, ss = 0.0f;
        for (int c = nl; c < 8; c += 2) {
            int   nn = g * 8 + c;
            float v  = Cs[nn * CPITCH + wi];
            if (wv) {
                Y[ybase + (size_t)(k0 + nn) * HW] = v;
                s  += v;
                ss += v * v;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s  += __shfl_down_sync(0xffffffffu, s, off);
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        }
        if ((tid & 31) == 0) { sh_s[g][tid >> 5] = s; sh_ss[g][tid >> 5] = ss; }
    }
    __syncthreads();

    if (tid < 16) {
        float s  = ((sh_s[tid][0]  + sh_s[tid][1])  + (sh_s[tid][2]  + sh_s[tid][3]));
        float ss = ((sh_ss[tid][0] + sh_ss[tid][1]) + (sh_ss[tid][2] + sh_ss[tid][3]));
        int    gg = kb * 16 + tid;
        int    bs = h * WB + wb;
        size_t o  = ((size_t)n * 32 + gg) * (size_t)NBS + bs;
        psum[o]   = s;
        psumsq[o] = ss;
    }
}

// ---------------------------------------------------------------------------
// Reduce block partials -> mean/rstd -> per-(n,c) folded affine.
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ psum,
                                   const float* __restrict__ psumsq,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ scale,
                                   float* __restrict__ bias,
                                   int NBS, double count, double eps, int C)
{
    const int idx = blockIdx.x;
    const int n   = idx >> 5;
    const int g   = idx & 31;
    const int t   = threadIdx.x;

    const float* ps  = psum   + (size_t)idx * NBS;
    const float* pss = psumsq + (size_t)idx * NBS;

    double s = 0.0, ss = 0.0;
    for (int i = t; i < NBS; i += blockDim.x) { s += (double)ps[i]; ss += (double)pss[i]; }

    __shared__ double rs[256], rss[256];
    rs[t] = s; rss[t] = ss;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) { rs[t] += rs[t + stride]; rss[t] += rss[t + stride]; }
        __syncthreads();
    }

    __shared__ float m_, r_;
    if (t == 0) {
        double mean = rs[0] / count;
        double var  = rss[0] / count - mean * mean;
        if (var < 0.0) var = 0.0;
        m_ = (float)mean;
        r_ = (float)(1.0 / sqrt(var + eps));
    }
    __syncthreads();

    if (t < 8) {
        int   c  = g * 8 + t;
        float gm = gamma[c];
        float bt = beta[c];
        scale[n * C + c] = gm * r_;
        bias [n * C + c] = bt - m_ * gm * r_;
    }
}

// ---------------------------------------------------------------------------
// Final fused epilogue: silu(y2*scale + bias) + residual
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_aff(float y, float s, float b)
{
    float v = y * s + b;
    return __fdividef(v, 1.0f + __expf(-v));
}

__global__ void final_kernel(const float* __restrict__ Y2,
                             const float* __restrict__ Xr,
                             const float* __restrict__ scale,
                             const float* __restrict__ bias,
                             float* __restrict__ out,
                             int HW, int C, int vec)
{
    const int row = blockIdx.x;
    const int n   = row / C;
    const int c   = row - n * C;
    const float s = scale[n * C + c];
    const float b = bias [n * C + c];
    const size_t off = (size_t)row * HW;

    if (vec) {
        const float4* y4 = (const float4*)(Y2 + off);
        const float4* x4 = (const float4*)(Xr + off);
        float4*       o4 = (float4*)(out + off);
        int n4 = HW >> 2;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float4 yv = y4[i], xv = x4[i], r;
            r.x = silu_aff(yv.x, s, b) + xv.x;
            r.y = silu_aff(yv.y, s, b) + xv.y;
            r.z = silu_aff(yv.z, s, b) + xv.z;
            r.w = silu_aff(yv.w, s, b) + xv.w;
            o4[i] = r;
        }
    } else {
        const float* y = Y2 + off;
        const float* x = Xr + off;
        float*       o = out + off;
        for (int i = threadIdx.x; i < HW; i += blockDim.x)
            o[i] = silu_aff(y[i], s, b) + x[i];
    }
}

// ---------------------------------------------------------------------------
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                             torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                             double eps)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    auto xc  = x.is_contiguous()  ? x  : x.contiguous();
    auto w1c = w1.is_contiguous() ? w1 : w1.contiguous();
    auto w2c = w2.is_contiguous() ? w2 : w2.contiguous();
    auto g1c = g1.is_contiguous() ? g1 : g1.contiguous();
    auto b1c = b1.is_contiguous() ? b1 : b1.contiguous();
    auto g2c = g2.is_contiguous() ? g2 : g2.contiguous();
    auto b2c = b2.is_contiguous() ? b2 : b2.contiguous();

    const int N = (int)xc.size(0);
    const int C = (int)xc.size(1);
    const int H = (int)xc.size(2);
    const int W = (int)xc.size(3);
    TORCH_CHECK(C == 256, "specialized for C=256");
    TORCH_CHECK(w1c.size(0) == 256 && w2c.size(0) == 256, "specialized for K=256");

    const int WB  = (W + BM - 1) / BM;
    const int NBS = H * WB;
    const int HW  = H * W;

    auto opts = xc.options();
    auto Bt1  = torch::empty({9, C, C}, opts);
    auto Bt2  = torch::empty({9, C, C}, opts);
    auto y1   = torch::empty({N, C, H, W}, opts);
    auto y2   = torch::empty({N, C, H, W}, opts);
    auto out  = torch::empty({N, C, H, W}, opts);
    auto ps   = torch::empty({(long)N * 32 * NBS}, opts);
    auto pss  = torch::empty({(long)N * 32 * NBS}, opts);
    auto sc1  = torch::empty({N, C}, opts);
    auto bi1  = torch::empty({N, C}, opts);
    auto sc2  = torch::empty({N, C}, opts);
    auto bi2  = torch::empty({N, C}, opts);

    auto stream = at::cuda::getDefaultCUDAStream();

    dim3 gT((C * 9 + 31) / 32, (C + 31) / 32);
    wtrans_kernel<<<gT, dim3(32, 8), 0, stream>>>(w1c.data_ptr<float>(), Bt1.data_ptr<float>(), C, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wtrans_kernel<<<gT, dim3(32, 8), 0, stream>>>(w2c.data_ptr<float>(), Bt2.data_ptr<float>(), C, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    size_t smem_a = (size_t)(BK * APITCH + BK * BPITCH);
    size_t smem_c = (size_t)(BN * CPITCH);
    size_t smem   = (smem_a > smem_c ? smem_a : smem_c) * sizeof(float);

    dim3 gc(WB, H, N * 2);
    conv_gn_kernel<false><<<gc, 128, smem, stream>>>(
        xc.data_ptr<float>(), Bt1.data_ptr<float>(), nullptr, nullptr,
        y1.data_ptr<float>(), ps.data_ptr<float>(), pss.data_ptr<float>(), H, W, WB, NBS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    double count = 8.0 * (double)H * (double)W;
    gn_finalize_kernel<<<N * 32, 256, 0, stream>>>(
        ps.data_ptr<float>(), pss.data_ptr<float>(), g1c.data_ptr<float>(), b1c.data_ptr<float>(),
        sc1.data_ptr<float>(), bi1.data_ptr<float>(), NBS, count, eps, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    conv_gn_kernel<true><<<gc, 128, smem, stream>>>(
        y1.data_ptr<float>(), Bt2.data_ptr<float>(), sc1.data_ptr<float>(), bi1.data_ptr<float>(),
        y2.data_ptr<float>(), ps.data_ptr<float>(), pss.data_ptr<float>(), H, W, WB, NBS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_kernel<<<N * 32, 256, 0, stream>>>(
        ps.data_ptr<float>(), pss.data_ptr<float>(), g2c.data_ptr<float>(), b2c.data_ptr<float>(),
        sc2.data_ptr<float>(), bi2.data_ptr<float>(), NBS, count, eps, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    final_kernel<<<N * C, 256, 0, stream>>>(
        y2.data_ptr<float>(), xc.data_ptr<float>(), sc2.data_ptr<float>(), bi2.data_ptr<float>(),
        out.data_ptr<float>(), HW, C, (HW % 4 == 0) ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
'''

_CPP = (
    "torch::Tensor fused_resblock(torch::Tensor x, torch::Tensor w1, torch::Tensor g1,"
    " torch::Tensor b1, torch::Tensor w2, torch::Tensor g2, torch::Tensor b2, double eps);"
)

_ext = load_inline(
    name="fused_vae_resblock_d",
    cpp_sources=_CPP,
    cuda_sources=_SRC,
    functions=["fused_resblock"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-gencode=arch=compute_120,code=sm_120",
    ],
)


class ModelNew(nn.Module):
    """Fully rewritten forward (granularity D). See header comment above."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256 and conv1_weight.size(0) == 256
                and conv2_weight.size(0) == 256):
            return self.ext.fused_resblock(x, conv1_weight, norm1_weight, norm1_bias,
                                           conv2_weight, norm2_weight, norm2_bias, eps_f)

        # Guard path only (unsupported shape/dtype/device) — never the hot path.
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=eps_f)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=eps_f)
        out = F.silu(out)
        return out + x
