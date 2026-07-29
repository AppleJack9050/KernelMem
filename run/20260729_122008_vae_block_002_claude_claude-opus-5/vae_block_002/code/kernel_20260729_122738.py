# ==========================================================================
# ModelNew — fused VAE residual block
#   Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> + x
#
# SEED GRANULARITY: (C) "fuse many ops into one/few kernels"
#
# 1) Chosen granularity: (C). All non-convolution work is fused into a small set of
#    hand-written CUDA kernels; the two 3x3 convs stay on the vendor implicit-GEMM,
#    invoked from inside the extension (at::conv2d) with channels_last operands.
#
# 2) Ops replaced by custom CUDA kernels:
#      - cuDNN's internal NCHW<->NHWC layout transforms (nchwToNhwcKernel)
#      - both GroupNorm moment passes (RowwiseMomentsCUDAKernel + ComputeFusedParams)
#      - both GroupNorm affine applications (elementwise_kernel)
#      - both SiLU activations (vectorized_elementwise_kernel)
#      - the residual add (vectorized_elementwise_kernel)
#      - the final NHWC->NCHW transform back to the reference output layout
#
# 3) Fusion map (5 custom kernels + 2 vendor conv calls):
#      K0 nchw_to_nhwc_kernel       : x (NCHW) -> xn (NHWC), shared-mem tiled transpose
#      [vendor] at::conv2d(xn, w1)  : channels_last in/out => cuDNN needs NO layout kernels
#      K1 gn_stats_kernel           : coalesced NHWC partial sum/sumsq per (n, chunk, group)
#      K2 gn_finalize_kernel        : partial reduce -> mean/rstd -> per-(n,c) affine A,B
#      K3 gn_apply_nhwc_kernel      : fuses (v*A+B) + SiLU, float4 vectorised, NHWC->NHWC
#      [vendor] at::conv2d(y1, w2)
#      K1/K2 again on the conv2 output
#      K4 gn_apply_transpose_kernel : fuses (v*A+B) + SiLU + residual add + NHWC->NCHW
#                                     transpose in ONE pass (no extra memory traffic)
#
# 4) Left in PyTorch / vendor libraries:
#      - the two 3x3 256->256 convolutions: ~74% of runtime and already at ~95% of this
#        GPU's TF32 roofline; rewriting cannot win, and feeding them channels_last
#        tensors removes the layout-conversion kernels for free.
#      - a pure-PyTorch fallback for shapes/dtypes the fast path does not specialise for
#        (C not a multiple of 256, non-fp32, non-CUDA) — safety only, never hit here.
#
# Precision: fp32 storage/arithmetic throughout; GN reductions use fp32 partials combined
# in fp64; TF32 tensor cores are used by the convs exactly as the reference does.
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

#define TILE 32
#define TBY  8
#define STAT_BS 256

__device__ __forceinline__ float silu_f(float z) {
    return z / (1.0f + __expf(-z));
}

// ---------------------------------------------------------------- K0
__global__ void nchw_to_nhwc_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int HW, int C) {
    __shared__ float tile[TILE][TILE + 1];
    const int n  = blockIdx.z;
    const int s0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const float* pin = in + (size_t)n * (size_t)C * (size_t)HW;
    {
        int s = s0 + tx;
        #pragma unroll
        for (int k = 0; k < TILE / TBY; ++k) {
            int c = c0 + ty + k * TBY;
            float v = 0.0f;
            if (s < HW && c < C) v = pin[(size_t)c * (size_t)HW + (size_t)s];
            tile[ty + k * TBY][tx] = v;      // tile[c_local][s_local]
        }
    }
    __syncthreads();
    {
        float* pout = out + (size_t)n * (size_t)HW * (size_t)C;
        int c = c0 + tx;
        #pragma unroll
        for (int k = 0; k < TILE / TBY; ++k) {
            int s = s0 + ty + k * TBY;
            if (s < HW && c < C)
                pout[(size_t)s * (size_t)C + (size_t)c] = tile[tx][ty + k * TBY];
        }
    }
}

// ---------------------------------------------------------------- K1
// partial layout: [N, nblk, G, 2]
__global__ void gn_stats_kernel(const float* __restrict__ in,
                                float* __restrict__ partial,
                                int HW, int C, int G, int D,
                                int nblk, int chunk) {
    __shared__ float ssum[STAT_BS];
    __shared__ float ssq [STAT_BS];

    const int n = blockIdx.y;
    const int b = blockIdx.x;
    const int s0 = b * chunk;
    int s1 = s0 + chunk;
    if (s1 > HW) s1 = HW;

    const float* p = in + (size_t)n * (size_t)HW * (size_t)C;
    const int tid = threadIdx.x;

    for (int cbase = 0; cbase < C; cbase += STAT_BS) {
        int c = cbase + tid;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
        if (c < C && s0 < s1) {
            const float* q = p + (size_t)s0 * (size_t)C + (size_t)c;
            int si = s0;
            for (; si + 4 <= s1; si += 4) {
                float v0 = q[0];
                float v1 = q[(size_t)C];
                float v2 = q[(size_t)2 * C];
                float v3 = q[(size_t)3 * C];
                a0 += v0; q0 += v0 * v0;
                a1 += v1; q1 += v1 * v1;
                a2 += v2; q2 += v2 * v2;
                a3 += v3; q3 += v3 * v3;
                q += (size_t)4 * C;
            }
            for (; si < s1; ++si) {
                float v = *q;
                a0 += v; q0 += v * v;
                q += (size_t)C;
            }
        }
        ssum[tid] = (a0 + a1) + (a2 + a3);
        ssq [tid] = (q0 + q1) + (q2 + q3);
        __syncthreads();

        int ngrp = STAT_BS / D;
        if (tid < ngrp) {
            float sa = 0.f, sq = 0.f;
            for (int d = 0; d < D; ++d) {
                sa += ssum[tid * D + d];
                sq += ssq [tid * D + d];
            }
            int g = cbase / D + tid;
            if (g < G) {
                size_t o = ((size_t)((size_t)n * nblk + b) * G + g) * 2;
                partial[o + 0] = sa;
                partial[o + 1] = sq;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- K2
__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ A,
                                   float* __restrict__ B,
                                   int nblk, int G, int C, int D,
                                   float eps, float cnt) {
    __shared__ double smA[256];
    __shared__ double smB[256];

    const int idx = blockIdx.x;
    const int n = idx / G;
    const int g = idx - n * G;

    const float* p = partial + ((size_t)n * nblk * G + g) * 2;
    double s = 0.0, ss = 0.0;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        s  += (double)p[(size_t)b * G * 2 + 0];
        ss += (double)p[(size_t)b * G * 2 + 1];
    }
    smA[threadIdx.x] = s;
    smB[threadIdx.x] = ss;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) {
            smA[threadIdx.x] += smA[threadIdx.x + off];
            smB[threadIdx.x] += smB[threadIdx.x + off];
        }
        __syncthreads();
    }
    double mean = smA[0] / (double)cnt;
    double var  = smB[0] / (double)cnt - mean * mean;
    if (var < 0.0) var = 0.0;
    float rstd = (float)(1.0 / sqrt(var + (double)eps));
    float fm   = (float)mean;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int c = g * D + d;
        float gg = gamma[c];
        A[(size_t)n * C + c] = rstd * gg;
        B[(size_t)n * C + c] = beta[c] - fm * rstd * gg;
    }
}

// ---------------------------------------------------------------- K3
__global__ void gn_apply_nhwc_kernel(const float* __restrict__ in,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ out,
                                     long total4, int C4, int C) {
    extern __shared__ float sh[];
    float* As = sh;
    float* Bs = sh + C;

    const int n = blockIdx.y;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        As[c] = A[(size_t)n * C + c];
        Bs[c] = B[(size_t)n * C + c];
    }
    __syncthreads();

    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total4) return;

    int c4 = (int)(i - (i / (long)C4) * (long)C4);
    int c  = c4 * 4;

    const float4* pin  = (const float4*)(in  + (size_t)n * (size_t)total4 * 4);
    float4*       pout = (float4*)      (out + (size_t)n * (size_t)total4 * 4);

    float4 v = pin[i];
    float4 o;
    o.x = silu_f(v.x * As[c + 0] + Bs[c + 0]);
    o.y = silu_f(v.y * As[c + 1] + Bs[c + 1]);
    o.z = silu_f(v.z * As[c + 2] + Bs[c + 2]);
    o.w = silu_f(v.w * As[c + 3] + Bs[c + 3]);
    pout[i] = o;
}

// ---------------------------------------------------------------- K4
__global__ void gn_apply_transpose_kernel(const float* __restrict__ in,   // NHWC
                                          const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          const float* __restrict__ res,  // NCHW
                                          float* __restrict__ out,        // NCHW
                                          int HW, int C) {
    __shared__ float tile[TILE][TILE + 1];
    const int n  = blockIdx.z;
    const int s0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const float* pin = in + (size_t)n * (size_t)HW * (size_t)C;
    int c = c0 + tx;
    float a = 0.f, b = 0.f;
    if (c < C) {
        a = A[(size_t)n * C + c];
        b = B[(size_t)n * C + c];
    }
    #pragma unroll
    for (int k = 0; k < TILE / TBY; ++k) {
        int s = s0 + ty + k * TBY;
        float y = 0.f;
        if (s < HW && c < C) {
            float v = pin[(size_t)s * (size_t)C + (size_t)c];
            y = silu_f(v * a + b);
        }
        tile[ty + k * TBY][tx] = y;          // tile[s_local][c_local]
    }
    __syncthreads();

    const size_t nbase = (size_t)n * (size_t)C * (size_t)HW;
    int s = s0 + tx;
    #pragma unroll
    for (int k = 0; k < TILE / TBY; ++k) {
        int cc = c0 + ty + k * TBY;
        if (s < HW && cc < C) {
            size_t off = nbase + (size_t)cc * (size_t)HW + (size_t)s;
            out[off] = tile[tx][ty + k * TBY] + res[off];
        }
    }
}

// ---------------------------------------------------------------- host helpers
static void run_gn_stats(const at::Tensor& src_nhwc,
                         const at::Tensor& gamma, const at::Tensor& beta,
                         at::Tensor& A, at::Tensor& B, at::Tensor& partial,
                         int64_t N, int64_t C, int64_t HW,
                         int G, int D, int nblk, int chunk,
                         float eps, cudaStream_t stream) {
    dim3 gs((unsigned)nblk, (unsigned)N, 1);
    gn_stats_kernel<<<gs, STAT_BS, 0, stream>>>(
        src_nhwc.data_ptr<float>(), partial.data_ptr<float>(),
        (int)HW, (int)C, G, D, nblk, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_kernel<<<(unsigned)(N * G), 256, 0, stream>>>(
        partial.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        A.data_ptr<float>(), B.data_ptr<float>(),
        nblk, G, (int)C, D, eps, (float)((double)HW * (double)D));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor gamma1, torch::Tensor beta1,
                             torch::Tensor w2, torch::Tensor gamma2, torch::Tensor beta2,
                             double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");

    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t HW = H * W;
    const int G = 32;
    TORCH_CHECK(C % G == 0 && C % 256 == 0, "unsupported channel count");
    const int D = (int)(C / G);
    TORCH_CHECK(256 % D == 0, "unsupported group width");

    auto stream = at::cuda::getCurrentCUDAStream();

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto opts    = xc.options();
    auto opts_cl = opts.memory_format(at::MemoryFormat::ChannelsLast);

    // ---- K0 : x -> NHWC ----
    auto xn = at::empty({N, C, H, W}, opts_cl);
    {
        dim3 grid((unsigned)((HW + TILE - 1) / TILE),
                  (unsigned)((C + TILE - 1) / TILE), (unsigned)N);
        dim3 blk(TILE, TBY, 1);
        nchw_to_nhwc_kernel<<<grid, blk, 0, stream>>>(
            xc.data_ptr<float>(), xn.data_ptr<float>(), (int)HW, (int)C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1c = gamma1.is_contiguous() ? gamma1 : gamma1.contiguous();
    auto b1c = beta1.is_contiguous()  ? beta1  : beta1.contiguous();
    auto g2c = gamma2.is_contiguous() ? gamma2 : gamma2.contiguous();
    auto b2c = beta2.is_contiguous()  ? beta2  : beta2.contiguous();

    // partial-sum sizing: chunk derived from nblk, then nblk re-derived from chunk so
    // that EVERY block owns a non-empty spatial range (no unwritten partial tail).
    int64_t per_img = std::max<int64_t>(1, 680 / std::max<int64_t>(1, N));
    int nblk_init = (int)std::max<int64_t>(1, std::min<int64_t>((HW + 63) / 64, per_img));
    int chunk = (int)((HW + nblk_init - 1) / nblk_init);
    if (chunk < 1) chunk = 1;
    int nblk = (int)((HW + chunk - 1) / chunk);

    auto partial = at::empty({N, (int64_t)nblk, (int64_t)G, 2}, opts);
    auto A = at::empty({N, C}, opts);
    auto B = at::empty({N, C}, opts);

    // ---- conv1 (vendor, channels_last in/out) ----
    auto c1 = at::conv2d(xn, w1c, {}, {1, 1}, {1, 1}, {1, 1}, 1);
    if (!c1.is_contiguous(at::MemoryFormat::ChannelsLast))
        c1 = c1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GN1 stats + K3 (affine + SiLU) ----
    run_gn_stats(c1, g1c, b1c, A, B, partial, N, C, HW, G, D, nblk, chunk,
                 (float)eps, stream);

    auto y1 = at::empty({N, C, H, W}, opts_cl);
    {
        const int C4 = (int)(C / 4);
        const long total4 = (long)HW * (long)C4;
        const int bs = 256;
        dim3 grid((unsigned)((total4 + bs - 1) / bs), (unsigned)N, 1);
        size_t shm = (size_t)2 * C * sizeof(float);
        gn_apply_nhwc_kernel<<<grid, bs, shm, stream>>>(
            c1.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
            y1.data_ptr<float>(), total4, C4, (int)C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (vendor, channels_last in/out) ----
    auto c2 = at::conv2d(y1, w2c, {}, {1, 1}, {1, 1}, {1, 1}, 1);
    if (!c2.is_contiguous(at::MemoryFormat::ChannelsLast))
        c2 = c2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GN2 stats + K4 (affine + SiLU + residual + NHWC->NCHW) ----
    run_gn_stats(c2, g2c, b2c, A, B, partial, N, C, HW, G, D, nblk, chunk,
                 (float)eps, stream);

    auto out = at::empty({N, C, H, W}, opts);
    {
        dim3 grid((unsigned)((HW + TILE - 1) / TILE),
                  (unsigned)((C + TILE - 1) / TILE), (unsigned)N);
        dim3 blk(TILE, TBY, 1);
        gn_apply_transpose_kernel<<<grid, blk, 0, stream>>>(
            c2.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
            xc.data_ptr<float>(), out.data_ptr<float>(), (int)HW, (int)C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor gamma1, torch::Tensor beta1,
                             torch::Tensor w2, torch::Tensor gamma2, torch::Tensor beta2,
                             double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_cl",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
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
    """See header comment: granularity (C) — everything except the two vendor
    convolutions is fused into 5 custom CUDA kernels operating in NHWC."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        e = float(eps)
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) % 256 == 0 and x.size(1) % 32 == 0):
            return self._ext.fused_resblock(
                x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, e)

        # Generic fallback (never taken for the benchmarked shapes).
        residual = x
        out = F.conv2d(x, conv1_weight, None, 1, 1)
        out = F.group_norm(out, 32, norm1_weight, norm1_bias, e)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, None, 1, 1)
        out = F.group_norm(out, 32, norm2_weight, norm2_bias, e)
        out = F.silu(out)
        return out + residual
