# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY: (C) "fuse many ops into one/few kernels"
#
# This revision: cudnn_algo_autotune.  The two 3x3 NHWC convolutions (63.5% of
# forward GPU time, previously run through an UNTIMED heuristic cuDNN plan
# `sm90_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32`)
# are now issued via at::cudnn_convolution with benchmark=true so cuDNN times
# every candidate engine for these fixed shapes and caches the fastest one.
# Numerics are unchanged: fp32 storage + fp32 I/O, TF32 math inside cuDNN.
#
# Fusion map (all custom kernels byte-for-byte identical to the previous round):
#    K_A  nchw2nhwc_kernel        : 32x32 tiled NCHW->NHWC transpose (entry).
#    conv3x3_nhwc_autotuned       : conv1 -> y1 (channels_last, TIMED plan)
#    K_B  gn_stats_kernel         : float4 partial sum/sumsq per (n,tile,group)
#    K_C  gn_finalize_kernel      : deterministic double combine -> mean/rstd
#    K_D  gn_silu_nhwc_inplace    : GN affine + SiLU fused, in-place NHWC
#    conv3x3_nhwc_autotuned       : conv2 -> y2 (channels_last, TIMED plan)
#    K_B/K_C again                : statistics for the second GroupNorm
#    K_E  gn_silu_res_nchw_v4     : GN affine + SiLU + residual + NHWC->NCHW,
#                                   128-bit (float4) accesses on all three
#                                   memory streams (mem_vectorize, still live).
#                                   Scalar variant kept for HW%4!=0 and tails.
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <exception>

#define GN_GROUPS 32
#define GN_C 256
#define GN_CPG 8          // channels per group (256 / 32)

// Reject-if switch (plan item 8): if nsys ever shows new nchwToNhwc/nhwcToNchw
// transform kernels or an extra elementwise kernel appearing around the convs,
// set this to 0 to pin the heuristic (untimed) plan again.
#define CONV_AUTOTUNE 1

__device__ __forceinline__ float silu_f(float t) {
    return t / (1.0f + __expf(-t));
}

// ---------------------------------------------------------------- K_A
// NCHW -> NHWC, 32(channel) x 32(pixel) shared-memory tiles.
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int C, int HW) {
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const size_t in_base  = (size_t)n * (size_t)C * (size_t)HW;
    const size_t out_base = (size_t)n * (size_t)HW * (size_t)C;

    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int jj = j + ty;
        const int c  = c0 + jj;
        const int p  = p0 + tx;
        float v = 0.0f;
        if (p < HW) v = in[in_base + (size_t)c * HW + p];
        tile[jj][tx] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int jj = j + ty;
        const int p  = p0 + jj;
        if (p < HW) {
            out[out_base + (size_t)p * C + c0 + tx] = tile[tx][jj];
        }
    }
}

// ---------------------------------------------------------------- K_B
// Partial (sum, sumsq) per (n, tile, group).  blockDim = (64, 4).
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                float* __restrict__ partial,
                                int HW, int tiles, int ppb) {
    const int tile = blockIdx.x;
    const int n    = blockIdx.y;
    const int tx   = threadIdx.x;          // 0..63
    const int ty   = threadIdx.y;          // 0..3
    const int g    = tx >> 1;              // 0..31

    const int p_start = tile * ppb;
    int p_end = p_start + ppb;
    if (p_end > HW) p_end = HW;

    const float4* base =
        reinterpret_cast<const float4*>(y + (size_t)n * (size_t)HW * GN_C);

    float s = 0.0f, ss = 0.0f;
    for (int p = p_start + ty; p < p_end; p += 4) {
        float4 v = base[(size_t)p * 64 + tx];
        s  += v.x + v.y + v.z + v.w;
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s  += __shfl_down_sync(0xffffffffu, s, 1);
    ss += __shfl_down_sync(0xffffffffu, ss, 1);

    __shared__ float sm_s[4][32];
    __shared__ float sm_ss[4][32];
    if ((tx & 1) == 0) { sm_s[ty][g] = s; sm_ss[ty][g] = ss; }
    __syncthreads();

    const int lin = ty * 64 + tx;
    if (lin < 32) {
        float a = sm_s[0][lin]  + sm_s[1][lin]  + sm_s[2][lin]  + sm_s[3][lin];
        float b = sm_ss[0][lin] + sm_ss[1][lin] + sm_ss[2][lin] + sm_ss[3][lin];
        size_t off = ((size_t)n * tiles + tile) * 64 + (size_t)lin * 2;
        partial[off]     = a;
        partial[off + 1] = b;
    }
}

// ---------------------------------------------------------------- K_C
__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int tiles, double invN, double eps) {
    const int idx = blockIdx.x;          // n * 32 + g
    const int n   = idx >> 5;
    const int g   = idx & 31;

    const float* base = partial + (size_t)n * tiles * 64 + (size_t)g * 2;

    double s = 0.0, ss = 0.0;
    for (int t = threadIdx.x; t < tiles; t += blockDim.x) {
        s  += (double)base[(size_t)t * 64];
        ss += (double)base[(size_t)t * 64 + 1];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off);
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    __shared__ double sm_s[8];
    __shared__ double sm_ss[8];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) { sm_s[wid] = s; sm_ss[wid] = ss; }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = blockDim.x >> 5;
        double S = 0.0, SS = 0.0;
        for (int i = 0; i < nw; ++i) { S += sm_s[i]; SS += sm_ss[i]; }
        double m   = S * invN;
        double var = (SS - S * S * invN) * invN;
        if (!(var > 0.0)) var = 0.0;
        mean[idx] = (float)m;
        rstd[idx] = (float)(1.0 / sqrt(var + eps));
    }
}

// ---------------------------------------------------------------- K_D
__global__ void gn_silu_nhwc_inplace(float* __restrict__ y,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ rstd,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     int HW) {
    const int n = blockIdx.y;
    float4* yn = reinterpret_cast<float4*>(y + (size_t)n * (size_t)HW * GN_C);

    const long long total  = (long long)HW * 64;
    const long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const int q = (int)(i & 63);
        const int g = q >> 1;
        const float m = mean[n * GN_GROUPS + g];
        const float r = rstd[n * GN_GROUPS + g];

        const int c = q * 4;
        const float a0 = r * gamma[c + 0];
        const float a1 = r * gamma[c + 1];
        const float a2 = r * gamma[c + 2];
        const float a3 = r * gamma[c + 3];
        const float b0 = beta[c + 0] - m * a0;
        const float b1 = beta[c + 1] - m * a1;
        const float b2 = beta[c + 2] - m * a2;
        const float b3 = beta[c + 3] - m * a3;

        float4 v = yn[i];
        float t;
        t = v.x * a0 + b0; v.x = silu_f(t);
        t = v.y * a1 + b1; v.y = silu_f(t);
        t = v.z * a2 + b2; v.z = silu_f(t);
        t = v.w * a3 + b3; v.w = silu_f(t);
        yn[i] = v;
    }
}

// ---------------------------------------------------------------- K_E (scalar)
// GroupNorm affine + SiLU + residual add + NHWC->NCHW transpose, one pass.
__global__ void gn_silu_res_nchw_kernel(const float* __restrict__ y,
                                        const float* __restrict__ res,
                                        float* __restrict__ out,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ rstd,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        int C, int HW, int p_offset) {
    __shared__ float tile[32][33];
    const int p0 = p_offset + blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int c = c0 + tx;
    const int g = c / GN_CPG;
    const float m = mean[n * GN_GROUPS + g];
    const float r = rstd[n * GN_GROUPS + g];
    const float a = r * gamma[c];
    const float b = beta[c] - m * a;

    const float* yn = y + (size_t)n * (size_t)HW * (size_t)C;

    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int jj = j + ty;
        const int p  = p0 + jj;
        float v = 0.0f;
        if (p < HW) v = yn[(size_t)p * C + c];
        const float t = v * a + b;
        tile[jj][tx] = silu_f(t);
    }
    __syncthreads();

    const size_t nchw_base = (size_t)n * (size_t)C * (size_t)HW;
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int jj = j + ty;
        const int cc = c0 + jj;
        const int p  = p0 + tx;
        if (p < HW) {
            const size_t off = nchw_base + (size_t)cc * HW + p;
            out[off] = tile[tx][jj] + res[off];
        }
    }
}

// ---------------------------------------------------------------- K_E (float4)
// mem_vectorize: identical math, 128-bit accesses on ALL THREE streams.
__global__ void gn_silu_res_nchw_v4_kernel(const float* __restrict__ y,
                                           const float* __restrict__ res,
                                           float* __restrict__ out,
                                           const float* __restrict__ mean,
                                           const float* __restrict__ rstd,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           int C, int HW) {
    __shared__ float tile[32][36];          // [channel-in-tile][pixel-in-tile]

    const int p0  = blockIdx.x * 32;
    const int c0  = blockIdx.y * 32;
    const int n   = blockIdx.z;
    const int tid = threadIdx.x;            // 0..255

    // ---------------- load phase : float4 along C (NHWC) + fused affine/SiLU
    const int ch4 = tid & 7;                // which float4 of the 32-channel tile
    const int pl  = tid >> 3;               // pixel inside the 32-pixel tile
    const int c   = c0 + 4 * ch4;
    const int g   = c >> 3;                 // == c / GN_CPG, GN_CPG == 8

    const float m = mean[n * GN_GROUPS + g];
    const float r = rstd[n * GN_GROUPS + g];

    const float4 gm = *reinterpret_cast<const float4*>(gamma + c);
    const float4 bt = *reinterpret_cast<const float4*>(beta  + c);

    const float a0 = r * gm.x;
    const float a1 = r * gm.y;
    const float a2 = r * gm.z;
    const float a3 = r * gm.w;
    const float b0 = bt.x - m * a0;
    const float b1 = bt.y - m * a1;
    const float b2 = bt.z - m * a2;
    const float b3 = bt.w - m * a3;

    const float* yn = y + (size_t)n * (size_t)HW * (size_t)C;
    const float4 v = *reinterpret_cast<const float4*>(
                         yn + (size_t)(p0 + pl) * (size_t)C + (size_t)c);

    tile[4 * ch4 + 0][pl] = silu_f(fmaf(v.x, a0, b0));
    tile[4 * ch4 + 1][pl] = silu_f(fmaf(v.y, a1, b1));
    tile[4 * ch4 + 2][pl] = silu_f(fmaf(v.z, a2, b2));
    tile[4 * ch4 + 3][pl] = silu_f(fmaf(v.w, a3, b3));

    __syncthreads();

    // ---------------- store phase : float4 along W (NCHW) for res + out
    const int p4 = tid & 7;                 // which float4 of the 32-pixel row
    const int cr = tid >> 3;                // channel inside the 32-channel tile
    const int cc = c0 + cr;

    const size_t off = (size_t)n * (size_t)C * (size_t)HW
                     + (size_t)cc * (size_t)HW + (size_t)(p0 + 4 * p4);

    const float4 rr = *reinterpret_cast<const float4*>(res + off);
    const float4 s  = *reinterpret_cast<const float4*>(&tile[cr][4 * p4]);

    float4 o;
    o.x = s.x + rr.x;
    o.y = s.y + rr.y;
    o.z = s.z + rr.z;
    o.w = s.w + rr.w;
    *reinterpret_cast<float4*>(out + off) = o;
}

// ---------------------------------------------------------------- host
static inline void pick_tiling(int B, int HW, int& tiles, int& ppb) {
    int target = 114 * 16;
    int t = (target + B - 1) / B;
    if (t < 1)  t = 1;
    if (t > HW) t = HW;
    ppb = (HW + t - 1) / t;
    if (ppb < 8) ppb = 8;
    ppb = ((ppb + 3) / 4) * 4;
    tiles = (HW + ppb - 1) / ppb;
}

static inline bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

static void run_group_stats(const at::Tensor& y, at::Tensor& mean, at::Tensor& rstd,
                            int B, int HW, double eps, cudaStream_t stream) {
    int tiles, ppb;
    pick_tiling(B, HW, tiles, ppb);
    auto partial = at::empty({(long)B * tiles * GN_GROUPS * 2}, y.options());

    dim3 blk(64, 4);
    dim3 grd(tiles, B);
    gn_stats_kernel<<<grd, blk, 0, stream>>>(
        y.data_ptr<float>(), partial.data_ptr<float>(), HW, tiles, ppb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const double invN = 1.0 / ((double)HW * (double)GN_CPG);
    gn_finalize_kernel<<<B * GN_GROUPS, 256, 0, stream>>>(
        partial.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        tiles, invN, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -------------------------------------------------------------------------
// cudnn_algo_autotune: timed (benchmark=true) engine selection for the two
// fixed-shape 3x3 NHWC convolutions.  fp32 in / fp32 out, TF32 math (identical
// numerical contract to at::conv2d with cudnn.allow_tf32=True).  The cuDNN plan
// cache is process-global and keyed by (shapes, dtype, layout, conv params), so
// the search runs only during warmup and steady-state iterations pay nothing.
// If a run ever shows nchwToNhwc/nhwcToNchw transform kernels appearing around
// the convs, set CONV_AUTOTUNE to 0 above to pin the heuristic plan.
// -------------------------------------------------------------------------
static at::Tensor conv3x3_nhwc_autotuned(const at::Tensor& in, const at::Tensor& w) {
#if CONV_AUTOTUNE
    try {
        return at::cudnn_convolution(in, w,
                                     /*padding=*/{1, 1},
                                     /*stride=*/{1, 1},
                                     /*dilation=*/{1, 1},
                                     /*groups=*/1,
                                     /*benchmark=*/true,
                                     /*deterministic=*/false,
                                     /*allow_tf32=*/true);
    } catch (const std::exception&) {
        return at::conv2d(in, w, at::Tensor(), {1, 1}, {1, 1}, {1, 1}, 1);
    }
#else
    return at::conv2d(in, w, at::Tensor(), {1, 1}, {1, 1}, {1, 1}, 1);
#endif
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps) {
    // Belt-and-braces: even the at::conv2d fallback uses timed plan selection.
    // allow_tf32 is intentionally NOT touched (must stay true = reference math).
    static const bool _cudnn_bench_once = [](){
        at::globalContext().setBenchmarkCuDNN(true);
        return true;
    }();
    (void)_cudnn_bench_once;

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "fp32 only");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    const int B = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    TORCH_CHECK(C == GN_C, "this kernel is specialised for C=256");
    const int HW = H * W;

    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c  = x.is_contiguous() ? x : x.contiguous();
    auto g1_c = n1w.is_contiguous() ? n1w : n1w.contiguous();
    auto b1_c = n1b.is_contiguous() ? n1b : n1b.contiguous();
    auto g2_c = n2w.is_contiguous() ? n2w : n2w.contiguous();
    auto b2_c = n2b.is_contiguous() ? n2b : n2b.contiguous();

    // ---- K_A : one-shot NCHW -> NHWC for the activations
    auto x_nhwc = at::empty({B, C, H, W},
                            x_c.options().memory_format(at::MemoryFormat::ChannelsLast));
    {
        dim3 blk(32, 8);
        dim3 grd((HW + 31) / 32, C / 32, B);
        nchw2nhwc_kernel<<<grd, blk, 0, stream>>>(
            x_c.data_ptr<float>(), x_nhwc.data_ptr<float>(), C, HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);

    auto mean = at::empty({B, GN_GROUPS}, x_c.options());
    auto rstd = at::empty({B, GN_GROUPS}, x_c.options());

    // ---- conv1 (cuDNN NHWC TF32, TIMED engine selection)
    auto y1 = conv3x3_nhwc_autotuned(x_nhwc, w1c);
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GN1 + SiLU (K_B, K_C, K_D)  [untouched]
    run_group_stats(y1, mean, rstd, B, HW, eps, stream);
    {
        long long total = (long long)HW * 64;
        int bx = (int)((total + 255) / 256);
        if (bx > 4096) bx = 4096;
        if (bx < 1) bx = 1;
        dim3 grd(bx, B);
        gn_silu_nhwc_inplace<<<grd, 256, 0, stream>>>(
            y1.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            g1_c.data_ptr<float>(), b1_c.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (cuDNN NHWC TF32, TIMED engine selection)
    auto y2 = conv3x3_nhwc_autotuned(y1, w2c);
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GN2 + SiLU + residual + NHWC->NCHW (K_B, K_C, K_E)  [untouched]
    run_group_stats(y2, mean, rstd, B, HW, eps, stream);

    auto out = at::empty({B, C, H, W}, x_c.options());  // contiguous NCHW

    const float* y2p  = y2.data_ptr<float>();
    const float* resp = x_c.data_ptr<float>();
    float*       outp = out.data_ptr<float>();

    // vector path requires 16 B alignment on all three streams and HW % 4 == 0
    const bool use_v4 = (HW % 4 == 0) && (HW >= 32) &&
                        aligned16(y2p) && aligned16(resp) && aligned16(outp);

    if (use_v4) {
        const int full_tiles = HW / 32;
        if (full_tiles > 0) {
            dim3 grd(full_tiles, C / 32, B);
            gn_silu_res_nchw_v4_kernel<<<grd, 256, 0, stream>>>(
                y2p, resp, outp,
                mean.data_ptr<float>(), rstd.data_ptr<float>(),
                g2_c.data_ptr<float>(), b2_c.data_ptr<float>(), C, HW);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const int tail_start = full_tiles * 32;
        if (tail_start < HW) {  // ragged tail columns -> scalar kernel, offset grid
            dim3 blk(32, 8);
            dim3 grd(1, C / 32, B);
            gn_silu_res_nchw_kernel<<<grd, blk, 0, stream>>>(
                y2p, resp, outp,
                mean.data_ptr<float>(), rstd.data_ptr<float>(),
                g2_c.data_ptr<float>(), b2_c.data_ptr<float>(), C, HW, tail_start);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } else {
        dim3 blk(32, 8);
        dim3 grd((HW + 31) / 32, C / 32, B);
        gn_silu_res_nchw_kernel<<<grd, blk, 0, stream>>>(
            y2p, resp, outp,
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            g2_c.data_ptr<float>(), b2_c.data_ptr<float>(), C, HW, 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_v5",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
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
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """Granularity (C): the whole residual block collapsed into a few fused kernels
    plus two vendor cuDNN convs, all issued from inside the load_inline extension.
    This revision turns on TIMED cuDNN engine selection (cudnn_algo_autotune) for
    both 3x3 NHWC convolutions, which own 63.5% of forward GPU time; every custom
    kernel (including the 128-bit mem_vectorize epilogue) is unchanged."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.reshape(()).item())
        else:
            eps_f = float(eps)

        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256 and conv1_weight.dtype == torch.float32
                and conv2_weight.dtype == torch.float32):
            return self.ext.fused_block(x, conv1_weight, norm1_weight, norm1_bias,
                                        conv2_weight, norm2_weight, norm2_bias, eps_f)

        # Fallback (never taken for this problem's workloads): reference semantics.
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=eps_f)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=eps_f)
        out = F.silu(out)
        return out + x
