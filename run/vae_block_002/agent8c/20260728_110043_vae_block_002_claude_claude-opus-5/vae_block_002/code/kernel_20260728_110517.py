# =============================================================================
# ModelNew — SOL problem 002: Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual
#
# SEED GRANULARITY: (C) fuse many ops into one/few custom CUDA kernels.
#
# 1) Chosen granularity: (C) operator fusion into a few hand-written CUDA kernels.
# 2) Ops replaced by custom CUDA kernels:
#      - group_norm(#1) + silu(#1)                    -> fused kernel set #1
#      - group_norm(#2) + silu(#2) + residual add     -> fused kernel set #2
#        (the 2nd set also performs an on-chip NHWC->NCHW transpose so the final
#         output is a plain contiguous NCHW tensor with no extra memory pass)
# 3) Fusion map:
#      kernel gn_stats_kernel        : blocked partial sum/sumsq  (fission of the
#                                      reduction for parallelism: grid = (K, N*G))
#      kernel gn_finalize_kernel     : combines partials -> (mean, rstd)
#      kernel gn_silu_apply_nhwc     : normalize * gamma + beta, SiLU, NHWC out   (fused)
#      kernel gn_silu_res_apply_nchw : normalize, SiLU, + residual, shared-memory
#                                      transpose, NCHW contiguous store          (fused)
# 4) Left in PyTorch (intentionally):
#      - F.conv2d (x2): cuDNN NHWC/TF32 implicit-GEMM is already at/near SOL for
#        256->256 3x3 fp32; we only feed it channels_last tensors so it hits the
#        tensor-core path. Layout conversions of x / weights are cheap PyTorch copies.
#      - dtype/format bookkeeping in Python (no device moves, no global state).
#
# Precision policy: everything stays fp32 (storage + arithmetic); reductions are
# fp32 accumulators (never narrower). TF32 for conv is inherited from the default
# torch/cuDNN setting, exactly as the reference does.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

#define CPG   8          // channels per group (C=256, G=32)
#define VPP   2          // float4 vectors per pixel per group (CPG/4)
#define NTHR  256
#define PCHUNK 128       // pixels processed per shared-memory tile (NTHR/VPP)

__device__ __forceinline__ void warpReduce2(float &a, float &b) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, off);
        b += __shfl_down_sync(0xffffffffu, b, off);
    }
}

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// Pass 1: partial sums / sums-of-squares over a pixel chunk of one (n,g) group.
// grid = (K, N*G), block = 256.   Input is channels-last (NHWC).
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ inp,
                                float* __restrict__ partial,
                                int HW, int C, int G, int chunk, int K)
{
    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    const float* base = inp + (long long)n * HW * C + (long long)g * CPG;

    const int tid     = threadIdx.x;
    const int j       = tid & (VPP - 1);
    const int p0      = tid / VPP;
    const int pstride = NTHR / VPP;

    float s = 0.f, ss = 0.f;
    for (int p = pstart + p0; p < pend; p += pstride) {
        const float4 v = *reinterpret_cast<const float4*>(base + (long long)p * C + j * 4);
        s  += v.x + v.y + v.z + v.w;
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    __shared__ float sa[NTHR / 32];
    __shared__ float sb[NTHR / 32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    warpReduce2(s, ss);
    if (lane == 0) { sa[wid] = s; sb[wid] = ss; }
    __syncthreads();
    if (wid == 0) {
        const int nw = NTHR / 32;
        s  = (lane < nw) ? sa[lane] : 0.f;
        ss = (lane < nw) ? sb[lane] : 0.f;
        warpReduce2(s, ss);
        if (lane == 0) {
            partial[((long long)ng * K + blk) * 2 + 0] = s;
            partial[((long long)ng * K + blk) * 2 + 1] = ss;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: finalize (mean, rstd) per (n,g).  grid = N*G, block = 32.
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ stats,
                                   int K, float invcount, float eps)
{
    const int ng = blockIdx.x;
    float s = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < K; i += 32) {
        s  += partial[((long long)ng * K + i) * 2 + 0];
        ss += partial[((long long)ng * K + i) * 2 + 1];
    }
    warpReduce2(s, ss);
    if (threadIdx.x == 0) {
        const float mean = s * invcount;
        float var = ss * invcount - mean * mean;
        if (var < 0.f) var = 0.f;
        stats[ng * 2 + 0] = mean;
        stats[ng * 2 + 1] = rsqrtf(var + eps);
    }
}

// ---------------------------------------------------------------------------
// Pass 3a: normalize * gamma + beta -> SiLU, NHWC in / NHWC out.
// grid = (K, N*G), block = 256.
// ---------------------------------------------------------------------------
__global__ void gn_silu_apply_nhwc(const float* __restrict__ inp,
                                   float* __restrict__ out,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ stats,
                                   int HW, int C, int G, int chunk)
{
    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const float mean = stats[ng * 2 + 0];
    const float rstd = stats[ng * 2 + 1];

    const int tid     = threadIdx.x;
    const int j       = tid & (VPP - 1);
    const int p0      = tid / VPP;
    const int pstride = NTHR / VPP;

    const float4 gm = *reinterpret_cast<const float4*>(gamma + g * CPG + j * 4);
    const float4 bt = *reinterpret_cast<const float4*>(beta  + g * CPG + j * 4);

    const float* ibase = inp + (long long)n * HW * C + (long long)g * CPG;
    float*       obase = out + (long long)n * HW * C + (long long)g * CPG;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    for (int p = pstart + p0; p < pend; p += pstride) {
        const long long off = (long long)p * C + j * 4;
        const float4 v = *reinterpret_cast<const float4*>(ibase + off);
        float4 o;
        o.x = silu_f((v.x - mean) * rstd * gm.x + bt.x);
        o.y = silu_f((v.y - mean) * rstd * gm.y + bt.y);
        o.z = silu_f((v.z - mean) * rstd * gm.z + bt.z);
        o.w = silu_f((v.w - mean) * rstd * gm.w + bt.w);
        *reinterpret_cast<float4*>(obase + off) = o;
    }
}

// ---------------------------------------------------------------------------
// Pass 3b: normalize -> SiLU -> + residual, NHWC in / NCHW contiguous out.
// On-chip transpose through shared memory keeps both loads and stores coalesced.
// grid = (K, N*G), block = 256 (must equal NTHR).
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_apply_nchw(const float* __restrict__ inp,
                                       const float* __restrict__ res,
                                       float* __restrict__ out,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ stats,
                                       int HW, int C, int G, int chunk)
{
    __shared__ float sh[CPG * (PCHUNK + 1)];

    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const float mean = stats[ng * 2 + 0];
    const float rstd = stats[ng * 2 + 1];

    const int tid = threadIdx.x;
    const int j   = tid & (VPP - 1);   // which float4 inside the group
    const int pl  = tid / VPP;         // local pixel index (0..127)

    const float4 gm = *reinterpret_cast<const float4*>(gamma + g * CPG + j * 4);
    const float4 bt = *reinterpret_cast<const float4*>(beta  + g * CPG + j * 4);

    const float* ibase = inp + (long long)n * HW * C + (long long)g * CPG;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    for (int p0 = pstart; p0 < pend; p0 += PCHUNK) {
        const int p = p0 + pl;
        if (p < pend) {
            const float4 v = *reinterpret_cast<const float4*>(ibase + (long long)p * C + j * 4);
            const int c0 = j * 4;
            sh[(c0 + 0) * (PCHUNK + 1) + pl] = silu_f((v.x - mean) * rstd * gm.x + bt.x);
            sh[(c0 + 1) * (PCHUNK + 1) + pl] = silu_f((v.y - mean) * rstd * gm.y + bt.y);
            sh[(c0 + 2) * (PCHUNK + 1) + pl] = silu_f((v.z - mean) * rstd * gm.z + bt.z);
            sh[(c0 + 3) * (PCHUNK + 1) + pl] = silu_f((v.w - mean) * rstd * gm.w + bt.w);
        }
        __syncthreads();

        const int cnt = (pend - p0) < PCHUNK ? (pend - p0) : PCHUNK;
#pragma unroll
        for (int it = 0; it < (CPG * PCHUNK) / NTHR; ++it) {
            const int idx = tid + it * NTHR;
            const int c   = idx / PCHUNK;
            const int lp  = idx - c * PCHUNK;
            if (lp < cnt) {
                const long long oidx =
                    ((long long)(n * C + g * CPG + c)) * HW + (long long)(p0 + lp);
                out[oidx] = sh[c * (PCHUNK + 1) + lp] + res[oidx];
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static inline void pick_chunk(long long NG, int HW, int &chunk, int &K)
{
    const long long target_blocks = 2048;
    int k = (int)((target_blocks + NG - 1) / NG);
    if (k < 1) k = 1;
    int per = (HW + k - 1) / k;
    chunk = ((per + PCHUNK - 1) / PCHUNK) * PCHUNK;
    if (chunk < PCHUNK) chunk = PCHUNK;
    K = (HW + chunk - 1) / chunk;
}

static void run_stats(const torch::Tensor &inp, torch::Tensor &stats,
                      int N, int C, int G, int HW, int chunk, int K, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const long long NG = (long long)N * G;
    auto partial = torch::empty({NG * K * 2}, inp.options());

    dim3 grid(K, (unsigned)NG);
    gn_stats_kernel<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), partial.data_ptr<float>(), HW, C, G, chunk, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float invcount = 1.0f / (float)((long long)CPG * HW);
    gn_finalize_kernel<<<(unsigned)NG, 32, 0, stream>>>(
        partial.data_ptr<float>(), stats.data_ptr<float>(), K, invcount, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_nhwc(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, double eps)
{
    TORCH_CHECK(inp.is_cuda(), "input must be CUDA");
    TORCH_CHECK(inp.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(inp.dim() == 4, "4D only");
    TORCH_CHECK(inp.is_contiguous(at::MemoryFormat::ChannelsLast), "need channels_last");

    const int N = (int)inp.size(0), C = (int)inp.size(1);
    const int H = (int)inp.size(2), W = (int)inp.size(3);
    const int G = 32, HW = H * W;
    TORCH_CHECK(C % G == 0 && (C / G) == CPG, "unsupported channel/group config");

    auto out = torch::empty_like(inp);           // preserves channels_last
    auto stats = torch::empty({(long long)N * G * 2}, inp.options());

    int chunk = 0, K = 0;
    pick_chunk((long long)N * G, HW, chunk, K);
    run_stats(inp, stats, N, C, G, HW, chunk, K, eps);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(K, (unsigned)((long long)N * G));
    gn_silu_apply_nhwc<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        stats.data_ptr<float>(), HW, C, G, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_res_nchw(torch::Tensor inp, torch::Tensor gamma,
                               torch::Tensor beta, torch::Tensor res, double eps)
{
    TORCH_CHECK(inp.is_cuda() && res.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(inp.scalar_type() == torch::kFloat32 &&
                res.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(inp.is_contiguous(at::MemoryFormat::ChannelsLast), "need channels_last");
    TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");

    const int N = (int)inp.size(0), C = (int)inp.size(1);
    const int H = (int)inp.size(2), W = (int)inp.size(3);
    const int G = 32, HW = H * W;
    TORCH_CHECK(C % G == 0 && (C / G) == CPG, "unsupported channel/group config");

    auto out = torch::empty({N, C, H, W}, inp.options());   // contiguous NCHW
    auto stats = torch::empty({(long long)N * G * 2}, inp.options());

    int chunk = 0, K = 0;
    pick_chunk((long long)N * G, HW, chunk, K);
    run_stats(inp, stats, N, C, G, HW, chunk, K, eps);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(K, (unsigned)((long long)N * G));
    gn_silu_res_apply_nchw<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), res.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        stats.data_ptr<float>(), HW, C, G, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor gn_silu_nhwc(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, double eps);
torch::Tensor gn_silu_res_nchw(torch::Tensor inp, torch::Tensor gamma,
                               torch::Tensor beta, torch::Tensor res, double eps);
'''

_ext = load_inline(
    name="vae_resblock_gn_silu_fused",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu_nhwc", "gn_silu_res_nchw"],
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
    """
    Granularity (C): GroupNorm+SiLU (+residual add, +NHWC->NCHW transpose) fused
    into custom CUDA kernels; the two 3x3 convolutions stay on cuDNN (channels_last).
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self,
                x,
                conv1_weight,
                norm1_weight,
                norm1_bias,
                conv2_weight,
                norm2_weight,
                norm2_bias,
                eps):
        eps_f = eps if isinstance(eps, float) else float(eps)

        C = x.size(1)
        num_groups = 32

        # Residual must stay untouched, contiguous NCHW.
        res = x if x.is_contiguous() else x.contiguous()

        # Fallback (never hit for this problem's fixed C=256/G=32) -> pure PyTorch.
        if C % num_groups != 0 or (C // num_groups) != 8 or x.dtype != torch.float32:
            out = F.conv2d(x, conv1_weight, None, 1, 1)
            out = F.silu(F.group_norm(out, num_groups, norm1_weight, norm1_bias, eps_f))
            out = F.conv2d(out, conv2_weight, None, 1, 1)
            out = F.silu(F.group_norm(out, num_groups, norm2_weight, norm2_bias, eps_f))
            return out + res

        cl = torch.channels_last
        xc = x if x.is_contiguous(memory_format=cl) else x.contiguous(memory_format=cl)
        w1 = conv1_weight if conv1_weight.is_contiguous(memory_format=cl) \
            else conv1_weight.contiguous(memory_format=cl)
        w2 = conv2_weight if conv2_weight.is_contiguous(memory_format=cl) \
            else conv2_weight.contiguous(memory_format=cl)

        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        y = F.conv2d(xc, w1, None, 1, 1)
        if not y.is_contiguous(memory_format=cl):
            y = y.contiguous(memory_format=cl)
        y = _ext.gn_silu_nhwc(y, g1, b1, eps_f)

        y = F.conv2d(y, w2, None, 1, 1)
        if not y.is_contiguous(memory_format=cl):
            y = y.contiguous(memory_format=cl)

        return _ext.gn_silu_res_nchw(y, g2, b2, res, eps_f)
