# ==========================================================================
# ModelNew — fused VAE residual block (Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +x)
#
# GRANULARITY (STRICT, EXACTLY ONE): (D) FULL FORWARD REWRITE.
#   The whole reference `run()` body is re-implemented inside a single
#   load_inline extension entry point `fused_resblock(...)`; ModelNew.forward
#   does nothing but argument hygiene + one extension call.
#
# 1) Chosen granularity : (D) fully rewrite forward.
# 2) Ops replaced       : F.conv2d x2, F.group_norm x2, F.silu x2, residual add
#                         -> all executed inside the extension.
# 3) Fusion map         :
#      - conv1 / conv2   : cuDNN via ATen `at::conv2d` called *inside* the
#                          extension (vendor-library path, allowed). It reads the
#                          same global cuDNN/TF32 context as the reference, so the
#                          convolution numerics are bit-for-bit the reference ones.
#      - kernel `gn_stats_kernel`  : deterministic 2-stage (no atomics) per-group
#                          sum / sum-of-squares reduction, float4 vectorized.
#      - kernel `gn_apply_kernel`  : fuses  GroupNorm-affine + SiLU + (optional)
#                          residual add into ONE pass. Used twice:
#                             (a) after conv1, written in-place into the conv1
#                                 temporary (our own buffer, never a user tensor),
#                             (b) after conv2, with residual = x, into a fresh out.
#                          This collapses reference passes
#                          (gn-mean/var, gn-affine, silu, gn2, silu2, add)
#                          from ~9 global memory passes down to 5.
# 4) Left in PyTorch/ATen : only the two 3x3 convolutions (cuDNN implicit-GEMM /
#                          Winograd is already near-peak and, more importantly,
#                          keeping the identical library call guarantees the
#                          tight max_atol=2.8e-3 tolerance is met).
#
# Notes:
#   * No global state mutated (TF32 / benchmark flags are read implicitly by
#     at::conv2d exactly as the reference does), no randomness, deterministic
#     reduction order (fixed partial buffer, no float atomics).
#   * Inputs are never mutated; residual `x` is only read.
#   * No device moves; all outputs allocated on the input device.
# ==========================================================================

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

#define GN_BLOCK 256

__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------------------
// Stage 1: per-(sample,group) partial sums / sums of squares.
// grid = (B*G, seg), block = GN_BLOCK.  Deterministic: fixed partition, no atomics.
// n = number of (vector) elements per group.
// ---------------------------------------------------------------------------
template <bool V4>
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ part,
                                int n, int seg) {
    const int g = blockIdx.x;
    const int s = blockIdx.y;

    const int per   = (n + seg - 1) / seg;
    const int start = s * per;
    int end = start + per;
    if (end > n) end = n;

    float sum = 0.f, sq = 0.f;

    if (V4) {
        const float4* xp = reinterpret_cast<const float4*>(x) + (long long)g * (long long)n;
        for (int i = start + (int)threadIdx.x; i < end; i += GN_BLOCK) {
            float4 v = xp[i];
            sum += v.x + v.y + v.z + v.w;
            sq  += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        const float* xp = x + (long long)g * (long long)n;
        for (int i = start + (int)threadIdx.x; i < end; i += GN_BLOCK) {
            float v = xp[i];
            sum += v;
            sq  += v * v;
        }
    }

    __shared__ float ssum[GN_BLOCK / 32];
    __shared__ float ssq[GN_BLOCK / 32];

    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    sum = warp_reduce_sum(sum);
    sq  = warp_reduce_sum(sq);
    if (lane == 0) { ssum[wid] = sum; ssq[wid] = sq; }
    __syncthreads();

    if (wid == 0) {
        const int nw = GN_BLOCK / 32;
        float a = (lane < nw) ? ssum[lane] : 0.f;
        float b = (lane < nw) ? ssq[lane]  : 0.f;
        a = warp_reduce_sum(a);
        b = warp_reduce_sum(b);
        if (lane == 0) {
            part[2 * (g * seg + s) + 0] = a;
            part[2 * (g * seg + s) + 1] = b;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: normalize + affine + SiLU + optional residual add, single pass.
// grid = (B*G, by), block = GN_BLOCK.
// ---------------------------------------------------------------------------
template <bool V4, bool RES>
__global__ void gn_apply_kernel(const float* __restrict__ x,
                                const float* __restrict__ part,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                const float* __restrict__ res,
                                float* __restrict__ out,
                                int n, int seg, int hwv, int shift,
                                int cpg, int G, float eps, float inv_n) {
    const int g = blockIdx.x;

    __shared__ float sstat[2];
    __shared__ float sc[64];
    __shared__ float sb[64];

    if (threadIdx.x == 0) {
        const float* p = part + 2 * (long long)g * (long long)seg;
        float sum = 0.f, sq = 0.f;
        for (int i = 0; i < seg; ++i) { sum += p[2 * i]; sq += p[2 * i + 1]; }
        float mean = sum * inv_n;
        float var  = sq * inv_n - mean * mean;
        if (var < 0.f) var = 0.f;
        sstat[0] = mean;
        sstat[1] = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = sstat[0];
    const float rstd = sstat[1];
    const int   gi   = g % G;

    if ((int)threadIdx.x < cpg) {
        float gm = gamma[gi * cpg + (int)threadIdx.x];
        float bt = beta[gi * cpg + (int)threadIdx.x];
        float a  = gm * rstd;
        sc[threadIdx.x] = a;
        sb[threadIdx.x] = bt - mean * a;
    }
    __syncthreads();

    const long long base   = (long long)g * (long long)n;
    const int        stride = gridDim.y * GN_BLOCK;
    int i0 = blockIdx.y * GN_BLOCK + (int)threadIdx.x;

    if (V4) {
        const float4* xp = reinterpret_cast<const float4*>(x) + base;
        float4*       op = reinterpret_cast<float4*>(out) + base;
        const float4* rp = RES ? (reinterpret_cast<const float4*>(res) + base) : nullptr;
        for (int i = i0; i < n; i += stride) {
            int ch = (shift >= 0) ? (i >> shift) : (i / hwv);
            float a = sc[ch], b = sb[ch];
            float4 v = xp[i];
            float4 o;
            float t;
            t = fmaf(v.x, a, b); o.x = t / (1.f + expf(-t));
            t = fmaf(v.y, a, b); o.y = t / (1.f + expf(-t));
            t = fmaf(v.z, a, b); o.z = t / (1.f + expf(-t));
            t = fmaf(v.w, a, b); o.w = t / (1.f + expf(-t));
            if (RES) {
                float4 r = rp[i];
                o.x += r.x; o.y += r.y; o.z += r.z; o.w += r.w;
            }
            op[i] = o;
        }
    } else {
        const float* xp = x + base;
        float*       op = out + base;
        const float* rp = RES ? (res + base) : nullptr;
        for (int i = i0; i < n; i += stride) {
            int ch = (shift >= 0) ? (i >> shift) : (i / hwv);
            float t = fmaf(xp[i], sc[ch], sb[ch]);
            float o = t / (1.f + expf(-t));
            if (RES) o += rp[i];
            op[i] = o;
        }
    }
}

// ---------------------------------------------------------------------------
static void gn_silu_launch(const at::Tensor& in,
                           const at::Tensor& gamma,
                           const at::Tensor& beta,
                           const float* res_ptr,
                           at::Tensor& out,
                           double eps) {
    const int64_t B = in.size(0), C = in.size(1), H = in.size(2), W = in.size(3);
    const int G = 32;
    TORCH_CHECK(C % G == 0, "channels must be divisible by 32");
    const int64_t cpg = C / G;
    TORCH_CHECK(cpg <= 64, "channels per group must be <= 64");
    const int64_t HW = H * W;
    const int64_t gsize = cpg * HW;
    TORCH_CHECK(gsize > 0, "empty group");

    const bool v4 = (HW % 4 == 0);
    const int n   = (int)(v4 ? (gsize / 4) : gsize);
    const int hwv = (int)(v4 ? (HW / 4) : HW);

    int shift = -1;
    for (int s = 0; s < 31; ++s) if ((1 << s) == hwv) { shift = s; break; }

    int seg = (int)((n + 2047) / 2048);
    if (seg < 1) seg = 1;
    if (seg > 32) seg = 32;

    const int ngroups = (int)(B * G);
    auto part = at::empty({(int64_t)ngroups * seg * 2}, in.options());

    auto stream = at::cuda::getDefaultCUDAStream();

    dim3 grid_s(ngroups, seg);
    if (v4) {
        gn_stats_kernel<true><<<grid_s, GN_BLOCK, 0, stream>>>(
            in.data_ptr<float>(), part.data_ptr<float>(), n, seg);
    } else {
        gn_stats_kernel<false><<<grid_s, GN_BLOCK, 0, stream>>>(
            in.data_ptr<float>(), part.data_ptr<float>(), n, seg);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int by = (n + GN_BLOCK * 8 - 1) / (GN_BLOCK * 8);
    if (by < 1) by = 1;
    if (by > 256) by = 256;
    dim3 grid_a(ngroups, by);

    const float inv_n = 1.0f / (float)gsize;
    const float epsf  = (float)eps;

    if (v4) {
        if (res_ptr) {
            gn_apply_kernel<true, true><<<grid_a, GN_BLOCK, 0, stream>>>(
                in.data_ptr<float>(), part.data_ptr<float>(),
                gamma.data_ptr<float>(), beta.data_ptr<float>(),
                res_ptr, out.data_ptr<float>(),
                n, seg, hwv, shift, (int)cpg, G, epsf, inv_n);
        } else {
            gn_apply_kernel<true, false><<<grid_a, GN_BLOCK, 0, stream>>>(
                in.data_ptr<float>(), part.data_ptr<float>(),
                gamma.data_ptr<float>(), beta.data_ptr<float>(),
                nullptr, out.data_ptr<float>(),
                n, seg, hwv, shift, (int)cpg, G, epsf, inv_n);
        }
    } else {
        if (res_ptr) {
            gn_apply_kernel<false, true><<<grid_a, GN_BLOCK, 0, stream>>>(
                in.data_ptr<float>(), part.data_ptr<float>(),
                gamma.data_ptr<float>(), beta.data_ptr<float>(),
                res_ptr, out.data_ptr<float>(),
                n, seg, hwv, shift, (int)cpg, G, epsf, inv_n);
        } else {
            gn_apply_kernel<false, false><<<grid_a, GN_BLOCK, 0, stream>>>(
                in.data_ptr<float>(), part.data_ptr<float>(),
                gamma.data_ptr<float>(), beta.data_ptr<float>(),
                nullptr, out.data_ptr<float>(),
                n, seg, hwv, shift, (int)cpg, G, epsf, inv_n);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight,
                             torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight,
                             torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias,
                             double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B,C,H,W)");

    at::Tensor xc  = x.is_contiguous() ? x : x.contiguous();
    at::Tensor w1c = conv1_weight.is_contiguous() ? conv1_weight : conv1_weight.contiguous();
    at::Tensor w2c = conv2_weight.is_contiguous() ? conv2_weight : conv2_weight.contiguous();
    at::Tensor g1  = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
    at::Tensor b1  = norm1_bias.is_contiguous()   ? norm1_bias   : norm1_bias.contiguous();
    at::Tensor g2  = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
    at::Tensor b2  = norm2_bias.is_contiguous()   ? norm2_bias   : norm2_bias.contiguous();

    std::vector<int64_t> ones{1, 1};
    std::vector<int64_t> pads{1, 1};

    // ---- conv1 (cuDNN through ATen, identical numerics to reference F.conv2d)
    at::Tensor y = at::conv2d(xc, w1c, {}, ones, pads, ones, 1);
    if (!y.is_contiguous()) y = y.contiguous();

    // ---- fused GroupNorm + SiLU (in-place into our own temporary buffer)
    gn_silu_launch(y, g1, b1, nullptr, y, eps);

    // ---- conv2
    at::Tensor z = at::conv2d(y, w2c, {}, ones, pads, ones, 1);
    if (!z.is_contiguous()) z = z.contiguous();

    // ---- fused GroupNorm + SiLU + residual add
    at::Tensor out = at::empty_like(xc);
    gn_silu_launch(z, g2, b2, xc.data_ptr<float>(), out, eps);

    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight,
                             torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight,
                             torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias,
                             double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_v1",
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
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """Fully-rewritten forward (granularity D). See header comment at top of file."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

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
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)
        return self._ext.fused_resblock(
            x, conv1_weight, norm1_weight, norm1_bias,
            conv2_weight, norm2_weight, norm2_bias, eps_f,
        )
