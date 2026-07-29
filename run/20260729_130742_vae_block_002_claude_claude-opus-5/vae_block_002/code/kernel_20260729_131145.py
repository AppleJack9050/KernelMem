# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY: (C) fuse many ops into one/few custom kernels.
#
# 1) Chosen granularity: (C).  All the non-GEMM work of the block is fused into
#    two custom CUDA kernels per GroupNorm stage; the two 3x3 convolutions stay
#    as vendor (cuDNN) calls issued from *inside* the load_inline extension.
#
# 2) Ops replaced by custom CUDA / in-extension vendor calls:
#      - F.group_norm  (both)          -> custom split-reduction kernels
#      - F.silu        (both)          -> fused into the GroupNorm apply kernel
#      - residual add  (out + x)       -> fused into the 2nd apply kernel
#      - NCHW<->NHWC layout shuffles   -> eliminated: the whole pipeline is run
#                                        in channels_last so cuDNN never has to
#                                        insert nchwToNhwc / nhwcToNchw kernels
#                                        (these were 20.8% of reference time).
#      - F.conv2d (x2)                 -> at::conv2d called from the extension,
#                                        fed channels_last activations+weights
#                                        so it lands directly on the
#                                        sm90 NHWC TF32 implicit-GEMM kernel
#                                        (the very kernel the reference uses).
#
# 3) Fusion map:
#      kernel gn_partial_kernel : split-reduction over pixels producing
#                                 (sum, sumsq) partials per (n, group).
#                                 Grid = (nsplit, G, N); every block writes its
#                                 slot unconditionally (zeros for empty tails)
#                                 so no partial-buffer tail can be left unwritten
#                                 for H*W not divisible by the tile size.
#      kernel gn_apply_kernel<RESID=false> :
#                                 finalizes mean/rstd from the partials (warp
#                                 reduce, no extra launch) and applies
#                                 normalize * gamma + beta -> SiLU, writing
#                                 NHWC so conv2 consumes it directly.
#      kernel gn_apply_kernel<RESID=true>  :
#                                 same, plus residual add, and transposes on the
#                                 fly: reads NHWC (32B/thread, sector-aligned),
#                                 writes NCHW fully coalesced -> the final
#                                 layout conversion is free.
#
# 4) Left in PyTorch/vendor:
#      - the two 3x3 convolutions: cuDNN's TF32 NHWC implicit GEMM is at/near
#        roofline for 256->256 3x3; re-implementing it wins nothing, the win is
#        removing the layout traffic and the norm/activation passes around it.
#      - a generic fallback (at::group_norm/at::silu) for shapes where
#        channels-per-group != 8, purely for safety; never hit by this problem.
#
# Precision: everything stored and computed in fp32; reductions accumulate in
# fp32 (per-thread -> warp -> block, so error stays tiny). TF32 tensor cores are
# used only by conv, exactly as the fp32 reference does.
# ==========================================================================
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// Pass 1: split reduction of (sum, sumsq) over pixels for every (n, group).
// Input y is NHWC (channels fastest), group g owns channels [g*CPG, g*CPG+CPG).
// Every block writes its partial slot, including empty tails  -> no unwritten
// entries for arbitrary H*W.
// ---------------------------------------------------------------------------
template <int CPG>
__global__ void gn_partial_kernel(const float* __restrict__ y,
                                  float2* __restrict__ partial,
                                  int HW, int nsplit, int G, int C, int chunk) {
    const int s = blockIdx.x;
    const int g = blockIdx.y;
    const int n = blockIdx.z;

    int p0 = s * chunk;
    int p1 = min(HW, p0 + chunk);

    float sum = 0.f, sq = 0.f;
    const float* base = y + (size_t)n * (size_t)HW * (size_t)C + (size_t)g * CPG;

    for (int p = p0 + (int)threadIdx.x; p < p1; p += (int)blockDim.x) {
        const float4* v = reinterpret_cast<const float4*>(base + (size_t)p * (size_t)C);
#pragma unroll
        for (int k = 0; k < CPG / 4; ++k) {
            float4 a = v[k];
            sum += a.x + a.y + a.z + a.w;
            sq  += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        }
    }

    __shared__ float ssum[32];
    __shared__ float ssq[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    sum = warp_sum(sum);
    sq  = warp_sum(sq);
    if (lane == 0) { ssum[wid] = sum; ssq[wid] = sq; }
    __syncthreads();
    const int nw = (int)(blockDim.x >> 5);
    if (wid == 0) {
        sum = (lane < nw) ? ssum[lane] : 0.f;
        sq  = (lane < nw) ? ssq[lane]  : 0.f;
        sum = warp_sum(sum);
        sq  = warp_sum(sq);
        if (lane == 0) {
            partial[((size_t)n * (size_t)G + (size_t)g) * (size_t)nsplit + (size_t)s] =
                make_float2(sum, sq);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: finalize stats (warp reduce over nsplit partials, done redundantly
// per block -> saves a kernel launch) then normalize * gamma + beta -> SiLU,
// optional residual add. RESID=false writes NHWC, RESID=true writes NCHW.
// ---------------------------------------------------------------------------
template <int CPG, bool RESID>
__global__ void gn_apply_kernel(const float* __restrict__ y,
                                const float* __restrict__ resid,
                                float* __restrict__ out,
                                const float2* __restrict__ partial,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                int HW, int nsplit, int G, int C,
                                float eps, float inv_cnt) {
    const int g = blockIdx.y;
    const int n = blockIdx.z;

    __shared__ float sm[2];
    if (threadIdx.x < 32) {
        float s = 0.f, q = 0.f;
        const size_t pb = ((size_t)n * (size_t)G + (size_t)g) * (size_t)nsplit;
        for (int i = (int)threadIdx.x; i < nsplit; i += 32) {
            float2 v = partial[pb + (size_t)i];
            s += v.x;
            q += v.y;
        }
        s = warp_sum(s);
        q = warp_sum(q);
        if (threadIdx.x == 0) {
            float mean = s * inv_cnt;
            float var  = fmaxf(q * inv_cnt - mean * mean, 0.f);
            sm[0] = mean;
            sm[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    const float mean = sm[0];
    const float rstd = sm[1];

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= HW) return;

    const int c0 = g * CPG;
    float gm[CPG], bt[CPG];
#pragma unroll
    for (int j = 0; j < CPG; ++j) { gm[j] = gamma[c0 + j]; bt[j] = beta[c0 + j]; }

    const float* src = y + (size_t)n * (size_t)HW * (size_t)C
                         + (size_t)p * (size_t)C + (size_t)c0;
    float val[CPG];
#pragma unroll
    for (int k = 0; k < CPG / 4; ++k) {
        float4 a = *reinterpret_cast<const float4*>(src + 4 * k);
        val[4 * k + 0] = a.x; val[4 * k + 1] = a.y;
        val[4 * k + 2] = a.z; val[4 * k + 3] = a.w;
    }
#pragma unroll
    for (int j = 0; j < CPG; ++j) {
        float v = (val[j] - mean) * rstd * gm[j] + bt[j];
        v = v / (1.f + expf(-v));   // SiLU
        val[j] = v;
    }

    if (RESID) {
#pragma unroll
        for (int j = 0; j < CPG; ++j) {
            size_t idx = ((size_t)n * (size_t)C + (size_t)(c0 + j)) * (size_t)HW + (size_t)p;
            out[idx] = val[j] + resid[idx];
        }
    } else {
        float* dst = out + (size_t)n * (size_t)HW * (size_t)C
                         + (size_t)p * (size_t)C + (size_t)c0;
#pragma unroll
        for (int k = 0; k < CPG / 4; ++k) {
            *reinterpret_cast<float4*>(dst + 4 * k) =
                make_float4(val[4 * k + 0], val[4 * k + 1], val[4 * k + 2], val[4 * k + 3]);
        }
    }
}

static inline int pick_nsplit(int64_t NG, int64_t HW) {
    int64_t target = (2048 + NG - 1) / NG;
    int64_t ns = std::max<int64_t>(1, std::min<int64_t>(32, target));
    int64_t cap = std::max<int64_t>(1, HW / 256);
    ns = std::min(ns, cap);
    return (int)ns;
}

// gn+silu(+residual) stage: y_nhwc -> out
static void gn_stage(const at::Tensor& y_nhwc, const at::Tensor& gamma, const at::Tensor& beta,
                     at::Tensor& out, const float* resid_ptr, bool resid,
                     int64_t N, int64_t C, int64_t HW, int G, float eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int CPG = 8;
    const int nsplit = pick_nsplit(N * (int64_t)G, HW);
    const int chunk = (int)((HW + nsplit - 1) / nsplit);

    auto partial = at::empty({N * (int64_t)G * (int64_t)nsplit * 2}, y_nhwc.options());

    dim3 gridP(nsplit, G, (unsigned)N);
    gn_partial_kernel<8><<<gridP, 256, 0, stream>>>(
        y_nhwc.data_ptr<float>(),
        reinterpret_cast<float2*>(partial.data_ptr<float>()),
        (int)HW, nsplit, G, (int)C, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int threads = 256;
    const int nblk = (int)((HW + threads - 1) / threads);
    dim3 gridA(nblk, G, (unsigned)N);
    const float inv_cnt = 1.0f / (float)((double)HW * (double)CPG);

    if (resid) {
        gn_apply_kernel<8, true><<<gridA, threads, 0, stream>>>(
            y_nhwc.data_ptr<float>(), resid_ptr, out.data_ptr<float>(),
            reinterpret_cast<const float2*>(partial.data_ptr<float>()),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            (int)HW, nsplit, G, (int)C, eps, inv_cnt);
    } else {
        gn_apply_kernel<8, false><<<gridA, threads, 0, stream>>>(
            y_nhwc.data_ptr<float>(), nullptr, out.data_ptr<float>(),
            reinterpret_cast<const float2*>(partial.data_ptr<float>()),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            (int)HW, nsplit, G, (int)C, eps, inv_cnt);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor vae_resblock(torch::Tensor x,
                           torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                           torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                           double eps_d) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be 4-D NCHW");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "fp32 only");

    const int G = 32;
    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t HW = H * W;
    const float eps = (float)eps_d;

    auto xc = x.is_contiguous() ? x : x.contiguous();

    if (!(C == 256 && C % G == 0 && (C / G) == 8)) {
        // generic safety fallback (never taken for this problem's shapes)
        auto o = at::conv2d(xc, w1.contiguous(), {}, {1, 1}, {1, 1});
        o = at::group_norm(o, G, g1.contiguous(), b1.contiguous(), eps);
        o = at::silu(o);
        o = at::conv2d(o, w2.contiguous(), {}, {1, 1}, {1, 1});
        o = at::group_norm(o, G, g2.contiguous(), b2.contiguous(), eps);
        o = at::silu(o);
        return o + xc;
    }

    const auto ml = at::MemoryFormat::ChannelsLast;
    auto x_cl  = xc.contiguous(ml);
    auto w1_cl = w1.contiguous(ml);
    auto w2_cl = w2.contiguous(ml);
    auto gam1 = g1.is_contiguous() ? g1 : g1.contiguous();
    auto bet1 = b1.is_contiguous() ? b1 : b1.contiguous();
    auto gam2 = g2.is_contiguous() ? g2 : g2.contiguous();
    auto bet2 = b2.is_contiguous() ? b2 : b2.contiguous();

    // conv1 (cuDNN NHWC TF32 implicit GEMM, no layout shuffles)
    auto y1 = at::conv2d(x_cl, w1_cl, {}, {1, 1}, {1, 1});
    if (!y1.is_contiguous(ml)) y1 = y1.contiguous(ml);

    auto n1 = at::empty({N, C, H, W}, x.options().memory_format(ml));
    gn_stage(y1, gam1, bet1, n1, nullptr, false, N, C, HW, G, eps);

    // conv2
    auto y2 = at::conv2d(n1, w2_cl, {}, {1, 1}, {1, 1});
    if (!y2.is_contiguous(ml)) y2 = y2.contiguous(ml);

    auto out = at::empty({N, C, H, W}, x.options());  // NCHW contiguous result
    gn_stage(y2, gam2, bet2, out, xc.data_ptr<float>(), true, N, C, HW, G, eps);

    return out;
}
"""

_cpp_src = r"""
torch::Tensor vae_resblock(torch::Tensor x,
                           torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                           torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                           double eps_d);
"""

_ext = load_inline(
    name="vae_resblock_fused_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["vae_resblock"],
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
    """Fused VAE residual block (see header comment for the granularity plan)."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            e = float(eps.reshape(-1)[0].item())
        else:
            e = float(eps)
        return self.ext.vae_resblock(x, conv1_weight, norm1_weight, norm1_bias,
                                     conv2_weight, norm2_weight, norm2_bias, e)
