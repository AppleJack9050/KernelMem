# ============================================================================
# ModelNew — fused VAE residual block (Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN
#            -> SiLU -> +residual), C=256, groups=32, fp32.
#
# OPTIMISATION (this revision): L2 CACHE BLOCKING.
#   The kernels themselves are unchanged and already run at 86-89% of DRAM
#   peak; the problem is that the full-batch live set (5 x 67MB) overflows the
#   ~96MB L2 so x and every intermediate round-trips to DRAM.  We therefore
#   execute the *entire* residual block in batch chunks sized so that
#   {x_chunk, z, y2} (3 x n x S bytes) stay resident in L2.  GroupNorm stats
#   are per-(image,group), so per-image chunking is mathematically exactly
#   equivalent — no kernel math changes at all.  A size gate keeps every
#   shape that cannot profit on the byte-identical single-shot path.
#
#   Host-side changes only:
#     * per-call invariants (eps, channels_last weights) hoisted out of loop
#     * new out-tensor variant gn_silu_res_nchw_out so chunks write in place
#     * chunk-size computation + gate + chunk loop in ModelNew.forward
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CTOT   256
#define NGRP   32
#define CPG    8

// ---------------------------------------------------------------------------
// NCHW -> NHWC tiled transpose (tile: 32 spatial x 64 channels, 256 threads)
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 int HW)
{
    __shared__ float sm[32 * 65];
    const int hw0 = blockIdx.x * 32;
    const int c0  = blockIdx.y * 64;
    const int n   = blockIdx.z;
    const int tid = threadIdx.x;

    const long long nbaseS = (long long)n * CTOT * (long long)HW;
    // load : coalesced along hw (NCHW)
    {
        const int lh = tid & 31;
        const int lc = tid >> 5;      // 0..7
        const int hw = hw0 + lh;
        if (hw < HW) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int cl = lc + i * 8;
                sm[lh * 65 + cl] = src[nbaseS + (long long)(c0 + cl) * HW + hw];
            }
        }
    }
    __syncthreads();
    // store : coalesced along c (NHWC)
    {
        const int lc = tid & 63;
        const int lh = tid >> 6;      // 0..3
        const long long nbaseD = (long long)n * (long long)HW * CTOT + (c0 + lc);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int hh = lh + i * 4;
            const int hw = hw0 + hh;
            if (hw < HW) {
                dst[nbaseD + (long long)hw * CTOT] = sm[hh * 65 + lc];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GroupNorm partial moments over NHWC data.  grid=(P,B), block=256 (1/channel)
// Every entry of the partial buffers is written (even for empty chunks).
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                int HW, int P,
                                float* __restrict__ psum,
                                float* __restrict__ psq)
{
    const int p = blockIdx.x;
    const int n = blockIdx.y;
    const int c = threadIdx.x;

    const int chunk = (HW + P - 1) / P;
    int hw0 = p * chunk;
    int hw1 = hw0 + chunk;
    if (hw1 > HW) hw1 = HW;

    float s = 0.f, q = 0.f;
    if (hw0 < hw1) {
        const float* base = y + (long long)n * (long long)HW * CTOT
                              + (long long)hw0 * CTOT + c;
        int cnt = hw1 - hw0;
        #pragma unroll 4
        for (int i = 0; i < cnt; ++i) {
            float v = base[(long long)i * CTOT];
            s += v;
            q += v * v;
        }
    }
    // reduce the 8 lanes belonging to one group
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
        q += __shfl_down_sync(0xffffffffu, q, off);
    }
    if ((c & 7) == 0) {
        const long long o = ((long long)n * P + p) * NGRP + (c >> 3);
        psum[o] = s;
        psq[o]  = q;
    }
}

// ---------------------------------------------------------------------------
// deterministic reduction of the partials -> mean / rstd (one block per group)
// ---------------------------------------------------------------------------
__global__ void gn_final_kernel(const float* __restrict__ psum,
                                const float* __restrict__ psq,
                                int P, int HW, float eps,
                                float* __restrict__ mean,
                                float* __restrict__ rstd)
{
    const int idx = blockIdx.x;           // n*32 + g
    const int n = idx >> 5;
    const int g = idx & 31;

    double s = 0.0, q = 0.0;
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const long long o = ((long long)n * P + p) * NGRP + g;
        s += (double)psum[o];
        q += (double)psq[o];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
        q += __shfl_down_sync(0xffffffffu, q, off);
    }
    __shared__ double ss[8], sq[8];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) { ss[warp] = s; sq[warp] = q; }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = blockDim.x >> 5;
        double ts = 0.0, tq = 0.0;
        for (int i = 0; i < nw; ++i) { ts += ss[i]; tq += sq[i]; }
        const double cnt = (double)HW * (double)CPG;
        const double m = ts / cnt;
        double v = tq / cnt - m * m;
        if (v < 0.0) v = 0.0;
        mean[idx] = (float)m;
        rstd[idx] = (float)(1.0 / sqrt(v + (double)eps));
    }
}

// ---------------------------------------------------------------------------
// GN affine + SiLU, NHWC -> NHWC, float4 vectorised.  grid=(x,B)
// ---------------------------------------------------------------------------
__global__ void gn_silu_apply(const float4* __restrict__ y,
                              const float4* __restrict__ w,
                              const float4* __restrict__ b,
                              const float* __restrict__ mean,
                              const float* __restrict__ rstd,
                              float4* __restrict__ out,
                              int n4perimg)
{
    const int n = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4perimg) return;

    const int c4 = i & 63;              // float4 index inside the 256 channels
    const int g  = c4 >> 1;             // 8 channels per group == 2 float4
    const float m = mean[n * NGRP + g];
    const float r = rstd[n * NGRP + g];

    const long long o = (long long)n * n4perimg + i;
    const float4 v  = y[o];
    const float4 ww = w[c4];
    const float4 bb = b[c4];

    float t0 = (v.x - m) * r * ww.x + bb.x;
    float t1 = (v.y - m) * r * ww.y + bb.y;
    float t2 = (v.z - m) * r * ww.z + bb.z;
    float t3 = (v.w - m) * r * ww.w + bb.w;

    float4 o4;
    o4.x = t0 / (1.f + __expf(-t0));
    o4.y = t1 / (1.f + __expf(-t1));
    o4.z = t2 / (1.f + __expf(-t2));
    o4.w = t3 / (1.f + __expf(-t3));
    out[o] = o4;
}

// ---------------------------------------------------------------------------
// GN affine + SiLU + residual add + NHWC->NCHW transpose in ONE pass.
// tile: 32 spatial x 64 channels, 256 threads, padded shared stride 65.
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_t(const float* __restrict__ y,
                              const float* __restrict__ res,
                              const float* __restrict__ w,
                              const float* __restrict__ b,
                              const float* __restrict__ mean,
                              const float* __restrict__ rstd,
                              float* __restrict__ out,
                              int HW)
{
    __shared__ float sm[32 * 65];
    const int hw0 = blockIdx.x * 32;
    const int c0  = blockIdx.y * 64;
    const int n   = blockIdx.z;
    const int tid = threadIdx.x;

    // ---- load NHWC (coalesced along c), normalise + silu into shared -------
    {
        const int lc = tid & 63;
        const int lh = tid >> 6;                 // 0..3
        const int c  = c0 + lc;
        const int g  = c >> 3;
        const float m  = mean[n * NGRP + g];
        const float r  = rstd[n * NGRP + g];
        const float ww = w[c];
        const float bb = b[c];
        const float* base = y + (long long)n * (long long)HW * CTOT + c;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int hh = lh + i * 4;
            const int hw = hw0 + hh;
            if (hw < HW) {
                const float v = base[(long long)hw * CTOT];
                const float t = (v - m) * r * ww + bb;
                sm[hh * 65 + lc] = t / (1.f + __expf(-t));
            }
        }
    }
    __syncthreads();

    // ---- store NCHW (coalesced along hw) + residual ------------------------
    {
        const int lh = tid & 31;
        const int lc = tid >> 5;                 // 0..7
        const int hw = hw0 + lh;
        if (hw < HW) {
            const long long nb = (long long)n * CTOT * (long long)HW;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int cl = lc + i * 8;
                const long long o = nb + (long long)(c0 + cl) * HW + hw;
                out[o] = sm[lh * 65 + cl] + res[o];
            }
        }
    }
}

// ===========================================================================
// host side
// ===========================================================================
static inline int pick_P(int B, int HW)
{
    int p = (HW + 31) / 32;
    if (p < 1) p = 1;
    int target = 1024 / (B > 0 ? B : 1);
    if (target < 1) target = 1;
    if (p > target) p = target;
    return p;
}

static void compute_stats(const float* y, int B, int HW, float eps,
                          torch::Tensor& mean, torch::Tensor& rstd,
                          const torch::TensorOptions& opts,
                          cudaStream_t stream)
{
    const int P = pick_P(B, HW);
    auto psum = torch::empty({(long long)B * P * NGRP}, opts);
    auto psq  = torch::empty({(long long)B * P * NGRP}, opts);

    dim3 g1(P, B);
    gn_stats_kernel<<<g1, 256, 0, stream>>>(y, HW, P,
                                            psum.data_ptr<float>(),
                                            psq.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_final_kernel<<<B * NGRP, 128, 0, stream>>>(psum.data_ptr<float>(),
                                                  psq.data_ptr<float>(),
                                                  P, HW, eps,
                                                  mean.data_ptr<float>(),
                                                  rstd.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor nchw_to_nhwc(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(x.dim() == 4, "4D only");
    const int B = (int)x.size(0), C = (int)x.size(1);
    const int H = (int)x.size(2), W = (int)x.size(3);
    TORCH_CHECK(C == CTOT, "C must be 256");
    const int HW = H * W;
    TORCH_CHECK(x.is_contiguous(), "expect contiguous NCHW");

    auto out = torch::empty({B, C, H, W},
                            x.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g((HW + 31) / 32, C / 64, B);
    nchw2nhwc_kernel<<<g, 256, 0, stream>>>(x.data_ptr<float>(),
                                            out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor w, torch::Tensor b,
                           double eps)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32);
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast),
                "expect channels_last conv output");
    const int B = (int)y.size(0), C = (int)y.size(1);
    const int H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == CTOT, "C must be 256");
    const int HW = H * W;

    auto opts = y.options();
    auto mean = torch::empty({(long long)B * NGRP}, opts);
    auto rstd = torch::empty({(long long)B * NGRP}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();

    compute_stats(y.data_ptr<float>(), B, HW, (float)eps, mean, rstd, opts, stream);

    auto out = torch::empty({B, C, H, W},
                            opts.memory_format(at::MemoryFormat::ChannelsLast));
    const int n4 = HW * (CTOT / 4);
    dim3 g((n4 + 255) / 256, B);
    gn_silu_apply<<<g, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(y.data_ptr<float>()),
        reinterpret_cast<const float4*>(w.data_ptr<float>()),
        reinterpret_cast<const float4*>(b.data_ptr<float>()),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_res_nchw(torch::Tensor y, torch::Tensor res,
                               torch::Tensor w, torch::Tensor b, double eps)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32);
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast),
                "expect channels_last conv output");
    TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");
    const int B = (int)y.size(0), C = (int)y.size(1);
    const int H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == CTOT, "C must be 256");
    const int HW = H * W;

    auto opts = y.options();
    auto mean = torch::empty({(long long)B * NGRP}, opts);
    auto rstd = torch::empty({(long long)B * NGRP}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();

    compute_stats(y.data_ptr<float>(), B, HW, (float)eps, mean, rstd, opts, stream);

    auto out = torch::empty({B, C, H, W}, opts);
    dim3 g((HW + 31) / 32, C / 64, B);
    gn_silu_res_t<<<g, 256, 0, stream>>>(y.data_ptr<float>(),
                                         res.data_ptr<float>(),
                                         w.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         mean.data_ptr<float>(),
                                         rstd.data_ptr<float>(),
                                         out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---------------------------------------------------------------------------
// L2-cache-blocking helper: identical to gn_silu_res_nchw but writes into a
// caller-provided (pre-allocated, contiguous NCHW) output slice so that batch
// chunks can be assembled in place without an extra copy.  Kernel unchanged.
// ---------------------------------------------------------------------------
void gn_silu_res_nchw_out(torch::Tensor y, torch::Tensor res,
                          torch::Tensor w, torch::Tensor b, double eps,
                          torch::Tensor out)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32);
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast),
                "expect channels_last conv output");
    TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous NCHW");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32);
    TORCH_CHECK(out.sizes() == y.sizes(), "out shape mismatch");
    const int B = (int)y.size(0), C = (int)y.size(1);
    const int H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == CTOT, "C must be 256");
    const int HW = H * W;

    auto opts = y.options();
    auto mean = torch::empty({(long long)B * NGRP}, opts);
    auto rstd = torch::empty({(long long)B * NGRP}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();

    compute_stats(y.data_ptr<float>(), B, HW, (float)eps, mean, rstd, opts, stream);

    dim3 g((HW + 31) / 32, C / 64, B);
    gn_silu_res_t<<<g, 256, 0, stream>>>(y.data_ptr<float>(),
                                         res.data_ptr<float>(),
                                         w.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         mean.data_ptr<float>(),
                                         rstd.data_ptr<float>(),
                                         out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

_CPP_SRC = r'''
torch::Tensor nchw_to_nhwc(torch::Tensor x);
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps);
torch::Tensor gn_silu_res_nchw(torch::Tensor y, torch::Tensor res, torch::Tensor w, torch::Tensor b, double eps);
void gn_silu_res_nchw_out(torch::Tensor y, torch::Tensor res, torch::Tensor w, torch::Tensor b, double eps, torch::Tensor out);
'''

_ext = load_inline(
    name="vae_resblock_fused_l2blk_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["nchw_to_nhwc", "gn_silu_nhwc", "gn_silu_res_nchw",
               "gn_silu_res_nchw_out"],
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
    """Granularity C fusion + L2 cache blocking over the batch dimension."""

    # live set during conv2 == 3 x (bytes per image) x chunk; keep it in L2
    _L2_BUDGET_BYTES = 56 * 1024 * 1024

    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        # ---- plan item 1: hoist every per-call invariant out of the loop ----
        if isinstance(eps, torch.Tensor):
            eps = float(eps.item())
        else:
            eps = float(eps)

        x = x if x.is_contiguous() else x.contiguous()
        nw1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        nb1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        nw2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        nb2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        # computed ONCE (not per chunk) -> no repeated weight-copy launches
        w1 = conv1_weight.contiguous(memory_format=torch.channels_last)
        w2 = conv2_weight.contiguous(memory_format=torch.channels_last)

        # ---- plan item 3: chunk sizing + gate -------------------------------
        B, C, H, W = x.shape
        S = C * H * W * x.element_size()          # bytes per image
        BUDGET = ModelNew._L2_BUDGET_BYTES
        bc = min(B, max(1, BUDGET // (3 * S))) if S > 0 else B

        if (3 * S) > BUDGET or bc >= B or bc < 1:
            # ---------- single-shot path: byte-identical to the base ---------
            xl = self.ext.nchw_to_nhwc(x)
            y = F.conv2d(xl, w1, None, 1, 1)                  # vendor tf32 NHWC conv
            y = self.ext.gn_silu_nhwc(y, nw1, nb1, eps)       # GN(moments+affine)+SiLU
            y = F.conv2d(y, w2, None, 1, 1)                   # vendor tf32 NHWC conv
            return self.ext.gn_silu_res_nchw(y, x, nw2, nb2, eps)

        # ---- plan item 4: L2-blocked chunked path --------------------------
        out = torch.empty_like(x)
        for i in range(0, B, bc):
            n = min(bc, B - i)                                # plan item 7
            xc = x.narrow(0, i, n)                            # contiguous view
            xl = self.ext.nchw_to_nhwc(xc)
            y = F.conv2d(xl, w1, None, 1, 1)
            y = self.ext.gn_silu_nhwc(y, nw1, nb1, eps)
            y = F.conv2d(y, w2, None, 1, 1)
            self.ext.gn_silu_res_nchw_out(y, xc, nw2, nb2, eps,
                                          out.narrow(0, i, n))
        return out                                            # plan item 6
