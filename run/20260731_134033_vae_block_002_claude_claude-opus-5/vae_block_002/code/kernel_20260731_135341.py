# =============================================================================
# ModelNew — SOL-ExecBench 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY: (D) FULL FORWARD REWRITE.
#   The whole reference `run()` body (2x [Conv3x3 -> GroupNorm(32) -> SiLU] +
#   residual add) is executed by ONE C++/CUDA entry point `fused_resblock`
#   built with load_inline; Python only dispatches into it.
#
# 1) Chosen granularity: (D) fully rewrite forward.
#
# 2) Ops replaced (all of them, inside the extension):
#      - layout handling  : explicit NCHW->NHWC (channels_last) staging of x and
#                           of both conv weights, done ONCE, so cuDNN never has
#                           to insert its own nchwToNhwc / nhwcToNchw kernels
#                           (those were 20.8% = 308 us of the reference profile).
#      - F.conv2d x2      : at::conv2d called FROM INSIDE the extension on
#                           channels_last tensors -> cuDNN NHWC TF32 implicit
#                           GEMM (the same vendor kernel the reference reaches,
#                           but with zero surrounding layout conversions).
#      - F.group_norm x2  : custom 2-kernel split reduction
#                           (gn_stats_kernel -> gn_finalize_kernel), float4
#                           coalesced NHWC loads, per-channel partials folded
#                           into per-group partials in shared memory, final
#                           accumulation in double.  Split over spatial blocks
#                           so the B*32 group-count never collapses occupancy.
#      - F.silu x2        : fused into the two normalization epilogue kernels.
#      - residual add     : fused into the last epilogue kernel.
#
# 3) Fusion map:
#      k1 gn_stats_kernel        : NHWC float4 read of conv output -> per-(n,g)
#                                  partial sum / sum-of-squares (no atomics,
#                                  every block always writes its slot, so there
#                                  is no unwritten partial-buffer tail when
#                                  H*W is not a multiple of the tile).
#      k2 gn_finalize_kernel     : partials -> mean/rstd per (n,group), double acc.
#      k3 gn_silu_nhwc_kernel    : (normalize + affine + SiLU) of conv1 output,
#                                  stays NHWC to feed conv2 directly.
#      k4 gn_silu_res_nchw_kernel: (normalize + affine + SiLU + residual add) of
#                                  conv2 output AND the NHWC->NCHW transpose of
#                                  the result, via a padded 32-pixel shared-memory
#                                  tile -> final output is plain contiguous NCHW
#                                  with no extra pass over memory.
#
# 4) What stays in PyTorch / vendor land and why:
#      - conv2d itself      : cuDNN NHWC TF32 implicit GEMM is at/near roofline
#                             for 3x3 C=256; owning it buys nothing, the win is
#                             removing the layout traffic and glue around it.
#      - x.contiguous(channels_last) : one vendor transpose replaces the four the
#                             reference paid for.
#      - a pure-PyTorch fallback path is kept for shapes/dtypes outside the
#                             fast-path contract (C%128!=0, non-fp32, non-CUDA).
#
# PRECISION: everything stays fp32 (storage + arithmetic); GroupNorm reductions
# accumulate fp32 partials then double.  TF32 is used only in conv, exactly as
# the reference does (cudnn.allow_tf32 default True).
# =============================================================================

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
#include <vector>
#include <algorithm>

#define TILE_W 32   // pixels per tile in the NHWC->NCHW epilogue (== warpSize)

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + __expf(-v));
}

// ---------------------------------------------------------------------------
// k1: per-(batch, group) partial sums over a spatial slice.
// blockDim = (C4, PT), grid = (nsb, B).  Layout of y: (B, H*W, C) = channels_last
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                int S, int C4, int cg4, int G)
{
    extern __shared__ float smem_stats[];
    const int nthr = blockDim.x * blockDim.y;
    float* ss = smem_stats;
    float* sq = smem_stats + nthr;

    const int c4 = threadIdx.x;
    const int ty = threadIdx.y;
    const int PT = blockDim.y;
    const int n  = blockIdx.y;
    const int C  = C4 << 2;
    const long long baseN = (long long)n * (long long)S * (long long)C;

    float s = 0.0f, q = 0.0f;
    for (int p = blockIdx.x * PT + ty; p < S; p += gridDim.x * PT) {
        const float4 v = *reinterpret_cast<const float4*>(
            y + baseN + (long long)p * (long long)C + (c4 << 2));
        s += v.x + v.y + v.z + v.w;
        q += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    const int tid = ty * blockDim.x + c4;
    ss[tid] = s;
    sq[tid] = q;
    __syncthreads();

    if (tid < G) {
        float gs = 0.0f, gq = 0.0f;
        for (int t = 0; t < PT; ++t) {
            const int rowb = t * blockDim.x + tid * cg4;
            for (int k = 0; k < cg4; ++k) {
                gs += ss[rowb + k];
                gq += sq[rowb + k];
            }
        }
        const long long o = ((long long)(blockIdx.y * gridDim.x + blockIdx.x)) * (long long)G + tid;
        psum[o] = gs;
        psq[o]  = gq;
    }
}

// ---------------------------------------------------------------------------
// k2: reduce spatial partials -> mean / rstd per (batch, group).  grid = B*G.
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ psum,
                                   const float* __restrict__ psq,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nsb, int G, double invCnt, double eps)
{
    __shared__ double shs[128];
    __shared__ double shq[128];

    const int idx = blockIdx.x;
    const int n   = idx / G;
    const int g   = idx - n * G;

    double s = 0.0, q = 0.0;
    for (int b = threadIdx.x; b < nsb; b += blockDim.x) {
        const long long o = ((long long)n * nsb + b) * (long long)G + g;
        s += (double)psum[o];
        q += (double)psq[o];
    }
    shs[threadIdx.x] = s;
    shq[threadIdx.x] = q;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) {
            shs[threadIdx.x] += shs[threadIdx.x + st];
            shq[threadIdx.x] += shq[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double m = shs[0] * invCnt;
        double v = shq[0] * invCnt - m * m;
        if (v < 0.0) v = 0.0;
        mean[idx] = (float)m;
        rstd[idx] = (float)(1.0 / sqrt(v + eps));
    }
}

// ---------------------------------------------------------------------------
// k3: normalize + affine + SiLU, NHWC in / NHWC out.
// blockDim = (C4, PT), grid = (gx, B)
// ---------------------------------------------------------------------------
__global__ void gn_silu_nhwc_kernel(const float* __restrict__ y,
                                    float* __restrict__ out,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ rstd,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    int S, int C4, int cg4, int G)
{
    const int c4 = threadIdx.x;
    const int C  = C4 << 2;
    const int g  = c4 / cg4;
    const int n  = blockIdx.y;

    const float4 gm = *reinterpret_cast<const float4*>(gamma + (c4 << 2));
    const float4 bt = *reinterpret_cast<const float4*>(beta  + (c4 << 2));
    const float m = mean[n * G + g];
    const float r = rstd[n * G + g];
    const long long baseN = (long long)n * (long long)S * (long long)C;

    for (int p = blockIdx.x * blockDim.y + threadIdx.y; p < S;
         p += gridDim.x * blockDim.y) {
        const long long o = baseN + (long long)p * (long long)C + (c4 << 2);
        const float4 v = *reinterpret_cast<const float4*>(y + o);
        float4 res;
        res.x = silu_f((v.x - m) * r * gm.x + bt.x);
        res.y = silu_f((v.y - m) * r * gm.y + bt.y);
        res.z = silu_f((v.z - m) * r * gm.z + bt.z);
        res.w = silu_f((v.w - m) * r * gm.w + bt.w);
        *reinterpret_cast<float4*>(out + o) = res;
    }
}

// ---------------------------------------------------------------------------
// k4: normalize + affine + SiLU + residual, NHWC in -> NCHW out (smem transpose)
// blockDim = (C4, PT), grid = (ceil(S/TILE_W), B), shared = TILE_W*(C+1) floats
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_nchw_kernel(const float* __restrict__ y,
                                        const float* __restrict__ xres,
                                        float* __restrict__ out,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ rstd,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        int S, int C4, int cg4, int G)
{
    extern __shared__ float tile[];             // TILE_W * (C+1)

    const int c4 = threadIdx.x;
    const int C  = C4 << 2;
    const int ld = C + 1;                       // odd stride -> conflict-free reads
    const int g  = c4 / cg4;
    const int n  = blockIdx.y;
    const int hw0 = blockIdx.x * TILE_W;

    const float4 gm = *reinterpret_cast<const float4*>(gamma + (c4 << 2));
    const float4 bt = *reinterpret_cast<const float4*>(beta  + (c4 << 2));
    const float m = mean[n * G + g];
    const float r = rstd[n * G + g];
    const long long baseN = (long long)n * (long long)S * (long long)C;

    for (int j = threadIdx.y; j < TILE_W; j += blockDim.y) {
        const int p = hw0 + j;
        if (p < S) {
            const long long o = baseN + (long long)p * (long long)C + (c4 << 2);
            const float4 v = *reinterpret_cast<const float4*>(y + o);
            const int so = j * ld + (c4 << 2);
            tile[so + 0] = silu_f((v.x - m) * r * gm.x + bt.x);
            tile[so + 1] = silu_f((v.y - m) * r * gm.y + bt.y);
            tile[so + 2] = silu_f((v.z - m) * r * gm.z + bt.z);
            tile[so + 3] = silu_f((v.w - m) * r * gm.w + bt.w);
        }
    }
    __syncthreads();

    const int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int nwarps = (blockDim.x * blockDim.y) >> 5;

    const int p = hw0 + lane;                   // TILE_W == 32
    if (p < S) {
        const long long base = (long long)n * (long long)C * (long long)S + p;
        for (int c = warp; c < C; c += nwarps) {
            const long long o = base + (long long)c * (long long)S;
            out[o] = tile[lane * ld + c] + xres[o];
        }
    }
}

// ---------------------------------------------------------------------------
static inline int calc_nsb(int B, int S, int PT) {
    const int target = 448;
    int nsb = (target + B - 1) / B;
    const int maxb = (S + PT - 1) / PT;
    if (nsb > maxb) nsb = maxb;
    if (nsb < 1) nsb = 1;
    return nsb;
}

static void run_gn_stats(const float* y, float* psum, float* psq,
                         int B, int S, int C4, int cg4, int G, int nsb,
                         cudaStream_t stream)
{
    dim3 blk(C4, 4);
    dim3 grd(nsb, B);
    size_t shm = (size_t)2 * (size_t)(C4 * 4) * sizeof(float);
    gn_stats_kernel<<<grd, blk, shm, stream>>>(y, psum, psq, S, C4, cg4, G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                             torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                             double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "fp32 only");
    TORCH_CHECK(x.dim() == 4, "NCHW expected");

    const int B = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int S = H * W;
    const int G = 32;

    TORCH_CHECK(C % 128 == 0 && C <= 256, "fast path needs C%128==0 && C<=256");
    const int C4  = C / 4;
    const int cg  = C / G;
    const int cg4 = cg / 4;

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto xl = xc.contiguous(at::MemoryFormat::ChannelsLast);
    auto w1l = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2l = w2.contiguous(at::MemoryFormat::ChannelsLast);
    auto gam1 = g1.is_contiguous() ? g1 : g1.contiguous();
    auto bet1 = b1.is_contiguous() ? b1 : b1.contiguous();
    auto gam2 = g2.is_contiguous() ? g2 : g2.contiguous();
    auto bet2 = b2.is_contiguous() ? b2 : b2.contiguous();

    std::vector<int64_t> stride1{1, 1};
    std::vector<int64_t> pad1{1, 1};
    std::vector<int64_t> dil1{1, 1};

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = xc.options();

    const int nsb = calc_nsb(B, S, 4);
    auto psum = torch::empty({B, nsb, G}, opts);
    auto psq  = torch::empty({B, nsb, G}, opts);
    auto mean = torch::empty({B, G}, opts);
    auto rstd = torch::empty({B, G}, opts);

    const double invCnt = 1.0 / ((double)cg * (double)S);

    // ---------------- stage 1 : conv -> GN -> SiLU (stays NHWC) -------------
    auto y1 = at::conv2d(xl, w1l, torch::Tensor(), stride1, pad1, dil1, 1);

    run_gn_stats(y1.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
                 B, S, C4, cg4, G, nsb, stream);

    gn_finalize_kernel<<<B * G, 128, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        nsb, G, invCnt, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto y1n = torch::empty_like(y1);
    {
        int gx = (S + 3) / 4;
        if (gx > 2048) gx = 2048;
        if (gx < 1) gx = 1;
        dim3 blk(C4, 4);
        dim3 grd(gx, B);
        gn_silu_nhwc_kernel<<<grd, blk, 0, stream>>>(
            y1.data_ptr<float>(), y1n.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gam1.data_ptr<float>(), bet1.data_ptr<float>(),
            S, C4, cg4, G);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---------------- stage 2 : conv -> GN -> SiLU + residual -> NCHW ------
    auto y2 = at::conv2d(y1n, w2l, torch::Tensor(), stride1, pad1, dil1, 1);

    run_gn_stats(y2.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
                 B, S, C4, cg4, G, nsb, stream);

    gn_finalize_kernel<<<B * G, 128, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        nsb, G, invCnt, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out = torch::empty({B, C, H, W}, opts);
    {
        dim3 blk(C4, 4);
        dim3 grd((S + TILE_W - 1) / TILE_W, B);
        size_t shm = (size_t)TILE_W * (size_t)(C + 1) * sizeof(float);
        gn_silu_res_nchw_kernel<<<grd, blk, shm, stream>>>(
            y2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gam2.data_ptr<float>(), bet2.data_ptr<float>(),
            S, C4, cg4, G);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                             torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                             double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_nhwc",
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
        "-gencode=arch=compute_90,code=sm_90",
    ],
)


class ModelNew(nn.Module):
    """Granularity (D): full forward rewrite; see header block at top of file."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

    @staticmethod
    def _reference(x, conv1_weight, norm1_weight, norm1_bias,
                   conv2_weight, norm2_weight, norm2_bias, eps):
        # Fallback for shapes/dtypes outside the fast-path contract.
        num_groups = 32
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
        out = F.silu(out)
        return out + residual

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        with torch.no_grad():
            fast = (
                x.is_cuda
                and x.dtype == torch.float32
                and x.dim() == 4
                and conv1_weight.dtype == torch.float32
                and conv2_weight.dtype == torch.float32
                and (x.size(1) % 128 == 0)
                and (x.size(1) <= 256)
            )
            if fast:
                return self._ext.fused_resblock(
                    x, conv1_weight, norm1_weight, norm1_bias,
                    conv2_weight, norm2_weight, norm2_bias, float(eps),
                )
            return self._reference(
                x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, float(eps),
            )
