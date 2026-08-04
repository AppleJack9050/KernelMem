# ==========================================================================
# ModelNew — SOL-ExecBench 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY:  (C) fuse many ops into one/few kernels
#
# 1) Chosen granularity: (C).  Every non-conv op in the reference graph is
#    fused away into 4 custom CUDA kernels; the two 3x3 convolutions stay as
#    vendor (cuDNN) calls issued *from inside the extension* via at::conv2d.
#
# 2) Ops replaced by custom CUDA:
#      - the NCHW->NHWC staging of x            (custom tiled transpose kernel)
#      - group_norm #1 (RowwiseMoments + fused-params + affine)   -> 3 kernels
#      - silu #1                                 (fused into the GN apply)
#      - group_norm #2 statistics/affine         (same 2 stat kernels reused)
#      - silu #2 + residual add + NHWC->NCHW     (single fused epilogue kernel)
#    This removes the 4x cudnn::nchwToNhwcKernel / 2x nhwcToNchwKernel layout
#    conversions (21% of reference time), the RowwiseMoments kernels (10.6%)
#    and all 4 elementwise kernels (27.5%) from the profile.
#
# 3) Fusion map:
#      K1 nchw2nhwc_kernel        : x (NCHW) -> x (NHWC), 32x32 shared tiles
#      cuDNN conv2d (NHWC, TF32)  : conv1, conv2  (vendor, left in place)
#      K2 gn_stats_kernel         : per-(n,group) sum/sumsq partials, ALL 32
#                                   groups accumulated per block so global
#                                   reads stay fully coalesced (vec4)
#      K3 gn_finalize_kernel      : partial reduce (double) -> per-(n,channel)
#                                   scale = rstd*gamma, shift = beta-mean*rstd*gamma
#      K4 gn_silu_apply_kernel    : y = silu(y*scale+shift), NHWC, float4, in
#                                   place on our own temp (GN1+SiLU1)
#      K5 gn_silu_res_t_kernel    : GN2 affine + SiLU + residual add + NHWC->NCHW
#                                   transpose, all in one shared-memory tile pass
#                                   (transpose is free: both sides stay coalesced)
#
# 4) What stays in PyTorch/vendor and why:
#      - conv2d x2 : sm90 implicit-GEMM TF32 tensor-core kernel is at roofline;
#                    re-implementing wins nothing, so we only remove the layout
#                    conversions around it by feeding/consuming NHWC directly.
#      - nothing else remains: forward is one extension call.
#
# Precision: everything is float32 storage+math; reductions accumulate in
# float per-thread then double across blocks.  TF32 is used only where the
# reference already uses it (cuDNN conv, torch default allow_tf32=True).
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define TILE 32
#define BLK_ROWS 8

// ---------------------------------------------------------------- K1
// NCHW -> NHWC : per batch this is a (C x HW) -> (HW x C) transpose
__global__ void nchw2nhwc_kernel(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 int C, int HW) {
    __shared__ float tile[TILE][TILE + 1];   // [c_local][p_local]
    const int p0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int n  = blockIdx.z;
    const float* s = src + (size_t)n * (size_t)C * (size_t)HW;
    float*       d = dst + (size_t)n * (size_t)C * (size_t)HW;

    #pragma unroll
    for (int i = 0; i < TILE; i += BLK_ROWS) {
        int c = c0 + threadIdx.y + i;
        int p = p0 + threadIdx.x;
        float v = 0.f;
        if (c < C && p < HW) v = s[(size_t)c * (size_t)HW + p];
        tile[threadIdx.y + i][threadIdx.x] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < TILE; i += BLK_ROWS) {
        int p = p0 + threadIdx.y + i;
        int c = c0 + threadIdx.x;
        if (c < C && p < HW) d[(size_t)p * (size_t)C + c] = tile[threadIdx.x][threadIdx.y + i];
    }
}

// ---------------------------------------------------------------- K2
// GroupNorm partial statistics on an NHWC tensor.
// grid = (T, B), block = 256 threads.  Each block walks a slab of pixels and
// accumulates sum / sumsq for ALL groups, so every global read is a full
// coalesced float4 stream (no strided per-group gather).
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                int C, int HW, int G, int Cg,
                                int T, int pixPerBlock) {
    extern __shared__ float sh[];
    float* ss = sh;
    float* sq = sh + G;

    const int t = blockIdx.x;
    const int n = blockIdx.y;

    for (int i = threadIdx.x; i < G; i += blockDim.x) { ss[i] = 0.f; sq[i] = 0.f; }
    __syncthreads();

    const int VPP     = C >> 2;                 // float4 units per pixel
    const int pixIter = blockDim.x / VPP;       // pixels handled per iteration
    const int v       = threadIdx.x % VPP;
    const int pOff    = threadIdx.x / VPP;
    const int c       = v << 2;
    const int g       = c / Cg;

    const int pStart = t * pixPerBlock;
    const int pEnd   = min(pStart + pixPerBlock, HW);

    const float4* base = reinterpret_cast<const float4*>(y) +
                         (size_t)n * (size_t)HW * (size_t)VPP;

    float s = 0.f, q = 0.f;
    for (int p = pStart + pOff; p < pEnd; p += pixIter) {
        float4 a = base[(size_t)p * (size_t)VPP + v];
        s += a.x + a.y + a.z + a.w;
        q += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    atomicAdd(&ss[g], s);
    atomicAdd(&sq[g], q);
    __syncthreads();

    for (int i = threadIdx.x; i < G; i += blockDim.x) {
        psum[((size_t)n * G + i) * (size_t)T + t] = ss[i];
        psq [((size_t)n * G + i) * (size_t)T + t] = sq[i];
    }
}

// ---------------------------------------------------------------- K3
// Reduce partials -> per-(batch, channel) scale / shift.
__global__ void gn_finalize_kernel(const float* __restrict__ psum,
                                   const float* __restrict__ psq,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ scale,
                                   float* __restrict__ shift,
                                   int T, int G, int Cg, int C,
                                   double eps, double inv_cnt) {
    __shared__ double sh_s[32];
    __shared__ double sh_q[32];

    const int idx = blockIdx.x;          // n*G + g
    const int n   = idx / G;
    const int g   = idx - n * G;

    const float* ps = psum + (size_t)idx * (size_t)T;
    const float* pq = psq  + (size_t)idx * (size_t)T;

    double s = 0.0, q = 0.0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        s += (double)ps[t];
        q += (double)pq[t];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
        q += __shfl_down_sync(0xffffffffu, q, off);
    }
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) { sh_s[wid] = s; sh_q[wid] = q; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nw = blockDim.x >> 5;
        double ts = 0.0, tq = 0.0;
        for (int i = 0; i < nw; ++i) { ts += sh_s[i]; tq += sh_q[i]; }
        sh_s[0] = ts; sh_q[0] = tq;
    }
    __syncthreads();

    const double mean = sh_s[0] * inv_cnt;
    double var = sh_q[0] * inv_cnt - mean * mean;
    if (var < 0.0) var = 0.0;
    const float rstd = (float)(1.0 / sqrt(var + eps));
    const float m    = (float)mean;

    for (int i = threadIdx.x; i < Cg; i += blockDim.x) {
        int c = g * Cg + i;
        float gm = gamma[c];
        float bt = beta[c];
        scale[(size_t)n * C + c] = rstd * gm;
        shift[(size_t)n * C + c] = bt - m * rstd * gm;
    }
}

__device__ __forceinline__ float silu_f(float t) {
    return t / (1.f + __expf(-t));
}

// ---------------------------------------------------------------- K4
// y = silu(y*scale + shift) on NHWC, in place, float4.
__global__ void gn_silu_apply_kernel(float* __restrict__ y,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ shift,
                                     int VPP, int nvecPerBatch) {
    const int n = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nvecPerBatch) return;

    float4* yb = reinterpret_cast<float4*>(y) + (size_t)n * (size_t)nvecPerBatch;
    const float4* sc = reinterpret_cast<const float4*>(scale) + (size_t)n * (size_t)VPP;
    const float4* sf = reinterpret_cast<const float4*>(shift) + (size_t)n * (size_t)VPP;

    const int v = i % VPP;
    float4 a = yb[i];
    float4 s = sc[v];
    float4 t = sf[v];
    a.x = silu_f(a.x * s.x + t.x);
    a.y = silu_f(a.y * s.y + t.y);
    a.z = silu_f(a.z * s.z + t.z);
    a.w = silu_f(a.w * s.w + t.w);
    yb[i] = a;
}

// ---------------------------------------------------------------- K5
// out(NCHW) = silu(y(NHWC)*scale + shift) + residual(NCHW)
// fused with the NHWC->NCHW transpose (shared-memory tiles keep both the
// read side and the write side fully coalesced).
__global__ void gn_silu_res_t_kernel(const float* __restrict__ y,
                                     const float* __restrict__ res,
                                     float* __restrict__ out,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ shift,
                                     int C, int HW) {
    __shared__ float tile[TILE][TILE + 1];   // [p_local][c_local]
    const int p0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int n  = blockIdx.z;

    const float* yb = y     + (size_t)n * (size_t)HW * (size_t)C;
    const float* sc = scale + (size_t)n * (size_t)C;
    const float* sf = shift + (size_t)n * (size_t)C;

    #pragma unroll
    for (int i = 0; i < TILE; i += BLK_ROWS) {
        int p = p0 + threadIdx.y + i;
        int c = c0 + threadIdx.x;
        float v = 0.f;
        if (p < HW && c < C) {
            float t = yb[(size_t)p * (size_t)C + c] * sc[c] + sf[c];
            v = silu_f(t);
        }
        tile[threadIdx.y + i][threadIdx.x] = v;
    }
    __syncthreads();

    float*       ob = out + (size_t)n * (size_t)C * (size_t)HW;
    const float* rb = res + (size_t)n * (size_t)C * (size_t)HW;
    #pragma unroll
    for (int i = 0; i < TILE; i += BLK_ROWS) {
        int c = c0 + threadIdx.y + i;
        int p = p0 + threadIdx.x;
        if (p < HW && c < C) {
            size_t off = (size_t)c * (size_t)HW + p;
            ob[off] = tile[threadIdx.x][threadIdx.y + i] + rb[off];
        }
    }
}

// ================================================================ host side
torch::Tensor fused_res_block(torch::Tensor x,
                              torch::Tensor conv1_weight,
                              torch::Tensor norm1_weight,
                              torch::Tensor norm1_bias,
                              torch::Tensor conv2_weight,
                              torch::Tensor norm2_weight,
                              torch::Tensor norm2_bias,
                              double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");

    const int B  = (int)x.size(0);
    const int C  = (int)x.size(1);
    const int H  = (int)x.size(2);
    const int W  = (int)x.size(3);
    const int HW = H * W;
    const int G  = 32;
    TORCH_CHECK(C % G == 0, "C must be divisible by 32");
    const int Cg  = C / G;
    const int VPP = C / 4;
    TORCH_CHECK(C % 4 == 0 && Cg % 4 == 0 && VPP <= 256 && (256 % VPP) == 0,
                "unsupported channel count");

    auto stream = at::cuda::getCurrentCUDAStream();   // default stream in bench

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto g1 = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
    auto b1 = norm1_bias.is_contiguous()   ? norm1_bias   : norm1_bias.contiguous();
    auto g2 = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
    auto b2 = norm2_bias.is_contiguous()   ? norm2_bias   : norm2_bias.contiguous();

    auto fopts = x.options();
    auto cl    = fopts.memory_format(at::MemoryFormat::ChannelsLast);

    // ---- K1 : stage x into NHWC once (kills 4x cudnn nchwToNhwc) ----------
    auto x_nhwc = at::empty({B, C, H, W}, cl);
    {
        dim3 grid((HW + TILE - 1) / TILE, (C + TILE - 1) / TILE, B);
        dim3 block(TILE, BLK_ROWS);
        nchw2nhwc_kernel<<<grid, block, 0, stream>>>(
            xc.data_ptr<float>(), x_nhwc.data_ptr<float>(), C, HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = conv1_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = conv2_weight.contiguous(at::MemoryFormat::ChannelsLast);

    std::vector<int64_t> st{1, 1}, pd{1, 1}, dl{1, 1};

    // ---- conv1 (vendor cuDNN, NHWC in -> NHWC out) -----------------------
    auto y = at::conv2d(x_nhwc, w1c, at::Tensor(), st, pd, dl, 1);
    if (!y.is_contiguous(at::MemoryFormat::ChannelsLast))
        y = y.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- statistics geometry (no untouched tail: T = ceil(HW/per)) -------
    const int64_t totalPix = (int64_t)B * (int64_t)HW;
    int per = (int)std::max<int64_t>(64, (totalPix + 2047) / 2048);
    per = ((per + 63) / 64) * 64;
    if (per > HW) per = ((HW + 63) / 64) * 64;
    const int T = (HW + per - 1) / per;

    auto psum = at::empty({(int64_t)B * G * T}, fopts);
    auto psq  = at::empty({(int64_t)B * G * T}, fopts);
    auto scale = at::empty({(int64_t)B * C}, fopts);
    auto shift = at::empty({(int64_t)B * C}, fopts);

    const double inv_cnt = 1.0 / ((double)Cg * (double)HW);

    // ---- K2/K3/K4 : GroupNorm1 + SiLU1 (in place, NHWC) ------------------
    {
        dim3 grid(T, B);
        gn_stats_kernel<<<grid, 256, 2 * G * sizeof(float), stream>>>(
            y.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
            C, HW, G, Cg, T, per);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        gn_finalize_kernel<<<B * G, 128, 0, stream>>>(
            psum.data_ptr<float>(), psq.data_ptr<float>(),
            g1.data_ptr<float>(), b1.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(),
            T, G, Cg, C, eps, inv_cnt);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int nvecPerBatch = HW * VPP;
        dim3 grid2((nvecPerBatch + 255) / 256, B);
        gn_silu_apply_kernel<<<grid2, 256, 0, stream>>>(
            y.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
            VPP, nvecPerBatch);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (vendor cuDNN, NHWC in -> NHWC out) -----------------------
    auto z = at::conv2d(y, w2c, at::Tensor(), st, pd, dl, 1);
    if (!z.is_contiguous(at::MemoryFormat::ChannelsLast))
        z = z.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- K2/K3 : GroupNorm2 statistics ----------------------------------
    {
        dim3 grid(T, B);
        gn_stats_kernel<<<grid, 256, 2 * G * sizeof(float), stream>>>(
            z.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
            C, HW, G, Cg, T, per);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        gn_finalize_kernel<<<B * G, 128, 0, stream>>>(
            psum.data_ptr<float>(), psq.data_ptr<float>(),
            g2.data_ptr<float>(), b2.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(),
            T, G, Cg, C, eps, inv_cnt);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- K5 : affine + SiLU + residual + NHWC->NCHW ----------------------
    auto out = at::empty({B, C, H, W}, fopts);
    {
        dim3 grid((HW + TILE - 1) / TILE, (C + TILE - 1) / TILE, B);
        dim3 block(TILE, BLK_ROWS);
        gn_silu_res_t_kernel<<<grid, block, 0, stream>>>(
            z.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(), C, HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor fused_res_block(torch::Tensor x,
                              torch::Tensor conv1_weight,
                              torch::Tensor norm1_weight,
                              torch::Tensor norm1_bias,
                              torch::Tensor conv2_weight,
                              torch::Tensor norm2_weight,
                              torch::Tensor norm2_bias,
                              double eps);
'''

_ext = load_inline(
    name="vae_res_block_fused_c",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["fused_res_block"],
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
    # Granularity (C): all non-conv ops fused into 4 custom CUDA kernels;
    # the two 3x3 convs remain vendor cuDNN calls issued from the extension,
    # but now consume/produce NHWC directly so cuDNN's layout-conversion
    # kernels disappear entirely.  See module header for the full fusion map.
    def __init__(self):
        super().__init__()
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        with torch.no_grad():
            C = x.size(1)
            if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                    and C % 32 == 0 and (C // 32) % 4 == 0
                    and (C // 4) <= 256 and (256 % (C // 4)) == 0):
                return self._ext.fused_res_block(
                    x, conv1_weight, norm1_weight, norm1_bias,
                    conv2_weight, norm2_weight, norm2_bias, float(eps))
            # off-spec safety path (never taken for this problem's shapes)
            out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=eps)
            out = F.silu(out)
            out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=eps)
            out = F.silu(out)
            return out + x
