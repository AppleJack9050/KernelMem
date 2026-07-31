# =============================================================================
# ModelNew — fused VAE residual block (Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +x)
#
# HEADER / PLAN (required):
# 1) CHOSEN GRANULARITY: (C) "fuse many ops into one/few kernels" + whole-forward
#    CUDA-GRAPH capture (launch/dispatch-bound at the segment level).
#
# 2) OPS REPLACED BY CUSTOM CUDA (unchanged from base):
#      - implicit NCHW->NHWC layout conversions -> `to_nhwc_cuda`
#      - group_norm #1 + silu                   -> gn_stats / gn_reduce / gn_apply_silu
#      - group_norm #2 + silu + residual add + NHWC->NCHW relayout -> one fused kernel
#
# 3) NEW IN THIS VERSION: the ENTIRE forward (transpose -> conv1 -> GN1+SiLU ->
#    conv2 -> GN2+SiLU+add+transpose) is captured into a single CUDA graph, keyed
#    by (shape, every input data_ptr, eps). Replay issues ONE graph launch instead
#    of 15 kernel launches + 2 cuDNN dispatches, removing the ~30% GPU-idle gap.
#    NO static input buffers and NO copy_ of inputs: the graph reads the caller's
#    tensors in place, so zero extra DRAM traffic is added and every kernel's
#    duration/DRAM-throughput is bit-for-bit unchanged.
#
# PRECISION: float32 storage/arithmetic throughout, float32 reductions.
# TF32 for conv only (cudnn.allow_tf32 default), exactly as the reference.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CCH 256          // channels (const per problem definition)
#define NGRP 32          // num_groups (const)
#define CPG  8           // channels per group

__device__ __forceinline__ float silu_f(float t) {
    return t / (1.0f + __expf(-t));
}

// ---------------------------------------------------------------------------
// K0: NCHW -> NHWC tiled transpose (32 pixels x 32 channels per block)
// ---------------------------------------------------------------------------
__global__ void to_nhwc_kernel(const float* __restrict__ x,
                               float* __restrict__ out,
                               int HW, int C) {
    __shared__ float sh[32][33];
    const int n  = blockIdx.z;
    const int cb = blockIdx.y * 32;
    const int pb = blockIdx.x * 32;
    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..7

    const float* xb = x + (long)n * C * HW;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c = cb + ty + i * 8;
        int p = pb + tx;
        float v = 0.0f;
        if (p < HW) v = xb[(long)c * HW + p];
        sh[ty + i * 8][tx] = v;
    }
    __syncthreads();

    float* ob = out + (long)n * (long)HW * C;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int p = pb + ty + i * 8;
        int c = cb + tx;
        if (p < HW) ob[(long)p * C + c] = sh[tx][ty + i * 8];
    }
}

// ---------------------------------------------------------------------------
// K1: per-(n, group) partial sums over a tile of pixels. NHWC, float4 loads.
//     blockDim = 256 : 64 threads (=256 ch via float4) x 4 pixels
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                int HW, int tile, int ntiles) {
    const int n = blockIdx.y;
    const int t = blockIdx.x;
    const int c4 = threadIdx.x & 63;     // float4 index within the 256 channels
    const int sp = threadIdx.x >> 6;     // 0..3 pixel phase

    const float4* y4 = (const float4*)(y + (long)n * (long)HW * CCH);

    int p0 = t * tile;
    int p1 = p0 + tile;
    if (p1 > HW) p1 = HW;

    float s = 0.0f, q = 0.0f;
    for (int p = p0 + sp; p < p1; p += 4) {
        float4 v = y4[(long)p * 64 + c4];
        s += v.x + v.y + v.z + v.w;
        q += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // channels 4*c4 .. 4*c4+3 all live in group (c4>>1); pair adjacent lanes.
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    q += __shfl_xor_sync(0xffffffffu, q, 1);

    __shared__ float ss[4][32];
    __shared__ float sq[4][32];
    if ((threadIdx.x & 1) == 0) {
        int g = c4 >> 1;                 // 0..31
        ss[sp][g] = s;
        sq[sp][g] = q;
    }
    __syncthreads();

    if (threadIdx.x < NGRP) {
        int g = threadIdx.x;
        float ts = ss[0][g] + ss[1][g] + ss[2][g] + ss[3][g];
        float tq = sq[0][g] + sq[1][g] + sq[2][g] + sq[3][g];
        long idx = ((long)n * NGRP + g) * ntiles + t;
        psum[idx] = ts;
        psq[idx]  = tq;
    }
}

// ---------------------------------------------------------------------------
// K2: reduce partials -> mean/rstd -> per-(n,channel) scale/shift
// ---------------------------------------------------------------------------
__global__ void gn_reduce_kernel(const float* __restrict__ psum,
                                 const float* __restrict__ psq,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ scale,
                                 float* __restrict__ shift,
                                 int ntiles, int HW, float eps) {
    const int ng = blockIdx.x;
    const int n  = ng / NGRP;
    const int g  = ng % NGRP;

    const float* pb_s = psum + (long)ng * ntiles;
    const float* pb_q = psq  + (long)ng * ntiles;

    float s = 0.0f, q = 0.0f;
    for (int i = threadIdx.x; i < ntiles; i += blockDim.x) {
        s += pb_s[i];
        q += pb_q[i];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
        q += __shfl_down_sync(0xffffffffu, q, off);
    }
    __shared__ float ws[8], wq[8];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) { ws[warp] = s; wq[warp] = q; }
    __syncthreads();

    if (threadIdx.x < CPG) {
        int nwarps = blockDim.x >> 5;
        float ts = 0.0f, tq = 0.0f;
        for (int i = 0; i < nwarps; ++i) { ts += ws[i]; tq += wq[i]; }
        float cnt  = (float)HW * (float)CPG;
        float mean = ts / cnt;
        float var  = tq / cnt - mean * mean;
        if (var < 0.0f) var = 0.0f;
        float rstd = rsqrtf(var + eps);
        int c = g * CPG + (int)threadIdx.x;
        float gm = gamma[c];
        float bt = beta[c];
        scale[n * CCH + c] = rstd * gm;
        shift[n * CCH + c] = bt - mean * rstd * gm;
    }
}

// ---------------------------------------------------------------------------
// K3: GN affine + SiLU, NHWC in / NHWC out, float4
// ---------------------------------------------------------------------------
__global__ void gn_apply_silu_nhwc_kernel(const float* __restrict__ y,
                                          const float* __restrict__ scale,
                                          const float* __restrict__ shift,
                                          float* __restrict__ out,
                                          int HW) {
    const int n = blockIdx.y;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)HW * 64;
    if (i >= total) return;

    const float4* y4 = (const float4*)(y   + (long)n * (long)HW * CCH);
    float4*       o4 = (float4*)      (out + (long)n * (long)HW * CCH);
    const float4* s4 = (const float4*)(scale + (long)n * CCH);
    const float4* b4 = (const float4*)(shift + (long)n * CCH);

    int c4 = (int)(i & 63);
    float4 v = y4[i];
    float4 s = s4[c4];
    float4 b = b4[c4];
    float4 r;
    r.x = silu_f(v.x * s.x + b.x);
    r.y = silu_f(v.y * s.y + b.y);
    r.z = silu_f(v.z * s.z + b.z);
    r.w = silu_f(v.w * s.w + b.w);
    o4[i] = r;
}

// ---------------------------------------------------------------------------
// K4: GN affine + SiLU + residual add + NHWC->NCHW transpose (one pass)
// ---------------------------------------------------------------------------
__global__ void gn_apply_silu_add_t_kernel(const float* __restrict__ y,     // NHWC
                                           const float* __restrict__ res,   // NCHW
                                           const float* __restrict__ scale,
                                           const float* __restrict__ shift,
                                           float* __restrict__ out,         // NCHW
                                           int HW) {
    __shared__ float sh[32][33];
    const int n  = blockIdx.z;
    const int cb = blockIdx.y * 32;
    const int pb = blockIdx.x * 32;
    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..7

    const float* yb = y + (long)n * (long)HW * CCH;
    const int c_in = cb + tx;
    const float sc = scale[n * CCH + c_in];
    const float sf = shift[n * CCH + c_in];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int p = pb + ty + i * 8;
        float o = 0.0f;
        if (p < HW) {
            float v = yb[(long)p * CCH + c_in];
            o = silu_f(v * sc + sf);
        }
        sh[ty + i * 8][tx] = o;
    }
    __syncthreads();

    const float* rb = res + (long)n * CCH * (long)HW;
    float*       ob = out + (long)n * CCH * (long)HW;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c = cb + ty + i * 8;
        int p = pb + tx;
        if (p < HW) {
            ob[(long)c * HW + p] = sh[tx][ty + i * 8] + rb[(long)c * HW + p];
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static void compute_affine(const at::Tensor& y, const at::Tensor& gamma,
                           const at::Tensor& beta, int N, int HW, float eps,
                           at::Tensor& scale, at::Tensor& shift) {
    int bpi = (HW + 15) / 16;
    int cap = std::max(1, 8192 / std::max(1, N));
    if (bpi > cap) bpi = cap;
    int tile = (HW + bpi - 1) / bpi;
    tile = ((tile + 3) / 4) * 4;              // multiple of 4 pixels
    if (tile < 4) tile = 4;
    bpi = (HW + tile - 1) / tile;             // no empty tile

    auto opts = y.options();
    auto psum = at::empty({(long)N * NGRP * bpi}, opts);
    auto psq  = at::empty({(long)N * NGRP * bpi}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    gn_stats_kernel<<<dim3(bpi, N), 256, 0, stream>>>(
        y.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
        HW, tile, bpi);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_reduce_kernel<<<N * NGRP, 256, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        bpi, HW, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor to_nhwc_cuda(at::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "float32 cuda tensor expected");
    TORCH_CHECK(x.dim() == 4, "4d tensor expected");
    int N = (int)x.size(0), C = (int)x.size(1);
    int H = (int)x.size(2), W = (int)x.size(3);
    TORCH_CHECK(C % 32 == 0, "channels must be a multiple of 32");
    int HW = H * W;
    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto out = at::empty_like(xc, xc.options(), at::MemoryFormat::ChannelsLast);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((HW + 31) / 32, C / 32, N);
    dim3 blk(32, 8);
    to_nhwc_kernel<<<grid, blk, 0, stream>>>(xc.data_ptr<float>(), out.data_ptr<float>(), HW, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor gn_silu_nhwc_cuda(at::Tensor y, at::Tensor gamma, at::Tensor beta, double eps) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kFloat, "float32 cuda tensor expected");
    TORCH_CHECK(y.dim() == 4 && y.size(1) == CCH, "expected (N,256,H,W)");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "expected channels_last");
    int N = (int)y.size(0);
    int HW = (int)(y.size(2) * y.size(3));

    auto scale = at::empty({(long)N * CCH}, y.options());
    auto shift = at::empty({(long)N * CCH}, y.options());
    compute_affine(y, gamma, beta, N, HW, (float)eps, scale, shift);

    auto out = at::empty_like(y);
    auto stream = at::cuda::getCurrentCUDAStream();
    long per_img = (long)HW * 64;
    int blocks = (int)((per_img + 255) / 256);
    dim3 grid(blocks, N);
    gn_apply_silu_nhwc_kernel<<<grid, 256, 0, stream>>>(
        y.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
        out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor gn_silu_add_nchw_cuda(at::Tensor y, at::Tensor res, at::Tensor gamma,
                                 at::Tensor beta, double eps) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kFloat, "float32 cuda tensor expected");
    TORCH_CHECK(y.dim() == 4 && y.size(1) == CCH, "expected (N,256,H,W)");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "expected channels_last");
    TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");
    int N = (int)y.size(0);
    int H = (int)y.size(2), W = (int)y.size(3);
    int HW = H * W;

    auto scale = at::empty({(long)N * CCH}, y.options());
    auto shift = at::empty({(long)N * CCH}, y.options());
    compute_affine(y, gamma, beta, N, HW, (float)eps, scale, shift);

    auto out = at::empty({(long)N, (long)CCH, (long)H, (long)W}, res.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((HW + 31) / 32, CCH / 32, N);
    dim3 blk(32, 8);
    gn_apply_silu_add_t_kernel<<<grid, blk, 0, stream>>>(
        y.data_ptr<float>(), res.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CPP_SRC = r"""
at::Tensor to_nhwc_cuda(at::Tensor x);
at::Tensor gn_silu_nhwc_cuda(at::Tensor y, at::Tensor gamma, at::Tensor beta, double eps);
at::Tensor gn_silu_add_nchw_cuda(at::Tensor y, at::Tensor res, at::Tensor gamma, at::Tensor beta, double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_v1",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["to_nhwc_cuda", "gn_silu_nhwc_cuda", "gn_silu_add_nchw_cuda"],
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


def _is_capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


class ModelNew(nn.Module):
    """Fused VAE res-block (level-C fusion) + whole-forward CUDA-graph capture."""

    _MAX_GRAPHS = 8

    def __init__(self):
        super().__init__()
        self.ext = _ext
        # LRU cache: key -> (graph, out_tensor)  |  key -> None (capture refused)
        self._graphs = OrderedDict()

    # ------------------------------------------------------------------
    # Plan item 1: the original forward body, verbatim, as the impl function.
    # Capture-safe: no .item(), no .cpu(), no synchronize, no host allocation.
    # ------------------------------------------------------------------
    def _forward_impl(self, x, conv1_weight, norm1_weight, norm1_bias,
                      conv2_weight, norm2_weight, norm2_bias, eps_f):
        x_c = x if x.is_contiguous() else x.contiguous()

        # NCHW -> NHWC once, with our own tiled transpose.
        x_nhwc = self.ext.to_nhwc_cuda(x_c)

        w1 = conv1_weight if conv1_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv1_weight.contiguous(memory_format=torch.channels_last)
        w2 = conv2_weight if conv2_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv2_weight.contiguous(memory_format=torch.channels_last)

        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        # conv #1 (vendor cuDNN implicit GEMM, native NHWC -> no relayout)
        y1 = F.conv2d(x_nhwc, w1, None, 1, 1)
        if not y1.is_contiguous(memory_format=torch.channels_last):
            y1 = y1.contiguous(memory_format=torch.channels_last)

        # fused GroupNorm(32) + SiLU, NHWC in/out
        z = self.ext.gn_silu_nhwc_cuda(y1, g1, b1, eps_f)

        # conv #2
        y2 = F.conv2d(z, w2, None, 1, 1)
        if not y2.is_contiguous(memory_format=torch.channels_last):
            y2 = y2.contiguous(memory_format=torch.channels_last)

        # fused GroupNorm(32) + SiLU + residual add + NHWC->NCHW relayout
        out = self.ext.gn_silu_add_nchw_cuda(y2, x_c, g2, b2, eps_f)
        return out

    # ------------------------------------------------------------------
    # Plan items 4 + 5: side-stream warmup, then capture reading the CALLER's
    # tensors directly (no static buffers, no copy_ of inputs).
    # ------------------------------------------------------------------
    def _warmup_and_capture(self, args):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            tmp = None
            for _ in range(3):
                tmp = self._forward_impl(*args)
            del tmp
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self._forward_impl(*args)
        return g, out

    # ------------------------------------------------------------------
    # Plan items 2/3/6/9/10: dispatcher only.
    # ------------------------------------------------------------------
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        eps_f = float(eps)
        assert x.dtype == torch.float32, "this kernel supports float32 only"

        args = (x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps_f)

        # Guards: nested capture / non-CUDA / non-fp32 -> plain eager path.
        if (not x.is_cuda) or (x.dtype != torch.float32) or _is_capturing():
            return self._forward_impl(*args)

        key = (tuple(x.shape),
               x.data_ptr(),
               conv1_weight.data_ptr(),
               conv2_weight.data_ptr(),
               norm1_weight.data_ptr(),
               norm1_bias.data_ptr(),
               norm2_weight.data_ptr(),
               norm2_bias.data_ptr(),
               eps_f)

        if key in self._graphs:
            entry = self._graphs[key]
            if entry is None:            # capture previously refused -> eager
                return self._forward_impl(*args)
            self._graphs.move_to_end(key)
            g, out = entry
            g.replay()
            return out

        # Bound the private memory pools (LRU-drop oldest).
        while len(self._graphs) >= self._MAX_GRAPHS:
            old_key, old_entry = self._graphs.popitem(last=False)
            del old_key, old_entry

        try:
            g, out = self._warmup_and_capture(args)
            # Capture itself does not execute the work: replay once to fill `out`.
            g.replay()
            self._graphs[key] = (g, out)
            return out
        except Exception:
            self._graphs[key] = None     # sentinel: permanently eager for this key
            return self._forward_impl(*args)
