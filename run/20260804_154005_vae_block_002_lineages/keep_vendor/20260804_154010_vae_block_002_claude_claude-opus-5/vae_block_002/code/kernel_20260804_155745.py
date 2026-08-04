# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER (required):
# 1) GRANULARITY: (C) — fuse many ops into one/few kernels.
#
# 2) OPS REPLACED (all non-conv work is replaced by custom CUDA kernels):
#      - the implicit NCHW<->NHWC layout conversions cuDNN inserts (4 kernels
#        in the reference profile, 309 us) -> replaced by ONE tiled
#        shared-memory transpose on the way in, and ZERO on the way out
#        (the outgoing transpose is fused into the final epilogue kernel).
#      - group_norm #1 (RowwiseMoments + ComputeFusedParams + elementwise)
#      - silu #1        (vectorized_elementwise)
#      - group_norm #2, silu #2, residual add (elementwise + vectorized_elementwise)
#
# 3) FUSION MAP:
#      k_nchw2nhwc      : NCHW->NHWC tiled transpose (32x32 tile, +1 pad)
#      at::conv2d       : cuDNN NHWC/TF32 implicit-GEMM (vendor, called from
#                         inside the extension) — conv1 and conv2
#      k_gn_partial     : split-reduction sum / sum-of-squares per (batch,group),
#                         256 threads = 256 channels, 8-lane butterfly shuffle
#      k_gn_finalize    : partials -> mean/var -> per-(b,c) scale & shift
#                         (double accumulation over the split partials)
#      k_gn_silu_nhwc   : {affine scale/shift} + {SiLU} fused, float4 vectorized
#                         (GroupNorm-apply + SiLU in ONE pass, NHWC->NHWC)
#      k_gn_silu_res_out: {affine scale/shift} + {SiLU} + {residual add} +
#                         {NHWC->NCHW transpose} fused into ONE epilogue kernel
#                         so the output layout conversion costs no extra traffic.
#
# 4) LEFT IN PYTORCH:
#      - the two 3x3 convolutions: they are the vendor TF32 tensor-core
#        implicit-GEMM (40.7% of runtime, at/near roofline); re-implementing
#        wins nothing, so instead they are *fed* the channels-last layout they
#        want, which removes the vendor's own transpose kernels.
#      - nothing else: every remaining reference op is inside a fused kernel.
#
# NOTES: fp32 in / fp32 out everywhere; reductions accumulate in fp32 per
# split and fp64 across splits. TF32 for conv only (matches the reference,
# which runs with cudnn.allow_tf32 = True). No in-place mutation of inputs,
# no hidden state, no device moves.
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

torch.backends.cudnn.benchmark = True

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// NCHW -> NHWC tiled transpose (32x32 tiles, 33-wide shared rows).
// in : (B, C, P)   out : (B, P, C)
// ---------------------------------------------------------------------------
__global__ void k_nchw2nhwc(const float* __restrict__ in,
                            float* __restrict__ out,
                            int C, long P) {
    __shared__ float sm[32][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int   c0 = blockIdx.x * 32;
    const long  p0 = (long)blockIdx.y * 32;
    const int   b  = blockIdx.z;

    const float* inb  = in  + (long)b * C * P;
    float*       outb = out + (long)b * P * C;

    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int  c = c0 + ty + k;
        long p = p0 + tx;
        float v = 0.0f;
        if (c < C && p < P) v = inb[(long)c * P + p];
        sm[ty + k][tx] = v;      // sm[channel_local][pixel_local]
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        long p = p0 + ty + k;
        int  c = c0 + tx;
        if (p < P && c < C) outb[p * C + c] = sm[tx][ty + k];
    }
}

// ---------------------------------------------------------------------------
// GroupNorm split reduction over pixels, NHWC.  blockDim.x == C (<=1024)
// grid = (nsplit, B).  psum/psq layout: [(b*nsplit + s)*G + g]
// ---------------------------------------------------------------------------
__global__ void k_gn_partial(const float* __restrict__ y,
                             float* __restrict__ psum,
                             float* __restrict__ psq,
                             long P, int C, int G, int cpg, int nsplit) {
    const int c = threadIdx.x;
    const int s = blockIdx.x;
    const int b = blockIdx.y;

    const long start = (P * (long)s) / nsplit;
    const long end   = (P * (long)(s + 1)) / nsplit;

    const float* base = y + (long)b * P * C + c;

    float sum = 0.0f, sq = 0.0f;
    long p = start;
    for (; p + 3 < end; p += 4) {
        float v0 = base[(p + 0) * C];
        float v1 = base[(p + 1) * C];
        float v2 = base[(p + 2) * C];
        float v3 = base[(p + 3) * C];
        sum += v0 + v1 + v2 + v3;
        sq  += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for (; p < end; ++p) {
        float v = base[p * C];
        sum += v;
        sq  += v * v;
    }

    // butterfly reduce inside each cpg-lane block (cpg is a power of two <= 32)
    for (int off = 1; off < cpg; off <<= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
        sq  += __shfl_xor_sync(0xffffffffu, sq,  off);
    }
    if ((c % cpg) == 0) {
        int g = c / cpg;
        long o = ((long)b * nsplit + s) * G + g;
        psum[o] = sum;
        psq[o]  = sq;
    }
}

// ---------------------------------------------------------------------------
// partials -> per-(b,c) scale / shift
// ---------------------------------------------------------------------------
__global__ void k_gn_finalize(const float* __restrict__ psum,
                              const float* __restrict__ psq,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              float* __restrict__ scale,
                              float* __restrict__ shift,
                              int B, int C, int G, int cpg,
                              int nsplit, double cnt, double eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * G) return;
    int b = idx / G;
    int g = idx - b * G;

    double s = 0.0, q = 0.0;
    for (int i = 0; i < nsplit; ++i) {
        long o = ((long)b * nsplit + i) * G + g;
        s += (double)psum[o];
        q += (double)psq[o];
    }
    double mean = s / cnt;
    double var  = q / cnt - mean * mean;
    if (var < 0.0) var = 0.0;
    double inv = rsqrt(var + eps);

    for (int j = 0; j < cpg; ++j) {
        int c = g * cpg + j;
        double sc = (double)gamma[c] * inv;
        scale[(long)b * C + c] = (float)sc;
        shift[(long)b * C + c] = (float)((double)beta[c] - mean * sc);
    }
}

// ---------------------------------------------------------------------------
// affine + SiLU, NHWC -> NHWC, float4 vectorized.  grid = (gx, B)
// ---------------------------------------------------------------------------
__global__ void k_gn_silu_nhwc(const float4* __restrict__ y,
                               const float4* __restrict__ scale,
                               const float4* __restrict__ shift,
                               float4* __restrict__ out,
                               long P, int C4) {
    const int b = blockIdx.y;
    const long base = (long)b * P * C4;
    const float4* sc = scale + (long)b * C4;
    const float4* sh = shift + (long)b * C4;
    const long total = P * (long)C4;
    const long stride = (long)gridDim.x * blockDim.x;

    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        int c4 = (int)(i % (long)C4);
        float4 v = y[base + i];
        float4 s = sc[c4];
        float4 t = sh[c4];
        float4 r;
        r.x = silu_f(v.x * s.x + t.x);
        r.y = silu_f(v.y * s.y + t.y);
        r.z = silu_f(v.z * s.z + t.z);
        r.w = silu_f(v.w * s.w + t.w);
        out[base + i] = r;
    }
}

// ---------------------------------------------------------------------------
// epilogue: affine + SiLU + residual add + NHWC->NCHW transpose (one pass)
// y : (B,P,C) NHWC   resid/out : (B,C,P) NCHW
// grid = (ceil(C/32), ceil(P/32), B), block = (32,8)
// ---------------------------------------------------------------------------
__global__ void k_gn_silu_res_out(const float* __restrict__ y,
                                  const float* __restrict__ scale,
                                  const float* __restrict__ shift,
                                  const float* __restrict__ resid,
                                  float* __restrict__ out,
                                  int C, long P) {
    __shared__ float sm[32][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int  c0 = blockIdx.x * 32;
    const long p0 = (long)blockIdx.y * 32;
    const int  b  = blockIdx.z;

    const float* yb = y + (long)b * P * C;
    const float* sc = scale + (long)b * C;
    const float* sh = shift + (long)b * C;

    const int cin = c0 + tx;
    float s = 0.0f, t = 0.0f;
    if (cin < C) { s = sc[cin]; t = sh[cin]; }

    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        long p = p0 + ty + k;
        float v = 0.0f;
        if (p < P && cin < C) {
            v = yb[p * C + cin];
            v = silu_f(v * s + t);
        }
        sm[ty + k][tx] = v;      // sm[pixel_local][channel_local]
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int  c = c0 + ty + k;
        long p = p0 + tx;
        if (c < C && p < P) {
            long o = (long)b * C * P + (long)c * P + p;
            out[o] = sm[tx][ty + k] + resid[o];
        }
    }
}

// ---------------------------------------------------------------------------
// host driver
// ---------------------------------------------------------------------------
static void gn_stats(const at::Tensor& y, const at::Tensor& gamma, const at::Tensor& beta,
                     at::Tensor& scale, at::Tensor& shift,
                     int B, int C, long P, int G, double eps, cudaStream_t stream) {
    const int cpg = C / G;

    long want = (512 + B - 1) / B;
    long nsl  = std::min<long>(std::max<long>(want, 1), 256);
    long maxs = std::max<long>((P + 63) / 64, 1);
    if (nsl > maxs) nsl = maxs;
    const int nsplit = (int)nsl;

    auto fopt = y.options();
    auto psum = at::empty({(long)B * nsplit * G}, fopt);
    auto psq  = at::empty({(long)B * nsplit * G}, fopt);

    dim3 gp(nsplit, B);
    k_gn_partial<<<gp, C, 0, stream>>>(y.data_ptr<float>(),
                                       psum.data_ptr<float>(), psq.data_ptr<float>(),
                                       P, C, G, cpg, nsplit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int nbg = B * G;
    int thr = nbg < 256 ? nbg : 256;
    int blk = (nbg + thr - 1) / thr;
    k_gn_finalize<<<blk, thr, 0, stream>>>(psum.data_ptr<float>(), psq.data_ptr<float>(),
                                           gamma.data_ptr<float>(), beta.data_ptr<float>(),
                                           scale.data_ptr<float>(), shift.data_ptr<float>(),
                                           B, C, G, cpg, nsplit, (double)P * (double)cpg, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                          torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                          double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "fp32 only");
    TORCH_CHECK(w1.scalar_type() == at::kFloat && w2.scalar_type() == at::kFloat, "fp32 only");

    const int  B = (int)x.size(0);
    const int  C = (int)x.size(1);
    const int  H = (int)x.size(2);
    const int  W = (int)x.size(3);
    const long P = (long)H * W;
    const int  G = 32;
    TORCH_CHECK(C % G == 0 && C <= 1024 && C % 4 == 0, "unsupported channel count");
    const int cpg = C / G;
    TORCH_CHECK(cpg <= 32 && (cpg & (cpg - 1)) == 0, "channels-per-group must be pow2 <= 32");

    auto stream = at::cuda::getCurrentCUDAStream();

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto opts = x.options();
    auto cl = opts.memory_format(at::MemoryFormat::ChannelsLast);

    // --- NCHW -> NHWC (tiled) --------------------------------------------
    auto xn = at::empty({B, C, H, W}, cl);
    {
        dim3 grid((C + 31) / 32, (unsigned)((P + 31) / 32), B);
        dim3 blk(32, 8);
        k_nchw2nhwc<<<grid, blk, 0, stream>>>(xc.data_ptr<float>(), xn.data_ptr<float>(), C, P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1c = g1.is_contiguous() ? g1 : g1.contiguous();
    auto b1c = b1.is_contiguous() ? b1 : b1.contiguous();
    auto g2c = g2.is_contiguous() ? g2 : g2.contiguous();
    auto b2c = b2.is_contiguous() ? b2 : b2.contiguous();

    at::Tensor no_bias;

    // --- conv1 (vendor, NHWC/TF32) ---------------------------------------
    auto y1 = at::conv2d(xn, w1c, no_bias, {1, 1}, {1, 1}, {1, 1}, 1);
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    auto scale = at::empty({B, C}, opts);
    auto shift = at::empty({B, C}, opts);
    gn_stats(y1, g1c, b1c, scale, shift, B, C, P, G, eps, stream);

    // --- GN-apply + SiLU (fused, NHWC) -----------------------------------
    auto y1n = at::empty({B, C, H, W}, cl);
    {
        const int C4 = C / 4;
        long total = P * (long)C4;
        int thr = 256;
        long gx = (total + thr - 1) / thr;
        if (gx > 2048) gx = 2048;
        if (gx < 1) gx = 1;
        dim3 grid((unsigned)gx, B);
        k_gn_silu_nhwc<<<grid, thr, 0, stream>>>(
            reinterpret_cast<const float4*>(y1.data_ptr<float>()),
            reinterpret_cast<const float4*>(scale.data_ptr<float>()),
            reinterpret_cast<const float4*>(shift.data_ptr<float>()),
            reinterpret_cast<float4*>(y1n.data_ptr<float>()), P, C4);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // --- conv2 (vendor, NHWC/TF32) ---------------------------------------
    auto y2 = at::conv2d(y1n, w2c, no_bias, {1, 1}, {1, 1}, {1, 1}, 1);
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    auto scale2 = at::empty({B, C}, opts);
    auto shift2 = at::empty({B, C}, opts);
    gn_stats(y2, g2c, b2c, scale2, shift2, B, C, P, G, eps, stream);

    // --- GN-apply + SiLU + residual + NHWC->NCHW (fused epilogue) --------
    auto out = at::empty({B, C, H, W}, opts);
    {
        dim3 grid((C + 31) / 32, (unsigned)((P + 31) / 32), B);
        dim3 blk(32, 8);
        k_gn_silu_res_out<<<grid, blk, 0, stream>>>(
            y2.data_ptr<float>(), scale2.data_ptr<float>(), shift2.data_ptr<float>(),
            xc.data_ptr<float>(), out.data_ptr<float>(), C, P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                          torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                          double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_c",
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
    """Granularity (C): conv3x3 -> GN -> SiLU -> conv3x3 -> GN -> SiLU -> +x
    with every non-conv op fused into custom CUDA kernels (see file header)."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        e = float(eps)
        if (not x.is_cuda) or x.dtype != torch.float32:
            # rare fallback: keep reference semantics exactly (no device moves)
            with torch.no_grad():
                out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
                out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=e)
                out = F.silu(out)
                out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
                out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=e)
                out = F.silu(out)
                return out + x
        with torch.no_grad():
            return self._ext.fused_block(x, conv1_weight, norm1_weight, norm1_bias,
                                         conv2_weight, norm2_weight, norm2_bias, e)
