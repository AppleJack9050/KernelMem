# ==========================================================================
# ModelNew — 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER / PLAN
# 1) Chosen granularity: (C) fuse many ops into one/few custom CUDA kernels.
#    (Vendor conv is deliberately NOT re-implemented; we accept the Amdahl cap
#     and instead remove ALL non-conv traffic + the cuDNN layout transposes.)
#
# 2) Ops replaced (all executed inside the load_inline extension):
#      - F.group_norm  (both instances)      -> custom 2-pass stats kernels
#      - F.silu        (both instances)      -> fused into the GN apply kernels
#      - residual add  (out + x)             -> fused into the 2nd apply kernel
#      - NHWC<->NCHW layout traffic          -> the 2nd apply kernel writes NCHW
#                                               directly via a shared-memory
#                                               transpose
#      - F.conv2d      (both instances)      -> at::conv2d called FROM the
#                                               extension, fed channels-last so
#                                               cuDNN runs its native
#                                               sm90 nhwckrsc TF32 implicit GEMM
#                                               with zero layout conversions.
#
# 3) THIS ROUND (cudnn_algo_autotune): the two conv mainloops are wrapped in an
#    RAII guard that locally enables cuDNN exhaustive algorithm search
#    (benchmark=true, TF32=true, benchmark_limit=10) and restores the previous
#    process-global flags on scope exit. No kernel body or launch config moves.
#
# Precision: everything stays float32; reductions accumulate in fp32 per-thread
# then double across chunks. TF32 conv matches reference default behaviour.
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

#define NCH  256
#define NGRP 32
#define GSZ  8
#define TP   32

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + __expf(-v) * 0.0f + expf(-v) * 1.0f - expf(-v) * 0.0f);
}

__device__ __forceinline__ float silu_fast(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// pass 1 : per-(batch, chunk, group) partial sum / sum of squares, NHWC input
//          block = 256 threads (one per channel)  -> fully coalesced 1KB rows
// ---------------------------------------------------------------------------
__global__ void gn_stats_p1(const float* __restrict__ x,
                            float* __restrict__ part,
                            int npix, int ppc, int nchunks)
{
    const int b     = blockIdx.y;
    const int chunk = blockIdx.x;
    const int p0    = chunk * ppc;
    int p1 = p0 + ppc; if (p1 > npix) p1 = npix;

    const float* base = x + (size_t)b * (size_t)npix * NCH;
    const int c = threadIdx.x;

    float s = 0.f, q = 0.f;
    for (int p = p0; p < p1; ++p) {
        float v = base[(size_t)p * NCH + c];
        s += v;
        q += v * v;
    }
    // reduce over the 8 consecutive lanes that form one group
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off, 8);
        q += __shfl_down_sync(0xffffffffu, q, off, 8);
    }
    if ((c & (GSZ - 1)) == 0) {
        const int g = c >> 3;
        size_t o = ((size_t)b * (size_t)nchunks + (size_t)chunk) * NGRP + g;
        part[2 * o + 0] = s;
        part[2 * o + 1] = q;
    }
}

// ---------------------------------------------------------------------------
// pass 2 : deterministic tree reduction over chunks -> per-channel scale/shift
//          grid = (NGRP, B), block = 256
// ---------------------------------------------------------------------------
__global__ void gn_stats_p2(const float* __restrict__ part,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            float* __restrict__ scale,
                            float* __restrict__ shift,
                            int nchunks, int npix, float eps)
{
    const int g = blockIdx.x;
    const int b = blockIdx.y;

    __shared__ double ss[256];
    __shared__ double sq[256];

    double s = 0.0, q = 0.0;
    for (int i = threadIdx.x; i < nchunks; i += blockDim.x) {
        size_t o = ((size_t)b * (size_t)nchunks + (size_t)i) * NGRP + g;
        s += (double)part[2 * o + 0];
        q += (double)part[2 * o + 1];
    }
    ss[threadIdx.x] = s;
    sq[threadIdx.x] = q;
    __syncthreads();

    for (int st = 128; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) {
            ss[threadIdx.x] += ss[threadIdx.x + st];
            sq[threadIdx.x] += sq[threadIdx.x + st];
        }
        __syncthreads();
    }

    if (threadIdx.x < GSZ) {
        const double N    = (double)npix * (double)GSZ;
        const double mean = ss[0] / N;
        double var        = sq[0] / N - mean * mean;
        if (var < 0.0) var = 0.0;
        const double rstd = 1.0 / sqrt(var + (double)eps);
        const int c = g * GSZ + (int)threadIdx.x;
        const double gm = (double)gamma[c];
        scale[b * NCH + c] = (float)(rstd * gm);
        shift[b * NCH + c] = (float)((double)beta[c] - mean * rstd * gm);
    }
}

// ---------------------------------------------------------------------------
// apply 1 : y = silu(x*scale + shift), NHWC -> NHWC, float4 vectorised
//           grid = (ceil(nvec/256), B)
// ---------------------------------------------------------------------------
__global__ void gn_apply1(const float* __restrict__ x,
                          const float* __restrict__ scale,
                          const float* __restrict__ shift,
                          float* __restrict__ out,
                          long nvec_per_img)
{
    const int b = blockIdx.y;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nvec_per_img) return;

    const float4* xin = (const float4*)(x   + (size_t)b * (size_t)nvec_per_img * 4);
    float4*       o   = (float4*)      (out + (size_t)b * (size_t)nvec_per_img * 4);

    const int c4 = (int)(i & 63);          // 64 float4 per pixel (256 channels)
    float4 v  = xin[i];
    float4 sc = ((const float4*)(scale + b * NCH))[c4];
    float4 sh = ((const float4*)(shift + b * NCH))[c4];

    float4 r;
    r.x = silu_fast(v.x * sc.x + sh.x);
    r.y = silu_fast(v.y * sc.y + sh.y);
    r.z = silu_fast(v.z * sc.z + sh.z);
    r.w = silu_fast(v.w * sc.w + sh.w);
    o[i] = r;
}

// ---------------------------------------------------------------------------
// apply 2 : out(NCHW) = silu(y*scale + shift) + residual(NCHW)
//           NHWC read (coalesced) -> shared transpose -> NCHW write
//           grid = (ceil(npix/32), B), block = 256
// ---------------------------------------------------------------------------
__global__ void gn_apply2(const float* __restrict__ y,
                          const float* __restrict__ res,
                          const float* __restrict__ scale,
                          const float* __restrict__ shift,
                          float* __restrict__ out,
                          int npix)
{
    __shared__ float sm[NCH * (TP + 1)];

    const int b  = blockIdx.y;
    const int p0 = blockIdx.x * TP;
    int np = npix - p0; if (np > TP) np = TP;
    if (np <= 0) return;

    const float* base = y + (size_t)b * (size_t)npix * NCH;
    const int c = threadIdx.x;
    const float sc = scale[b * NCH + c];
    const float sh = shift[b * NCH + c];

    for (int j = 0; j < np; ++j) {
        float v = base[(size_t)(p0 + j) * NCH + c];
        v = v * sc + sh;
        sm[c * (TP + 1) + j] = silu_fast(v);
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int p    = p0 + lane;
    if (p < npix) {
        const size_t imgbase = (size_t)b * NCH * (size_t)npix;
        for (int cc = warp; cc < NCH; cc += 8) {
            float v = sm[cc * (TP + 1) + lane];
            size_t off = imgbase + (size_t)cc * (size_t)npix + (size_t)p;
            out[off] = v + res[off];
        }
    }
}

// ---------------------------------------------------------------------------
// RAII guard : locally enable cuDNN exhaustive algorithm autotune (+TF32) for
//              the two 3x3 NHWC conv mainloops, restore the previous
//              process-global flags on scope exit (also on exception).
// ---------------------------------------------------------------------------
struct CudnnAutotuneGuard {
    static constexpr int kBenchmarkLimit = 10;   // bound trial count per shape
    bool prev_bm;
    bool prev_tf;
    int  prev_lim;

    CudnnAutotuneGuard() {
        prev_bm  = at::globalContext().benchmarkCuDNN();
        prev_tf  = at::globalContext().allowTF32CuDNN();
        prev_lim = at::globalContext().benchmarkLimitCuDNN();
        at::globalContext().setBenchmarkCuDNN(true);
        at::globalContext().setAllowTF32CuDNN(true);
        if (prev_lim <= 0) at::globalContext().setBenchmarkLimitCuDNN(kBenchmarkLimit);
    }

    ~CudnnAutotuneGuard() {
        at::globalContext().setBenchmarkCuDNN(prev_bm);
        at::globalContext().setAllowTF32CuDNN(prev_tf);
        at::globalContext().setBenchmarkLimitCuDNN(prev_lim);
    }
};

// ---------------------------------------------------------------------------
// host entry
// ---------------------------------------------------------------------------
static void gn_stats(const at::Tensor& t, const at::Tensor& gamma, const at::Tensor& beta,
                     at::Tensor& scale, at::Tensor& shift,
                     int B, int npix, float eps, cudaStream_t stream)
{
    int ppc = (npix + 1023) / 1024;
    if (ppc < 32) ppc = 32;
    const int nchunks = (npix + ppc - 1) / ppc;

    auto part = at::empty({(long)B * nchunks * NGRP * 2}, t.options());

    dim3 g1(nchunks, B), b1(NCH);
    gn_stats_p1<<<g1, b1, 0, stream>>>(t.data_ptr<float>(), part.data_ptr<float>(),
                                       npix, ppc, nchunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 g2(NGRP, B), b2(256);
    gn_stats_p2<<<g2, b2, 0, stream>>>(part.data_ptr<float>(),
                                       gamma.data_ptr<float>(), beta.data_ptr<float>(),
                                       scale.data_ptr<float>(), shift.data_ptr<float>(),
                                       nchunks, npix, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4 && x.size(1) == NCH, "expect (B,256,H,W)");

    const int B = (int)x.size(0);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int npix = H * W;

    auto stream = at::cuda::getCurrentCUDAStream();

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto xl = xc.contiguous(at::MemoryFormat::ChannelsLast);
    auto w1l = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2l = w2.contiguous(at::MemoryFormat::ChannelsLast);

    auto n1wc = n1w.is_contiguous() ? n1w : n1w.contiguous();
    auto n1bc = n1b.is_contiguous() ? n1b : n1b.contiguous();
    auto n2wc = n2w.is_contiguous() ? n2w : n2w.contiguous();
    auto n2bc = n2b.is_contiguous() ? n2b : n2b.contiguous();

    std::vector<int64_t> st{1, 1}, pd{1, 1}, dl{1, 1};

    auto opts   = x.options();
    auto scale  = at::empty({(long)B * NCH}, opts);
    auto shift  = at::empty({(long)B * NCH}, opts);
    auto scale2 = at::empty({(long)B * NCH}, opts);
    auto shift2 = at::empty({(long)B * NCH}, opts);

    at::Tensor y2;

    // ======== cuDNN algo-autotuned conv segment (flags restored on exit) =====
    {
        CudnnAutotuneGuard _autotune_guard;   // benchmark=true, TF32=true, limit<=10

        // ---- conv 1 (cuDNN NHWC TF32 implicit GEMM, no layout conversion) ----
        auto y1 = at::conv2d(xl, w1l, c10::optional<at::Tensor>(),
                             at::IntArrayRef(st), at::IntArrayRef(pd), at::IntArrayRef(dl), 1);
        if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
            y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

        gn_stats(y1, n1wc, n1bc, scale, shift, B, npix, (float)eps, stream);

        auto y1n = at::empty({(long)B, (long)NCH, (long)H, (long)W},
                             opts.memory_format(at::MemoryFormat::ChannelsLast));

        const long nvec = (long)npix * 64L;              // float4 units per image
        {
            const int blk = 256;
            dim3 g((unsigned)((nvec + blk - 1) / blk), (unsigned)B);
            gn_apply1<<<g, blk, 0, stream>>>(y1.data_ptr<float>(),
                                             scale.data_ptr<float>(), shift.data_ptr<float>(),
                                             y1n.data_ptr<float>(), nvec);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // ---- conv 2 ----
        y2 = at::conv2d(y1n, w2l, c10::optional<at::Tensor>(),
                        at::IntArrayRef(st), at::IntArrayRef(pd), at::IntArrayRef(dl), 1);
        if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
            y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);
    }
    // ======== flags restored here (guard destructor) =========================

    gn_stats(y2, n2wc, n2bc, scale2, shift2, B, npix, (float)eps, stream);

    auto out = at::empty({(long)B, (long)NCH, (long)H, (long)W}, opts);
    {
        dim3 g((unsigned)((npix + TP - 1) / TP), (unsigned)B);
        gn_apply2<<<g, NCH, 0, stream>>>(y2.data_ptr<float>(), xc.data_ptr<float>(),
                                         scale2.data_ptr<float>(), shift2.data_ptr<float>(),
                                         out.data_ptr<float>(), npix);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps);
'''

_ext = load_inline(
    name="vae_resblock_fused_v2",
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
    """See top-of-file header comment for the granularity/fusion plan (level C)."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        # Fast path: fixed C=256 / G=32 float32 CUDA specialisation.
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256 and conv1_weight.dtype == torch.float32
                and conv2_weight.dtype == torch.float32):
            return self._ext.fused_block(x, conv1_weight, norm1_weight, norm1_bias,
                                         conv2_weight, norm2_weight, norm2_bias, eps_f)

        # Reference fallback (never taken by the benchmarked workloads).
        num_groups = 32
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps_f)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps_f)
        out = F.silu(out)
        return out + residual
