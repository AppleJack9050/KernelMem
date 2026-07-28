# ==========================================================================
# ModelNew — 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# ROUND-N METHOD: persistent_cooperative_gn_fusion
#   Each GroupNorm's three kernels (partial stats -> cross-chunk reduce ->
#   affine+SiLU(+residual/transpose)) are collapsed into ONE grid-persistent
#   kernel:  phase1 accumulate -> grid barrier -> phase2 in-register stats ->
#   phase3 apply over the SAME pixel range (phase-3 re-read hits L2).
#   gn_stats_p2 (8 us, 0.28 waves, 22% occupancy => pure launch latency) is
#   gone; per-forward launches 11 -> 7 (2 conv + 2 fused + allocator).
#   Legacy 3-kernel path is kept verbatim as a deadlock/compat fallback.
#
# Ops replaced (all inside the extension): both F.group_norm, both F.silu,
# residual add, NHWC<->NCHW transposes.  Vendor-kept: the two 3x3 convs
# (cuDNN sm90 nhwckrsc TF32 implicit GEMM, channels-last, zero layout copies).
# Precision: fp32 storage/arith, fp32 per-thread partials, double final reduce.
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
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

#define NCH  256
#define NGRP 32
#define GSZ  8
#define TP   32

__device__ __forceinline__ float silu_fast(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// grid-wide barrier (sense reversing, generation counter never needs a reset).
// Equivalent to cg::this_grid().sync(); all blocks are guaranteed co-resident
// because the kernel is launched with cudaLaunchCooperativeKernel.
// bar[0] = arrival count, bar[1] = generation.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void grid_bar(unsigned int* bar, unsigned int nblocks)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int gen = atomicAdd(&bar[1], 0u);
        __threadfence();
        if (atomicAdd(&bar[0], 1u) == nblocks - 1u) {
            atomicExch(&bar[0], 0u);
            __threadfence();
            atomicAdd(&bar[1], 1u);
        } else {
            while (atomicAdd(&bar[1], 0u) == gen) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_block();
}

// ===========================================================================
//  PERSISTENT FUSED KERNEL 1 :  stats + reduce + (affine GN) + SiLU, NHWC->NHWC
// ===========================================================================
__global__ void gn_fused1(const float* __restrict__ y,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          float* __restrict__ out,
                          float* __restrict__ part,
                          unsigned int* __restrict__ bar,
                          int npix, int bpb, int chunk, float eps,
                          unsigned int nblocks)
{
    const int b  = blockIdx.x / bpb;
    const int k  = blockIdx.x - b * bpb;
    const int p0 = k * chunk;
    int p1 = p0 + chunk; if (p1 > npix) p1 = npix;

    const int c = threadIdx.x;
    const float* base = y + (size_t)b * (size_t)npix * NCH;

    // ---------------- PHASE 1 : partial sum / sumsq (never early-returns) ---
    {
        float s = 0.f, q = 0.f;
        for (int p = p0; p < p1; ++p) {
            float v = base[(size_t)p * NCH + c];
            s += v;
            q += v * v;
        }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off, 8);
            q += __shfl_down_sync(0xffffffffu, q, off, 8);
        }
        if ((c & (GSZ - 1)) == 0) {
            const int g = c >> 3;
            size_t o = ((size_t)b * (size_t)bpb + (size_t)k) * NGRP + g;
            part[2 * o + 0] = s;
            part[2 * o + 1] = q;
        }
    }

    // ---------------- GRID SYNC --------------------------------------------
    grid_bar(bar, nblocks);

    // ---------------- PHASE 2 : in-register stats (fixed order, double) ----
    __shared__ __align__(16) float ssc[NCH];
    __shared__ __align__(16) float ssh[NCH];
    {
        const int g = c >> 3;
        double S = 0.0, Q = 0.0;
        for (int i = 0; i < bpb; ++i) {
            size_t o = ((size_t)b * (size_t)bpb + (size_t)i) * NGRP + g;
            S += (double)part[2 * o + 0];
            Q += (double)part[2 * o + 1];
        }
        const double N    = (double)npix * (double)GSZ;
        const double mean = S / N;
        double var        = Q / N - mean * mean;
        if (var < 0.0) var = 0.0;
        const double rstd = 1.0 / sqrt(var + (double)eps);
        const double gm   = (double)gamma[c];
        ssc[c] = (float)(rstd * gm);
        ssh[c] = (float)((double)beta[c] - mean * rstd * gm);
    }
    __syncthreads();

    // ---------------- PHASE 3 : silu(x*sc+sh), float4, same [p0,p1) --------
    {
        const long nvec = (long)(p1 - p0) * 64L;      // 64 float4 per pixel
        if (nvec > 0) {
            const float4* xin = (const float4*)(base + (size_t)p0 * NCH);
            float4* o4 = (float4*)(out + (size_t)b * (size_t)npix * NCH
                                       + (size_t)p0 * NCH);
            const float4* sc4 = (const float4*)ssc;
            const float4* sh4 = (const float4*)ssh;
            for (long i = threadIdx.x; i < nvec; i += blockDim.x) {
                const int c4 = (int)(i & 63);
                float4 v  = xin[i];
                float4 sc = sc4[c4];
                float4 sh = sh4[c4];
                float4 r;
                r.x = silu_fast(v.x * sc.x + sh.x);
                r.y = silu_fast(v.y * sc.y + sh.y);
                r.z = silu_fast(v.z * sc.z + sh.z);
                r.w = silu_fast(v.w * sc.w + sh.w);
                o4[i] = r;
            }
        }
    }
}

// ===========================================================================
//  PERSISTENT FUSED KERNEL 2 : stats + reduce + GN + SiLU + residual,
//                              NHWC read -> smem transpose -> NCHW write
// ===========================================================================
__global__ void gn_fused2(const float* __restrict__ y,
                          const float* __restrict__ res,
                          const float* __restrict__ gamma,
                          const float* __restrict__ beta,
                          float* __restrict__ out,
                          float* __restrict__ part,
                          unsigned int* __restrict__ bar,
                          int npix, int bpb, int chunk, float eps,
                          unsigned int nblocks)
{
    __shared__ float sm[NCH * (TP + 1)];

    const int b  = blockIdx.x / bpb;
    const int k  = blockIdx.x - b * bpb;
    const int p0 = k * chunk;
    int p1 = p0 + chunk; if (p1 > npix) p1 = npix;

    const int c = threadIdx.x;
    const float* base = y + (size_t)b * (size_t)npix * NCH;

    // ---------------- PHASE 1 ----------------------------------------------
    {
        float s = 0.f, q = 0.f;
        for (int p = p0; p < p1; ++p) {
            float v = base[(size_t)p * NCH + c];
            s += v;
            q += v * v;
        }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off, 8);
            q += __shfl_down_sync(0xffffffffu, q, off, 8);
        }
        if ((c & (GSZ - 1)) == 0) {
            const int g = c >> 3;
            size_t o = ((size_t)b * (size_t)bpb + (size_t)k) * NGRP + g;
            part[2 * o + 0] = s;
            part[2 * o + 1] = q;
        }
    }

    // ---------------- GRID SYNC --------------------------------------------
    grid_bar(bar, nblocks);

    // ---------------- PHASE 2 : registers ----------------------------------
    float sc, sh;
    {
        const int g = c >> 3;
        double S = 0.0, Q = 0.0;
        for (int i = 0; i < bpb; ++i) {
            size_t o = ((size_t)b * (size_t)bpb + (size_t)i) * NGRP + g;
            S += (double)part[2 * o + 0];
            Q += (double)part[2 * o + 1];
        }
        const double N    = (double)npix * (double)GSZ;
        const double mean = S / N;
        double var        = Q / N - mean * mean;
        if (var < 0.0) var = 0.0;
        const double rstd = 1.0 / sqrt(var + (double)eps);
        const double gm   = (double)gamma[c];
        sc = (float)(rstd * gm);
        sh = (float)((double)beta[c] - mean * rstd * gm);
    }

    // ---------------- PHASE 3 : smem NHWC->NCHW tiles over [p0,p1) ---------
    const int warp = c >> 5;
    const int lane = c & 31;
    const size_t imgbase = (size_t)b * NCH * (size_t)npix;

    for (int pt = p0; pt < p1; pt += TP) {
        int np = p1 - pt; if (np > TP) np = TP;

        __syncthreads();
        for (int j = 0; j < np; ++j) {
            float v = base[(size_t)(pt + j) * NCH + c];
            v = v * sc + sh;
            sm[c * (TP + 1) + j] = silu_fast(v);
        }
        __syncthreads();

        if (lane < np) {
            const int p = pt + lane;
            for (int cc = warp; cc < NCH; cc += 8) {
                float v = sm[cc * (TP + 1) + lane];
                size_t off = imgbase + (size_t)cc * (size_t)npix + (size_t)p;
                out[off] = v + res[off];
            }
        }
    }
}

// ===========================================================================
//  LEGACY 3-KERNEL PATH (fallback, unchanged)
// ===========================================================================
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

    const int c4 = (int)(i & 63);
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
// host helpers
// ---------------------------------------------------------------------------
static int get_nsm() {
    static int n = -1;
    if (n < 0) { int d = 0; cudaGetDevice(&d);
                 cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
    return n;
}

static int coop_supported() {
    static int s = -1;
    if (s < 0) { int d = 0; cudaGetDevice(&d);
                 if (cudaDeviceGetAttribute(&s, cudaDevAttrCooperativeLaunch, d) != cudaSuccess) s = 0; }
    return s;
}

static at::Tensor get_bar(const at::Tensor& ref) {
    static at::Tensor bar;
    if (!bar.defined() || bar.device() != ref.device()) {
        bar = at::zeros({2}, ref.options().dtype(at::kInt));
    }
    return bar;
}

// returns false if a cooperative launch is not possible -> caller uses legacy
static bool coop_cfg(const void* fn, int B, int npix, int& bpb, int& chunk)
{
    if (!coop_supported()) return false;
    int mb = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, fn, NCH, 0);
    if (e != cudaSuccess || mb <= 0) { cudaGetLastError(); return false; }
    const int maxBlocks = mb * get_nsm();
    if (maxBlocks < B) return false;
    int cap = (npix + TP - 1) / TP; if (cap < 1) cap = 1;
    bpb = maxBlocks / B;
    if (bpb > cap) bpb = cap;
    if (bpb < 1)   bpb = 1;
    chunk = (npix + bpb - 1) / bpb;
    return true;
}

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

static bool run_fused_gn1(const at::Tensor& y1, const at::Tensor& gam, const at::Tensor& bet,
                          at::Tensor& out, int B, int npix, float eps, cudaStream_t stream)
{
    int bpb = 1, chunk = npix;
    if (!coop_cfg((const void*)gn_fused1, B, npix, bpb, chunk)) return false;

    auto part = at::empty({(long)B * bpb * NGRP * 2}, y1.options());
    auto bar  = get_bar(y1);

    const float* py = y1.data_ptr<float>();
    const float* pg = gam.data_ptr<float>();
    const float* pb = bet.data_ptr<float>();
    float* po  = out.data_ptr<float>();
    float* ppt = part.data_ptr<float>();
    unsigned int* pbar = (unsigned int*)bar.data_ptr<int>();
    unsigned int nb = (unsigned int)(B * bpb);
    int npix_ = npix, bpb_ = bpb, chunk_ = chunk;
    float eps_ = eps;

    void* args[] = { (void*)&py, (void*)&pg, (void*)&pb, (void*)&po, (void*)&ppt,
                     (void*)&pbar, (void*)&npix_, (void*)&bpb_, (void*)&chunk_,
                     (void*)&eps_, (void*)&nb };

    cudaError_t e = cudaLaunchCooperativeKernel((const void*)gn_fused1,
                                                dim3(nb), dim3(NCH), args, 0, stream);
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

static bool run_fused_gn2(const at::Tensor& y2, const at::Tensor& res,
                          const at::Tensor& gam, const at::Tensor& bet,
                          at::Tensor& out, int B, int npix, float eps, cudaStream_t stream)
{
    int bpb = 1, chunk = npix;
    if (!coop_cfg((const void*)gn_fused2, B, npix, bpb, chunk)) return false;

    auto part = at::empty({(long)B * bpb * NGRP * 2}, y2.options());
    auto bar  = get_bar(y2);

    const float* py = y2.data_ptr<float>();
    const float* pr = res.data_ptr<float>();
    const float* pg = gam.data_ptr<float>();
    const float* pb = bet.data_ptr<float>();
    float* po  = out.data_ptr<float>();
    float* ppt = part.data_ptr<float>();
    unsigned int* pbar = (unsigned int*)bar.data_ptr<int>();
    unsigned int nb = (unsigned int)(B * bpb);
    int npix_ = npix, bpb_ = bpb, chunk_ = chunk;
    float eps_ = eps;

    void* args[] = { (void*)&py, (void*)&pr, (void*)&pg, (void*)&pb, (void*)&po,
                     (void*)&ppt, (void*)&pbar, (void*)&npix_, (void*)&bpb_,
                     (void*)&chunk_, (void*)&eps_, (void*)&nb };

    cudaError_t e = cudaLaunchCooperativeKernel((const void*)gn_fused2,
                                                dim3(nb), dim3(NCH), args, 0, stream);
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// host entry
// ---------------------------------------------------------------------------
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

    // ---- conv 1 (cuDNN NHWC TF32 implicit GEMM, no layout conversion) ----
    auto y1 = at::conv2d(xl, w1l, c10::optional<at::Tensor>(),
                         at::IntArrayRef(st), at::IntArrayRef(pd), at::IntArrayRef(dl), 1);
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    auto opts = x.options();
    auto y1n = at::empty({(long)B, (long)NCH, (long)H, (long)W},
                         opts.memory_format(at::MemoryFormat::ChannelsLast));

    // ---- fused GN1 + SiLU (persistent cooperative) -----------------------
    if (!run_fused_gn1(y1, n1wc, n1bc, y1n, B, npix, (float)eps, stream)) {
        auto scale = at::empty({(long)B * NCH}, opts);
        auto shift = at::empty({(long)B * NCH}, opts);
        gn_stats(y1, n1wc, n1bc, scale, shift, B, npix, (float)eps, stream);
        const long nvec = (long)npix * 64L;
        const int blk = 256;
        dim3 g((unsigned)((nvec + blk - 1) / blk), (unsigned)B);
        gn_apply1<<<g, blk, 0, stream>>>(y1.data_ptr<float>(),
                                         scale.data_ptr<float>(), shift.data_ptr<float>(),
                                         y1n.data_ptr<float>(), nvec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv 2 ----
    auto y2 = at::conv2d(y1n, w2l, c10::optional<at::Tensor>(),
                         at::IntArrayRef(st), at::IntArrayRef(pd), at::IntArrayRef(dl), 1);
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    auto out = at::empty({(long)B, (long)NCH, (long)H, (long)W}, opts);

    // ---- fused GN2 + SiLU + residual + NHWC->NCHW ------------------------
    if (!run_fused_gn2(y2, xc, n2wc, n2bc, out, B, npix, (float)eps, stream)) {
        auto scale2 = at::empty({(long)B * NCH}, opts);
        auto shift2 = at::empty({(long)B * NCH}, opts);
        gn_stats(y2, n2wc, n2bc, scale2, shift2, B, npix, (float)eps, stream);
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
    name="vae_resblock_fused_coop_v2",
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
    """Persistent cooperative GroupNorm fusion (see top-of-file header)."""

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
