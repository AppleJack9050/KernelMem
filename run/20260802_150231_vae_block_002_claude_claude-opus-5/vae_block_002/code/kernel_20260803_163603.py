# ============================================================================
# ModelNew — fused VAE residual block (Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN
#            -> SiLU -> +residual), C=256, groups=32, fp32.
#
# THIS REVISION: reduction-kernel fusion.
#   gn_final_kernel is deleted; its final reduction is folded into
#   gn_stats_kernel using a deadlock-free "last-arriving block" pattern
#   (__threadfence + atomicAdd on a self-resetting arrival counter).  Blocks
#   that are not last EXIT immediately (no spin-wait), so there is no residency
#   requirement and multistream concurrency / graph capture are untouched.
#   This removes 8 launches (and 8 stream dependencies) per forward whose cost
#   was ~75% launch/drain latency (5.2us duration for ~1.3us of work).
#
#   Everything else (L2 chunk-blocking, evict-policy hints, multistream
#   fork/join pipeline, CUDA-graph capture, exhaustive cuDNN engine autotune
#   during warm-up only, grids, block sizes) is preserved verbatim.
#
# PRECISION: float32 end to end; TF32 only inside the convolutions, exactly as
# the reference does.  Reductions accumulate in float/double as before.
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
#include <mutex>

#define CTOT   256
#define NGRP   32
#define CPG    8

// ---------------------------------------------------------------------------
// portable cache-hint helpers (inline PTX, defined once).
//   ld_cs  : ld.global.cs.f32      -> evict-first / streaming scalar load
//   ld_cs4 : ld.global.cs.v4.f32   -> evict-first / streaming float4 load
//   ld_cg  : ld.global.cg.f32      -> cache in L2, bypass L1 (L2-resident)
//   st_cs  : st.global.cs.f32      -> evict-first / streaming store
// Value-neutral: the returned/stored bits are exactly the ones a plain
// load/store would produce.  ld_cg additionally guarantees the fused tail
// reduction sees the L2-coherent psum/psq data published by __threadfence().
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ld_cs(const float* p)
{
    float v;
    asm("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float ld_cg(const float* p)
{
    float v;
    asm("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float4 ld_cs4(const float4* p)
{
    float4 v;
    asm("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p));
    return v;
}

__device__ __forceinline__ void st_cs(float* p, float v)
{
    asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}

// ---------------------------------------------------------------------------
// NCHW -> NHWC tiled transpose (tile: 32 spatial x 64 channels, 256 threads)
// UNCHANGED.
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
                sm[lh * 65 + cl] =
                    ld_cs(&src[nbaseS + (long long)(c0 + cl) * HW + hw]);
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
// PLAN ITEMS 1/2/3/6/7: GroupNorm partial moments over NHWC data WITH the
// final reduction fused in (gn_final_kernel is gone).
//   grid=(P,nb), block=256 (1 thread / channel).  Every entry of the partial
//   buffers is written (even for empty chunks).  After publishing its partials
//   each block votes on a per-call arrival counter; the single last-arriving
//   block re-executes gn_final's exact math for all nb*NGRP groups (its input
//   is already L2-resident, ~89% hit) and then resets the counter to 0 so the
//   slot is reusable by the next launch / graph replay.
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                int HW, int P,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                float* __restrict__ mean,
                                float* __restrict__ rstd,
                                float eps, int nb,
                                int* __restrict__ cnt)
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
        int cnt_i = hw1 - hw0;
        #pragma unroll 4
        for (int i = 0; i < cnt_i; ++i) {
            float v = ld_cg(&base[(long long)i * CTOT]);
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

    // ---- PLAN ITEM 1: publish + single arrival vote ------------------------
    __threadfence();
    __shared__ int s_last;
    if (threadIdx.x == 0) {
        const int old = atomicAdd(cnt, 1);
        s_last = (old == (int)(gridDim.x * gridDim.y) - 1) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) return;                 // non-last blocks EXIT (no spin-wait)

    // ---- PLAN ITEM 2: fused final reduction (gn_final's math) --------------
    const int G = nb * NGRP;             // total groups in this chunk
    int tpg = 1;                         // lanes per group (power of two, <=32)
    {
        const int t = (int)blockDim.x / (G > 0 ? G : 1);   // 0 -> tpg stays 1
        while ((tpg << 1) <= t && tpg < 32) tpg <<= 1;
    }
    const int gpb   = (int)blockDim.x / tpg;               // groups per pass
    const int l     = (int)threadIdx.x & (tpg - 1);        // lane inside group
    const int gbase = (int)threadIdx.x / tpg;
    const int niter = (G + gpb - 1) / gpb;                 // uniform trip count
    const double cntd = (double)HW * (double)CPG;

    for (int it = 0; it < niter; ++it) {
        const int gid = it * gpb + gbase;
        double s2 = 0.0, q2 = 0.0;
        if (gid < G) {
            const int nl = gid / NGRP;
            const int gg = gid - nl * NGRP;
            for (int pp = l; pp < P; pp += tpg) {
                const long long o2 = ((long long)nl * P + pp) * NGRP + gg;
                s2 += (double)ld_cg(&psum[o2]);
                q2 += (double)ld_cg(&psq[o2]);
            }
        }
        // reduce the tpg contiguous lanes (all threads participate)
        for (int off = tpg >> 1; off > 0; off >>= 1) {
            s2 += __shfl_down_sync(0xffffffffu, s2, off);
            q2 += __shfl_down_sync(0xffffffffu, q2, off);
        }
        if (l == 0 && gid < G) {
            const double m = s2 / cntd;
            double v = q2 / cntd - m * m;
            if (v < 0.0) v = 0.0;
            mean[gid] = (float)m;
            rstd[gid] = (float)(1.0 / sqrt(v + (double)eps));
        }
    }

    // self-reset so the slot is 0 for the next launch / graph replay
    if (threadIdx.x == 0) cnt[0] = 0;
}

// ---------------------------------------------------------------------------
// GN affine + SiLU, NHWC -> NHWC, float4 vectorised.  grid=(x,nb)
// UNCHANGED (reverse block traversal + evict hints preserved verbatim).
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
    const int i = (gridDim.x - 1 - blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n4perimg) return;

    const int c4 = i & 63;              // float4 index inside the 256 channels
    const int g  = c4 >> 1;             // 8 channels per group == 2 float4
    const float m = mean[n * NGRP + g];
    const float r = rstd[n * NGRP + g];

    const long long o = (long long)n * n4perimg + i;
    const float4 v  = ld_cs4(y + o);
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
// UNCHANGED (reverse spatial-tile traversal + evict hints preserved verbatim).
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
    const int bx  = gridDim.x - 1 - blockIdx.x;
    const int hw0 = bx * 32;
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
                const float v = ld_cs(&base[(long long)hw * CTOT]);
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
                const float r0 = ld_cs(&res[o]);
                st_cs(&out[o], sm[lh * 65 + cl] + r0);
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

// ---------------------------------------------------------------------------
// L2 chunk size in images (16 MB budget) + node-count guard (<= 4 chunks).
// UNCHANGED -> l2_cache_blocking preserved bit-for-bit.
// ---------------------------------------------------------------------------
static inline int chunk_imgs(int B, int HW)
{
    const long long bpi = (long long)CTOT * (long long)HW * 4LL;  // bytes/image
    int k = (int)((16LL << 20) / (bpi > 0 ? bpi : 1));
    if (k < 1) k = 1;
    if (k > B) k = B;
    const int nch = (B + k - 1) / k;
    if (nch > 4) {
        k = (B + 3) / 4;
        if (k < 1) k = 1;
        if (k > B) k = B;
    }
    return k;
}

// ---------------------------------------------------------------------------
// PLAN ITEM 5: graph-safe arrival-counter pool.  NSLOT ints, allocated exactly
// once (cudaMalloc + cudaMemset(0)) on the first host call -- i.e. during the
// eager / warm-up phase, never inside a CUDA-graph capture region.  Each
// stats_chunk call takes the next slot round-robin, so concurrent streams and
// distinct capture call sites never share a counter.
// ---------------------------------------------------------------------------
#define NSLOT 64
static int*            g_cnt = nullptr;
static std::once_flag  g_cnt_once;

static void init_cnt_impl()
{
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, (size_t)NSLOT * sizeof(int));
    TORCH_CHECK(e == cudaSuccess, "counter alloc failed");
    e = cudaMemset(p, 0, (size_t)NSLOT * sizeof(int));
    TORCH_CHECK(e == cudaSuccess, "counter memset failed");
    g_cnt = (int*)p;
}

static inline int* cnt_slot(int s)
{
    std::call_once(g_cnt_once, init_cnt_impl);
    return g_cnt + s;
}

void init_counters()
{
    std::call_once(g_cnt_once, init_cnt_impl);
}

static int slot_rr = 0;

// ---------------------------------------------------------------------------
// PLAN ITEM 4: one chunk's stats -> now exactly ONE kernel launch.
// grid is exactly (P, nb) and the fused tail reads exactly the nb*P*NGRP slots
// that were written -> no unwritten tail.
// ---------------------------------------------------------------------------
static inline void stats_chunk(const float* y, int nb, int HW, int P, float eps,
                               float* psum, float* psq,
                               float* mean, float* rstd,
                               cudaStream_t stream)
{
    const int s   = (slot_rr++) & (NSLOT - 1);
    int* cptr     = cnt_slot(s);

    dim3 g1(P, nb);
    gn_stats_kernel<<<g1, 256, 0, stream>>>(y, HW, P, psum, psq,
                                            mean, rstd, eps, nb, cptr);
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

// ---------------------------------------------------------------------------
// chunked stats -> apply (L2 cache-blocking).  Chunk loop identical to base.
// ---------------------------------------------------------------------------
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
    auto stream = at::cuda::getCurrentCUDAStream();

    const int ic = chunk_imgs(B, HW);
    const int P  = pick_P(ic, HW);

    // hoisted allocations (capture-friendly: fixed sizes, done once)
    auto mean = torch::empty({(long long)B * NGRP}, opts);
    auto rstd = torch::empty({(long long)B * NGRP}, opts);
    auto psum = torch::empty({(long long)ic * P * NGRP}, opts);
    auto psq  = torch::empty({(long long)ic * P * NGRP}, opts);

    auto out = torch::empty({B, C, H, W},
                            opts.memory_format(at::MemoryFormat::ChannelsLast));

    const float* ybase = y.data_ptr<float>();
    float*       obase = out.data_ptr<float>();
    float*       mbase = mean.data_ptr<float>();
    float*       rbase = rstd.data_ptr<float>();
    float*       ps    = psum.data_ptr<float>();
    float*       pq    = psq.data_ptr<float>();

    const int n4 = HW * (CTOT / 4);
    const long long img_elems = (long long)CTOT * (long long)HW;

    for (int n0 = 0; n0 < B; n0 += ic) {
        const int nb = (ic < (B - n0)) ? ic : (B - n0);
        const float* yc = ybase + (long long)n0 * img_elems;
        float*       oc = obase + (long long)n0 * img_elems;
        float*       mc = mbase + (long long)n0 * NGRP;
        float*       rc = rbase + (long long)n0 * NGRP;

        stats_chunk(yc, nb, HW, P, (float)eps, ps, pq, mc, rc, stream);

        dim3 g((n4 + 255) / 256, nb);
        gn_silu_apply<<<g, 256, 0, stream>>>(
            reinterpret_cast<const float4*>(yc),
            reinterpret_cast<const float4*>(w.data_ptr<float>()),
            reinterpret_cast<const float4*>(b.data_ptr<float>()),
            mc, rc,
            reinterpret_cast<float4*>(oc), n4);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// ---------------------------------------------------------------------------
// identical chunk loop for the fused residual/transpose variant.
// ---------------------------------------------------------------------------
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
    auto stream = at::cuda::getCurrentCUDAStream();

    const int ic = chunk_imgs(B, HW);
    const int P  = pick_P(ic, HW);

    auto mean = torch::empty({(long long)B * NGRP}, opts);
    auto rstd = torch::empty({(long long)B * NGRP}, opts);
    auto psum = torch::empty({(long long)ic * P * NGRP}, opts);
    auto psq  = torch::empty({(long long)ic * P * NGRP}, opts);

    auto out = torch::empty({B, C, H, W}, opts);

    const float* ybase = y.data_ptr<float>();
    const float* rsbase = res.data_ptr<float>();
    float*       obase = out.data_ptr<float>();
    float*       mbase = mean.data_ptr<float>();
    float*       rbase = rstd.data_ptr<float>();
    float*       ps    = psum.data_ptr<float>();
    float*       pq    = psq.data_ptr<float>();

    const long long img_elems = (long long)CTOT * (long long)HW;

    for (int n0 = 0; n0 < B; n0 += ic) {
        const int nb = (ic < (B - n0)) ? ic : (B - n0);
        const float* yc  = ybase  + (long long)n0 * img_elems;
        const float* rsc = rsbase + (long long)n0 * img_elems;
        float*       oc  = obase  + (long long)n0 * img_elems;
        float*       mc  = mbase  + (long long)n0 * NGRP;
        float*       rc  = rbase  + (long long)n0 * NGRP;

        stats_chunk(yc, nb, HW, P, (float)eps, ps, pq, mc, rc, stream);

        dim3 g((HW + 31) / 32, C / 64, nb);
        gn_silu_res_t<<<g, 256, 0, stream>>>(yc, rsc,
                                             w.data_ptr<float>(),
                                             b.data_ptr<float>(),
                                             mc, rc, oc, HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// ---------------------------------------------------------------------------
// out-parameter variant.  Same chunked schedule; writes into the caller-supplied
// contiguous NCHW tensor (a batch slice of the final output).
// ---------------------------------------------------------------------------
void gn_silu_res_nchw_out(torch::Tensor y, torch::Tensor res,
                          torch::Tensor w, torch::Tensor b, double eps,
                          torch::Tensor out)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32);
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast),
                "expect channels_last conv output");
    TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32,
                "out must be CUDA fp32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous NCHW");
    TORCH_CHECK(out.dim() == 4, "out must be 4D");
    TORCH_CHECK(out.size(0) == y.size(0) && out.size(1) == y.size(1) &&
                out.size(2) == y.size(2) && out.size(3) == y.size(3),
                "out shape mismatch");
    TORCH_CHECK(res.size(0) == y.size(0) && res.size(1) == y.size(1) &&
                res.size(2) == y.size(2) && res.size(3) == y.size(3),
                "res shape mismatch");

    const int B = (int)y.size(0), C = (int)y.size(1);
    const int H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == CTOT, "C must be 256");
    const int HW = H * W;

    auto opts = y.options();
    auto stream = at::cuda::getCurrentCUDAStream();

    const int ic = chunk_imgs(B, HW);
    const int P  = pick_P(ic, HW);

    auto mean = torch::empty({(long long)B * NGRP}, opts);
    auto rstd = torch::empty({(long long)B * NGRP}, opts);
    auto psum = torch::empty({(long long)ic * P * NGRP}, opts);
    auto psq  = torch::empty({(long long)ic * P * NGRP}, opts);

    const float* ybase  = y.data_ptr<float>();
    const float* rsbase = res.data_ptr<float>();
    float*       obase  = out.data_ptr<float>();
    float*       mbase  = mean.data_ptr<float>();
    float*       rbase  = rstd.data_ptr<float>();
    float*       ps     = psum.data_ptr<float>();
    float*       pq     = psq.data_ptr<float>();

    const long long img_elems = (long long)CTOT * (long long)HW;

    for (int n0 = 0; n0 < B; n0 += ic) {
        const int nb = (ic < (B - n0)) ? ic : (B - n0);
        const float* yc  = ybase  + (long long)n0 * img_elems;
        const float* rsc = rsbase + (long long)n0 * img_elems;
        float*       oc  = obase  + (long long)n0 * img_elems;
        float*       mc  = mbase  + (long long)n0 * NGRP;
        float*       rc  = rbase  + (long long)n0 * NGRP;

        stats_chunk(yc, nb, HW, P, (float)eps, ps, pq, mc, rc, stream);

        dim3 g((HW + 31) / 32, C / 64, nb);
        gn_silu_res_t<<<g, 256, 0, stream>>>(yc, rsc,
                                             w.data_ptr<float>(),
                                             b.data_ptr<float>(),
                                             mc, rc, oc, HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
'''

_CPP_SRC = r'''
void init_counters();
torch::Tensor nchw_to_nhwc(torch::Tensor x);
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps);
torch::Tensor gn_silu_res_nchw(torch::Tensor y, torch::Tensor res, torch::Tensor w, torch::Tensor b, double eps);
void gn_silu_res_nchw_out(torch::Tensor y, torch::Tensor res, torch::Tensor w, torch::Tensor b, double eps, torch::Tensor out);
'''

_ext = load_inline(
    name="vae_resblock_fused_redfuse_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["init_counters", "nchw_to_nhwc", "gn_silu_nhwc",
               "gn_silu_res_nchw", "gn_silu_res_nchw_out"],
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

# PLAN ITEM 5: allocate the arrival-counter pool once, at import time, i.e.
# far outside any CUDA-graph capture region.
try:
    if torch.cuda.is_available():
        _ext.init_counters()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Locally-scoped cuDNN exhaustive-autotune context manager, unchanged.
# benchmark=True + benchmark_limit=0 -> exhaustive engine timing.
# Always restored on exit; never active during CUDA-graph capture.
# ---------------------------------------------------------------------------
class _Autotune:
    def __enter__(self):
        self._ok = False
        try:
            self.pb = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = True
            self._ok = True
        except Exception:
            self.pb = None
        self._lim_ok = False
        try:
            self.pl = getattr(torch.backends.cudnn, "benchmark_limit", 10)
            torch.backends.cudnn.benchmark_limit = 0
            self._lim_ok = True
        except Exception:
            self.pl = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._lim_ok:
            try:
                torch.backends.cudnn.benchmark_limit = self.pl
            except Exception:
                pass
        if self._ok:
            try:
                torch.backends.cudnn.benchmark = self.pb
            except Exception:
                pass
        return False


class ModelNew(nn.Module):
    """Fused VAE residual block: L2 cache-blocked GN/SiLU with fused two-phase
    reduction (last-block pattern) + evict-policy hints + multistream +
    CUDA-graph capture + cuDNN engine autotune."""

    _MAX_GRAPHS = 8
    _MAX_TUNED = 8

    def __init__(self):
        super().__init__()
        self.ext = _ext
        try:
            self.ext.init_counters()
        except Exception:
            pass
        # multi-stream resources (lazily created)
        self._streams = None
        # graph cache + shared memory pool
        self._graphs = {}
        self._pool = None
        # per-(shape, memory-format) autotuned key set
        self._tuned = set()

    # ---- lazy stream + event creation with safe fallback --------------------
    def _get_streams(self):
        if self._streams is None:
            try:
                sA = torch.cuda.Stream()
                sB = torch.cuda.Stream()
                ev0 = torch.cuda.Event()
                evStage = torch.cuda.Event()
                evA = torch.cuda.Event()
                evB = torch.cuda.Event()
                self._streams = (sA, sB, ev0, evStage, evA, evB)
            except Exception:
                self._streams = False  # permanent single-stream fallback
        return self._streams if self._streams else None

    # ======================================================================
    # The graph-capturable pipeline body.
    #   * B < 4  -> legacy single-stream path
    #   * B >= 4 -> two-chunk, two-stream fork/join pipeline
    # ======================================================================
    def _run_pipeline(self, x, w1c, nw1, nb1, w2c, nw2, nb2, eps):
        B = int(x.size(0))
        st = self._get_streams()

        if (st is None) or (B < 4):
            xl = self.ext.nchw_to_nhwc(x)
            y = F.conv2d(xl, w1c, None, 1, 1)
            y = self.ext.gn_silu_nhwc(y, nw1, nb1, eps)
            y = F.conv2d(y, w2c, None, 1, 1)
            return self.ext.gn_silu_res_nchw(y, x, nw2, nb2, eps)

        sA, sB, ev0, evStage, evA, evB = st
        s0 = B // 2

        out = torch.empty_like(x)
        cur = torch.cuda.current_stream()

        # fork
        ev0.record(cur)
        sA.wait_event(ev0)
        sB.wait_event(ev0)

        # chunk 0 on stream A
        with torch.cuda.stream(sA):
            xa = x[0:s0]
            xla = self.ext.nchw_to_nhwc(xa)
            evStage.record(sA)                       # stagger marker
            ya = F.conv2d(xla, w1c, None, 1, 1)
            ya = self.ext.gn_silu_nhwc(ya, nw1, nb1, eps)
            ya = F.conv2d(ya, w2c, None, 1, 1)
            self.ext.gn_silu_res_nchw_out(ya, xa, nw2, nb2, eps, out[0:s0])
            evA.record(sA)

        # chunk 1 on stream B, one stage out of phase
        with torch.cuda.stream(sB):
            sB.wait_event(evStage)
            xb = x[s0:B]
            xlb = self.ext.nchw_to_nhwc(xb)
            yb = F.conv2d(xlb, w1c, None, 1, 1)
            yb = self.ext.gn_silu_nhwc(yb, nw1, nb1, eps)
            yb = F.conv2d(yb, w2c, None, 1, 1)
            self.ext.gn_silu_res_nchw_out(yb, xb, nw2, nb2, eps, out[s0:B])
            evB.record(sB)

        # join
        cur.wait_event(evA)
        cur.wait_event(evB)
        return out

    # ======================================================================
    # One exhaustive-autotune pipeline run per new key.  Result discarded;
    # the selected cuDNN plan stays in torch's plan cache.
    # ======================================================================
    def _autotune_once(self, key, x, w1c, nw1, nb1, w2c, nw2, nb2, eps):
        if (key in self._tuned) or (len(self._tuned) >= self._MAX_TUNED):
            return
        self._tuned.add(key)
        try:
            with _Autotune():
                tmp = self._run_pipeline(x, w1c, nw1, nb1, w2c, nw2, nb2, eps)
                del tmp
            torch.cuda.synchronize()
        except Exception:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    @staticmethod
    def _tune_key(x):
        try:
            cl = bool(x.is_contiguous(memory_format=torch.channels_last))
        except Exception:
            cl = False
        return (tuple(x.shape), str(x.dtype), cl)

    # ---- eager (non-captured) path, keeps record_stream bookkeeping --------
    def _forward_eager(self, x, conv1_weight, norm1_weight, norm1_bias,
                       conv2_weight, norm2_weight, norm2_bias, eps):
        x = x if x.is_contiguous() else x.contiguous()
        nw1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        nb1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        nw2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        nb2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()
        w1c = conv1_weight.contiguous(memory_format=torch.channels_last)
        w2c = conv2_weight.contiguous(memory_format=torch.channels_last)

        if x.is_cuda and x.dim() == 4 and x.dtype == torch.float32:
            self._autotune_once(self._tune_key(x), x, w1c, nw1, nb1,
                                w2c, nw2, nb2, eps)

        B = int(x.size(0))
        st = self._get_streams()
        if (st is None) or (B < 4) or (x.dim() != 4) or \
           (x.dtype != torch.float32) or (not x.is_cuda):
            xl = self.ext.nchw_to_nhwc(x)
            y = F.conv2d(xl, w1c, None, 1, 1)
            y = self.ext.gn_silu_nhwc(y, nw1, nb1, eps)
            y = F.conv2d(y, w2c, None, 1, 1)
            return self.ext.gn_silu_res_nchw(y, x, nw2, nb2, eps)

        sA, sB, ev0, evStage, evA, evB = st
        s0 = B // 2
        out = torch.empty_like(x)
        cur = torch.cuda.current_stream()
        ev0.record(cur)
        sA.wait_event(ev0)
        sB.wait_event(ev0)

        with torch.cuda.stream(sA):
            x.record_stream(sA); out.record_stream(sA)
            w1c.record_stream(sA); w2c.record_stream(sA)
            nw1.record_stream(sA); nb1.record_stream(sA)
            nw2.record_stream(sA); nb2.record_stream(sA)
            xa = x[0:s0]
            xla = self.ext.nchw_to_nhwc(xa)
            evStage.record(sA)
            ya = F.conv2d(xla, w1c, None, 1, 1)
            ya = self.ext.gn_silu_nhwc(ya, nw1, nb1, eps)
            ya = F.conv2d(ya, w2c, None, 1, 1)
            self.ext.gn_silu_res_nchw_out(ya, xa, nw2, nb2, eps, out[0:s0])
            evA.record(sA)

        with torch.cuda.stream(sB):
            sB.wait_event(evStage)
            x.record_stream(sB); out.record_stream(sB)
            w1c.record_stream(sB); w2c.record_stream(sB)
            nw1.record_stream(sB); nb1.record_stream(sB)
            nw2.record_stream(sB); nb2.record_stream(sB)
            xb = x[s0:B]
            xlb = self.ext.nchw_to_nhwc(xb)
            yb = F.conv2d(xlb, w1c, None, 1, 1)
            yb = self.ext.gn_silu_nhwc(yb, nw1, nb1, eps)
            yb = F.conv2d(yb, w2c, None, 1, 1)
            self.ext.gn_silu_res_nchw_out(yb, xb, nw2, nb2, eps, out[s0:B])
            evB.record(sB)

        cur.wait_event(evA)
        cur.wait_event(evB)
        return out

    # ======================================================================
    # Warm up (with autotune), capture, cache.  Shapes are static so the
    # chunked GN sequence is a fixed node list; buffers come from the pool.
    # ======================================================================
    def _try_capture(self, key, x, conv1_weight, norm1_weight, norm1_bias,
                     conv2_weight, norm2_weight, norm2_bias, eps):
        try:
            nw1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
            nb1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
            nw2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
            nb2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()
            w1c = conv1_weight.contiguous(memory_format=torch.channels_last)
            w2c = conv2_weight.contiguous(memory_format=torch.channels_last)

            s_warm = torch.cuda.Stream()
            s_warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_warm):
                tmp = None
                with _Autotune():
                    for _ in range(3):
                        tmp = self._run_pipeline(x, w1c, nw1, nb1,
                                                 w2c, nw2, nb2, eps)
                    del tmp
                tmp = self._run_pipeline(x, w1c, nw1, nb1, w2c, nw2, nb2, eps)
                del tmp
            torch.cuda.current_stream().wait_stream(s_warm)
            torch.cuda.synchronize()

            self._tuned.add(self._tune_key(x))

            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                out = self._run_pipeline(x, w1c, nw1, nb1, w2c, nw2, nb2, eps)

            self._graphs[key] = (g, out, w1c, w2c, nw1, nb1, nw2, nb2, x)

            g.replay()
            return out
        except Exception:
            self._graphs[key] = False
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return None

    # ======================================================================
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps = float(eps.item())
        else:
            eps = float(eps)

        if x.is_cuda and x.dim() == 4 and x.dtype == torch.float32 and x.is_contiguous():
            key = (tuple(x.shape), x.dtype, x.data_ptr(),
                   conv1_weight.data_ptr(), conv2_weight.data_ptr(),
                   norm1_weight.data_ptr(), norm1_bias.data_ptr(),
                   norm2_weight.data_ptr(), norm2_bias.data_ptr(), eps)

            entry = self._graphs.get(key, None)
            if entry is not None:
                if entry is not False:
                    g, out = entry[0], entry[1]
                    g.replay()
                    return out
            elif len(self._graphs) < self._MAX_GRAPHS:
                res = self._try_capture(key, x, conv1_weight, norm1_weight,
                                        norm1_bias, conv2_weight, norm2_weight,
                                        norm2_bias, eps)
                if res is not None:
                    return res

        return self._forward_eager(x, conv1_weight, norm1_weight, norm1_bias,
                                   conv2_weight, norm2_weight, norm2_bias, eps)
