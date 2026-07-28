# =============================================================================
# ModelNew : fused VAE residual block
#            Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual
#
# v2: persistent_fused_groupnorm_grid_barrier
#   Each GroupNorm (stats + finalize + affine/SiLU [+ residual + NHWC->NCHW])
#   is now ONE grid-persistent kernel with a device-wide barrier:
#     phase 1 : blocks own a CONTIGUOUS tile range, accumulate per-(b,group)
#               sum/sumsq into a zeroed workspace with atomicAdd
#     barrier : counter in global memory (grid guaranteed fully resident)
#     phase 2 : each block re-reads ITS OWN slice (L2 hit) and applies
#               affine + SiLU (+ residual + smem transpose back to NCHW)
#   Custom launches per forward: 7 -> 3 (+1 workspace zero-fill).
#   The legacy 3-kernel path is kept for tiny shapes / non-resident grids.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

#define CH       256
#define NGROUPS  32
#define CPG      8
#define TP       64
#define TC       64
#define NTHR     256
#define TILE     64   /* pixels per tile (one tile never crosses a batch) */

// ---------------------------------------------------------------- transpose
// NCHW (contiguous) -> NHWC (channels_last), shared memory tiled.  (unchanged)
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int HW)
{
    __shared__ float sm[TP * (TC + 1)];
    const int b   = blockIdx.z;
    const int c0  = blockIdx.y * TC;
    const int p0  = blockIdx.x * TP;
    const int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int cl  = idx >> 6;
        int pl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            sm[pl * (TC + 1) + cl] =
                in[((long)b * CH + (c0 + cl)) * (long)HW + p];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int pl  = idx >> 6;
        int cl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            out[((long)b * HW + p) * CH + (c0 + cl)] = sm[pl * (TC + 1) + cl];
        }
    }
}

// ============================ legacy (fallback) kernels ======================
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ part,
                                int HW, int nchunk, int ppc, long sq_off)
{
    const int b     = blockIdx.y;
    const int chunk = blockIdx.x;
    const int c     = threadIdx.x;

    int pstart = chunk * ppc;
    if (pstart >= HW) return;
    int pend = pstart + ppc;
    if (pend > HW) pend = HW;
    int n = pend - pstart;

    const float* px = x + (long)b * HW * CH + (long)pstart * CH + c;

    float s = 0.f, s2 = 0.f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float v0 = px[0];
        float v1 = px[CH];
        float v2 = px[2 * CH];
        float v3 = px[3 * CH];
        s  += (v0 + v1) + (v2 + v3);
        s2 += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
        px += 4 * CH;
    }
    for (; i < n; ++i) {
        float v = px[0];
        s  += v;
        s2 += v * v;
        px += CH;
    }

    #pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off, 8);
        s2 += __shfl_down_sync(0xffffffffu, s2, off, 8);
    }
    if ((c & 7) == 0) {
        int g  = c >> 3;
        long o = ((long)b * NGROUPS + g) * (long)nchunk + chunk;
        part[o]          = s;
        part[sq_off + o] = s2;
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ part,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nchunk, long sq_off,
                                   float invN, float eps)
{
    __shared__ float sh1[NTHR];
    __shared__ float sh2[NTHR];
    const int bg  = blockIdx.x;
    const int tid = threadIdx.x;

    float s = 0.f, s2 = 0.f;
    for (int i = tid; i < nchunk; i += NTHR) {
        s  += part[(long)bg * nchunk + i];
        s2 += part[sq_off + (long)bg * nchunk + i];
    }
    sh1[tid] = s;
    sh2[tid] = s2;
    __syncthreads();
    for (int st = NTHR / 2; st > 0; st >>= 1) {
        if (tid < st) {
            sh1[tid] += sh1[tid + st];
            sh2[tid] += sh2[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float m = sh1[0] * invN;
        float v = sh2[0] * invN - m * m;
        if (!(v > 0.f)) v = 0.f;
        mean[bg] = m;
        rstd[bg] = rsqrtf(v + eps);
    }
}

__global__ void gn_apply_silu_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ rstd,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     int HW)
{
    const int b = blockIdx.y;
    const int c = threadIdx.x;
    const int g = c >> 3;
    const float m  = mean[b * NGROUPS + g];
    const float r  = rstd[b * NGROUPS + g];
    const float sc = gamma[c] * r;
    const float sh = beta[c] - m * sc;

    const int p0 = blockIdx.x * 8;
    if (p0 >= HW) return;
    long off = (long)b * HW * CH + (long)p0 * CH + c;

    if (p0 + 8 <= HW) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = x[off + (long)i * CH] * sc + sh;
            y[off + (long)i * CH] = v / (1.f + expf(-v));
        }
    } else {
        for (int i = 0; p0 + i < HW; ++i) {
            float v = x[off + (long)i * CH] * sc + sh;
            y[off + (long)i * CH] = v / (1.f + expf(-v));
        }
    }
}

__global__ void gn_apply_silu_res_nchw_kernel(const float* __restrict__ x,
                                              const float* __restrict__ res,
                                              float* __restrict__ out,
                                              const float* __restrict__ mean,
                                              const float* __restrict__ rstd,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              int HW)
{
    __shared__ float sm[TC * (TP + 1)];
    __shared__ float ssc[TC];
    __shared__ float ssh[TC];

    const int b   = blockIdx.z;
    const int c0  = blockIdx.y * TC;
    const int p0  = blockIdx.x * TP;
    const int tid = threadIdx.x;

    if (tid < TC) {
        int c   = c0 + tid;
        int g   = c >> 3;
        float r = rstd[b * NGROUPS + g];
        float m = mean[b * NGROUPS + g];
        float sc = gamma[c] * r;
        ssc[tid] = sc;
        ssh[tid] = beta[c] - m * sc;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int pl  = idx >> 6;
        int cl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            float v = x[((long)b * HW + p) * CH + (c0 + cl)] * ssc[cl] + ssh[cl];
            sm[cl * (TP + 1) + pl] = v / (1.f + expf(-v));
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int cl  = idx >> 6;
        int pl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            long o = ((long)b * CH + (c0 + cl)) * (long)HW + p;
            out[o] = sm[cl * (TP + 1) + pl] + res[o];
        }
    }
}

// ==================== persistent fused GroupNorm kernels =====================

__device__ __forceinline__ void flush_stats(float s, float s2, int c, int b,
                                            float* __restrict__ acc, long sq_off)
{
    #pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off, 8);
        s2 += __shfl_down_sync(0xffffffffu, s2, off, 8);
    }
    if ((c & 7) == 0) {
        int g  = c >> 3;
        long o = (long)b * NGROUPS + g;
        atomicAdd(acc + o, s);
        atomicAdd(acc + sq_off + o, s2);
    }
}

// device-wide barrier: valid only because the grid is fully resident.
__device__ __forceinline__ void grid_barrier(int* __restrict__ bar)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1);
        while (((volatile int*)bar)[0] < (int)gridDim.x) {
            __nanosleep(64);
        }
        __threadfence();
    }
    __syncthreads();
}

// ---- fused GN #1 : stats + finalize + affine + SiLU  (NHWC -> NHWC) --------
__global__ __launch_bounds__(NTHR, 4)
void gn_fused_kernel(const float* __restrict__ x,
                     float* __restrict__ y,
                     float* __restrict__ acc,
                     int*   __restrict__ bar,
                     const float* __restrict__ gamma,
                     const float* __restrict__ beta,
                     int HW, int TPB, int ntiles, int chunk,
                     long sq_off, float invN, float eps)
{
    const int c  = threadIdx.x;
    const int g  = c >> 3;
    const int t0 = blockIdx.x * chunk;
    int t1 = t0 + chunk;
    if (t1 > ntiles) t1 = ntiles;

    // ---------------- phase 1 : partial sums over the OWN tile range --------
    {
        float s = 0.f, s2 = 0.f;
        int cur_b = -1;
        for (int t = t0; t < t1; ++t) {
            int b  = t / TPB;
            int p0 = (t - b * TPB) * TILE;
            if (b != cur_b) {
                if (cur_b >= 0) flush_stats(s, s2, c, cur_b, acc, sq_off);
                s = 0.f; s2 = 0.f; cur_b = b;
            }
            int n = HW - p0; if (n > TILE) n = TILE;
            const float* px = x + ((long)b * HW + p0) * (long)CH + c;
            int i = 0;
            for (; i + 4 <= n; i += 4) {
                float v0 = px[0];
                float v1 = px[CH];
                float v2 = px[2 * CH];
                float v3 = px[3 * CH];
                s  += (v0 + v1) + (v2 + v3);
                s2 += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
                px += 4 * CH;
            }
            for (; i < n; ++i) {
                float v = px[0];
                s  += v; s2 += v * v;
                px += CH;
            }
        }
        if (cur_b >= 0) flush_stats(s, s2, c, cur_b, acc, sq_off);
    }

    // ---------------- device-wide barrier -----------------------------------
    grid_barrier(bar);

    // ---------------- phase 2 : re-read own slice (L2) + affine + SiLU ------
    {
        int cur_b = -1;
        float sc = 0.f, sh = 0.f;
        for (int t = t0; t < t1; ++t) {
            int b  = t / TPB;
            int p0 = (t - b * TPB) * TILE;
            if (b != cur_b) {
                cur_b = b;
                long o    = (long)b * NGROUPS + g;
                float sum = __ldcg(acc + o);
                float sq  = __ldcg(acc + sq_off + o);
                float m   = sum * invN;
                float v   = sq * invN - m * m;
                if (!(v > 0.f)) v = 0.f;
                float r = rsqrtf(v + eps);
                sc = gamma[c] * r;
                sh = beta[c] - m * sc;
            }
            int n = HW - p0; if (n > TILE) n = TILE;
            const float* px = x + ((long)b * HW + p0) * (long)CH + c;
            float*       py = y + ((long)b * HW + p0) * (long)CH + c;
            for (int i = 0; i < n; ++i) {
                float v = fmaf(px[0], sc, sh);
                py[0] = v / (1.f + expf(-v));
                px += CH; py += CH;
            }
        }
    }
}

// ---- fused GN #2 : stats + finalize + affine + SiLU + residual + NHWC->NCHW -
__global__ __launch_bounds__(NTHR, 4)
void gn_fused_res_nchw_kernel(const float* __restrict__ x,
                              const float* __restrict__ res,
                              float* __restrict__ out,
                              float* __restrict__ acc,
                              int*   __restrict__ bar,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              int HW, int TPB, int ntiles, int chunk,
                              long sq_off, float invN, float eps)
{
    __shared__ float sm[TC * (TP + 1)];   // 16 640 B
    __shared__ float ssc[CH];             //  1 024 B
    __shared__ float ssh[CH];             //  1 024 B

    const int tid = threadIdx.x;
    const int c   = tid;
    const int g   = c >> 3;
    const int t0  = blockIdx.x * chunk;
    int t1 = t0 + chunk;
    if (t1 > ntiles) t1 = ntiles;

    // ---------------- phase 1 ------------------------------------------------
    {
        float s = 0.f, s2 = 0.f;
        int cur_b = -1;
        for (int t = t0; t < t1; ++t) {
            int b  = t / TPB;
            int p0 = (t - b * TPB) * TILE;
            if (b != cur_b) {
                if (cur_b >= 0) flush_stats(s, s2, c, cur_b, acc, sq_off);
                s = 0.f; s2 = 0.f; cur_b = b;
            }
            int n = HW - p0; if (n > TILE) n = TILE;
            const float* px = x + ((long)b * HW + p0) * (long)CH + c;
            int i = 0;
            for (; i + 4 <= n; i += 4) {
                float v0 = px[0];
                float v1 = px[CH];
                float v2 = px[2 * CH];
                float v3 = px[3 * CH];
                s  += (v0 + v1) + (v2 + v3);
                s2 += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
                px += 4 * CH;
            }
            for (; i < n; ++i) {
                float v = px[0];
                s  += v; s2 += v * v;
                px += CH;
            }
        }
        if (cur_b >= 0) flush_stats(s, s2, c, cur_b, acc, sq_off);
    }

    // ---------------- device-wide barrier ------------------------------------
    grid_barrier(bar);

    // ---------------- phase 2 : affine + SiLU + residual + transpose ---------
    {
        int cur_b = -1;
        for (int t = t0; t < t1; ++t) {
            int b  = t / TPB;
            int p0 = (t - b * TPB) * TILE;
            if (b != cur_b) {
                cur_b = b;
                long o    = (long)b * NGROUPS + g;
                float sum = __ldcg(acc + o);
                float sq  = __ldcg(acc + sq_off + o);
                float m   = sum * invN;
                float v   = sq * invN - m * m;
                if (!(v > 0.f)) v = 0.f;
                float r  = rsqrtf(v + eps);
                float sc = gamma[c] * r;
                ssc[c] = sc;
                ssh[c] = beta[c] - m * sc;
            }
            #pragma unroll
            for (int ct = 0; ct < CH / TC; ++ct) {
                const int c0 = ct * TC;
                __syncthreads();
                #pragma unroll
                for (int i = 0; i < (TP * TC) / NTHR; ++i) {
                    int idx = i * NTHR + tid;
                    int pl  = idx >> 6;
                    int cl  = idx & 63;
                    int p   = p0 + pl;
                    if (p < HW) {
                        float v = fmaf(x[((long)b * HW + p) * CH + (c0 + cl)],
                                       ssc[c0 + cl], ssh[c0 + cl]);
                        sm[cl * (TP + 1) + pl] = v / (1.f + expf(-v));
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int i = 0; i < (TP * TC) / NTHR; ++i) {
                    int idx = i * NTHR + tid;
                    int cl  = idx >> 6;
                    int pl  = idx & 63;
                    int p   = p0 + pl;
                    if (p < HW) {
                        long o = ((long)b * CH + (c0 + cl)) * (long)HW + p;
                        out[o] = sm[cl * (TP + 1) + pl] + res[o];
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------- driver
static void run_groupnorm_stats(const at::Tensor& t, at::Tensor& part,
                                at::Tensor& mean, at::Tensor& rstd,
                                int B, int HW, int nchunk, float eps,
                                cudaStream_t stream)
{
    int ppc = (HW + nchunk - 1) / nchunk;
    long sq_off = (long)B * NGROUPS * nchunk;

    dim3 g1(nchunk, B);
    gn_stats_kernel<<<g1, NTHR, 0, stream>>>(
        t.data_ptr<float>(), part.data_ptr<float>(), HW, nchunk, ppc, sq_off);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float invN = 1.0f / (float)((long)HW * CPG);
    gn_finalize_kernel<<<B * NGROUPS, NTHR, 0, stream>>>(
        part.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        nchunk, sq_off, invN, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static inline int grid_for(int cap, int ntiles, int* chunk_out)
{
    int g = cap < ntiles ? cap : ntiles;
    if (g < 1) g = 1;
    int chunk = (ntiles + g - 1) / g;
    if (chunk < 1) chunk = 1;
    g = (ntiles + chunk - 1) / chunk;
    *chunk_out = chunk;
    return g;
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps)
{
    at::NoGradGuard nograd;
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    auto xc = x.is_contiguous() ? x : x.contiguous();
    const int B  = (int)xc.size(0);
    const int C  = (int)xc.size(1);
    const int H  = (int)xc.size(2);
    const int W  = (int)xc.size(3);
    TORCH_CHECK(C == CH, "specialized for C == 256");
    const int HW = H * W;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts   = xc.options();

    // ---- x -> NHWC (unchanged) --------------------------------------------
    auto x_nhwc = at::empty({B, C, H, W}, opts, at::MemoryFormat::ChannelsLast);
    {
        dim3 grid((HW + TP - 1) / TP, C / TC, B);
        nchw2nhwc_kernel<<<grid, NTHR, 0, stream>>>(
            xc.data_ptr<float>(), x_nhwc.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1c = n1w.is_contiguous() ? n1w : n1w.contiguous();
    auto b1c = n1b.is_contiguous() ? n1b : n1b.contiguous();
    auto g2c = n2w.is_contiguous() ? n2w : n2w.contiguous();
    auto b2c = n2b.is_contiguous() ? n2b : n2b.contiguous();

    std::vector<int64_t> stride{1, 1}, pad{1, 1}, dil{1, 1};

    // ---- conv1 (cuDNN, NHWC) ----------------------------------------------
    auto t1 = at::conv2d(x_nhwc, w1c, {}, stride, pad, dil, 1);
    if (!t1.is_contiguous(at::MemoryFormat::ChannelsLast))
        t1 = t1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- residency / grid sizing for the persistent fused kernels ----------
    static int sm_count = -1;
    if (sm_count < 0)
        sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    static int bpsm_f = -1, bpsm_r = -1;
    if (bpsm_f < 0) {
        int b = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &b, (const void*)gn_fused_kernel, NTHR, 0);
        bpsm_f = b;
    }
    if (bpsm_r < 0) {
        int b = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &b, (const void*)gn_fused_res_nchw_kernel, NTHR, 0);
        bpsm_r = b;
    }

    const int TPB    = (HW + TILE - 1) / TILE;
    const long ntl   = (long)B * TPB;
    const int  ntiles = (int)ntl;
    const float invN = 1.0f / (float)((long)HW * CPG);

    const bool use_fused = (bpsm_f > 0) && (bpsm_r > 0) &&
                           (ntl <= 2147483647L) &&
                           (ntiles >= 2 * sm_count) &&
                           ((long)B * HW >= 16384L);

    auto out = at::empty({B, C, H, W}, opts);

    if (use_fused) {
        // one zeroed workspace: [GN1 sum | GN1 sq | GN2 sum | GN2 sq | bar x4]
        const long accN = 4L * B * NGROUPS;
        auto acc = at::zeros({accN + 4}, opts);
        float* accp = acc.data_ptr<float>();
        int*   barp = reinterpret_cast<int*>(accp + accN);
        const long sq_off = (long)B * NGROUPS;

        // ---- GN1 + SiLU (fused) -------------------------------------------
        auto y1 = at::empty({B, C, H, W}, opts, at::MemoryFormat::ChannelsLast);
        {
            int chunk = 1;
            int grid  = grid_for(bpsm_f * sm_count, ntiles, &chunk);
            gn_fused_kernel<<<grid, NTHR, 0, stream>>>(
                t1.data_ptr<float>(), y1.data_ptr<float>(),
                accp, barp,
                g1c.data_ptr<float>(), b1c.data_ptr<float>(),
                HW, TPB, ntiles, chunk, sq_off, invN, (float)eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // ---- conv2 (cuDNN, NHWC) ------------------------------------------
        auto t2 = at::conv2d(y1, w2c, {}, stride, pad, dil, 1);
        if (!t2.is_contiguous(at::MemoryFormat::ChannelsLast))
            t2 = t2.contiguous(at::MemoryFormat::ChannelsLast);

        // ---- GN2 + SiLU + residual + NHWC->NCHW (fused) --------------------
        {
            int chunk = 1;
            int grid  = grid_for(bpsm_r * sm_count, ntiles, &chunk);
            gn_fused_res_nchw_kernel<<<grid, NTHR, 0, stream>>>(
                t2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
                accp + 2L * B * NGROUPS, barp + 1,
                g2c.data_ptr<float>(), b2c.data_ptr<float>(),
                HW, TPB, ntiles, chunk, sq_off, invN, (float)eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out;
    }

    // =========================== legacy 3-kernel path ========================
    int nchunk = HW / 16;
    if (nchunk < 1) nchunk = 1;
    int cap = 2048 / (B > 0 ? B : 1);
    if (cap < 1) cap = 1;
    if (nchunk > cap) nchunk = cap;
    if (nchunk > 2048) nchunk = 2048;

    auto part = at::empty({2L * B * NGROUPS * nchunk}, opts);
    auto mean = at::empty({(long)B * NGROUPS}, opts);
    auto rstd = at::empty({(long)B * NGROUPS}, opts);

    run_groupnorm_stats(t1, part, mean, rstd, B, HW, nchunk, (float)eps, stream);

    auto y1 = at::empty({B, C, H, W}, opts, at::MemoryFormat::ChannelsLast);
    {
        dim3 grid((HW + 7) / 8, B);
        gn_apply_silu_kernel<<<grid, NTHR, 0, stream>>>(
            t1.data_ptr<float>(), y1.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            g1c.data_ptr<float>(), b1c.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto t2 = at::conv2d(y1, w2c, {}, stride, pad, dil, 1);
    if (!t2.is_contiguous(at::MemoryFormat::ChannelsLast))
        t2 = t2.contiguous(at::MemoryFormat::ChannelsLast);

    run_groupnorm_stats(t2, part, mean, rstd, B, HW, nchunk, (float)eps, stream);

    {
        dim3 grid((HW + TP - 1) / TP, C / TC, B);
        gn_apply_silu_res_nchw_kernel<<<grid, NTHR, 0, stream>>>(
            t2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            g2c.data_ptr<float>(), b2c.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_block"],
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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


class ModelNew(nn.Module):
    """Fused Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual.

    Each GroupNorm is now a single grid-persistent kernel (stats + finalize +
    affine/SiLU [+ residual + NHWC->NCHW]) synchronised by a device-wide
    barrier. This module is stateless (all weights arrive as forward inputs),
    so there is no parameter holder to mirror.
    """

    def __init__(self):
        super().__init__()
        self._ext = _ext

    @staticmethod
    def _ref(x, w1, n1w, n1b, w2, n2w, n2b, eps):
        out = F.conv2d(x, w1, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n1w, bias=n1b, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, w2, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n2w, bias=n2b, eps=eps)
        out = F.silu(out)
        return out + x

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        ok = (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.size(1) == 256
            and conv1_weight.dtype == torch.float32
            and conv2_weight.dtype == torch.float32
        )
        if not ok:
            return self._ref(x, conv1_weight, norm1_weight, norm1_bias,
                             conv2_weight, norm2_weight, norm2_bias, eps_f)

        with torch.no_grad():
            return self._ext.fused_block(
                x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps_f,
            )
