# ==========================================================================
# ModelNew — SOL problem 002: Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual
#
# Optimisation applied here: gridsync_gn_fusion
#   Each GroupNorm's 3 launches (reduce -> finalize -> apply) are merged into ONE
#   persistent kernel whose phase-2 blocks re-touch exactly the (n, pixel) range they
#   read in phase 1, so the second read of the conv output hits L2 instead of DRAM.
#   A manual generation/sense grid barrier is used (no -rdc needed); the launch goes
#   through cudaLaunchCooperativeKernel so residency is validated by the driver, and
#   any failure falls back to the original 3-kernel path.
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE  32
#define TROWS 8
#define FT    256   // threads per block of the fused persistent kernels

__device__ __forceinline__ float silu_f(float z){
    return z / (1.0f + expf(-z));
}

// ---------------------------------------------------------------- transpose (unchanged)
__global__ void nchw2nhwc_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int C, long HW)
{
    __shared__ float tile[TILE][TILE + 1];
    long p0 = (long)blockIdx.x * TILE;
    int  c0 = blockIdx.y * TILE;
    int  n  = blockIdx.z;
    const float* xn = x + (size_t)n * (size_t)C * (size_t)HW;
    float*       yn = y + (size_t)n * (size_t)HW * (size_t)C;
    int tx = threadIdx.x, ty = threadIdx.y;

    #pragma unroll
    for (int i = 0; i < TILE; i += TROWS) {
        int  c = c0 + ty + i;
        long p = p0 + tx;
        float v = 0.f;
        if (c < C && p < HW) v = xn[(size_t)c * (size_t)HW + (size_t)p];
        tile[ty + i][tx] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < TILE; i += TROWS) {
        long p = p0 + ty + i;
        int  c = c0 + tx;
        if (c < C && p < HW) yn[(size_t)p * (size_t)C + (size_t)c] = tile[tx][ty + i];
    }
}

// ================================================================= legacy 3-kernel path
__global__ void gn_reduce_kernel(const float* __restrict__ y,
                                 double* __restrict__ psum,
                                 double* __restrict__ psq,
                                 int C, long HW, int G, int CPG,
                                 int ppb, long chunk, int nch)
{
    extern __shared__ float sh[];
    float* s_s = sh;
    float* s_q = sh + blockDim.x;

    int tid = threadIdx.x;
    int g   = tid % G;
    int dp  = tid / G;
    int n   = blockIdx.y;

    long pstart = (long)blockIdx.x * chunk;
    long pend   = pstart + chunk;
    if (pend > HW) pend = HW;

    const float* base = y + (size_t)n * (size_t)HW * (size_t)C + (size_t)g * (size_t)CPG;

    float s = 0.f, q = 0.f;
    if ((CPG & 3) == 0) {
        int nv = CPG >> 2;
        for (long p = pstart + dp; p < pend; p += ppb) {
            const float4* v4 = reinterpret_cast<const float4*>(base + (size_t)p * (size_t)C);
            for (int j = 0; j < nv; ++j) {
                float4 v = v4[j];
                s += v.x + v.y + v.z + v.w;
                q += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    } else {
        for (long p = pstart + dp; p < pend; p += ppb) {
            const float* v = base + (size_t)p * (size_t)C;
            for (int j = 0; j < CPG; ++j) { float t = v[j]; s += t; q += t * t; }
        }
    }

    s_s[tid] = s; s_q[tid] = q;
    __syncthreads();
    for (int stride = ppb >> 1; stride > 0; stride >>= 1) {
        if (dp < stride) {
            s_s[tid] += s_s[tid + stride * G];
            s_q[tid] += s_q[tid + stride * G];
        }
        __syncthreads();
    }
    if (dp == 0) {
        size_t off = (size_t)(n * G + g) * (size_t)nch + (size_t)blockIdx.x;
        psum[off] = (double)s_s[tid];
        psq[off]  = (double)s_q[tid];
    }
}

__global__ void gn_finalize_kernel(const double* __restrict__ psum,
                                   const double* __restrict__ psq,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nch, double count, double eps)
{
    __shared__ double sh[256];
    int bg  = blockIdx.x;
    int tid = threadIdx.x;
    double s = 0.0, q = 0.0;
    for (int i = tid; i < nch; i += blockDim.x) {
        s += psum[(size_t)bg * (size_t)nch + i];
        q += psq [(size_t)bg * (size_t)nch + i];
    }
    sh[tid] = s; sh[128 + tid] = q;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) { sh[tid] += sh[tid + st]; sh[128 + tid] += sh[128 + tid + st]; }
        __syncthreads();
    }
    if (tid == 0) {
        double m = sh[0] / count;
        double v = sh[128] / count - m * m;
        if (v < 0.0) v = 0.0;
        mean[bg] = (float)m;
        rstd[bg] = (float)(1.0 / sqrt(v + eps));
    }
}

__global__ void gn_apply_nhwc_fast(float* __restrict__ y,
                                   const float* __restrict__ mean,
                                   const float* __restrict__ rstd,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   int C4, long HW, int G, int CPG, int ppb)
{
    int n  = blockIdx.y;
    int c4 = threadIdx.x % C4;
    int dp = threadIdx.x / C4;
    int c  = c4 << 2;
    int g  = c / CPG;

    float m = mean[n * G + g];
    float r = rstd[n * G + g];
    float a0 = r * w[c],     a1 = r * w[c + 1], a2 = r * w[c + 2], a3 = r * w[c + 3];
    float d0 = b[c]     - m * a0;
    float d1 = b[c + 1] - m * a1;
    float d2 = b[c + 2] - m * a2;
    float d3 = b[c + 3] - m * a3;

    float4* base = reinterpret_cast<float4*>(y) + (size_t)n * (size_t)HW * (size_t)C4;
    for (long p = (long)blockIdx.x * ppb + dp; p < HW; p += (long)gridDim.x * ppb) {
        float4 v = base[(size_t)p * (size_t)C4 + (size_t)c4];
        v.x = silu_f(v.x * a0 + d0);
        v.y = silu_f(v.y * a1 + d1);
        v.z = silu_f(v.z * a2 + d2);
        v.w = silu_f(v.w * a3 + d3);
        base[(size_t)p * (size_t)C4 + (size_t)c4] = v;
    }
}

__global__ void gn_apply_nhwc_gen(float* __restrict__ y,
                                  const float* __restrict__ mean,
                                  const float* __restrict__ rstd,
                                  const float* __restrict__ w,
                                  const float* __restrict__ b,
                                  int C, long HW, int G, int CPG, long total)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        int  c   = (int)(i % (long)C);
        long pix = i / (long)C;
        int  n   = (int)(pix / HW);
        int  g   = c / CPG;
        float v = y[i];
        float z = (v - mean[n * G + g]) * rstd[n * G + g] * w[c] + b[c];
        y[i] = silu_f(z);
    }
}

__global__ void gn_apply_out_kernel(const float* __restrict__ y,   // NHWC
                                    const float* __restrict__ res, // NCHW
                                    float* __restrict__ out,       // NCHW
                                    const float* __restrict__ mean,
                                    const float* __restrict__ rstd,
                                    const float* __restrict__ w,
                                    const float* __restrict__ b,
                                    int C, long HW, int G, int CPG)
{
    __shared__ float tile[TILE][TILE + 1];
    long p0 = (long)blockIdx.x * TILE;
    int  c0 = blockIdx.y * TILE;
    int  n  = blockIdx.z;
    const float* yn = y + (size_t)n * (size_t)HW * (size_t)C;
    int tx = threadIdx.x, ty = threadIdx.y;

    #pragma unroll
    for (int i = 0; i < TILE; i += TROWS) {
        int  c = c0 + tx;
        long p = p0 + ty + i;
        float t = 0.f;
        if (c < C && p < HW) {
            float v = yn[(size_t)p * (size_t)C + (size_t)c];
            int   g = c / CPG;
            float z = (v - mean[n * G + g]) * rstd[n * G + g] * w[c] + b[c];
            t = silu_f(z);
        }
        tile[ty + i][tx] = t;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < TILE; i += TROWS) {
        int  c = c0 + ty + i;
        long p = p0 + tx;
        if (c < C && p < HW) {
            size_t off = ((size_t)n * (size_t)C + (size_t)c) * (size_t)HW + (size_t)p;
            out[off] = tile[tx][ty + i] + res[off];
        }
    }
}

// ===================================================== fused persistent GN (gridsync)
// ------------------------------------------------------------------- grid barrier
__device__ unsigned int g_bar_count = 0u;
__device__ unsigned int g_bar_gen   = 0u;

__device__ __forceinline__ void grid_bar(unsigned int nblk)
{
    __syncthreads();                       // block-wide release of all prior global writes
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned int gen = *(volatile unsigned int*)&g_bar_gen;
        if (atomicAdd(&g_bar_count, 1u) == nblk - 1u) {
            atomicExch(&g_bar_count, 0u);
            __threadfence();
            atomicAdd(&g_bar_gen, 1u);
        } else {
            while (*(volatile unsigned int*)&g_bar_gen == gen) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(96);
#endif
            }
        }
        __threadfence();
    }
    __syncthreads();
}

// ------------------------------------------------------- phase-1 helpers (shared code)
__device__ __forceinline__ void gn_block_flush(float s, float q,
        float* s_s, float* s_q,
        int C4, int ppb, int G, int per, int n,
        double* __restrict__ psum, double* __restrict__ psq, unsigned int nblk)
{
    const int tid = threadIdx.x;
    __syncthreads();
    s_s[tid] = s; s_q[tid] = q;
    __syncthreads();
    for (int st = ppb >> 1; st > 0; st >>= 1) {
        if (tid < st * C4) {
            s_s[tid] += s_s[tid + st * C4];
            s_q[tid] += s_q[tid + st * C4];
        }
        __syncthreads();
    }
    for (int g = tid; g < G; g += FT) {
        float ss = 0.f, qq = 0.f;
        for (int j = 0; j < per; ++j) { ss += s_s[g * per + j]; qq += s_q[g * per + j]; }
        size_t off = (size_t)(n * G + g) * (size_t)nblk + (size_t)blockIdx.x;
        psum[off] = (double)ss;
        psq[off]  = (double)qq;
    }
    __syncthreads();
}

// phase 1: (n, pixel-chunk) partitioned statistics, NHWC float4 reads
__device__ __forceinline__ void gn_phase1_stats(const float* __restrict__ y,
        double* __restrict__ psum, double* __restrict__ psq,
        float* s_s, float* s_q,
        int B, int C4, long HW, int G, int CPG, int ppb,
        int nchunk, long total_chunks, unsigned int nblk)
{
    const int tid = threadIdx.x;
    const int c4  = tid % C4;
    const int dp  = tid / C4;
    const int per = CPG >> 2;

    // every block writes ALL of its (n,g) slots -> no unwritten tail
    for (int k = tid; k < B * G; k += FT) {
        psum[(size_t)k * (size_t)nblk + (size_t)blockIdx.x] = 0.0;
        psq [(size_t)k * (size_t)nblk + (size_t)blockIdx.x] = 0.0;
    }
    __syncthreads();

    float s = 0.f, q = 0.f;
    int cur_n = -1;
    for (long i = (long)blockIdx.x; i < total_chunks; i += (long)nblk) {
        int  n  = (int)(i / (long)nchunk);
        long p0 = (i - (long)n * (long)nchunk) * (long)TILE;
        if (n != cur_n) {
            if (cur_n >= 0)
                gn_block_flush(s, q, s_s, s_q, C4, ppb, G, per, cur_n, psum, psq, nblk);
            s = 0.f; q = 0.f; cur_n = n;
        }
        const float4* base = reinterpret_cast<const float4*>(y)
                             + (size_t)n * (size_t)HW * (size_t)C4;
        for (int t = dp; t < TILE; t += ppb) {
            long p = p0 + (long)t;
            if (p < HW) {
                float4 v = base[(size_t)p * (size_t)C4 + (size_t)c4];
                s += v.x + v.y + v.z + v.w;
                q = fmaf(v.x, v.x, q);
                q = fmaf(v.y, v.y, q);
                q = fmaf(v.z, v.z, q);
                q = fmaf(v.w, v.w, q);
            }
        }
    }
    if (cur_n >= 0)
        gn_block_flush(s, q, s_s, s_q, C4, ppb, G, per, cur_n, psum, psq, nblk);
}

// in-kernel finalize: blocks bg = blockIdx.x, blockIdx.x+nblk, ... over B*G
__device__ __forceinline__ void gn_finalize_inline(const double* __restrict__ psum,
        const double* __restrict__ psq,
        float* __restrict__ mean, float* __restrict__ rstd,
        double* d_a, double* d_b,
        int BG, unsigned int nblk, double count, double eps)
{
    const int tid = threadIdx.x;
    for (int bg = (int)blockIdx.x; bg < BG; bg += (int)nblk) {
        const volatile double* ps = psum + (size_t)bg * (size_t)nblk;
        const volatile double* pq = psq  + (size_t)bg * (size_t)nblk;
        double sa = 0.0, qa = 0.0;
        for (unsigned int k = (unsigned int)tid; k < nblk; k += FT) {
            sa += ps[k]; qa += pq[k];
        }
        __syncthreads();
        d_a[tid] = sa; d_b[tid] = qa;
        __syncthreads();
        for (int st = FT >> 1; st > 0; st >>= 1) {
            if (tid < st) { d_a[tid] += d_a[tid + st]; d_b[tid] += d_b[tid + st]; }
            __syncthreads();
        }
        if (tid == 0) {
            double m = d_a[0] / count;
            double v = d_b[0] / count - m * m;
            if (v < 0.0) v = 0.0;
            mean[bg] = (float)m;
            rstd[bg] = (float)(1.0 / sqrt(v + eps));
        }
        __syncthreads();
    }
}

// ------------------------------------- GN1: stats + affine + SiLU, NHWC in-place, fused
__global__ void gn_fused_inplace_kernel(float* __restrict__ y,
        const float* __restrict__ w, const float* __restrict__ b,
        float* __restrict__ mean, float* __restrict__ rstd,
        double* __restrict__ psum, double* __restrict__ psq,
        int B, int C4, long HW, int G, int CPG, int ppb, int nchunk,
        long total_chunks, unsigned int nblk, double count, double eps)
{
    extern __shared__ __align__(16) char smem_raw[];
    float*  s_s = (float*)smem_raw;
    float*  s_q = s_s + FT;
    double* d_a = (double*)smem_raw;
    double* d_b = d_a + FT;

    const int tid = threadIdx.x;
    const int c4  = tid % C4;
    const int dp  = tid / C4;
    const int c   = c4 << 2;
    const int gg  = c / CPG;

    // ---- phase 1
    gn_phase1_stats(y, psum, psq, s_s, s_q, B, C4, HW, G, CPG, ppb,
                    nchunk, total_chunks, nblk);

    // ---- barrier A
    grid_bar(nblk);

    // ---- finalize (in-kernel)
    gn_finalize_inline(psum, psq, mean, rstd, d_a, d_b, B * G, nblk, count, eps);

    // ---- barrier B
    grid_bar(nblk);

    // ---- phase 2 : same chunks, same order -> L2-resident re-read
    const float4 wv = reinterpret_cast<const float4*>(w)[c4];
    const float4 bv = reinterpret_cast<const float4*>(b)[c4];
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    int prev_n = -1;

    for (long i = (long)blockIdx.x; i < total_chunks; i += (long)nblk) {
        int  n  = (int)(i / (long)nchunk);
        long p0 = (i - (long)n * (long)nchunk) * (long)TILE;
        if (n != prev_n) {
            float m = *(const volatile float*)(mean + n * G + gg);
            float r = *(const volatile float*)(rstd + n * G + gg);
            a0 = r * wv.x; d0 = bv.x - m * a0;
            a1 = r * wv.y; d1 = bv.y - m * a1;
            a2 = r * wv.z; d2 = bv.z - m * a2;
            a3 = r * wv.w; d3 = bv.w - m * a3;
            prev_n = n;
        }
        float4* base = reinterpret_cast<float4*>(y) + (size_t)n * (size_t)HW * (size_t)C4;
        for (int t = dp; t < TILE; t += ppb) {
            long p = p0 + (long)t;
            if (p < HW) {
                size_t off = (size_t)p * (size_t)C4 + (size_t)c4;
                float4 v = base[off];
                v.x = silu_f(fmaf(v.x, a0, d0));
                v.y = silu_f(fmaf(v.y, a1, d1));
                v.z = silu_f(fmaf(v.z, a2, d2));
                v.w = silu_f(fmaf(v.w, a3, d3));
                base[off] = v;
            }
        }
    }
    (void)c;
}

// ---------- GN2: stats + affine + SiLU + residual + NHWC->NCHW transpose, fused
__global__ void gn_fused_out_kernel(const float* __restrict__ y,    // NHWC
        const float* __restrict__ res,                              // NCHW
        float* __restrict__ out,                                    // NCHW
        const float* __restrict__ w, const float* __restrict__ b,
        float* __restrict__ mean, float* __restrict__ rstd,
        double* __restrict__ psum, double* __restrict__ psq,
        int B, int C, int C4, long HW, int G, int CPG, int ppb, int nchunk,
        long total_chunks, unsigned int nblk, int s0off, double count, double eps)
{
    extern __shared__ __align__(16) char smem_raw[];
    float*  s_s  = (float*)smem_raw;
    float*  s_q  = s_s + FT;
    double* d_a  = (double*)smem_raw;
    double* d_b  = d_a + FT;
    float*  tile = (float*)smem_raw;                 // TILE*(TILE+1) floats
    float*  sm_m = (float*)(smem_raw + s0off);       // G floats
    float*  sm_r = sm_m + G;                         // G floats

    const int tid = threadIdx.x;

    // ---- phase 1
    gn_phase1_stats(y, psum, psq, s_s, s_q, B, C4, HW, G, CPG, ppb,
                    nchunk, total_chunks, nblk);

    // ---- barrier A
    grid_bar(nblk);

    // ---- finalize (in-kernel)
    gn_finalize_inline(psum, psq, mean, rstd, d_a, d_b, B * G, nblk, count, eps);

    // ---- barrier B
    grid_bar(nblk);

    // ---- phase 2 : same (n, pixel chunk), sweep ALL channels through a 32x33 tile
    const int tx  = tid & 31;
    const int ty  = tid >> 5;
    const int nct = (C + TILE - 1) / TILE;
    int prev_n = -1;

    for (long i = (long)blockIdx.x; i < total_chunks; i += (long)nblk) {
        int  n  = (int)(i / (long)nchunk);
        long p0 = (i - (long)n * (long)nchunk) * (long)TILE;
        if (n != prev_n) {
            __syncthreads();
            for (int g = tid; g < G; g += FT) {
                sm_m[g] = *(const volatile float*)(mean + n * G + g);
                sm_r[g] = *(const volatile float*)(rstd + n * G + g);
            }
            __syncthreads();
            prev_n = n;
        }
        const float* yn = y + (size_t)n * (size_t)HW * (size_t)C;

        for (int ct = 0; ct < nct; ++ct) {
            int c0 = ct * TILE;
            int cx = c0 + tx;
            float wx = 0.f, bx = 0.f, mx = 0.f, rx = 0.f;
            if (cx < C) {
                wx = w[cx]; bx = b[cx];
                int g = cx / CPG;
                mx = sm_m[g]; rx = sm_r[g];
            }
            #pragma unroll
            for (int r = 0; r < TILE; r += TROWS) {
                long p = p0 + (long)(ty + r);
                float t = 0.f;
                if (cx < C && p < HW) {
                    float v = yn[(size_t)p * (size_t)C + (size_t)cx];
                    float z = (v - mx) * rx * wx + bx;
                    t = silu_f(z);
                }
                tile[(ty + r) * (TILE + 1) + tx] = t;
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < TILE; r += TROWS) {
                int  cy = c0 + ty + r;
                long p  = p0 + (long)tx;
                if (cy < C && p < HW) {
                    size_t off = ((size_t)n * (size_t)C + (size_t)cy) * (size_t)HW + (size_t)p;
                    out[off] = tile[tx * (TILE + 1) + ty + r] + res[off];
                }
            }
            __syncthreads();
        }
    }
}

// =========================================================== host side helpers
static inline int pow2_floor(int v) {
    int r = 1;
    while ((r << 1) <= v) r <<= 1;
    return r;
}

static int get_num_sms() {
    static int sms = -1;
    if (sms < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return sms;
}

static bool coop_supported() {
    static int s = -1;
    if (s < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&s, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess) s = 0;
    }
    return s != 0;
}

// eligibility + grid sizing for the fused persistent kernels
static bool fused_cfg(const void* kern, int C, int G, long HW, int B,
                      int shmem, int& C4, int& ppb, int& nchunk,
                      long& total_chunks, int& grid)
{
    if (!coop_supported()) return false;
    if (C % 4 != 0) return false;
    int CPG = C / G;
    if (CPG % 4 != 0) return false;
    C4 = C / 4;
    if (C4 > FT || (FT % C4) != 0) return false;
    ppb = FT / C4;
    if (pow2_floor(ppb) != ppb) return false;
    nchunk = (int)((HW + TILE - 1) / TILE);
    if (nchunk < 1) return false;
    total_chunks = (long)B * (long)nchunk;
    if (total_chunks < 1) return false;

    int blocksPerSM = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kern, FT,
                                                      (size_t)shmem) != cudaSuccess)
        return false;
    if (blocksPerSM < 1) return false;
    long maxg = (long)get_num_sms() * (long)blocksPerSM;
    if (maxg < 1) return false;
    long g = total_chunks < maxg ? total_chunks : maxg;
    if (g < 1 || g > 2147483000L) return false;
    grid = (int)g;
    return true;
}

static void compute_stats(const float* yptr, int B, int C, long HW, int G, double eps,
                          const torch::TensorOptions& fopts,
                          torch::Tensor& mean, torch::Tensor& rstd)
{
    int CPG = C / G;
    auto stream = at::cuda::getCurrentCUDAStream();

    int ppb = 1;
    if (G <= 256) ppb = pow2_floor(256 / G);
    if (ppb < 1) ppb = 1;
    int block = G * ppb;
    TORCH_CHECK(block <= 1024, "group count too large");

    int nch_target = (int)((1024 + B - 1) / B);
    long nch_max_l = (HW + ppb - 1) / ppb;
    int nch = nch_target;
    if ((long)nch > nch_max_l) nch = (int)nch_max_l;
    if (nch < 1) nch = 1;
    long chunk = (HW + nch - 1) / nch;
    chunk = ((chunk + ppb - 1) / ppb) * ppb;
    if (chunk < 1) chunk = 1;
    nch = (int)((HW + chunk - 1) / chunk);

    auto dopts = fopts.dtype(torch::kDouble);
    auto psum = torch::empty({(long)B * G * nch}, dopts);
    auto psq  = torch::empty({(long)B * G * nch}, dopts);

    dim3 grid(nch, B);
    size_t shmem = (size_t)2 * block * sizeof(float);
    gn_reduce_kernel<<<grid, block, shmem, stream>>>(
        yptr, psum.data_ptr<double>(), psq.data_ptr<double>(),
        C, HW, G, CPG, ppb, chunk, nch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mean = torch::empty({(long)B * G}, fopts);
    rstd = torch::empty({(long)B * G}, fopts);
    double count = (double)HW * (double)CPG;
    gn_finalize_kernel<<<B * G, 128, 0, stream>>>(
        psum.data_ptr<double>(), psq.data_ptr<double>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), nch, count, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------ exported
torch::Tensor nchw_to_nhwc(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(x.dim() == 4 && x.is_contiguous(), "contiguous NCHW expected");
    int B = (int)x.size(0), C = (int)x.size(1);
    long H = x.size(2), W = x.size(3), HW = H * W;
    auto out = torch::empty({B, H, W, C}, x.options());
    dim3 block(TILE, TROWS);
    dim3 grid((unsigned)((HW + TILE - 1) / TILE), (unsigned)((C + TILE - 1) / TILE), (unsigned)B);
    auto stream = at::cuda::getCurrentCUDAStream();
    nchw2nhwc_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), C, HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void gn_silu_nhwc_(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps, int64_t G_)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(y.dim() == 4, "4D expected");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "channels_last expected");
    int B = (int)y.size(0), C = (int)y.size(1), G = (int)G_;
    long HW = y.size(2) * y.size(3);
    TORCH_CHECK(C % G == 0, "C%G");
    int CPG = C / G;
    auto wc = w.is_contiguous() ? w : w.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();

    // ---------------- fused persistent path (gridsync_gn_fusion) ----------------
    {
        int shmem = 2 * FT * (int)sizeof(double);        // == max(2*FT*float, 2*FT*double)
        int C4 = 0, ppb = 0, nchunk = 0, grid = 0;
        long total_chunks = 0;
        if (fused_cfg((const void*)gn_fused_inplace_kernel, C, G, HW, B, shmem,
                      C4, ppb, nchunk, total_chunks, grid)) {
            auto dopts = y.options().dtype(torch::kDouble);
            auto psum = torch::empty({(long)B * G * grid}, dopts);
            auto psq  = torch::empty({(long)B * G * grid}, dopts);
            auto mean = torch::empty({(long)B * G}, y.options());
            auto rstd = torch::empty({(long)B * G}, y.options());

            float*  p_y    = y.data_ptr<float>();
            const float* p_w = wc.data_ptr<float>();
            const float* p_b = bc.data_ptr<float>();
            float*  p_mean = mean.data_ptr<float>();
            float*  p_rstd = rstd.data_ptr<float>();
            double* p_ps   = psum.data_ptr<double>();
            double* p_pq   = psq.data_ptr<double>();
            int  a_B = B, a_C4 = C4, a_G = G, a_CPG = CPG, a_ppb = ppb, a_nchunk = nchunk;
            long a_HW = HW, a_tc = total_chunks;
            unsigned int a_nblk = (unsigned int)grid;
            double a_count = (double)HW * (double)CPG, a_eps = eps;

            void* args[] = { &p_y, &p_w, &p_b, &p_mean, &p_rstd, &p_ps, &p_pq,
                             &a_B, &a_C4, &a_HW, &a_G, &a_CPG, &a_ppb, &a_nchunk,
                             &a_tc, &a_nblk, &a_count, &a_eps };
            cudaError_t err = cudaLaunchCooperativeKernel(
                (const void*)gn_fused_inplace_kernel, dim3((unsigned)grid), dim3(FT),
                args, (size_t)shmem, stream);
            if (err == cudaSuccess) return;
            (void)cudaGetLastError();     // fall through to the legacy 3-kernel path
        }
    }

    // ---------------------------- legacy fallback ----------------------------
    torch::Tensor mean, rstd;
    compute_stats(y.data_ptr<float>(), B, C, HW, G, eps, y.options(), mean, rstd);

    bool fast = (C % 4 == 0) && (CPG % 4 == 0);
    int C4 = C / 4;
    int T = 0, ppb = 0;
    if (fast) {
        if (C4 <= 256 && (256 % C4) == 0) { T = 256; ppb = 256 / C4; }
        else if (C4 <= 1024)              { T = C4;  ppb = 1; }
        else fast = false;
    }
    if (fast) {
        long gx = (HW + ppb - 1) / ppb;
        if (gx > 8192) gx = 8192;
        if (gx < 1) gx = 1;
        dim3 grid((unsigned)gx, (unsigned)B);
        gn_apply_nhwc_fast<<<grid, T, 0, stream>>>(
            y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            wc.data_ptr<float>(), bc.data_ptr<float>(), C4, HW, G, CPG, ppb);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        long total = (long)B * HW * (long)C;
        long gx = (total + 255) / 256;
        if (gx > 65535) gx = 65535;
        if (gx < 1) gx = 1;
        gn_apply_nhwc_gen<<<(unsigned)gx, 256, 0, stream>>>(
            y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            wc.data_ptr<float>(), bc.data_ptr<float>(), C, HW, G, CPG, total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

torch::Tensor gn_silu_add_nchw(torch::Tensor y, torch::Tensor res,
                               torch::Tensor w, torch::Tensor b,
                               double eps, int64_t G_)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "channels_last expected");
    TORCH_CHECK(res.is_contiguous(), "contiguous residual expected");
    TORCH_CHECK(res.sizes() == y.sizes(), "shape mismatch");
    int B = (int)y.size(0), C = (int)y.size(1), G = (int)G_;
    long H = y.size(2), W = y.size(3), HW = H * W;
    TORCH_CHECK(C % G == 0, "C%G");
    int CPG = C / G;
    auto wc = w.is_contiguous() ? w : w.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto out = torch::empty({B, C, H, W}, res.options());

    // ---------------- fused persistent path (gridsync_gn_fusion) ----------------
    {
        int s0 = TILE * (TILE + 1) * (int)sizeof(float);     // 4224 B tile
        int s0d = 2 * FT * (int)sizeof(double);              // 4096 B fp64 finalize
        if (s0d > s0) s0 = s0d;
        s0 = ((s0 + 15) / 16) * 16;
        int shmem = s0 + 2 * G * (int)sizeof(float);
        int C4 = 0, ppb = 0, nchunk = 0, grid = 0;
        long total_chunks = 0;
        if (fused_cfg((const void*)gn_fused_out_kernel, C, G, HW, B, shmem,
                      C4, ppb, nchunk, total_chunks, grid)) {
            auto dopts = y.options().dtype(torch::kDouble);
            auto psum = torch::empty({(long)B * G * grid}, dopts);
            auto psq  = torch::empty({(long)B * G * grid}, dopts);
            auto mean = torch::empty({(long)B * G}, y.options());
            auto rstd = torch::empty({(long)B * G}, y.options());

            const float* p_y  = y.data_ptr<float>();
            const float* p_r  = res.data_ptr<float>();
            float* p_o        = out.data_ptr<float>();
            const float* p_w  = wc.data_ptr<float>();
            const float* p_b  = bc.data_ptr<float>();
            float* p_mean     = mean.data_ptr<float>();
            float* p_rstd     = rstd.data_ptr<float>();
            double* p_ps      = psum.data_ptr<double>();
            double* p_pq      = psq.data_ptr<double>();
            int a_B = B, a_C = C, a_C4 = C4, a_G = G, a_CPG = CPG;
            int a_ppb = ppb, a_nchunk = nchunk, a_s0 = s0;
            long a_HW = HW, a_tc = total_chunks;
            unsigned int a_nblk = (unsigned int)grid;
            double a_count = (double)HW * (double)CPG, a_eps = eps;

            void* args[] = { &p_y, &p_r, &p_o, &p_w, &p_b, &p_mean, &p_rstd, &p_ps, &p_pq,
                             &a_B, &a_C, &a_C4, &a_HW, &a_G, &a_CPG, &a_ppb, &a_nchunk,
                             &a_tc, &a_nblk, &a_s0, &a_count, &a_eps };
            cudaError_t err = cudaLaunchCooperativeKernel(
                (const void*)gn_fused_out_kernel, dim3((unsigned)grid), dim3(FT),
                args, (size_t)shmem, stream);
            if (err == cudaSuccess) return out;
            (void)cudaGetLastError();     // fall through to the legacy 3-kernel path
        }
    }

    // ---------------------------- legacy fallback ----------------------------
    torch::Tensor mean, rstd;
    compute_stats(y.data_ptr<float>(), B, C, HW, G, eps, y.options(), mean, rstd);

    dim3 block(TILE, TROWS);
    dim3 grid((unsigned)((HW + TILE - 1) / TILE), (unsigned)((C + TILE - 1) / TILE), (unsigned)B);
    gn_apply_out_kernel<<<grid, block, 0, stream>>>(
        y.data_ptr<float>(), res.data_ptr<float>(), out.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        wc.data_ptr<float>(), bc.data_ptr<float>(), C, HW, G, CPG);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor nchw_to_nhwc(torch::Tensor x);
void gn_silu_nhwc_(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps, int64_t G_);
torch::Tensor gn_silu_add_nchw(torch::Tensor y, torch::Tensor res, torch::Tensor w,
                               torch::Tensor b, double eps, int64_t G_);
'''

_ext = load_inline(
    name="sol002_gn_silu_res_gridsync",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["nchw_to_nhwc", "gn_silu_nhwc_", "gn_silu_add_nchw"],
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

_NUM_GROUPS = 32


class ModelNew(nn.Module):
    """GN stats / affine+SiLU / residual / layout conversions run in custom CUDA; each
    GroupNorm is now ONE persistent cooperative kernel (stats -> grid barrier -> finalize
    -> grid barrier -> apply) so the conv output is re-read from L2 instead of DRAM.
    The two 3x3 convolutions stay on the vendor TF32 NHWC implicit-GEMM, fed channels_last
    so cuDNN performs no internal layout copies."""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        e = float(eps.item()) if torch.is_tensor(eps) else float(eps)

        xc = x if x.is_contiguous() else x.contiguous()

        # NCHW -> NHWC (custom tiled transpose); view it back as a channels_last (B,C,H,W)
        xh = _ext.nchw_to_nhwc(xc)               # (B, H, W, C) contiguous
        xl = xh.permute(0, 3, 1, 2)              # (B, C, H, W), channels_last strides

        w1 = conv1_weight
        if not w1.is_contiguous(memory_format=torch.channels_last):
            w1 = w1.contiguous(memory_format=torch.channels_last)
        o = F.conv2d(xl, w1, None, 1, 1)
        if not o.is_contiguous(memory_format=torch.channels_last):
            o = o.contiguous(memory_format=torch.channels_last)

        # GroupNorm + SiLU, single fused persistent kernel, in-place
        _ext.gn_silu_nhwc_(o, norm1_weight, norm1_bias, e, _NUM_GROUPS)

        w2 = conv2_weight
        if not w2.is_contiguous(memory_format=torch.channels_last):
            w2 = w2.contiguous(memory_format=torch.channels_last)
        o2 = F.conv2d(o, w2, None, 1, 1)
        if not o2.is_contiguous(memory_format=torch.channels_last):
            o2 = o2.contiguous(memory_format=torch.channels_last)

        # GroupNorm + SiLU + residual add + NHWC->NCHW, single fused persistent kernel
        return _ext.gn_silu_add_nchw(o2, xc, norm2_weight, norm2_bias, e, _NUM_GROUPS)
