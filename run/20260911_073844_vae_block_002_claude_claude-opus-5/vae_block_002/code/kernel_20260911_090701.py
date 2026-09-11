import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ void block_reduce2(float& s, float& sq, float* smem) {
    // smem must have 2 * 32 floats
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(FULL_MASK, s,  off);
        sq += __shfl_down_sync(FULL_MASK, sq, off);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) { smem[warp] = s; smem[32 + warp] = sq; }
    __syncthreads();
    if (warp == 0) {
        s  = (lane < nwarps) ? smem[lane] : 0.f;
        sq = (lane < nwarps) ? smem[32 + lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            s  += __shfl_down_sync(FULL_MASK, s,  off);
            sq += __shfl_down_sync(FULL_MASK, sq, off);
        }
    }
}

// ---- stage 1: partial sums over an NHWC tensor, per (n, group) ----------------
__global__ void gn_partial_kernel(const float* __restrict__ in,
                                  float2* __restrict__ partial,
                                  int P, int C, int CPG, int G,
                                  int nparts, int pixPerPart) {
    int bg   = blockIdx.y;
    int n    = bg / G;
    int g    = bg - n * G;
    int part = blockIdx.x;
    int p0   = part * pixPerPart;
    int p1   = p0 + pixPerPart; if (p1 > P) p1 = P;

    const float* base = in + (long long)n * (long long)P * (long long)C + (long long)(g * CPG);

    float s = 0.f, sq = 0.f;
    if ((CPG & 3) == 0 && (C & 3) == 0) {
        int v = CPG >> 2;
        for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
            const float4* ptr = reinterpret_cast<const float4*>(base + (long long)p * (long long)C);
            #pragma unroll 2
            for (int j = 0; j < v; ++j) {
                float4 t = ptr[j];
                s  += t.x + t.y + t.z + t.w;
                sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
            }
        }
    } else {
        for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
            const float* ptr = base + (long long)p * (long long)C;
            for (int j = 0; j < CPG; ++j) {
                float t = ptr[j];
                s += t; sq += t * t;
            }
        }
    }
    __shared__ float smem[64];
    block_reduce2(s, sq, smem);
    if (threadIdx.x == 0) {
        partial[(long long)bg * nparts + part] = make_float2(s, sq);
    }
}

// ---- stage 1b: partial sums over an NCHW tensor, per (n, group) ---------------
// For a contiguous NCHW tensor the whole group is one contiguous span of length
// CPG*P starting at ((n*C + g*CPG) * P). Grid-stride over that flat span.
__global__ void gn_partial_nchw_kernel(const float* __restrict__ in,
                                       float2* __restrict__ partial,
                                       int P, int C, int CPG, int G,
                                       int nparts, int pixPerPart) {
    int bg   = blockIdx.y;
    int n    = bg / G;
    int g    = bg - n * G;
    int part = blockIdx.x;

    long long spanLen = (long long)CPG * (long long)P;
    long long base_off = ((long long)n * C + (long long)g * CPG) * (long long)P;
    const float* base = in + base_off;

    long long e0 = (long long)part * pixPerPart;
    long long e1 = e0 + pixPerPart; if (e1 > spanLen) e1 = spanLen;

    float s = 0.f, sq = 0.f;
    bool vec_ok = ((spanLen & 3) == 0) && ((base_off & 3) == 0) && ((e0 & 3) == 0);
    if (vec_ok && (e1 - e0) >= 4) {
        long long v0 = e0 >> 2;
        long long v1 = e1 >> 2;                 // floor; tail handled below
        const float4* ptr = reinterpret_cast<const float4*>(base);
        for (long long v = v0 + threadIdx.x; v < v1; v += blockDim.x) {
            float4 t = ptr[v];
            s  += t.x + t.y + t.z + t.w;
            sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
        }
        long long tail0 = v1 << 2;
        for (long long e = tail0 + threadIdx.x; e < e1; e += blockDim.x) {
            float t = base[e];
            s += t; sq += t * t;
        }
    } else {
        for (long long e = e0 + threadIdx.x; e < e1; e += blockDim.x) {
            float t = base[e];
            s += t; sq += t * t;
        }
    }
    __shared__ float smem[64];
    block_reduce2(s, sq, smem);
    if (threadIdx.x == 0) {
        partial[(long long)bg * nparts + part] = make_float2(s, sq);
    }
}

// ---- stage 2: finalize stats -> per (n,c) scale/shift -------------------------
__global__ void gn_finalize_kernel(const float2* __restrict__ partial,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ scale,
                                   float* __restrict__ shift,
                                   int nparts, int C, int CPG, int G,
                                   float eps, float invCount) {
    int bg = blockIdx.x;
    int n  = bg / G;
    int g  = bg - n * G;

    float s = 0.f, sq = 0.f;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) {
        float2 t = partial[(long long)bg * nparts + i];
        s += t.x; sq += t.y;
    }
    __shared__ float smem[64];
    __shared__ float mean_s, rstd_s;
    block_reduce2(s, sq, smem);
    if (threadIdx.x == 0) {
        float mean = s * invCount;
        float var  = sq * invCount - mean * mean;
        if (var < 0.f) var = 0.f;
        mean_s = mean;
        rstd_s = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = mean_s, rstd = rstd_s;
    for (int c = threadIdx.x; c < CPG; c += blockDim.x) {
        int ch = g * CPG + c;
        float sc = gamma[ch] * rstd;
        scale[n * C + ch] = sc;
        shift[n * C + ch] = beta[ch] - mean * sc;
    }
}

// ---- GroupNorm affine + SiLU, NHWC in / NHWC out ------------------------------
__global__ void norm_silu_nhwc_kernel(const float4* __restrict__ in,
                                      float4* __restrict__ out,
                                      const float4* __restrict__ scale,
                                      const float4* __restrict__ shift,
                                      long long totalVec, int vecC, int P) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= totalVec) return;
    int cv       = (int)(v % (long long)vecC);
    long long px = v / (long long)vecC;
    int n        = (int)(px / (long long)P);

    float4 t  = in[v];
    float4 sc = scale[(long long)n * vecC + cv];
    float4 sh = shift[(long long)n * vecC + cv];
    float y0 = t.x * sc.x + sh.x;
    float y1 = t.y * sc.y + sh.y;
    float y2 = t.z * sc.z + sh.z;
    float y3 = t.w * sc.w + sh.w;
    float4 r;
    r.x = y0 / (1.f + __expf(-y0));
    r.y = y1 / (1.f + __expf(-y1));
    r.z = y2 / (1.f + __expf(-y2));
    r.w = y3 / (1.f + __expf(-y3));
    out[v] = r;
}

// scalar fallback (C not divisible by 4)
__global__ void norm_silu_nhwc_scalar_kernel(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             const float* __restrict__ scale,
                                             const float* __restrict__ shift,
                                             long long total, int C, int P) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c        = (int)(i % (long long)C);
    long long px = i / (long long)C;
    int n        = (int)(px / (long long)P);
    float y = in[i] * scale[(long long)n * C + c] + shift[(long long)n * C + c];
    out[i]  = y / (1.f + __expf(-y));
}

// ---- GroupNorm affine + SiLU + residual, NHWC in -> NCHW out ------------------
__global__ void norm_silu_res_t_kernel(const float* __restrict__ in,
                                       const float* __restrict__ res,
                                       float* __restrict__ out,
                                       const float* __restrict__ scale,
                                       const float* __restrict__ shift,
                                       int P, int C) {
    __shared__ float tile[32][33];
    int p0 = blockIdx.x * 32;
    int c0 = blockIdx.y * 32;
    int n  = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c = c0 + tx;
    float sc = 0.f, sh = 0.f;
    if (c < C) { sc = scale[n * C + c]; sh = shift[n * C + c]; }

    const float* ib = in + (long long)n * (long long)P * (long long)C;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int p = p0 + ty + k;
        float v = 0.f;
        if (p < P && c < C) {
            float y = ib[(long long)p * (long long)C + c] * sc + sh;
            v = y / (1.f + __expf(-y));
        }
        tile[ty + k][tx] = v;
    }
    __syncthreads();
    int p = p0 + tx;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int cc = c0 + ty + k;
        if (p < P && cc < C) {
            long long o = ((long long)n * C + cc) * (long long)P + p;
            out[o] = tile[tx][ty + k] + res[o];
        }
    }
}

// ---- GroupNorm affine + SiLU + residual, NCHW in -> NCHW out (flat, no transpose) --
__global__ void norm_silu_res_nchw_in_kernel(const float* __restrict__ in,
                                             const float* __restrict__ res,
                                             float* __restrict__ out,
                                             const float* __restrict__ scale,
                                             const float* __restrict__ shift,
                                             long long total, int C, int P) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    long long cp = i / (long long)P;
    int c        = (int)(cp % (long long)C);
    int n        = (int)(cp / (long long)C);
    float y = in[i] * scale[(long long)n * C + c] + shift[(long long)n * C + c];
    out[i]  = y / (1.f + __expf(-y)) + res[i];
}

// vectorized variant: valid only when (P & 3) == 0, so a float4 group of 4
// consecutive elements along P all share the same (n, c).
__global__ void norm_silu_res_nchw_in_vec_kernel(const float4* __restrict__ in,
                                                  const float4* __restrict__ res,
                                                  float4* __restrict__ out,
                                                  const float* __restrict__ scale,
                                                  const float* __restrict__ shift,
                                                  long long totalVec, int C, int Pvec) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= totalVec) return;
    long long cp = v / (long long)Pvec;
    int c        = (int)(cp % (long long)C);
    int n        = (int)(cp / (long long)C);
    float sc = scale[(long long)n * C + c];
    float sh = shift[(long long)n * C + c];
    float4 t = in[v];
    float4 r4 = res[v];
    float y0 = t.x * sc + sh;
    float y1 = t.y * sc + sh;
    float y2 = t.z * sc + sh;
    float y3 = t.w * sc + sh;
    float4 r;
    r.x = y0 / (1.f + __expf(-y0)) + r4.x;
    r.y = y1 / (1.f + __expf(-y1)) + r4.y;
    r.z = y2 / (1.f + __expf(-y2)) + r4.z;
    r.w = y3 / (1.f + __expf(-y3)) + r4.w;
    out[v] = r;
}

// ---- GroupNorm affine + SiLU, NCHW in -> NHWC out (shared-memory transpose) ---
__global__ void norm_silu_nchw2nhwc_kernel(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           const float* __restrict__ scale,
                                           const float* __restrict__ shift,
                                           int P, int C) {
    __shared__ float tile[32][33];
    int p0 = blockIdx.x * 32;
    int c0 = blockIdx.y * 32;
    int n  = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // read NCHW coalesced along p: element (n, cc, p)
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int cc = c0 + ty + k;
        int p  = p0 + tx;
        float v = 0.f;
        if (p < P && cc < C) {
            float sc = scale[n * C + cc];
            float sh = shift[n * C + cc];
            long long idx = ((long long)n * C + cc) * (long long)P + p;
            float y = in[idx] * sc + sh;
            v = y / (1.f + __expf(-y));
        }
        tile[ty + k][tx] = v;
    }
    __syncthreads();
    // write NHWC coalesced along c: element (n, p, c)
    int c = c0 + tx;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int p = p0 + ty + k;
        if (p < P && c < C) {
            long long o = ((long long)n * P + p) * (long long)C + c;
            out[o] = tile[tx][ty + k];
        }
    }
}

// ------------------------------ host helpers ----------------------------------
static void compute_scale_shift(const torch::Tensor& in_nhwc,
                                const torch::Tensor& gamma,
                                const torch::Tensor& beta,
                                double eps, int G, int B, int C, int P,
                                torch::Tensor& scale, torch::Tensor& shift) {
    int CPG = C / G;
    long long bg = (long long)B * G;
    int target = 1024;
    int nparts = (int)((target + bg - 1) / bg);
    int maxparts = (P + 255) / 256;
    if (maxparts < 1) maxparts = 1;
    if (nparts > maxparts) nparts = maxparts;
    if (nparts < 1) nparts = 1;
    int pixPerPart = (P + nparts - 1) / nparts;
    nparts = (P + pixPerPart - 1) / pixPerPart;   // every part holds >=1 pixel

    auto opts = in_nhwc.options();
    auto partial = torch::empty({bg * nparts, 2}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(nparts, (unsigned)bg);
    gn_partial_kernel<<<grid, 256, 0, stream>>>(
        in_nhwc.data_ptr<float>(),
        reinterpret_cast<float2*>(partial.data_ptr<float>()),
        P, C, CPG, G, nparts, pixPerPart);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float invCount = 1.0f / (float)((double)P * (double)CPG);
    gn_finalize_kernel<<<(unsigned)bg, 256, 0, stream>>>(
        reinterpret_cast<const float2*>(partial.data_ptr<float>()),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        nparts, C, CPG, G, (float)eps, invCount);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// NCHW-input version of compute_scale_shift: uses gn_partial_nchw_kernel to read
// the contiguous per-group span directly out of an NCHW tensor (no NHWC layout
// needed for the reduction pass). gn_finalize_kernel is reused unchanged.
static void compute_scale_shift_nchw(const torch::Tensor& in_nchw,
                                     const torch::Tensor& gamma,
                                     const torch::Tensor& beta,
                                     double eps, int G, int B, int C, int P,
                                     torch::Tensor& scale, torch::Tensor& shift) {
    int CPG = C / G;
    long long bg = (long long)B * G;
    long long spanLen = (long long)CPG * (long long)P;
    int target = 1024;
    int nparts = (int)((target + bg - 1) / bg);
    long long maxparts = (spanLen + 255) / 256;
    if (maxparts < 1) maxparts = 1;
    if ((long long)nparts > maxparts) nparts = (int)maxparts;
    if (nparts < 1) nparts = 1;
    long long pixPerPart = (spanLen + nparts - 1) / nparts;
    nparts = (int)((spanLen + pixPerPart - 1) / pixPerPart);

    auto opts = in_nchw.options();
    auto partial = torch::empty({bg * nparts, 2}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(nparts, (unsigned)bg);
    gn_partial_nchw_kernel<<<grid, 256, 0, stream>>>(
        in_nchw.data_ptr<float>(),
        reinterpret_cast<float2*>(partial.data_ptr<float>()),
        P, C, CPG, G, nparts, (int)pixPerPart);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float invCount = 1.0f / (float)((double)P * (double)CPG);
    gn_finalize_kernel<<<(unsigned)bg, 256, 0, stream>>>(
        reinterpret_cast<const float2*>(partial.data_ptr<float>()),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        nparts, C, CPG, G, (float)eps, invCount);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_nhwc(torch::Tensor in_nhwc, torch::Tensor gamma, torch::Tensor beta,
                           double eps, int64_t G, int64_t B, int64_t C, int64_t P) {
    TORCH_CHECK(in_nhwc.scalar_type() == torch::kFloat32, "float32 only");
    auto opts = in_nhwc.options();
    auto scale = torch::empty({B * C}, opts);
    auto shift = torch::empty({B * C}, opts);
    compute_scale_shift(in_nhwc, gamma, beta, eps, (int)G, (int)B, (int)C, (int)P, scale, shift);

    auto out = torch::empty_like(in_nhwc);
    auto stream = at::cuda::getCurrentCUDAStream();
    long long total = (long long)B * C * P;
    if ((C & 3) == 0) {
        int vecC = (int)C >> 2;
        long long totalVec = total >> 2;
        int threads = 256;
        long long blocks = (totalVec + threads - 1) / threads;
        norm_silu_nhwc_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in_nhwc.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            reinterpret_cast<const float4*>(scale.data_ptr<float>()),
            reinterpret_cast<const float4*>(shift.data_ptr<float>()),
            totalVec, vecC, (int)P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        norm_silu_nhwc_scalar_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
            in_nhwc.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(),
            total, (int)C, (int)P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// gn_silu_res_nchw writes into a caller-provided `out` tensor (contiguous NCHW,
// same sizes as res_nchw) instead of allocating one, so a batch-sliced view of a
// single pre-allocated output buffer can be written directly by either the
// single-stream path or one of the per-chunk streams in the multi-stream path.
torch::Tensor gn_silu_res_nchw(torch::Tensor in_nhwc, torch::Tensor res_nchw, torch::Tensor out,
                               torch::Tensor gamma, torch::Tensor beta,
                               double eps, int64_t G, int64_t B, int64_t C, int64_t P) {
    TORCH_CHECK(in_nhwc.scalar_type() == torch::kFloat32, "float32 only");
    auto opts = in_nhwc.options();
    auto scale = torch::empty({B * C}, opts);
    auto shift = torch::empty({B * C}, opts);
    compute_scale_shift(in_nhwc, gamma, beta, eps, (int)G, (int)B, (int)C, (int)P, scale, shift);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(32, 8);
    dim3 grid((unsigned)((P + 31) / 32), (unsigned)((C + 31) / 32), (unsigned)B);
    norm_silu_res_t_kernel<<<grid, block, 0, stream>>>(
        in_nhwc.data_ptr<float>(), res_nchw.data_ptr<float>(), out.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(), (int)P, (int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- NCHW-input variant of the final fused stage: GroupNorm+SiLU+residual, no
// transpose needed since input is already NCHW (folds the layout restore away
// entirely rather than paying for a shared-memory transpose). ------------------
torch::Tensor gn_silu_res_from_nchw(torch::Tensor in_nchw, torch::Tensor res_nchw, torch::Tensor out,
                                    torch::Tensor gamma, torch::Tensor beta,
                                    double eps, int64_t G, int64_t B, int64_t C, int64_t P) {
    TORCH_CHECK(in_nchw.scalar_type() == torch::kFloat32, "float32 only");
    auto opts = in_nchw.options();
    auto scale = torch::empty({B * C}, opts);
    auto shift = torch::empty({B * C}, opts);
    compute_scale_shift_nchw(in_nchw, gamma, beta, eps, (int)G, (int)B, (int)C, (int)P, scale, shift);

    auto stream = at::cuda::getCurrentCUDAStream();
    long long total = (long long)B * C * P;
    if ((P & 3) == 0) {
        int Pvec = (int)P >> 2;
        long long totalVec = total >> 2;
        int threads = 256;
        long long blocks = (totalVec + threads - 1) / threads;
        norm_silu_res_nchw_in_vec_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in_nchw.data_ptr<float>()),
            reinterpret_cast<const float4*>(res_nchw.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            scale.data_ptr<float>(), shift.data_ptr<float>(),
            totalVec, (int)C, Pvec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        norm_silu_res_nchw_in_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
            in_nchw.data_ptr<float>(), res_nchw.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(),
            total, (int)C, (int)P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// ---- NCHW-input variant of the first fused stage: GroupNorm+SiLU, output as an
// NHWC (channels_last-tagged) tensor via shared-memory transpose, so conv2 still
// gets a channels_last input and keeps its fast cuDNN NHWC tensor-core path. -----
torch::Tensor gn_silu_nhwc_from_nchw(torch::Tensor in_nchw, torch::Tensor gamma, torch::Tensor beta,
                                     double eps, int64_t G, int64_t B, int64_t C, int64_t P) {
    TORCH_CHECK(in_nchw.scalar_type() == torch::kFloat32, "float32 only");
    auto opts = in_nchw.options();
    auto scale = torch::empty({B * C}, opts);
    auto shift = torch::empty({B * C}, opts);
    compute_scale_shift_nchw(in_nchw, gamma, beta, eps, (int)G, (int)B, (int)C, (int)P, scale, shift);

    // sizes must match the logical NCHW tensor shape; strides are channels_last.
    auto out = torch::empty(in_nchw.sizes(), opts.memory_format(at::MemoryFormat::ChannelsLast));

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(32, 8);
    dim3 grid((unsigned)((P + 31) / 32), (unsigned)((C + 31) / 32), (unsigned)B);
    norm_silu_nchw2nhwc_kernel<<<grid, block, 0, stream>>>(
        in_nchw.data_ptr<float>(), out.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(), (int)P, (int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor gn_silu_nhwc(torch::Tensor in_nhwc, torch::Tensor gamma, torch::Tensor beta,
                           double eps, int64_t G, int64_t B, int64_t C, int64_t P);
torch::Tensor gn_silu_res_nchw(torch::Tensor in_nhwc, torch::Tensor res_nchw, torch::Tensor out,
                               torch::Tensor gamma, torch::Tensor beta,
                               double eps, int64_t G, int64_t B, int64_t C, int64_t P);
torch::Tensor gn_silu_res_from_nchw(torch::Tensor in_nchw, torch::Tensor res_nchw, torch::Tensor out,
                                    torch::Tensor gamma, torch::Tensor beta,
                                    double eps, int64_t G, int64_t B, int64_t C, int64_t P);
torch::Tensor gn_silu_nhwc_from_nchw(torch::Tensor in_nchw, torch::Tensor gamma, torch::Tensor beta,
                                     double eps, int64_t G, int64_t B, int64_t C, int64_t P);
'''

_ext = load_inline(
    name="vae_resblock_gn_silu_ms_v2",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu_nhwc", "gn_silu_res_nchw", "gn_silu_res_from_nchw", "gn_silu_nhwc_from_nchw"],
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
    def __init__(self):
        super().__init__()
        self._streams = None  # lazily created on first forward (avoid ctx issues in __init__)

    @staticmethod
    def _cl_weight(w):
        # NOTE: no caching across calls - tensor addresses get recycled by the
        # caching allocator, so a data_ptr-keyed cache can silently return a
        # stale weight. The copy is ~2MB and costs a few microseconds.
        return w if w.is_contiguous(memory_format=torch.channels_last) \
            else w.contiguous(memory_format=torch.channels_last)

    @staticmethod
    def _block(x_nchw, w1, n1w, n1b, w2, n2w, n2b, eps, out):
        # the full fused chain (conv1 -> gn+silu -> conv2 -> gn+silu+res+layout)
        # applied to one batch chunk, writing its result into `out`.
        num_groups = 32
        B, C, H, W = x_nchw.shape
        P = H * W

        # conv #1 : cuDNN
        o = F.conv2d(x_nchw, w1, None, 1, 1)
        if o.is_contiguous(memory_format=torch.channels_last):
            # already NHWC: fused GroupNorm + SiLU (NHWC -> NHWC)
            o = _ext.gn_silu_nhwc(o, n1w, n1b, eps, num_groups, B, C, P)
        elif o.is_contiguous():
            # conv returned plain NCHW: fold the NCHW->NHWC transpose into the
            # GroupNorm+SiLU pass instead of paying for a separate
            # .contiguous(memory_format=channels_last) copy first.
            o = _ext.gn_silu_nhwc_from_nchw(o, n1w, n1b, eps, num_groups, B, C, P)
        else:
            o = o.contiguous(memory_format=torch.channels_last)
            o = _ext.gn_silu_nhwc(o, n1w, n1b, eps, num_groups, B, C, P)

        # conv #2
        o = F.conv2d(o, w2, None, 1, 1)
        if o.is_contiguous(memory_format=torch.channels_last):
            # fused GroupNorm + SiLU + residual + NHWC->NCHW restore, into `out`
            _ext.gn_silu_res_nchw(o, x_nchw, out, n2w, n2b, eps, num_groups, B, C, P)
        elif o.is_contiguous():
            # conv returned plain NCHW already: no transpose needed at all, the
            # final stage becomes a pure flat elementwise+residual pass.
            _ext.gn_silu_res_from_nchw(o, x_nchw, out, n2w, n2b, eps, num_groups, B, C, P)
        else:
            o = o.contiguous(memory_format=torch.channels_last)
            _ext.gn_silu_res_nchw(o, x_nchw, out, n2w, n2b, eps, num_groups, B, C, P)
        return out

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps = float(eps.item())

        assert x.dtype == torch.float32, "float32 only"
        res = x if x.is_contiguous() else x.contiguous()

        B, C, H, W = res.shape

        w1 = self._cl_weight(conv1_weight)
        w2 = self._cl_weight(conv2_weight)
        n1w = norm1_weight.contiguous()
        n1b = norm1_bias.contiguous()
        n2w = norm2_weight.contiguous()
        n2b = norm2_bias.contiguous()

        out = torch.empty_like(res)

        use_ms = (B >= 4) and (res.numel() >= (1 << 23))

        if not use_ms:
            self._block(res, w1, n1w, n1b, w2, n2w, n2b, eps, out)
            return out

        if self._streams is None:
            self._streams = [torch.cuda.Stream(), torch.cuda.Stream()]

        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cur)

        b0 = B // 2
        chunks = [(0, b0), (b0, B)]

        end_events = []
        for i, (s, e) in enumerate(chunks):
            st = self._streams[i]
            st.wait_event(ev)
            with torch.cuda.stream(st):
                res.record_stream(st)
                out.record_stream(st)
                w1.record_stream(st)
                w2.record_stream(st)
                self._block(res[s:e], w1, n1w, n1b, w2, n2w, n2b, eps, out[s:e])
            e2 = torch.cuda.Event()
            e2.record(st)
            end_events.append(e2)

        for e2 in end_events:
            cur.wait_event(e2)

        return out
