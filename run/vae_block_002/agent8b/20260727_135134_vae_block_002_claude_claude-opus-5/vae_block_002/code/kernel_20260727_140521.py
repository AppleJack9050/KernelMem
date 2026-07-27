# ==========================================================================
# ModelNew — SOL problem 002_vae_conv3x3_groupnorm_silu_residual_fused
#   Optimisation: nhwc_layout_switch
#     * cuDNN mainloop (F.conv2d) is fed channels_last tensors so it dispatches
#       NHWC TF32 tensor-core implicit-GEMM kernels instead of NCHW FFMA.
#     * All GN-stats / GN-affine / SiLU / residual-add work is re-indexed for
#       NHWC in custom CUDA kernels (no ATen for those ops).
#     * NCHW custom kernels are retained purely as an anti-regression fallback
#       selected by a one-shot CUDA-event autotune.
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

#define CDIV(a, b) (((a) + (b) - 1) / (b))

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ===========================================================================
//                              NCHW  (fallback path)
// ===========================================================================
template <bool VEC>
__global__ void gn_stats_partial_kernel(const float* __restrict__ y,
                                        float* __restrict__ partial,
                                        long long group_size,
                                        int splits) {
    const long long g = blockIdx.y;
    const int s = blockIdx.x;
    const float* __restrict__ base = y + g * group_size;

    float sum = 0.f, sq = 0.f;

    if (VEC) {
        const long long nvec = group_size >> 2;
        const long long start = ((long long)s * nvec) / splits;
        const long long end = ((long long)(s + 1) * nvec) / splits;
        const float4* __restrict__ b4 = reinterpret_cast<const float4*>(base);
        for (long long i = start + threadIdx.x; i < end; i += blockDim.x) {
            float4 v = b4[i];
            sum += v.x + v.y + v.z + v.w;
            sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        const long long start = ((long long)s * group_size) / splits;
        const long long end = ((long long)(s + 1) * group_size) / splits;
        for (long long i = start + threadIdx.x; i < end; i += blockDim.x) {
            float v = base[i];
            sum += v;
            sq += v * v;
        }
    }

    __shared__ float ws[32];
    __shared__ float wq[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    sum = warp_sum(sum);
    sq = warp_sum(sq);
    if (lane == 0) { ws[wid] = sum; wq[wid] = sq; }
    __syncthreads();

    if (wid == 0) {
        const int nw = blockDim.x >> 5;
        float a = (lane < nw) ? ws[lane] : 0.f;
        float b = (lane < nw) ? wq[lane] : 0.f;
        a = warp_sum(a);
        b = warp_sum(b);
        if (lane == 0) {
            partial[(g * splits + s) * 2 + 0] = a;
            partial[(g * splits + s) * 2 + 1] = b;
        }
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ ms,
                                   int splits, float inv_n, float eps) {
    const int g = blockIdx.x;
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < splits; i += 32) {
        s += partial[(long long)(g * splits + i) * 2 + 0];
        q += partial[(long long)(g * splits + i) * 2 + 1];
    }
    s = warp_sum(s);
    q = warp_sum(q);
    if (threadIdx.x == 0) {
        float m = s * inv_n;
        float var = q * inv_n - m * m;
        var = fmaxf(var, 0.f);
        ms[2 * g + 0] = m;
        ms[2 * g + 1] = rsqrtf(var + eps);
    }
}

template <bool VEC, bool RESID>
__global__ void gn_silu_apply_kernel(const float* __restrict__ y,
                                     const float* __restrict__ res,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     const float* __restrict__ ms,
                                     float* __restrict__ out,
                                     int C, int cpg, int G, long long HW) {
    const int nc = blockIdx.y;
    const int c = nc % C;
    const int n = nc / C;
    const int gid = n * G + (c / cpg);

    const float mean = ms[2 * gid + 0];
    const float rstd = ms[2 * gid + 1];
    const float gam = gamma[c];
    const float bet = beta[c];
    const float scale = gam * rstd;
    const float bias = bet - mean * scale;

    const long long off = (long long)nc * HW;

    if (VEC) {
        const long long nvec = HW >> 2;
        const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < nvec) {
            const float4* __restrict__ y4 = reinterpret_cast<const float4*>(y + off);
            float4* __restrict__ o4 = reinterpret_cast<float4*>(out + off);
            float4 v = y4[idx];
            float a0 = v.x * scale + bias;
            float a1 = v.y * scale + bias;
            float a2 = v.z * scale + bias;
            float a3 = v.w * scale + bias;
            a0 = a0 / (1.f + expf(-a0));
            a1 = a1 / (1.f + expf(-a1));
            a2 = a2 / (1.f + expf(-a2));
            a3 = a3 / (1.f + expf(-a3));
            if (RESID) {
                const float4* __restrict__ r4 = reinterpret_cast<const float4*>(res + off);
                float4 r = r4[idx];
                a0 += r.x; a1 += r.y; a2 += r.z; a3 += r.w;
            }
            float4 o;
            o.x = a0; o.y = a1; o.z = a2; o.w = a3;
            o4[idx] = o;
        }
    } else {
        const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < HW) {
            float a = y[off + idx] * scale + bias;
            a = a / (1.f + expf(-a));
            if (RESID) a += res[off + idx];
            out[off + idx] = a;
        }
    }
}

// ===========================================================================
//                              NHWC  (primary path)
// ===========================================================================
// Stage 1: per-(n, group, pixel-chunk) partial sums.
// grid = (chunks, N), block = C threads; thread t owns channel t of a pixel row
// (row of C contiguous floats -> perfectly coalesced).
template <bool SHFL>
__global__ void gn_stats_nhwc_kernel(const float* __restrict__ y,
                                     float* __restrict__ partial,
                                     int C, int cpg, int G,
                                     long long HW, int chunks) {
    extern __shared__ float smem[];
    const int n = blockIdx.y;
    const int chunk = blockIdx.x;
    const int c = threadIdx.x;

    const long long p0 = ((long long)chunk * HW) / chunks;
    const long long p1 = ((long long)(chunk + 1) * HW) / chunks;

    const float* __restrict__ p = y + ((long long)n * HW + p0) * (long long)C + c;

    float sum = 0.f, sq = 0.f;
#pragma unroll 4
    for (long long i = p0; i < p1; ++i) {
        float v = __ldg(p);
        p += C;
        sum += v;
        sq = fmaf(v, v, sq);
    }

    if (SHFL) {
        // cpg is a power of two <= 32 and groups are lane-aligned inside a warp
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < cpg) {
                sum += __shfl_down_sync(0xffffffffu, sum, off);
                sq += __shfl_down_sync(0xffffffffu, sq, off);
            }
        }
        if ((c % cpg) == 0) {
            const int g = c / cpg;
            const long long o = ((long long)(n * G + g) * chunks + chunk) * 2;
            partial[o + 0] = sum;
            partial[o + 1] = sq;
        }
    } else {
        smem[c] = sum;
        smem[C + c] = sq;
        __syncthreads();
        if (c < G) {
            float s = 0.f, q = 0.f;
            for (int t = 0; t < cpg; ++t) {
                s += smem[c * cpg + t];
                q += smem[C + c * cpg + t];
            }
            const long long o = ((long long)(n * G + c) * chunks + chunk) * 2;
            partial[o + 0] = s;
            partial[o + 1] = q;
        }
    }
}

// Stage 2: finalize mean/rstd AND precompute per-(n,c) scale / bias.
// grid = (N*G), block = 128
__global__ void gn_finalize_nhwc_kernel(const float* __restrict__ partial,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ scale,
                                        float* __restrict__ bias,
                                        int chunks, int C, int cpg, int G,
                                        float inv_n, float eps) {
    const int ng = blockIdx.x;
    const int n = ng / G;
    const int g = ng - n * G;

    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < chunks; i += blockDim.x) {
        const long long o = ((long long)ng * chunks + i) * 2;
        s += partial[o + 0];
        q += partial[o + 1];
    }

    __shared__ float ws[32];
    __shared__ float wq[32];
    __shared__ float s_mean, s_rstd;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    s = warp_sum(s);
    q = warp_sum(q);
    if (lane == 0) { ws[wid] = s; wq[wid] = q; }
    __syncthreads();

    if (wid == 0) {
        const int nw = blockDim.x >> 5;
        float a = (lane < nw) ? ws[lane] : 0.f;
        float b = (lane < nw) ? wq[lane] : 0.f;
        a = warp_sum(a);
        b = warp_sum(b);
        if (lane == 0) {
            float m = a * inv_n;
            float var = fmaxf(b * inv_n - m * m, 0.f);
            s_mean = m;
            s_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    const float m = s_mean;
    const float r = s_rstd;
    for (int t = threadIdx.x; t < cpg; t += blockDim.x) {
        const int c = g * cpg + t;
        const float sc = gamma[c] * r;
        scale[(long long)n * C + c] = sc;
        bias[(long long)n * C + c] = beta[c] - m * sc;
    }
}

// Stage 3 (vectorized): out = silu(scale*y + bias) [+ residual], NHWC float4.
// blockDim = (C/4, TY), grid = (ceil(HW/(TY*ITER)), N)
template <bool RESID, int ITER>
__global__ void gn_silu_apply_nhwc_vec_kernel(const float4* __restrict__ y,
                                              const float4* __restrict__ res,
                                              const float4* __restrict__ scale4,
                                              const float4* __restrict__ bias4,
                                              float4* __restrict__ out,
                                              int C4, long long HW) {
    const int n = blockIdx.y;
    const int cv = threadIdx.x;

    const float4 s = scale4[(long long)n * C4 + cv];
    const float4 b = bias4[(long long)n * C4 + cv];

    const long long base = (long long)n * HW * (long long)C4;
    const long long p0 = (long long)blockIdx.x * (long long)(blockDim.y * ITER) + threadIdx.y;

#pragma unroll
    for (int it = 0; it < ITER; ++it) {
        const long long p = p0 + (long long)it * blockDim.y;
        if (p < HW) {
            const long long idx = base + p * (long long)C4 + cv;
            float4 v = y[idx];
            float a0 = fmaf(v.x, s.x, b.x);
            float a1 = fmaf(v.y, s.y, b.y);
            float a2 = fmaf(v.z, s.z, b.z);
            float a3 = fmaf(v.w, s.w, b.w);
            a0 = a0 / (1.f + expf(-a0));
            a1 = a1 / (1.f + expf(-a1));
            a2 = a2 / (1.f + expf(-a2));
            a3 = a3 / (1.f + expf(-a3));
            if (RESID) {
                float4 r = res[idx];
                a0 += r.x; a1 += r.y; a2 += r.z; a3 += r.w;
            }
            float4 o;
            o.x = a0; o.y = a1; o.z = a2; o.w = a3;
            out[idx] = o;
        }
    }
}

// Stage 3 (scalar fallback, C % 4 != 0)
template <bool RESID>
__global__ void gn_silu_apply_nhwc_scalar_kernel(const float* __restrict__ y,
                                                 const float* __restrict__ res,
                                                 const float* __restrict__ scale,
                                                 const float* __restrict__ bias,
                                                 float* __restrict__ out,
                                                 int C, long long HW, long long total) {
    const long long HWC = HW * (long long)C;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x) {
        const long long n = idx / HWC;
        const int c = (int)(idx % (long long)C);
        const float sc = scale[n * C + c];
        const float bi = bias[n * C + c];
        float a = fmaf(y[idx], sc, bi);
        a = a / (1.f + expf(-a));
        if (RESID) a += res[idx];
        out[idx] = a;
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
static torch::Tensor gn_silu_impl(const torch::Tensor& y,
                                  const torch::Tensor& gamma,
                                  const torch::Tensor& beta,
                                  const torch::Tensor* res,
                                  int64_t num_groups,
                                  double eps) {
    TORCH_CHECK(y.is_cuda(), "input must be CUDA");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(y.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(y.is_contiguous(), "input must be contiguous");

    const int64_t N = y.size(0);
    const int64_t C = y.size(1);
    const int64_t HW = y.size(2) * y.size(3);
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");
    const int64_t cpg = C / num_groups;
    const int64_t group_size = cpg * HW;
    const int64_t NG = N * num_groups;

    if (res != nullptr) {
        TORCH_CHECK(res->is_cuda() && res->scalar_type() == at::kFloat, "residual dtype/device");
        TORCH_CHECK(res->is_contiguous(), "residual must be contiguous");
        TORCH_CHECK(res->numel() == y.numel(), "residual numel mismatch");
    }

    auto out = torch::empty_like(y);
    auto opts = y.options();
    auto ms = torch::empty({NG * 2}, opts);

    int splits = (int)std::min<int64_t>(32, std::max<int64_t>(1, 2048 / std::max<int64_t>(NG, 1)));
    while (splits > 1 && group_size / splits < 4096) splits >>= 1;
    if (splits < 1) splits = 1;

    auto partial = torch::empty({NG * (int64_t)splits * 2}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();

    const bool vec_stats = (group_size % 4 == 0);
    dim3 sgrid((unsigned)splits, (unsigned)NG);
    if (vec_stats) {
        gn_stats_partial_kernel<true><<<sgrid, 256, 0, stream>>>(
            y.data_ptr<float>(), partial.data_ptr<float>(), (long long)group_size, splits);
    } else {
        gn_stats_partial_kernel<false><<<sgrid, 256, 0, stream>>>(
            y.data_ptr<float>(), partial.data_ptr<float>(), (long long)group_size, splits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_kernel<<<(unsigned)NG, 32, 0, stream>>>(
        partial.data_ptr<float>(), ms.data_ptr<float>(), splits,
        (float)(1.0 / (double)group_size), (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const bool vec_apply = (HW % 4 == 0);
    const int block = 256;
    const int64_t units = vec_apply ? (HW / 4) : HW;
    dim3 agrid((unsigned)CDIV(units, (int64_t)block), (unsigned)(N * C));

    const float* rptr = (res != nullptr) ? res->data_ptr<float>() : y.data_ptr<float>();

    if (vec_apply) {
        if (res != nullptr) {
            gn_silu_apply_kernel<true, true><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
        } else {
            gn_silu_apply_kernel<true, false><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
        }
    } else {
        if (res != nullptr) {
            gn_silu_apply_kernel<false, true><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
        } else {
            gn_silu_apply_kernel<false, false><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

static torch::Tensor gn_silu_nhwc_impl(const torch::Tensor& y,
                                       const torch::Tensor& gamma,
                                       const torch::Tensor& beta,
                                       const torch::Tensor* res,
                                       int64_t num_groups,
                                       double eps) {
    TORCH_CHECK(y.is_cuda(), "input must be CUDA");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(y.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast),
                "input must be channels_last contiguous");

    const int64_t N = y.size(0);
    const int64_t C = y.size(1);
    const int64_t HW = y.size(2) * y.size(3);
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");
    TORCH_CHECK(C <= 1024, "NHWC path requires C <= 1024");
    const int64_t cpg = C / num_groups;
    const int64_t group_size = cpg * HW;
    const int64_t NG = N * num_groups;

    if (res != nullptr) {
        TORCH_CHECK(res->is_cuda() && res->scalar_type() == at::kFloat, "residual dtype/device");
        TORCH_CHECK(res->is_contiguous(at::MemoryFormat::ChannelsLast),
                    "residual must be channels_last contiguous");
        TORCH_CHECK(res->numel() == y.numel(), "residual numel mismatch");
    }

    auto out = torch::empty_like(y, y.options(), at::MemoryFormat::ChannelsLast);
    auto opts = y.options();

    // ---- stage 1: partial stats ------------------------------------------
    int chunks = (int)std::max<int64_t>(1,
                    std::min<int64_t>(std::max<int64_t>(HW / 64, 1),
                                      1024 / std::max<int64_t>(N, 1)));
    auto partial = torch::empty({NG * (int64_t)chunks * 2}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    const bool pow2_cpg = (cpg > 0) && ((cpg & (cpg - 1)) == 0) && (cpg <= 32);
    const bool use_shfl = pow2_cpg && (C % 32 == 0);

    dim3 sgrid((unsigned)chunks, (unsigned)N);
    if (use_shfl) {
        gn_stats_nhwc_kernel<true><<<sgrid, (unsigned)C, 0, stream>>>(
            y.data_ptr<float>(), partial.data_ptr<float>(),
            (int)C, (int)cpg, (int)num_groups, (long long)HW, chunks);
    } else {
        gn_stats_nhwc_kernel<false><<<sgrid, (unsigned)C, (size_t)(2 * C * sizeof(float)), stream>>>(
            y.data_ptr<float>(), partial.data_ptr<float>(),
            (int)C, (int)cpg, (int)num_groups, (long long)HW, chunks);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ---- stage 2: finalize + per-(n,c) scale/bias -------------------------
    auto scale = torch::empty({N * C}, opts);
    auto bias = torch::empty({N * C}, opts);

    gn_finalize_nhwc_kernel<<<(unsigned)NG, 128, 0, stream>>>(
        partial.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(),
        chunks, (int)C, (int)cpg, (int)num_groups,
        (float)(1.0 / (double)group_size), (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ---- stage 3: affine + SiLU (+ residual) ------------------------------
    const float* rptr = (res != nullptr) ? res->data_ptr<float>() : y.data_ptr<float>();

    if (C % 4 == 0 && (C / 4) <= 1024) {
        const int C4 = (int)(C / 4);
        int ty = 256 / C4;
        if (ty < 1) ty = 1;
        if ((long long)ty * C4 > 1024) ty = 1024 / C4;
        if (ty < 1) ty = 1;
        constexpr int ITER = 4;
        dim3 blk((unsigned)C4, (unsigned)ty);
        dim3 grd((unsigned)CDIV(HW, (int64_t)(ty * ITER)), (unsigned)N);

        const float4* y4 = reinterpret_cast<const float4*>(y.data_ptr<float>());
        const float4* r4 = reinterpret_cast<const float4*>(rptr);
        const float4* s4 = reinterpret_cast<const float4*>(scale.data_ptr<float>());
        const float4* b4 = reinterpret_cast<const float4*>(bias.data_ptr<float>());
        float4* o4 = reinterpret_cast<float4*>(out.data_ptr<float>());

        if (res != nullptr) {
            gn_silu_apply_nhwc_vec_kernel<true, ITER><<<grd, blk, 0, stream>>>(
                y4, r4, s4, b4, o4, C4, (long long)HW);
        } else {
            gn_silu_apply_nhwc_vec_kernel<false, ITER><<<grd, blk, 0, stream>>>(
                y4, r4, s4, b4, o4, C4, (long long)HW);
        }
    } else {
        const long long total = (long long)N * HW * C;
        const int block = 256;
        int gridx = (int)std::min<long long>(CDIV(total, (long long)block), 65535LL * 8);
        if (res != nullptr) {
            gn_silu_apply_nhwc_scalar_kernel<true><<<gridx, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, scale.data_ptr<float>(), bias.data_ptr<float>(),
                out.data_ptr<float>(), (int)C, (long long)HW, total);
        } else {
            gn_silu_apply_nhwc_scalar_kernel<false><<<gridx, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, scale.data_ptr<float>(), bias.data_ptr<float>(),
                out.data_ptr<float>(), (int)C, (long long)HW, total);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor gn_silu(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                      int64_t num_groups, double eps) {
    return gn_silu_impl(y, gamma, beta, nullptr, num_groups, eps);
}

torch::Tensor gn_silu_add(torch::Tensor y, torch::Tensor residual, torch::Tensor gamma,
                          torch::Tensor beta, int64_t num_groups, double eps) {
    return gn_silu_impl(y, gamma, beta, &residual, num_groups, eps);
}

torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                           int64_t num_groups, double eps) {
    return gn_silu_nhwc_impl(y, gamma, beta, nullptr, num_groups, eps);
}

torch::Tensor gn_silu_add_nhwc(torch::Tensor y, torch::Tensor residual, torch::Tensor gamma,
                               torch::Tensor beta, int64_t num_groups, double eps) {
    return gn_silu_nhwc_impl(y, gamma, beta, &residual, num_groups, eps);
}
"""

_CPP_SRC = r"""
torch::Tensor gn_silu(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                      int64_t num_groups, double eps);
torch::Tensor gn_silu_add(torch::Tensor y, torch::Tensor residual, torch::Tensor gamma,
                          torch::Tensor beta, int64_t num_groups, double eps);
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                           int64_t num_groups, double eps);
torch::Tensor gn_silu_add_nhwc(torch::Tensor y, torch::Tensor residual, torch::Tensor gamma,
                               torch::Tensor beta, int64_t num_groups, double eps);
"""

_ext = load_inline(
    name="vae_gn_silu_res_nhwc_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu", "gn_silu_add", "gn_silu_nhwc", "gn_silu_add_nhwc"],
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

# Plan item 2: enable cuDNN autotune (fixed shapes) and keep TF32 conv path.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

# Plan item 8: module-level cache of the winning layout per (shape, dtype).
_PATH_CACHE = {}


class ModelNew(nn.Module):
    """
    replaced : group_norm x2, silu x2, residual add -> custom CUDA kernels (NHWC & NCHW)
    kept     : F.conv2d (cuDNN vendor mainloop) — now fed channels_last tensors so
               it dispatches NHWC TF32 tensor-core implicit-GEMM kernels.
    """

    def __init__(self):
        super().__init__()
        self.ext = _ext
        self.num_groups = 32
        self.force_nchw_output = False

    # ------------------------------------------------------------------ NCHW
    def _fwd_nchw(self, x, w1_raw, g1_raw, b1_raw, w2_raw, g2_raw, b2_raw, eps_f):
        x_c = x if x.is_contiguous() else x.contiguous()
        w1 = w1_raw if w1_raw.is_contiguous() else w1_raw.contiguous()
        w2 = w2_raw if w2_raw.is_contiguous() else w2_raw.contiguous()
        g1 = g1_raw if g1_raw.is_contiguous() else g1_raw.contiguous()
        b1 = b1_raw if b1_raw.is_contiguous() else b1_raw.contiguous()
        g2 = g2_raw if g2_raw.is_contiguous() else g2_raw.contiguous()
        b2 = b2_raw if b2_raw.is_contiguous() else b2_raw.contiguous()

        out = F.conv2d(x_c, w1, bias=None, stride=1, padding=1)
        out = out if out.is_contiguous() else out.contiguous()
        out = self.ext.gn_silu(out, g1, b1, self.num_groups, eps_f)

        out = F.conv2d(out, w2, bias=None, stride=1, padding=1)
        out = out if out.is_contiguous() else out.contiguous()
        out = self.ext.gn_silu_add(out, x_c, g2, b2, self.num_groups, eps_f)
        return out

    # ------------------------------------------------------------------ NHWC
    def _fwd_nhwc(self, x, w1_raw, g1_raw, b1_raw, w2_raw, g2_raw, b2_raw, eps_f):
        cl = torch.channels_last
        x_c = x if x.is_contiguous(memory_format=cl) else x.contiguous(memory_format=cl)
        w1 = w1_raw if w1_raw.is_contiguous(memory_format=cl) else w1_raw.contiguous(memory_format=cl)
        w2 = w2_raw if w2_raw.is_contiguous(memory_format=cl) else w2_raw.contiguous(memory_format=cl)
        g1 = g1_raw if g1_raw.is_contiguous() else g1_raw.contiguous()
        b1 = b1_raw if b1_raw.is_contiguous() else b1_raw.contiguous()
        g2 = g2_raw if g2_raw.is_contiguous() else g2_raw.contiguous()
        b2 = b2_raw if b2_raw.is_contiguous() else b2_raw.contiguous()

        out = F.conv2d(x_c, w1, bias=None, stride=1, padding=1)
        if not out.is_contiguous(memory_format=cl):
            out = out.contiguous(memory_format=cl)
        out = self.ext.gn_silu_nhwc(out, g1, b1, self.num_groups, eps_f)

        out = F.conv2d(out, w2, bias=None, stride=1, padding=1)
        if not out.is_contiguous(memory_format=cl):
            out = out.contiguous(memory_format=cl)
        out = self.ext.gn_silu_add_nhwc(out, x_c, g2, b2, self.num_groups, eps_f)

        if self.force_nchw_output:
            out = out.contiguous()
        return out

    # -------------------------------------------------------------- autotune
    def _autotune(self, args):
        x = args[0]
        C = x.size(1)
        nhwc_ok = (x.dim() == 4 and x.dtype == torch.float32 and C <= 1024
                   and C % self.num_groups == 0)
        if not nhwc_ok:
            return "nchw"
        try:
            with torch.no_grad():
                for _ in range(3):
                    self._fwd_nchw(*args)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(5):
                    self._fwd_nchw(*args)
                e.record()
                torch.cuda.synchronize()
                t_nchw = s.elapsed_time(e)

                for _ in range(3):
                    self._fwd_nhwc(*args)
                torch.cuda.synchronize()
                s2 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                s2.record()
                for _ in range(5):
                    self._fwd_nhwc(*args)
                e2.record()
                torch.cuda.synchronize()
                t_nhwc = s2.elapsed_time(e2)
        except Exception:
            return "nchw"
        return "nhwc" if t_nhwc <= 0.95 * t_nchw else "nchw"

    # ------------------------------------------------------------------ main
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        args = (x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps_f)

        key = (tuple(x.shape), x.dtype, tuple(conv1_weight.shape), self.num_groups)
        mode = _PATH_CACHE.get(key)
        if mode is None:
            mode = self._autotune(args)
            _PATH_CACHE[key] = mode

        if mode == "nhwc":
            return self._fwd_nhwc(*args)
        return self._fwd_nchw(*args)
