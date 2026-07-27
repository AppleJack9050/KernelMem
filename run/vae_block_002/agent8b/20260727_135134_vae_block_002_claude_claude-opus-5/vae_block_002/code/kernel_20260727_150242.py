import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

torch.backends.cudnn.benchmark = True

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
//                      FP32 NCHW PATH (fallback, identical to base)
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
    const float scale = gamma[c] * rstd;
    const float bias = beta[c] - mean * scale;

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
        if (res != nullptr)
            gn_silu_apply_kernel<true, true><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
        else
            gn_silu_apply_kernel<true, false><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
    } else {
        if (res != nullptr)
            gn_silu_apply_kernel<false, true><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
        else
            gn_silu_apply_kernel<false, false><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
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

// ===========================================================================
//                  FP16 NHWC PATH (primary optimisation)
// ===========================================================================

// --- Plan item 3: stats over NHWC half, channel-major thread mapping -------
// grid = (splits, N), block = TPB; thread owns a channel PAIR (half2 load).
__global__ void gn_stats_nhwc_half_kernel(const __half* __restrict__ in,
                                          float* __restrict__ partial,
                                          int C, long long HW, int splits) {
    const int n = blockIdx.y;
    const int s = blockIdx.x;
    const int C2 = C >> 1;

    const long long start = ((long long)s * HW) / splits;
    const long long end = ((long long)(s + 1) * HW) / splits;

    const __half2* __restrict__ base =
        reinterpret_cast<const __half2*>(in) + (long long)n * HW * C2;

    for (int c2 = threadIdx.x; c2 < C2; c2 += blockDim.x) {
        float s0 = 0.f, q0 = 0.f, s1 = 0.f, q1 = 0.f;
        long long p = start;
        for (; p + 3 < end; p += 4) {
            __half2 v0 = base[(p + 0) * C2 + c2];
            __half2 v1 = base[(p + 1) * C2 + c2];
            __half2 v2 = base[(p + 2) * C2 + c2];
            __half2 v3 = base[(p + 3) * C2 + c2];
            float a0 = __low2float(v0), b0 = __high2float(v0);
            float a1 = __low2float(v1), b1 = __high2float(v1);
            float a2 = __low2float(v2), b2 = __high2float(v2);
            float a3 = __low2float(v3), b3 = __high2float(v3);
            s0 += a0 + a1 + a2 + a3;
            q0 += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
            s1 += b0 + b1 + b2 + b3;
            q1 += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
        }
        for (; p < end; ++p) {
            __half2 v = base[p * C2 + c2];
            float a = __low2float(v), b = __high2float(v);
            s0 += a; q0 += a * a;
            s1 += b; q1 += b * b;
        }
        const long long o = ((long long)(n * splits + s) * C + 2 * c2);
        partial[o * 2 + 0] = s0;
        partial[o * 2 + 1] = q0;
        partial[(o + 1) * 2 + 0] = s1;
        partial[(o + 1) * 2 + 1] = q1;
    }
}

// --- Plan item 4: finalize (sum per-channel partials inside each group) ----
__global__ void gn_finalize_nhwc_kernel(const float* __restrict__ partial,
                                        float* __restrict__ ms,
                                        int C, int cpg, int G, int splits,
                                        float inv_n, float eps) {
    const int ng = blockIdx.x;
    const int n = ng / G;
    const int g = ng - n * G;

    float s = 0.f, q = 0.f;
    const int total = cpg * splits;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int si = i / cpg;
        const int cc = i - si * cpg;
        const long long o = ((long long)(n * splits + si) * C + g * cpg + cc);
        s += partial[o * 2 + 0];
        q += partial[o * 2 + 1];
    }
    s = warp_sum(s);
    q = warp_sum(q);

    __shared__ float ss[8], sq[8];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) { ss[wid] = s; sq[wid] = q; }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = blockDim.x >> 5;
        float S = 0.f, Q = 0.f;
        for (int i = 0; i < nw; ++i) { S += ss[i]; Q += sq[i]; }
        float m = S * inv_n;
        float var = Q * inv_n - m * m;
        var = fmaxf(var, 0.f);
        ms[2 * ng + 0] = m;
        ms[2 * ng + 1] = rsqrtf(var + eps);
    }
}

// --- Plan item 5: GN + SiLU, NHWC half in -> NHWC half out (feeds conv2) ---
// grid = (px_blocks, N), block = 256, dynamic smem = 2*C floats.
__global__ void gn_silu_apply_nhwc_half_kernel(const __half* __restrict__ in,
                                               const float* __restrict__ gamma,
                                               const float* __restrict__ beta,
                                               const float* __restrict__ ms,
                                               __half* __restrict__ out,
                                               int C, int cpg, int G, long long HW) {
    extern __shared__ float smem[];
    const int C8 = C >> 3;
    float* sscale = smem;            // layout: [j][c8]  (conflict-free reads)
    float* sbias = smem + C;

    const int n = blockIdx.y;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const int gid = n * G + (c / cpg);
        const float m = ms[2 * gid + 0];
        const float r = ms[2 * gid + 1];
        const float sc = gamma[c] * r;
        const int j = c & 7;
        const int c8 = c >> 3;
        sscale[j * C8 + c8] = sc;
        sbias[j * C8 + c8] = beta[c] - m * sc;
    }
    __syncthreads();

    const int c8 = threadIdx.x % C8;
    const int pl = threadIdx.x / C8;
    const int prows = blockDim.x / C8;

    const uint4* __restrict__ in4 =
        reinterpret_cast<const uint4*>(in) + (long long)n * HW * C8;
    uint4* __restrict__ out4 =
        reinterpret_cast<uint4*>(out) + (long long)n * HW * C8;

    union PK { uint4 u; __half2 h[4]; };

    for (long long p = (long long)blockIdx.x * prows + pl; p < HW;
         p += (long long)gridDim.x * prows) {
        const long long idx = p * C8 + c8;
        PK pk;
        pk.u = in4[idx];
        PK ok;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a0 = __low2float(pk.h[j]);
            float a1 = __high2float(pk.h[j]);
            const float s0 = sscale[(2 * j) * C8 + c8];
            const float b0 = sbias[(2 * j) * C8 + c8];
            const float s1 = sscale[(2 * j + 1) * C8 + c8];
            const float b1 = sbias[(2 * j + 1) * C8 + c8];
            a0 = a0 * s0 + b0;
            a1 = a1 * s1 + b1;
            a0 = a0 / (1.f + expf(-a0));
            a1 = a1 / (1.f + expf(-a1));
            ok.h[j] = __floats2half2_rn(a0, a1);
        }
        out4[idx] = ok.u;
    }
}

// --- Plan item 6: GN + SiLU + fp32 residual, NHWC half in -> NCHW fp32 out -
// 32(chan) x 32(pixel) shared tile so global stores/residual reads coalesce.
__global__ void gn_silu_res_nhwc2nchw_kernel(const __half* __restrict__ in,
                                             const float* __restrict__ res,
                                             const float* __restrict__ gamma,
                                             const float* __restrict__ beta,
                                             const float* __restrict__ ms,
                                             float* __restrict__ out,
                                             int C, int cpg, int G, long long HW) {
    extern __shared__ float smem[];
    float* sscale = smem;
    float* sbias = smem + C;
    float* tile = smem + 2 * C;   // 32 * 33 floats (padded, conflict-free)

    const int n = blockIdx.y;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const int gid = n * G + (c / cpg);
        const float m = ms[2 * gid + 0];
        const float r = ms[2 * gid + 1];
        const float sc = gamma[c] * r;
        sscale[c] = sc;
        sbias[c] = beta[c] - m * sc;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;   // 0..7

    const __half* __restrict__ inb = in + (long long)n * HW * C;

    for (long long pbase = (long long)blockIdx.x * 32; pbase < HW;
         pbase += (long long)gridDim.x * 32) {
        for (int cb = 0; cb < C; cb += 32) {
            __syncthreads();
            // ---- load: lane == channel (64B coalesced per warp) ----
            const int c = cb + lane;
            const float sc = sscale[c];
            const float bi = sbias[c];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int plc = wid + 8 * j;
                const long long p = pbase + plc;
                float a = 0.f;
                if (p < HW) {
                    a = __half2float(inb[p * C + c]) * sc + bi;
                    a = a / (1.f + expf(-a));
                }
                tile[lane * 33 + plc] = a;
            }
            __syncthreads();
            // ---- store: lane == pixel (coalesced fp32 NCHW) ----
            const long long p = pbase + lane;
            if (p < HW) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int cl = wid * 4 + j;
                    const long long o = ((long long)n * C + cb + cl) * HW + p;
                    out[o] = tile[cl * 33 + lane] + res[o];
                }
            }
        }
    }
}

// --------------------------- host helpers ----------------------------------
static torch::Tensor compute_ms_nhwc(const torch::Tensor& y, int64_t num_groups,
                                     double eps, cudaStream_t stream) {
    const int64_t N = y.size(0);
    const int64_t C = y.size(1);
    const int64_t HW = y.size(2) * y.size(3);
    const int64_t cpg = C / num_groups;

    int64_t splits = std::max<int64_t>(1, std::min<int64_t>(64, 2048 / std::max<int64_t>(N, 1)));
    while (splits > 1 && HW / splits < 64) splits >>= 1;

    auto optsf = y.options().dtype(at::kFloat);
    auto partial = torch::empty({N * splits * C * 2}, optsf);
    auto ms = torch::empty({N * num_groups * 2}, optsf);

    int tpb = (int)std::min<int64_t>(C / 2, 256);
    tpb = ((tpb + 31) / 32) * 32;
    if (tpb < 32) tpb = 32;

    dim3 sgrid((unsigned)splits, (unsigned)N);
    gn_stats_nhwc_half_kernel<<<sgrid, tpb, 0, stream>>>(
        reinterpret_cast<const __half*>(y.data_ptr<at::Half>()),
        partial.data_ptr<float>(), (int)C, (long long)HW, (int)splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_nhwc_kernel<<<(unsigned)(N * num_groups), 128, 0, stream>>>(
        partial.data_ptr<float>(), ms.data_ptr<float>(), (int)C, (int)cpg,
        (int)num_groups, (int)splits,
        (float)(1.0 / (double)(cpg * HW)), (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return ms;
}

// Plan item 9: guards
static void check_nhwc(const torch::Tensor& y, int64_t num_groups) {
    TORCH_CHECK(y.is_cuda() && y.dim() == 4, "nhwc input must be 4D CUDA");
    TORCH_CHECK(y.scalar_type() == at::kHalf, "nhwc input must be half");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast),
                "nhwc input must be channels_last contiguous");
    const int64_t C = y.size(1);
    TORCH_CHECK(C % num_groups == 0, "C % num_groups");
    TORCH_CHECK(C % 32 == 0 && C % 8 == 0, "C must be a multiple of 32");
    TORCH_CHECK((C / 8) <= 256 && (256 % (C / 8)) == 0, "unsupported C for half8 mapping");
    TORCH_CHECK(C <= 1024, "C too large for shared scale/bias");
}

torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                           int64_t num_groups, double eps) {
    check_nhwc(y, num_groups);
    const int64_t N = y.size(0);
    const int64_t C = y.size(1);
    const int64_t HW = y.size(2) * y.size(3);
    const int64_t cpg = C / num_groups;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto ms = compute_ms_nhwc(y, num_groups, eps, stream);
    auto out = torch::empty_like(y);

    const int C8 = (int)(C / 8);
    const int block = 256;
    const int prows = block / C8;
    int64_t gx = CDIV(HW, (int64_t)prows);
    gx = std::min<int64_t>(gx, 8192);
    dim3 grid((unsigned)std::max<int64_t>(gx, 1), (unsigned)N);
    const size_t shm = (size_t)(2 * C) * sizeof(float);

    gn_silu_apply_nhwc_half_kernel<<<grid, block, shm, stream>>>(
        reinterpret_cast<const __half*>(y.data_ptr<at::Half>()),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), ms.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        (int)C, (int)cpg, (int)num_groups, (long long)HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_res_nhwc(torch::Tensor y, torch::Tensor res, torch::Tensor gamma,
                               torch::Tensor beta, int64_t num_groups, double eps) {
    check_nhwc(y, num_groups);
    TORCH_CHECK(res.is_cuda() && res.scalar_type() == at::kFloat && res.is_contiguous(),
                "residual must be contiguous fp32 NCHW");
    TORCH_CHECK(res.numel() == y.numel(), "residual numel mismatch");

    const int64_t N = y.size(0);
    const int64_t C = y.size(1);
    const int64_t H = y.size(2);
    const int64_t W = y.size(3);
    const int64_t HW = H * W;
    const int64_t cpg = C / num_groups;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto ms = compute_ms_nhwc(y, num_groups, eps, stream);
    auto out = torch::empty({N, C, H, W}, y.options().dtype(at::kFloat));

    const int block = 256;
    int64_t gx = std::min<int64_t>(CDIV(HW, (int64_t)32), 4096);
    dim3 grid((unsigned)std::max<int64_t>(gx, 1), (unsigned)N);
    const size_t shm = (size_t)(2 * C + 32 * 33) * sizeof(float);

    gn_silu_res_nhwc2nchw_kernel<<<grid, block, shm, stream>>>(
        reinterpret_cast<const __half*>(y.data_ptr<at::Half>()),
        res.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        ms.data_ptr<float>(), out.data_ptr<float>(),
        (int)C, (int)cpg, (int)num_groups, (long long)HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor gn_silu(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                      int64_t num_groups, double eps);
torch::Tensor gn_silu_add(torch::Tensor y, torch::Tensor residual, torch::Tensor gamma,
                          torch::Tensor beta, int64_t num_groups, double eps);
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                           int64_t num_groups, double eps);
torch::Tensor gn_silu_res_nhwc(torch::Tensor y, torch::Tensor res, torch::Tensor gamma,
                               torch::Tensor beta, int64_t num_groups, double eps);
"""

_ext = load_inline(
    name="vae_gn_silu_res_fp16_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu", "gn_silu_add", "gn_silu_nhwc", "gn_silu_res_nhwc"],
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
    """
    conv(FP16 NHWC tensor-core) -> fused GN+SiLU (custom NHWC half kernels)
    conv(FP16 NHWC tensor-core) -> fused GN+SiLU+residual (NHWC half -> NCHW fp32)
    """

    def __init__(self):
        super().__init__()
        self.ext = _ext
        self.num_groups = 32
        self._w_cache = {}
        self._checked = False
        self._use_fp16 = True

    # ---- cached half channels_last weights (plan items 1 & 8) --------------
    def _half_w(self, w):
        key = (w.data_ptr(), tuple(w.shape))
        wh = self._w_cache.get(key)
        if wh is None:
            wh = w.to(torch.half, memory_format=torch.channels_last)
            self._w_cache[key] = wh
        return wh

    def _nhwc_supported(self, x):
        if x.dim() != 4 or x.dtype != torch.float32:
            return False
        C = x.size(1)
