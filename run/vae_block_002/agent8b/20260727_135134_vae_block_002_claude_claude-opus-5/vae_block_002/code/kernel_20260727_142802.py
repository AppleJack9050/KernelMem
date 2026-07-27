# ==========================================================================
# ModelNew — SOL problem 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# Optimisation applied: cudnn_winograd_algo_autotune
#   - cudnn.benchmark / TF32 enabled (exhaustive algo search + cache)
#   - custom cuDNN-direct conv3x3 with cudnnFindConvolutionForwardAlgorithmEx,
#     shape-keyed static algo cache, cached workspace, Winograd accuracy guard,
#     and a one-time perf comparison vs at::conv2d (falls back if not faster).
#   - GN/SiLU/residual fused kernels are left EXACTLY as in the base kernel
#     (they already run at 87-89% of DRAM SOL).
# ==========================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------------
# Plan item 1: exhaustive cuDNN algo search + TF32 (same math mode as base)
# --------------------------------------------------------------------------
def _enable_cudnn_autotune():
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass


_enable_cudnn_autotune()

# --------------------------------------------------------------------------
# (1) GN / SiLU / residual CUDA source  -- UNCHANGED from base kernel
# --------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Stage 1: partial sums / sums-of-squares per (n, group) chunk.
// grid = (splits, num_groups_total), block = 256
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Stage 2: finalize mean / rstd.  grid = (num_groups_total), block = 32
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Stage 3: y_hat = gamma*(y-mean)*rstd + beta ; out = silu(y_hat) (+ residual)
// grid = (chunks_over_HW, N*C), block = 256
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Host launcher
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
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "affine params must be CUDA");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
                "affine params must be float32");

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

    // choose split factor: aim for ~2048 blocks, >= 4096 elements per block
    int splits = (int)std::min<int64_t>(32, std::max<int64_t>(1, 2048 / std::max<int64_t>(NG, 1)));
    while (splits > 1 && group_size / splits < 4096) splits >>= 1;
    if (splits < 1) splits = 1;

    auto partial = torch::empty({NG * (int64_t)splits * 2}, opts);

    auto stream = at::cuda::getDefaultCUDAStream();

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

    const float* rptr = (res != nullptr) ? res->data_ptr<float>() : nullptr;

    if (vec_apply) {
        if (res != nullptr) {
            gn_silu_apply_kernel<true, true><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), rptr, gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
        } else {
            gn_silu_apply_kernel<true, false><<<agrid, block, 0, stream>>>(
                y.data_ptr<float>(), y.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
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
                y.data_ptr<float>(), y.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                ms.data_ptr<float>(), out.data_ptr<float>(),
                (int)C, (int)cpg, (int)num_groups, (long long)HW);
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
"""

# --------------------------------------------------------------------------
# (2) cuDNN algorithm-autotuned 3x3 convolution (plan items 2,4,5,7,8,9)
# --------------------------------------------------------------------------
_CONV_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cuda/CUDAContext.h>
#include <chrono>
#include <mutex>
#include <vector>
#include <unordered_map>

// ---------------------------------------------------------------------------
// descriptor bundle (NCHW, FLOAT, 3x3 / pad1 / stride1 / cross-correlation)
// ---------------------------------------------------------------------------
struct Descs {
    cudnnTensorDescriptor_t xd = nullptr, yd = nullptr;
    cudnnFilterDescriptor_t wd = nullptr;
    cudnnConvolutionDescriptor_t cd = nullptr;

    bool build(int N, int C, int H, int W, int K, int R, int S, int pad) {
        if (cudnnCreateTensorDescriptor(&xd) != CUDNN_STATUS_SUCCESS) return false;
        if (cudnnCreateTensorDescriptor(&yd) != CUDNN_STATUS_SUCCESS) return false;
        if (cudnnCreateFilterDescriptor(&wd) != CUDNN_STATUS_SUCCESS) return false;
        if (cudnnCreateConvolutionDescriptor(&cd) != CUDNN_STATUS_SUCCESS) return false;
        if (cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       N, C, H, W) != CUDNN_STATUS_SUCCESS) return false;
        if (cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       N, K, H, W) != CUDNN_STATUS_SUCCESS) return false;
        if (cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                       K, C, R, S) != CUDNN_STATUS_SUCCESS) return false;
        if (cudnnSetConvolution2dDescriptor(cd, pad, pad, 1, 1, 1, 1,
                                            CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) return false;
        // same math mode PyTorch uses when allow_tf32 == true
        cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        return true;
    }
    ~Descs() {
        if (cd) cudnnDestroyConvolutionDescriptor(cd);
        if (wd) cudnnDestroyFilterDescriptor(wd);
        if (yd) cudnnDestroyTensorDescriptor(yd);
        if (xd) cudnnDestroyTensorDescriptor(xd);
    }
};

struct ConvPlan {
    bool valid = false;                     // true -> use cuDNN-direct path
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t ws_bytes = 0;
};

static std::unordered_map<unsigned long long, ConvPlan> g_plans;   // shape-keyed cache
static std::mutex g_mtx;
static at::Tensor g_ws;                                           // cached workspace

static inline unsigned long long make_key(int N, int C, int H, int W, int K) {
    return ((unsigned long long)N * 1000003ull) ^ ((unsigned long long)C << 12) ^
           ((unsigned long long)H << 24) ^ ((unsigned long long)W << 38) ^
           ((unsigned long long)K << 50);
}

static inline bool is_winograd(cudnnConvolutionFwdAlgo_t a) {
    return a == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD ||
           a == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
}

static void* ensure_ws(size_t bytes, const at::TensorOptions& opts) {
    if (bytes == 0) return nullptr;
    if (!g_ws.defined() || (size_t)g_ws.numel() < bytes) {
        g_ws = torch::empty({(int64_t)bytes}, opts.dtype(at::kByte));
    }
    return g_ws.data_ptr();
}

// raw cuDNN forward into a freshly allocated output on the input's device
static at::Tensor run_cudnn_conv(const at::Tensor& xc, const at::Tensor& wc,
                                 Descs& d, cudnnConvolutionFwdAlgo_t algo,
                                 size_t ws_bytes, int N, int K, int H, int W) {
    auto handle = at::native::getCudnnHandle();
    cudnnSetStream(handle, at::cuda::getCurrentCUDAStream());
    auto y = torch::empty({N, K, H, W}, xc.options());
    void* ws = ensure_ws(ws_bytes, xc.options());
    const float alpha = 1.f, beta = 0.f;
    cudnnStatus_t st = cudnnConvolutionForward(handle, &alpha, d.xd, xc.data_ptr(),
                                               d.wd, wc.data_ptr(), d.cd, algo,
                                               ws, ws_bytes, &beta, d.yd, y.data_ptr());
    if (st != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnnConvolutionForward failed: ") +
                                 cudnnGetErrorString(st));
    }
    return y;
}

template <typename F>
static double time_ms(F&& f, int iters) {
    auto stream = at::cuda::getCurrentCUDAStream();
    f(); f();
    cudaStreamSynchronize(stream);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) f();
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)iters;
}

// ---------------------------------------------------------------------------
// One-time exhaustive algorithm search + accuracy guard + perf guard.
// ---------------------------------------------------------------------------
static ConvPlan search_plan(const at::Tensor& xc, const at::Tensor& wc,
                            int N, int C, int H, int W, int K, int R, int S, int pad) {
    ConvPlan plan;  // valid == false -> fall back to at::conv2d forever
    try {
        auto handle = at::native::getCudnnHandle();
        cudnnSetStream(handle, at::cuda::getCurrentCUDAStream());

        Descs d;
        if (!d.build(N, C, H, W, K, R, S, pad)) return plan;

        auto y = torch::empty({N, K, H, W}, xc.options());
        const size_t ws_limit = (size_t)256 << 20;   // 256 MB search workspace
        at::Tensor search_ws = torch::empty({(int64_t)ws_limit},
                                            xc.options().dtype(at::kByte));

        const int req = 8;
        int returned = 0;
        std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(req);
        cudnnStatus_t st = cudnnFindConvolutionForwardAlgorithmEx(
            handle, d.xd, xc.data_ptr(), d.wd, wc.data_ptr(), d.cd, d.yd, y.data_ptr(),
            req, &returned, perf.data(), search_ws.data_ptr(), ws_limit);
        if (st != CUDNN_STATUS_SUCCESS || returned <= 0) return plan;

        // reference for the accuracy guard (same TF32 math mode as the baseline)
        at::Tensor ref = at::conv2d(xc, wc, at::Tensor(), {1, 1}, {pad, pad});

        bool skip_winograd = false;
        bool found = false;
        cudnnConvolutionFwdAlgo_t chosen = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        size_t chosen_ws = 0;

        for (int pass = 0; pass < 2 && !found; ++pass) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
                if (perf[i].memory > ws_limit) continue;
                if (skip_winograd && is_winograd(perf[i].algo)) continue;
                try {
                    at::Tensor test = run_cudnn_conv(xc, wc, d, perf[i].algo,
                                                     perf[i].memory, N, K, H, W);
                    double diff = (test - ref).abs().max().item<double>();
                    if (diff <= 1.0e-3) {
                        chosen = perf[i].algo;
                        chosen_ws = perf[i].memory;
                        found = true;
                        break;
                    } else if (is_winograd(perf[i].algo)) {
                        skip_winograd = true;   // blacklist winograd, restart scan
                        break;
                    }
                } catch (...) {
                    continue;
                }
            }
            if (!skip_winograd) break;   // no restart needed
        }
        if (!found) return plan;

        // perf guard: only keep the cuDNN-direct path if clearly faster (>=5%)
        double t_cudnn = time_ms([&]() {
            (void)run_cudnn_conv(xc, wc, d, chosen, chosen_ws, N, K, H, W);
        }, 5);
        double t_aten = time_ms([&]() {
            (void)at::conv2d(xc, wc, at::Tensor(), {1, 1}, {pad, pad});
        }, 5);

        if (t_cudnn <= 0.95 * t_aten) {
            plan.valid = true;
            plan.algo = chosen;
            plan.ws_bytes = chosen_ws;
            ensure_ws(chosen_ws, xc.options());   // allocate steady-state workspace once
        }
    } catch (...) {
        plan.valid = false;
    }
    return plan;
}

static ConvPlan get_plan(const at::Tensor& xc, const at::Tensor& wc,
                         int N, int C, int H, int W, int K, int R, int S, int pad) {
    unsigned long long key = make_key(N, C, H, W, K);
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_plans.find(key);
    if (it != g_plans.end()) return it->second;
    ConvPlan p = search_plan(xc, wc, N, C, H, W, K, R, S, pad);
    g_plans[key] = p;
    return p;
}

// ---------------------------------------------------------------------------
// Public entry point: 3x3, stride 1, pad 1, NCHW fp32 convolution.
// ---------------------------------------------------------------------------
torch::Tensor conv3x3(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "conv3x3: inputs must be CUDA");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "conv3x3: inputs must be 4D");

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto wc = w.is_contiguous() ? w : w.contiguous();

    const int N = (int)xc.size(0), C = (int)xc.size(1);
    const int H = (int)xc.size(2), W = (int)xc.size(3);
    const int K = (int)wc.size(0), R = (int)wc.size(2), S = (int)wc.size(3);
    const int pad = 1;

    if (xc.scalar_type() != at::kFloat || wc.scalar_type() != at::kFloat ||
        R != 3 || S != 3 || (int)wc.size(1) != C) {
        return at::conv2d(xc, wc, at::Tensor(), {1, 1}, {pad, pad});
    }

    ConvPlan plan = get_plan(xc, wc, N, C, H, W, K, R, S, pad);
    if (!plan.valid) {
        // keeps cudnn.benchmark=True exhaustive selection (plan step 1 only)
        return at::conv2d(xc, wc, at::Tensor(), {1, 1}, {pad, pad});
    }

    try {
        Descs d;
        if (!d.build(N, C, H, W, K, R, S, pad)) {
            return at::conv2d(xc, wc, at::Tensor(), {1, 1}, {pad, pad});
        }
        return run_cudnn_conv(xc, wc, d, plan.algo, plan.ws_bytes, N, K, H, W);
    } catch (...) {
        return at::conv2d(xc, wc, at::Tensor(), {1, 1}, {pad, pad});
    }
}
"""

# --------------------------------------------------------------------------
# (3) prototypes
# --------------------------------------------------------------------------
_CPP_SRC = r"""
torch::Tensor gn_silu(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                      int64_t num_groups, double eps);
torch::Tensor gn_silu_add(torch::Tensor y, torch::Tensor residual, torch::Tensor gamma,
                          torch::Tensor beta, int64_t num_groups, double eps);
"""

_CONV_CPP_SRC = r"""
torch::Tensor conv3x3(torch::Tensor x, torch::Tensor w);
"""

# --------------------------------------------------------------------------
# (4) builds: GN kernel group + cuDNN conv group
# --------------------------------------------------------------------------
_ext = load_inline(
    name="vae_gn_silu_res_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu", "gn_silu_add"],
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


def _cudnn_paths():
    incs, ldflags = [], []
    try:
        import nvidia.cudnn as _c

        base = os.path.dirname(_c.__file__)
        inc = os.path.join(base, "include")
        lib = os.path.join(base, "lib")
        if os.path.isdir(inc):
            incs.append(inc)
        if os.path.isdir(lib):
            ldflags += ["-L" + lib, "-Wl,-rpath," + lib]
    except Exception:
        pass
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    for p in (os.path.join(cuda_home, "include"),):
        if os.path.isdir(p):
            incs.append(p)
    for p in (os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")):
        if os.path.isdir(p):
            ldflags.append("-L" + p)
    return incs, ldflags


try:
    _inc, _ld = _cudnn_paths()
    _conv_ext = load_inline(
        name="vae_cudnn_conv3x3_autotune_ext",
        cpp_sources=_CONV_CPP_SRC,
        cuda_sources=_CONV_CUDA_SRC,
        functions=["conv3x3"],
        verbose=False,
        extra_include_paths=_inc,
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        extra_ldflags=_ld + ["-lcudnn"],
    )
except Exception:
    _conv_ext = None


class ModelNew(nn.Module):
    """
    replaced : group_norm x2, silu x2, residual add -> custom CUDA kernels (unchanged)
    tuned    : 3x3 convolutions -> cuDNN with exhaustive, shape-cached algorithm
               search (Winograd-nonfused preferred), with accuracy + perf guards.
    """

    def __init__(self):
        super().__init__()
        _enable_cudnn_autotune()          # plan item 1
        self.ext = _ext
        self.ext_conv = _conv_ext
        self.num_groups = 32

    def _conv3x3(self, x, w):
        if self.ext_conv is not None:
            try:
                return self.ext_conv.conv3x3(x, w)
            except Exception:
                self.ext_conv = None
        return F.conv2d(x, w, bias=None, stride=1, padding=1)

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        x_c = x if x.is_contiguous() else x.contiguous()
        w1 = conv1_weight if conv1_weight.is_contiguous() else conv1_weight.contiguous()
        w2 = conv2_weight if conv2_weight.is_contiguous() else conv2_weight.contiguous()
        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        # --- path 1: autotuned cuDNN conv -> fused GroupNorm + SiLU
        out = self._conv3x3(x_c, w1)
        out = out if out.is_contiguous() else out.contiguous()
        out = self.ext.gn_silu(out, g1, b1, self.num_groups, eps_f)

        # --- path 2: autotuned cuDNN conv -> fused GroupNorm + SiLU + residual add
        out = self._conv3x3(out, w2)
        out = out if out.is_contiguous() else out.contiguous()
        out = self.ext.gn_silu_add(out, x_c, g2, b2, self.num_groups, eps_f)

        return out
