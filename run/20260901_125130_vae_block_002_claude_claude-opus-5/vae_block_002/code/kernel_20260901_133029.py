import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#define TP 32

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + __expf(-v));
}

// ---------------------------------------------------------------- statistics
// x: NHWC contiguous. one block handles (n,g) pixel-segment s.
// Specialized for C=256, G=32, CPG=8 (cpg4 = 2) -> pure 32-bit index math.
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ partial,   // [2, BG, S]
                                int HW, int S, int BG) {
    constexpr int C = 256;
    constexpr int G = 32;
    constexpr int CPG = 8;

    const int s  = blockIdx.x;
    const int bg = blockIdx.y;
    const int n  = bg / G;
    const int g  = bg - n * G;

    const long long base = (long long)n * HW * (long long)C + (long long)g * CPG;
    const int p_start = (int)(((long long)HW * s) / S);
    const int p_end   = (int)(((long long)HW * (s + 1)) / S);

    float sum = 0.f, sq = 0.f;
    const int cnt = (p_end - p_start) << 1;   // (p_end-p_start) * cpg4, cpg4=2
    #pragma unroll 4
    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
        int p = p_start + (i >> 1);
        int j = i & 1;
        const float4 v = *reinterpret_cast<const float4*>(x + base + (long long)p * C + (j << 2));
        sum += v.x + v.y + v.z + v.w;
        sq  += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // block reduce
    __shared__ float ws[2][32];
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(mask, sum, off);
        sq  += __shfl_down_sync(mask, sq,  off);
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { ws[0][warp] = sum; ws[1][warp] = sq; }
    __syncthreads();
    int nw = blockDim.x >> 5;
    if (warp == 0) {
        sum = (lane < nw) ? ws[0][lane] : 0.f;
        sq  = (lane < nw) ? ws[1][lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(mask, sum, off);
            sq  += __shfl_down_sync(mask, sq,  off);
        }
        if (lane == 0) {
            partial[(long long)bg * S + s]                 = sum;
            partial[(long long)BG * S + (long long)bg * S + s] = sq;
        }
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int S, int BG, float inv_count, float eps) {
    const int bg = blockIdx.x;
    float sum = 0.f, sq = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        sum += partial[(long long)bg * S + i];
        sq  += partial[(long long)BG * S + (long long)bg * S + i];
    }
    __shared__ float ws[2][32];
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(mask, sum, off);
        sq  += __shfl_down_sync(mask, sq,  off);
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { ws[0][warp] = sum; ws[1][warp] = sq; }
    __syncthreads();
    int nw = blockDim.x >> 5;
    if (warp == 0) {
        sum = (lane < nw) ? ws[0][lane] : 0.f;
        sq  = (lane < nw) ? ws[1][lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(mask, sum, off);
            sq  += __shfl_down_sync(mask, sq,  off);
        }
        if (lane == 0) {
            float m = sum * inv_count;
            float v = sq * inv_count - m * m;
            v = v < 0.f ? 0.f : v;
            mean[bg] = m;
            rstd[bg] = rsqrtf(v + eps);
        }
    }
}

// ------------------------------------------- GN(affine) + SiLU, NHWC -> NHWC
// Specialized: C4 = 64, CPG = 8  ->  idx % C4 = idx & 63, c0 / CPG = c0 >> 3
__global__ void gn_silu_nhwc_kernel(const float* __restrict__ x,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ rstd,
                                    float* __restrict__ y,
                                    int HWC4, int G) {
    constexpr int C4 = 64;
    const int n = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HWC4) return;
    const int c0 = (idx & (C4 - 1)) << 2;
    const int g  = c0 >> 3;
    const int bg = n * G + g;
    const float mu = mean[bg], rs = rstd[bg];

    const long long off = (long long)n * HWC4 + idx;
    const float4 v  = *reinterpret_cast<const float4*>(x + (off << 2));
    const float4 gm = *reinterpret_cast<const float4*>(gamma + c0);
    const float4 bt = *reinterpret_cast<const float4*>(beta + c0);
    float4 o;
    o.x = silu_f((v.x - mu) * rs * gm.x + bt.x);
    o.y = silu_f((v.y - mu) * rs * gm.y + bt.y);
    o.z = silu_f((v.z - mu) * rs * gm.z + bt.z);
    o.w = silu_f((v.w - mu) * rs * gm.w + bt.w);
    *reinterpret_cast<float4*>(y + (off << 2)) = o;
}

// ---------------- GN(affine) + SiLU + residual add, NHWC in -> NCHW out
// Specialized: CPG = 8, TPX = 128, static shared sm[128*9], 256 threads/block.
__global__ void gn_silu_res_nchw_kernel(const float* __restrict__ x,
                                        const float* __restrict__ res,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ rstd,
                                        float* __restrict__ out,
                                        int HW, int G) {
    constexpr int C = 256;
    constexpr int CPG = 8;
    constexpr int TPX = 128;
    __shared__ float sm[TPX * 9];

    const int bg = blockIdx.y;
    const int n  = bg / G;
    const int g  = bg - n * G;
    const int p0 = blockIdx.x * TPX;
    const float mu = mean[bg], rs = rstd[bg];

    const long long base = (long long)n * HW * (long long)C + (long long)g * CPG;
    const int t = threadIdx.x;          // 0..255
    const int p = t >> 1;               // 0..127
    const int j = t & 1;                // 0 or 1
    const int pg = p0 + p;

    if (pg < HW) {
        const float4 v  = *reinterpret_cast<const float4*>(x + base + (long long)pg * C + (j << 2));
        const float4 gm = *reinterpret_cast<const float4*>(gamma + (g << 3) + (j << 2));
        const float4 bt = *reinterpret_cast<const float4*>(beta  + (g << 3) + (j << 2));
        float r0 = silu_f((v.x - mu) * rs * gm.x + bt.x);
        float r1 = silu_f((v.y - mu) * rs * gm.y + bt.y);
        float r2 = silu_f((v.z - mu) * rs * gm.z + bt.z);
        float r3 = silu_f((v.w - mu) * rs * gm.w + bt.w);
        int base_c = j << 2;
        sm[p * 9 + base_c + 0] = r0;
        sm[p * 9 + base_c + 1] = r1;
        sm[p * 9 + base_c + 2] = r2;
        sm[p * 9 + base_c + 3] = r3;
    }
    __syncthreads();

    const long long obase = (long long)n * C * (long long)HW + (long long)g * CPG * HW;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int lin = (k << 8) + t;      // k*256 + t, ranges 0..1023
        int c   = lin >> 7;          // 0..7
        int pl  = lin & 127;         // 0..127
        int pgv = p0 + pl;
        if (pgv < HW) {
            long long oi = obase + (long long)c * HW + pgv;
            out[oi] = sm[pl * 9 + c] + res[oi];
        }
    }
}

// ------------------------------------------------- NCHW -> NHWC layout transform
// x: NCHW contiguous (float32), C = 256. out: NHWC-linear buffer.
__global__ void nchw_to_nhwc_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int HW) {
    __shared__ float t[32][33];
    const int hw0 = blockIdx.x * 32;
    const int c0  = blockIdx.y * 32;
    const int n   = blockIdx.z;
    const int tx  = threadIdx.x;   // 0..31 (pixel offset)
    const int ty  = threadIdx.y;   // 0..7  (channel sub-offset)

    const long long xbase = (long long)n * 256 * (long long)HW;
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int cc = ty + it * 8;
        int hh = tx;
        int hw = hw0 + hh;
        if (hw < HW) {
            t[cc][hh] = x[xbase + (long long)(c0 + cc) * HW + hw];
        }
    }
    __syncthreads();

    const long long obase = (long long)n * HW * 256;
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int hh = ty + it * 8;
        int cc = tx;
        int hw = hw0 + hh;
        if (hw < HW) {
            out[obase + (long long)hw * 256 + c0 + cc] = t[cc][hh];
        }
    }
}

// ------------------------------------------------- weight OIHW -> O(HWI) (channels_last)
// src: (256,256,3,3) contiguous. dst: (256,256,3,3) channels_last-linear (O,kH,kW,I).
__global__ void weight_to_nhwc_kernel(const float* __restrict__ w1,
                                      const float* __restrict__ w2,
                                      float* __restrict__ d1,
                                      float* __restrict__ d2) {
    __shared__ float sm[2304];
    const int k = blockIdx.x;               // output channel 0..255
    const int which = blockIdx.y;            // 0 -> w1, 1 -> w2
    const float* src = which ? w2 : w1;
    float* dst = which ? d2 : d1;
    const int tid = threadIdx.x;             // 0..255

    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        sm[tid + 256 * it] = src[(long long)k * 2304 + tid + 256 * it];
    }
    __syncthreads();

    #pragma unroll
    for (int rs = 0; rs < 9; ++rs) {
        dst[(long long)k * 2304 + rs * 256 + tid] = sm[tid * 9 + rs];
    }
}

// ------------------------------------------------------------------- launchers
static void compute_stats(const at::Tensor& x, int B, int HW, int C, int G,
                          float eps, at::Tensor& mean, at::Tensor& rstd) {
    const int BG = B * G;
    int maxS = (HW + 255) / 256;
    if (maxS < 1) maxS = 1;
    int S = (2048 + BG - 1) / BG;
    if (S < 1) S = 1;
    if (S > maxS) S = maxS;
    auto opts = x.options();
    auto partial = at::empty({2, BG, S}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(S, BG);
    gn_stats_kernel<<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), partial.data_ptr<float>(), HW, S, BG);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int CPG = C / G;
    float inv_count = 1.0f / (float)((long long)HW * CPG);
    gn_finalize_kernel<<<BG, 256, 0, stream>>>(
        partial.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        S, BG, inv_count, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                           int64_t B, int64_t HW, int64_t C, int64_t G, double eps) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    auto opts = x.options();
    auto mean = at::empty({B * G}, opts);
    auto rstd = at::empty({B * G}, opts);
    compute_stats(x, (int)B, (int)HW, (int)C, (int)G, (float)eps, mean, rstd);

    auto y = at::empty_like(x);
    const int C4 = (int)C / 4;
    const int HWC4 = (int)HW * C4;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((HWC4 + 255) / 256, (unsigned)B);
    gn_silu_nhwc_kernel<<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), y.data_ptr<float>(),
        HWC4, (int)G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor gn_silu_res_nchw(torch::Tensor x, torch::Tensor res,
                               torch::Tensor gamma, torch::Tensor beta,
                               int64_t B, int64_t HW, int64_t C, int64_t G, double eps) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    auto opts = x.options();
    auto mean = at::empty({B * G}, opts);
    auto rstd = at::empty({B * G}, opts);
    compute_stats(x, (int)B, (int)HW, (int)C, (int)G, (float)eps, mean, rstd);

    auto out = at::empty({B * C * HW}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(((int)HW + 128 - 1) / 128, (unsigned)(B * G));
    gn_silu_res_nchw_kernel<<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), res.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        out.data_ptr<float>(), (int)HW, (int)G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nchw_to_nhwc(torch::Tensor x, torch::Tensor out, int64_t B, int64_t HW) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(32, 8);
    dim3 grid(((int)HW + 31) / 32, 8, (unsigned)B);
    nchw_to_nhwc_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), (int)HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void weight_to_nhwc(torch::Tensor w1, torch::Tensor w2, torch::Tensor d1, torch::Tensor d2) {
    TORCH_CHECK(w1.scalar_type() == at::kFloat, "float32 only");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(256, 2);
    weight_to_nhwc_kernel<<<grid, 256, 0, stream>>>(
        w1.data_ptr<float>(), w2.data_ptr<float>(), d1.data_ptr<float>(), d2.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP = r"""
torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                           int64_t B, int64_t HW, int64_t C, int64_t G, double eps);
torch::Tensor gn_silu_res_nchw(torch::Tensor x, torch::Tensor res,
                               torch::Tensor gamma, torch::Tensor beta,
                               int64_t B, int64_t HW, int64_t C, int64_t G, double eps);
torch::Tensor nchw_to_nhwc(torch::Tensor x, torch::Tensor out, int64_t B, int64_t HW);
void weight_to_nhwc(torch::Tensor w1, torch::Tensor w2, torch::Tensor d1, torch::Tensor d2);
"""

_ext = load_inline(
    name="vae_res_block_fused_v2",
    cpp_sources=_CPP,
    cuda_sources=_SRC,
    functions=["gn_silu_nhwc", "gn_silu_res_nchw", "nchw_to_nhwc", "weight_to_nhwc"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-use_fast_math",
        "-gencode=arch=compute_90,code=sm_90",
    ],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        G = 32
        B, C, H, W = x.shape
        HW = H * W

        if x.dtype != torch.float32 or x.dim() != 4 or C != 256 or C % (4 * G) != 0:
            # generic fallback (keeps semantics for unsupported configs)
            out = F.conv2d(x, conv1_weight, None, 1, 1)
            out = F.group_norm(out, G, norm1_weight, norm1_bias, eps)
            out = F.silu(out)
            out = F.conv2d(out, conv2_weight, None, 1, 1)
            out = F.group_norm(out, G, norm2_weight, norm2_bias, eps)
            out = F.silu(out)
            return out + x

        xc = x if x.is_contiguous() else x.contiguous()
        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()
        w1c = conv1_weight if conv1_weight.is_contiguous() else conv1_weight.contiguous()
        w2c = conv2_weight if conv2_weight.is_contiguous() else conv2_weight.contiguous()

        # custom NCHW -> NHWC layout transform for the input (replaces aten copy)
        xl = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype,
                          memory_format=torch.channels_last)
        _ext.nchw_to_nhwc(xc, xl, B, HW)

        # custom weight layout transform: OIHW -> channels_last (O,kH,kW,I), both weights fused
        w1l = torch.empty((256, 256, 3, 3), device=x.device, dtype=x.dtype,
                           memory_format=torch.channels_last)
        w2l = torch.empty((256, 256, 3, 3), device=x.device, dtype=x.dtype,
                           memory_format=torch.channels_last)
        _ext.weight_to_nhwc(w1c, w2c, w1l, w2l)

        y = F.conv2d(xl, w1l, None, 1, 1)                       # cuDNN NHWC (vendor)
        y = _ext.gn_silu_nhwc(y, g1, b1, B, HW, C, G, float(eps))
        y = F.conv2d(y, w2l, None, 1, 1)                        # cuDNN NHWC (vendor)
        out = _ext.gn_silu_res_nchw(y, xc, g2, b2, B, HW, C, G, float(eps))
        return out.view(B, C, H, W)
