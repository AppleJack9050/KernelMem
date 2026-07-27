# ==========================================================================
# ModelNew — fused VAE residual block (Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +x)
#
# SEED GRANULARITY: (C) fuse many ops into one/few kernels.
#   1) Chosen granularity: (C) — the whole normalization/activation/residual
#      chain is collapsed into a small number of hand-written CUDA kernels,
#      driven by a single C++ entry point (`fused_resblock`) built with
#      load_inline.
#   2) Ops replaced by custom CUDA kernels:
#        - F.group_norm  (both occurrences)  -> gstats_partial + gstats_finalize
#                                               (two-stage group reduction)
#        - F.silu        (both occurrences)  -> fused into gn_silu_kernel
#        - out + residual                    -> fused into gn_silu_kernel<HAS_RES=true>
#   3) Fusion map:
#        kernel `gstats_partial_kernel` + `gstats_finalize_kernel`
#             -> per-(batch,group) mean / rstd (split-K style, ~2048 blocks so the
#                170 SMs stay saturated even with only B*G=256 groups).
#        kernel `gn_silu_kernel<BLOCK,false>`
#             -> (x-mean)*rstd*gamma+beta  AND  SiLU, single read + single write.
#        kernel `gn_silu_kernel<BLOCK,true>`
#             -> same, PLUS the residual add, still one pass.
#      This removes ~600 MB of round-trip DRAM traffic vs. the eager reference
#      (group_norm write, silu read/write, add read/write are all eliminated).
#   4) Remaining in PyTorch (called from *inside* the extension, i.e. cuDNN via
#      at::conv2d): the two 3x3 convolutions.
#        - Reason: cuDNN's implicit-GEMM/Winograd fp32 kernels for C=256,K=256
#          are already near hardware limits, and calling the *identical* path
#          (same NCHW layout, same global TF32 setting) guarantees bit-level
#          agreement with the reference for the dominant arithmetic, so the
#          fused epilogues are the only source of (1e-7-level) deviation.
#      Everything else (stats, affine, SiLU, residual) is custom CUDA.
#
# Parameter parity: this problem is stateless (get_init_inputs() == []); all
# weights arrive as forward() arguments and are used verbatim — no new
# parameters are created, no tensor is mutated in place, output is out-of-place.
# ==========================================================================

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// block reduction of two floats
// ---------------------------------------------------------------------------
template <int BLOCK>
__device__ __forceinline__ void block_reduce2(float& a, float& b) {
    __shared__ float sa[BLOCK / 32];
    __shared__ float sb[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, off);
        b += __shfl_down_sync(0xffffffffu, b, off);
    }
    if (lane == 0) { sa[wid] = a; sb[wid] = b; }
    __syncthreads();
    if (threadIdx.x < (BLOCK / 32)) { a = sa[threadIdx.x]; b = sb[threadIdx.x]; }
    else                            { a = 0.f;             b = 0.f;             }
    if (wid == 0) {
#pragma unroll
        for (int off = BLOCK / 64; off > 0; off >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, off);
            b += __shfl_down_sync(0xffffffffu, b, off);
        }
    }
}

// ---------------------------------------------------------------------------
// stage 1 : partial sums / sums-of-squares per (batch,group) chunk
//   grid = (S, B*G)   ; each (batch,group) owns L = (C/G)*H*W contiguous floats
// ---------------------------------------------------------------------------
template <int BLOCK>
__global__ void gstats_partial_kernel(const float* __restrict__ x,
                                      float* __restrict__ psum,
                                      float* __restrict__ psq,
                                      int L, int S, int vec4) {
    const int s  = blockIdx.x;
    const int bg = blockIdx.y;
    const float* p = x + (long long)bg * (long long)L;

    float sm = 0.f, sq = 0.f;

    if (vec4) {
        const int L4    = L >> 2;
        const int chunk = (L4 + S - 1) / S;
        const int i0    = s * chunk;
        const int i1    = min(i0 + chunk, L4);
        const float4* p4 = reinterpret_cast<const float4*>(p);
        for (int i = i0 + threadIdx.x; i < i1; i += BLOCK) {
            float4 v = p4[i];
            sm += v.x + v.y + v.z + v.w;
            sq  = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, sq))));
        }
    } else {
        const int chunk = (L + S - 1) / S;
        const int i0    = s * chunk;
        const int i1    = min(i0 + chunk, L);
        for (int i = i0 + threadIdx.x; i < i1; i += BLOCK) {
            float v = p[i];
            sm += v;
            sq  = fmaf(v, v, sq);
        }
    }

    block_reduce2<BLOCK>(sm, sq);
    if (threadIdx.x == 0) {
        psum[(long long)bg * S + s] = sm;
        psq [(long long)bg * S + s] = sq;
    }
}

// ---------------------------------------------------------------------------
// stage 2 : finalize mean / rstd
// ---------------------------------------------------------------------------
__global__ void gstats_finalize_kernel(const float* __restrict__ psum,
                                       const float* __restrict__ psq,
                                       float* __restrict__ mean,
                                       float* __restrict__ rstd,
                                       int L, int S, float eps, int nG) {
    int bg = blockIdx.x * blockDim.x + threadIdx.x;
    if (bg >= nG) return;
    float s = 0.f, q = 0.f;
    const float* ps = psum + (long long)bg * S;
    const float* pq = psq  + (long long)bg * S;
    for (int i = 0; i < S; ++i) { s += ps[i]; q += pq[i]; }
    const float invL = 1.f / (float)L;
    float m   = s * invL;
    float var = q * invL - m * m;
    if (!(var > 0.f)) var = 0.f;
    mean[bg] = m;
    rstd[bg] = rsqrtf(var + eps);
}

// ---------------------------------------------------------------------------
// fused: groupnorm affine + SiLU (+ optional residual add)
//   grid = B*C blocks, one output channel-plane (HW elements) per block
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_f(float v) {
    return v * (1.f / (1.f + expf(-v)));
}

template <int BLOCK, bool HAS_RES>
__global__ void gn_silu_kernel(const float* __restrict__ x,
                               const float* __restrict__ mean,
                               const float* __restrict__ rstd,
                               const float* __restrict__ gamma,
                               const float* __restrict__ beta,
                               const float* __restrict__ res,
                               float* __restrict__ out,
                               int HW, int C, int cpg, int G, int vec4) {
    const int blk = blockIdx.x;
    const int n   = blk / C;
    const int c   = blk - n * C;
    const int g   = c / cpg;

    const float m = mean[n * G + g];
    const float r = rstd[n * G + g];
    const float a = r * gamma[c];
    const float b = beta[c] - m * a;

    const long long base = (long long)blk * (long long)HW;

    if (vec4) {
        const int n4 = HW >> 2;
        const float4* xp = reinterpret_cast<const float4*>(x + base);
        float4*       op = reinterpret_cast<float4*>(out + base);
        const float4* rp = HAS_RES ? reinterpret_cast<const float4*>(res + base)
                                   : (const float4*)nullptr;
        for (int i = threadIdx.x; i < n4; i += BLOCK) {
            float4 v = xp[i];
            float4 o;
            float t;
            t = fmaf(v.x, a, b); o.x = silu_f(t);
            t = fmaf(v.y, a, b); o.y = silu_f(t);
            t = fmaf(v.z, a, b); o.z = silu_f(t);
            t = fmaf(v.w, a, b); o.w = silu_f(t);
            if (HAS_RES) {
                float4 rv = rp[i];
                o.x += rv.x; o.y += rv.y; o.z += rv.z; o.w += rv.w;
            }
            op[i] = o;
        }
    } else {
        const float* xp = x + base;
        float*       op = out + base;
        const float* rp = HAS_RES ? (res + base) : (const float*)nullptr;
        for (int i = threadIdx.x; i < HW; i += BLOCK) {
            float t = fmaf(xp[i], a, b);
            float o = silu_f(t);
            if (HAS_RES) o += rp[i];
            op[i] = o;
        }
    }
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
static void group_stats(const at::Tensor& y, at::Tensor& mean, at::Tensor& rstd,
                        int64_t B, int64_t C, int64_t HW, int64_t G, double eps) {
    const int64_t cpg = C / G;
    const int64_t L   = cpg * HW;
    const int64_t nG  = B * G;

    int S = 1;
    while (S < 32 && (int64_t)nG * S < 1360 && (L / (int64_t)(S * 2)) >= 8192) S *= 2;

    const int vec4 = (L % 4 == 0) ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();  // == default stream in the harness

    auto part = at::empty({2, nG * S}, y.options());
    float* psum = part.data_ptr<float>();
    float* psq  = psum + nG * S;

    dim3 grid((unsigned)S, (unsigned)nG);
    gstats_partial_kernel<256><<<grid, 256, 0, stream>>>(
        y.data_ptr<float>(), psum, psq, (int)L, S, vec4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int fb = 128;
    const int fg = (int)((nG + fb - 1) / fb);
    gstats_finalize_kernel<<<fg, fb, 0, stream>>>(
        psum, psq, mean.data_ptr<float>(), rstd.data_ptr<float>(),
        (int)L, S, (float)eps, (int)nG);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void gn_silu_launch(const at::Tensor& y, const at::Tensor& mean, const at::Tensor& rstd,
                           const at::Tensor& gamma, const at::Tensor& beta,
                           const at::Tensor* res, at::Tensor& out,
                           int64_t B, int64_t C, int64_t HW, int64_t G) {
    const int64_t cpg = C / G;
    const int vec4 = (HW % 4 == 0) ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = (int)(B * C);

    if (res != nullptr) {
        gn_silu_kernel<256, true><<<grid, 256, 0, stream>>>(
            y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(), res->data_ptr<float>(),
            out.data_ptr<float>(), (int)HW, (int)C, (int)cpg, (int)G, vec4);
    } else {
        gn_silu_kernel<256, false><<<grid, 256, 0, stream>>>(
            y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(), (const float*)nullptr,
            out.data_ptr<float>(), (int)HW, (int)C, (int)cpg, (int)G, vec4);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight,
                             torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight,
                             torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias,
                             double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x must be 4-D (B,C,H,W)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only float32 is supported");
    TORCH_CHECK(conv1_weight.scalar_type() == at::kFloat &&
                conv2_weight.scalar_type() == at::kFloat, "weights must be float32");

    auto xc  = x.is_contiguous()            ? x            : x.contiguous();
    auto w1c = conv1_weight.is_contiguous() ? conv1_weight : conv1_weight.contiguous();
    auto w2c = conv2_weight.is_contiguous() ? conv2_weight : conv2_weight.contiguous();
    auto g1  = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
    auto b1  = norm1_bias.is_contiguous()   ? norm1_bias   : norm1_bias.contiguous();
    auto g2  = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
    auto b2  = norm2_bias.is_contiguous()   ? norm2_bias   : norm2_bias.contiguous();

    const int64_t B  = xc.size(0);
    const int64_t C  = xc.size(1);
    const int64_t H  = xc.size(2);
    const int64_t W  = xc.size(3);
    const int64_t G  = 32;
    const int64_t HW = H * W;
    TORCH_CHECK(C % G == 0, "channels must be divisible by 32 groups");

    std::vector<int64_t> one{1, 1};

    // ---- conv1 (cuDNN, identical call/layout/precision-mode as the reference)
    auto y = at::conv2d(xc, w1c, at::Tensor(), one, one, one, 1);
    TORCH_CHECK(y.is_contiguous(), "conv1 output must be contiguous NCHW");

    auto opts  = xc.options();
    auto mean1 = at::empty({B * G}, opts);
    auto rstd1 = at::empty({B * G}, opts);
    group_stats(y, mean1, rstd1, B, C, HW, G, eps);

    auto z = at::empty_like(y);
    gn_silu_launch(y, mean1, rstd1, g1, b1, nullptr, z, B, C, HW, G);

    // ---- conv2
    auto y2 = at::conv2d(z, w2c, at::Tensor(), one, one, one, 1);
    TORCH_CHECK(y2.is_contiguous(), "conv2 output must be contiguous NCHW");

    auto mean2 = at::empty({B * G}, opts);
    auto rstd2 = at::empty({B * G}, opts);
    group_stats(y2, mean2, rstd2, B, C, HW, G, eps);

    auto out = at::empty_like(y2);
    gn_silu_launch(y2, mean2, rstd2, g2, b2, &xc, out, B, C, HW, G);

    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight,
                             torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight,
                             torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias,
                             double eps);
'''

_ext = load_inline(
    name="fused_vae_resblock_gn_silu",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["fused_resblock"],
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


class ModelNew(nn.Module):
    """See module header comment for the granularity/fusion plan (level C)."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self,
                x,
                conv1_weight,
                norm1_weight,
                norm1_bias,
                conv2_weight,
                norm2_weight,
                norm2_bias,
                eps):
        if torch.is_tensor(eps):
            eps = float(eps.reshape(-1)[0].item())
        return self.ext.fused_resblock(x,
                                       conv1_weight,
                                       norm1_weight,
                                       norm1_bias,
                                       conv2_weight,
                                       norm2_weight,
                                       norm2_bias,
                                       float(eps))
