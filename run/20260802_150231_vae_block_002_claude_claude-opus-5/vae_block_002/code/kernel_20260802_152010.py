# =============================================================================
# ModelNew — fused Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> +residual
#
# HEADER (required):
# 1) CHOSEN GRANULARITY: (C) "fuse many ops into one/few kernels".
#    Everything except the two vendor convolutions is folded into custom CUDA
#    kernels; the convolutions stay as vendor (cuDNN) calls because the profile
#    shows them running at ~90% of the TF32 roofline (1615us of 2180us).
#
# 2) OPS REPLACED BY CUSTOM CUDA:
#      - the NCHW->NHWC layout transform that cuDNN was doing internally
#        (cudnn::nchwToNhwcKernel) for the network input          -> nchw2nhwc_kernel
#      - group_norm #1 moments (RowwiseMoments + ComputeFusedParams) -> moments_kernel + finalize_kernel
#      - group_norm #1 affine apply + SiLU                        -> gnsilu_nhwc_kernel
#      - group_norm #2 moments                                    -> moments_kernel + finalize_kernel
#      - group_norm #2 affine apply + SiLU + residual add + the
#        NHWC->NCHW output transform                              -> gnsilu_res_nchw_kernel
#
# 3) FUSIONS:
#      - kernel gnsilu_nhwc_kernel        = (x-mean)*rstd*w+b  AND  SiLU, single pass, NHWC in/out.
#      - kernel gnsilu_res_nchw_kernel    = (x-mean)*rstd*w+b  AND  SiLU  AND  "+residual"
#                                           AND the NHWC->NCHW re-layout, all in one pass
#                                           (the reference paid 3 separate elementwise kernels
#                                            plus a cuDNN layout kernel for this).
#      - moments are computed with a deterministic 2-stage tree reduction
#        (partial buffer, no atomics) so results are run-to-run identical.
#      - the whole pipeline is kept in NHWC so cuDNN never inserts layout kernels
#        between the two convolutions.
#
# 4) LEFT IN PYTORCH:
#      - F.conv2d x2      : vendor tensor-core implicit-GEMM is already near roofline;
#                           re-implementing it cannot win, so we only feed it NHWC.
#      - weight .contiguous(channels_last) : 2.4MB each, negligible, keeps cuDNN on the
#                           NHWC tensor-core path without a per-call transform kernel of ours.
#      - a full PyTorch fallback path is kept for shapes outside C=256/G=32 (never hit by
#        this benchmark's workloads) purely for safety.
#
# Precision: everything stays float32 (TF32 tensor cores in conv exactly as the
# reference, which runs with cudnn.allow_tf32=True). Reductions accumulate in
# float32 per-thread and finish in double.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

torch.backends.cudnn.benchmark = True

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define GRP 32
#define CPG 8

__device__ __forceinline__ void blockReduce2(float &a, float &b) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, off);
        b += __shfl_down_sync(0xffffffffu, b, off);
    }
    __shared__ float sa[32];
    __shared__ float sb[32];
    int w = threadIdx.x >> 5;
    int l = threadIdx.x & 31;
    if (l == 0) { sa[w] = a; sb[w] = b; }
    __syncthreads();
    int nw = (int)((blockDim.x + 31) >> 5);
    if (threadIdx.x < 32) {
        a = (threadIdx.x < (unsigned)nw) ? sa[threadIdx.x] : 0.f;
        b = (threadIdx.x < (unsigned)nw) ? sb[threadIdx.x] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, off);
            b += __shfl_down_sync(0xffffffffu, b, off);
        }
    }
}

__device__ __forceinline__ float gnsilu_one(float x, float m, float r, float g, float b) {
    float v = (x - m) * r * g + b;
    return v / (1.f + expf(-v));
}

// ---------------------------------------------------------------- NCHW -> NHWC
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int S, int C) {
    int n  = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= S) return;
    const float* p = in + (size_t)n * C * (size_t)S + hw;
    float* o = out + ((size_t)n * S + hw) * C;
    for (int c = 0; c < C; c += 8) {
        float4 a, b;
        a.x = p[(size_t)(c + 0) * S];
        a.y = p[(size_t)(c + 1) * S];
        a.z = p[(size_t)(c + 2) * S];
        a.w = p[(size_t)(c + 3) * S];
        b.x = p[(size_t)(c + 4) * S];
        b.y = p[(size_t)(c + 5) * S];
        b.z = p[(size_t)(c + 6) * S];
        b.w = p[(size_t)(c + 7) * S];
        reinterpret_cast<float4*>(o + c)[0] = a;
        reinterpret_cast<float4*>(o + c)[1] = b;
    }
}

// ---------------------------------------------------------- GroupNorm moments
__global__ void moments_kernel(const float* __restrict__ in,
                               float* __restrict__ part,
                               int S, int C, int P) {
    int idx = blockIdx.y;          // n*GRP + g
    int n   = idx >> 5;
    int g   = idx & 31;
    int chunk = (S + P - 1) / P;
    int start = blockIdx.x * chunk;
    int end   = min(S, start + chunk);
    const float* base = in + (size_t)n * S * (size_t)C + g * CPG;
    float s = 0.f, s2 = 0.f;
    for (int hw = start + (int)threadIdx.x; hw < end; hw += (int)blockDim.x) {
        const float4* p = reinterpret_cast<const float4*>(base + (size_t)hw * C);
        float4 a = p[0];
        float4 b = p[1];
        s  += (a.x + a.y + a.z + a.w) + (b.x + b.y + b.z + b.w);
        s2 += (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w)
            + (b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w);
    }
    blockReduce2(s, s2);
    if (threadIdx.x == 0) {
        size_t o = ((size_t)idx * P + blockIdx.x) * 2;
        part[o]     = s;
        part[o + 1] = s2;
    }
}

__global__ void finalize_kernel(const float* __restrict__ part,
                                float* __restrict__ mean,
                                float* __restrict__ rstd,
                                int P, long long count, float eps) {
    int idx = blockIdx.x;
    const float* p = part + (size_t)idx * P * 2;
    float s = 0.f, s2 = 0.f;
    for (int i = (int)threadIdx.x; i < P; i += (int)blockDim.x) {
        s  += p[2 * i];
        s2 += p[2 * i + 1];
    }
    blockReduce2(s, s2);
    if (threadIdx.x == 0) {
        double cnt = (double)count;
        double m = (double)s / cnt;
        double v = (double)s2 / cnt - m * m;
        if (v < 0.0) v = 0.0;
        mean[idx] = (float)m;
        rstd[idx] = (float)(1.0 / sqrt(v + (double)eps));
    }
}

// ------------------------------------------------ GroupNorm affine + SiLU (NHWC)
__global__ void gnsilu_nhwc_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ mean,
                                   const float* __restrict__ rstd,
                                   int S, int C) {
    int n = blockIdx.y;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tot = (long long)S * GRP;
    if (t >= tot) return;
    int hw = (int)(t >> 5);
    int g  = (int)(t & 31);

    const float* p = in  + ((size_t)n * S + hw) * C + g * CPG;
    float*       o = out + ((size_t)n * S + hw) * C + g * CPG;

    float m = mean[n * GRP + g];
    float r = rstd[n * GRP + g];

    float4 a  = reinterpret_cast<const float4*>(p)[0];
    float4 b  = reinterpret_cast<const float4*>(p)[1];
    float4 ga = reinterpret_cast<const float4*>(gamma + g * CPG)[0];
    float4 gb = reinterpret_cast<const float4*>(gamma + g * CPG)[1];
    float4 ba = reinterpret_cast<const float4*>(beta  + g * CPG)[0];
    float4 bb = reinterpret_cast<const float4*>(beta  + g * CPG)[1];

    float4 ra, rb;
    ra.x = gnsilu_one(a.x, m, r, ga.x, ba.x);
    ra.y = gnsilu_one(a.y, m, r, ga.y, ba.y);
    ra.z = gnsilu_one(a.z, m, r, ga.z, ba.z);
    ra.w = gnsilu_one(a.w, m, r, ga.w, ba.w);
    rb.x = gnsilu_one(b.x, m, r, gb.x, bb.x);
    rb.y = gnsilu_one(b.y, m, r, gb.y, bb.y);
    rb.z = gnsilu_one(b.z, m, r, gb.z, bb.z);
    rb.w = gnsilu_one(b.w, m, r, gb.w, bb.w);

    reinterpret_cast<float4*>(o)[0] = ra;
    reinterpret_cast<float4*>(o)[1] = rb;
}

// ------------- GroupNorm affine + SiLU + residual + NHWC->NCHW re-layout ------
__global__ void gnsilu_res_nchw_kernel(const float* __restrict__ in,
                                       const float* __restrict__ res,
                                       float* __restrict__ out,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ rstd,
                                       int S, int C) {
    int n  = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= S) return;

    const float* p  = in  + ((size_t)n * S + hw) * C;
    const float* r0 = res + (size_t)n * C * (size_t)S + hw;
    float*       o  = out + (size_t)n * C * (size_t)S + hw;
    const float* mn = mean + n * GRP;
    const float* rs = rstd + n * GRP;

    for (int g = 0; g < GRP; ++g) {
        float m = mn[g];
        float r = rs[g];
        float4 a  = reinterpret_cast<const float4*>(p + g * CPG)[0];
        float4 b  = reinterpret_cast<const float4*>(p + g * CPG)[1];
        float4 ga = reinterpret_cast<const float4*>(gamma + g * CPG)[0];
        float4 gb = reinterpret_cast<const float4*>(gamma + g * CPG)[1];
        float4 ba = reinterpret_cast<const float4*>(beta  + g * CPG)[0];
        float4 bb = reinterpret_cast<const float4*>(beta  + g * CPG)[1];

        float v[8];
        v[0] = gnsilu_one(a.x, m, r, ga.x, ba.x);
        v[1] = gnsilu_one(a.y, m, r, ga.y, ba.y);
        v[2] = gnsilu_one(a.z, m, r, ga.z, ba.z);
        v[3] = gnsilu_one(a.w, m, r, ga.w, ba.w);
        v[4] = gnsilu_one(b.x, m, r, gb.x, bb.x);
        v[5] = gnsilu_one(b.y, m, r, gb.y, bb.y);
        v[6] = gnsilu_one(b.z, m, r, gb.z, bb.z);
        v[7] = gnsilu_one(b.w, m, r, gb.w, bb.w);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            size_t off = (size_t)(g * CPG + i) * (size_t)S;
            o[off] = v[i] + r0[off];
        }
    }
}

// ------------------------------------------------------------------ launchers
static inline int pick_P(int ng, int S) {
    int P = (2048 + ng - 1) / ng;
    int maxP = (S + 255) / 256;
    if (maxP < 1) maxP = 1;
    if (P > maxP) P = maxP;
    if (P < 1) P = 1;
    return P;
}

torch::Tensor to_nhwc(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "float32 cuda expected");
    TORCH_CHECK(x.dim() == 4, "4D expected");
    int N = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
    TORCH_CHECK(C % 8 == 0, "C%8");
    int S = H * W;
    auto out = torch::empty({N, C, H, W},
                            x.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto stream = at::cuda::getDefaultCUDAStream();
    dim3 blk(256);
    dim3 grd((S + 255) / 256, N);
    nchw2nhwc_kernel<<<grd, blk, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), S, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

static void compute_stats(const float* in, int N, int S, int C, float eps,
                          torch::Tensor& mean, torch::Tensor& rstd,
                          const torch::TensorOptions& opts) {
    int ng = N * GRP;
    int P = pick_P(ng, S);
    auto part = torch::empty({(long)ng * P * 2}, opts);
    auto stream = at::cuda::getDefaultCUDAStream();
    dim3 blk(256);
    dim3 grd(P, ng);
    moments_kernel<<<grd, blk, 0, stream>>>(in, part.data_ptr<float>(), S, C, P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int fb = 128;
    while (fb > 32 && fb >= 2 * P) fb >>= 1;
    finalize_kernel<<<dim3(ng), dim3(fb), 0, stream>>>(part.data_ptr<float>(),
                                                       mean.data_ptr<float>(),
                                                       rstd.data_ptr<float>(),
                                                       P, (long long)S * CPG, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "float32 cuda expected");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "channels_last expected");
    int N = (int)y.size(0), C = (int)y.size(1), H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == 256, "C==256 expected");
    int S = H * W;
    auto opts = y.options();
    auto mean = torch::empty({(long)N * GRP}, opts);
    auto rstd = torch::empty({(long)N * GRP}, opts);
    compute_stats(y.data_ptr<float>(), N, S, C, (float)eps, mean, rstd, opts);

    auto out = torch::empty({N, C, H, W}, opts.memory_format(at::MemoryFormat::ChannelsLast));
    auto stream = at::cuda::getDefaultCUDAStream();
    long long tot = (long long)S * GRP;
    dim3 blk(256);
    dim3 grd((unsigned)((tot + 255) / 256), N);
    gnsilu_nhwc_kernel<<<grd, blk, 0, stream>>>(y.data_ptr<float>(), out.data_ptr<float>(),
                                                gamma.data_ptr<float>(), beta.data_ptr<float>(),
                                                mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                                S, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_res(torch::Tensor y, torch::Tensor res,
                          torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "float32 cuda expected");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "channels_last expected");
    TORCH_CHECK(res.is_contiguous(), "residual must be NCHW contiguous");
    int N = (int)y.size(0), C = (int)y.size(1), H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == 256, "C==256 expected");
    int S = H * W;
    auto opts = y.options();
    auto mean = torch::empty({(long)N * GRP}, opts);
    auto rstd = torch::empty({(long)N * GRP}, opts);
    compute_stats(y.data_ptr<float>(), N, S, C, (float)eps, mean, rstd, opts);

    auto out = torch::empty({N, C, H, W}, opts);   // NCHW contiguous
    auto stream = at::cuda::getDefaultCUDAStream();
    dim3 blk(256);
    dim3 grd((S + 255) / 256, N);
    gnsilu_res_nchw_kernel<<<grd, blk, 0, stream>>>(y.data_ptr<float>(), res.data_ptr<float>(),
                                                    out.data_ptr<float>(),
                                                    gamma.data_ptr<float>(), beta.data_ptr<float>(),
                                                    mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                                    S, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_cpp_src = r"""
torch::Tensor to_nhwc(torch::Tensor x);
torch::Tensor gn_silu(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor gn_silu_res(torch::Tensor y, torch::Tensor res, torch::Tensor gamma, torch::Tensor beta, double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["to_nhwc", "gn_silu", "gn_silu_res"],
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
    # See file header for the granularity/fusion plan (granularity C).
    def __init__(self):
        super().__init__()
        self.ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps = float(eps.item())
        eps = float(eps)

        num_groups = 32
        C = x.size(1)

        # Safety fallback for shapes this kernel family does not specialise for.
        if (x.dim() != 4) or (C != 256) or (x.dtype != torch.float32) or (not x.is_cuda):
            out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
            out = F.silu(out)
            out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
            out = F.silu(out)
            return out + x

        x_c = x if x.is_contiguous() else x.contiguous()

        # NCHW -> NHWC once; the whole pipeline then stays in NHWC so cuDNN never
        # inserts its own layout-conversion kernels.
        xn = self.ext.to_nhwc(x_c)

        w1 = conv1_weight if conv1_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv1_weight.contiguous(memory_format=torch.channels_last)
        y1 = F.conv2d(xn, w1, None, 1, 1)
        if not y1.is_contiguous(memory_format=torch.channels_last):
            y1 = y1.contiguous(memory_format=torch.channels_last)

        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        z1 = self.ext.gn_silu(y1, g1, b1, eps)

        w2 = conv2_weight if conv2_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv2_weight.contiguous(memory_format=torch.channels_last)
        y2 = F.conv2d(z1, w2, None, 1, 1)
        if not y2.is_contiguous(memory_format=torch.channels_last):
            y2 = y2.contiguous(memory_format=torch.channels_last)

        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()
        return self.ext.gn_silu_res(y2, x_c, g2, b2, eps)
