# =============================================================================
# ModelNew — fused VAE residual block:
#   Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> +residual
#
# HEADER (required):
# 1) CHOSEN GRANULARITY: (C) "fuse many ops into one/few kernels".
#    Everything except the two vendor convolutions is folded into custom CUDA
#    kernels, and the whole pipeline is run in NHWC (channels_last) so that the
#    cuDNN NHWC tensor-core kernels need NO layout conversions at all.
#
# 2) OPS REPLACED BY CUSTOM CUDA:
#      - NCHW -> NHWC layout conversion of the input          (nchw2nhwc_kernel,
#        tiled 32x32 shared-memory transpose; replaces cudnn's nchwToNhwcKernel
#        which cost 7.6% of the reference runtime)
#      - GroupNorm(32 groups) statistics                      (gn_stats_kernel +
#        gn_finalize_kernel; replaces RowwiseMomentsCUDAKernel /
#        ComputeFusedParamsCUDAKernel)
#      - GroupNorm affine + SiLU                              (gn_apply_silu_kernel)
#      - GroupNorm affine + SiLU + residual add + NHWC->NCHW  (final_kernel)
#
# 3) FUSIONS:
#      - kernel gn_apply_silu_kernel : (x-mean)*rstd*w + b  AND  SiLU, one pass,
#        float4-vectorised, writing directly the NHWC input of conv2.
#      - kernel final_kernel : GroupNorm affine + SiLU + residual add + the
#        NHWC->NCHW transpose, all in one shared-memory-tiled pass. This removes
#        the separate SiLU kernel, the separate add kernel and the output layout
#        conversion (3 library kernels -> 1).
#      - GroupNorm is done as a deterministic 2-stage reduction (per-block
#        partials, then a fixed-order finalize) -- no atomics, so the result is
#        bit-reproducible run to run.
#
# 4) WHAT STAYS IN PYTORCH:
#      - F.conv2d (x2): the vendor sm80_xmma implicit-GEMM TF32 NHWC kernel is
#        measured at ~91% of this GPU's TF32 roofline; re-implementing it cannot
#        win, and cuDNN's default heuristic (benchmark left OFF) picks exactly
#        the same kernel as the reference, so conv numerics are unchanged.
#      - weight -> channels_last conversion: 2.4 MB, negligible, torch's copy is
#        fine there.
#      - a generic PyTorch fallback path is kept for C != 256 (kernels are
#        specialised for C=256 / 32 groups, which the problem fixes as const).
#
# PRECISION: everything is float32 storage + float32 arithmetic; reductions
# accumulate in float32 (short per-thread chains). TF32 is used only inside the
# vendor conv, exactly as the reference does.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CDIM 256
#define GDIM 32
#define CPG  8
#define TILE 32

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// NCHW -> NHWC tiled transpose (32 pixels x 32 channels per block)
// sm[c_local][p_local]
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int HW, int C) {
    __shared__ float sm[TILE][TILE + 1];
    const int p0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..7

    const float* ibase = in  + (size_t)n * (size_t)C * (size_t)HW;
    float*       obase = out + (size_t)n * (size_t)HW * (size_t)C;

    #pragma unroll
    for (int k = 0; k < TILE; k += 8) {
        const int c = c0 + ty + k;
        const int p = p0 + tx;
        float v = 0.0f;
        if (p < HW) v = ibase[(size_t)c * (size_t)HW + (size_t)p];
        sm[ty + k][tx] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; k += 8) {
        const int p = p0 + ty + k;
        const int c = c0 + tx;
        if (p < HW) obase[(size_t)p * (size_t)C + (size_t)c] = sm[tx][ty + k];
    }
}

// ---------------------------------------------------------------------------
// GroupNorm stage 1: per-(n, pixel-chunk) partial sums / sums of squares,
// C = 256 channels handled by 256 threads (thread == channel).
// psum / psq layout: [(n*32 + g) * nblk + blk]
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                int HW, int nblk, int chunk) {
    const int blk = blockIdx.x;
    const int n   = blockIdx.y;
    const int c   = threadIdx.x;

    int p0 = blk * chunk;
    int p1 = p0 + chunk;
    if (p1 > HW) p1 = HW;

    const float* base = x + (size_t)n * (size_t)HW * (size_t)CDIM + (size_t)c;

    float s = 0.0f, ss = 0.0f;
    for (int p = p0; p < p1; ++p) {
        float v = base[(size_t)p * (size_t)CDIM];
        s  += v;
        ss += v * v;
    }
    // reduce over the 8 channels of a group (lanes are contiguous)
    #pragma unroll
    for (int off = 1; off < CPG; off <<= 1) {
        s  += __shfl_xor_sync(0xffffffffu, s,  off);
        ss += __shfl_xor_sync(0xffffffffu, ss, off);
    }
    if ((c & (CPG - 1)) == 0) {
        const int g = c >> 3;
        const size_t idx = ((size_t)n * GDIM + g) * (size_t)nblk + (size_t)blk;
        psum[idx] = s;
        psq[idx]  = ss;
    }
}

// ---------------------------------------------------------------------------
// GroupNorm stage 2: deterministic reduction over nblk partials + fused affine
// scale/shift per (n, c):  y = x*scale + shift
// grid = N*32, block = 128
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ psum,
                                   const float* __restrict__ psq,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   float* __restrict__ scale,
                                   float* __restrict__ shift,
                                   int nblk, float eps, float M) {
    __shared__ float sh[2][4];
    const int idx = blockIdx.x;             // n*32 + g
    const int tid = threadIdx.x;

    float s = 0.0f, ss = 0.0f;
    for (int i = tid; i < nblk; i += blockDim.x) {
        const size_t o = (size_t)idx * (size_t)nblk + (size_t)i;
        s  += psum[o];
        ss += psq[o];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off);
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    if ((tid & 31) == 0) {
        sh[0][tid >> 5] = s;
        sh[1][tid >> 5] = ss;
    }
    __syncthreads();

    if (tid < CPG) {
        float ts = 0.0f, tss = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { ts += sh[0][i]; tss += sh[1][i]; }
        const float mean = ts / M;
        float var = tss / M - mean * mean;
        var = fmaxf(var, 0.0f);
        const float rstd = rsqrtf(var + eps);

        const int n = idx >> 5;
        const int g = idx & 31;
        const int c = g * CPG + tid;
        const float sc = rstd * w[c];
        scale[n * CDIM + c] = sc;
        shift[n * CDIM + c] = b[c] - mean * sc;
    }
}

// ---------------------------------------------------------------------------
// GroupNorm affine + SiLU (NHWC in, NHWC out), float4 vectorised.
// grid = (ceil(HW*64/256), N), block = 256
// ---------------------------------------------------------------------------
__global__ void gn_apply_silu_kernel(const float* __restrict__ x,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ shift,
                                     float* __restrict__ out,
                                     int HW4) {
    const int n = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW4) return;

    const size_t off = (size_t)n * (size_t)HW4 + (size_t)i;
    const float4 v = reinterpret_cast<const float4*>(x)[off];
    const int c4 = i & 63;
    const float4 sc = reinterpret_cast<const float4*>(scale)[n * 64 + c4];
    const float4 sf = reinterpret_cast<const float4*>(shift)[n * 64 + c4];

    float4 r;
    r.x = silu_f(v.x * sc.x + sf.x);
    r.y = silu_f(v.y * sc.y + sf.y);
    r.z = silu_f(v.z * sc.z + sf.z);
    r.w = silu_f(v.w * sc.w + sf.w);
    reinterpret_cast<float4*>(out)[off] = r;
}

// ---------------------------------------------------------------------------
// Final fused epilogue: GroupNorm affine + SiLU (NHWC) + residual (NCHW)
// + NHWC->NCHW transpose, one shared-memory tiled pass.
// sm[p_local][c_local]
// ---------------------------------------------------------------------------
__global__ void final_kernel(const float* __restrict__ y,
                             const float* __restrict__ scale,
                             const float* __restrict__ shift,
                             const float* __restrict__ res,
                             float* __restrict__ out,
                             int HW) {
    __shared__ float sm[TILE][TILE + 1];
    const int p0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const float* ybase = y + (size_t)n * (size_t)HW * (size_t)CDIM;

    const int cl = c0 + tx;
    const float sc = scale[n * CDIM + cl];
    const float sf = shift[n * CDIM + cl];

    #pragma unroll
    for (int k = 0; k < TILE; k += 8) {
        const int p = p0 + ty + k;
        float v = 0.0f;
        if (p < HW) {
            v = ybase[(size_t)p * (size_t)CDIM + (size_t)cl];
            v = silu_f(v * sc + sf);
        }
        sm[ty + k][tx] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; k += 8) {
        const int c = c0 + ty + k;
        const int p = p0 + tx;
        if (p < HW) {
            const size_t o = ((size_t)n * CDIM + (size_t)c) * (size_t)HW + (size_t)p;
            out[o] = sm[tx][ty + k] + res[o];
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static void compute_affine(const float* y, int N, int HW,
                           const torch::Tensor& w, const torch::Tensor& b,
                           float eps,
                           torch::Tensor& scale, torch::Tensor& shift,
                           cudaStream_t stream) {
    int target = 512;
    int nblk = (target + N - 1) / N;
    int cap  = (HW + 63) / 64;
    if (nblk > cap) nblk = cap;
    if (nblk < 1)   nblk = 1;
    int chunk = (HW + nblk - 1) / nblk;
    nblk = (HW + chunk - 1) / chunk;      // drop empty trailing blocks

    auto opts = w.options();
    auto psum = at::empty({(long)N * GDIM * (long)nblk}, opts);
    auto psq  = at::empty({(long)N * GDIM * (long)nblk}, opts);

    dim3 g1(nblk, N);
    gn_stats_kernel<<<g1, CDIM, 0, stream>>>(y,
        psum.data_ptr<float>(), psq.data_ptr<float>(), HW, nblk, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_kernel<<<N * GDIM, 128, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        w.data_ptr<float>(), b.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        nblk, eps, (float)((double)HW * (double)CPG));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor to_nhwc(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be cuda");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "4d only");
    TORCH_CHECK(x.is_contiguous(), "expect contiguous NCHW");
    const int N = (int)x.size(0), C = (int)x.size(1);
    const int H = (int)x.size(2), W = (int)x.size(3);
    TORCH_CHECK(C % TILE == 0, "C must be multiple of 32");
    const int HW = H * W;

    auto out = at::empty({N, C, H, W}, x.options(), at::MemoryFormat::ChannelsLast);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blk(32, 8);
    dim3 grd((HW + TILE - 1) / TILE, C / TILE, N);
    nchw2nhwc_kernel<<<grd, blk, 0, stream>>>(x.data_ptr<float>(),
                                              out.data_ptr<float>(), HW, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kFloat);
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "expect channels_last");
    const int N = (int)y.size(0), C = (int)y.size(1);
    const int H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == CDIM, "specialised for C=256");
    const int HW = H * W;

    auto out   = at::empty_like(y);
    auto scale = at::empty({N, C}, y.options());
    auto shift = at::empty({N, C}, y.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    compute_affine(y.data_ptr<float>(), N, HW, w, b, (float)eps, scale, shift, stream);

    const int HW4 = HW * (CDIM / 4);
    dim3 grd((HW4 + 255) / 256, N);
    gn_apply_silu_kernel<<<grd, 256, 0, stream>>>(
        y.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
        out.data_ptr<float>(), HW4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_add(torch::Tensor y, torch::Tensor w, torch::Tensor b,
                          double eps, torch::Tensor res) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kFloat);
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "expect channels_last");
    TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");
    const int N = (int)y.size(0), C = (int)y.size(1);
    const int H = (int)y.size(2), W = (int)y.size(3);
    TORCH_CHECK(C == CDIM, "specialised for C=256");
    const int HW = H * W;

    auto out   = at::empty({N, C, H, W}, res.options());
    auto scale = at::empty({N, C}, y.options());
    auto shift = at::empty({N, C}, y.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    compute_affine(y.data_ptr<float>(), N, HW, w, b, (float)eps, scale, shift, stream);

    dim3 blk(32, 8);
    dim3 grd((HW + TILE - 1) / TILE, C / TILE, N);
    final_kernel<<<grd, blk, 0, stream>>>(
        y.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
        res.data_ptr<float>(), out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor to_nhwc(torch::Tensor x);
torch::Tensor gn_silu(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps);
torch::Tensor gn_silu_add(torch::Tensor y, torch::Tensor w, torch::Tensor b,
                          double eps, torch::Tensor res);
'''

_ext = load_inline(
    name="vae_resblock_fused_v1",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["to_nhwc", "gn_silu", "gn_silu_add"],
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
    """See file header for granularity / fusion plan (granularity C)."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    @staticmethod
    def _fallback(x, w1, n1w, n1b, w2, n2w, n2b, eps):
        out = F.conv2d(x, w1, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n1w, bias=n1b, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, w2, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n2w, bias=n2b, eps=eps)
        out = F.silu(out)
        return out + x

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        e = float(eps)

        if (x.dtype != torch.float32) or (not x.is_cuda) or (x.dim() != 4) \
                or (x.size(1) != 256):
            return self._fallback(x, conv1_weight, norm1_weight, norm1_bias,
                                  conv2_weight, norm2_weight, norm2_bias, e)

        xn = x if x.is_contiguous() else x.contiguous()

        # NCHW -> NHWC once (custom tiled transpose); conv then runs natively
        # in NHWC and cuDNN performs no layout conversion at all.
        xc = self.ext.to_nhwc(xn)

        w1 = conv1_weight if conv1_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv1_weight.contiguous(memory_format=torch.channels_last)
        w2 = conv2_weight if conv2_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv2_weight.contiguous(memory_format=torch.channels_last)

        o1 = F.conv2d(xc, w1, bias=None, stride=1, padding=1)
        if not o1.is_contiguous(memory_format=torch.channels_last):
            o1 = o1.contiguous(memory_format=torch.channels_last)

        nw1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        nb1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        h = self.ext.gn_silu(o1, nw1, nb1, e)

        o2 = F.conv2d(h, w2, bias=None, stride=1, padding=1)
        if not o2.is_contiguous(memory_format=torch.channels_last):
            o2 = o2.contiguous(memory_format=torch.channels_last)

        nw2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        nb2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()
        return self.ext.gn_silu_add(o2, nw2, nb2, e, xn)
