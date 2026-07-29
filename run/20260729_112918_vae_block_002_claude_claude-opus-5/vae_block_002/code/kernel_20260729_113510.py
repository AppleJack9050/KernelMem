# =============================================================================
# ModelNew — fused VAE residual block:
#   Conv3x3 -> GroupNorm(32) -> SiLU -> Conv3x3 -> GroupNorm(32) -> SiLU -> +x
#
# SEED GRANULARITY: (C) "fuse many ops into one/few kernels".
#
# 1) Chosen granularity: (C). Everything except the two 3x3 convolutions is
#    expressed as a small set of hand-written fused CUDA kernels; no per-op
#    PyTorch boundaries are kept for the replaced part.
#
# 2) Ops replaced by custom CUDA kernels:
#      - NCHW -> NHWC layout conversion of the input activation
#        (removes the cudnn `nchwToNhwcKernel` calls seen in the profile)
#      - group_norm #1 (mean/var reduction + affine) + silu
#      - group_norm #2 (mean/var reduction + affine) + silu + residual add
#        + NHWC -> NCHW layout conversion of the final result
#
# 3) Fusion map:
#      k_nchw2nhwc            : tiled shared-memory transpose (input -> NHWC)
#      k_gn_stats             : per-(n,group) partial sum / sumsq, vectorised
#                               float4 loads over the 8 contiguous NHWC channels
#      k_gn_finish            : partials -> mean/rstd -> per-(n,c) scale/shift
#                               (folds gamma/beta into the affine pair)
#      k_gn_apply_nhwc        : y = silu(x*scale+shift)         [NHWC -> NHWC]
#      k_gn_apply_add_nchw    : out = silu(x*scale+shift) + residual, fused with
#                               the NHWC->NCHW transpose in shared memory so the
#                               final layout conversion costs no extra traffic
#    => GroupNorm + SiLU + residual-add + 2 layout conversions collapse into
#       3 memory-bound passes instead of ~9 library kernels.
#
# 4) What stays in PyTorch and why:
#      - F.conv2d (channels_last): the vendor tf32 implicit-GEMM conv already
#        runs at ~92% of the TF32 tensor-core roofline for this shape; a
#        hand-written replacement cannot win, and feeding/consuming NHWC
#        removes the layout conversions that surrounded it.
#      - weight -> channels_last: 2.4 MB one-shot copy, negligible.
#      - a pure-PyTorch fallback path is kept only for non-CUDA/non-fp32 input.
#
# Precision: all storage and arithmetic stay float32 (TF32 tensor cores are used
# by the conv exactly as in the reference); reductions accumulate in fp32 per
# thread and fp64 for the final partial combination.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define TILE 32

__device__ __forceinline__ float silu_f(float z) {
    return z / (1.0f + __expf(-z));
}

// ---------------------------------------------------------------------------
// NCHW -> NHWC tiled transpose (writes a channels_last tensor's raw storage)
// ---------------------------------------------------------------------------
__global__ void k_nchw2nhwc(const float* __restrict__ in,
                            float* __restrict__ out,
                            int C, int HW) {
    __shared__ float tile[TILE][TILE + 1];
    const int n  = blockIdx.z;
    const int c0 = blockIdx.y * TILE;
    const int p0 = blockIdx.x * TILE;
    const int tx = threadIdx.x, ty = threadIdx.y;

    int c = c0 + ty, p = p0 + tx;
    if (p < HW && c < C)
        tile[ty][tx] = in[((size_t)n * C + c) * (size_t)HW + p];
    __syncthreads();

    int c2 = c0 + tx, p2 = p0 + ty;
    if (p2 < HW && c2 < C)
        out[((size_t)n * HW + p2) * (size_t)C + c2] = tile[tx][ty];
}

// ---------------------------------------------------------------------------
// GroupNorm statistics over NHWC (group channels are contiguous)
// grid = (NBLK, B*G), block = 256
// ---------------------------------------------------------------------------
__global__ void k_gn_stats(const float* __restrict__ x,
                           float2* __restrict__ part,
                           int C, int HW, int GS, int G, int NBLK) {
    const int bg = blockIdx.y;
    const int n  = bg / G;
    const int g  = bg - n * G;

    const long long base = (long long)n * HW * C + (long long)g * GS;
    const int chunk  = (HW + NBLK - 1) / NBLK;
    const int pstart = blockIdx.x * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    float s = 0.f, ss = 0.f;

    if ((GS & 3) == 0 && (C & 3) == 0) {
        const int GS4 = GS >> 2;
        for (int p = pstart + threadIdx.x; p < pend; p += blockDim.x) {
            const float4* ptr = (const float4*)(x + base + (long long)p * C);
            #pragma unroll 2
            for (int k = 0; k < GS4; ++k) {
                float4 v = ptr[k];
                s  += v.x + v.y + v.z + v.w;
                ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    } else {
        for (int p = pstart + threadIdx.x; p < pend; p += blockDim.x) {
            const float* ptr = x + base + (long long)p * C;
            for (int k = 0; k < GS; ++k) {
                float v = ptr[k];
                s += v; ss += v * v;
            }
        }
    }

    // block reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s, off);
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    __shared__ float sm_s[32], sm_ss[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    if (lane == 0) { sm_s[warp] = s; sm_ss[warp] = ss; }
    __syncthreads();
    if (warp == 0) {
        s  = (lane < nwarp) ? sm_s[lane]  : 0.f;
        ss = (lane < nwarp) ? sm_ss[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s  += __shfl_down_sync(0xffffffffu, s, off);
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        }
        if (lane == 0) part[(long long)bg * NBLK + blockIdx.x] = make_float2(s, ss);
    }
}

// ---------------------------------------------------------------------------
// partials -> per-(n,c) affine pair.  grid = (B*G), block = 32
// ---------------------------------------------------------------------------
__global__ void k_gn_finish(const float2* __restrict__ part,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            float* __restrict__ scale,
                            float* __restrict__ shift,
                            int C, int HW, int GS, int G, int NBLK, float eps) {
    const int bg = blockIdx.x;
    const int n  = bg / G;
    const int g  = bg - n * G;

    __shared__ float sh_mean, sh_rstd;
    if (threadIdx.x == 0) {
        double s = 0.0, ss = 0.0;
        const float2* p = part + (long long)bg * NBLK;
        for (int i = 0; i < NBLK; ++i) { s += (double)p[i].x; ss += (double)p[i].y; }
        double cnt = (double)((long long)GS * (long long)HW);
        double mean = s / cnt;
        double var  = ss / cnt - mean * mean;
        if (var < 0.0) var = 0.0;
        sh_mean = (float)mean;
        sh_rstd = (float)(1.0 / sqrt(var + (double)eps));
    }
    __syncthreads();

    const float mean = sh_mean, rstd = sh_rstd;
    for (int k = threadIdx.x; k < GS; k += blockDim.x) {
        int c = g * GS + k;
        float gm = gamma ? gamma[c] : 1.0f;
        float bt = beta  ? beta[c]  : 0.0f;
        float a = rstd * gm;
        scale[(long long)n * C + c] = a;
        shift[(long long)n * C + c] = bt - mean * a;
    }
}

// ---------------------------------------------------------------------------
// y = silu(x*scale + shift), NHWC -> NHWC   (vectorised)
// ---------------------------------------------------------------------------
__global__ void k_gn_apply_nhwc_v4(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   const float* __restrict__ scale,
                                   const float* __restrict__ shift,
                                   int C, int C4, int imgs4) {
    const int n = blockIdx.y;
    const float4* xp = (const float4*)x + (size_t)n * imgs4;
    float4*       yp = (float4*)y       + (size_t)n * imgs4;
    const float4* sc = (const float4*)(scale + (size_t)n * C);
    const float4* sf = (const float4*)(shift + (size_t)n * C);

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < imgs4;
         j += gridDim.x * blockDim.x) {
        int c4 = j % C4;
        float4 v = xp[j];
        float4 a = sc[c4];
        float4 b = sf[c4];
        float4 o;
        o.x = silu_f(v.x * a.x + b.x);
        o.y = silu_f(v.y * a.y + b.y);
        o.z = silu_f(v.z * a.z + b.z);
        o.w = silu_f(v.w * a.w + b.w);
        yp[j] = o;
    }
}

__global__ void k_gn_apply_nhwc_s(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  const float* __restrict__ scale,
                                  const float* __restrict__ shift,
                                  int C, int imgs) {
    const int n = blockIdx.y;
    const float* xp = x + (size_t)n * imgs;
    float*       yp = y + (size_t)n * imgs;
    const float* sc = scale + (size_t)n * C;
    const float* sf = shift + (size_t)n * C;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < imgs;
         j += gridDim.x * blockDim.x) {
        int c = j % C;
        yp[j] = silu_f(xp[j] * sc[c] + sf[c]);
    }
}

// ---------------------------------------------------------------------------
// out(NCHW) = silu(x_nhwc*scale + shift) + residual(NCHW)   (fused transpose)
// ---------------------------------------------------------------------------
__global__ void k_gn_apply_add_nchw(const float* __restrict__ x,
                                    const float* __restrict__ res,
                                    float* __restrict__ out,
                                    const float* __restrict__ scale,
                                    const float* __restrict__ shift,
                                    int C, int HW) {
    __shared__ float tile[TILE][TILE + 1];
    const int n  = blockIdx.z;
    const int c0 = blockIdx.y * TILE;
    const int p0 = blockIdx.x * TILE;
    const int tx = threadIdx.x, ty = threadIdx.y;

    int c = c0 + tx;      // channel  (fast NHWC dim)
    int p = p0 + ty;      // pixel
    if (p < HW && c < C) {
        float v = x[((size_t)n * HW + p) * (size_t)C + c];
        float a = scale[(size_t)n * C + c];
        float b = shift[(size_t)n * C + c];
        tile[ty][tx] = silu_f(v * a + b);
    }
    __syncthreads();

    int cw = c0 + ty;
    int pw = p0 + tx;
    if (pw < HW && cw < C) {
        size_t o = ((size_t)n * C + cw) * (size_t)HW + pw;
        out[o] = tile[tx][ty] + res[o];
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static int compute_nblk(long long HW) {
    long long v = (HW + 255) / 256;
    if (v < 1) v = 1;
    if (v > 64) v = 64;
    return (int)v;
}

static void gn_scale_shift(const torch::Tensor& x, const torch::Tensor& gamma,
                           const torch::Tensor& beta, double eps, int64_t G,
                           torch::Tensor& scale, torch::Tensor& shift) {
    const int B  = (int)x.size(0);
    const int C  = (int)x.size(1);
    const int HW = (int)(x.size(2) * x.size(3));
    const int GS = C / (int)G;
    const int NBLK = compute_nblk(HW);

    auto opts = x.options();
    auto part = torch::empty({(long long)B * G * NBLK * 2}, opts);
    scale = torch::empty({(long long)B * C}, opts);
    shift = torch::empty({(long long)B * C}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 g1(NBLK, B * (int)G);
    k_gn_stats<<<g1, 256, 0, stream>>>(
        x.data_ptr<float>(), (float2*)part.data_ptr<float>(),
        C, HW, GS, (int)G, NBLK);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    k_gn_finish<<<B * (int)G, 32, 0, stream>>>(
        (const float2*)part.data_ptr<float>(),
        gamma.defined() ? gamma.data_ptr<float>() : nullptr,
        beta.defined() ? beta.data_ptr<float>() : nullptr,
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        C, HW, GS, (int)G, NBLK, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// exported ops
// ---------------------------------------------------------------------------
torch::Tensor nchw_to_nhwc(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(x.dim() == 4, "4D expected");
    auto xc = x.is_contiguous() ? x : x.contiguous();
    const int B = (int)xc.size(0), C = (int)xc.size(1);
    const int H = (int)xc.size(2), W = (int)xc.size(3);
    const int HW = H * W;
    auto out = torch::empty({B, C, H, W},
                            xc.options().memory_format(at::MemoryFormat::ChannelsLast));
    dim3 blk(TILE, TILE);
    dim3 grd((HW + TILE - 1) / TILE, (C + TILE - 1) / TILE, B);
    auto stream = at::cuda::getCurrentCUDAStream();
    k_nchw2nhwc<<<grd, blk, 0, stream>>>(xc.data_ptr<float>(), out.data_ptr<float>(), C, HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                           double eps, int64_t G) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(x.dim() == 4, "4D expected");
    auto xl = x.is_contiguous(at::MemoryFormat::ChannelsLast)
                  ? x : x.contiguous(at::MemoryFormat::ChannelsLast);
    const int B = (int)xl.size(0), C = (int)xl.size(1);
    const int H = (int)xl.size(2), W = (int)xl.size(3);
    const int HW = H * W;
    TORCH_CHECK(C % G == 0, "C must be divisible by groups");

    torch::Tensor scale, shift;
    gn_scale_shift(xl, gamma, beta, eps, G, scale, shift);

    auto out = torch::empty({B, C, H, W},
                            xl.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto stream = at::cuda::getCurrentCUDAStream();

    if ((C & 3) == 0) {
        const int C4 = C >> 2;
        const int imgs4 = C4 * HW;
        int gx = (imgs4 + 255) / 256;
        if (gx > 4096) gx = 4096;
        if (gx < 1) gx = 1;
        dim3 grd(gx, B);
        k_gn_apply_nhwc_v4<<<grd, 256, 0, stream>>>(
            xl.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(), C, C4, imgs4);
    } else {
        const int imgs = C * HW;
        int gx = (imgs + 255) / 256;
        if (gx > 4096) gx = 4096;
        if (gx < 1) gx = 1;
        dim3 grd(gx, B);
        k_gn_apply_nhwc_s<<<grd, 256, 0, stream>>>(
            xl.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(), C, imgs);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_add_nchw(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                               double eps, int64_t G, torch::Tensor residual) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    auto xl = x.is_contiguous(at::MemoryFormat::ChannelsLast)
                  ? x : x.contiguous(at::MemoryFormat::ChannelsLast);
    auto rc = residual.is_contiguous() ? residual : residual.contiguous();

    const int B = (int)xl.size(0), C = (int)xl.size(1);
    const int H = (int)xl.size(2), W = (int)xl.size(3);
    const int HW = H * W;
    TORCH_CHECK(C % G == 0, "C must be divisible by groups");

    torch::Tensor scale, shift;
    gn_scale_shift(xl, gamma, beta, eps, G, scale, shift);

    auto out = torch::empty({B, C, H, W}, xl.options());  // contiguous NCHW
    dim3 blk(TILE, TILE);
    dim3 grd((HW + TILE - 1) / TILE, (C + TILE - 1) / TILE, B);
    auto stream = at::cuda::getCurrentCUDAStream();
    k_gn_apply_add_nchw<<<grd, blk, 0, stream>>>(
        xl.data_ptr<float>(), rc.data_ptr<float>(), out.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(), C, HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_cpp_src = r"""
torch::Tensor nchw_to_nhwc(torch::Tensor x);
torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                           double eps, int64_t G);
torch::Tensor gn_silu_add_nchw(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                               double eps, int64_t G, torch::Tensor residual);
"""

_ext = load_inline(
    name="vae_resblock_fused_v1",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["nchw_to_nhwc", "gn_silu_nhwc", "gn_silu_add_nchw"],
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
    """See file header for the granularity/fusion plan (granularity C)."""

    NUM_GROUPS = 32

    def __init__(self):
        super().__init__()
        self.ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        G = ModelNew.NUM_GROUPS

        # ---- safety fallback: only for dtypes/devices the kernels don't cover
        if (not x.is_cuda) or x.dtype != torch.float32:
            out = F.conv2d(x, conv1_weight, None, 1, 1)
            out = F.group_norm(out, G, norm1_weight, norm1_bias, eps_f)
            out = F.silu(out)
            out = F.conv2d(out, conv2_weight, None, 1, 1)
            out = F.group_norm(out, G, norm2_weight, norm2_bias, eps_f)
            out = F.silu(out)
            return out + x

        xr = x if x.is_contiguous() else x.contiguous()

        # NCHW -> NHWC once (custom tiled transpose); everything downstream of
        # this point lives in channels_last so cuDNN never re-lays-out anything.
        xl = self.ext.nchw_to_nhwc(xr)

        w1 = conv1_weight if conv1_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv1_weight.contiguous(memory_format=torch.channels_last)
        out = F.conv2d(xl, w1, None, 1, 1)

        # GroupNorm + SiLU fused (NHWC in / NHWC out)
        out = self.ext.gn_silu_nhwc(out, norm1_weight, norm1_bias, eps_f, G)

        w2 = conv2_weight if conv2_weight.is_contiguous(memory_format=torch.channels_last) \
            else conv2_weight.contiguous(memory_format=torch.channels_last)
        out = F.conv2d(out, w2, None, 1, 1)

        # GroupNorm + SiLU + residual add + NHWC->NCHW transpose, single pass
        return self.ext.gn_silu_add_nchw(out, norm2_weight, norm2_bias, eps_f, G, xr)
