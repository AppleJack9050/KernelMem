# ==========================================================================
# ModelNew: fused Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> +residual
#
# HEADER (required):
#   1) Chosen granularity: (C) fuse many ops into one/few custom kernels.
#   2) Ops replaced by custom CUDA:
#        - group_norm #1 (stats + affine)  + silu #1            -> custom kernels
#        - group_norm #2 (stats + affine)  + silu #2 + residual add
#          + the NHWC->NCHW layout conversion of the result     -> ONE custom kernel
#   3) Fusion map:
#        kernel gn_stats_kernel   : per-(n,group) partial sum/sumsq over NHWC data
#        kernel gn_finalize_kernel: partials -> mean/rstd (double accumulation)
#        kernel gn_silu_apply_nhwc: (normalize + affine + SiLU) fused, NHWC->NHWC
#        kernel gn_silu_res_nchw  : (normalize + affine + SiLU + residual add +
#                                    shared-memory NHWC->NCHW transpose) fused
#   4) Left in PyTorch:
#        - the two 3x3 convolutions: they are vendor tensor-core (TF32) implicit
#          GEMMs at/near roofline; re-implementing them is not worthwhile here.
#          They are run in channels_last so cuDNN uses its native NHWC path and
#          performs NO nchwToNhwc/nhwcToNchw conversions at all (the reference
#          spent 7.2% of its time in those conversions); the single unavoidable
#          NCHW->NHWC of the input is a torch .contiguous(channels_last), and the
#          final NHWC->NCHW is fused into the last custom kernel for free.
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

#define CHANS 256
#define GROUPS 32
#define CPG 8            // channels per group
#define TP 32            // positions per tile in the transposing kernel
#define SSTRIDE 257      // padded shared row stride (avoids bank conflicts)

// ---------------------------------------------------------------------------
// stage 1: partial sums / sums-of-squares per (n, group)
// grid = (nblk, B), block = 256 threads (one per channel)
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                float* __restrict__ partial,   // [2][B*G][nblk]
                                int HW, int nblk, long long pstride) {
    const int blk = blockIdx.x;
    const int n   = blockIdx.y;
    const int c   = threadIdx.x;
    const int g   = c >> 3;

    const int p0 = (int)(((long long)blk * HW) / nblk);
    const int p1 = (int)(((long long)(blk + 1) * HW) / nblk);

    const float* base = y + ((long long)n * HW + p0) * CHANS + c;
    float s = 0.f, q = 0.f;
    for (int p = p0; p < p1; ++p) {
        float v = *base;
        base += CHANS;
        s += v;
        q += v * v;
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
        q += __shfl_down_sync(0xffffffffu, q, off);
    }
    if ((c & 7) == 0) {
        long long idx = (long long)(n * GROUPS + g) * nblk + blk;
        partial[idx] = s;
        partial[pstride + idx] = q;
    }
}

// ---------------------------------------------------------------------------
// stage 2: partials -> mean / rstd.  grid = B*G blocks, 256 threads.
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nblk, long long pstride,
                                   float inv_count, float eps) {
    __shared__ double ss[256];
    __shared__ double sq[256];
    const int bg = blockIdx.x;
    const int t  = threadIdx.x;

    double s = 0.0, q = 0.0;
    const float* ps = partial + (long long)bg * nblk;
    const float* pq = partial + pstride + (long long)bg * nblk;
    for (int i = t; i < nblk; i += 256) { s += (double)ps[i]; q += (double)pq[i]; }
    ss[t] = s; sq[t] = q;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) { ss[t] += ss[t + stride]; sq[t] += sq[t + stride]; }
        __syncthreads();
    }
    if (t == 0) {
        double m = ss[0] * (double)inv_count;
        double v = sq[0] * (double)inv_count - m * m;
        if (v < 0.0) v = 0.0;
        mean[bg] = (float)m;
        rstd[bg] = (float)(1.0 / sqrt(v + (double)eps));
    }
}

__device__ __forceinline__ float silu(float v) {
    return v / (1.0f + __expf(-v));
}

// ---------------------------------------------------------------------------
// stage 3a: normalize + affine + silu, NHWC -> NHWC (float4 vectorized)
// grid = (ceil(HW*64 / 256), B)
// ---------------------------------------------------------------------------
__global__ void gn_silu_apply_nhwc(const float* __restrict__ y,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ mean,
                                   const float* __restrict__ rstd,
                                   float* __restrict__ out,
                                   int HW) {
    const int n = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // float4 index
    const int total4 = HW * (CHANS / 4);
    if (idx >= total4) return;

    const int c4 = idx & 63;
    const int c  = c4 << 2;
    const int g  = c >> 3;

    const float m = mean[n * GROUPS + g];
    const float r = rstd[n * GROUPS + g];

    const float4* src = (const float4*)(y + (long long)n * HW * CHANS);
    float4* dst = (float4*)(out + (long long)n * HW * CHANS);
    const float4 gv = ((const float4*)gamma)[c4];
    const float4 bv = ((const float4*)beta)[c4];

    float4 v = src[idx];
    float4 o;
    o.x = silu((v.x - m) * r * gv.x + bv.x);
    o.y = silu((v.y - m) * r * gv.y + bv.y);
    o.z = silu((v.z - m) * r * gv.z + bv.z);
    o.w = silu((v.w - m) * r * gv.w + bv.w);
    dst[idx] = o;
}

// ---------------------------------------------------------------------------
// stage 3b: normalize + affine + silu + residual + NHWC->NCHW transpose
// grid = (ceil(HW/TP), B), block = 256
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_nchw(const float* __restrict__ y,
                                 const float* __restrict__ res,   // NCHW
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 const float* __restrict__ mean,
                                 const float* __restrict__ rstd,
                                 float* __restrict__ out,         // NCHW
                                 int HW) {
    __shared__ float sm[TP * SSTRIDE];
    __shared__ float smean[GROUPS];
    __shared__ float srstd[GROUPS];
    __shared__ float sg[CHANS];
    __shared__ float sb[CHANS];

    const int n  = blockIdx.y;
    const int p0 = blockIdx.x * TP;
    const int npos = min(TP, HW - p0);
    const int t = threadIdx.x;

    sg[t] = gamma[t];
    sb[t] = beta[t];
    if (t < GROUPS) {
        smean[t] = mean[n * GROUPS + t];
        srstd[t] = rstd[n * GROUPS + t];
    }
    __syncthreads();

    const float4* src = (const float4*)(y + ((long long)n * HW + p0) * CHANS);
    const int nElem4 = npos * (CHANS / 4);
    for (int i = t; i < nElem4; i += 256) {
        const int p  = i >> 6;
        const int c4 = i & 63;
        const int c  = c4 << 2;
        const int g  = c >> 3;
        const float m = smean[g];
        const float r = srstd[g];
        float4 v = src[i];
        float* d = &sm[p * SSTRIDE + c];
        d[0] = silu((v.x - m) * r * sg[c + 0] + sb[c + 0]);
        d[1] = silu((v.y - m) * r * sg[c + 1] + sb[c + 1]);
        d[2] = silu((v.z - m) * r * sg[c + 2] + sb[c + 2]);
        d[3] = silu((v.w - m) * r * sg[c + 3] + sb[c + 3]);
    }
    __syncthreads();

    const int warp = t >> 5;
    const int lane = t & 31;
    if (lane < npos) {
        const long long nbase = (long long)n * CHANS * HW + p0 + lane;
        #pragma unroll 4
        for (int k = 0; k < CHANS / 8; ++k) {
            const int c = k * 8 + warp;
            const long long off = nbase + (long long)c * HW;
            out[off] = sm[lane * SSTRIDE + c] + res[off];
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static inline int pick_nblk(int B, int HW) {
    int target = (512 + B - 1) / B;
    int maxb = (HW + 31) / 32;
    if (maxb < 1) maxb = 1;
    int nblk = target < maxb ? target : maxb;
    if (nblk < 1) nblk = 1;
    return nblk;
}

static void compute_stats(const torch::Tensor& y, int B, int HW, float eps,
                          torch::Tensor& mean, torch::Tensor& rstd,
                          cudaStream_t stream) {
    int nblk = pick_nblk(B, HW);
    auto opts = y.options();
    long long pstride = (long long)B * GROUPS * nblk;
    auto partial = torch::empty({2 * pstride}, opts);

    dim3 grid1(nblk, B);
    gn_stats_kernel<<<grid1, 256, 0, stream>>>(
        y.data_ptr<float>(), partial.data_ptr<float>(), HW, nblk, pstride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float inv_count = 1.0f / (float)((long long)CPG * HW);
    gn_finalize_kernel<<<B * GROUPS, 256, 0, stream>>>(
        partial.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        nblk, pstride, inv_count, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "float32 cuda expected");
    TORCH_CHECK(y.size(1) == CHANS, "channels must be 256");
    const int B = y.size(0), H = y.size(2), W = y.size(3);
    const int HW = H * W;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto fopts = y.options();
    auto mean = torch::empty({B * GROUPS}, fopts);
    auto rstd = torch::empty({B * GROUPS}, fopts);
    compute_stats(y, B, HW, (float)eps, mean, rstd, stream);

    auto out = torch::empty_like(y, fopts.memory_format(at::MemoryFormat::ChannelsLast));

    int total4 = HW * (CHANS / 4);
    dim3 grid((total4 + 255) / 256, B);
    gn_silu_apply_nhwc<<<grid, 256, 0, stream>>>(
        y.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_res(torch::Tensor y, torch::Tensor res, torch::Tensor gamma,
                          torch::Tensor beta, double eps) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "float32 cuda expected");
    TORCH_CHECK(y.size(1) == CHANS, "channels must be 256");
    const int B = y.size(0), H = y.size(2), W = y.size(3);
    const int HW = H * W;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto fopts = y.options();
    auto mean = torch::empty({B * GROUPS}, fopts);
    auto rstd = torch::empty({B * GROUPS}, fopts);
    compute_stats(y, B, HW, (float)eps, mean, rstd, stream);

    auto out = torch::empty({B, CHANS, H, W}, fopts);

    dim3 grid((HW + TP - 1) / TP, B);
    gn_silu_res_nchw<<<grid, 256, 0, stream>>>(
        y.data_ptr<float>(), res.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor gn_silu_res(torch::Tensor y, torch::Tensor res, torch::Tensor gamma,
                          torch::Tensor beta, double eps);
"""

_ext = load_inline(
    name="vae_resblock_gnsilu_v1",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu_nhwc", "gn_silu_res"],
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

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        cl = torch.channels_last
        xc = x if x.is_contiguous(memory_format=cl) else x.contiguous(memory_format=cl)
        w1 = conv1_weight if conv1_weight.is_contiguous(memory_format=cl) \
            else conv1_weight.contiguous(memory_format=cl)
        w2 = conv2_weight if conv2_weight.is_contiguous(memory_format=cl) \
            else conv2_weight.contiguous(memory_format=cl)

        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        res = x if x.is_contiguous() else x.contiguous()

        y1 = F.conv2d(xc, w1, bias=None, stride=1, padding=1)
        if not y1.is_contiguous(memory_format=cl):
            y1 = y1.contiguous(memory_format=cl)
        z1 = _ext.gn_silu_nhwc(y1, g1, b1, float(eps))
        y2 = F.conv2d(z1, w2, bias=None, stride=1, padding=1)
        if not y2.is_contiguous(memory_format=cl):
            y2 = y2.contiguous(memory_format=cl)
        return _ext.gn_silu_res(y2, res, g2, b2, float(eps))
