# ==========================================================================
# ModelNew — SOL problem 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER / PLAN (required):
# 1) Chosen granularity: (C) fuse many ops into one/few custom CUDA kernels.
#
# 2) Ops replaced by custom CUDA kernels:
#      - the NCHW->NHWC layout conversions that cuDNN was doing internally
#        (4 nchwToNhwc launches in the reference profile)  -> custom tiled
#        transpose kernel `nchw2nhwc_kernel`
#      - group_norm #1 (RowwiseMoments + ComputeFusedParams + elementwise)
#        + SiLU #1                                        -> `gn_partial_kernel`,
#                                                            `gn_stats_kernel`,
#                                                            `gn_silu_nhwc_kernel`
#      - group_norm #2 + SiLU #2 + residual add + NHWC->NCHW output layout
#                                                         -> `gn_partial_kernel`,
#                                                            `gn_stats_kernel`,
#                                                            `gn_silu_res_nchw_kernel`
#
# 3) Fusion map:
#      kernel A (nchw2nhwc)       : layout conversion of the input, once, instead
#                                   of cuDNN doing it per-conv.
#      kernel B (gn_partial)      : per-(image,group) sum / sum-of-squares over an
#                                   NHWC chunk, fully coalesced float4 loads.
#      kernel C (gn_stats)        : chunk reduction -> mean/rstd -> per-(n,c)
#                                   affine scale/shift (folds gamma/beta in).
#      kernel D (gn_silu_nhwc)    : normalize + affine + SiLU in one pass, NHWC in/out
#                                   (feeds conv2 directly in channels-last).
#      kernel E (gn_silu_res_nchw): normalize + affine + SiLU + residual add +
#                                   NHWC->NCHW transpose, all in one shared-memory
#                                   tiled pass (the transpose is free traffic-wise).
#
# 4) What stays in PyTorch and why:
#      - F.conv2d (x2): vendor cuDNN TF32 implicit-GEMM NHWC kernel is ~74% of the
#        reference time and already sits near the compute roofline (155 GFLOP /
#        ~105 TFLOPS ~= 1.5 ms); re-implementing it wins nothing. We instead feed it
#        channels-last activations AND channels-last weights so cuDNN performs zero
#        internal layout conversions.
#      - weight .contiguous(channels_last): tiny (2.4 MB) one-shot relayout, cheaper
#        than any custom path and keeps parameter parity with the reference inputs.
#      - a pure-PyTorch fallback is kept for shapes/dtypes outside the specialized
#        (C=256, G=32, fp32, cuda) fast path, for correctness safety.
# ==========================================================================

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
#include <algorithm>

#define CH  256
#define NG  32
#define CPG 8

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// A) NCHW -> NHWC tiled transpose (C == 256, multiple of 32)
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 long long S)
{
    __shared__ float tile[32][33];   // [c_local][s_local]
    const int s0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const float* inp = in  + (long long)n * CH * S;
    float*       outp = out + (long long)n * S * CH;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int cc = ty + k * 8;
        int ss = tx;
        if (s0 + ss < S)
            tile[cc][ss] = inp[(long long)(c0 + cc) * S + (s0 + ss)];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int cc = tx;
        int ss = ty + k * 8;
        if (s0 + ss < S)
            outp[(long long)(s0 + ss) * CH + (c0 + cc)] = tile[cc][ss];
    }
}

// ---------------------------------------------------------------------------
// B) per-(image, group) partial sums over NHWC chunks
//    thread tid always covers group ((tid & 63) >> 1)
// ---------------------------------------------------------------------------
__global__ void gn_partial_kernel(const float* __restrict__ x,
                                  float* __restrict__ psum,
                                  float* __restrict__ psq,
                                  long long S, long long chunkSize, int numChunks)
{
    const int chunk = blockIdx.x;
    const int n     = blockIdx.y;
    const long long s0 = (long long)chunk * chunkSize;
    long long s1 = s0 + chunkSize;
    if (s1 > S) s1 = S;
    const long long len = s1 - s0;

    const float4* xp = reinterpret_cast<const float4*>(x) + (long long)(n * S + s0) * 64;
    const long long total = len * 64;

    float sum = 0.f, sq = 0.f;
    for (long long i = threadIdx.x; i < total; i += 256) {
        float4 v = xp[i];
        sum += v.x + v.y + v.z + v.w;
        sq  += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    __shared__ float sm[256];
    __shared__ float sq_[256];
    sm[threadIdx.x]  = sum;
    sq_[threadIdx.x] = sq;
    __syncthreads();

    if (threadIdx.x < NG) {
        const int g = threadIdx.x;
        float a = 0.f, b = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int base = j * 64 + 2 * g;
            a += sm[base]  + sm[base + 1];
            b += sq_[base] + sq_[base + 1];
        }
        psum[(long long)(n * NG + g) * numChunks + chunk] = a;
        psq [(long long)(n * NG + g) * numChunks + chunk] = b;
    }
}

// ---------------------------------------------------------------------------
// C) chunk reduction -> mean/rstd -> per-(n,c) scale/shift
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ psum,
                                const float* __restrict__ psq,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                float* __restrict__ scale,
                                float* __restrict__ shift,
                                int numChunks, double invCount, double eps)
{
    const int idx = blockIdx.x;              // n * NG + g
    const float* ps = psum + (long long)idx * numChunks;
    const float* pq = psq  + (long long)idx * numChunks;

    double s = 0.0, q = 0.0;
    for (int i = threadIdx.x; i < numChunks; i += blockDim.x) {
        s += (double)ps[i];
        q += (double)pq[i];
    }
    __shared__ double rs[128];
    __shared__ double rq[128];
    rs[threadIdx.x] = s;
    rq[threadIdx.x] = q;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            rs[threadIdx.x] += rs[threadIdx.x + off];
            rq[threadIdx.x] += rq[threadIdx.x + off];
        }
        __syncthreads();
    }
    __shared__ float mean_s, rstd_s;
    if (threadIdx.x == 0) {
        double m = rs[0] * invCount;
        double v = rq[0] * invCount - m * m;
        if (v < 0.0) v = 0.0;
        mean_s = (float)m;
        rstd_s = (float)(1.0 / sqrt(v + eps));
    }
    __syncthreads();

    if (threadIdx.x < CPG) {
        const int n = idx / NG;
        const int g = idx - n * NG;
        const int c = g * CPG + (int)threadIdx.x;
        float sc = rstd_s * gamma[c];
        scale[n * CH + c] = sc;
        shift[n * CH + c] = beta[c] - mean_s * sc;
    }
}

// ---------------------------------------------------------------------------
// D) normalize + affine + SiLU, NHWC in / NHWC out
// ---------------------------------------------------------------------------
__global__ void gn_silu_nhwc_kernel(const float4* __restrict__ in,
                                    float4* __restrict__ out,
                                    const float4* __restrict__ scale,
                                    const float4* __restrict__ shift,
                                    long long perImage)
{
    const int n = blockIdx.y;
    const float4* ip = in  + (long long)n * perImage;
    float4*       op = out + (long long)n * perImage;

    const long long start = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int c4 = (int)(start & 63);
    const float4 sc = scale[n * 64 + c4];
    const float4 sh = shift[n * 64 + c4];

    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = start; i < perImage; i += stride) {
        float4 v = ip[i];
        float4 r;
        r.x = silu_f(v.x * sc.x + sh.x);
        r.y = silu_f(v.y * sc.y + sh.y);
        r.z = silu_f(v.z * sc.z + sh.z);
        r.w = silu_f(v.w * sc.w + sh.w);
        op[i] = r;
    }
}

// ---------------------------------------------------------------------------
// E) normalize + affine + SiLU + residual + NHWC->NCHW transpose
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_nchw_kernel(const float* __restrict__ in,   // NHWC
                                        const float* __restrict__ scale,
                                        const float* __restrict__ shift,
                                        const float* __restrict__ res,  // NCHW
                                        float* __restrict__ out,        // NCHW
                                        long long S)
{
    __shared__ float tile[32][33];   // [s_local][c_local]
    const int s0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const float* ip = in + (long long)n * S * CH;
    const float* sc = scale + n * CH;
    const float* sh = shift + n * CH;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int cc = tx;
        int ss = ty + k * 8;
        if (s0 + ss < S) {
            float v = ip[(long long)(s0 + ss) * CH + (c0 + cc)];
            v = v * sc[c0 + cc] + sh[c0 + cc];
            tile[ss][cc] = silu_f(v);
        }
    }
    __syncthreads();
    const long long nbase = (long long)n * CH * S;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int cc = ty + k * 8;
        int ss = tx;
        if (s0 + ss < S) {
            long long o = nbase + (long long)(c0 + cc) * S + (s0 + ss);
            out[o] = tile[ss][cc] + res[o];
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static void compute_scale_shift(const torch::Tensor& y,
                                const torch::Tensor& gamma,
                                const torch::Tensor& beta,
                                double eps,
                                torch::Tensor& scale,
                                torch::Tensor& shift)
{
    const long long N = y.size(0);
    const long long H = y.size(2);
    const long long W = y.size(3);
    const long long S = H * W;

    int desiredBlocks = 680;
    int chunksPerImage = (int)((desiredBlocks + N - 1) / N);
    if (chunksPerImage < 1) chunksPerImage = 1;
    long long chunkSize = (S + chunksPerImage - 1) / chunksPerImage;
    if (chunkSize < 8) chunkSize = 8;
    if (chunkSize > S) chunkSize = S;
    int numChunks = (int)((S + chunkSize - 1) / chunkSize);

    auto fopt = y.options();
    auto psum = torch::empty({N * NG * numChunks}, fopt);
    auto psq  = torch::empty({N * NG * numChunks}, fopt);

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 gridP(numChunks, (unsigned)N);
    gn_partial_kernel<<<gridP, 256, 0, stream>>>(
        y.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
        S, chunkSize, numChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    double invCount = 1.0 / (double)(S * CPG);
    gn_stats_kernel<<<(unsigned)(N * NG), 128, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        numChunks, invCount, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor nchw_to_nhwc(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda required");
    TORCH_CHECK(x.dim() == 4 && x.size(1) == CH, "expect NCHW with C=256");
    TORCH_CHECK(x.is_contiguous(), "expect contiguous NCHW");
    const long long N = x.size(0), H = x.size(2), W = x.size(3);
    const long long S = H * W;
    auto out = torch::empty({N, CH, H, W},
                            x.options().memory_format(at::MemoryFormat::ChannelsLast));
    dim3 grid((unsigned)((S + 31) / 32), CH / 32, (unsigned)N);
    dim3 block(32, 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    nchw2nhwc_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta, double eps)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda required");
    TORCH_CHECK(y.dim() == 4 && y.size(1) == CH, "expect C=256");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "expect channels_last");
    const long long N = y.size(0), H = y.size(2), W = y.size(3);
    const long long S = H * W;

    auto scale = torch::empty({N * CH}, y.options());
    auto shift = torch::empty({N * CH}, y.options());
    compute_scale_shift(y, gamma, beta, eps, scale, shift);

    auto out = torch::empty({N, CH, H, W},
                            y.options().memory_format(at::MemoryFormat::ChannelsLast));

    const long long perImage = S * (CH / 4);
    long long gx = (perImage + 255) / 256;
    if (gx > 2048) gx = 2048;
    if (gx < 1) gx = 1;
    dim3 grid((unsigned)gx, (unsigned)N);
    auto stream = at::cuda::getCurrentCUDAStream();
    gn_silu_nhwc_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(y.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        reinterpret_cast<const float4*>(scale.data_ptr<float>()),
        reinterpret_cast<const float4*>(shift.data_ptr<float>()),
        perImage);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_res_nchw(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                               double eps, torch::Tensor res)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda required");
    TORCH_CHECK(y.dim() == 4 && y.size(1) == CH, "expect C=256");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "expect channels_last");
    TORCH_CHECK(res.is_contiguous() && res.sizes() == y.sizes(), "residual must be contiguous NCHW");
    const long long N = y.size(0), H = y.size(2), W = y.size(3);
    const long long S = H * W;

    auto scale = torch::empty({N * CH}, y.options());
    auto shift = torch::empty({N * CH}, y.options());
    compute_scale_shift(y, gamma, beta, eps, scale, shift);

    auto out = torch::empty({N, CH, H, W}, y.options());

    dim3 grid((unsigned)((S + 31) / 32), CH / 32, (unsigned)N);
    dim3 block(32, 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    gn_silu_res_nchw_kernel<<<grid, block, 0, stream>>>(
        y.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
        res.data_ptr<float>(), out.data_ptr<float>(), S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_cpp_src = r"""
torch::Tensor nchw_to_nhwc(torch::Tensor x);
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor gn_silu_res_nchw(torch::Tensor y, torch::Tensor gamma, torch::Tensor beta,
                               double eps, torch::Tensor res);
"""

_ext = load_inline(
    name="vae_resblock_fused_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["nchw_to_nhwc", "gn_silu_nhwc", "gn_silu_res_nchw"],
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
    """See module header comment for the granularity/fusion plan (level C)."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        num_groups = 32
        fast = (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.size(1) == 256
            and conv1_weight.dtype == torch.float32
            and conv2_weight.dtype == torch.float32
            and conv1_weight.shape == (256, 256, 3, 3)
            and conv2_weight.shape == (256, 256, 3, 3)
            and norm1_weight.numel() == 256
            and norm2_weight.numel() == 256
        )
        if not fast:
            residual = x
            out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
            out = F.silu(out)
            out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
            out = F.silu(out)
            return out + residual

        xc = x if x.is_contiguous() else x.contiguous()
        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        # single NCHW->NHWC conversion; everything downstream stays channels_last
        x_nhwc = self.ext.nchw_to_nhwc(xc)

        w1 = conv1_weight.contiguous(memory_format=torch.channels_last)
        y = F.conv2d(x_nhwc, w1, None, 1, 1)          # vendor cuDNN NHWC TF32 GEMM
        y = self.ext.gn_silu_nhwc(y, g1, b1, float(eps))

        w2 = conv2_weight.contiguous(memory_format=torch.channels_last)
        y = F.conv2d(y, w2, None, 1, 1)               # vendor cuDNN NHWC TF32 GEMM
        out = self.ext.gn_silu_res_nchw(y, g2, b2, float(eps), xc)
        return out
