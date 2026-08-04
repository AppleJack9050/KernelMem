# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY: (C) fuse many ops into one/few kernels.
#
# 1) Chosen granularity: (C). Everything except the two 3x3 convolutions is
#    collapsed into a small set of hand-written fused CUDA kernels; the convs
#    stay in the vendor (cuDNN) implementation but are fed/consumed in the
#    layout the vendor kernel actually wants (NHWC), so the library's own
#    nchwToNhwc / nhwcToNchw transposes (20.8% of reference GPU time) vanish.
#
# 2) Ops replaced by custom CUDA:
#    - group_norm #1 and #2  (RowwiseMomentsCUDAKernel + ComputeFusedParams +
#      elementwise_kernel)
#    - silu #1 and #2        (vectorized_elementwise_kernel)
#    - residual add          (vectorized_elementwise_kernel)
#    - both activation layout conversions (NCHW<->NHWC) performed by cuDNN
#
# 3) Fusion map (4 custom kernels + 1 tiny reduction kernel):
#    K0 nchw2nhwc_kernel      : shared-memory 32x32 tiled transpose of x into
#                               channels_last, once, so conv1 + conv2 run pure
#                               NHWC TF32 implicit-GEMM with zero library
#                               transposes around them.
#    K1 gn_stats_kernel       : deterministic split reduction (sum, sum of
#                               squares) over each (n, group) of an NHWC
#                               tensor; fully coalesced 1024B-per-warp reads.
#    K2 gn_finalize_kernel    : one block per (n,group); reduces the partials
#                               in fixed order (double accumulators), turns
#                               them into per-(n,channel) affine coefficients
#                               a = rstd*gamma, b = beta - mean*rstd*gamma
#                               (i.e. GroupNorm + weight/bias folded).
#    K3 gn_silu_nhwc_kernel   : GroupNorm-apply + SiLU fused, float4
#                               vectorized, NHWC in -> NHWC out (feeds conv2).
#    K4 gn_silu_add_nchw_kernel: GroupNorm-apply + SiLU + residual add +
#                               NHWC->NCHW transpose fused into ONE pass over
#                               the tensor (shared-memory tile keeps both the
#                               NHWC read and the NCHW write coalesced, and the
#                               residual is read coalesced in NCHW at store
#                               time). This single kernel replaces 3 ATen
#                               elementwise kernels + cuDNN's output transpose.
#    K1/K2 are a fission of one GroupNorm (global reduction cannot be fused
#    into the same pass as its consumer), everything else is fused.
#
# 4) What stays in PyTorch/vendor and why:
#    - at::conv2d (cuDNN NHWC TF32 implicit GEMM): tensor-core conv already at
#      ~roofline; called from inside the extension, not from the reference.
#    - weight .contiguous(ChannelsLast): a 2.4MB one-off layout copy per conv,
#      cheaper than letting cuDNN redo it inside the timed region.
#    - a pure-PyTorch fallback path exists only for shapes outside the
#      definition's constants (C==256, G==32, fp32); never taken by the bench.
#
# Precision: all storage and arithmetic in float32 (input dtype); reductions
# accumulate in float32 per-thread and double across chunks. TF32 conv matches
# the reference (cudnn.allow_tf32 default True). Nothing is downcast.
# Determinism: no atomics anywhere; every reduction has a fixed tree order.
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define C_FIX   256
#define G_FIX   32
#define CPG_FIX 8

__device__ __forceinline__ float silu_f(float t) {
    return t / (1.0f + __expf(-t));
}

// ---------------------------------------------------------------------------
// K0: NCHW -> NHWC, 32x32 shared-memory tiled transpose (coalesced both ways)
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int C, int HW) {
    __shared__ float tile[32][33];
    const int hw0 = blockIdx.x * 32;
    const int c0  = blockIdx.y * 32;
    const int n   = blockIdx.z;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;

    const float* inb  = in  + (size_t)n * (size_t)C * (size_t)HW;
    float*       outb = out + (size_t)n * (size_t)HW * (size_t)C;

    const int p = hw0 + tx;
    for (int j = ty; j < 32; j += 8) {
        float v = 0.0f;
        if (p < HW) v = inb[(size_t)(c0 + j) * (size_t)HW + (size_t)p];
        tile[j][tx] = v;
    }
    __syncthreads();
    for (int i = ty; i < 32; i += 8) {
        const int pp = hw0 + i;
        if (pp < HW) outb[(size_t)pp * (size_t)C + (size_t)(c0 + tx)] = tile[tx][i];
    }
}

// ---------------------------------------------------------------------------
// K1: per-(n,group) partial sum / sum-of-squares over an NHWC tensor.
//     blockDim.x == C_FIX : each thread owns one channel, so a warp reads
//     128 contiguous bytes -> perfectly coalesced 1KB rows.
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                int HW, int nchunk, int chunk) {
    __shared__ float ss[C_FIX];
    __shared__ float sq[C_FIX];

    const int n   = blockIdx.y;
    const int ci  = blockIdx.x;
    const int tid = threadIdx.x;

    const int p0 = ci * chunk;
    int p1 = p0 + chunk;
    if (p1 > HW) p1 = HW;

    float s = 0.0f, q = 0.0f;
    if (p0 < p1) {
        const float* base = y + (size_t)n * (size_t)HW * (size_t)C_FIX
                              + (size_t)p0 * (size_t)C_FIX + (size_t)tid;
        for (int p = p0; p < p1; ++p) {
            const float v = *base;
            base += C_FIX;
            s += v;
            q += v * v;
        }
    }
    ss[tid] = s;
    sq[tid] = q;
    __syncthreads();

    if (tid < G_FIX) {
        float a = 0.0f, b = 0.0f;
#pragma unroll
        for (int j = 0; j < CPG_FIX; ++j) {
            a += ss[tid * CPG_FIX + j];
            b += sq[tid * CPG_FIX + j];
        }
        const int o = (n * G_FIX + tid) * nchunk + ci;
        psum[o] = a;
        psq[o]  = b;
    }
}

// ---------------------------------------------------------------------------
// K2: reduce partials (fixed order, double acc) -> per-(n,channel) affine
//     coefficients with gamma/beta folded in.
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ psum,
                                   const float* __restrict__ psq,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ A,
                                   float* __restrict__ B,
                                   int nchunk, double invM, double eps) {
    __shared__ double rs[128];
    __shared__ double rq[128];
    __shared__ float  s_rstd;
    __shared__ float  s_mean;

    const int idx = blockIdx.x;      // n * G + g
    const int tid = threadIdx.x;

    const float* ps = psum + (size_t)idx * (size_t)nchunk;
    const float* pq = psq  + (size_t)idx * (size_t)nchunk;

    double s = 0.0, q = 0.0;
    for (int i = tid; i < nchunk; i += 128) {
        s += (double)ps[i];
        q += (double)pq[i];
    }
    rs[tid] = s;
    rq[tid] = q;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (tid < off) {
            rs[tid] += rs[tid + off];
            rq[tid] += rq[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const double mean = rs[0] * invM;
        double var = rq[0] * invM - mean * mean;
        if (var < 0.0) var = 0.0;
        s_rstd = (float)(1.0 / sqrt(var + eps));
        s_mean = (float)mean;
    }
    __syncthreads();

    if (tid < CPG_FIX) {
        const int n = idx / G_FIX;
        const int g = idx - n * G_FIX;
        const int c = g * CPG_FIX + tid;
        const float a = s_rstd * gamma[c];
        A[n * C_FIX + c] = a;
        B[n * C_FIX + c] = beta[c] - s_mean * a;
    }
}

// ---------------------------------------------------------------------------
// K3: GroupNorm-apply + SiLU, NHWC -> NHWC, float4 vectorized along C.
// ---------------------------------------------------------------------------
__global__ void gn_silu_nhwc_kernel(const float4* __restrict__ y,
                                    float4* __restrict__ out,
                                    const float4* __restrict__ A,
                                    const float4* __restrict__ B,
                                    long total4) {
    const int n = blockIdx.y;
    const float4* Ab = A + (size_t)n * (C_FIX / 4);
    const float4* Bb = B + (size_t)n * (C_FIX / 4);
    const float4* yb = y   + (size_t)n * (size_t)total4;
    float4*       ob = out + (size_t)n * (size_t)total4;

    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total4; i += stride) {
        const int c4 = (int)(i & (long)(C_FIX / 4 - 1));
        const float4 v = yb[i];
        const float4 a = Ab[c4];
        const float4 b = Bb[c4];
        float4 o;
        o.x = silu_f(v.x * a.x + b.x);
        o.y = silu_f(v.y * a.y + b.y);
        o.z = silu_f(v.z * a.z + b.z);
        o.w = silu_f(v.w * a.w + b.w);
        ob[i] = o;
    }
}

// ---------------------------------------------------------------------------
// K4: GroupNorm-apply + SiLU + residual add + NHWC->NCHW transpose, one pass.
// ---------------------------------------------------------------------------
__global__ void gn_silu_add_nchw_kernel(const float* __restrict__ y,
                                        const float* __restrict__ res,
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ out,
                                        int C, int HW) {
    __shared__ float tile[32][33];
    const int hw0 = blockIdx.x * 32;
    const int c0  = blockIdx.y * 32;
    const int n   = blockIdx.z;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;

    const float* yb = y + (size_t)n * (size_t)HW * (size_t)C;
    const float* Ab = A + (size_t)n * (size_t)C;
    const float* Bb = B + (size_t)n * (size_t)C;

    // ---- load phase: NHWC coalesced along C (tx indexes channel) ----
    const int c_ld = c0 + tx;
    const float a = Ab[c_ld];
    const float b = Bb[c_ld];
    for (int i = ty; i < 32; i += 8) {
        const int p = hw0 + i;
        float t = 0.0f;
        if (p < HW) {
            const float v = yb[(size_t)p * (size_t)C + (size_t)c_ld];
            t = silu_f(v * a + b);
        }
        tile[i][tx] = t;
    }
    __syncthreads();

    // ---- store phase: NCHW coalesced along HW (tx indexes pixel) ----
    const int p = hw0 + tx;
    if (p < HW) {
        const float* resb = res + (size_t)n * (size_t)C * (size_t)HW;
        float*       ob   = out + (size_t)n * (size_t)C * (size_t)HW;
        for (int j = ty; j < 32; j += 8) {
            const size_t o = (size_t)(c0 + j) * (size_t)HW + (size_t)p;
            ob[o] = tile[tx][j] + resb[o];
        }
    }
}

// ---------------------------------------------------------------------------
// host driver
// ---------------------------------------------------------------------------
static inline void gn_stats_launch(const at::Tensor& y_nhwc, at::Tensor& psum,
                                   at::Tensor& psq, int N, int HW,
                                   int nchunk, int chunk, cudaStream_t stream) {
    dim3 grid(nchunk, N);
    gn_stats_kernel<<<grid, C_FIX, 0, stream>>>(
        y_nhwc.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
        HW, nchunk, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                          torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                          double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "fp32 only");

    const int N  = (int)x.size(0);
    const int C  = (int)x.size(1);
    const int H  = (int)x.size(2);
    const int W  = (int)x.size(3);
    TORCH_CHECK(C == C_FIX, "this kernel is specialised for C=256");
    const int HW = H * W;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto xc = x.is_contiguous() ? x : x.contiguous();

    auto opts   = x.options();
    auto cl     = opts.memory_format(at::MemoryFormat::ChannelsLast);

    // ---- K0: x -> NHWC ----
    auto xn = torch::empty({N, C, H, W}, cl);
    {
        dim3 tb(32, 8);
        dim3 gr((HW + 31) / 32, C / 32, N);
        nchw2nhwc_kernel<<<gr, tb, 0, stream>>>(
            xc.data_ptr<float>(), xn.data_ptr<float>(), C, HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1c = g1.is_contiguous() ? g1 : g1.contiguous();
    auto b1c = b1.is_contiguous() ? b1 : b1.contiguous();
    auto g2c = g2.is_contiguous() ? g2 : g2.contiguous();
    auto b2c = b2.is_contiguous() ? b2 : b2.contiguous();

    // split-reduction geometry (>=64 pixels per block, ~912 blocks target)
    int nchunk = (HW + 63) / 64;
    int maxc   = (912 + N - 1) / N;
    if (maxc < 1) maxc = 1;
    if (nchunk > maxc) nchunk = maxc;
    if (nchunk < 1) nchunk = 1;
    const int chunk = (HW + nchunk - 1) / nchunk;
    const double invM = 1.0 / ((double)CPG_FIX * (double)HW);

    auto psum = torch::empty({(long)N * G_FIX * nchunk}, opts);
    auto psq  = torch::empty({(long)N * G_FIX * nchunk}, opts);
    auto A    = torch::empty({(long)N * C}, opts);
    auto B    = torch::empty({(long)N * C}, opts);

    // ---- conv1 (cuDNN, NHWC TF32) ----
    auto y1 = at::conv2d(xn, w1c, torch::Tensor(), {1, 1}, {1, 1}, {1, 1}, 1);
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- K1/K2: stats + folded affine params ----
    gn_stats_launch(y1, psum, psq, N, HW, nchunk, chunk, stream);
    gn_finalize_kernel<<<N * G_FIX, 128, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        g1c.data_ptr<float>(), b1c.data_ptr<float>(),
        A.data_ptr<float>(), B.data_ptr<float>(),
        nchunk, invM, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ---- K3: GN + SiLU (NHWC -> NHWC) ----
    auto y1n = torch::empty({N, C, H, W}, cl);
    {
        const long total4 = (long)HW * (C / 4);
        const int threads = 256;
        long nb = (total4 + threads - 1) / threads;
        if (nb > 65535) nb = 65535;
        dim3 gr((unsigned)nb, N);
        gn_silu_nhwc_kernel<<<gr, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(y1.data_ptr<float>()),
            reinterpret_cast<float4*>(y1n.data_ptr<float>()),
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<const float4*>(B.data_ptr<float>()),
            total4);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (cuDNN, NHWC TF32) ----
    auto y2 = at::conv2d(y1n, w2c, torch::Tensor(), {1, 1}, {1, 1}, {1, 1}, 1);
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    gn_stats_launch(y2, psum, psq, N, HW, nchunk, chunk, stream);
    gn_finalize_kernel<<<N * G_FIX, 128, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        g2c.data_ptr<float>(), b2c.data_ptr<float>(),
        A.data_ptr<float>(), B.data_ptr<float>(),
        nchunk, invM, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ---- K4: GN + SiLU + residual + transpose back to NCHW ----
    auto out = torch::empty({N, C, H, W}, opts);
    {
        dim3 tb(32, 8);
        dim3 gr((HW + 31) / 32, C / 32, N);
        gn_silu_add_nchw_kernel<<<gr, tb, 0, stream>>>(
            y2.data_ptr<float>(), xc.data_ptr<float>(),
            A.data_ptr<float>(), B.data_ptr<float>(),
            out.data_ptr<float>(), C, HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                          torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                          double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_c",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["fused_block"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-gencode=arch=compute_90,code=sm_90",
    ],
)

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


class ModelNew(nn.Module):
    """Fused VAE residual block (granularity C). See module header comment."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        # Fast fused path: fp32, CUDA, C == 256 (the definition's constant).
        if (x.is_cuda and x.dim() == 4 and x.dtype == torch.float32
                and x.size(1) == 256):
            return self.ext.fused_block(
                x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, float(eps))

        # Defensive fallback (never taken by this benchmark's workloads).
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=eps)
        out = F.silu(out)
        return out + residual
