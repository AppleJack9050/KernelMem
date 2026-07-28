# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# ROUND N: stream_pipeline_overlap
#
# Granularity: (D) FULL FORWARD REWRITE (unchanged from base).
# The entire reference run() body (2x [conv3x3 -> GroupNorm(32) -> SiLU] +
# residual) executes inside the extension entry point `fused_block`.
#
# THIS ROUND'S CHANGE (execution overlap only — no kernel body edits):
#   The block body was extracted into `run_chunk(...)`, parameterised by an
#   input batch-slice, an output batch-slice and a CUDA stream.  `fused_block`
#   now splits the batch into 2 independent chunks (when B >= 4, B*H*W >=
#   65536 and B is even) and runs them on two streams with event fork/join.
#   GroupNorm reduces per (sample, group) and conv is per-sample, so the split
#   is mathematically exact and bitwise identical.  The point is to let the
#   DRAM-bound GN/transpose kernels of one chunk co-reside with the
#   compute-bound tf32 cuDNN convs of the other (gn_stats: 8.1% SM / 81.9%
#   DRAM; convs: ~19% DRAM).
#
# Kernels (unchanged):
#   K1 nchw2nhwc_kernel      : NCHW->NHWC transpose of x (32x33 shared tile).
#   K2 gn_stats_kernel       : per-(batch,chunk) partial sum/sumsq, float4.
#   K3 gn_finalize_kernel    : deterministic tree reduce -> mean/rstd.
#   K4 gn_silu_nhwc_kernel   : GN affine + SiLU, NHWC->NHWC, float4.
#   K5 gn_silu_add_t_kernel  : GN affine + SiLU + residual + NHWC->NCHW.
#   at::conv2d (cuDNN sm90 NHWC TF32) owns the two 3x3 convs.
#
# PRECISION: float32 storage + arithmetic everywhere; reductions in float32;
# TF32 tensor cores only inside conv, exactly as the reference does.
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CH   256          // channels (const per problem definition)
#define NG   32           // num groups
#define CPG  (CH/NG)      // 8 channels per group
#define CHUNK 128         // pixels per stats block

// ---------------------------------------------------------------------------
// K1 : NCHW -> NHWC transpose (per batch, C x N  ->  N x C)
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 int N)
{
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int b  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const float* s = src + (size_t)b * CH * (size_t)N;
    float*       d = dst + (size_t)b * (size_t)N * CH;

    const int col = p0 + tx;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = c0 + ty + 8 * i;              // channel
        float v = 0.f;
        if (col < N) v = s[(size_t)row * N + col];
        tile[ty + 8 * i][tx] = v;                     // [c_local][p_local]
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int p = p0 + ty + 8 * i;
        if (p < N) d[(size_t)p * CH + c0 + tx] = tile[tx][ty + 8 * i];
    }
}

// ---------------------------------------------------------------------------
// K2 : per (batch, chunk) partial sums / sums of squares for the 32 groups
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                float* __restrict__ partials,
                                int N, int nchunks)
{
    const int chunk = blockIdx.x;
    const int b     = blockIdx.y;
    const int t     = threadIdx.x;        // 0..255
    const int v     = t & 63;             // float4 index inside a pixel row
    const int prow  = t >> 6;             // 0..3

    const int p0   = chunk * CHUNK;
    int pend = p0 + CHUNK; if (pend > N) pend = N;

    const float4* base =
        reinterpret_cast<const float4*>(y + (size_t)b * (size_t)N * CH);

    float s = 0.f, sq = 0.f;
    for (int p = p0 + prow; p < pend; p += 4) {
        float4 val = base[(size_t)p * 64 + v];
        s  += val.x + val.y + val.z + val.w;
        sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    __shared__ float ss[256];
    __shared__ float sqq[256];
    ss[t]  = s;
    sqq[t] = sq;
    __syncthreads();

    if (t < NG) {
        float gs = 0.f, gq = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int i0 = k * 64 + 2 * t;   // threads whose v/2 == t
            gs += ss[i0]  + ss[i0 + 1];
            gq += sqq[i0] + sqq[i0 + 1];
        }
        float* o = partials + ((size_t)b * nchunks + chunk) * (2 * NG);
        o[2 * t]     = gs;
        o[2 * t + 1] = gq;
    }
}

// ---------------------------------------------------------------------------
// K3 : reduce partials -> mean / rstd   (one block per (b,g))
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ partials,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nchunks, float invcount, float eps)
{
    const int bg = blockIdx.x;
    const int b  = bg / NG;
    const int g  = bg - b * NG;
    const int t  = threadIdx.x;

    float s = 0.f, q = 0.f;
    for (int c = t; c < nchunks; c += blockDim.x) {
        const float* p = partials + ((size_t)b * nchunks + c) * (2 * NG) + 2 * g;
        s += p[0];
        q += p[1];
    }
    __shared__ float ss[128];
    __shared__ float sq[128];
    ss[t] = s; sq[t] = q;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (t < stride) { ss[t] += ss[t + stride]; sq[t] += sq[t + stride]; }
        __syncthreads();
    }
    if (t == 0) {
        float m   = ss[0] * invcount;
        float var = sq[0] * invcount - m * m;
        if (!(var > 0.f)) var = 0.f;
        mean[bg] = m;
        rstd[bg] = rsqrtf(var + eps);
    }
}

// ---------------------------------------------------------------------------
// K4 : GroupNorm affine + SiLU, NHWC -> NHWC, float4 vectorised
// ---------------------------------------------------------------------------
__global__ void gn_silu_nhwc_kernel(const float4* __restrict__ y,
                                    float4* __restrict__ out,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ rstd,
                                    const float4* __restrict__ gamma,
                                    const float4* __restrict__ beta,
                                    long total /* = N*64 float4 per batch */)
{
    const int b = blockIdx.y;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int v = (int)(idx & 63);
    const int g = v >> 1;                    // 4 channels/thread -> one group
    const float m = mean[b * NG + g];
    const float r = rstd[b * NG + g];
    const float4 gm = gamma[v];
    const float4 bt = beta[v];

    const long off = (long)b * total + idx;
    float4 val = y[off];

    float a0 = (val.x - m) * r * gm.x + bt.x;
    float a1 = (val.y - m) * r * gm.y + bt.y;
    float a2 = (val.z - m) * r * gm.z + bt.z;
    float a3 = (val.w - m) * r * gm.w + bt.w;

    float4 o;
    o.x = a0 / (1.f + expf(-a0));
    o.y = a1 / (1.f + expf(-a1));
    o.z = a2 / (1.f + expf(-a2));
    o.w = a3 / (1.f + expf(-a3));
    out[off] = o;
}

// ---------------------------------------------------------------------------
// K5 : GroupNorm affine + SiLU + residual add + NHWC->NCHW transpose
// ---------------------------------------------------------------------------
__global__ void gn_silu_add_t_kernel(const float* __restrict__ y,     // NHWC
                                     const float* __restrict__ res,   // NCHW
                                     const float* __restrict__ mean,
                                     const float* __restrict__ rstd,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ out,         // NCHW
                                     int N)
{
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int b  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int c = c0 + tx;
    const int g = c >> 3;                        // CPG == 8
    const float m  = mean[b * NG + g];
    const float r  = rstd[b * NG + g];
    const float gm = gamma[c];
    const float bt = beta[c];

    const float* yb = y + (size_t)b * (size_t)N * CH;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int p = p0 + ty + 8 * i;
        float v = 0.f;
        if (p < N) {
            float t0 = yb[(size_t)p * CH + c];
            float a  = (t0 - m) * r * gm + bt;
            v = a / (1.f + expf(-a));
        }
        tile[ty + 8 * i][tx] = v;                // [p_local][c_local]
    }
    __syncthreads();

    const float* rb = res + (size_t)b * CH * (size_t)N;
    float*       ob = out + (size_t)b * CH * (size_t)N;
    const int p = p0 + tx;
    if (p < N) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int cc = c0 + ty + 8 * i;
            ob[(size_t)cc * N + p] = tile[tx][ty + 8 * i] + rb[(size_t)cc * N + p];
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static inline at::Tensor to_cl(const at::Tensor& t) {
    return t.is_contiguous(at::MemoryFormat::ChannelsLast)
           ? t : t.contiguous(at::MemoryFormat::ChannelsLast);
}

static void compute_stats(const at::Tensor& y, int64_t B, int64_t N,
                          at::Tensor& mean, at::Tensor& rstd, float eps,
                          cudaStream_t stream)
{
    const int nchunks = (int)((N + CHUNK - 1) / CHUNK);
    auto partials = at::empty({B * (int64_t)nchunks * (2 * NG)}, y.options());

    dim3 grd((unsigned)nchunks, (unsigned)B, 1);
    gn_stats_kernel<<<grd, 256, 0, stream>>>(
        y.data_ptr<float>(), partials.data_ptr<float>(), (int)N, nchunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float invcount = 1.0f / (float)((double)CPG * (double)N);
    gn_finalize_kernel<<<(unsigned)(B * NG), 128, 0, stream>>>(
        partials.data_ptr<float>(), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), nchunks, invcount, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// PLAN ITEM 1 : the whole block body, per batch-chunk, on one stream.
// Body is byte-for-byte the previous fused_block core; only B, the stream and
// the destination pointer (out_slice) are parameterised.
// ---------------------------------------------------------------------------
static void run_chunk(const at::Tensor& xc,        // NCHW contiguous slice
                      const at::Tensor& w1,        // channels-last
                      const at::Tensor& g1,
                      const at::Tensor& b1,
                      const at::Tensor& w2,        // channels-last
                      const at::Tensor& g2,
                      const at::Tensor& b2,
                      at::Tensor out_slice,        // NCHW contiguous slice
                      float eps,
                      at::cuda::CUDAStream cstream)
{
    // PLAN ITEM 4/6 : guard makes at::conv2d + the caching allocator use this
    // stream for every temporary created below.
    at::cuda::CUDAStreamGuard guard(cstream);
    cudaStream_t stream = cstream.stream();

    const int64_t B = xc.size(0);
    const int64_t C = xc.size(1), H = xc.size(2), W = xc.size(3);
    const int64_t N = H * W;
    auto opts = xc.options();

    // ---- K1 : x (NCHW) -> xn (NHWC) --------------------------------------
    auto xn = at::empty({B, C, H, W},
                        opts.memory_format(at::MemoryFormat::ChannelsLast));
    {
        dim3 blk(32, 8, 1);
        dim3 grd((unsigned)((N + 31) / 32), (unsigned)(C / 32), (unsigned)B);
        nchw2nhwc_kernel<<<grd, blk, 0, stream>>>(
            xc.data_ptr<float>(), xn.data_ptr<float>(), (int)N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv1 (cuDNN, NHWC TF32) ----------------------------------------
    auto y1 = at::conv2d(xn, w1, {}, {1, 1}, {1, 1}, {1, 1}, 1);
    y1 = to_cl(y1);

    auto mean = at::empty({B, (int64_t)NG}, opts);
    auto rstd = at::empty({B, (int64_t)NG}, opts);
    compute_stats(y1, B, N, mean, rstd, eps, stream);

    auto z1 = at::empty({B, C, H, W},
                        opts.memory_format(at::MemoryFormat::ChannelsLast));
    {
        const long total = (long)N * 64;
        const int threads = 256;
        dim3 grd((unsigned)((total + threads - 1) / threads), (unsigned)B, 1);
        gn_silu_nhwc_kernel<<<grd, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(y1.data_ptr<float>()),
            reinterpret_cast<float4*>(z1.data_ptr<float>()),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            reinterpret_cast<const float4*>(g1.data_ptr<float>()),
            reinterpret_cast<const float4*>(b1.data_ptr<float>()),
            total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (cuDNN, NHWC TF32) ----------------------------------------
    auto y2 = at::conv2d(z1, w2, {}, {1, 1}, {1, 1}, {1, 1}, 1);
    y2 = to_cl(y2);

    auto mean2 = at::empty({B, (int64_t)NG}, opts);
    auto rstd2 = at::empty({B, (int64_t)NG}, opts);
    compute_stats(y2, B, N, mean2, rstd2, eps, stream);

    // ---- K5 : GN + SiLU + residual + NHWC->NCHW, straight into out_slice --
    {
        dim3 blk(32, 8, 1);
        dim3 grd((unsigned)((N + 31) / 32), (unsigned)(C / 32), (unsigned)B);
        gn_silu_add_t_kernel<<<grd, blk, 0, stream>>>(
            y2.data_ptr<float>(), xc.data_ptr<float>(),
            mean2.data_ptr<float>(), rstd2.data_ptr<float>(),
            g2.data_ptr<float>(), b2.data_ptr<float>(),
            out_slice.data_ptr<float>(), (int)N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// ---------------------------------------------------------------------------
// PLAN ITEMS 2/3/4/5 : fork/join driver
// ---------------------------------------------------------------------------
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight,
                          torch::Tensor norm1_weight,
                          torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight,
                          torch::Tensor norm2_weight,
                          torch::Tensor norm2_bias,
                          double eps)
{
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(x.dim() == 4, "input must be 4D");
    const int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C == CH, "specialised for C=256");

    auto s0 = at::cuda::getCurrentCUDAStream();

    // ---- one-time prep on the current stream (PLAN ITEM 2) ---------------
    auto xc   = x.is_contiguous() ? x : x.contiguous();
    auto opts = x.options();

    auto w1 = to_cl(conv1_weight);
    auto w2 = to_cl(conv2_weight);
    auto g1 = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
    auto b1 = norm1_bias.is_contiguous()   ? norm1_bias   : norm1_bias.contiguous();
    auto g2 = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
    auto b2 = norm2_bias.is_contiguous()   ? norm2_bias   : norm2_bias.contiguous();

    auto out = at::empty({B, C, H, W}, opts);   // plain NCHW contiguous

    // ---- chunk-count heuristic (PLAN ITEM 3) -----------------------------
    const int64_t M = B * H * W;
    int nchunks = (B >= 4 && M >= 65536 && (B % 2 == 0)) ? 2 : 1;

    if (nchunks == 1) {
        run_chunk(xc, w1, g1, b1, w2, g2, b2, out, (float)eps, s0);
        return out;
    }

    // ---- side stream (PLAN ITEM 4) ---------------------------------------
    static thread_local at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool();

    // ---- allocator safety (PLAN ITEM 5) ----------------------------------
    xc.record_stream(s1);
    out.record_stream(s1);
    w1.record_stream(s1);
    w2.record_stream(s1);
    g1.record_stream(s1);
    b1.record_stream(s1);
    g2.record_stream(s1);
    b2.record_stream(s1);

    // fork
    at::cuda::CUDAEvent ev0;
    ev0.record(s0);
    ev0.block(s1);

    const int64_t nb = B / nchunks;
    for (int c = 0; c < nchunks; ++c) {
        const int64_t off = (int64_t)c * nb;
        auto xs = xc.narrow(0, off, nb);    // contiguous NCHW batch slice
        auto os = out.narrow(0, off, nb);
        run_chunk(xs, w1, g1, b1, w2, g2, b2, os, (float)eps,
                  (c == 0) ? s0 : s1);
    }

    // join
    at::cuda::CUDAEvent ev1;
    ev1.record(s1);
    ev1.block(s0);

    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight,
                          torch::Tensor norm1_weight,
                          torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight,
                          torch::Tensor norm2_weight,
                          torch::Tensor norm2_bias,
                          double eps);
'''

_ext = load_inline(
    name="vae_resblock_fused_v2_streams",
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
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    # Granularity (D): full forward rewrite; the two 3x3 convs stay as cuDNN
    # NHWC-TF32 calls issued from inside the extension, every norm/act/add/
    # layout op is a custom CUDA kernel (K1..K5).  This round adds batch-split
    # two-stream pipelining (stream_pipeline_overlap) around that same code.
    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256 and conv1_weight.dtype == torch.float32):
            return self._ext.fused_block(x, conv1_weight, norm1_weight,
                                         norm1_bias, conv2_weight,
                                         norm2_weight, norm2_bias, eps_f)

        # Fallback (never taken for the benchmarked workloads: C is const 256).
        num_groups = 32
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, num_groups, weight=norm1_weight,
                           bias=norm1_bias, eps=eps_f)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, num_groups, weight=norm2_weight,
                           bias=norm2_bias, eps=eps_f)
        out = F.silu(out)
        return out + residual
