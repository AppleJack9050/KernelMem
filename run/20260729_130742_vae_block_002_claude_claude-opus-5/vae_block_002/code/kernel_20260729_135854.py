# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER / PLAN:
#   Granularity (C): the whole reference body runs through ONE extension entry
#   point `fused_res_block(...)`; the two 3x3 convolutions remain vendor
#   (cuDNN/TF32) mainloops, everything else (layout conversions, GroupNorm
#   statistics, affine, SiLU, residual add, NHWC->NCHW) is my own CUDA code.
#
#   THIS REVISION: stream_pipeline_overlap.
#   The residual block decomposes EXACTLY along the batch axis (GroupNorm
#   statistics are per (n, group); conv/residual are per sample).  We split N
#   into up to 4 chunks and run each chunk's full 9-kernel sequence on one of
#   two side streams from the ATen stream pool, so that chunk i's cuDNN
#   tensor-core mainloop (84% of DRAM idle) overlaps chunk i-1's DRAM-bound
#   stats / gn_silu / transpose chain (80%+ DRAM, <50% SM).  The vendor conv is
#   untouched; only scheduling changes.  For N < 4 (or small working sets) we
#   fall back to the previous, bit-identical single-stream sequence.
#
#   KERNEL MAP (per chunk, unchanged):
#     K1 nchw2nhwc_kernel        : x (NCHW) -> xl (NHWC), tiled 32x32 transpose
#     cuDNN conv2d               : xl * w1(channels_last) -> y1 (NHWC), TF32
#     K2 stats_partial_kernel    : per-channel sum/sumsq partials (coalesced)
#     K3 stats_finalize_kernel   : partials -> per-(n,c) {scale, shift}
#     K4 gn_silu_nhwc_kernel     : silu(y1*scale+shift), float4 vectorised
#     cuDNN conv2d               : y1s * w2(channels_last) -> y2 (NHWC), TF32
#     K2/K3 again on y2
#     K5 gn_silu_res_nchw_kernel : affine + SiLU + residual + NHWC->NCHW
#
#   PRECISION: float32 storage + float32 arithmetic everywhere; reductions
#   accumulate in float32.  TF32 only inside the vendor conv, exactly as the
#   reference (cudnn.allow_tf32 default True).
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
#include <algorithm>

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// K1: NCHW -> NHWC tiled transpose (32x32 tile, 32x8 threads)
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int P, int C) {
    __shared__ float tile[32][33];
    const int n  = blockIdx.z;
    const int c0 = blockIdx.y * 32;
    const int p0 = blockIdx.x * 32;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const size_t base_in = (size_t)n * (size_t)C * (size_t)P;
    #pragma unroll
    for (int r = 0; r < 32; r += 8) {
        const int c = c0 + ty + r;
        const int p = p0 + tx;
        if (p < P) tile[ty + r][tx] = in[base_in + (size_t)c * (size_t)P + (size_t)p];
    }
    __syncthreads();

    const size_t base_out = (size_t)n * (size_t)P * (size_t)C;
    #pragma unroll
    for (int r = 0; r < 32; r += 8) {
        const int p = p0 + ty + r;
        const int c = c0 + tx;
        if (p < P) out[base_out + (size_t)p * (size_t)C + (size_t)c] = tile[tx][ty + r];
    }
}

// ---------------------------------------------------------------------------
// K2: per-channel partial sum / sum-of-squares over a chunk of spatial
//     positions.  blockDim.x == C  (thread == channel  => fully coalesced).
// ---------------------------------------------------------------------------
__global__ void stats_partial_kernel(const float* __restrict__ in,
                                     float* __restrict__ psum,
                                     float* __restrict__ psq,
                                     int P, int C, int chunk, int nchunks) {
    const int n  = blockIdx.y;
    const int ck = blockIdx.x;
    const int c  = threadIdx.x;

    int p0 = ck * chunk;
    int p1 = p0 + chunk;
    if (p1 > P) p1 = P;

    float s = 0.0f, q = 0.0f;
    if (p0 < p1) {
        const float* ptr = in + (size_t)n * (size_t)P * (size_t)C
                              + (size_t)p0 * (size_t)C + (size_t)c;
        for (int p = p0; p < p1; ++p) {
            float v = __ldg(ptr);
            s += v;
            q += v * v;
            ptr += C;
        }
    }
    // always written -> no unwritten tail even when nchunks*chunk > P
    const size_t o = ((size_t)n * (size_t)nchunks + (size_t)ck) * (size_t)C + (size_t)c;
    psum[o] = s;
    psq[o]  = q;
}

// ---------------------------------------------------------------------------
// K3: reduce partials per (n, group) -> per-(n,c) scale / shift
// ---------------------------------------------------------------------------
__global__ void stats_finalize_kernel(const float* __restrict__ psum,
                                      const float* __restrict__ psq,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ scale,
                                      float* __restrict__ shift,
                                      int C, int G, int CPG, int nchunks,
                                      float inv_count, float eps) {
    __shared__ float ss[256];
    __shared__ float sq[256];

    const int blk = blockIdx.x;
    const int n   = blk / G;
    const int g   = blk - n * G;

    const int items = nchunks * CPG;
    float s = 0.0f, q = 0.0f;
    for (int i = threadIdx.x; i < items; i += blockDim.x) {
        const int cc = i / nchunks;
        const int ck = i - cc * nchunks;
        const size_t o = ((size_t)n * (size_t)nchunks + (size_t)ck) * (size_t)C
                       + (size_t)(g * CPG + cc);
        s += psum[o];
        q += psq[o];
    }
    ss[threadIdx.x] = s;
    sq[threadIdx.x] = q;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            ss[threadIdx.x] += ss[threadIdx.x + st];
            sq[threadIdx.x] += sq[threadIdx.x + st];
        }
        __syncthreads();
    }

    const float mean = ss[0] * inv_count;
    float var = sq[0] * inv_count - mean * mean;
    if (var < 0.0f) var = 0.0f;
    const float rstd = rsqrtf(var + eps);

    for (int t = threadIdx.x; t < CPG; t += blockDim.x) {
        const int c = g * CPG + t;
        const float gm = (gamma != nullptr) ? gamma[c] : 1.0f;
        const float bt = (beta  != nullptr) ? beta[c]  : 0.0f;
        scale[(size_t)n * (size_t)C + (size_t)c] = rstd * gm;
        shift[(size_t)n * (size_t)C + (size_t)c] = bt - mean * rstd * gm;
    }
}

// ---------------------------------------------------------------------------
// K4: NHWC  ->  silu(x*scale + shift)  (NHWC), float4 vectorised
// ---------------------------------------------------------------------------
__global__ void gn_silu_nhwc_kernel(const float4* __restrict__ in,
                                    float4* __restrict__ out,
                                    const float4* __restrict__ scale,
                                    const float4* __restrict__ shift,
                                    long P, int C4) {
    const int n = blockIdx.y;
    const long p = (long)blockIdx.x * (long)blockDim.y + (long)threadIdx.y;
    if (p >= P) return;
    const int c4 = threadIdx.x;

    const long idx = ((long)n * P + p) * (long)C4 + (long)c4;
    const long sidx = (long)n * (long)C4 + (long)c4;

    float4 v = in[idx];
    float4 s = scale[sidx];
    float4 b = shift[sidx];

    float4 r;
    r.x = silu_f(v.x * s.x + b.x);
    r.y = silu_f(v.y * s.y + b.y);
    r.z = silu_f(v.z * s.z + b.z);
    r.w = silu_f(v.w * s.w + b.w);
    out[idx] = r;
}

// ---------------------------------------------------------------------------
// K5: NHWC -> (affine + SiLU + residual) -> NCHW, fused transpose epilogue
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_nchw_kernel(const float* __restrict__ in,   // NHWC
                                        const float* __restrict__ res,  // NCHW
                                        float* __restrict__ out,        // NCHW
                                        const float* __restrict__ scale,
                                        const float* __restrict__ shift,
                                        int P, int C) {
    __shared__ float tile[32][33];
    const int n  = blockIdx.z;
    const int c0 = blockIdx.y * 32;
    const int p0 = blockIdx.x * 32;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const size_t base_in = (size_t)n * (size_t)P * (size_t)C;
    const size_t sbase   = (size_t)n * (size_t)C;

    {
        const int c = c0 + tx;
        const float sc = scale[sbase + (size_t)c];
        const float sh = shift[sbase + (size_t)c];
        #pragma unroll
        for (int r = 0; r < 32; r += 8) {
            const int p = p0 + ty + r;
            if (p < P) {
                float v = in[base_in + (size_t)p * (size_t)C + (size_t)c];
                tile[ty + r][tx] = silu_f(v * sc + sh);
            }
        }
    }
    __syncthreads();

    const size_t base_out = (size_t)n * (size_t)C * (size_t)P;
    #pragma unroll
    for (int r = 0; r < 32; r += 8) {
        const int c = c0 + ty + r;
        const int p = p0 + tx;
        if (p < P) {
            const size_t o = base_out + (size_t)c * (size_t)P + (size_t)p;
            out[o] = tile[tx][ty + r] + res[o];
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
// NOTE: `n_total` is the FULL batch of the forward (not the chunk batch); it is
// used only for the partial-block cap heuristic, so that the decomposition of P
// into stat chunks is identical to the unchunked path => per-sample partial
// sums (and thus the GroupNorm statistics) are bit-identical when chunking.
static void compute_scale_shift(const float* data, int N, int C, int P, int G,
                                const at::Tensor& gamma, const at::Tensor& beta,
                                float eps, const at::TensorOptions& opts,
                                cudaStream_t stream, int n_total,
                                at::Tensor& scale, at::Tensor& shift) {
    int nchunks = (P + 63) / 64;
    int cap = 2048 / (n_total > 0 ? n_total : 1);
    if (cap < 1) cap = 1;
    if (nchunks > cap) nchunks = cap;
    if (nchunks < 1) nchunks = 1;
    int chunk = (P + nchunks - 1) / nchunks;
    if (chunk < 1) chunk = 1;
    nchunks = (P + chunk - 1) / chunk;
    if (nchunks < 1) nchunks = 1;

    at::Tensor partial = at::empty({2, (long)N * (long)nchunks * (long)C}, opts);
    float* psum = partial.data_ptr<float>();
    float* psq  = psum + (size_t)N * (size_t)nchunks * (size_t)C;

    dim3 blk1(C, 1, 1);
    dim3 grd1(nchunks, N, 1);
    stats_partial_kernel<<<grd1, blk1, 0, stream>>>(data, psum, psq, P, C, chunk, nchunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scale = at::empty({(long)N, (long)C}, opts);
    shift = at::empty({(long)N, (long)C}, opts);

    const int CPG = C / G;
    const float inv_count = 1.0f / (float)((double)P * (double)CPG);

    const float* gptr = gamma.defined() ? gamma.data_ptr<float>() : nullptr;
    const float* bptr = beta.defined()  ? beta.data_ptr<float>()  : nullptr;

    stats_finalize_kernel<<<N * G, 256, 0, stream>>>(
        psum, psq, gptr, bptr,
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        C, G, CPG, nchunks, inv_count, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// One chunk of the residual block: all 9 launches go to the CURRENT stream.
static void run_chunk(const float* x_base, float* out_base,
                      const at::Tensor& w1c, const at::Tensor& g1, const at::Tensor& b1,
                      const at::Tensor& w2c, const at::Tensor& g2, const at::Tensor& b2,
                      int nc, int C, int H, int W, int P, int G,
                      float eps, int n_total, const at::TensorOptions& opts) {
    auto stream = at::cuda::getCurrentCUDAStream();

    // ---- K1: NCHW -> NHWC ----------------------------------------------
    at::Tensor xl = at::empty({nc, C, H, W},
                              opts.memory_format(at::MemoryFormat::ChannelsLast));
    dim3 blkT(32, 8, 1);
    dim3 grdT((P + 31) / 32, C / 32, nc);
    nchw2nhwc_kernel<<<grdT, blkT, 0, stream>>>(x_base, xl.data_ptr<float>(), P, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor nobias;

    // ---- conv1 (vendor, NHWC / TF32) ------------------------------------
    at::Tensor y1 = at::conv2d(xl, w1c, nobias, {1, 1}, {1, 1});
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GroupNorm1 stats + fused affine/SiLU ----------------------------
    at::Tensor sc1, sh1;
    compute_scale_shift(y1.data_ptr<float>(), nc, C, P, G, g1, b1, eps,
                        opts, stream, n_total, sc1, sh1);

    at::Tensor y1s = at::empty({nc, C, H, W},
                               opts.memory_format(at::MemoryFormat::ChannelsLast));
    {
        const int C4 = C / 4;
        int by = 256 / C4;
        if (by < 1) by = 1;
        dim3 blkN(C4, by, 1);
        dim3 grdN((P + by - 1) / by, nc, 1);
        gn_silu_nhwc_kernel<<<grdN, blkN, 0, stream>>>(
            reinterpret_cast<const float4*>(y1.data_ptr<float>()),
            reinterpret_cast<float4*>(y1s.data_ptr<float>()),
            reinterpret_cast<const float4*>(sc1.data_ptr<float>()),
            reinterpret_cast<const float4*>(sh1.data_ptr<float>()),
            (long)P, C4);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (vendor, NHWC / TF32) ------------------------------------
    at::Tensor y2 = at::conv2d(y1s, w2c, nobias, {1, 1}, {1, 1});
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GroupNorm2 stats -------------------------------------------------
    at::Tensor sc2, sh2;
    compute_scale_shift(y2.data_ptr<float>(), nc, C, P, G, g2, b2, eps,
                        opts, stream, n_total, sc2, sh2);

    // ---- K5: affine + SiLU + residual + NHWC->NCHW ------------------------
    gn_silu_res_nchw_kernel<<<grdT, blkT, 0, stream>>>(
        y2.data_ptr<float>(), x_base, out_base,
        sc2.data_ptr<float>(), sh2.data_ptr<float>(), P, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_res_block(torch::Tensor x,
                              torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                              torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                              double eps_d, int64_t num_groups) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(w1.scalar_type() == at::kFloat && w2.scalar_type() == at::kFloat);

    auto xc = x.is_contiguous() ? x : x.contiguous();

    const int N = (int)xc.size(0);
    const int C = (int)xc.size(1);
    const int H = (int)xc.size(2);
    const int W = (int)xc.size(3);
    const int P = H * W;
    const int G = (int)num_groups;
    const float eps = (float)eps_d;

    TORCH_CHECK(C % 32 == 0 && C % 4 == 0 && C <= 1024, "unsupported channel count");
    TORCH_CHECK(C % G == 0, "channels not divisible by groups");

    auto opts = xc.options();
    auto main_stream = at::cuda::getCurrentCUDAStream();

    // ---- hoisted: weights + output live on the main stream ----------------
    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    at::Tensor out = at::empty({N, C, H, W}, opts);

    const float* x_ptr = xc.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();
    const size_t sample_elems = (size_t)C * (size_t)P;

    // ---- plan item 1: exact decomposition ---------------------------------
    int nchunks = (N >= 4 && (size_t)N * (size_t)C * (size_t)P * 4ull >= (32u << 20))
                      ? std::min(4, N) : 1;
    int cs = (N + nchunks - 1) / nchunks;
    nchunks = (N + cs - 1) / cs;

    // ---- plan item 7: chunk-size guard (keep the conv GEMM wide enough) ---
    while (nchunks > 1) {
        long ctas = (long)cs * (long)P / 64 * 2;
        if (ctas >= 256) break;
        nchunks = (nchunks > 2) ? 2 : 1;
        cs = (N + nchunks - 1) / nchunks;
        nchunks = (N + cs - 1) / cs;
    }

    if (nchunks <= 1) {
        // Mandatory fallback: byte-identical kernel sequence on the default stream.
        run_chunk(x_ptr, o_ptr, w1c, g1, b1, w2c, g2, b2,
                  N, C, H, W, P, G, eps, N, opts);
        return out;
    }

    // ---- plan item 2: stream plumbing -------------------------------------
    c10::cuda::CUDAStream sides[2] = {at::cuda::getStreamFromPool(),
                                      at::cuda::getStreamFromPool()};

    at::cuda::CUDAEvent start;
    start.record(main_stream);
    start.block(sides[0]);
    start.block(sides[1]);

    // ---- plan item 4: cross-stream lifetime safety ------------------------
    for (int i = 0; i < 2; ++i) {
        xc.record_stream(sides[i]);
        w1c.record_stream(sides[i]);
        w2c.record_stream(sides[i]);
        out.record_stream(sides[i]);
    }

    for (int c = 0; c < nchunks; ++c) {
        const int n0 = c * cs;
        int nc = cs;
        if (n0 + nc > N) nc = N - n0;
        if (nc <= 0) continue;

        at::cuda::CUDAStreamGuard guard(sides[c & 1]);
        run_chunk(x_ptr + (size_t)n0 * sample_elems,
                  o_ptr + (size_t)n0 * sample_elems,
                  w1c, g1, b1, w2c, g2, b2,
                  nc, C, H, W, P, G, eps, N, opts);
    }

    for (int i = 0; i < 2; ++i) {
        at::cuda::CUDAEvent done;
        done.record(sides[i]);
        done.block(main_stream);
    }

    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor fused_res_block(torch::Tensor x,
                              torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                              torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                              double eps_d, int64_t num_groups);
'''

_ext = load_inline(
    name="vae_res_block_fused_v2_streams",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["fused_res_block"],
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
    """See module header for the granularity/fusion plan (granularity C)."""

    def __init__(self):
        super().__init__()
        self.ext = _ext
        self.num_groups = 32  # const per the SOL problem definition

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.reshape(-1)[0].item()) if eps.numel() > 0 else 1e-6
        else:
            eps_f = float(eps)

        # Fast path: fused extension (single entry point, all kernels inside).
        if (x.is_cuda and x.dim() == 4 and x.dtype == torch.float32
                and x.size(1) % 32 == 0 and x.size(1) <= 1024
                and x.size(1) % self.num_groups == 0):
            return self.ext.fused_res_block(
                x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias,
                eps_f, self.num_groups)

        # Conservative fallback (shape/dtype outside the supported envelope).
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps_f)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, self.num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps_f)
        out = F.silu(out)
        return out + residual
