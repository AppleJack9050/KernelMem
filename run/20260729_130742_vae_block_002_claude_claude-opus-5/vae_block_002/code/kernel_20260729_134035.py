# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER / PLAN (required):
#   1) CHOSEN GRANULARITY: (C) — fuse many ops into one/few custom kernels.
#      The whole reference body is executed by a single extension entry point
#      `fused_res_block(...)`, but the two 3x3 convolutions are still issued as
#      vendor (cuDNN/TF32) calls from inside that extension.
#
#   2) OPS REPLACED (all done by my own CUDA kernels):
#      - the 4 cudnn nchwToNhwc / 2 nhwcToNchw layout conversions  -> removed;
#        exactly ONE explicit NCHW->NHWC transpose of x remains, and the
#        NHWC->NCHW conversion of the result is fused into the epilogue kernel.
#      - RowwiseMomentsCUDAKernel + ComputeFusedParams (both GroupNorms).
#      - GroupNorm affine + SiLU + residual add -> fused.
#
#   3) NEW IN THIS REVISION — l2_resident_batch_chunking:
#      The per-stage working set (C*H*W*4 * N) exceeds the 50 MB L2, so every
#      re-read of an intermediate (stats1, K4, conv2 input, stats2, K5) was a
#      full DRAM round trip.  The whole pipeline
#         K1 -> conv1 -> stats -> K4 -> conv2 -> stats -> K5
#      is now cache-blocked over the batch axis: with a 16 MB budget we process
#      chunkN = BUDGET / (C*P*4) samples at a time so each chunk's intermediates
#      stay L2-resident across the five re-reads.  GroupNorm statistics are
#      per-(n,group) and the residual is elementwise, so batch chunking is
#      exactly equivalent.  Shapes whose full batch already fits (N <= chunkN,
#      e.g. 2x256x64x64 and 1x256x131x131) take the bit-identical unchunked
#      path.  Single stream, strictly sequential — the win is temporal locality.
#
#   PRECISION: float32 storage + float32 arithmetic; reductions accumulate in
#   float32 hierarchically.  TF32 only inside the conv, as in the reference.
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
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
static void compute_scale_shift(const float* data, int N, int C, int P, int G,
                                const at::Tensor& gamma, const at::Tensor& beta,
                                float eps, const at::TensorOptions& opts,
                                cudaStream_t stream,
                                at::Tensor& scale, at::Tensor& shift) {
    int nchunks = (P + 63) / 64;
    int cap = 2048 / (N > 0 ? N : 1);
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

    TORCH_CHECK(C % 32 == 0 && C % 4 == 0 && C <= 1024, "unsupported channel count");
    TORCH_CHECK(C % G == 0, "channels not divisible by groups");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = xc.options();
    const float eps_f = (float)eps_d;

    // ================= plan item 1: L2 cache-blocking decision ==============
    const size_t bps = (size_t)C * (size_t)P * 4;          // bytes / sample / stage
    const size_t BUDGET = 16ull << 20;                     // 16 MB L2 budget
    int chunkN = (int)(BUDGET / (bps > 0 ? bps : 1));
    if (chunkN < 1) chunkN = 1;
    if (chunkN > N) chunkN = N;
    if (chunkN > 0 && (N / chunkN) > 8) chunkN = (N + 7) / 8;   // cap #chunks at 8
    if (chunkN < 1) chunkN = 1;
    if (chunkN > N) chunkN = N;
    const bool chunked = (N > chunkN);

    dim3 blkT(32, 8, 1);
    at::Tensor nobias;

    if (!chunked) {
        // ---------------- original (bit-identical) full-batch path ----------
        at::Tensor xl = at::empty({N, C, H, W},
                                  opts.memory_format(at::MemoryFormat::ChannelsLast));
        dim3 grdT((P + 31) / 32, C / 32, N);
        nchw2nhwc_kernel<<<grdT, blkT, 0, stream>>>(
            xc.data_ptr<float>(), xl.data_ptr<float>(), P, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
        at::Tensor y1 = at::conv2d(xl, w1c, nobias, {1, 1}, {1, 1});
        if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
            y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

        at::Tensor sc1, sh1;
        compute_scale_shift(y1.data_ptr<float>(), N, C, P, G, g1, b1, eps_f,
                            opts, stream, sc1, sh1);

        at::Tensor y1s = at::empty({N, C, H, W},
                                   opts.memory_format(at::MemoryFormat::ChannelsLast));
        {
            const int C4 = C / 4;
            int by = 256 / C4;
            if (by < 1) by = 1;
            dim3 blkN(C4, by, 1);
            dim3 grdN((P + by - 1) / by, N, 1);
            gn_silu_nhwc_kernel<<<grdN, blkN, 0, stream>>>(
                reinterpret_cast<const float4*>(y1.data_ptr<float>()),
                reinterpret_cast<float4*>(y1s.data_ptr<float>()),
                reinterpret_cast<const float4*>(sc1.data_ptr<float>()),
                reinterpret_cast<const float4*>(sh1.data_ptr<float>()),
                (long)P, C4);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
        at::Tensor y2 = at::conv2d(y1s, w2c, nobias, {1, 1}, {1, 1});
        if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
            y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

        at::Tensor sc2, sh2;
        compute_scale_shift(y2.data_ptr<float>(), N, C, P, G, g2, b2, eps_f,
                            opts, stream, sc2, sh2);

        at::Tensor out = at::empty({N, C, H, W}, opts);
        gn_silu_res_nchw_kernel<<<grdT, blkT, 0, stream>>>(
            y2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
            sc2.data_ptr<float>(), sh2.data_ptr<float>(), P, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return out;
    }

    // =============== plan item 2: L2-resident batch-chunked path ============
    // Hoisted, full-size allocations + one-time weight layout conversions.
    at::Tensor xl  = at::empty({N, C, H, W},
                               opts.memory_format(at::MemoryFormat::ChannelsLast));
    at::Tensor y1s = at::empty({N, C, H, W},
                               opts.memory_format(at::MemoryFormat::ChannelsLast));
    at::Tensor out = at::empty({N, C, H, W}, opts);

    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);

    float* xptr   = xc.data_ptr<float>();
    float* xlptr  = xl.data_ptr<float>();
    float* y1sptr = y1s.data_ptr<float>();
    float* outptr = out.data_ptr<float>();

    const int C4 = C / 4;
    int by = 256 / C4;
    if (by < 1) by = 1;
    dim3 blkN(C4, by, 1);

    for (int n0 = 0; n0 < N; n0 += chunkN) {
        const int nb = std::min(chunkN, N - n0);
        TORCH_CHECK(nb >= 1, "empty chunk");
        const size_t off = (size_t)n0 * (size_t)C * (size_t)P;

        dim3 grdT((P + 31) / 32, C / 32, nb);

        // ---- K1: NCHW -> NHWC (chunk) ----------------------------------
        nchw2nhwc_kernel<<<grdT, blkT, 0, stream>>>(xptr + off, xlptr + off, P, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // ---- conv1 (vendor, NHWC / TF32) on the chunk view --------------
        at::Tensor xl_chunk = xl.narrow(0, n0, nb);
        at::Tensor y1 = at::conv2d(xl_chunk, w1c, nobias, {1, 1}, {1, 1});
        if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
            y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

        // ---- GroupNorm1 stats (nb samples) ------------------------------
        at::Tensor sc1, sh1;
        compute_scale_shift(y1.data_ptr<float>(), nb, C, P, G, g1, b1, eps_f,
                            opts, stream, sc1, sh1);

        // ---- K4: affine + SiLU (NHWC) -----------------------------------
        {
            dim3 grdN((P + by - 1) / by, nb, 1);
            gn_silu_nhwc_kernel<<<grdN, blkN, 0, stream>>>(
                reinterpret_cast<const float4*>(y1.data_ptr<float>()),
                reinterpret_cast<float4*>(y1sptr + off),
                reinterpret_cast<const float4*>(sc1.data_ptr<float>()),
                reinterpret_cast<const float4*>(sh1.data_ptr<float>()),
                (long)P, C4);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // ---- conv2 (vendor, NHWC / TF32) --------------------------------
        at::Tensor y1s_chunk = y1s.narrow(0, n0, nb);
        at::Tensor y2 = at::conv2d(y1s_chunk, w2c, nobias, {1, 1}, {1, 1});
        if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
            y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

        // ---- GroupNorm2 stats (nb samples) ------------------------------
        at::Tensor sc2, sh2;
        compute_scale_shift(y2.data_ptr<float>(), nb, C, P, G, g2, b2, eps_f,
                            opts, stream, sc2, sh2);

        // ---- K5: affine + SiLU + residual + NHWC->NCHW -------------------
        gn_silu_res_nchw_kernel<<<grdT, blkT, 0, stream>>>(
            y2.data_ptr<float>(), xptr + off, outptr + off,
            sc2.data_ptr<float>(), sh2.data_ptr<float>(), P, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    name="vae_res_block_fused_v2_l2chunk",
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
    """See module header for the granularity/fusion plan (granularity C) and the
    l2_resident_batch_chunking revision."""

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
