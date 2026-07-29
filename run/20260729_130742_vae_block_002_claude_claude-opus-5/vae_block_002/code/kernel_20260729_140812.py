# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER / PLAN (required):
#   1) CHOSEN GRANULARITY: (C) — fuse many ops into one/few custom kernels.
#      The whole reference body is executed by a single extension entry point
#      `fused_res_block(...)`, but the two 3x3 convolutions are still issued as
#      vendor (cuDNN/TF32) calls from inside that extension (near roofline).
#
#   2) OPS REPLACED (all done by my own CUDA kernels):
#      - the cudnn layout conversions -> one tiled NCHW->NHWC transpose, and
#        the NHWC->NCHW conversion fused into the epilogue kernel (free).
#      - RowwiseMomentsCUDAKernel + ComputeFusedParams (both GroupNorms)
#        -> ONE single-pass atomic-privatized reduction kernel per norm
#           (in-register 8-lane group reduce -> [N][G][2] accumulator ->
#            last-block finalize emits scale/shift).  The 2 MB partial
#            round-trip and the latency-bound finalize launch are gone.
#      - GroupNorm affine + SiLU + residual add -> fused epilogue kernels.
#
#   3) FUSION MAP (kernel by kernel):
#      K1 nchw2nhwc_kernel        : x (NCHW) -> xl (NHWC), tiled 32x32.
#      cuDNN conv2d               : xl * w1(channels_last) -> y1 (NHWC), TF32.
#      K2 stats_atomic_kernel     : per-channel partials -> warp-shuffle group
#                                   reduce -> atomicAdd into [N][G][2] ->
#                                   last block emits {scale, shift}[N][C].
#      K4 gn_silu_nhwc_kernel     : y1 -> silu(y1*scale+shift) NHWC, float4.
#      cuDNN conv2d               : y1s * w2(channels_last) -> y2 (NHWC), TF32.
#      K2 again on y2 (second scratch slot of the same zeroed buffer).
#      K5 gn_silu_res_nchw_kernel : affine + SiLU + residual + NHWC->NCHW.
#
#   4) WHAT STAYS IN PYTORCH / VENDOR AND WHY:
#      - at::conv2d (cuDNN TF32 NHWC): vendor GEMM already at roofline.
#      - weight .contiguous(ChannelsLast): 2.4 MB each, ~2 us.
#
#   PRECISION: float32 storage + float32 arithmetic everywhere; reductions
#   accumulate in float32 (register -> warp shuffle -> fp32 atomicAdd).
#   TF32 is used only inside the conv, exactly as the reference does.
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
// K2 (plan items 2-6): single-pass, atomic-privatized GroupNorm statistics.
//   blockDim.x == C  (thread == channel  => fully coalesced streaming loads).
//   * accumulation loop kept byte-for-byte from stats_partial_kernel
//   * in-register group reduction over CPG contiguous lanes (__shfl_down_sync)
//   * one (sum,sumsq) atomicAdd pair per group per block into acc[N][G][2]
//   * last arriving block per image emits the coalesced scale/shift row
// ---------------------------------------------------------------------------
__global__ void stats_atomic_kernel(const float* __restrict__ in,
                                    float* __restrict__ acc,     // [N][G][2]
                                    int*   __restrict__ ctr,     // [N]
                                    float* __restrict__ scale,   // [N][C]
                                    float* __restrict__ shift,   // [N][C]
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    int P, int C, int chunk, int nchunks,
                                    int G, int CPG,
                                    float inv_count, float eps,
                                    int use_shfl) {
    extern __shared__ float smem[];           // only used by the generic path

    const int n  = blockIdx.y;
    const int ck = blockIdx.x;
    const int c  = threadIdx.x;

    int p0 = ck * chunk;
    int p1 = p0 + chunk;
    if (p1 > P) p1 = P;

    // ---- unchanged coalesced accumulation loop (80.8% DRAM) ---------------
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
    // NOTE: no early return -- every thread reaches the shuffles / syncs.

    // ---- plan item 3: in-register group reduction + privatized atomics ----
    if (use_shfl) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < CPG) {
                s += __shfl_down_sync(0xffffffffu, s, off, CPG);
                q += __shfl_down_sync(0xffffffffu, q, off, CPG);
            }
        }
        if ((c & (CPG - 1)) == 0) {
            const int g = c / CPG;
            atomicAdd(&acc[((size_t)n * (size_t)G + (size_t)g) * 2 + 0], s);
            atomicAdd(&acc[((size_t)n * (size_t)G + (size_t)g) * 2 + 1], q);
        }
    } else {
        // generic fallback: shared-memory segmented reduction (CPG not a
        // power of two, or CPG > 32).  Never taken for C=256, G=32.
        float* ss = smem;
        float* sq = smem + C;
        ss[c] = s;
        sq[c] = q;
        __syncthreads();
        if ((c % CPG) == 0) {
            float as = 0.0f, aq = 0.0f;
            for (int k = 0; k < CPG; ++k) { as += ss[c + k]; aq += sq[c + k]; }
            const int g = c / CPG;
            atomicAdd(&acc[((size_t)n * (size_t)G + (size_t)g) * 2 + 0], as);
            atomicAdd(&acc[((size_t)n * (size_t)G + (size_t)g) * 2 + 1], aq);
        }
        __syncthreads();
    }

    // ---- plan item 4: last-block finalize (replaces stats_finalize_kernel) -
    __threadfence();
    __shared__ int isLast;
    if (c == 0) isLast = (atomicAdd(&ctr[n], 1) == (nchunks - 1)) ? 1 : 0;
    __syncthreads();

    if (isLast) {
        const volatile float* vacc = acc;     // bypass L1 for cross-block data
        const int g = c / CPG;
        const size_t go = ((size_t)n * (size_t)G + (size_t)g) * 2;
        const float sm = vacc[go + 0] * inv_count;
        float v = vacc[go + 1] * inv_count - sm * sm;
        if (v < 0.0f) v = 0.0f;                     // plan item 6: var clamp
        const float r  = rsqrtf(v + eps);
        const float gm = (gamma != nullptr) ? gamma[c] : 1.0f;
        const float bt = (beta  != nullptr) ? beta[c]  : 0.0f;
        scale[(size_t)n * (size_t)C + (size_t)c] = r * gm;
        shift[(size_t)n * (size_t)C + (size_t)c] = bt - sm * r * gm;
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
// plan item 7: takes the scratch base + slot index (0 = norm1, 1 = norm2) and
// issues EXACTLY ONE kernel launch; scale/shift layout [N][C] is unchanged.
static void compute_scale_shift(const float* data, int N, int C, int P, int G,
                                const at::Tensor& gamma, const at::Tensor& beta,
                                float eps, const at::TensorOptions& opts,
                                cudaStream_t stream,
                                float* scratch_base, int slot, int slot_stride,
                                at::Tensor& scale, at::Tensor& shift) {
    int nchunks = (P + 63) / 64;
    int cap = 2048 / (N > 0 ? N : 1);
    if (cap < 1) cap = 1;
    if (nchunks > cap) nchunks = cap;
    if (nchunks > 1024) nchunks = 1024;          // plan item 6: contention cap
    if (nchunks < 1) nchunks = 1;
    int chunk = (P + nchunks - 1) / nchunks;
    if (chunk < 1) chunk = 1;
    nchunks = (P + chunk - 1) / chunk;           // exact ceil -> counter target
    if (nchunks < 1) nchunks = 1;

    scale = at::empty({(long)N, (long)C}, opts);
    shift = at::empty({(long)N, (long)C}, opts);

    const int CPG = C / G;
    const float inv_count = 1.0f / (float)((double)P * (double)CPG);

    const float* gptr = gamma.defined() ? gamma.data_ptr<float>() : nullptr;
    const float* bptr = beta.defined()  ? beta.data_ptr<float>()  : nullptr;

    float* accp = scratch_base + (size_t)slot * (size_t)slot_stride;
    int*   ctrp = reinterpret_cast<int*>(accp + (size_t)N * (size_t)G * 2);

    const int use_shfl = (CPG <= 32 && (CPG & (CPG - 1)) == 0) ? 1 : 0;
    const size_t shmem = use_shfl ? 0 : (size_t)2 * (size_t)C * sizeof(float);

    dim3 blk(C, 1, 1);
    dim3 grd(nchunks, N, 1);
    stats_atomic_kernel<<<grd, blk, shmem, stream>>>(
        data, accp, ctrp,
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        gptr, bptr,
        P, C, chunk, nchunks, G, CPG, inv_count, eps, use_shfl);
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

    // ---- plan item 1: ONE tiny zeroed scratch for BOTH norms --------------
    const int slot_stride = N * G * 2 + N;   // {float acc[N][G][2]; int ctr[N];}
    at::Tensor acc_scratch = at::zeros({2, (long)slot_stride}, opts);
    float* scratch_base = acc_scratch.data_ptr<float>();

    // ---- K1: NCHW -> NHWC ----------------------------------------------
    at::Tensor xl = at::empty({N, C, H, W}, opts.memory_format(at::MemoryFormat::ChannelsLast));
    dim3 blkT(32, 8, 1);
    dim3 grdT((P + 31) / 32, C / 32, N);
    nchw2nhwc_kernel<<<grdT, blkT, 0, stream>>>(xc.data_ptr<float>(), xl.data_ptr<float>(), P, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor nobias;

    // ---- conv1 (vendor, NHWC / TF32) ------------------------------------
    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    at::Tensor y1 = at::conv2d(xl, w1c, nobias, {1, 1}, {1, 1});
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GroupNorm1 stats (single fused launch, slot 0) -------------------
    at::Tensor sc1, sh1;
    compute_scale_shift(y1.data_ptr<float>(), N, C, P, G, g1, b1, (float)eps_d,
                        opts, stream, scratch_base, 0, slot_stride, sc1, sh1);

    at::Tensor y1s = at::empty({N, C, H, W}, opts.memory_format(at::MemoryFormat::ChannelsLast));
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

    // ---- conv2 (vendor, NHWC / TF32) ------------------------------------
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    at::Tensor y2 = at::conv2d(y1s, w2c, nobias, {1, 1}, {1, 1});
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GroupNorm2 stats (single fused launch, slot 1) -------------------
    at::Tensor sc2, sh2;
    compute_scale_shift(y2.data_ptr<float>(), N, C, P, G, g2, b2, (float)eps_d,
                        opts, stream, scratch_base, 1, slot_stride, sc2, sh2);

    // ---- K5: affine + SiLU + residual + NHWC->NCHW ------------------------
    at::Tensor out = at::empty({N, C, H, W}, opts);
    gn_silu_res_nchw_kernel<<<grdT, blkT, 0, stream>>>(
        y2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
        sc2.data_ptr<float>(), sh2.data_ptr<float>(), P, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

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
    name="vae_res_block_fused_v2",
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
