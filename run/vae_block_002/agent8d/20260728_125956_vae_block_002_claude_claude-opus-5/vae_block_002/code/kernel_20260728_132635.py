# =============================================================================
# ModelNew : fused VAE residual block
#            Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual
#
# v2: L2 CACHE-BLOCKING of the GroupNorm two-pass.
#
# Each GroupNorm previously read the whole conv output twice from DRAM
# (lts__t_sector_hit_rate 0.73% / 6%).  Now every GN is partitioned into
# (batch-block x channel-block) chunks whose slab fits the L2 working-set
# budget, and stats-then-apply are issued back-to-back for the SAME chunk, so
# the apply pass re-reads data still resident in L2.
#
# Kernel list (launch order):
#   K0 nchw2nhwc_kernel                : tiled transpose of x            (once)
#   -- at::conv2d (cuDNN, NHWC/TF32)
#   per chunk: K1 gn_stats_kernel      -> atomicAdd into acc[2*B*32]
#              K2 gn_apply_silu_kernel  (mean/rstd derived in prologue)
#   -- at::conv2d (cuDNN, NHWC/TF32)
#   per chunk: K1 gn_stats_kernel
#              K3 gn_apply_silu_res_nchw_kernel (affine+SiLU+residual+NHWC->NCHW)
#
# gn_finalize_kernel is gone: the per-(batch,group) sums are accumulated with
# float atomics into a 2 KB buffer that is zeroed once per GN with
# cudaMemsetAsync; the apply kernels turn (sum, sumsq) into (mean, rstd) with
# two scalar loads + one rsqrt per block.
#
# Precision: float32 storage + float32 arithmetic everywhere; TF32 only inside
# cuDNN's conv, exactly as the reference does.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <vector>
#include <math.h>

#define CH       256
#define NGROUPS  32
#define CPG      8
#define TP       64
#define TC       64
#define NTHR     256

// ---------------------------------------------------------------- transpose
// NCHW (contiguous) -> NHWC (channels_last), shared memory tiled.  (unchanged)
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int HW)
{
    __shared__ float sm[TP * (TC + 1)];
    const int b   = blockIdx.z;
    const int c0  = blockIdx.y * TC;
    const int p0  = blockIdx.x * TP;
    const int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int cl  = idx >> 6;
        int pl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            sm[pl * (TC + 1) + cl] =
                in[((long)b * CH + (c0 + cl)) * (long)HW + p];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int pl  = idx >> 6;
        int cl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            out[((long)b * HW + p) * CH + (c0 + cl)] = sm[pl * (TC + 1) + cl];
        }
    }
}

// ------------------------------------------------------------- GN statistics
// Chunked: covers batches [n0, n0+gridDim.y) and channels [c0, c0+cblk).
// blockDim.x == NTHR == 256, viewed as (nsub = 256/cblk) pixel sub-chunks
// x cblk channels.  Per-(b,g) sums are atomically accumulated into acc.
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ acc,
                                int HW, int n0, int c0, int cblk, int nsub,
                                int ppc, long sq_off)
{
    __shared__ float sh1[NTHR / 8];   // 32 slots  (nsub * ngl == 32)
    __shared__ float sh2[NTHR / 8];

    const int b   = n0 + blockIdx.y;
    const int tid = threadIdx.x;
    const int cl  = tid & (cblk - 1);   // channel inside the chunk
    const int sub = tid / cblk;         // pixel sub-chunk
    const int ngl = cblk >> 3;          // groups covered by this chunk

    const int chunk  = blockIdx.x * nsub + sub;
    const int pstart = chunk * ppc;

    float s = 0.f, s2 = 0.f;
    if (pstart < HW) {
        int pend = pstart + ppc;
        if (pend > HW) pend = HW;
        int n = pend - pstart;
        const float* px = x + (long)b * HW * CH + (long)pstart * CH + (c0 + cl);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float v0 = px[0];
            float v1 = px[CH];
            float v2 = px[2 * CH];
            float v3 = px[3 * CH];
            s  += (v0 + v1) + (v2 + v3);
            s2 += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
            px += 4 * CH;
        }
        for (; i < n; ++i) {
            float v = px[0];
            s  += v;
            s2 += v * v;
            px += CH;
        }
    }

    // reduce over the 8 lanes that belong to the same group
    #pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off, 8);
        s2 += __shfl_down_sync(0xffffffffu, s2, off, 8);
    }
    if ((cl & 7) == 0) {
        int gl = cl >> 3;
        sh1[sub * ngl + gl] = s;
        sh2[sub * ngl + gl] = s2;
    }
    __syncthreads();

    // fold the pixel sub-chunks, then one atomic pair per group per block
    if (tid < ngl) {
        float a = 0.f, q = 0.f;
        for (int su = 0; su < nsub; ++su) {
            a += sh1[su * ngl + tid];
            q += sh2[su * ngl + tid];
        }
        long o = (long)b * NGROUPS + (c0 >> 3) + tid;
        atomicAdd(&acc[o], a);
        atomicAdd(&acc[sq_off + o], q);
    }
}

// -------------------------------------------- GN affine + SiLU (NHWC -> NHWC)
// grid (ceil(HW/8), nb), block cblk threads == cblk channels of the chunk.
__global__ void gn_apply_silu_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     const float* __restrict__ acc,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     int HW, int n0, int c0,
                                     long sq_off, float invN, float eps)
{
    const int b = n0 + blockIdx.y;
    const int c = c0 + threadIdx.x;
    const int g = c >> 3;

    const long o = (long)b * NGROUPS + g;
    float m = acc[o] * invN;
    float v = acc[sq_off + o] * invN - m * m;
    if (!(v > 0.f)) v = 0.f;
    const float r  = rsqrtf(v + eps);
    const float sc = gamma[c] * r;
    const float sh = beta[c] - m * sc;

    const int p0 = blockIdx.x * 8;
    if (p0 >= HW) return;
    long off = (long)b * HW * CH + (long)p0 * CH + c;

    if (p0 + 8 <= HW) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float t = x[off + (long)i * CH] * sc + sh;
            y[off + (long)i * CH] = t / (1.f + expf(-t));
        }
    } else {
        for (int i = 0; p0 + i < HW; ++i) {
            float t = x[off + (long)i * CH] * sc + sh;
            y[off + (long)i * CH] = t / (1.f + expf(-t));
        }
    }
}

// ----------- GN affine + SiLU + residual add + NHWC -> NCHW (one pass)
// grid (ceil(HW/TP), cblk/TC, nb), block NTHR.
__global__ void gn_apply_silu_res_nchw_kernel(const float* __restrict__ x,
                                              const float* __restrict__ res,
                                              float* __restrict__ out,
                                              const float* __restrict__ acc,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              int HW, int n0, int c0,
                                              long sq_off, float invN, float eps)
{
    __shared__ float sm[TC * (TP + 1)];
    __shared__ float ssc[TC];
    __shared__ float ssh[TC];

    const int b   = n0 + blockIdx.z;
    const int ct0 = c0 + blockIdx.y * TC;
    const int p0  = blockIdx.x * TP;
    const int tid = threadIdx.x;

    if (tid < TC) {
        int c = ct0 + tid;
        int g = c >> 3;
        long o = (long)b * NGROUPS + g;
        float m = acc[o] * invN;
        float v = acc[sq_off + o] * invN - m * m;
        if (!(v > 0.f)) v = 0.f;
        float r  = rsqrtf(v + eps);
        float sc = gamma[c] * r;
        ssc[tid] = sc;
        ssh[tid] = beta[c] - m * sc;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int pl  = idx >> 6;
        int cl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            float v = x[((long)b * HW + p) * CH + (ct0 + cl)] * ssc[cl] + ssh[cl];
            sm[cl * (TP + 1) + pl] = v / (1.f + expf(-v));
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < (TP * TC) / NTHR; ++i) {
        int idx = i * NTHR + tid;
        int cl  = idx >> 6;
        int pl  = idx & 63;
        int p   = p0 + pl;
        if (p < HW) {
            long o = ((long)b * CH + (ct0 + cl)) * (long)HW + p;
            out[o] = sm[cl * (TP + 1) + pl] + res[o];
        }
    }
}

// ------------------------------------------------------------- host helpers
struct ChunkCfg {
    int cblk;       // channels per chunk (64 / 128 / 256)
    int nblk;       // batches per chunk
    int nsub;       // NTHR / cblk  (pixel sub-chunks per stats block)
    int nblocks_p;  // stats grid.x
    int ppc;        // pixels per stats sub-chunk
};

static ChunkCfg pick_cfg(int B, int HW)
{
    // L2 working-set budget for one (batch-block x channel-block) slab.
    const long L2_BUDGET = 36L * 1024 * 1024;

    ChunkCfg cfg;
    long total = (long)B * HW * CH * 4;

    if (total <= L2_BUDGET) {
        // small shape: single chunk, identical launch pattern to the base
        cfg.cblk = CH;
        cfg.nblk = B;
    } else {
        int cblk = CH;
        while (cblk > 64 && (long)HW * cblk * 4 > L2_BUDGET) cblk >>= 1;
        long slab = (long)HW * cblk * 4;
        int nblk = (int)(L2_BUDGET / (slab > 0 ? slab : 1));
        if (nblk < 1) nblk = 1;
        if (nblk > B) nblk = B;
        // keep the chunk count (and therefore launch count) bounded
        while ((((B + nblk - 1) / nblk) * (CH / cblk)) > 8) {
            if (cblk < CH)      cblk <<= 1;
            else if (nblk < B)  nblk = std::min(B, nblk * 2);
            else                break;
        }
        cfg.cblk = cblk;
        cfg.nblk = nblk;
    }

    cfg.nsub = NTHR / cfg.cblk;

    int total_chunks = HW / 32;
    if (total_chunks < 1)    total_chunks = 1;
    if (total_chunks > 4096) total_chunks = 4096;
    cfg.nblocks_p = (total_chunks + cfg.nsub - 1) / cfg.nsub;
    if (cfg.nblocks_p < 1) cfg.nblocks_p = 1;
    total_chunks  = cfg.nblocks_p * cfg.nsub;
    cfg.ppc       = (HW + total_chunks - 1) / total_chunks;
    if (cfg.ppc < 1) cfg.ppc = 1;
    return cfg;
}

// cache of channels_last weight copies (keyed by data pointer); the original
// tensor is kept alive so the address cannot be recycled by another buffer.
static at::Tensor get_cl_weight(const at::Tensor& w)
{
    static std::unordered_map<const void*, std::pair<at::Tensor, at::Tensor>> cache;
    const void* key = w.data_ptr();
    auto it = cache.find(key);
    if (it != cache.end()) {
        const at::Tensor& orig = it->second.first;
        if (orig.data_ptr() == w.data_ptr() &&
            orig.sizes() == w.sizes() &&
            orig.scalar_type() == w.scalar_type()) {
            return it->second.second;
        }
    }
    at::Tensor cl = w.contiguous(at::MemoryFormat::ChannelsLast);
    cache[key] = std::make_pair(w, cl);
    return cl;
}

// ------------------------------------------------------------- driver
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps)
{
    at::NoGradGuard nograd;
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    auto xc = x.is_contiguous() ? x : x.contiguous();
    const int B  = (int)xc.size(0);
    const int C  = (int)xc.size(1);
    const int H  = (int)xc.size(2);
    const int W  = (int)xc.size(3);
    TORCH_CHECK(C == CH, "specialized for C == 256");
    const int HW = H * W;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts   = xc.options();

    // ---- x -> NHWC ---------------------------------------------------------
    auto x_nhwc = at::empty({B, C, H, W}, opts, at::MemoryFormat::ChannelsLast);
    {
        dim3 grid((HW + TP - 1) / TP, C / TC, B);
        nchw2nhwc_kernel<<<grid, NTHR, 0, stream>>>(
            xc.data_ptr<float>(), x_nhwc.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = get_cl_weight(w1);
    auto w2c = get_cl_weight(w2);
    auto g1c = n1w.is_contiguous() ? n1w : n1w.contiguous();
    auto b1c = n1b.is_contiguous() ? n1b : n1b.contiguous();
    auto g2c = n2w.is_contiguous() ? n2w : n2w.contiguous();
    auto b2c = n2b.is_contiguous() ? n2b : n2b.contiguous();

    std::vector<int64_t> stride{1, 1}, pad{1, 1}, dil{1, 1};

    // ---- conv1 (cuDNN, NHWC) ----------------------------------------------
    auto t1 = at::conv2d(x_nhwc, w1c, {}, stride, pad, dil, 1);
    if (!t1.is_contiguous(at::MemoryFormat::ChannelsLast))
        t1 = t1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- L2-blocking configuration + tiny accumulator ----------------------
    ChunkCfg cfg    = pick_cfg(B, HW);
    const long sq_off = (long)B * NGROUPS;
    auto acc          = at::empty({2L * B * NGROUPS}, opts);
    const float invN  = 1.0f / (float)((long)HW * CPG);
    const float epsf  = (float)eps;
    const size_t accB = sizeof(float) * (size_t)(2L * B * NGROUPS);

    // ---- GN1 + SiLU (chunked: stats then apply per chunk) ------------------
    auto y1 = at::empty({B, C, H, W}, opts, at::MemoryFormat::ChannelsLast);
    C10_CUDA_CHECK(cudaMemsetAsync(acc.data_ptr<float>(), 0, accB, stream));
    for (int n0 = 0; n0 < B; n0 += cfg.nblk) {
        int nb = std::min(cfg.nblk, B - n0);
        for (int c0 = 0; c0 < CH; c0 += cfg.cblk) {
            gn_stats_kernel<<<dim3(cfg.nblocks_p, nb), NTHR, 0, stream>>>(
                t1.data_ptr<float>(), acc.data_ptr<float>(),
                HW, n0, c0, cfg.cblk, cfg.nsub, cfg.ppc, sq_off);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            gn_apply_silu_kernel<<<dim3((HW + 7) / 8, nb), cfg.cblk, 0, stream>>>(
                t1.data_ptr<float>(), y1.data_ptr<float>(),
                acc.data_ptr<float>(),
                g1c.data_ptr<float>(), b1c.data_ptr<float>(),
                HW, n0, c0, sq_off, invN, epsf);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // ---- conv2 (cuDNN, NHWC) ----------------------------------------------
    auto t2 = at::conv2d(y1, w2c, {}, stride, pad, dil, 1);
    if (!t2.is_contiguous(at::MemoryFormat::ChannelsLast))
        t2 = t2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GN2 + SiLU + residual + back to NCHW (chunked) --------------------
    auto out = at::empty({B, C, H, W}, opts);
    C10_CUDA_CHECK(cudaMemsetAsync(acc.data_ptr<float>(), 0, accB, stream));
    for (int n0 = 0; n0 < B; n0 += cfg.nblk) {
        int nb = std::min(cfg.nblk, B - n0);
        for (int c0 = 0; c0 < CH; c0 += cfg.cblk) {
            gn_stats_kernel<<<dim3(cfg.nblocks_p, nb), NTHR, 0, stream>>>(
                t2.data_ptr<float>(), acc.data_ptr<float>(),
                HW, n0, c0, cfg.cblk, cfg.nsub, cfg.ppc, sq_off);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            dim3 grid((HW + TP - 1) / TP, cfg.cblk / TC, nb);
            gn_apply_silu_res_nchw_kernel<<<grid, NTHR, 0, stream>>>(
                t2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
                acc.data_ptr<float>(),
                g2c.data_ptr<float>(), b2c.data_ptr<float>(),
                HW, n0, c0, sq_off, invN, epsf);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_v2_l2blk",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_block"],
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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


class ModelNew(nn.Module):
    """Fused Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual.

    L2 cache-blocked GroupNorm two-pass: stats and apply are issued per
    (batch-block x channel-block) chunk so the apply pass re-reads L2-resident
    data instead of DRAM. This module is stateless (all weights arrive as
    forward inputs), so there is no parameter holder to mirror.
    """

    def __init__(self):
        super().__init__()
        self._ext = _ext

    @staticmethod
    def _ref(x, w1, n1w, n1b, w2, n2w, n2b, eps):
        out = F.conv2d(x, w1, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n1w, bias=n1b, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, w2, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n2w, bias=n2b, eps=eps)
        out = F.silu(out)
        return out + x

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        ok = (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.size(1) == 256
            and conv1_weight.dtype == torch.float32
            and conv2_weight.dtype == torch.float32
        )
        if not ok:
            return self._ref(x, conv1_weight, norm1_weight, norm1_bias,
                             conv2_weight, norm2_weight, norm2_bias, eps_f)

        with torch.no_grad():
            return self._ext.fused_block(
                x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps_f,
            )
