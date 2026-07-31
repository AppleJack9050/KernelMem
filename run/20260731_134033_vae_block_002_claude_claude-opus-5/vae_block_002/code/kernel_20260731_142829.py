# =============================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY: (C) "fuse many ops into one/few kernels"
#
# Same fusion contract as before (K1..K5 custom kernels + 2 vendor convs).
# THIS REVISION: METHOD = stream_pipeline_overlap. The block is split along N into
# 2 chunks issued on 2 side streams, so one chunk's DRAM-bound GN/transpose kernels
# co-reside with the other chunk's compute-bound cuDNN TF32 conv mainloop.
#
# 1) Chosen granularity: (C).
# 2) Ops replaced by custom CUDA: NCHW<->NHWC layout conversions, both GroupNorm
#    statistic passes, both GroupNorm affine applications, both SiLU activations,
#    the residual add.
# 3) Fusion map (5 custom kernels + 2 vendor conv calls) per chunk:
#      K1 nchw2nhwc_kernel        : x(NCHW) -> x_nhwc (tiled smem transpose)
#      V1 cudnn_convolution(NHWC, TF32, autotuned) -> y1
#      K2 gn_partial_kernel       : deterministic per-(n,group) sum/sumsq partials (fp64)
#      K3 gn_finalize_kernel      : partials -> per-(n,c) {scale, shift}
#      K4 gn_silu_kernel          : GroupNorm-affine + SiLU, float4, NHWC->NHWC
#      V2 cudnn_convolution(NHWC, TF32, autotuned) -> y2
#      K2/K3 again for norm2
#      K5 gn_silu_res_t_kernel    : GN-affine + SiLU + residual + NHWC->NCHW transpose
# 4) Vendor kept: the 3x3 conv mainloop (cuDNN sm90 TF32 implicit GEMM) with per-shape
#    plan autotuning; weights' channels-last conversion; Python fallback.
#
# Precision: storage+arithmetic stay float32 (TF32 only inside conv, exactly as the
# reference which runs with cudnn.allow_tf32=True). All GroupNorm reductions accumulate
# in float64 in a fixed order -> deterministic, no atomics.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

constexpr int CC  = 256;          // channels (const in the SOL definition)
constexpr int GG  = 32;           // groups   (const in the SOL definition)
constexpr int CPG = CC / GG;      // 8 channels per group

// ---------------------------------------------------------------- K1: NCHW -> NHWC
__global__ void nchw2nhwc_kernel(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 int HW) {
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const float* s = src + (size_t)n * (size_t)CC * (size_t)HW;
    float*       d = dst + (size_t)n * (size_t)HW * (size_t)CC;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + ty + 8 * k;
        const int p = p0 + tx;
        float v = 0.f;
        if (p < HW) v = s[(size_t)c * (size_t)HW + p];
        tile[ty + 8 * k][tx] = v;            // tile[c_local][p_local]
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = p0 + ty + 8 * k;
        const int c = c0 + tx;
        if (p < HW) d[(size_t)p * (size_t)CC + c] = tile[tx][ty + 8 * k];
    }
}

// ------------------------------------------------- K2: per-(n,group) partial moments
// grid = (nchunks, Nc), block = 256 threads (one per channel) -> fully coalesced reads.
__global__ void gn_partial_kernel(const float* __restrict__ y,
                                  double2* __restrict__ part,
                                  int HW, int nchunks, int ppc) {
    const int n  = blockIdx.y;
    const int ch = blockIdx.x;
    const int c  = threadIdx.x;

    const int pstart = ch * ppc;
    int pend = pstart + ppc;
    if (pend > HW) pend = HW;

    const float* base = y + (size_t)n * (size_t)HW * (size_t)CC + c;

    double s = 0.0, ss = 0.0;
    int p = pstart;
    for (; p + 3 < pend; p += 4) {
        float v0 = base[(size_t)(p    ) * CC];
        float v1 = base[(size_t)(p + 1) * CC];
        float v2 = base[(size_t)(p + 2) * CC];
        float v3 = base[(size_t)(p + 3) * CC];
        s  += (double)v0 + (double)v1 + (double)v2 + (double)v3;
        ss += (double)v0 * (double)v0 + (double)v1 * (double)v1
            + (double)v2 * (double)v2 + (double)v3 * (double)v3;
    }
    for (; p < pend; ++p) {
        float v = base[(size_t)p * CC];
        s  += (double)v;
        ss += (double)v * (double)v;
    }

    // reduce the 8 lanes belonging to one group (groups are 8-lane aligned inside a warp)
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off);
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    if ((c & (CPG - 1)) == 0) {
        const int g = c >> 3;
        part[(size_t)(n * GG + g) * (size_t)nchunks + ch] = make_double2(s, ss);
    }
}

// ------------------------------------------- K3: partials -> per-(n,c) scale / shift
__global__ void gn_finalize_kernel(const double2* __restrict__ part,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ scale,
                                   float* __restrict__ shift,
                                   int nchunks, double M, double eps) {
    __shared__ double sm_s[128];
    __shared__ double sm_ss[128];
    const int ng  = blockIdx.x;          // n * GG + g   (n local to the chunk)
    const int tid = threadIdx.x;

    double s = 0.0, ss = 0.0;
    for (int i = tid; i < nchunks; i += 128) {
        double2 v = part[(size_t)ng * (size_t)nchunks + i];
        s  += v.x;
        ss += v.y;
    }
    sm_s[tid] = s; sm_ss[tid] = ss;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) { sm_s[tid] += sm_s[tid + st]; sm_ss[tid] += sm_ss[tid + st]; }
        __syncthreads();
    }
    if (tid < CPG) {
        const double mean = sm_s[0] / M;
        double var = sm_ss[0] / M - mean * mean;
        if (!(var > 0.0)) var = 0.0;
        const double rstd = 1.0 / sqrt(var + eps);
        const int g = ng % GG;
        const int n = ng / GG;
        const int c = g * CPG + tid;
        const double gm = (double)gamma[c];
        const double bt = (double)beta[c];
        scale[n * CC + c] = (float)(rstd * gm);
        shift[n * CC + c] = (float)(bt - mean * rstd * gm);
    }
}

// ------------------------------------------------ K4: GN-affine + SiLU (NHWC -> NHWC)
__global__ void gn_silu_kernel(const float* __restrict__ y,
                               const float* __restrict__ scale,
                               const float* __restrict__ shift,
                               float* __restrict__ out,
                               long HW) {
    __shared__ float ssc[CC];
    __shared__ float ssh[CC];
    const int n = blockIdx.y;
    const int tid = threadIdx.x;
    ssc[tid] = scale[n * CC + tid];
    ssh[tid] = shift[n * CC + tid];
    __syncthreads();

    const float4* yp = reinterpret_cast<const float4*>(y   + (size_t)n * (size_t)HW * CC);
    float4*       op = reinterpret_cast<float4*>      (out + (size_t)n * (size_t)HW * CC);
    const long nq = HW * (long)(CC / 4);

    for (long q = (long)blockIdx.x * blockDim.x + tid; q < nq;
         q += (long)gridDim.x * blockDim.x) {
        const int cq = (int)(q & (long)(CC / 4 - 1));
        const float4 v  = yp[q];
        const float sc0 = ssc[cq * 4 + 0], sh0 = ssh[cq * 4 + 0];
        const float sc1 = ssc[cq * 4 + 1], sh1 = ssh[cq * 4 + 1];
        const float sc2 = ssc[cq * 4 + 2], sh2 = ssh[cq * 4 + 2];
        const float sc3 = ssc[cq * 4 + 3], sh3 = ssh[cq * 4 + 3];
        float t0 = fmaf(v.x, sc0, sh0);
        float t1 = fmaf(v.y, sc1, sh1);
        float t2 = fmaf(v.z, sc2, sh2);
        float t3 = fmaf(v.w, sc3, sh3);
        float4 r;
        r.x = t0 / (1.0f + expf(-t0));
        r.y = t1 / (1.0f + expf(-t1));
        r.z = t2 / (1.0f + expf(-t2));
        r.w = t3 / (1.0f + expf(-t3));
        op[q] = r;
    }
}

// ------------ K5: GN-affine + SiLU + residual add + NHWC -> NCHW transpose (one pass)
__global__ void gn_silu_res_t_kernel(const float* __restrict__ y,
                                     const float* __restrict__ xres,   // NCHW contiguous
                                     const float* __restrict__ scale,
                                     const float* __restrict__ shift,
                                     float* __restrict__ out,          // NCHW contiguous
                                     int HW) {
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const float* yb = y + (size_t)n * (size_t)HW * (size_t)CC;
    const float* sc_b = scale + n * CC;
    const float* sh_b = shift + n * CC;

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = p0 + ty + 8 * k;
        const int c = c0 + tx;
        float t = 0.f;
        if (p < HW) {
            const float v = yb[(size_t)p * (size_t)CC + c];
            t = fmaf(v, sc_b[c], sh_b[c]);
            t = t / (1.0f + expf(-t));
        }
        tile[tx][ty + 8 * k] = t;          // tile[c_local][p_local]
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + ty + 8 * k;
        const int p = p0 + tx;
        if (p < HW) {
            const size_t off = ((size_t)n * CC + c) * (size_t)HW + p;
            out[off] = tile[ty + 8 * k][tx] + xres[off];
        }
    }
}

} // namespace

// ------------------------------------------------------------- cudnn_algo_autotune
// dispatch helpers. The (int, ...) overload binds only when the ATen symbol with the
// 9-argument (input, weight, padding, stride, dilation, groups, benchmark,
// deterministic, allow_tf32) signature exists; otherwise the (long, ...) fallback
// throws and the caller's catch restores the at::conv2d behaviour.
namespace detail {

template <class... Args>
auto cudnn_conv_dispatch(int, Args&&... args)
    -> decltype(at::cudnn_convolution(std::forward<Args>(args)...)) {
    return at::cudnn_convolution(std::forward<Args>(args)...);
}

template <class... Args>
at::Tensor cudnn_conv_dispatch(long, Args&&...) {
    throw std::runtime_error("at::cudnn_convolution unavailable with this signature");
}

template <class Ctx>
auto set_bm_limit(Ctx& c, int v, int) -> decltype(c.setBenchmarkLimitCuDNN(v), void()) {
    c.setBenchmarkLimitCuDNN(v);
}
template <class Ctx>
void set_bm_limit(Ctx&, int, long) {}

// plan item 4: keep the caching allocator from recycling a caller-stream allocation
// while a side stream is still reading/writing it.
inline void rec_stream(const at::Tensor& t, c10::cuda::CUDAStream s) {
    if (t.defined() && t.is_cuda() && t.has_storage()) {
        c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), s);
    }
}

} // namespace detail

// ------------------------------------------------------------------------------
// One batch chunk = the complete K1..K5 + 2 vendor conv chain, all launches issued
// on `stream`.  ATen/cuDNN work follows the CUDAStreamGuard installed by the caller.
// Plan item 9: every grid uses `ncs` (the chunk's batch size) for the N dimension.
// ------------------------------------------------------------------------------
static void run_chunk(const at::Tensor& xs,        // NCHW contiguous slice
                      at::Tensor&       xcl_s,     // NHWC slice (dst of K1)
                      at::Tensor&       out_s,     // NCHW slice (dst of K5)
                      at::Tensor&       part_s,    // {ncs*GG*nchunks, 2} fp64 slice
                      at::Tensor&       sc_s,      // {ncs, C}
                      at::Tensor&       sh_s,      // {ncs, C}
                      const at::Tensor& w1c,
                      const at::Tensor& n1w, const at::Tensor& n1b,
                      const at::Tensor& w2c,
                      const at::Tensor& n2w, const at::Tensor& n2b,
                      int64_t ncs, int64_t H, int64_t W,
                      int nchunks, int ppc, double M, double eps,
                      int silu_blocks, cudaStream_t stream) {
    const int64_t HW = H * W;
    auto opts = xs.options();
    std::vector<int64_t> stride{1, 1}, pad{1, 1}, dil{1, 1};

    // --- K1: single layout conversion for this chunk ------------------------
    {
        dim3 blk(32, 8);
        dim3 grd((unsigned)((HW + 31) / 32), (unsigned)(CC / 32), (unsigned)ncs);
        nchw2nhwc_kernel<<<grd, blk, 0, stream>>>(xs.data_ptr<float>(),
                                                  xcl_s.data_ptr<float>(), (int)HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // --- conv1 (cuDNN v8, NHWC TF32, autotuned) -----------------------------
    at::Tensor y1;
    try {
        y1 = detail::cudnn_conv_dispatch(0, xcl_s, w1c, pad, stride, dil,
                                         (int64_t)1,
                                         /*benchmark=*/true,
                                         /*deterministic=*/false,
                                         /*allow_tf32=*/true);
    } catch (const std::exception&) {
        y1 = at::conv2d(xcl_s, w1c, {}, stride, pad, dil, 1);
    }
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    // --- K2 + K3: GroupNorm statistics for norm1 ----------------------------
    gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)ncs), 256, 0, stream>>>(
        y1.data_ptr<float>(), reinterpret_cast<double2*>(part_s.data_ptr<double>()),
        (int)HW, nchunks, ppc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gn_finalize_kernel<<<(unsigned)(ncs * GG), 128, 0, stream>>>(
        reinterpret_cast<const double2*>(part_s.data_ptr<double>()),
        n1w.data_ptr<float>(), n1b.data_ptr<float>(),
        sc_s.data_ptr<float>(), sh_s.data_ptr<float>(), nchunks, M, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // --- K4: GN + SiLU fused ------------------------------------------------
    auto z = at::empty({ncs, (int64_t)CC, H, W},
                       opts.memory_format(at::MemoryFormat::ChannelsLast));
    gn_silu_kernel<<<dim3((unsigned)silu_blocks, (unsigned)ncs), 256, 0, stream>>>(
        y1.data_ptr<float>(), sc_s.data_ptr<float>(), sh_s.data_ptr<float>(),
        z.data_ptr<float>(), (long)HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // --- conv2 (cuDNN v8, NHWC TF32, autotuned) -----------------------------
    at::Tensor y2;
    try {
        y2 = detail::cudnn_conv_dispatch(0, z, w2c, pad, stride, dil,
                                         (int64_t)1,
                                         /*benchmark=*/true,
                                         /*deterministic=*/false,
                                         /*allow_tf32=*/true);
    } catch (const std::exception&) {
        y2 = at::conv2d(z, w2c, {}, stride, pad, dil, 1);
    }
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    // --- K2 + K3: GroupNorm statistics for norm2 ----------------------------
    gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)ncs), 256, 0, stream>>>(
        y2.data_ptr<float>(), reinterpret_cast<double2*>(part_s.data_ptr<double>()),
        (int)HW, nchunks, ppc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gn_finalize_kernel<<<(unsigned)(ncs * GG), 128, 0, stream>>>(
        reinterpret_cast<const double2*>(part_s.data_ptr<double>()),
        n2w.data_ptr<float>(), n2b.data_ptr<float>(),
        sc_s.data_ptr<float>(), sh_s.data_ptr<float>(), nchunks, M, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // --- K5: GN + SiLU + residual + transpose back to NCHW ------------------
    {
        dim3 blk(32, 8);
        dim3 grd((unsigned)((HW + 31) / 32), (unsigned)(CC / 32), (unsigned)ncs);
        gn_silu_res_t_kernel<<<grd, blk, 0, stream>>>(
            y2.data_ptr<float>(), xs.data_ptr<float>(),
            sc_s.data_ptr<float>(), sh_s.data_ptr<float>(),
            out_s.data_ptr<float>(), (int)HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// ------------------------------------------------------------------ host entry point
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight,
                          torch::Tensor norm1_weight,
                          torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight,
                          torch::Tensor norm2_weight,
                          torch::Tensor norm2_bias,
                          double eps) {
    at::NoGradGuard no_grad;

    // --- plan item 8: one-time global cuDNN autotune switch (unchanged) ------
    static const bool _bm_init = [](){
        at::globalContext().setBenchmarkCuDNN(true);
        detail::set_bm_limit(at::globalContext(), 16, 0);
        return true;
    }();
    (void)_bm_init;

    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "expected NCHW");
    TORCH_CHECK(x.size(1) == CC, "specialised for C=256");

    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t HW = H * W;

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto opts = xc.options();
    auto cur = at::cuda::getCurrentCUDAStream();
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    // --- plan item 3: function-local static side-stream pool -----------------
    static std::vector<c10::cuda::CUDAStream> pool{ c10::cuda::getStreamFromPool(),
                                                    c10::cuda::getStreamFromPool() };

    // --- plan item 1: chunk-eligibility guard --------------------------------
    // Keeps per-chunk conv CTA count large enough that wave quantization does not
    // regress the mainloop; small / awkward shapes keep the exact old schedule.
    int nsplit = (N >= 4 && (N / 2) * HW >= 32768) ? 2 : 1;
    if (nsplit == 2 && (pool[0].device_index() != x.get_device() ||
                        pool[1].device_index() != x.get_device())) {
        nsplit = 1;   // stream pool belongs to another device -> stay single-stream
    }

    // plan item 7: weight / affine-param preparation hoisted out of the chunk loop
    auto w1c = conv1_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = conv2_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto n1w = norm1_weight.contiguous();
    auto n1b = norm1_bias.contiguous();
    auto n2w = norm2_weight.contiguous();
    auto n2b = norm2_bias.contiguous();

    // --- plan item 4: reduction schedule derived from the CHUNK's batch size --
    const int64_t Nc = N / nsplit;                    // base chunk batch
    const int min_ppc = 32;
    long nc = ((long)8 * sms + Nc - 1) / Nc;
    const long max_nc = (HW + min_ppc - 1) / min_ppc;
    if (nc > max_nc) nc = max_nc;
    if (nc < 1) nc = 1;
    const int ppc     = (int)((HW + nc - 1) / nc);
    const int nchunks = (int)((HW + ppc - 1) / ppc);

    // --- plan item 4: all full-size buffers allocated on the caller stream ----
    auto x_cl = at::empty({N, C, H, W},
                          opts.memory_format(at::MemoryFormat::ChannelsLast));
    auto out   = at::empty({N, C, H, W}, opts);
    auto part  = at::empty({(int64_t)N * GG * nchunks, 2}, opts.dtype(at::kDouble));
    auto scale = at::empty({N, C}, opts);
    auto shift = at::empty({N, C}, opts);
    const double M = (double)CPG * (double)HW;

    const long nq = HW * (long)(CC / 4);
    int silu_blocks = (int)std::min<long>((nq + 255) / 256, (long)(6 * sms));
    if (silu_blocks < 1) silu_blocks = 1;

    // ---------------- single-stream path: byte-identical to the base ---------
    if (nsplit == 1) {
        run_chunk(xc, x_cl, out, part, scale, shift,
                  w1c, n1w, n1b, w2c, n2w, n2b,
                  N, H, W, nchunks, ppc, M, eps, silu_blocks, cur.stream());
        return out;
    }

    // ------------------------------------------------------------------------
    // SEMANTIC PROOF (plan item 2): GroupNorm reduces over (C/G, H, W) *within one
    // sample*, and the residual add is elementwise, so slicing the batch axis is an
    // exact rewrite -- no value crosses a chunk boundary. The statistic buffers stay
    // disjoint because `part` is indexed (n*GG+g)*nchunks+ch and `scale`/`shift` by
    // n*CC+c, hence narrow(0, n0*GG*nchunks, ...) / narrow(0, n0, nc) slices of the
    // two chunks never alias. narrow() on dim 0 also preserves channels-last
    // contiguity, so cuDNN still sees a plain NHWC tensor.
    // ------------------------------------------------------------------------
    at::cuda::CUDAEvent start;
    start.record(cur);

    at::cuda::CUDAEvent done[2];
    int64_t n0 = 0;
    for (int i = 0; i < nsplit; ++i) {
        const int64_t ncs = (i == nsplit - 1) ? (N - n0) : Nc;

        // side stream waits for the caller's inputs / allocations
        start.block(pool[i]);

        // plan items 4 & 7: protect caller-stream allocations used on this stream
        detail::rec_stream(xc,    pool[i]);
        detail::rec_stream(x_cl,  pool[i]);
        detail::rec_stream(out,   pool[i]);
        detail::rec_stream(part,  pool[i]);
        detail::rec_stream(scale, pool[i]);
        detail::rec_stream(shift, pool[i]);
        detail::rec_stream(w1c,   pool[i]);
        detail::rec_stream(w2c,   pool[i]);
        detail::rec_stream(n1w,   pool[i]);
        detail::rec_stream(n1b,   pool[i]);
        detail::rec_stream(n2w,   pool[i]);
        detail::rec_stream(n2b,   pool[i]);

        // plan item 5: per-chunk views
        auto xs     = xc.narrow(0, n0, ncs);
        auto xcl_s  = x_cl.narrow(0, n0, ncs);
        auto out_s  = out.narrow(0, n0, ncs);
        auto part_s = part.narrow(0, n0 * GG * nchunks, ncs * GG * nchunks);
        auto sc_s   = scale.narrow(0, n0, ncs);
        auto sh_s   = shift.narrow(0, n0, ncs);

        {   // ATen/cuDNN work is redirected to pool[i] by this guard
            c10::cuda::CUDAStreamGuard g(pool[i]);
            run_chunk(xs, xcl_s, out_s, part_s, sc_s, sh_s,
                      w1c, n1w, n1b, w2c, n2w, n2b,
                      ncs, H, W, nchunks, ppc, M, eps, silu_blocks,
                      pool[i].stream());
            done[i].record(pool[i]);
        }
        n0 += ncs;
    }

    // --- plan item 6: join back onto the caller stream -----------------------
    for (int i = 0; i < nsplit; ++i) done[i].block(cur);

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
    name="vae_resblock_fused_v3_streams",
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


class ModelNew(nn.Module):
    """See the file header for the granularity / fusion contract (granularity C)."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

    @torch.no_grad()
    def forward(self,
                x,
                conv1_weight,
                norm1_weight,
                norm1_bias,
                conv2_weight,
                norm2_weight,
                norm2_bias,
                eps):
        e = float(eps)
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256 and conv1_weight.dtype == torch.float32
                and conv2_weight.dtype == torch.float32):
            return self._ext.fused_block(x, conv1_weight, norm1_weight, norm1_bias,
                                         conv2_weight, norm2_weight, norm2_bias, e)

        # ---- fallback: exact reference semantics for unsupported configurations ----
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=e)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=e)
        out = F.silu(out)
        return out + residual
