# =============================================================================
# ModelNew — SOL problem 002: Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual
#
# Base granularity (C) preserved: GroupNorm+SiLU (+residual add, +NHWC->NCHW
# transpose) fused into custom CUDA kernels; the two 3x3 convolutions stay on
# cuDNN (channels_last / TF32), exactly as in the base kernel.
#
# THIS REVISION: CUDA_Graph_Capture_Replay_StaticBuffers.
#   The 180 per-iteration kernel launches (48 cuDNN convertTensor, 36
#   elementwise, 24 conv, 72 GN) are collapsed into a single graph replay.
#   Zero-copy policy: the caller's own tensors are adopted as the graph's static
#   buffers (kept alive by strong references, validated by data_ptr identity), so
#   the replay path performs NO device-to-device input copies.
#
# Precision policy: unchanged — fp32 storage + fp32 arithmetic, fp32 reduction
# accumulators; TF32 for conv inherited from the default torch/cuDNN setting.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

#define CPG   8          // channels per group (C=256, G=32)
#define VPP   2          // float4 vectors per pixel per group (CPG/4)
#define NTHR  256
#define PCHUNK 128       // pixels processed per shared-memory tile (NTHR/VPP)

__device__ __forceinline__ void warpReduce2(float &a, float &b) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, off);
        b += __shfl_down_sync(0xffffffffu, b, off);
    }
}

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// Pass 1: partial sums / sums-of-squares over a pixel chunk of one (n,g) group.
// grid = (K, N*G), block = 256.   Input is channels-last (NHWC).
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ inp,
                                float* __restrict__ partial,
                                int HW, int C, int G, int chunk, int K)
{
    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    const float* base = inp + (long long)n * HW * C + (long long)g * CPG;

    const int tid     = threadIdx.x;
    const int j       = tid & (VPP - 1);
    const int p0      = tid / VPP;
    const int pstride = NTHR / VPP;

    float s = 0.f, ss = 0.f;
    for (int p = pstart + p0; p < pend; p += pstride) {
        const float4 v = *reinterpret_cast<const float4*>(base + (long long)p * C + j * 4);
        s  += v.x + v.y + v.z + v.w;
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    __shared__ float sa[NTHR / 32];
    __shared__ float sb[NTHR / 32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    warpReduce2(s, ss);
    if (lane == 0) { sa[wid] = s; sb[wid] = ss; }
    __syncthreads();
    if (wid == 0) {
        const int nw = NTHR / 32;
        s  = (lane < nw) ? sa[lane] : 0.f;
        ss = (lane < nw) ? sb[lane] : 0.f;
        warpReduce2(s, ss);
        if (lane == 0) {
            partial[((long long)ng * K + blk) * 2 + 0] = s;
            partial[((long long)ng * K + blk) * 2 + 1] = ss;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: finalize (mean, rstd) per (n,g).  grid = N*G, block = 32.
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ stats,
                                   int K, float invcount, float eps)
{
    const int ng = blockIdx.x;
    float s = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < K; i += 32) {
        s  += partial[((long long)ng * K + i) * 2 + 0];
        ss += partial[((long long)ng * K + i) * 2 + 1];
    }
    warpReduce2(s, ss);
    if (threadIdx.x == 0) {
        const float mean = s * invcount;
        float var = ss * invcount - mean * mean;
        if (var < 0.f) var = 0.f;
        stats[ng * 2 + 0] = mean;
        stats[ng * 2 + 1] = rsqrtf(var + eps);
    }
}

// ---------------------------------------------------------------------------
// Pass 3a: normalize * gamma + beta -> SiLU, NHWC in / NHWC out.
// grid = (K, N*G), block = 256.
// ---------------------------------------------------------------------------
__global__ void gn_silu_apply_nhwc(const float* __restrict__ inp,
                                   float* __restrict__ out,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ stats,
                                   int HW, int C, int G, int chunk)
{
    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const float mean = stats[ng * 2 + 0];
    const float rstd = stats[ng * 2 + 1];

    const int tid     = threadIdx.x;
    const int j       = tid & (VPP - 1);
    const int p0      = tid / VPP;
    const int pstride = NTHR / VPP;

    const float4 gm = *reinterpret_cast<const float4*>(gamma + g * CPG + j * 4);
    const float4 bt = *reinterpret_cast<const float4*>(beta  + g * CPG + j * 4);

    const float* ibase = inp + (long long)n * HW * C + (long long)g * CPG;
    float*       obase = out + (long long)n * HW * C + (long long)g * CPG;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    for (int p = pstart + p0; p < pend; p += pstride) {
        const long long off = (long long)p * C + j * 4;
        const float4 v = *reinterpret_cast<const float4*>(ibase + off);
        float4 o;
        o.x = silu_f((v.x - mean) * rstd * gm.x + bt.x);
        o.y = silu_f((v.y - mean) * rstd * gm.y + bt.y);
        o.z = silu_f((v.z - mean) * rstd * gm.z + bt.z);
        o.w = silu_f((v.w - mean) * rstd * gm.w + bt.w);
        *reinterpret_cast<float4*>(obase + off) = o;
    }
}

// ---------------------------------------------------------------------------
// Pass 3b: normalize -> SiLU -> + residual, NHWC in / NCHW contiguous out.
// On-chip transpose through shared memory keeps both loads and stores coalesced.
// grid = (K, N*G), block = 256 (must equal NTHR).
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_apply_nchw(const float* __restrict__ inp,
                                       const float* __restrict__ res,
                                       float* __restrict__ out,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ stats,
                                       int HW, int C, int G, int chunk)
{
    __shared__ float sh[CPG * (PCHUNK + 1)];

    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const float mean = stats[ng * 2 + 0];
    const float rstd = stats[ng * 2 + 1];

    const int tid = threadIdx.x;
    const int j   = tid & (VPP - 1);   // which float4 inside the group
    const int pl  = tid / VPP;         // local pixel index (0..127)

    const float4 gm = *reinterpret_cast<const float4*>(gamma + g * CPG + j * 4);
    const float4 bt = *reinterpret_cast<const float4*>(beta  + g * CPG + j * 4);

    const float* ibase = inp + (long long)n * HW * C + (long long)g * CPG;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    for (int p0 = pstart; p0 < pend; p0 += PCHUNK) {
        const int p = p0 + pl;
        if (p < pend) {
            const float4 v = *reinterpret_cast<const float4*>(ibase + (long long)p * C + j * 4);
            const int c0 = j * 4;
            sh[(c0 + 0) * (PCHUNK + 1) + pl] = silu_f((v.x - mean) * rstd * gm.x + bt.x);
            sh[(c0 + 1) * (PCHUNK + 1) + pl] = silu_f((v.y - mean) * rstd * gm.y + bt.y);
            sh[(c0 + 2) * (PCHUNK + 1) + pl] = silu_f((v.z - mean) * rstd * gm.z + bt.z);
            sh[(c0 + 3) * (PCHUNK + 1) + pl] = silu_f((v.w - mean) * rstd * gm.w + bt.w);
        }
        __syncthreads();

        const int cnt = (pend - p0) < PCHUNK ? (pend - p0) : PCHUNK;
#pragma unroll
        for (int it = 0; it < (CPG * PCHUNK) / NTHR; ++it) {
            const int idx = tid + it * NTHR;
            const int c   = idx / PCHUNK;
            const int lp  = idx - c * PCHUNK;
            if (lp < cnt) {
                const long long oidx =
                    ((long long)(n * C + g * CPG + c)) * HW + (long long)(p0 + lp);
                out[oidx] = sh[c * (PCHUNK + 1) + lp] + res[oidx];
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static inline void pick_chunk(long long NG, int HW, int &chunk, int &K)
{
    const long long target_blocks = 2048;
    int k = (int)((target_blocks + NG - 1) / NG);
    if (k < 1) k = 1;
    int per = (HW + k - 1) / k;
    chunk = ((per + PCHUNK - 1) / PCHUNK) * PCHUNK;
    if (chunk < PCHUNK) chunk = PCHUNK;
    K = (HW + chunk - 1) / chunk;
}

static void run_stats(const torch::Tensor &inp, torch::Tensor &stats,
                      int N, int C, int G, int HW, int chunk, int K, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const long long NG = (long long)N * G;
    auto partial = torch::empty({NG * K * 2}, inp.options());

    dim3 grid(K, (unsigned)NG);
    gn_stats_kernel<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), partial.data_ptr<float>(), HW, C, G, chunk, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float invcount = 1.0f / (float)((long long)CPG * HW);
    gn_finalize_kernel<<<(unsigned)NG, 32, 0, stream>>>(
        partial.data_ptr<float>(), stats.data_ptr<float>(), K, invcount, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_nhwc(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, double eps)
{
    TORCH_CHECK(inp.is_cuda(), "input must be CUDA");
    TORCH_CHECK(inp.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(inp.dim() == 4, "4D only");
    TORCH_CHECK(inp.is_contiguous(at::MemoryFormat::ChannelsLast), "need channels_last");

    const int N = (int)inp.size(0), C = (int)inp.size(1);
    const int H = (int)inp.size(2), W = (int)inp.size(3);
    const int G = 32, HW = H * W;
    TORCH_CHECK(C % G == 0 && (C / G) == CPG, "unsupported channel/group config");

    auto out = torch::empty_like(inp);           // preserves channels_last
    auto stats = torch::empty({(long long)N * G * 2}, inp.options());

    int chunk = 0, K = 0;
    pick_chunk((long long)N * G, HW, chunk, K);
    run_stats(inp, stats, N, C, G, HW, chunk, K, eps);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(K, (unsigned)((long long)N * G));
    gn_silu_apply_nhwc<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        stats.data_ptr<float>(), HW, C, G, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gn_silu_res_nchw(torch::Tensor inp, torch::Tensor gamma,
                               torch::Tensor beta, torch::Tensor res, double eps)
{
    TORCH_CHECK(inp.is_cuda() && res.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(inp.scalar_type() == torch::kFloat32 &&
                res.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(inp.is_contiguous(at::MemoryFormat::ChannelsLast), "need channels_last");
    TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");

    const int N = (int)inp.size(0), C = (int)inp.size(1);
    const int H = (int)inp.size(2), W = (int)inp.size(3);
    const int G = 32, HW = H * W;
    TORCH_CHECK(C % G == 0 && (C / G) == CPG, "unsupported channel/group config");

    auto out = torch::empty({N, C, H, W}, inp.options());   // contiguous NCHW
    auto stats = torch::empty({(long long)N * G * 2}, inp.options());

    int chunk = 0, K = 0;
    pick_chunk((long long)N * G, HW, chunk, K);
    run_stats(inp, stats, N, C, G, HW, chunk, K, eps);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(K, (unsigned)((long long)N * G));
    gn_silu_res_apply_nchw<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), res.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        stats.data_ptr<float>(), HW, C, G, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor gn_silu_nhwc(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, double eps);
torch::Tensor gn_silu_res_nchw(torch::Tensor inp, torch::Tensor gamma,
                               torch::Tensor beta, torch::Tensor res, double eps);
'''

_ext = load_inline(
    name="vae_resblock_gn_silu_fused",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu_nhwc", "gn_silu_res_nchw"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
    ],
)


class ModelNew(nn.Module):
    """
    Granularity (C): GroupNorm+SiLU (+residual add, +NHWC->NCHW transpose) fused
    into custom CUDA kernels; the two 3x3 convolutions stay on cuDNN
    (channels_last).  The whole fused pipeline is additionally captured into a
    CUDA graph with a zero-copy static-buffer policy: the caller's tensors ARE
    the static buffers (pinned by strong references + data_ptr identity check),
    so replay performs no per-iteration input copies.
    """

    _MAX_RECAPTURES = 2

    def __init__(self):
        super().__init__()
        # --- CUDA graph state (plan item 2) ---
        self._graph = None
        self._captured = False
        self._graph_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._static_refs = None      # strong refs to the 7 captured input tensors
        self._static_ptrs = None      # their data_ptr() values (address identity)
        self._static_out = None       # graph-pool output buffer
        self._sig = None              # shape/dtype/device/eps signature
        self._recapture_count = 0
        self._graph_disabled = False

    # ------------------------------------------------------------------
    # Fused fast path (verbatim body of the base kernel's fast path).
    # ------------------------------------------------------------------
    def _run_fused(self, x, w1, g1, b1, w2, g2, b2, eps_f):
        cl = torch.channels_last

        # Residual must stay untouched, contiguous NCHW.
        res = x if x.is_contiguous() else x.contiguous()

        xc = x if x.is_contiguous(memory_format=cl) else x.contiguous(memory_format=cl)
        w1c = w1 if w1.is_contiguous(memory_format=cl) else w1.contiguous(memory_format=cl)
        w2c = w2 if w2.is_contiguous(memory_format=cl) else w2.contiguous(memory_format=cl)

        g1c = g1 if g1.is_contiguous() else g1.contiguous()
        b1c = b1 if b1.is_contiguous() else b1.contiguous()
        g2c = g2 if g2.is_contiguous() else g2.contiguous()
        b2c = b2 if b2.is_contiguous() else b2.contiguous()

        y = F.conv2d(xc, w1c, None, 1, 1)
        if not y.is_contiguous(memory_format=cl):
            y = y.contiguous(memory_format=cl)
        y = _ext.gn_silu_nhwc(y, g1c, b1c, eps_f)

        y = F.conv2d(y, w2c, None, 1, 1)
        if not y.is_contiguous(memory_format=cl):
            y = y.contiguous(memory_format=cl)

        return _ext.gn_silu_res_nchw(y, g2c, b2c, res, eps_f)

    # ------------------------------------------------------------------
    # Pure-PyTorch reference fallback (unsupported configs).
    # ------------------------------------------------------------------
    @staticmethod
    def _pytorch_fallback(x, w1, g1, b1, w2, g2, b2, eps_f, num_groups):
        res = x if x.is_contiguous() else x.contiguous()
        out = F.conv2d(x, w1, None, 1, 1)
        out = F.silu(F.group_norm(out, num_groups, g1, b1, eps_f))
        out = F.conv2d(out, w2, None, 1, 1)
        out = F.silu(F.group_norm(out, num_groups, g2, b2, eps_f))
        return out + res

    # ------------------------------------------------------------------
    # Signature / address helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _make_sig(tensors, eps_f):
        return tuple(
            (tuple(t.shape), t.dtype, t.device, t.stride()) for t in tensors
        ) + (eps_f,)

    @staticmethod
    def _make_ptrs(tensors):
        return tuple(t.data_ptr() for t in tensors)

    # ------------------------------------------------------------------
    # Graph capture with zero-copy static buffers.
    # ------------------------------------------------------------------
    def _capture(self, tensors, eps_f, sig):
        self._static_refs = tuple(tensors)
        self._static_ptrs = self._make_ptrs(self._static_refs)

        s = self._graph_stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self._run_fused(*self._static_refs, eps_f)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_out = self._run_fused(*self._static_refs, eps_f)
        torch.cuda.synchronize()

        self._captured = True
        self._sig = sig

    # ------------------------------------------------------------------
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
        eps_f = eps if isinstance(eps, float) else float(eps)

        C = x.size(1)
        num_groups = 32

        supported = (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and C % num_groups == 0
            and (C // num_groups) == 8
        )

        if not supported:
            return self._pytorch_fallback(x, conv1_weight, norm1_weight, norm1_bias,
                                          conv2_weight, norm2_weight, norm2_bias,
                                          eps_f, num_groups)

        tensors = (x, conv1_weight, norm1_weight, norm1_bias,
                   conv2_weight, norm2_weight, norm2_bias)

        if self.training or self._graph_disabled or self._graph_stream is None:
            return self._run_fused(x, conv1_weight, norm1_weight, norm1_bias,
                                   conv2_weight, norm2_weight, norm2_bias, eps_f)

        sig = self._make_sig(tensors, eps_f)
        ptrs = self._make_ptrs(tensors)

        if (self._captured
                and sig == self._sig
                and ptrs == self._static_ptrs):
            self._graph.replay()
            return self._static_out

        self._recapture_count += 1
        if self._recapture_count > self._MAX_RECAPTURES:
            self._graph_disabled = True
            self._graph = None
            self._static_refs = None
            self._static_ptrs = None
            self._static_out = None
            self._captured = False
            return self._run_fused(x, conv1_weight, norm1_weight, norm1_bias,
                                   conv2_weight, norm2_weight, norm2_bias, eps_f)

        self._captured = False
        try:
            self._capture(tensors, eps_f, sig)
        except Exception:
            # Any capture failure -> permanently fall back to eager fused path.
            self._graph_disabled = True
            self._graph = None
            self._static_refs = None
            self._static_ptrs = None
            self._static_out = None
            self._captured = False
            return self._run_fused(x, conv1_weight, norm1_weight, norm1_bias,
                                   conv2_weight, norm2_weight, norm2_bias, eps_f)

        # Capture only RECORDS work; the graph-pool output is still
        # uninitialized.  Execute the graph once so the returned buffer holds
        # real results.
        self._graph.replay()
        return self._static_out
