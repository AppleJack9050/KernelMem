# =============================================================================
# ModelNew : fused VAE residual block
#            Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual
#
# Base fusion (unchanged):
#      K0 nchw2nhwc_kernel        : shared-mem tiled transpose of x  (once)
#      -- at::conv2d (cuDNN, NHWC/TF32, called from inside the extension)
#      K1 gn_stats_kernel         : partial sum / sumsq per (batch,group)
#      K2 gn_finalize_kernel      : partials -> mean, rstd
#      K3 gn_apply_silu_kernel    : (x-mean)*rstd*g+b  then SiLU, NHWC->NHWC
#      -- at::conv2d (cuDNN, NHWC/TF32)
#      K1,K2 again for the 2nd GroupNorm
#      K4 gn_apply_silu_res_nchw_kernel : affine + SiLU + residual + NHWC->NCHW
#
# THIS REVISION: CUDA_Graph_Capture_Replay_StaticBuffers.
#   The forward is launch/dispatch bound (242 host dispatches per call, many of
#   them tiny: gn_finalize at 3.1 us / 0.25 waves, 88 cuDNN convertTensor).
#   Per-kernel DRAM SOL is already 77.8-88.7%, so the remaining wall-clock is
#   host dispatch + inter-kernel gaps.  We therefore capture the ENTIRE forward
#   (transpose, both cuDNN convs, both GN stat/finalize/apply passes, residual)
#   into a per-shape CUDA Graph over static input/output buffers and replay it:
#   242 dispatches -> 1 graph launch.  No CUDA kernel code, launch geometry,
#   register usage or shared-memory footprint is changed, so numerics are
#   bit-identical to the base kernel.
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
#include <vector>
#include <math.h>

#define CH       256
#define NGROUPS  32
#define CPG      8
#define TP       64
#define TC       64
#define NTHR     256

// ---------------------------------------------------------------- transpose
// NCHW (contiguous) -> NHWC (channels_last), shared memory tiled.
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
// grid (nchunk, B), block 256 threads == 256 channels.
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ part,
                                int HW, int nchunk, int ppc, long sq_off)
{
    const int b     = blockIdx.y;
    const int chunk = blockIdx.x;
    const int c     = threadIdx.x;

    int pstart = chunk * ppc;
    if (pstart >= HW) return;
    int pend = pstart + ppc;
    if (pend > HW) pend = HW;
    int n = pend - pstart;

    const float* px = x + (long)b * HW * CH + (long)pstart * CH + c;

    float s = 0.f, s2 = 0.f;
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

    // reduce over the 8 lanes that belong to the same group
    #pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off, 8);
        s2 += __shfl_down_sync(0xffffffffu, s2, off, 8);
    }
    if ((c & 7) == 0) {
        int g  = c >> 3;
        long o = ((long)b * NGROUPS + g) * (long)nchunk + chunk;
        part[o]          = s;
        part[sq_off + o] = s2;
    }
}

// ------------------------------------------------------------ GN finalize
__global__ void gn_finalize_kernel(const float* __restrict__ part,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nchunk, long sq_off,
                                   float invN, float eps)
{
    __shared__ float sh1[NTHR];
    __shared__ float sh2[NTHR];
    const int bg  = blockIdx.x;
    const int tid = threadIdx.x;

    float s = 0.f, s2 = 0.f;
    for (int i = tid; i < nchunk; i += NTHR) {
        s  += part[(long)bg * nchunk + i];
        s2 += part[sq_off + (long)bg * nchunk + i];
    }
    sh1[tid] = s;
    sh2[tid] = s2;
    __syncthreads();
    for (int st = NTHR / 2; st > 0; st >>= 1) {
        if (tid < st) {
            sh1[tid] += sh1[tid + st];
            sh2[tid] += sh2[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float m = sh1[0] * invN;
        float v = sh2[0] * invN - m * m;
        if (!(v > 0.f)) v = 0.f;
        mean[bg] = m;
        rstd[bg] = rsqrtf(v + eps);
    }
}

// -------------------------------------------- GN affine + SiLU (NHWC -> NHWC)
// grid (ceil(HW/8), B), block 256 threads == 256 channels
__global__ void gn_apply_silu_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ rstd,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     int HW)
{
    const int b = blockIdx.y;
    const int c = threadIdx.x;
    const int g = c >> 3;
    const float m  = mean[b * NGROUPS + g];
    const float r  = rstd[b * NGROUPS + g];
    const float sc = gamma[c] * r;
    const float sh = beta[c] - m * sc;

    const int p0 = blockIdx.x * 8;
    if (p0 >= HW) return;
    long off = (long)b * HW * CH + (long)p0 * CH + c;

    if (p0 + 8 <= HW) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = x[off + (long)i * CH] * sc + sh;
            y[off + (long)i * CH] = v / (1.f + expf(-v));
        }
    } else {
        for (int i = 0; p0 + i < HW; ++i) {
            float v = x[off + (long)i * CH] * sc + sh;
            y[off + (long)i * CH] = v / (1.f + expf(-v));
        }
    }
}

// ----------- GN affine + SiLU + residual add + NHWC -> NCHW (one pass)
__global__ void gn_apply_silu_res_nchw_kernel(const float* __restrict__ x,
                                              const float* __restrict__ res,
                                              float* __restrict__ out,
                                              const float* __restrict__ mean,
                                              const float* __restrict__ rstd,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              int HW)
{
    __shared__ float sm[TC * (TP + 1)];
    __shared__ float ssc[TC];
    __shared__ float ssh[TC];

    const int b   = blockIdx.z;
    const int c0  = blockIdx.y * TC;
    const int p0  = blockIdx.x * TP;
    const int tid = threadIdx.x;

    if (tid < TC) {
        int c   = c0 + tid;
        int g   = c >> 3;
        float r = rstd[b * NGROUPS + g];
        float m = mean[b * NGROUPS + g];
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
            float v = x[((long)b * HW + p) * CH + (c0 + cl)] * ssc[cl] + ssh[cl];
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
            long o = ((long)b * CH + (c0 + cl)) * (long)HW + p;
            out[o] = sm[cl * (TP + 1) + pl] + res[o];
        }
    }
}

// ------------------------------------------------------------- driver
static void run_groupnorm_stats(const at::Tensor& t, at::Tensor& part,
                                at::Tensor& mean, at::Tensor& rstd,
                                int B, int HW, int nchunk, float eps,
                                cudaStream_t stream)
{
    int ppc = (HW + nchunk - 1) / nchunk;
    long sq_off = (long)B * NGROUPS * nchunk;

    dim3 g1(nchunk, B);
    gn_stats_kernel<<<g1, NTHR, 0, stream>>>(
        t.data_ptr<float>(), part.data_ptr<float>(), HW, nchunk, ppc, sq_off);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float invN = 1.0f / (float)((long)HW * CPG);
    gn_finalize_kernel<<<B * NGROUPS, NTHR, 0, stream>>>(
        part.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        nchunk, sq_off, invN, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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

    auto w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1c = n1w.is_contiguous() ? n1w : n1w.contiguous();
    auto b1c = n1b.is_contiguous() ? n1b : n1b.contiguous();
    auto g2c = n2w.is_contiguous() ? n2w : n2w.contiguous();
    auto b2c = n2b.is_contiguous() ? n2b : n2b.contiguous();

    std::vector<int64_t> stride{1, 1}, pad{1, 1}, dil{1, 1};

    // ---- conv1 (cuDNN, NHWC) ----------------------------------------------
    auto t1 = at::conv2d(x_nhwc, w1c, {}, stride, pad, dil, 1);
    if (!t1.is_contiguous(at::MemoryFormat::ChannelsLast))
        t1 = t1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- reduction workspace ----------------------------------------------
    int nchunk = HW / 16;
    if (nchunk < 1) nchunk = 1;
    int cap = 2048 / (B > 0 ? B : 1);
    if (cap < 1) cap = 1;
    if (nchunk > cap) nchunk = cap;
    if (nchunk > 2048) nchunk = 2048;

    auto part = at::empty({2L * B * NGROUPS * nchunk}, opts);
    auto mean = at::empty({(long)B * NGROUPS}, opts);
    auto rstd = at::empty({(long)B * NGROUPS}, opts);

    // ---- GN1 + SiLU --------------------------------------------------------
    run_groupnorm_stats(t1, part, mean, rstd, B, HW, nchunk, (float)eps, stream);

    auto y1 = at::empty({B, C, H, W}, opts, at::MemoryFormat::ChannelsLast);
    {
        dim3 grid((HW + 7) / 8, B);
        gn_apply_silu_kernel<<<grid, NTHR, 0, stream>>>(
            t1.data_ptr<float>(), y1.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            g1c.data_ptr<float>(), b1c.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (cuDNN, NHWC) ----------------------------------------------
    auto t2 = at::conv2d(y1, w2c, {}, stride, pad, dil, 1);
    if (!t2.is_contiguous(at::MemoryFormat::ChannelsLast))
        t2 = t2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GN2 + SiLU + residual + back to NCHW ------------------------------
    run_groupnorm_stats(t2, part, mean, rstd, B, HW, nchunk, (float)eps, stream);

    auto out = at::empty({B, C, H, W}, opts);
    {
        dim3 grid((HW + TP - 1) / TP, C / TC, B);
        gn_apply_silu_res_nchw_kernel<<<grid, NTHR, 0, stream>>>(
            t2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(),
            g2c.data_ptr<float>(), b2c.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    name="vae_resblock_fused_v2_graph",
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
    """Fused Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual,
    executed through a per-shape captured CUDA Graph (1 dispatch instead of 242).

    This module is stateless (all weights arrive as forward inputs), so there is
    no parameter holder to mirror; only graph bookkeeping is added.
    """

    def __init__(self):
        super().__init__()
        self._ext = _ext
        # --- plan item 1 / 9 / 12 : per-shape graph cache + guards -----------
        self._graphs = {}
        self._graph_ok = True
        self._max_graphs = 3

    @staticmethod
    def _ref(x, w1, n1w, n1b, w2, n2w, n2b, eps):
        out = F.conv2d(x, w1, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n1w, bias=n1b, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, w2, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=n2w, bias=n2b, eps=eps)
        out = F.silu(out)
        return out + x

    # ------------------------------------------------------------------ graph
    def _capture(self, key, x, w1, n1w, n1b, w2, n2w, n2b, eps_f):
        """Plan items 3-7: static buffers, side-stream warmup, capture, store."""
        try:
            # --- plan item 3 : static (fixed-address) input buffers ----------
            s_x = torch.empty_like(x).contiguous()
            s_w1 = torch.empty_like(w1).contiguous()
            s_n1w = torch.empty_like(n1w).contiguous()
            s_n1b = torch.empty_like(n1b).contiguous()
            s_w2 = torch.empty_like(w2).contiguous()
            s_n2w = torch.empty_like(n2w).contiguous()
            s_n2b = torch.empty_like(n2b).contiguous()

            s_x.copy_(x)
            s_w1.copy_(w1)
            s_n1w.copy_(n1w)
            s_n1b.copy_(n1b)
            s_w2.copy_(w2)
            s_n2w.copy_(n2w)
            s_n2b.copy_(n2b)

            # --- plan item 4 : warmup on a side stream (cuDNN autotune,
            #     workspaces, ChannelsLast weight packing) --------------------
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            warm = None
            with torch.cuda.stream(side):
                for _ in range(5):
                    warm = self._ext.fused_block(
                        s_x, s_w1, s_n1w, s_n1b, s_w2, s_n2w, s_n2b, eps_f
                    )
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            # --- plan item 5 : static output OUTSIDE the graph pool ----------
            s_out = torch.empty_like(warm)
            del warm

            # --- plan item 6 : capture the whole fused forward ---------------
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                r = self._ext.fused_block(
                    s_x, s_w1, s_n1w, s_n1b, s_w2, s_n2w, s_n2b, eps_f
                )
                s_out.copy_(r)
            del r

            # --- plan item 7 : store ----------------------------------------
            self._graphs[key] = (
                g, s_x, s_w1, s_n1w, s_n1b, s_w2, s_n2w, s_n2b, s_out
            )
            return True
        except Exception:
            # correctness-preserving fallback: never try graphs again
            self._graph_ok = False
            self._graphs.pop(key, None)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return False

    # ---------------------------------------------------------------- forward
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        # --- plan item 10 : unchanged generic fallback -----------------------
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
            # --- plan item 2 / 9 : per-shape key ---------------------------
            dev_idx = x.device.index if x.device.index is not None else -1
            key = (tuple(x.shape), x.dtype, dev_idx, eps_f)

            entry = self._graphs.get(key)
            if entry is None and self._graph_ok and len(self._graphs) < self._max_graphs:
                # --- plan item 12 : bounded number of captured graphs -------
                if self._capture(key, x, conv1_weight, norm1_weight, norm1_bias,
                                 conv2_weight, norm2_weight, norm2_bias, eps_f):
                    entry = self._graphs.get(key)

            if entry is not None:
                # --- plan item 8 : replay path ------------------------------
                (g, s_x, s_w1, s_n1w, s_n1b, s_w2, s_n2w, s_n2b, s_out) = entry
                s_x.copy_(x, non_blocking=True)
                s_w1.copy_(conv1_weight, non_blocking=True)
                s_n1w.copy_(norm1_weight, non_blocking=True)
                s_n1b.copy_(norm1_bias, non_blocking=True)
                s_w2.copy_(conv2_weight, non_blocking=True)
                s_n2w.copy_(norm2_weight, non_blocking=True)
                s_n2b.copy_(norm2_bias, non_blocking=True)
                g.replay()
                return s_out

            # --- eager fused path (unchanged base behaviour) ----------------
            return self._ext.fused_block(
                x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps_f,
            )
