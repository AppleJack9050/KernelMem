# =============================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY: (C) "fuse many ops into one/few kernels"
#
# Same fusion contract as before (4 custom kernels + 2 vendor cuDNN convs), but the
# HOST SCHEDULE is now a two-stream, batch-split software pipeline:
#
#   sC (compute stream) : conv1(chunk 0..k-1)  then  conv2(chunk 0..k-1)
#   sM (memory stream)  : GN1+SiLU(chunk 0..k-1) then GN2+SiLU+residual+transpose
#
# Events create a one-stage skew, so the compute-bound cuDNN implicit GEMM of chunk
# i+1 overlaps the DRAM-bound GroupNorm/SiLU kernels of chunk i. All five __global__
# kernels are byte-identical to the previous candidate => bitwise-identical outputs
# (per-sample reduction order unchanged; GN statistics are per-(n,group), so the
# split along N is exact and needs no halo).
# =============================================================================

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
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

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
// grid = (nchunks, cn), block = 256 threads (one per channel) -> fully coalesced reads.
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

// --------- allocator safety helper (plan item 5) ---------------------------
inline void rec(const at::Tensor& t, at::cuda::CUDAStream s) {
    if (t.defined() && t.is_cuda() && t.has_storage()) {
        c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), s);
    }
}

} // namespace

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

    // --- K1: one single layout conversion for the whole block (current stream) ---
    auto x_cl = at::empty({N, C, H, W},
                          opts.memory_format(at::MemoryFormat::ChannelsLast));
    {
        dim3 blk(32, 8);
        dim3 grd((unsigned)((HW + 31) / 32), (unsigned)(C / 32), (unsigned)N);
        nchw2nhwc_kernel<<<grd, blk, 0, cur.stream()>>>(xc.data_ptr<float>(),
                                                        x_cl.data_ptr<float>(), (int)HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto w1c = conv1_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = conv2_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1  = norm1_weight.contiguous();
    auto b1  = norm1_bias.contiguous();
    auto g2  = norm2_weight.contiguous();
    auto b2  = norm2_bias.contiguous();
    std::vector<int64_t> stride{1, 1}, pad{1, 1}, dil{1, 1};

    // ---------------- plan item 1 + 8: chunk decision -------------------------
    int k = 1;
    if (N >= 2) {
        k = (N >= 4) ? 4 : (int)N;
        // guardrail: per-chunk implicit-GEMM M = cn*H*W must stay >= 8192
        while (k > 1 && ((N + k - 1) / k) * HW < 8192) --k;
    }
    int64_t chunk = (N + k - 1) / k;
    k = (int)((N + chunk - 1) / chunk);          // drop empty trailing chunks

    // ---------------- plan item 7: reduction schedule from per-chunk work -----
    const int min_ppc = 32;
    long nc = ((long)8 * sms + chunk - 1) / chunk;
    const long max_nc = (HW + min_ppc - 1) / min_ppc;
    if (nc > max_nc) nc = max_nc;
    if (nc < 1) nc = 1;
    const int ppc     = (int)((HW + nc - 1) / nc);
    const int nchunks = (int)((HW + ppc - 1) / ppc);

    auto part  = at::empty({(int64_t)N * GG * nchunks, 2}, opts.dtype(at::kDouble));
    auto scale = at::empty({N, C}, opts);
    auto shift = at::empty({N, C}, opts);
    const double M = (double)CPG * (double)HW;

    const long nq = HW * (long)(CC / 4);
    int silu_blocks = (int)std::min<long>((nq + 255) / 256, (long)(6 * sms));
    if (silu_blocks < 1) silu_blocks = 1;

    auto z   = at::empty({N, C, H, W}, opts.memory_format(at::MemoryFormat::ChannelsLast));
    auto out = at::empty({N, C, H, W}, opts);

    double2* part_ptr = reinterpret_cast<double2*>(part.data_ptr<double>());
    float*   sc_ptr   = scale.data_ptr<float>();
    float*   sh_ptr   = shift.data_ptr<float>();

    // =====================================================================
    // N < 2 (or degenerate chunking): ORIGINAL single-stream path, untouched
    // =====================================================================
    if (k <= 1) {
        auto y1 = at::conv2d(x_cl, w1c, {}, stride, pad, dil, 1);
        if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
            y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

        gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)N), 256, 0, cur.stream()>>>(
            y1.data_ptr<float>(), part_ptr, (int)HW, nchunks, ppc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        gn_finalize_kernel<<<(unsigned)(N * GG), 128, 0, cur.stream()>>>(
            part_ptr, g1.data_ptr<float>(), b1.data_ptr<float>(),
            sc_ptr, sh_ptr, nchunks, M, eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        gn_silu_kernel<<<dim3((unsigned)silu_blocks, (unsigned)N), 256, 0, cur.stream()>>>(
            y1.data_ptr<float>(), sc_ptr, sh_ptr, z.data_ptr<float>(), (long)HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto y2 = at::conv2d(z, w2c, {}, stride, pad, dil, 1);
        if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
            y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

        gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)N), 256, 0, cur.stream()>>>(
            y2.data_ptr<float>(), part_ptr, (int)HW, nchunks, ppc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        gn_finalize_kernel<<<(unsigned)(N * GG), 128, 0, cur.stream()>>>(
            part_ptr, g2.data_ptr<float>(), b2.data_ptr<float>(),
            sc_ptr, sh_ptr, nchunks, M, eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        {
            dim3 blk(32, 8);
            dim3 grd((unsigned)((HW + 31) / 32), (unsigned)(C / 32), (unsigned)N);
            gn_silu_res_t_kernel<<<grd, blk, 0, cur.stream()>>>(
                y2.data_ptr<float>(), xc.data_ptr<float>(),
                sc_ptr, sh_ptr, out.data_ptr<float>(), (int)HW);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out;
    }

    // =====================================================================
    // Pipelined two-stream schedule (plan items 2-6)
    // =====================================================================
    static thread_local at::cuda::CUDAStream sC = at::cuda::getStreamFromPool();
    static thread_local at::cuda::CUDAStream sM = at::cuda::getStreamFromPool();

    at::cuda::CUDAEvent ev0;
    ev0.record(cur);
    ev0.block(sC);
    ev0.block(sM);

    std::vector<at::cuda::CUDAEvent> evConv1(k), evGN1(k), evConv2(k), evGN2(k);
    std::vector<at::Tensor> y1s(k), y2s(k);

    // allocator safety for cross-stream lifetimes
    rec(x_cl, sC); rec(w1c, sC); rec(w2c, sC); rec(z, sC);
    rec(xc, sM);   rec(z, sM);   rec(out, sM);
    rec(part, sM); rec(scale, sM); rec(shift, sM);
    rec(g1, sM);   rec(b1, sM);  rec(g2, sM);  rec(b2, sM);

    // ---- LOOP A (sC): conv1 for every chunk -------------------------------
    {
        c10::cuda::CUDAStreamGuard g(sC);
        for (int c = 0; c < k; ++c) {
            const int64_t n0 = (int64_t)c * chunk;
            const int64_t cn = std::min<int64_t>(chunk, N - n0);
            auto xin = x_cl.narrow(0, n0, cn);
            auto y = at::conv2d(xin, w1c, {}, stride, pad, dil, 1);
            if (!y.is_contiguous(at::MemoryFormat::ChannelsLast))
                y = y.contiguous(at::MemoryFormat::ChannelsLast);
            y1s[c] = y;
            rec(y, sM);
            evConv1[c].record(sC);
        }
    }

    // ---- LOOP B (sM): GN1 stats + GN1 apply + SiLU for every chunk ---------
    {
        c10::cuda::CUDAStreamGuard g(sM);
        for (int c = 0; c < k; ++c) {
            const int64_t n0 = (int64_t)c * chunk;
            const int64_t cn = std::min<int64_t>(chunk, N - n0);
            evConv1[c].block(sM);

            const float* yp = y1s[c].data_ptr<float>();
            double2* pp = part_ptr + (size_t)n0 * GG * nchunks;
            float* scp = sc_ptr + n0 * CC;
            float* shp = sh_ptr + n0 * CC;
            float* zp  = z.narrow(0, n0, cn).data_ptr<float>();

            gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)cn), 256, 0, sM.stream()>>>(
                yp, pp, (int)HW, nchunks, ppc);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            gn_finalize_kernel<<<(unsigned)(cn * GG), 128, 0, sM.stream()>>>(
                pp, g1.data_ptr<float>(), b1.data_ptr<float>(), scp, shp, nchunks, M, eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            gn_silu_kernel<<<dim3((unsigned)silu_blocks, (unsigned)cn), 256, 0, sM.stream()>>>(
                yp, scp, shp, zp, (long)HW);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            evGN1[c].record(sM);
        }
    }

    // ---- LOOP C (sC): conv2 for every chunk -------------------------------
    {
        c10::cuda::CUDAStreamGuard g(sC);
        for (int c = 0; c < k; ++c) {
            const int64_t n0 = (int64_t)c * chunk;
            const int64_t cn = std::min<int64_t>(chunk, N - n0);
            evGN1[c].block(sC);
            auto zin = z.narrow(0, n0, cn);
            auto y = at::conv2d(zin, w2c, {}, stride, pad, dil, 1);
            if (!y.is_contiguous(at::MemoryFormat::ChannelsLast))
                y = y.contiguous(at::MemoryFormat::ChannelsLast);
            y2s[c] = y;
            rec(y, sM);
            evConv2[c].record(sC);
        }
    }

    // ---- LOOP D (sM): GN2 stats + GN2 apply + SiLU + residual + transpose --
    {
        c10::cuda::CUDAStreamGuard g(sM);
        for (int c = 0; c < k; ++c) {
            const int64_t n0 = (int64_t)c * chunk;
            const int64_t cn = std::min<int64_t>(chunk, N - n0);
            evConv2[c].block(sM);

            const float* yp = y2s[c].data_ptr<float>();
            double2* pp = part_ptr + (size_t)n0 * GG * nchunks;
            float* scp = sc_ptr + n0 * CC;
            float* shp = sh_ptr + n0 * CC;
            const float* xrp = xc.narrow(0, n0, cn).data_ptr<float>();
            float* op = out.narrow(0, n0, cn).data_ptr<float>();

            gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)cn), 256, 0, sM.stream()>>>(
                yp, pp, (int)HW, nchunks, ppc);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            gn_finalize_kernel<<<(unsigned)(cn * GG), 128, 0, sM.stream()>>>(
                pp, g2.data_ptr<float>(), b2.data_ptr<float>(), scp, shp, nchunks, M, eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            {
                dim3 blk(32, 8);
                dim3 grd((unsigned)((HW + 31) / 32), (unsigned)(C / 32), (unsigned)cn);
                gn_silu_res_t_kernel<<<grd, blk, 0, sM.stream()>>>(
                    yp, xrp, scp, shp, op, (int)HW);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            evGN2[c].record(sM);
        }
    }

    // ---- join back onto the caller's stream --------------------------------
    evGN2[k - 1].block(cur);
    evConv2[k - 1].block(cur);
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
    name="vae_resblock_fused_v2_pipe",
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
