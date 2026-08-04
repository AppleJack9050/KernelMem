# ==========================================================================
# ModelNew - SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER (required):
# 1) GRANULARITY: (D) fully rewrite forward. The vendor implicit-GEMM conv
#    (sm90_xmma_fprop_implicit_gemm_..._tf32f32_...) is OWNED here: both 3x3
#    stride-1 convs are hand-written Winograd F(2x2,3x3) kernels (TF32 wmma
#    pointwise stage, fp32 accumulate). No cuDNN / at::conv2d anywhere.
# 2) OPS REPLACED: conv1, group_norm1, silu1, conv2, group_norm2, silu2,
#    residual add -- i.e. the entire reference forward.
# 3) FUSION MAP (5 launches, reference used ~17 kernels):
#    K1 wino_kernel<false>: Winograd conv1. Input transform B^T d B and filter
#       transform G g G^T are computed INSIDE the kernel and live only in
#       shared memory / registers, never in global (the transform domain is 4x
#       the tensor; materialising it gives back the whole 2.25x). Epilogue does
#       A^T M A, writes y1, and emits GroupNorm-1 partial sum/sumsq per block
#       (deterministic shared-mem tree reduction, no atomics) so y1 is never
#       re-read for statistics.
#    K2 finalize_stats: B*G-block reduction of partials -> mean/rstd (GN1).
#    K3 wino_kernel<true>: Winograd conv2 whose PROLOGUE fuses GN1+SiLU onto
#       activations while staging them into shared memory, so the normalised
#       tensor never round-trips through global (a full read AND write deleted).
#       Epilogue emits the GN2 partials.
#    K4 finalize_stats: mean/rstd for GN2.
#    K5 gn_silu_res_kernel: GN2 + SiLU + residual add in one bandwidth pass.
# 4) REMAINS IN PYTORCH: nothing computational; only allocation / contiguity
#    checks (no-ops when inputs are already contiguous).
#
# Design: F(2x2,3x3) only (transform entries 0,+-1/2,+-1 are exact in binary FP).
# 16 accumulators per tile is the binding register constraint, so the 16
# transform positions xi are SPLIT ACROSS WARPS (warp w owns xi=2w,2w+1) -> 8
# wmma accumulator fragments/warp (64 regs) instead of 128. Block tile: 32 out
# channels x 32 winograd tiles (16x8 out pixels), C-chunk 8 (= tf32 wmma k),
# 256 threads, 38 KB shared. fp32 storage/accumulate, TF32 tensor cores only in
# the pointwise stage - exactly what the reference conv itself runs.
# ==========================================================================

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <mma.h>

using namespace nvcuda;

#define KB       32
#define PB       32
#define CBK      8
#define TBX      8
#define TBY      4
#define RAW_H    10
#define RAW_W    18
#define RAW_LD   20
#define NTHR     256
#define NGRP     32
#define SMEM_FLOATS (CBK*RAW_H*RAW_LD + 16*CBK*PB + 16*CBK*KB)

__device__ __forceinline__ void in_tf(const float d[4][4], float v[16]) {
    float t[4][4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        t[0][c] = d[0][c] - d[2][c];
        t[1][c] = d[1][c] + d[2][c];
        t[2][c] = d[2][c] - d[1][c];
        t[3][c] = d[1][c] - d[3][c];
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        v[r*4+0] = t[r][0] - t[r][2];
        v[r*4+1] = t[r][1] + t[r][2];
        v[r*4+2] = t[r][2] - t[r][1];
        v[r*4+3] = t[r][1] - t[r][3];
    }
}

__device__ __forceinline__ void filt_tf(const float g[3][3], float u[16]) {
    float t[4][3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        t[0][c] = g[0][c];
        t[1][c] = 0.5f * (g[0][c] + g[1][c] + g[2][c]);
        t[2][c] = 0.5f * (g[0][c] - g[1][c] + g[2][c]);
        t[3][c] = g[2][c];
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        u[r*4+0] = t[r][0];
        u[r*4+1] = 0.5f * (t[r][0] + t[r][1] + t[r][2]);
        u[r*4+2] = 0.5f * (t[r][0] - t[r][1] + t[r][2]);
        u[r*4+3] = t[r][2];
    }
}

__device__ __forceinline__ void out_tf(const float m[16], float o[4]) {
    float t[2][4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        t[0][c] = m[0*4+c] + m[1*4+c] + m[2*4+c];
        t[1][c] = m[1*4+c] - m[2*4+c] - m[3*4+c];
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        o[r*2+0] = t[r][0] + t[r][1] + t[r][2];
        o[r*2+1] = t[r][1] - t[r][2] - t[r][3];
    }
}

template <bool NORM_IN>
__global__ __launch_bounds__(NTHR, 2) void wino_kernel(
    const float* __restrict__ xin,
    const float* __restrict__ wt,
    const float* __restrict__ gmean,
    const float* __restrict__ grstd,
    const float* __restrict__ gw,
    const float* __restrict__ gb,
    float* __restrict__ yout,
    float* __restrict__ psum,
    float* __restrict__ psq,
    int B, int C, int H, int W, int cpg,
    int tilesX, int tilesY, int nbx, int nsb)
{
    __shared__ float smem[SMEM_FLOATS];
    float* raw = smem;
    float* Vs  = smem + CBK*RAW_H*RAW_LD;
    float* Us  = Vs + 16*CBK*PB;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;

    const int sb = blockIdx.x;
    const int kb = blockIdx.y;
    const int bn = blockIdx.z;

    const int bx = sb % nbx;
    const int by = sb / nbx;
    const int tbx0 = bx * TBX;
    const int tby0 = by * TBY;
    const int k0abs = kb * KB;
    const int iy0 = tby0 * 2 - 1;
    const int ix0 = tbx0 * 2 - 1;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b2 = 0; b2 < 2; ++b2)
#pragma unroll
            for (int c2 = 0; c2 < 2; ++c2)
                wmma::fill_fragment(acc[a][b2][c2], 0.0f);

    for (int cc = 0; cc < C; cc += CBK) {
        __syncthreads();
        for (int idx = tid; idx < CBK*RAW_H*RAW_W; idx += NTHR) {
            int c   = idx / (RAW_H*RAW_W);
            int rem = idx - c*(RAW_H*RAW_W);
            int r   = rem / RAW_W;
            int s   = rem - r*RAW_W;
            int iy  = iy0 + r;
            int ix  = ix0 + s;
            float val = 0.0f;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                int ch = cc + c;
                val = xin[((size_t)(bn*C + ch)*H + iy)*W + ix];
                if (NORM_IN) {
                    int g = ch / cpg;
                    float v = (val - gmean[bn*NGRP + g]) * grstd[bn*NGRP + g]
                              * gw[ch] + gb[ch];
                    val = v / (1.0f + expf(-v));
                }
            }
            raw[c*(RAW_H*RAW_LD) + r*RAW_LD + s] = val;
        }
        __syncthreads();

        {   // input transform: one (channel, tile) pair per thread
            int c  = tid >> 5;
            int t  = tid & 31;
            int ty = t >> 3;
            int tx = t & 7;
            const float* p = raw + c*(RAW_H*RAW_LD) + (ty*2)*RAW_LD + tx*2;
            float d[4][4];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int s = 0; s < 4; ++s) d[r][s] = p[r*RAW_LD + s];
            float v[16];
            in_tf(d, v);
#pragma unroll
            for (int xi = 0; xi < 16; ++xi) Vs[xi*(CBK*PB) + c*PB + t] = v[xi];
        }
        {   // filter transform: one (out-ch, in-ch) pair per thread
            int k = tid >> 3;
            int c = tid & 7;
            const float* wp = wt + ((size_t)(k0abs + k)*C + cc + c)*9;
            float g3[3][3];
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j) g3[i][j] = wp[i*3 + j];
            float u[16];
            filt_tf(g3, u);
#pragma unroll
            for (int xi = 0; xi < 16; ++xi) Us[xi*(CBK*KB) + c*KB + k] = u[xi];
        }
        __syncthreads();

#pragma unroll
        for (int xi_i = 0; xi_i < 2; ++xi_i) {
            int xi = warp*2 + xi_i;
            const float* Up = Us + xi*(CBK*KB);
            const float* Vp = Vs + xi*(CBK*PB);
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                           wmma::col_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                           wmma::row_major> bf[2];
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                wmma::load_matrix_sync(af[ki], Up + ki*16, KB);
#pragma unroll
                for (int e = 0; e < af[ki].num_elements; ++e)
                    af[ki].x[e] = wmma::__float_to_tf32(af[ki].x[e]);
            }
#pragma unroll
            for (int pi = 0; pi < 2; ++pi) {
                wmma::load_matrix_sync(bf[pi], Vp + pi*16, PB);
#pragma unroll
                for (int e = 0; e < bf[pi].num_elements; ++e)
                    bf[pi].x[e] = wmma::__float_to_tf32(bf[pi].x[e]);
            }
#pragma unroll
            for (int ki = 0; ki < 2; ++ki)
#pragma unroll
                for (int pi = 0; pi < 2; ++pi)
                    wmma::mma_sync(acc[xi_i][ki][pi], af[ki], bf[pi],
                                   acc[xi_i][ki][pi]);
        }
    }

    // ---- epilogue: inverse transform + store + GroupNorm partials ----
    float* Mb = smem;   // [16][KB][16] overlay (V/U/raw are dead now)
    float gsum[4] = {0.f, 0.f, 0.f, 0.f};
    float gsq[4]  = {0.f, 0.f, 0.f, 0.f};

    for (int hp = 0; hp < 2; ++hp) {
        __syncthreads();
#pragma unroll
        for (int xi_i = 0; xi_i < 2; ++xi_i) {
            int xi = warp*2 + xi_i;
#pragma unroll
            for (int ki = 0; ki < 2; ++ki)
                wmma::store_matrix_sync(Mb + xi*(KB*16) + ki*256,
                                        acc[xi_i][ki][hp], 16,
                                        wmma::mem_row_major);
        }
        __syncthreads();
        for (int it = 0; it < 2; ++it) {
            int idx = tid + it*NTHR;
            int k = idx >> 4;
            int p = idx & 15;
            float m[16];
#pragma unroll
            for (int xi = 0; xi < 16; ++xi) m[xi] = Mb[xi*(KB*16) + k*16 + p];
            float o[4];
            out_tf(m, o);

            int pl  = hp*16 + p;
            int ty  = pl >> 3;
            int tx  = pl & 7;
            int oy0 = (tby0 + ty) * 2;
            int ox0 = (tbx0 + tx) * 2;
            int kabs = k0abs + k;
            int gi = k / cpg;
            float* yp = yout + (size_t)(bn*C + kabs)*H*W;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int oy = oy0 + i;
                if (oy >= H) continue;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int ox = ox0 + j;
                    if (ox >= W) continue;
                    float v = o[i*2 + j];
                    yp[(size_t)oy*W + ox] = v;
                    gsum[gi] += v;
                    gsq[gi]  += v*v;
                }
            }
        }
    }

    __syncthreads();
    float* rs = smem;              // [4][256]
    float* rq = smem + 4*NTHR;     // [4][256]
#pragma unroll
    for (int g = 0; g < 4; ++g) { rs[g*NTHR + tid] = gsum[g]; rq[g*NTHR + tid] = gsq[g]; }
    __syncthreads();
    for (int st = NTHR/2; st > 0; st >>= 1) {
        if (tid < st) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                rs[g*NTHR + tid] += rs[g*NTHR + tid + st];
                rq[g*NTHR + tid] += rq[g*NTHR + tid + st];
            }
        }
        __syncthreads();
    }
    if (tid < 4) {
        int gabs = (k0abs / cpg) + tid;
        size_t off = (size_t)(bn*NGRP + gabs)*nsb + sb;
        psum[off] = rs[tid*NTHR];
        psq[off]  = rq[tid*NTHR];
    }
}

__global__ void finalize_stats(const float* __restrict__ psum,
                               const float* __restrict__ psq,
                               float* __restrict__ mean,
                               float* __restrict__ rstd,
                               int nsb, float invN, float eps)
{
    __shared__ float ss[128], sq[128];
    int bg = blockIdx.x;
    const float* p1 = psum + (size_t)bg*nsb;
    const float* p2 = psq  + (size_t)bg*nsb;
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < nsb; i += blockDim.x) { s += p1[i]; q += p2[i]; }
    ss[threadIdx.x] = s; sq[threadIdx.x] = q;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            ss[threadIdx.x] += ss[threadIdx.x + st];
            sq[threadIdx.x] += sq[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = ss[0] * invN;
        float v = sq[0] * invN - m*m;
        if (!(v > 0.f)) v = 0.f;
        mean[bg] = m;
        rstd[bg] = rsqrtf(v + eps);
    }
}

__global__ void gn_silu_res(const float* __restrict__ y,
                            const float* __restrict__ res,
                            const float* __restrict__ mean,
                            const float* __restrict__ rstd,
                            const float* __restrict__ gw,
                            const float* __restrict__ gb,
                            float* __restrict__ out,
                            int C, long HW, int cpg, long numel)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < numel; i += stride) {
        long t = i / HW;
        int c  = (int)(t % C);
        int b  = (int)(t / C);
        int g  = c / cpg;
        float v = (y[i] - mean[b*NGRP + g]) * rstd[b*NGRP + g] * gw[c] + gb[c];
        v = v / (1.0f + expf(-v));
        out[i] = v + res[i];
    }
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "fp32 cuda input required");
    auto xc  = x.is_contiguous()  ? x  : x.contiguous();
    auto w1c = w1.is_contiguous() ? w1 : w1.contiguous();
    auto w2c = w2.is_contiguous() ? w2 : w2.contiguous();
    auto a1  = n1w.is_contiguous() ? n1w : n1w.contiguous();
    auto b1  = n1b.is_contiguous() ? n1b : n1b.contiguous();
    auto a2  = n2w.is_contiguous() ? n2w : n2w.contiguous();
    auto b2  = n2b.is_contiguous() ? n2b : n2b.contiguous();

    const int B = xc.size(0), C = xc.size(1), H = xc.size(2), W = xc.size(3);
    const int cpg = C / NGRP;
    TORCH_CHECK(C % KB == 0 && cpg == 8, "kernel specialised for C=256, G=32");

    const int tilesX = (W + 1) / 2;
    const int tilesY = (H + 1) / 2;
    const int nbx = (tilesX + TBX - 1) / TBX;
    const int nby = (tilesY + TBY - 1) / TBY;
    const int nsb = nbx * nby;

    auto opt = xc.options();
    auto y1 = torch::empty({B, C, H, W}, opt);
    auto y2 = torch::empty({B, C, H, W}, opt);
    auto out = torch::empty({B, C, H, W}, opt);
    auto ps  = torch::empty({B, NGRP, nsb}, opt);
    auto pq  = torch::empty({B, NGRP, nsb}, opt);
    auto m1  = torch::empty({B, NGRP}, opt);
    auto r1  = torch::empty({B, NGRP}, opt);
    auto m2  = torch::empty({B, NGRP}, opt);
    auto r2  = torch::empty({B, NGRP}, opt);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(nsb, C / KB, B);
    const float invN = 1.0f / (float)((long)cpg * H * W);

    wino_kernel<false><<<grid, NTHR, 0, stream>>>(
        xc.data_ptr<float>(), w1c.data_ptr<float>(),
        nullptr, nullptr, nullptr, nullptr,
        y1.data_ptr<float>(), ps.data_ptr<float>(), pq.data_ptr<float>(),
        B, C, H, W, cpg, tilesX, tilesY, nbx, nsb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_stats<<<B*NGRP, 128, 0, stream>>>(
        ps.data_ptr<float>(), pq.data_ptr<float>(),
        m1.data_ptr<float>(), r1.data_ptr<float>(), nsb, invN, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    wino_kernel<true><<<grid, NTHR, 0, stream>>>(
        y1.data_ptr<float>(), w2c.data_ptr<float>(),
        m1.data_ptr<float>(), r1.data_ptr<float>(),
        a1.data_ptr<float>(), b1.data_ptr<float>(),
        y2.data_ptr<float>(), ps.data_ptr<float>(), pq.data_ptr<float>(),
        B, C, H, W, cpg, tilesX, tilesY, nbx, nsb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_stats<<<B*NGRP, 128, 0, stream>>>(
        ps.data_ptr<float>(), pq.data_ptr<float>(),
        m2.data_ptr<float>(), r2.data_ptr<float>(), nsb, invN, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const long numel = (long)B * C * H * W;
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535 * 4) blocks = 65535 * 4;
    gn_silu_res<<<blocks, threads, 0, stream>>>(
        y2.data_ptr<float>(), xc.data_ptr<float>(),
        m2.data_ptr<float>(), r2.data_ptr<float>(),
        a2.data_ptr<float>(), b2.data_ptr<float>(),
        out.data_ptr<float>(), C, (long)H*W, cpg, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor n1w, torch::Tensor n1b,
                          torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b,
                          double eps);
"""

_ext = load_inline(
    name="wino_vae_block_v1",
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
    """Winograd F(2x2,3x3) conv x2 with GroupNorm/SiLU/residual fused into the
    conv prologue and epilogue (granularity D: the vendor conv is owned)."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        return self._ext.fused_block(
            x, conv1_weight, norm1_weight, norm1_bias,
            conv2_weight, norm2_weight, norm2_bias, float(eps))
