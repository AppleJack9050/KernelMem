# ============================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# 1) GRANULARITY: (D) fully rewrite forward.
# 2) OPS REPLACED: both F.conv2d 3x3 (the vendor sm90 implicit-GEMM kernels),
#    both F.group_norm, both F.silu, the residual add, and the cudnn
#    NCHW<->NHWC layout-conversion kernels (deleted: we stay NCHW).
#    No cuDNN / at::conv2d / at::cudnn_convolution anywhere.
# 3) FUSION MAP:
#    wtrans_kernel   : G g G^T filter transform, once per forward -> Ug[16][C][K].
#    winograd_kernel : OWNS the conv. F(2x2,3x3) minimal filtering (16 mults per
#                      2x2 tile vs 36 direct = 2.25x fewer). Input transform
#                      B^T d B and output transform A^T m A are FUSED inside:
#                      the 4x4 transform tiles live only in shared memory /
#                      registers and are NEVER written to global memory
#                      (materializing them costs ~4x tensor traffic and gives
#                      the 2.25x back -- why WINOGRAD_NONFUSED loses here).
#                      The 16 transform-domain GEMMs use TF32 wmma m16n16k8 with
#                      fp32 accumulate (same tensor-core precision the reference
#                      itself runs). The GroupNorm sum / sum-of-squares reduction
#                      is fused into the epilogue, so the conv output is never
#                      re-read to compute statistics.
#    gnstats_kernel  : tiny N*32 finalize of mean/rstd.
#    gnsilu_kernel   : GN affine + SiLU (+ residual add on the 2nd call) in one
#                      read-modify-write pass.
# 4) REMAINS IN PYTORCH: tensor allocation and the contiguity guard only.
#
# F(2x2,3x3) only: transform entries {0,+-1,+-1/2} are exact in binary FP.
# Register budget: the 16 transform positions are split 2-per-warp over 8 warps,
# so a thread carries 8 wmma accumulator fragments (64 regs) rather than 16
# scalar accumulators per output; the tile decomposition is chosen around that.
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define NGRP     32
#define CPG      8
#define KT       32
#define BTW      8
#define BTH      4
#define PT       (BTW*BTH)
#define CS       8
#define PWD      (BTW*2+2)
#define PHD      (BTH*2+2)
#define LDU      36
#define LDV      36
#define SH_PATCH (CS*PHD*PWD)
#define SH_U     (16*CS*LDU)
#define SH_V     (16*CS*LDV)
#define SMEM_FLOATS (16*KT*PT)

struct __align__(32) Align32 { float v[8]; };

__global__ void wtrans_kernel(const float* __restrict__ w,
                              float* __restrict__ Ug, int K, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * C) return;
    int k = idx % K;
    int c = idx / K;
    const float* g = w + ((size_t)k * C + c) * 9;
    float t[4][3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
        t[0][j] = g0;
        t[1][j] = 0.5f * (g0 + g1 + g2);
        t[2][j] = 0.5f * (g0 - g1 + g2);
        t[3][j] = g2;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a0 = t[i][0], a1 = t[i][1], a2 = t[i][2];
        Ug[((size_t)(i * 4 + 0) * C + c) * K + k] = a0;
        Ug[((size_t)(i * 4 + 1) * C + c) * K + k] = 0.5f * (a0 + a1 + a2);
        Ug[((size_t)(i * 4 + 2) * C + c) * K + k] = 0.5f * (a0 - a1 + a2);
        Ug[((size_t)(i * 4 + 3) * C + c) * K + k] = a2;
    }
}

__global__ void __launch_bounds__(256, 1)
winograd_gnstat_kernel(const float* __restrict__ X,
                       const float* __restrict__ Ug,
                       float* __restrict__ Y,
                       double* __restrict__ partials,
                       int C, int K, int H, int W,
                       int TH, int TW, int nbx)
{
    extern __shared__ Align32 smem_raw[];
    float* smem  = reinterpret_cast<float*>(smem_raw);
    float* patch = smem;
    float* Us    = smem + SH_PATCH;
    float* Vs    = Us + SH_U;
    float* Ms    = smem;

    __shared__ float sred[8][4][2];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int n     = blockIdx.z;
    const int kbase = blockIdx.y * KT;
    const int sb    = blockIdx.x;
    const int tbx   = sb % nbx;
    const int tby   = sb / nbx;
    const int t0x   = tbx * BTW;
    const int t0y   = tby * BTH;
    const int r0    = 2 * t0y - 1;
    const int c0g   = 2 * t0x - 1;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b)
#pragma unroll
            for (int c = 0; c < 2; ++c)
                wmma::fill_fragment(acc[a][b][c], 0.0f);

    const int cc_t = tid >> 5;
    const int p_t  = tid & 31;
    const int py_t = p_t >> 3;
    const int px_t = p_t & 7;

    for (int c0 = 0; c0 < C; c0 += CS) {
        __syncthreads();

        for (int i = tid; i < SH_PATCH; i += 256) {
            int pw_ = i % PWD;
            int t   = i / PWD;
            int ph_ = t % PHD;
            int cc  = t / PHD;
            int rr  = r0 + ph_;
            int col = c0g + pw_;
            float v = 0.0f;
            if (rr >= 0 && rr < H && col >= 0 && col < W)
                v = X[(((size_t)n * C + (c0 + cc)) * H + rr) * W + col];
            patch[cc * (PHD * PWD) + ph_ * PWD + pw_] = v;
        }
        for (int i = tid; i < 16 * CS * KT; i += 256) {
            int k  = i & 31;
            int cc = (i >> 5) & 7;
            int xi = i >> 8;
            Us[xi * (CS * LDU) + cc * LDU + k] =
                Ug[((size_t)xi * C + (c0 + cc)) * K + kbase + k];
        }
        __syncthreads();

        // ---- fused input transform: V = B^T d B, straight into shared ----
        {
            const float* pp = patch + cc_t * (PHD * PWD) + (2 * py_t) * PWD + (2 * px_t);
            float d0, d1, d2, d3;
            float t0[4], t1[4], t2[4], t3[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                d0 = pp[0 * PWD + j];
                d1 = pp[1 * PWD + j];
                d2 = pp[2 * PWD + j];
                d3 = pp[3 * PWD + j];
                t0[j] = d0 - d2;
                t1[j] = d1 + d2;
                t2[j] = d2 - d1;
                t3[j] = d1 - d3;
            }
            float* vb = Vs + cc_t * LDV + p_t;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a0 = (i == 0) ? t0[0] : (i == 1) ? t1[0] : (i == 2) ? t2[0] : t3[0];
                float a1 = (i == 0) ? t0[1] : (i == 1) ? t1[1] : (i == 2) ? t2[1] : t3[1];
                float a2 = (i == 0) ? t0[2] : (i == 1) ? t1[2] : (i == 2) ? t2[2] : t3[2];
                float a3 = (i == 0) ? t0[3] : (i == 1) ? t1[3] : (i == 2) ? t2[3] : t3[3];
                vb[(i * 4 + 0) * (CS * LDV)] = a0 - a2;
                vb[(i * 4 + 1) * (CS * LDV)] = a1 + a2;
                vb[(i * 4 + 2) * (CS * LDV)] = a2 - a1;
                vb[(i * 4 + 3) * (CS * LDV)] = a1 - a3;
            }
        }
        __syncthreads();

        // ---- 16 transform-domain GEMMs, 2 per warp, TF32 / fp32 accumulate ----
#pragma unroll
        for (int xl = 0; xl < 2; ++xl) {
            int xi = warp * 2 + xl;
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf[2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                wmma::load_matrix_sync(af[mi], Us + xi * (CS * LDU) + mi * 16, LDU);
#pragma unroll
                for (int e = 0; e < af[mi].num_elements; ++e)
                    af[mi].x[e] = wmma::__float_to_tf32(af[mi].x[e]);
            }
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                wmma::load_matrix_sync(bf[ni], Vs + xi * (CS * LDV) + ni * 16, LDV);
#pragma unroll
                for (int e = 0; e < bf[ni].num_elements; ++e)
                    bf[ni].x[e] = wmma::__float_to_tf32(bf[ni].x[e]);
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 2; ++ni)
                    wmma::mma_sync(acc[xl][mi][ni], af[mi], bf[ni], acc[xl][mi][ni]);
        }
    }

    __syncthreads();
#pragma unroll
    for (int xl = 0; xl < 2; ++xl) {
        int xi = warp * 2 + xl;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni)
                wmma::store_matrix_sync(Ms + xi * (KT * PT) + (mi * 16) * PT + ni * 16,
                                        acc[xl][mi][ni], PT, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- fused output transform A^T m A + GroupNorm partial reduction ----
    float lsum[4], lsq[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { lsum[i] = 0.0f; lsq[i] = 0.0f; }

    const bool weven = ((W & 1) == 0);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * 256;
        int k   = idx >> 5;
        int p   = idx & 31;
        int py  = p >> 3;
        int px  = p & 7;
        int ty  = t0y + py;
        int tx  = t0x + px;

        const float* mp = Ms + k * PT + p;
        float u0[4], u1[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float m0 = mp[(0 * 4 + j) * (KT * PT)];
            float m1 = mp[(1 * 4 + j) * (KT * PT)];
            float m2 = mp[(2 * 4 + j) * (KT * PT)];
            float m3 = mp[(3 * 4 + j) * (KT * PT)];
            u0[j] = m0 + m1 + m2;
            u1[j] = m1 - m2 - m3;
        }
        float o00 = u0[0] + u0[1] + u0[2];
        float o01 = u0[1] - u0[2] - u0[3];
        float o10 = u1[0] + u1[1] + u1[2];
        float o11 = u1[1] - u1[2] - u1[3];

        if (ty < TH && tx < TW) {
            int h0 = 2 * ty, w0 = 2 * tx;
            int kg = kbase + k;
            float* rp = Y + ((size_t)((size_t)n * C + kg) * H + h0) * W + w0;
            bool wv1 = (w0 + 1) < W;
            bool hv1 = (h0 + 1) < H;
            if (wv1 && weven) {
                *reinterpret_cast<float2*>(rp) = make_float2(o00, o01);
            } else {
                rp[0] = o00;
                if (wv1) rp[1] = o01;
            }
            lsum[i] += o00; lsq[i] += o00 * o00;
            if (wv1) { lsum[i] += o01; lsq[i] += o01 * o01; }
            if (hv1) {
                float* rq = rp + W;
                if (wv1 && weven) {
                    *reinterpret_cast<float2*>(rq) = make_float2(o10, o11);
                } else {
                    rq[0] = o10;
                    if (wv1) rq[1] = o11;
                }
                lsum[i] += o10; lsq[i] += o10 * o10;
                if (wv1) { lsum[i] += o11; lsq[i] += o11 * o11; }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float s = lsum[i], q = lsq[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
            q += __shfl_down_sync(0xffffffffu, q, off);
        }
        if (lane == 0) { sred[warp][i][0] = s; sred[warp][i][1] = q; }
    }
    __syncthreads();
    if (tid < 8) {
        int gi = tid >> 1, which = tid & 1;
        float s = 0.0f;
#pragma unroll
        for (int w2 = 0; w2 < 8; ++w2) s += sred[w2][gi][which];
        atomicAdd(&partials[((size_t)n * NGRP + (blockIdx.y * 4 + gi)) * 2 + which], (double)s);
    }
}

__global__ void gnstats_kernel(const double* __restrict__ part,
                               float* __restrict__ mean,
                               float* __restrict__ rstd,
                               int total, double cnt, double eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    double s = part[(size_t)idx * 2];
    double q = part[(size_t)idx * 2 + 1];
    double m = s / cnt;
    double v = q / cnt - m * m;
    if (!(v > 0.0)) v = 0.0;
    mean[idx] = (float)m;
    rstd[idx] = (float)(1.0 / sqrt(v + eps));
}

__global__ void gnsilu_kernel(const float* __restrict__ Yin,
                              const float* __restrict__ mean,
                              const float* __restrict__ rstd,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              const float* __restrict__ res,
                              float* __restrict__ out,
                              int C, long HW, int cpg)
{
    int c = blockIdx.x;
    int n = blockIdx.y;
    int g = n * NGRP + c / cpg;
    float m = mean[g], r = rstd[g];
    float a = r * gamma[c];
    float b = beta[c] - m * a;
    size_t base = ((size_t)n * C + c) * (size_t)HW;
    const float* pi = Yin + base;
    float* po = out + base;
    const float* pr = (res == nullptr) ? nullptr : (res + base);

    if ((HW & 3L) == 0) {
        long nv = HW >> 2;
        const float4* pi4 = reinterpret_cast<const float4*>(pi);
        float4* po4 = reinterpret_cast<float4*>(po);
        const float4* pr4 = (pr == nullptr) ? nullptr : reinterpret_cast<const float4*>(pr);
        for (long i = threadIdx.x; i < nv; i += blockDim.x) {
            float4 v = pi4[i];
            float x0 = v.x * a + b, x1 = v.y * a + b, x2 = v.z * a + b, x3 = v.w * a + b;
            x0 = x0 / (1.0f + __expf(-x0));
            x1 = x1 / (1.0f + __expf(-x1));
            x2 = x2 / (1.0f + __expf(-x2));
            x3 = x3 / (1.0f + __expf(-x3));
            if (pr4 != nullptr) {
                float4 rr = pr4[i];
                x0 += rr.x; x1 += rr.y; x2 += rr.z; x3 += rr.w;
            }
            po4[i] = make_float4(x0, x1, x2, x3);
        }
    } else {
        for (long i = threadIdx.x; i < HW; i += blockDim.x) {
            float x = pi[i] * a + b;
            x = x / (1.0f + __expf(-x));
            if (pr != nullptr) x += pr[i];
            po[i] = x;
        }
    }
}

static bool g_attr_set = false;

static void launch_winograd(const at::Tensor& X, const at::Tensor& Ug,
                            at::Tensor& Y, at::Tensor& part,
                            int N, int C, int K, int H, int W,
                            cudaStream_t stream)
{
    int TH = (H + 1) / 2, TW = (W + 1) / 2;
    int nbx = (TW + BTW - 1) / BTW;
    int nby = (TH + BTH - 1) / BTH;
    dim3 grid(nbx * nby, K / KT, N);
    size_t smem = (size_t)SMEM_FLOATS * sizeof(float);
    if (!g_attr_set) {
        cudaFuncSetAttribute((const void*)winograd_gnstat_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        g_attr_set = true;
    }
    winograd_gnstat_kernel<<<grid, 256, smem, stream>>>(
        X.data_ptr<float>(), Ug.data_ptr<float>(), Y.data_ptr<float>(),
        part.data_ptr<double>(), C, K, H, W, TH, TW, nbx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight,
                             torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight,
                             torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias,
                             double eps)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "fp32 cuda input required");
    TORCH_CHECK(x.dim() == 4, "expect NCHW");
    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto w1 = conv1_weight.is_contiguous() ? conv1_weight : conv1_weight.contiguous();
    auto w2 = conv2_weight.is_contiguous() ? conv2_weight : conv2_weight.contiguous();

    int N = (int)xc.size(0), C = (int)xc.size(1), H = (int)xc.size(2), W = (int)xc.size(3);
    int K = (int)w1.size(0);
    TORCH_CHECK(C == 256 && K == 256, "specialized for C=K=256");
    TORCH_CHECK(w1.size(2) == 3 && w1.size(3) == 3, "3x3 kernel required");
    TORCH_CHECK(C % NGRP == 0, "groups must divide channels");
    const int cpg = C / NGRP;
    TORCH_CHECK(cpg == CPG, "specialized for 8 channels per group");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto fopt = xc.options();
    auto dopt = fopt.dtype(torch::kDouble);

    auto Ug1 = torch::empty({16, C, K}, fopt);
    auto Ug2 = torch::empty({16, C, K}, fopt);
    int nw = K * C;
    wtrans_kernel<<<(nw + 255) / 256, 256, 0, stream>>>(w1.data_ptr<float>(), Ug1.data_ptr<float>(), K, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wtrans_kernel<<<(nw + 255) / 256, 256, 0, stream>>>(w2.data_ptr<float>(), Ug2.data_ptr<float>(), K, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto y1 = torch::empty({N, K, H, W}, fopt);
    auto p1 = torch::zeros({N, NGRP, 2}, dopt);
    launch_winograd(xc, Ug1, y1, p1, N, C, K, H, W, stream);

    auto mean = torch::empty({N * NGRP}, fopt);
    auto rstd = torch::empty({N * NGRP}, fopt);
    double cnt = (double)cpg * (double)H * (double)W;
    gnstats_kernel<<<(N * NGRP + 127) / 128, 128, 0, stream>>>(
        p1.data_ptr<double>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        N * NGRP, cnt, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 eg(K, N);
    gnsilu_kernel<<<eg, 256, 0, stream>>>(
        y1.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        norm1_weight.data_ptr<float>(), norm1_bias.data_ptr<float>(),
        nullptr, y1.data_ptr<float>(), K, (long)H * (long)W, cpg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto y2 = torch::empty({N, K, H, W}, fopt);
    auto p2 = torch::zeros({N, NGRP, 2}, dopt);
    launch_winograd(y1, Ug2, y2, p2, N, C, K, H, W, stream);

    auto mean2 = torch::empty({N * NGRP}, fopt);
    auto rstd2 = torch::empty({N * NGRP}, fopt);
    gnstats_kernel<<<(N * NGRP + 127) / 128, 128, 0, stream>>>(
        p2.data_ptr<double>(), mean2.data_ptr<float>(), rstd2.data_ptr<float>(),
        N * NGRP, cnt, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out = torch::empty({N, K, H, W}, fopt);
    gnsilu_kernel<<<eg, 256, 0, stream>>>(
        y2.data_ptr<float>(), mean2.data_ptr<float>(), rstd2.data_ptr<float>(),
        norm2_weight.data_ptr<float>(), norm2_bias.data_ptr<float>(),
        xc.data_ptr<float>(), out.data_ptr<float>(), K, (long)H * (long)W, cpg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
'''

cpp_src = r'''
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight,
                             torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight,
                             torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias,
                             double eps);
'''

_ext = load_inline(
    name="wino_vae_resblock_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_resblock"],
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
    """Winograd F(2x2,3x3) fused VAE residual block (granularity D)."""

    def __init__(self):
        super().__init__()
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps = float(eps.reshape(-1)[0].item())
        return self._ext.fused_resblock(
            x, conv1_weight, norm1_weight, norm1_bias,
            conv2_weight, norm2_weight, norm2_bias, float(eps),
        )
