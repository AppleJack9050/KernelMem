# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
# HEADER:
# 1) GRANULARITY: (D) FULL FORWARD REWRITE. Both 3x3 convolutions are OWNED by
#    a hand-written tiled implicit-GEMM kernel (wmma TF32, fp32 accumulate).
#    No cuDNN / at::conv2d is used anywhere on the fast path.
# 2) OPS REPLACED: conv1, group_norm1, silu1, conv2, group_norm2, silu2, add.
# 3) FUSION MAP:
#    K0 wt_transform   : (Co,Ci,3,3) -> (rs,Ci,Co) for coalesced B-tile loads.
#    K1 conv_gn<false> : conv1 implicit GEMM (BM128 x BN128 x BK32, 8 warps);
#                        epilogue also emits GN1 partial sum/sumsq per
#                        (b,group,m-tile) -> the GN1 stats pass is deleted.
#    K2 gn_finalize    : deterministic double reduction -> per-(b,c) affine.
#    K3 conv_gn<true>  : conv2 implicit GEMM whose PROLOGUE applies GN1+SiLU
#                        while staging the A tile (that tensor never round-
#                        trips through global memory); epilogue emits GN2
#                        partials the same way.
#    K4 gn_finalize    : same reduction for GN2.
#    K5 gn_silu_res    : GN2 affine + SiLU + residual add, float4 vectorised.
#    Traffic: read x, write y1, read y1, write y2, read y2+x, write out. The
#    reference materialises 6 intermediates plus two NCHW<->NHWC conversions.
# 4) LEFT IN PYTORCH: output allocation only, plus a correctness fallback for
#    unsupported configs (C != 256, non-fp32, non-CUDA) — never taken here.
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 32
#define NTHREADS 256
#define CPG 8
#define NGROUPS 32

__global__ void wt_transform_kernel(const float* __restrict__ w,
                                    float* __restrict__ wt, int C) {
    int total = C * C * 9;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int rs = idx % 9;
    int t  = idx / 9;
    int c  = t % C;
    int n  = t / C;
    wt[((size_t)rs * C + c) * C + n] = w[idx];
}

template<bool FUSE_IN>
__global__ void conv_gn_kernel(const float* __restrict__ inp,
                               const float* __restrict__ wt,
                               const float* __restrict__ aff,
                               float* __restrict__ out,
                               float* __restrict__ psum,
                               float* __restrict__ psq,
                               int C, int H, int W, int HW, int NMT) {
    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int b  = blockIdx.z;
    const int n0 = blockIdx.y * BN;
    const int m0 = blockIdx.x * BM;
    const size_t base_b = (size_t)b * C * HW;

    __shared__ int   s_off[9 * BM];
    __shared__ int   s_hw[BM];
    __shared__ float s_ab[2 * BK * BM];
    __shared__ float s_gs[2][16];
    __shared__ float s_gq[2][16];

    float* sA = s_ab;
    float* sB = s_ab + BK * BM;

    for (int i = tid; i < BM; i += NTHREADS) {
        int p = m0 + i;
        if (p < HW) {
            s_hw[i] = p;
            int h = p / W, w = p % W;
#pragma unroll
            for (int rs = 0; rs < 9; ++rs) {
                int hh = h + (rs / 3) - 1;
                int ww = w + (rs % 3) - 1;
                s_off[rs * BM + i] =
                    (hh >= 0 && hh < H && ww >= 0 && ww < W) ? (hh * W + ww) : -1;
            }
        } else {
            s_hw[i] = -1;
#pragma unroll
            for (int rs = 0; rs < 9; ++rs) s_off[rs * BM + i] = -1;
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[4][2];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[mi][j], 0.0f);

    const float* affb = FUSE_IN ? (aff + (size_t)b * 2 * C) : (const float*)0;
    __syncthreads();

    for (int rs = 0; rs < 9; ++rs) {
        const int* offp = s_off + rs * BM;
        for (int c0 = 0; c0 < C; c0 += BK) {
            __syncthreads();
            for (int i = tid; i < BK * BM; i += NTHREADS) {
                int kk = i >> 7;
                int m  = i & (BM - 1);
                int off = offp[m];
                float v = 0.0f;
                if (off >= 0) {
                    int c = c0 + kk;
                    v = inp[base_b + (size_t)c * HW + off];
                    if (FUSE_IN) {
                        float t = fmaf(v, affb[c], affb[C + c]);
                        v = t * (1.0f / (1.0f + __expf(-t)));
                    }
                }
                sA[kk * BM + m] = v;
            }
            const float* wp = wt + ((size_t)rs * C + c0) * C + n0;
            for (int i = tid; i < BK * BN; i += NTHREADS) {
                int kk = i >> 7;
                int n  = i & (BN - 1);
                sB[kk * BN + n] = wp[(size_t)kk * C + n];
            }
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < BK / 8; ++ks) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32, wmma::col_major> af[4];
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> bf[2];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    wmma::load_matrix_sync(af[mi],
                        sA + ks * 8 * BM + warp_m * 64 + mi * 16, BM);
#pragma unroll
                    for (int t = 0; t < af[mi].num_elements; ++t)
                        af[mi].x[t] = wmma::__float_to_tf32(af[mi].x[t]);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::load_matrix_sync(bf[j],
                        sB + ks * 8 * BN + warp_n * 32 + j * 16, BN);
#pragma unroll
                    for (int t = 0; t < bf[j].num_elements; ++t)
                        bf[j].x[t] = wmma::__float_to_tf32(bf[j].x[t]);
                }
#pragma unroll
                for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                    for (int j = 0; j < 2; ++j)
                        wmma::mma_sync(acc[mi][j], af[mi], bf[j], acc[mi][j]);
            }
        }
    }

    __syncthreads();
    float* wbuf = s_ab + warp * 1024;     // [16 n][64 m]
    const int mbase = warp_m * 64;

    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            wmma::store_matrix_sync(wbuf + mi * 16, acc[mi][j], 64,
                                    wmma::mem_col_major);
        __syncwarp();
        const int nbase = n0 + warp_n * 32 + j * 16;
        float gs0 = 0.f, gq0 = 0.f, gs1 = 0.f, gq1 = 0.f;
        for (int nl = 0; nl < 16; ++nl) {
            const int n = nbase + nl;
            const float* col = wbuf + nl * 64;
            float s = 0.f, q = 0.f;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                int mm = lane + u * 32;
                int hw = s_hw[mbase + mm];
                if (hw >= 0) {
                    float v = col[mm];
                    s += v; q += v * v;
                    out[base_b + (size_t)n * HW + hw] = v;
                }
            }
            if (nl < CPG) { gs0 += s; gq0 += q; }
            else          { gs1 += s; gq1 += q; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gs0 += __shfl_down_sync(0xffffffffu, gs0, off);
            gq0 += __shfl_down_sync(0xffffffffu, gq0, off);
            gs1 += __shfl_down_sync(0xffffffffu, gs1, off);
            gq1 += __shfl_down_sync(0xffffffffu, gq1, off);
        }
        if (lane == 0) {
            int gi = warp_n * 4 + j * 2;
            s_gs[warp_m][gi]     = gs0;  s_gq[warp_m][gi]     = gq0;
            s_gs[warp_m][gi + 1] = gs1;  s_gq[warp_m][gi + 1] = gq1;
        }
        __syncwarp();
    }
    __syncthreads();
    if (tid < 16) {
        int g = n0 / CPG + tid;
        size_t o = ((size_t)(b * NGROUPS + g)) * NMT + blockIdx.x;
        psum[o] = s_gs[0][tid] + s_gs[1][tid];
        psq[o]  = s_gq[0][tid] + s_gq[1][tid];
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ psum,
                                   const float* __restrict__ psq,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ aff,
                                   int NMT, int C, int HW, float eps) {
    const int tid = threadIdx.x;
    const int bg  = blockIdx.x;
    const int g   = bg % NGROUPS;
    const int b   = bg / NGROUPS;
    const float* ps = psum + (size_t)bg * NMT;
    const float* pq = psq  + (size_t)bg * NMT;

    double s = 0.0, q = 0.0;
    for (int i = tid; i < NMT; i += blockDim.x) { s += (double)ps[i]; q += (double)pq[i]; }

    __shared__ double sh_s[256], sh_q[256];
    sh_s[tid] = s; sh_q[tid] = q;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { sh_s[tid] += sh_s[tid + off]; sh_q[tid] += sh_q[tid + off]; }
        __syncthreads();
    }
    __shared__ float smean, srstd;
    if (tid == 0) {
        double N = (double)HW * (double)CPG;
        double mean = sh_s[0] / N;
        double var  = sh_q[0] / N - mean * mean;
        if (var < 0.0) var = 0.0;
        smean = (float)mean;
        srstd = (float)(1.0 / sqrt(var + (double)eps));
    }
    __syncthreads();
    if (tid < CPG) {
        int c = g * CPG + tid;
        float sc = gamma[c] * srstd;
        aff[(size_t)b * 2 * C + c]     = sc;
        aff[(size_t)b * 2 * C + C + c] = beta[c] - smean * sc;
    }
}

__global__ void gn_silu_res_kernel(const float* __restrict__ y,
                                   const float* __restrict__ aff,
                                   const float* __restrict__ res,
                                   float* __restrict__ out,
                                   int C, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW) return;
    int c = blockIdx.y, b = blockIdx.z;
    size_t idx = ((size_t)b * C + c) * HW + i;
    float sc = aff[(size_t)b * 2 * C + c];
    float sh = aff[(size_t)b * 2 * C + C + c];
    float t = fmaf(y[idx], sc, sh);
    out[idx] = t * (1.0f / (1.0f + __expf(-t))) + res[idx];
}

__global__ void gn_silu_res_kernel_v4(const float4* __restrict__ y,
                                      const float* __restrict__ aff,
                                      const float4* __restrict__ res,
                                      float4* __restrict__ out,
                                      int C, int HW4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW4) return;
    int c = blockIdx.y, b = blockIdx.z;
    size_t idx = ((size_t)b * C + c) * HW4 + i;
    float sc = aff[(size_t)b * 2 * C + c];
    float sh = aff[(size_t)b * 2 * C + C + c];
    float4 v = y[idx];
    float4 r = res[idx];
    float t0 = fmaf(v.x, sc, sh), t1 = fmaf(v.y, sc, sh);
    float t2 = fmaf(v.z, sc, sh), t3 = fmaf(v.w, sc, sh);
    float4 o;
    o.x = t0 * (1.0f / (1.0f + __expf(-t0))) + r.x;
    o.y = t1 * (1.0f / (1.0f + __expf(-t1))) + r.y;
    o.z = t2 * (1.0f / (1.0f + __expf(-t2))) + r.z;
    o.w = t3 * (1.0f / (1.0f + __expf(-t3))) + r.w;
    out[idx] = o;
}

torch::Tensor fused_block(torch::Tensor x, torch::Tensor w1, torch::Tensor g1,
                          torch::Tensor b1, torch::Tensor w2, torch::Tensor g2,
                          torch::Tensor b2, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "fp32 CUDA input required");
    TORCH_CHECK(x.dim() == 4, "expected NCHW");
    auto xc  = x.is_contiguous()  ? x  : x.contiguous();
    auto w1c = w1.is_contiguous() ? w1 : w1.contiguous();
    auto w2c = w2.is_contiguous() ? w2 : w2.contiguous();
    auto g1c = g1.is_contiguous() ? g1 : g1.contiguous();
    auto b1c = b1.is_contiguous() ? b1 : b1.contiguous();
    auto g2c = g2.is_contiguous() ? g2 : g2.contiguous();
    auto b2c = b2.is_contiguous() ? b2 : b2.contiguous();

    const int B = (int)xc.size(0), C = (int)xc.size(1);
    const int H = (int)xc.size(2), W = (int)xc.size(3);
    const int HW = H * W;
    TORCH_CHECK(C % BN == 0 && (C / NGROUPS) == CPG, "unsupported channel config");
    const int NMT = (HW + BM - 1) / BM;

    auto opts = xc.options();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto wt   = torch::empty({2, (long)9 * C * C}, opts);
    auto y1   = torch::empty_like(xc);
    auto y2   = torch::empty_like(xc);
    auto out  = torch::empty_like(xc);
    auto psum = torch::empty({(long)B * NGROUPS * NMT}, opts);
    auto psq  = torch::empty({(long)B * NGROUPS * NMT}, opts);
    auto aff  = torch::empty({2, (long)B * 2 * C}, opts);

    float* wtp   = wt.data_ptr<float>();
    float* affp  = aff.data_ptr<float>();
    float* psump = psum.data_ptr<float>();
    float* psqp  = psq.data_ptr<float>();
    const size_t wstride = (size_t)9 * C * C;
    const size_t astride = (size_t)B * 2 * C;

    int tot = 9 * C * C;
    int tb  = (tot + 255) / 256;
    wt_transform_kernel<<<tb, 256, 0, stream>>>(w1c.data_ptr<float>(), wtp, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wt_transform_kernel<<<tb, 256, 0, stream>>>(w2c.data_ptr<float>(), wtp + wstride, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid(NMT, C / BN, B), block(NTHREADS);

    conv_gn_kernel<false><<<grid, block, 0, stream>>>(
        xc.data_ptr<float>(), wtp, nullptr, y1.data_ptr<float>(),
        psump, psqp, C, H, W, HW, NMT);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_kernel<<<B * NGROUPS, 256, 0, stream>>>(
        psump, psqp, g1c.data_ptr<float>(), b1c.data_ptr<float>(),
        affp, NMT, C, HW, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    conv_gn_kernel<true><<<grid, block, 0, stream>>>(
        y1.data_ptr<float>(), wtp + wstride, affp, y2.data_ptr<float>(),
        psump, psqp, C, H, W, HW, NMT);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_kernel<<<B * NGROUPS, 256, 0, stream>>>(
        psump, psqp, g2c.data_ptr<float>(), b2c.data_ptr<float>(),
        affp + astride, NMT, C, HW, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (HW % 4 == 0) {
        int HW4 = HW / 4;
        dim3 eg((HW4 + 255) / 256, C, B);
        gn_silu_res_kernel_v4<<<eg, 256, 0, stream>>>(
            (const float4*)y2.data_ptr<float>(), affp + astride,
            (const float4*)xc.data_ptr<float>(),
            (float4*)out.data_ptr<float>(), C, HW4);
    } else {
        dim3 eg((HW + 255) / 256, C, B);
        gn_silu_res_kernel<<<eg, 256, 0, stream>>>(
            y2.data_ptr<float>(), affp + astride, xc.data_ptr<float>(),
            out.data_ptr<float>(), C, HW);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_block(torch::Tensor x, torch::Tensor w1, torch::Tensor g1,
                          torch::Tensor b1, torch::Tensor w2, torch::Tensor g2,
                          torch::Tensor b2, double eps);
"""

_ext = load_inline(
    name="vae_resblock_implicit_gemm_d",
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
    """Full-forward (D) rewrite: own implicit-GEMM convs with GN/SiLU fused
    into their prologue/epilogue. See file header for the fusion map."""

    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps = float(eps.reshape(()).item())
        else:
            eps = float(eps)

        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256 and x.size(1) % 128 == 0):
            return self.ext.fused_block(x, conv1_weight, norm1_weight, norm1_bias,
                                        conv2_weight, norm2_weight, norm2_bias, eps)

        # Correctness fallback for configurations the kernel does not cover.
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=eps)
        out = F.silu(out)
        return out + x
