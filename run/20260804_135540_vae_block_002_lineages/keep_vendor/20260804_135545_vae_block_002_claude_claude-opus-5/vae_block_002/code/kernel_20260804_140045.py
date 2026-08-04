# ==========================================================================
# ModelNew — SOL problem 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY HEADER (REQUIRED)
#   1) Chosen granularity: (C) fuse many ops into one/few kernels.
#
#   2) Ops replaced (all of them are executed inside ONE load_inline
#      extension entry point `fused_block`, which is the whole forward):
#        - the two F.conv2d 3x3 calls          -> at::conv2d called from the
#                                                 extension, driven in
#                                                 channels-last (NHWC) so the
#                                                 cuDNN sm90 TF32 NHWC kernel
#                                                 runs WITHOUT the 6
#                                                 nchwToNhwc / nhwcToNchw
#                                                 transpose kernels that the
#                                                 reference pays (21% of time).
#        - both F.group_norm  (RowwiseMoments + ComputeFusedParams +
#          elementwise affine)                  -> custom CUDA kernels
#        - both F.silu                          -> fused into the same kernels
#        - the residual add                     -> fused into the same kernel
#        - the final NHWC->NCHW layout restore  -> fused (shared-mem transpose)
#          into the second GroupNorm epilogue kernel
#
#   3) Fusion map (few kernels total):
#        K1 gn_stats_kernel      : per-(batch,group) partial sum / sumsq over
#                                  NHWC conv output (chunked, always writes its
#                                  partial slot -> no unwritten tail for odd HW)
#        K2 gn_finalize_kernel   : partials -> mean / rstd  (double accum)
#        K3 gn_silu_apply_vec4   : (y-mean)*rstd*gamma+beta -> SiLU, NHWC->NHWC,
#                                  float4 vectorised   [GroupNorm1+SiLU1 fused]
#           gn_silu_apply_scalar : generic fallback (C % 4 != 0)
#        K4 gn_silu_res_t_kernel : (y-mean)*rstd*gamma+beta -> SiLU -> +residual
#                                  AND NHWC->NCHW transpose in shared memory
#                                  [GroupNorm2+SiLU2+add+layout in one pass]
#        So GN1+SiLU1 = 3 kernels, GN2+SiLU2+add+relayout = 3 kernels, instead
#        of the reference's 4 memory passes per norm plus 6 layout kernels.
#
#   4) What stays in PyTorch / vendor libs and why:
#        - conv2d itself -> cuDNN implicit-GEMM TF32 NHWC is at/near roofline;
#          re-writing it wins nothing, we only remove its layout tax.
#        - weight relayout to channels_last -> one-off tiny permute (2.4 MB),
#          cheaper than any custom path, done through ATen inside the extension.
#
#   Precision: everything stays float32 (inputs are float32); reductions
#   accumulate in double. TF32 tensor cores are used for the convolutions
#   exactly as the reference does (cudnn.allow_tf32 default True).
# ==========================================================================
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <utility>

#define BLK 256

__device__ __forceinline__ float siluf(float t) {
    return t / (1.0f + expf(-t));
}

__device__ __forceinline__ void blockReduce2(double &s, double &q) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
        q += __shfl_down_sync(0xffffffffu, q, off);
    }
    __shared__ double ss[BLK / 32];
    __shared__ double qq[BLK / 32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) { ss[wid] = s; qq[wid] = q; }
    __syncthreads();
    if (wid == 0) {
        double a  = (lane < (BLK / 32)) ? ss[lane] : 0.0;
        double bq = (lane < (BLK / 32)) ? qq[lane] : 0.0;
        #pragma unroll
        for (int off = (BLK / 64); off > 0; off >>= 1) {
            a  += __shfl_down_sync(0xffffffffu, a,  off);
            bq += __shfl_down_sync(0xffffffffu, bq, off);
        }
        s = a; q = bq;
    }
}

// ---------------------------------------------------------------------------
// K1: partial (sum, sumsq) per (batch, group) over an NHWC tensor.
// grid = (nchunks, B*G), block = BLK. Every block writes its slot (also when
// its pixel range is empty) so the partial buffer never has an unwritten tail.
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ y,
                                double* __restrict__ psum,
                                double* __restrict__ psq,
                                int G, int C, int CPG, int HW,
                                int pix_per_chunk, int nchunks, int vec4) {
    const int bg = blockIdx.y;
    const int b  = bg / G;
    const int g  = bg - b * G;
    const int chunk = blockIdx.x;

    const long base = (long)b * (long)HW * (long)C + (long)g * (long)CPG;
    const int p0 = chunk * pix_per_chunk;
    int p1 = p0 + pix_per_chunk;
    if (p1 > HW) p1 = HW;

    double s = 0.0, q = 0.0;
    if (vec4) {
        const int CPG4 = CPG >> 2;
        for (int p = p0 + (int)threadIdx.x; p < p1; p += (int)blockDim.x) {
            const float4* row =
                reinterpret_cast<const float4*>(y + base + (long)p * (long)C);
            for (int k = 0; k < CPG4; ++k) {
                float4 v = row[k];
                s += (double)v.x + (double)v.y + (double)v.z + (double)v.w;
                q += (double)v.x * (double)v.x + (double)v.y * (double)v.y
                   + (double)v.z * (double)v.z + (double)v.w * (double)v.w;
            }
        }
    } else {
        for (int p = p0 + (int)threadIdx.x; p < p1; p += (int)blockDim.x) {
            const float* row = y + base + (long)p * (long)C;
            for (int k = 0; k < CPG; ++k) {
                float v = row[k];
                s += (double)v;
                q += (double)v * (double)v;
            }
        }
    }
    blockReduce2(s, q);
    if (threadIdx.x == 0) {
        psum[(long)bg * nchunks + chunk] = s;
        psq [(long)bg * nchunks + chunk] = q;
    }
}

// ---------------------------------------------------------------------------
// K2: partials -> mean / rstd  (one block per (batch, group))
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const double* __restrict__ psum,
                                   const double* __restrict__ psq,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nchunks, double N, double eps) {
    const int bg = blockIdx.x;
    double s = 0.0, q = 0.0;
    for (int i = (int)threadIdx.x; i < nchunks; i += (int)blockDim.x) {
        s += psum[(long)bg * nchunks + i];
        q += psq [(long)bg * nchunks + i];
    }
    blockReduce2(s, q);
    if (threadIdx.x == 0) {
        double m = s / N;
        double v = q / N - m * m;
        if (!(v > 0.0)) v = 0.0;
        mean[bg] = (float)m;
        rstd[bg] = (float)(1.0 / sqrt(v + eps));
    }
}

// ---------------------------------------------------------------------------
// K3: GroupNorm affine + SiLU, NHWC -> NHWC (vectorised by float4)
// ---------------------------------------------------------------------------
__global__ void gn_silu_apply_vec4(const float4* __restrict__ y,
                                   float4* __restrict__ z,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ mean,
                                   const float* __restrict__ rstd,
                                   int n4, int C4, int HW, int CPG4, int G) {
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * blockDim.x) {
        const int c4   = i % C4;
        const int rest = i / C4;          // b * HW + p
        const int b    = rest / HW;
        const int g    = c4 / CPG4;
        const int bg   = b * G + g;
        const float m = mean[bg];
        const float r = rstd[bg];
        const float4 v  = y[i];
        const float4 ga = g4[c4];
        const float4 be = b4[c4];
        float4 o;
        o.x = siluf((v.x - m) * r * ga.x + be.x);
        o.y = siluf((v.y - m) * r * ga.y + be.y);
        o.z = siluf((v.z - m) * r * ga.z + be.z);
        o.w = siluf((v.w - m) * r * ga.w + be.w);
        z[i] = o;
    }
}

__global__ void gn_silu_apply_scalar(const float* __restrict__ y,
                                     float* __restrict__ z,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ rstd,
                                     long n, int C, int HW, int CPG, int G) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) {
        const int c    = (int)(i % (long)C);
        const long rst = i / (long)C;
        const int b    = (int)(rst / (long)HW);
        const int bg   = b * G + c / CPG;
        const float t = ((float)y[i] - mean[bg]) * rstd[bg] * gamma[c] + beta[c];
        z[i] = siluf(t);
    }
}

// ---------------------------------------------------------------------------
// K4: GroupNorm affine + SiLU + residual add + NHWC -> NCHW transpose.
// grid = (ceil(HW/32), ceil(C/32), B), block = (32, 8)
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_t_kernel(const float* __restrict__ y,   // NHWC
                                     const float* __restrict__ res, // NCHW
                                     float* __restrict__ out,       // NCHW
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ rstd,
                                     int C, int HW, int CPG, int G) {
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int b  = blockIdx.z;
    const long ybase = (long)b * (long)HW * (long)C;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    {
        const int c = c0 + tx;
        float gm = 0.f, bt = 0.f, mn = 0.f, rs = 0.f;
        if (c < C) {
            gm = gamma[c];
            bt = beta[c];
            const int bg = b * G + c / CPG;
            mn = mean[bg];
            rs = rstd[bg];
        }
        for (int j = ty; j < 32; j += 8) {
            const int p = p0 + j;
            float v = 0.f;
            if (p < HW && c < C) {
                const float t = (y[ybase + (long)p * (long)C + c] - mn) * rs * gm + bt;
                v = siluf(t);
            }
            tile[j][tx] = v;
        }
    }
    __syncthreads();
    {
        const int p = p0 + tx;
        for (int j = ty; j < 32; j += 8) {
            const int c = c0 + j;
            if (p < HW && c < C) {
                const long o = ((long)b * (long)C + c) * (long)HW + p;
                out[o] = tile[tx][j] + res[o];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static std::pair<at::Tensor, at::Tensor> gn_stats(const at::Tensor& y,
                                                  int B, int C, int G, int HW,
                                                  double eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int CPG = C / G;
    const int BG  = B * G;

    const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int target = 6 * sm;
    int nchunks = (target + BG - 1) / BG;
    if (nchunks < 1) nchunks = 1;
    int maxchunks = (HW + 255) / 256;
    if (maxchunks < 1) maxchunks = 1;
    if (nchunks > maxchunks) nchunks = maxchunks;
    int pix = (HW + nchunks - 1) / nchunks;
    if (pix < 1) pix = 1;
    nchunks = (HW + pix - 1) / pix;          // no empty chunk
    if (nchunks < 1) nchunks = 1;

    auto dopt = y.options().dtype(at::kDouble);
    auto psum = at::empty({(long)BG * nchunks}, dopt);
    auto psq  = at::empty({(long)BG * nchunks}, dopt);
    auto mean = at::empty({(long)BG}, y.options());
    auto rstd = at::empty({(long)BG}, y.options());

    const int vec4 = ((CPG % 4 == 0) && (C % 4 == 0)) ? 1 : 0;

    dim3 grid(nchunks, BG);
    gn_stats_kernel<<<grid, BLK, 0, stream>>>(
        y.data_ptr<float>(), psum.data_ptr<double>(), psq.data_ptr<double>(),
        G, C, CPG, HW, pix, nchunks, vec4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finalize_kernel<<<BG, BLK, 0, stream>>>(
        psum.data_ptr<double>(), psq.data_ptr<double>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        nchunks, (double)HW * (double)CPG, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_pair(mean, rstd);
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor gam1, torch::Tensor bet1,
                          torch::Tensor w2, torch::Tensor gam2, torch::Tensor bet2,
                          double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(w1.scalar_type() == at::kFloat && w2.scalar_type() == at::kFloat,
                "float32 weights only");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    const int B = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int G = 32;
    TORCH_CHECK(C % G == 0, "C must be divisible by 32 groups");
    const int CPG = C / G;
    const int HW = H * W;

    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c    = x.is_contiguous() ? x : x.contiguous();
    auto x_nhwc = x_c.contiguous(at::MemoryFormat::ChannelsLast);
    auto w1c    = w1.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c    = w2.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1c    = gam1.is_contiguous() ? gam1 : gam1.contiguous();
    auto b1c    = bet1.is_contiguous() ? bet1 : bet1.contiguous();
    auto g2c    = gam2.is_contiguous() ? gam2 : gam2.contiguous();
    auto b2c    = bet2.is_contiguous() ? bet2 : bet2.contiguous();

    std::vector<int64_t> one{1, 1};

    // ---- conv1 (cuDNN TF32 NHWC, no layout tax) ----
    auto y1 = at::conv2d(x_nhwc, w1c, c10::optional<at::Tensor>(),
                         one, one, one, 1);
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GroupNorm1 + SiLU1 (fused) ----
    auto st1 = gn_stats(y1, B, C, G, HW, eps);
    auto z1 = at::empty(y1.sizes(),
                        y1.options().memory_format(at::MemoryFormat::ChannelsLast));
    {
        const long n = (long)B * (long)HW * (long)C;
        if ((C % 4 == 0) && (CPG % 4 == 0)) {
            const int n4 = (int)(n / 4);
            int blocks = (n4 + BLK - 1) / BLK;
            blocks = std::min(blocks, 8192);
            if (blocks < 1) blocks = 1;
            gn_silu_apply_vec4<<<blocks, BLK, 0, stream>>>(
                reinterpret_cast<const float4*>(y1.data_ptr<float>()),
                reinterpret_cast<float4*>(z1.data_ptr<float>()),
                g1c.data_ptr<float>(), b1c.data_ptr<float>(),
                st1.first.data_ptr<float>(), st1.second.data_ptr<float>(),
                n4, C / 4, HW, CPG / 4, G);
        } else {
            long blocksl = (n + BLK - 1) / BLK;
            int blocks = (int)std::min<long>(blocksl, 8192);
            if (blocks < 1) blocks = 1;
            gn_silu_apply_scalar<<<blocks, BLK, 0, stream>>>(
                y1.data_ptr<float>(), z1.data_ptr<float>(),
                g1c.data_ptr<float>(), b1c.data_ptr<float>(),
                st1.first.data_ptr<float>(), st1.second.data_ptr<float>(),
                n, C, HW, CPG, G);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 ----
    auto y2 = at::conv2d(z1, w2c, c10::optional<at::Tensor>(), one, one, one, 1);
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- GroupNorm2 + SiLU2 + residual + NHWC->NCHW (single fused kernel) ----
    auto st2 = gn_stats(y2, B, C, G, HW, eps);
    auto out = at::empty({B, C, H, W}, x.options());
    {
        dim3 block(32, 8);
        dim3 grid((HW + 31) / 32, (C + 31) / 32, B);
        gn_silu_res_t_kernel<<<grid, block, 0, stream>>>(
            y2.data_ptr<float>(), x_c.data_ptr<float>(), out.data_ptr<float>(),
            g2c.data_ptr<float>(), b2c.data_ptr<float>(),
            st2.first.data_ptr<float>(), st2.second.data_ptr<float>(),
            C, HW, CPG, G);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
'''

cpp_src = r'''
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor w1, torch::Tensor gam1, torch::Tensor bet1,
                          torch::Tensor w2, torch::Tensor gam2, torch::Tensor bet2,
                          double eps);
'''

_ext = load_inline(
    name="vae_resblock_fused_c",
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
        "-gencode=arch=compute_90,code=sm_90",
    ],
    extra_ldflags=[""],
)

# cuDNN algo selection for the (few) fixed shapes used by the harness.
torch.backends.cudnn.benchmark = True


class ModelNew(nn.Module):
    """
    Granularity (C): the whole residual block is executed by one extension
    entry point that issues 2 vendor convs (NHWC/TF32) plus 6 custom kernels
    which absorb both GroupNorms, both SiLUs, the residual add and the final
    layout restore. See the file header for the full fusion map.

    Stateless (the SOL reference is stateless): all weights arrive as inputs,
    so there is no parameter holder to keep in parity.
    """

    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self,
                x,
                conv1_weight,
                norm1_weight,
                norm1_bias,
                conv2_weight,
                norm2_weight,
                norm2_bias,
                eps):
        if isinstance(eps, torch.Tensor):
            eps = float(eps.item())
        return self.ext.fused_block(x,
                                    conv1_weight, norm1_weight, norm1_bias,
                                    conv2_weight, norm2_weight, norm2_bias,
                                    float(eps))
