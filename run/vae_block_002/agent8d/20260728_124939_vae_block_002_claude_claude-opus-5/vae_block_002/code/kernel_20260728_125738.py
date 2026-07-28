# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# SEED GRANULARITY: (C) "fuse many ops into one/few kernels".
#
# 1) Chosen granularity: (C). Every elementwise / normalization / layout op is
#    fused into a small number of hand-written CUDA kernels; the two 3x3
#    convolutions stay as vendor (cuDNN) calls issued from inside the
#    load_inline extension.
#
# 2) Ops replaced by custom CUDA:
#      - GroupNorm(32) statistics (RowwiseMoments + ComputeFusedParams)  x2
#      - GroupNorm affine apply                                          x2
#      - SiLU                                                            x2
#      - residual add                                                    x1
#      - ALL NCHW<->NHWC layout conversions that cuDNN was doing
#        internally (nchwToNhwc / nhwcToNchw copies, 168us + 94us in the
#        reference profile) are removed: the whole block is executed in
#        NHWC (channels_last) so the sm80_xmma NHWC implicit-GEMM sees
#        native layout, and the only remaining transposes are one explicit
#        NCHW->NHWC of x and one NHWC->NCHW folded into the last epilogue.
#
# 3) Fusion map (kernel <- ops):
#      K1 nchw2nhwc_kernel        <- layout conversion of x (shared-mem tiled)
#      cuDNN at::conv2d (NHWC)    <- conv1 (vendor, called in-extension)
#      K2 gn_moments_nhwc         <- partial sum/sumsq per (n,group) (norm1)
#      K3 gn_finalize             <- mean/rstd + fold gamma/beta -> per-(n,c) A,B
#      K4 gn_silu_nhwc            <- (x*A+B) + SiLU, vectorized float4, NHWC->NHWC
#      cuDNN at::conv2d (NHWC)    <- conv2 (vendor, called in-extension)
#      K2,K3 again                <- norm2 statistics / fused params
#      K5 gn_silu_res_nhwc2nchw   <- norm2 apply + SiLU + residual add +
#                                    NHWC->NCHW transpose, all in ONE pass
#    => GroupNorm+SiLU+add+layout traffic drops from ~1.4 GB to ~0.6 GB.
#
# 4) What stays in PyTorch / vendor land and why:
#      - conv3x3 256->256: the profiled vendor TF32 implicit GEMM runs at
#        ~92 TFLOPS (~88% of this GPU's TF32 peak); reimplementing it cannot
#        win, so it is kept (invoked via at::conv2d inside the extension).
#      - weight NCHW->channels_last packing: 2.4 MB, negligible, uses
#        .contiguous(ChannelsLast) inside the extension.
#      - a pure-PyTorch fallback path is kept only for shapes/dtypes the
#        custom kernels do not cover (C%128!=0, non-fp32, non-CUDA).
#
# Precision: fp32 storage/arith throughout; reductions accumulate fp32 per
# thread then fp64 for the final combine; TF32 tensor cores in conv only
# (identical to the reference, which runs with allow_tf32=True).
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>

#define GROUPS 32

__device__ __forceinline__ float silu_f(float y) {
    return y / (1.0f + expf(-y));
}

// ---------------------------------------------------------------- K1
// NCHW -> NHWC (channels_last) tiled transpose. C must be a multiple of 32.
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int HW, int C) {
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const size_t off = (size_t)blockIdx.z * (size_t)C * (size_t)HW;
    const float* inb = in + off;
    float*       outb = out + off;

#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        int c = c0 + threadIdx.y + i;
        int p = p0 + threadIdx.x;
        float v = 0.0f;
        if (p < HW) v = inb[(size_t)c * HW + p];
        tile[threadIdx.y + i][threadIdx.x] = v;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        int p = p0 + threadIdx.y + i;
        int c = c0 + threadIdx.x;
        if (p < HW) outb[(size_t)p * C + c] = tile[threadIdx.x][threadIdx.y + i];
    }
}

// ---------------------------------------------------------------- K2
// GroupNorm partial moments over an NHWC tensor.
// grid = (NCHUNK, GROUPS, N), block = 128
__global__ void gn_moments_nhwc(const float* __restrict__ in,
                                float* __restrict__ psum,
                                float* __restrict__ psumsq,
                                int HW, int C, int CPG, int NCHUNK, int LOGSPP) {
    const int chunk = blockIdx.x;
    const int g     = blockIdx.y;
    const int n     = blockIdx.z;
    const int G     = gridDim.y;

    const int spp = 1 << LOGSPP;              // float4 slots per pixel
    const int chunkPix = (HW + NCHUNK - 1) / NCHUNK;
    const int pb = chunk * chunkPix;
    int pe = pb + chunkPix;
    if (pe > HW) pe = HW;

    const float* base = in + (size_t)n * (size_t)HW * (size_t)C + (size_t)g * CPG;

    float s = 0.0f, ss = 0.0f;
    if (pb < pe) {
        const long nslots = (long)(pe - pb) * (long)spp;
        for (long i = threadIdx.x; i < nslots; i += blockDim.x) {
            const long p    = (long)pb + (i >> LOGSPP);
            const int  part = (int)(i & (long)(spp - 1));
            const float4 v = *reinterpret_cast<const float4*>(
                base + (size_t)p * (size_t)C + (size_t)(part << 2));
            s += v.x + v.y + v.z + v.w;
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
    }

    const unsigned
