# =============================================================================
# 002_vae_conv3x3_groupnorm_silu_residual_fused  --  ModelNew
#
# L2-PERSIST PRODUCER/CONSUMER CHUNK FUSION (l2_persist_chunk_fusion):
# each convolution is sliced along the batch axis into L2-sized chunks and the
# GroupNorm moment pass (gn_partial) is launched on each chunk immediately
# after that chunk's conv, on the same stream, with an L2 persisting
# access-policy window over the chunk's output -- so the moment pass reads the
# conv output out of L2 instead of re-reading it from DRAM.
# The persisting-L2 reservation is saved / raised / restored INSIDE forward.
# =============================================================================
import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


def _find_cutlass():
    cands = [
        os.environ.get("CUTLASS_PATH", ""),
        os.environ.get("CUTLASS_DIR", ""),
        "/home/otter77/git_project/KernelMem/third_party/cutlass",
        "/home/elek/KernelMem/third_party/cutlass",
        os.path.expanduser("~/KernelMem/third_party/cutlass"),
    ]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "include", "cutlass")):
            return os.path.join(c, "include")
        if c and os.path.isdir(os.path.join(c, "cutlass")):
            return c
    return "/home/otter77/git_project/KernelMem/third_party/cutlass/include"


CUTLASS_INC = _find_cutlass()

cuda_src = r'''
#include <cudaTypedefs.h>
using PFN_cuTensorMapEncodeTiled  = PFN_cuTensorMapEncodeTiled_v12000;
using PFN_cuTensorMapEncodeIm2col = PFN_cuTensorMapEncodeIm2col_v12000;

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstring>
#include <map>
#include <mutex>

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// Plan item 8: anti-regression compile-time switch -- set to 1 to force the
// original single-shot (un-chunked) conv + moments path.
#ifndef FORCE_SINGLE_SHOT
#define FORCE_SINGLE_SHOT 0
#endif

// ---------------------------------------------------------------------------
// CUTLASS TF32 implicit-GEMM 3x3 convolution (NHWC activations, KRSC filters).
// 128x128x16 threadblock, 64x64x16 warp tile, m16n8k8 MMA, fp32 accumulate.
// ---------------------------------------------------------------------------
using ConvKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    float, cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<ConvKernel>;

// Plan item 3: explicit base-pointer + N-override convolution range.
static void conv3x3_nhwc_range(const float* xptr, const float* wptr, float* outptr,
                               int Nsub, int H, int W, int C, int K, int R, int S,
                               const torch::TensorOptions& opts, cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize ps(Nsub, H, W, C, K, R, S, H, W,
                                      1, 1, 1, 1, 1, 1,
                                      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC> ra(
      const_cast<float*>(xptr), cutlass::layout::TensorNHWC::packed({Nsub, H, W, C}));
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC> rb(
      const_cast<float*>(wptr), cutlass::layout::TensorNHWC::packed({K, R, S, C}));
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC> rd(
      outptr, cutlass::layout::TensorNHWC::packed({Nsub, H, W, K}));
  typename ImplicitGemm::Arguments args{ps, ra, rb, rd, rd, {1.0f, 0.0f}};
  ImplicitGemm op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
              "cutlass conv: unsupported problem");
  size_t ws = op.get_workspace_size(args);
  void* wsptr = nullptr;
  torch::Tensor wsbuf;
  if (ws > 0) {
    wsbuf = torch::empty({(long)ws}, opts.dtype(torch::kUInt8));
    wsptr = wsbuf.data_ptr();
  }
  TORCH_CHECK(op.initialize(args, wsptr, stream) == cutlass::Status::kSuccess,
              "cutlass conv: initialize failed");
  TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "cutlass conv: launch failed");
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_f(float v) {
  return v / (1.0f + __expf(-v));
}

#define TT 64  // transpose tile (pixels x channels)

// NCHW -> NHWC ---------------------------------------------------------------
__global__ __launch_bounds__(256) void nchw2nhwc_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int HW, int C, int vec) {
  __shared__ float sm[TT][TT + 1];   // [c_local][p_local]
  const int p0 = blockIdx.x * TT;
  const int c0 = blockIdx.y * TT;
  const int n  = blockIdx.z;
  const float* base = src + (size_t)n * (size_t)C * HW;
  float*      obase = dst + (size_t)n * (size_t)HW * C;
  const int t = threadIdx.x;
  const int lc = t >> 4;
  const int lp = (t & 15) * 4;
  const int lp2 = t >> 4;
  const int lc2 = (t & 15) * 4;
  const bool full = (vec != 0) && (p0 + TT <= HW);

  if (full) {   // interior tile: no bounds checks at all
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = lc + r * 16;
      float4 v = *reinterpret_cast<const float4*>(base + (size_t)(c0 + c) * HW + p0 + lp);
      sm[c][lp] = v.x; sm[c][lp + 1] = v.y; sm[c][lp + 2] = v.z; sm[c][lp + 3] = v.w;
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp2 + r * 16;
      float4 v;
      v.x = sm[lc2][p]; v.y = sm[lc2 + 1][p]; v.z = sm[lc2 + 2][p]; v.w = sm[lc2 + 3][p];
      *reinterpret_cast<float4*>(obase + (size_t)(p0 + p) * C + c0 + lc2) = v;
    }
  } else {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = lc + r * 16;
      const float* s = base + (size_t)(c0 + c) * HW + p0 + lp;
      #pragma unroll
      for (int j = 0; j < 4; ++j) sm[c][lp + j] = (p0 + lp + j < HW) ? s[j] : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp2 + r * 16;
      if (p0 + p >= HW) continue;
      float4 v;
      v.x = sm[lc2][p]; v.y = sm[lc2 + 1][p]; v.z = sm[lc2 + 2][p]; v.w = sm[lc2 + 3][p];
      *reinterpret_cast<float4*>(obase + (size_t)(p0 + p) * C + c0 + lc2) = v;
    }
  }
}

// GroupNorm partial moments over NHWC tensor ---------------------------------
// Plan item 1: batch-range offset n0.  Every (n, chunk, g) slot is still
// written exactly once (empty chunks store zeros), and each slot's partial sum
// is bitwise identical to the un-chunked launch.
__global__ __launch_bounds__(256) void gn_partial_kernel(
    const float* __restrict__ y, float* __restrict__ part,
    int HW, int C, int G, int Cg, int pixPerChunk, int numChunks, int n0) {
  const int tpp = C >> 2;                 // float4 lanes per pixel
  const int ppp = 256 / tpp;              // pixels handled per pass
  const int c4  = threadIdx.x % tpp;
  const int pr  = threadIdx.x / tpp;
  const int n   = n0 + blockIdx.y;
  const int p0  = blockIdx.x * pixPerChunk;
  const int p1  = min(HW, p0 + pixPerChunk);
  const float4* yb = reinterpret_cast<const float4*>(y + (size_t)n * (size_t)HW * C);

  float s = 0.f, ss = 0.f;
  for (int p = p0 + pr; p < p1; p += ppp) {
    float4 v = yb[(size_t)p * tpp + c4];
    s  += v.x + v.y + v.z + v.w;
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  __shared__ float sms[256], smss[256];
  sms[threadIdx.x] = s; smss[threadIdx.x] = ss;
  __syncthreads();

  if ((int)threadIdx.x < G) {
    const int g = threadIdx.x;
    const int c4lo = (g * Cg) >> 2;
    const int c4hi = ((g + 1) * Cg) >> 2;
    float ts = 0.f, tss = 0.f;
    for (int i = 0; i < ppp; ++i)
      for (int c = c4lo; c < c4hi; ++c) {
        int idx = i * tpp + c;
        ts += sms[idx]; tss += smss[idx];
      }
    size_t o = (((size_t)n * numChunks + blockIdx.x) * G + g) * 2;
    part[o] = ts; part[o + 1] = tss;
  }
}

__global__ __launch_bounds__(128) void gn_finalize_kernel(
    const float* __restrict__ part, float* __restrict__ mean, float* __restrict__ rstd,
    int numChunks, int G, long cnt, float eps) {
  const int ng = blockIdx.x;              // n * G + g
  const int n = ng / G, g = ng % G;
  double s = 0.0, ss = 0.0;
  for (int c = threadIdx.x; c < numChunks; c += blockDim.x) {
    size_t o = (((size_t)n * numChunks + c) * G + g) * 2;
    s  += (double)part[o];
    ss += (double)part[o + 1];
  }
  __shared__ double bs[128], bss[128];
  bs[threadIdx.x] = s; bss[threadIdx.x] = ss;
  __syncthreads();
  for (int off = 64; off > 0; off >>= 1) {
    if ((int)threadIdx.x < off) {
      bs[threadIdx.x] += bs[threadIdx.x + off];
      bss[threadIdx.x] += bss[threadIdx.x + off];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double m = bs[0] / (double)cnt;
    double v = bss[0] / (double)cnt - m * m;
    if (v < 0.0) v = 0.0;
    mean[ng] = (float)m;
    rstd[ng] = (float)(1.0 / sqrt(v + (double)eps));
  }
}

// GroupNorm + SiLU, NHWC -> NHWC ---------------------------------------------
__global__ __launch_bounds__(256) void gn_silu_nhwc_kernel(
    const float* __restrict__ y, float* __restrict__ out,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    int HW, int C, int G, int Cg) {
  const int tpp = C >> 2;
  const int n = blockIdx.y;
  const float4* yb = reinterpret_cast<const float4*>(y + (size_t)n * (size_t)HW * C);
  float4* ob = reinterpret_cast<float4*>(out + (size_t)n * (size_t)HW * C);
  const size_t total = (size_t)HW * tpp;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (size_t)gridDim.x * blockDim.x) {
    const int c4 = (int)(i % (size_t)tpp);
    const int c = c4 * 4;
    const int g = c / Cg;
    const float mu = mean[n * G + g], rs = rstd[n * G + g];
    float4 v = yb[i];
    float4 gm = reinterpret_cast<const float4*>(gamma)[c4];
    float4 bt = reinterpret_cast<const float4*>(beta)[c4];
    float4 o;
    o.x = silu_f((v.x - mu) * rs * gm.x + bt.x);
    o.y = silu_f((v.y - mu) * rs * gm.y + bt.y);
    o.z = silu_f((v.z - mu) * rs * gm.z + bt.z);
    o.w = silu_f((v.w - mu) * rs * gm.w + bt.w);
    ob[i] = o;
  }
}

// GroupNorm + SiLU + residual + NHWC -> NCHW ---------------------------------
__global__ __launch_bounds__(256) void gn_silu_add_nhwc2nchw_kernel(
    const float* __restrict__ y, const float* __restrict__ res, float* __restrict__ out,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    int HW, int C, int G, int Cg, int vec) {
  __shared__ float sm[TT][TT + 1];   // [c_local][p_local]
  const int p0 = blockIdx.x * TT;
  const int c0 = blockIdx.y * TT;
  const int n  = blockIdx.z;
  const float* yb = y + (size_t)n * (size_t)HW * C;
  const int t = threadIdx.x;

  // phase 1: read NHWC, normalize + affine + silu, stage transposed in shared
  const int lp = t >> 4;             // pixel within tile
  const int lc = (t & 15) * 4;       // channel within tile
  const int c  = c0 + lc;
  const int g  = c / Cg;
  const float mu = mean[n * G + g], rs = rstd[n * G + g];
  const float4 gm = reinterpret_cast<const float4*>(gamma)[c >> 2];
  const float4 bt = reinterpret_cast<const float4*>(beta)[c >> 2];

  const int lc2 = t >> 4;
  const int lp2 = (t & 15) * 4;
  const size_t nbase = (size_t)n * (size_t)C * HW;
  const bool full = (vec != 0) && (p0 + TT <= HW);

  if (full) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp + r * 16;
      float4 v = *reinterpret_cast<const float4*>(yb + (size_t)(p0 + p) * C + c);
      sm[lc + 0][p] = silu_f((v.x - mu) * rs * gm.x + bt.x);
      sm[lc + 1][p] = silu_f((v.y - mu) * rs * gm.y + bt.y);
      sm[lc + 2][p] = silu_f((v.z - mu) * rs * gm.z + bt.z);
      sm[lc + 3][p] = silu_f((v.w - mu) * rs * gm.w + bt.w);
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int cc = lc2 + r * 16;
      const size_t off = nbase + (size_t)(c0 + cc) * HW + p0 + lp2;
      float4 rv = *reinterpret_cast<const float4*>(res + off);
      float4 o;
      o.x = sm[cc][lp2 + 0] + rv.x;
      o.y = sm[cc][lp2 + 1] + rv.y;
      o.z = sm[cc][lp2 + 2] + rv.z;
      o.w = sm[cc][lp2 + 3] + rv.w;
      *reinterpret_cast<float4*>(out + off) = o;
    }
  } else {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = lp + r * 16;
      if (p0 + p >= HW) continue;
      float4 v = *reinterpret_cast<const float4*>(yb + (size_t)(p0 + p) * C + c);
      sm[lc + 0][p] = silu_f((v.x - mu) * rs * gm.x + bt.x);
      sm[lc + 1][p] = silu_f((v.y - mu) * rs * gm.y + bt.y);
      sm[lc + 2][p] = silu_f((v.z - mu) * rs * gm.z + bt.z);
      sm[lc + 3][p] = silu_f((v.w - mu) * rs * gm.w + bt.w);
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int cc = lc2 + r * 16;
      const size_t off = nbase + (size_t)(c0 + cc) * HW + p0 + lp2;
      if (vec && p0 + lp2 + 3 < HW) {
        float4 rv = *reinterpret_cast<const float4*>(res + off);
        float4 o;
        o.x = sm[cc][lp2 + 0] + rv.x;
        o.y = sm[cc][lp2 + 1] + rv.y;
        o.z = sm[cc][lp2 + 2] + rv.z;
        o.w = sm[cc][lp2 + 3] + rv.w;
        *reinterpret_cast<float4*>(out + off) = o;
      } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          if (p0 + lp2 + j < HW) out[off + j] = sm[cc][lp2 + j] + res[off + j];
      }
    }
  }
}

// KCRS -> KRSC filter permutation --------------------------------------------
__global__ __launch_bounds__(256) void kcrs2krsc_kernel(
    const float* __restrict__ w, float* __restrict__ o, int C, int RS) {
  extern __shared__ float sm[];
  const int k = blockIdx.x;
  const float* wb = w + (size_t)k * C * RS;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float* s = wb + (size_t)c * RS;
    for (int j = 0; j < RS; ++j) sm[j * C + c] = s[j];
  }
  __syncthreads();
  float* ob = o + (size_t)k * C * RS;
  for (int i = threadIdx.x; i < C * RS; i += blockDim.x) ob[i] = sm[i];
}

// ---------------------------------------------------------------------------
// host driver
// ---------------------------------------------------------------------------
static inline torch::Tensor to_nhwc_weight(const torch::Tensor& w, cudaStream_t stream) {
  const int K = (int)w.size(0), C = (int)w.size(1);
  const int R = (int)w.size(2), S = (int)w.size(3);
  const int RS = R * S;
  const size_t smem = (size_t)C * RS * sizeof(float);
  if (smem <= 48 * 1024) {
    auto o = torch::empty({K, R, S, C}, w.options());
    kcrs2krsc_kernel<<<K, 256, smem, stream>>>(w.data_ptr<float>(), o.data_ptr<float>(), C, RS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
  }
  return w.permute({0, 2, 3, 1}).contiguous();
}

// Plan item 2: moment pass split into a batch-range partial launch and a
// single finalize launch.
static inline void moments_partial_range(const float* y, int n0, int nN, int HW, int C,
                                         int G, float* part, int numChunks,
                                         int pixPerChunk, cudaStream_t stream) {
  dim3 grid(numChunks, nN);
  gn_partial_kernel<<<grid, 256, 0, stream>>>(y, part, HW, C, G, C / G,
                                              pixPerChunk, numChunks, n0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static inline void moments_finalize(const float* part, float* mean, float* rstd,
                                    int NG, int numChunks, int G, long cnt, float eps,
                                    cudaStream_t stream) {
  gn_finalize_kernel<<<NG, 128, 0, stream>>>(part, mean, rstd, numChunks, G, cnt, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Plan item 5: L2 persisting access-policy window support (guarded; if the
// runtime/device does not support it we silently skip -- correctness never
// depends on it).
struct L2Info {
  int l2_size = 0;
  int max_persist = 0;
  int max_window = 0;
  bool ok = false;
};

static const L2Info& l2_info() {
  static L2Info info = [] {
    L2Info i;
#if CUDART_VERSION >= 11000
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return i;
    if (cudaDeviceGetAttribute(&i.l2_size, cudaDevAttrL2CacheSize, dev) != cudaSuccess)
      return i;
    if (cudaDeviceGetAttribute(&i.max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev)
        != cudaSuccess)
      return i;
    if (cudaDeviceGetAttribute(&i.max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev)
        != cudaSuccess)
      return i;
    i.ok = (i.l2_size > 0 && i.max_persist > 0 && i.max_window > 0);
#endif
    return i;
  }();
  return info;
}

// The persisting-L2 reservation is device-global state.  It must be raised and
// restored WITHIN a single forward() call, never from a static one-shot
// initializer, otherwise it outlives this kernel and slows unrelated work.
static inline size_t save_persist_limit() {
  size_t prev = 0;
#if CUDART_VERSION >= 11000
  if (cudaDeviceGetLimit(&prev, cudaLimitPersistingL2CacheSize) != cudaSuccess) {
    (void)cudaGetLastError();
    prev = 0;
  }
#endif
  return prev;
}

static inline void raise_persist_limit() {
#if CUDART_VERSION >= 11000
  const L2Info& info = l2_info();
  if (!info.ok) return;
  size_t budget = (size_t)info.l2_size / 2;
  if (budget > (size_t)info.max_persist) budget = (size_t)info.max_persist;
  if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, budget) != cudaSuccess)
    (void)cudaGetLastError();
#endif
}

static inline void restore_persist_limit(size_t prev) {
#if CUDART_VERSION >= 11000
  if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prev) != cudaSuccess)
    (void)cudaGetLastError();
#endif
}

static inline void set_persist_window(cudaStream_t stream, const void* base,
                                      size_t bytes) {
#if CUDART_VERSION >= 11000
  const L2Info& info = l2_info();
  if (!info.ok) return;
  cudaStreamAttrValue v;
  std::memset(&v, 0, sizeof(v));
  size_t nb = bytes;
  if (nb > (size_t)info.max_window) nb = (size_t)info.max_window;
  v.accessPolicyWindow.base_ptr = const_cast<void*>(base);
  v.accessPolicyWindow.num_bytes = nb;
  v.accessPolicyWindow.hitRatio = 1.0f;
  v.accessPolicyWindow.hitProp = (nb == 0) ? cudaAccessPropertyNormal
                                           : cudaAccessPropertyPersisting;
  v.accessPolicyWindow.missProp = (nb == 0) ? cudaAccessPropertyNormal
                                            : cudaAccessPropertyStreaming;
  cudaError_t e = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &v);
  if (e != cudaSuccess) (void)cudaGetLastError();
#endif
}

static inline void reset_persist(cudaStream_t stream, size_t prev_limit) {
#if CUDART_VERSION >= 11000
  if (l2_info().ok) {
    set_persist_window(stream, nullptr, 0);
    cudaError_t e = cudaCtxResetPersistingL2Cache();
    if (e != cudaSuccess) (void)cudaGetLastError();
  }
  restore_persist_limit(prev_limit);
#endif
}

// Plan item 4: producer/consumer chunk fusion -- each conv chunk is followed
// immediately by its GroupNorm partial pass on the SAME stream, with the
// chunk's output held in L2 by a persisting access-policy window.
static void conv_then_moments(const float* in, const float* wptr, float* y,
                              int N, int H, int W, int C, int G,
                              float* part, int numChunks, int pixPerChunk,
                              const torch::TensorOptions& opts, cudaStream_t stream) {
  const int HW = H * W;
  const size_t planeBytes = (size_t)HW * (size_t)C * sizeof(float);
  const size_t bytes = (size_t)N * planeBytes;   // 4a

  const L2Info& info = l2_info();
  size_t budget = info.ok ? ((size_t)info.l2_size / 2) : 0;

  // ---- gate (plan item 4b / item 8) -------------------------------------
  bool eligible = true;
#if FORCE_SINGLE_SHOT
  eligible = false;
#else
  if (!info.ok || N == 1 || bytes <= budget) eligible = false;
#endif

  if (!eligible) {   // exactly the original single-shot path
    conv3x3_nhwc_range(in, wptr, y, N, H, W, C, C, 3, 3, opts, stream);
    moments_partial_range(y, 0, N, HW, C, G, part, numChunks, pixPerChunk, stream);
    return;
  }

  // ---- chunk size (plan item 4c): L2 budget, raised until a chunk fills at
  // least one full wave of CTAs, then balanced so the last chunk is not a stub
  int nsub = (int)(budget / planeBytes);
  if (nsub < 1) nsub = 1;
  if (nsub > N) nsub = N;
  const long ktiles = (long)((C + 127) / 128);
  while (nsub < N) {
    long ctas = (long)(((size_t)nsub * (size_t)HW + 127) / 128) * ktiles;
    if (ctas >= 170) break;
    ++nsub;
  }
  {
    const int ncnk = (N + nsub - 1) / nsub;
    nsub = (N + ncnk - 1) / ncnk;
  }

  // ---- per-shape-key decision cache (plan item 8): if the chunked path is
  // not faster for this shape key, fall back to the single-shot path.
  struct Key { int N, H, W, C;
    bool operator<(const Key& o) const {
      if (N != o.N) return N < o.N; if (H != o.H) return H < o.H;
      if (W != o.W) return W < o.W; return C < o.C; } };
  static std::map<Key, int> s_choice;      // 0 = single-shot, 1 = chunked
  static std::mutex s_mtx;
  Key key{N, H, W, C};
  int choice = -1;
  { std::lock_guard<std::mutex> lk(s_mtx);
    auto it = s_choice.find(key);
    if (it != s_choice.end()) choice = it->second; }

  if (choice < 0) {
    float t_single = 0.f, t_chunk = 0.f;
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    // warm-up (first CUTLASS launch of this geometry pays module overhead)
    conv3x3_nhwc_range(in, wptr, y, N, H, W, C, C, 3, 3, opts, stream);
    moments_partial_range(y, 0, N, HW, C, G, part, numChunks, pixPerChunk, stream);
    cudaEventRecord(e0, stream);
    conv3x3_nhwc_range(in, wptr, y, N, H, W, C, C, 3, 3, opts, stream);
    moments_partial_range(y, 0, N, HW, C, G, part, numChunks, pixPerChunk, stream);
    cudaEventRecord(e1, stream);
    for (int n0 = 0; n0 < N; n0 += nsub) {
      const int nN = min(nsub, N - n0);
      const size_t off = (size_t)n0 * (size_t)HW * (size_t)C;
      set_persist_window(stream, y + off, (size_t)nN * planeBytes);
      conv3x3_nhwc_range(in + off, wptr, y + off, nN, H, W, C, C, 3, 3, opts, stream);
      moments_partial_range(y, n0, nN, HW, C, G, part, numChunks, pixPerChunk, stream);
    }
    set_persist_window(stream, nullptr, 0);
    cudaEventRecord(e2, stream);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&t_single, e0, e1);
    cudaEventElapsedTime(&t_chunk, e1, e2);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    choice = (t_chunk < t_single) ? 1 : 0;
    { std::lock_guard<std::mutex> lk(s_mtx); s_choice[key] = choice; }
    return;   // y / part are already fully written (both variants agree bitwise)
  }

  if (choice == 0) {
    conv3x3_nhwc_range(in, wptr, y, N, H, W, C, C, 3, 3, opts, stream);
    moments_partial_range(y, 0, N, HW, C, G, part, numChunks, pixPerChunk, stream);
    return;
  }

  // ---- fused producer/consumer chunk loop (plan item 4c + item 5) --------
  for (int n0 = 0; n0 < N; n0 += nsub) {
    const int nN = min(nsub, N - n0);
    const size_t off = (size_t)n0 * (size_t)HW * (size_t)C;
    set_persist_window(stream, y + off, (size_t)nN * planeBytes);
    conv3x3_nhwc_range(in + off, wptr, y + off, nN, H, W, C, C, 3, 3, opts, stream);
    moments_partial_range(y, n0, nN, HW, C, G, part, numChunks, pixPerChunk, stream);
  }
  set_persist_window(stream, nullptr, 0);
}

torch::Tensor fused_resblock(torch::Tensor x, torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight, torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight, torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "x must be cuda float32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  auto xc = x.is_contiguous() ? x : x.contiguous();
  const int N = (int)xc.size(0), C = (int)xc.size(1);
  const int H = (int)xc.size(2), W = (int)xc.size(3);
  const int HW = H * W;
  const int G = 32;
  TORCH_CHECK(C % G == 0 && (C / G) % 4 == 0 && C % TT == 0 && (C / 4) <= 256 &&
                  256 % (C / 4) == 0,
              "unsupported channel count");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = xc.options();

  // Save the caller's persisting-L2 reservation, raise it for the duration of
  // this call only, and restore it before returning (see reset_persist).
  const size_t prev_persist_limit = save_persist_limit();
  raise_persist_limit();

  auto w1 = to_nhwc_weight(conv1_weight.is_contiguous() ? conv1_weight
                                                        : conv1_weight.contiguous(), stream);
  auto w2 = to_nhwc_weight(conv2_weight.is_contiguous() ? conv2_weight
                                                        : conv2_weight.contiguous(), stream);
  auto g1 = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
  auto b1 = norm1_bias.is_contiguous() ? norm1_bias : norm1_bias.contiguous();
  auto g2 = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
  auto b2 = norm2_bias.is_contiguous() ? norm2_bias : norm2_bias.contiguous();

  auto xh = torch::empty({N, H, W, C}, opts);
  const int vec = (HW % 4 == 0) ? 1 : 0;
  dim3 tgrid((HW + TT - 1) / TT, C / TT, N);
  nchw2nhwc_kernel<<<tgrid, 256, 0, stream>>>(xc.data_ptr<float>(), xh.data_ptr<float>(),
                                              HW, C, vec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // moment-pass geometry: identical to the base (depends only on N and HW);
  // plan item 2 -- `part` allocated for the full N up front.
  int nc = 1024 / max(N, 1);
  if (nc < 1) nc = 1;
  int maxc = (HW + 255) / 256;
  if (maxc < 1) maxc = 1;
  if (nc > maxc) nc = maxc;
  const int numChunks = nc;
  const int pixPerChunk = (HW + numChunks - 1) / numChunks;
  auto part = torch::empty({(long)N * numChunks * G * 2}, opts);

  auto mean = torch::empty({N * G}, opts);
  auto rstd = torch::empty({N * G}, opts);

  // conv1 -> moments (plan item 6)
  auto y1 = torch::empty({N, H, W, C}, opts);
  conv_then_moments(xh.data_ptr<float>(), w1.data_ptr<float>(), y1.data_ptr<float>(),
                    N, H, W, C, G, part.data_ptr<float>(), numChunks, pixPerChunk,
                    opts, stream);
  moments_finalize(part.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                   N * G, numChunks, G, (long)HW * (C / G), (float)eps, stream);

  auto z1 = xh;  // reuse buffer (staged NHWC copy of x is dead by now)
  {
    int blocks = (int)(((size_t)HW * (C / 4) + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    dim3 gr(blocks, N);
    gn_silu_nhwc_kernel<<<gr, 256, 0, stream>>>(
        y1.data_ptr<float>(), z1.data_ptr<float>(), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), g1.data_ptr<float>(), b1.data_ptr<float>(), HW, C, G, C / G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // conv2 -> moments (plan item 6)
  auto y2 = y1;  // reuse buffer (conv1 output is dead by now)
  conv_then_moments(z1.data_ptr<float>(), w2.data_ptr<float>(), y2.data_ptr<float>(),
                    N, H, W, C, G, part.data_ptr<float>(), numChunks, pixPerChunk,
                    opts, stream);
  moments_finalize(part.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                   N * G, numChunks, G, (long)HW * (C / G), (float)eps, stream);

  auto out = torch::empty({N, C, H, W}, opts);
  gn_silu_add_nhwc2nchw_kernel<<<tgrid, 256, 0, stream>>>(
      y2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), g2.data_ptr<float>(),
      b2.data_ptr<float>(), HW, C, G, C / G, vec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  reset_persist(stream, prev_persist_limit);
  return out;
}
'''

cpp_src = r'''
torch::Tensor fused_resblock(torch::Tensor x, torch::Tensor conv1_weight,
                             torch::Tensor norm1_weight, torch::Tensor norm1_bias,
                             torch::Tensor conv2_weight, torch::Tensor norm2_weight,
                             torch::Tensor norm2_bias, double eps);
'''

_ext = load_inline(
    name="vae_resblock_fused_l2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_resblock"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo",
        "-gencode=arch=compute_120,code=sm_120",
        "-I" + CUTLASS_INC,
    ],
)


class ModelNew(nn.Module):
    # =====================================================================
    # Full forward rewrite; the convolution is our own CUTLASS TF32 tiled
    # implicit GEMM (no cuDNN / at::conv2d anywhere).  This module is
    # stateless (the reference is stateless too): all weights arrive as
    # forward() arguments, so there is nothing to hold and no parameter
    # parity to break.
    # =====================================================================
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        return self.ext.fused_resblock(x, conv1_weight, norm1_weight, norm1_bias,
                                       conv2_weight, norm2_weight, norm2_bias, float(eps))
