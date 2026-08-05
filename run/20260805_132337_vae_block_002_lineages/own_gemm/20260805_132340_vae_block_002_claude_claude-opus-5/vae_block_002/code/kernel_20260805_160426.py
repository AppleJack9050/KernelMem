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

#include <map>
#include <mutex>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// ---------------------------------------------------------------------------
// PLAN 1: templated CUTLASS TF32 implicit-GEMM 3x3 conv (NHWC, KRSC filters).
// Everything but TB tile / warp tile / stages / swizzle is identical to the
// base config: float NHWC A/B/C, OpClassTensorOp, Sm80, MMA 16x8x8,
// LinearCombination<float,4>, kOptimized, alpha=1 beta=0, fp32 accumulate.
// ---------------------------------------------------------------------------
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int STAGES, int SW>
struct ConvCfg {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      float, cutlass::layout::TensorNHWC,
      float, cutlass::layout::TensorNHWC,
      float, cutlass::layout::TensorNHWC,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, TBK>,
      cutlass::gemm::GemmShape<WM, WN, WK>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SW>,
      STAGES,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

  static size_t smem_bytes() { return sizeof(typename Kernel::SharedStorage); }

  static typename Op::Arguments make_args(const torch::Tensor& x, const torch::Tensor& w,
                                          torch::Tensor& out) {
    int N = (int)x.size(0), H = (int)x.size(1), W = (int)x.size(2), C = (int)x.size(3);
    int K = (int)w.size(0), R = (int)w.size(1), S = (int)w.size(2);
    cutlass::conv::Conv2dProblemSize ps(N, H, W, C, K, R, S, H, W,
                                        1, 1, 1, 1, 1, 1,
                                        cutlass::conv::Mode::kCrossCorrelation, 1, 1);
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> ra(
        (float*)x.data_ptr<float>(), cutlass::layout::TensorNHWC::packed({N, H, W, C}));
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> rb(
        (float*)w.data_ptr<float>(), cutlass::layout::TensorNHWC::packed({K, R, S, C}));
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> rd(
        out.data_ptr<float>(), cutlass::layout::TensorNHWC::packed({N, H, W, K}));
    return typename Op::Arguments{ps, ra, rb, rd, rd, {1.0f, 0.0f}};
  }

  // PLAN 3: guarded run -- returns false on can_implement / init / launch failure.
  static bool run(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                  cudaStream_t stream) {
    auto args = make_args(x, w, out);
    Op op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = op.get_workspace_size(args);
    void* wsptr = nullptr;
    torch::Tensor wsbuf;
    if (ws > 0) {
      wsbuf = torch::empty({(long)ws}, x.options().dtype(torch::kUInt8));
      wsptr = wsbuf.data_ptr();
    }
    if (op.initialize(args, wsptr, stream) != cutlass::Status::kSuccess) return false;
    if (op(stream) != cutlass::Status::kSuccess) return false;
    return true;
  }

  static bool can(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
    auto args = make_args(x, w, out);
    Op op;
    return op.can_implement(args) == cutlass::Status::kSuccess;
  }
};

struct CandDesc {
  int tbM, tbN, tbK, stages, swizzle;
  size_t smem;
  bool (*run)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&, cudaStream_t);
  bool (*can)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&);
};

template <class Cfg>
static CandDesc make_desc(int tbM, int tbN, int tbK, int stages, int sw) {
  CandDesc d;
  d.tbM = tbM; d.tbN = tbN; d.tbK = tbK; d.stages = stages; d.swizzle = sw;
  d.smem = Cfg::smem_bytes();
  d.run = &Cfg::run;
  d.can = &Cfg::can;
  return d;
}

// PLAN 2 + 6: fixed compile-time candidate list.  Index 0 is the permanent
// fallback (today's config).  Entries [0..kNumTiles) are the tile sweep at the
// default swizzle 4; the trailing entries are swizzle {1,8} variants used only
// for the second (winning-tile-only) pass.
static const int kNumTiles = 8;

static const std::vector<CandDesc>& candidates() {
  static std::vector<CandDesc> v = [] {
    std::vector<CandDesc> t;
    // ---- tile sweep, swizzle 4 -------------------------------------------
    t.push_back(make_desc<ConvCfg<128, 128, 16, 64, 64, 16, 3, 4>>(128, 128, 16, 3, 4)); // C0
    t.push_back(make_desc<ConvCfg<256, 128, 16, 64, 64, 16, 3, 4>>(256, 128, 16, 3, 4)); // C1
    t.push_back(make_desc<ConvCfg<128, 256, 16, 64, 64, 16, 3, 4>>(128, 256, 16, 3, 4)); // C2
    t.push_back(make_desc<ConvCfg<128, 128, 16, 32, 64, 16, 4, 4>>(128, 128, 16, 4, 4)); // C3
    t.push_back(make_desc<ConvCfg<128, 128, 32, 64, 64, 32, 3, 4>>(128, 128, 32, 3, 4)); // C4
    t.push_back(make_desc<ConvCfg<128,  64, 16, 64, 32, 16, 4, 4>>(128,  64, 16, 4, 4)); // C5
    t.push_back(make_desc<ConvCfg< 64, 128, 16, 32, 64, 16, 4, 4>>( 64, 128, 16, 4, 4)); // C6
    t.push_back(make_desc<ConvCfg<128, 128, 16, 64, 32, 16, 3, 4>>(128, 128, 16, 3, 4)); // C7
    // ---- swizzle variants (same tiles), used in pass 2 only ---------------
    t.push_back(make_desc<ConvCfg<128, 128, 16, 64, 64, 16, 3, 1>>(128, 128, 16, 3, 1));
    t.push_back(make_desc<ConvCfg<256, 128, 16, 64, 64, 16, 3, 1>>(256, 128, 16, 3, 1));
    t.push_back(make_desc<ConvCfg<128, 256, 16, 64, 64, 16, 3, 1>>(128, 256, 16, 3, 1));
    t.push_back(make_desc<ConvCfg<128, 128, 16, 32, 64, 16, 4, 1>>(128, 128, 16, 4, 1));
    t.push_back(make_desc<ConvCfg<128, 128, 32, 64, 64, 32, 3, 1>>(128, 128, 32, 3, 1));
    t.push_back(make_desc<ConvCfg<128,  64, 16, 64, 32, 16, 4, 1>>(128,  64, 16, 4, 1));
    t.push_back(make_desc<ConvCfg< 64, 128, 16, 32, 64, 16, 4, 1>>( 64, 128, 16, 4, 1));
    t.push_back(make_desc<ConvCfg<128, 128, 16, 64, 32, 16, 3, 1>>(128, 128, 16, 3, 1));
    t.push_back(make_desc<ConvCfg<128, 128, 16, 64, 64, 16, 3, 8>>(128, 128, 16, 3, 8));
    t.push_back(make_desc<ConvCfg<256, 128, 16, 64, 64, 16, 3, 8>>(256, 128, 16, 3, 8));
    t.push_back(make_desc<ConvCfg<128, 256, 16, 64, 64, 16, 3, 8>>(128, 256, 16, 3, 8));
    t.push_back(make_desc<ConvCfg<128, 128, 16, 32, 64, 16, 4, 8>>(128, 128, 16, 4, 8));
    t.push_back(make_desc<ConvCfg<128, 128, 32, 64, 64, 32, 3, 8>>(128, 128, 32, 3, 8));
    t.push_back(make_desc<ConvCfg<128,  64, 16, 64, 32, 16, 4, 8>>(128,  64, 16, 4, 8));
    t.push_back(make_desc<ConvCfg< 64, 128, 16, 32, 64, 16, 4, 8>>( 64, 128, 16, 4, 8));
    t.push_back(make_desc<ConvCfg<128, 128, 16, 64, 32, 16, 3, 8>>(128, 128, 16, 3, 8));
    return t;
  }();
  return v;
}

// PLAN 2: device shared-memory opt-in budget, queried once.
static size_t max_smem_optin() {
  static size_t s = [] {
    int dev = 0, v = 48 * 1024;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (size_t)v;
  }();
  return s;
}

static int num_sms() {
  static int s = [] {
    int dev = 0, v = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
    return v;
  }();
  return s;
}

struct ConvKey {
  int N, H, W, C, K;
  bool operator<(const ConvKey& o) const {
    if (N != o.N) return N < o.N;
    if (H != o.H) return H < o.H;
    if (W != o.W) return W < o.W;
    if (C != o.C) return C < o.C;
    return K < o.K;
  }
};

static std::map<ConvKey, int>& tune_cache() {
  static std::map<ConvKey, int> m;
  return m;
}
static std::mutex& tune_mutex() {
  static std::mutex m;
  return m;
}

// PLAN 4: shape dispatch / anti-regression CTA-count filter.
static bool cta_ok(const CandDesc& c, const CandDesc& c0, long M, int Kc) {
  auto ctas = [&](const CandDesc& d) {
    return (long)((M + d.tbM - 1) / d.tbM) * (long)((Kc + d.tbN - 1) / d.tbN);
  };
  long need = (long)num_sms();
  if (ctas(c0) < need) return true;      // C0 itself below one wave -> no filter
  return ctas(c) >= need;
}

// PLAN 5: time one candidate (1 warmup + 3 timed reps) on the real tensors.
static bool time_candidate(const CandDesc& c, const torch::Tensor& x, const torch::Tensor& w,
                           torch::Tensor& out, cudaStream_t stream, float* ms_out) {
  if (!c.run(x, w, out, stream)) return false;
  if (cudaStreamSynchronize(stream) != cudaSuccess) { cudaGetLastError(); return false; }
  cudaEvent_t e0, e1;
  if (cudaEventCreate(&e0) != cudaSuccess) { cudaGetLastError(); return false; }
  if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); cudaGetLastError(); return false; }
  bool ok = true;
  cudaEventRecord(e0, stream);
  for (int i = 0; i < 3 && ok; ++i) ok = c.run(x, w, out, stream);
  cudaEventRecord(e1, stream);
  if (cudaEventSynchronize(e1) != cudaSuccess) { cudaGetLastError(); ok = false; }
  float ms = 0.f;
  if (ok && cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess) { cudaGetLastError(); ok = false; }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  if (cudaGetLastError() != cudaSuccess) ok = false;
  if (!ok) return false;
  *ms_out = ms;
  return true;
}

static int autotune(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                    cudaStream_t stream) {
  const auto& cand = candidates();
  const long M = (long)x.size(0) * x.size(1) * x.size(2);
  const int Kc = (int)w.size(0);
  const size_t smem_cap = max_smem_optin();

  int best = 0;
  float best_ms = 3.4e38f, c0_ms = 3.4e38f;
  // pass 1: tile sweep at swizzle 4 (indices [0, kNumTiles)).
  for (int i = 0; i < kNumTiles && i < (int)cand.size(); ++i) {
    const CandDesc& c = cand[i];
    if (i != 0) {
      if (c.smem > smem_cap) continue;                 // PLAN 2 smem budget
      if (!cta_ok(c, cand[0], M, Kc)) continue;        // PLAN 4 CTA filter
    }
    if (!c.can(x, w, out)) continue;                   // PLAN 3
    float ms = 0.f;
    if (!time_candidate(c, x, w, out, stream, &ms)) continue;
    if (i == 0) c0_ms = ms;
    if (ms < best_ms) { best_ms = ms; best = i; }
  }
  if (best_ms > 3.0e38f) return 0;                     // nothing timed -> C0

  // PLAN 6: cheap second pass, swizzle {1,8} for the winning tile only.
  int tile = best;
  for (int s = 1; s <= 2; ++s) {
    int idx = tile + s * kNumTiles;
    if (idx >= (int)cand.size()) break;
    const CandDesc& c = cand[idx];
    if (c.smem > smem_cap) continue;
    if (!cta_ok(c, cand[0], M, Kc)) continue;
    if (!c.can(x, w, out)) continue;
    float ms = 0.f;
    if (!time_candidate(c, x, w, out, stream, &ms)) continue;
    if (ms < best_ms) { best_ms = ms; best = idx; }
  }
  // PLAN 8: anti-regression -- only keep a non-default config if it beats the
  // fallback C0 by more than measurement noise; otherwise cache C0.
  if (best != 0 && !(best_ms < 0.98f * c0_ms)) best = 0;
  return best;
}

static void conv3x3_nhwc_into(const torch::Tensor& x, const torch::Tensor& w,
                              torch::Tensor& out, cudaStream_t stream) {
  const auto& cand = candidates();
  ConvKey key{(int)x.size(0), (int)x.size(1), (int)x.size(2), (int)x.size(3), (int)w.size(0)};
  int sel;
  {
    std::lock_guard<std::mutex> lk(tune_mutex());
    auto& cache = tune_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
      int b = 0;
      b = autotune(x, w, out, stream);
      cache.emplace(key, b);
      sel = b;
    } else {
      sel = it->second;
    }
  }
  if (sel < 0 || sel >= (int)cand.size()) sel = 0;
  if (!cand[sel].run(x, w, out, stream)) {
    TORCH_CHECK(cand[0].run(x, w, out, stream), "cutlass conv: unsupported problem");
  }
}

// ---------------------------------------------------------------------------
// PLAN 7: every other kernel below is byte-identical to the base version.
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
__global__ __launch_bounds__(256) void gn_partial_kernel(
    const float* __restrict__ y, float* __restrict__ part,
    int HW, int C, int G, int Cg, int pixPerChunk, int numChunks) {
  const int tpp = C >> 2;                 // float4 lanes per pixel
  const int ppp = 256 / tpp;              // pixels handled per pass
  const int c4  = threadIdx.x % tpp;
  const int pr  = threadIdx.x / tpp;
  const int n   = blockIdx.y;
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

static void group_moments(const torch::Tensor& y, int N, int HW, int C, int G,
                          float eps, torch::Tensor& mean, torch::Tensor& rstd,
                          cudaStream_t stream) {
  const int Cg = C / G;
  int nc = 1024 / max(N, 1);
  if (nc < 1) nc = 1;
  int maxc = (HW + 255) / 256;
  if (maxc < 1) maxc = 1;
  if (nc > maxc) nc = maxc;
  const int numChunks = nc;
  const int pixPerChunk = (HW + numChunks - 1) / numChunks;
  auto part = torch::empty({(long)N * numChunks * G * 2}, y.options());
  dim3 grid(numChunks, N);
  gn_partial_kernel<<<grid, 256, 0, stream>>>(
      y.data_ptr<float>(), part.data_ptr<float>(), HW, C, G, Cg, pixPerChunk, numChunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gn_finalize_kernel<<<N * G, 128, 0, stream>>>(
      part.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      numChunks, G, (long)HW * Cg, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

  auto y1 = torch::empty({N, H, W, C}, opts);
  conv3x3_nhwc_into(xh, w1, y1, stream);

  auto mean = torch::empty({N * G}, opts);
  auto rstd = torch::empty({N * G}, opts);
  group_moments(y1, N, HW, C, G, (float)eps, mean, rstd, stream);

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

  auto y2 = y1;  // reuse buffer (conv1 output is dead by now)
  conv3x3_nhwc_into(z1, w2, y2, stream);

  group_moments(y2, N, HW, C, G, (float)eps, mean, rstd, stream);

  auto out = torch::empty({N, C, H, W}, opts);
  gn_silu_add_nhwc2nchw_kernel<<<tgrid, 256, 0, stream>>>(
      y2.data_ptr<float>(), xc.data_ptr<float>(), out.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), g2.data_ptr<float>(),
      b2.data_ptr<float>(), HW, C, G, C / G, vec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    name="vae_resblock_fused_tuned",
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
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        return self.ext.fused_resblock(x, conv1_weight, norm1_weight, norm1_bias,
                                       conv2_weight, norm2_weight, norm2_bias, float(eps))
