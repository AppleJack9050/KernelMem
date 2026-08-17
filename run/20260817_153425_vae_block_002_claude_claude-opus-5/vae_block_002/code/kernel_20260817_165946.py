import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

__device__ __forceinline__ void warp_red2(float &a, float &b) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    a += __shfl_down_sync(0xffffffff, a, off);
    b += __shfl_down_sync(0xffffffff, b, off);
  }
}
__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

/* ---------------- generic NCHW fallback path ---------------- */
__global__ void gn_stats_kernel(const float* __restrict__ x, float* __restrict__ psum,
                                float* __restrict__ psum2, long epg, int nchunk) {
  const int bg = blockIdx.x, chunk = blockIdx.y;
  const long base = (long)bg * epg;
  const long begin = (long)(chunk) * epg / nchunk;
  const long end = (long)(chunk + 1) * epg / nchunk;
  float s = 0.f, s2 = 0.f;
  const long n = end - begin, p0 = base + begin;
  if (((p0 & 3L) == 0) && ((n & 3L) == 0)) {
    const float4* xv = reinterpret_cast<const float4*>(x + p0);
    long nv = n >> 2;
    for (long i = threadIdx.x; i < nv; i += blockDim.x) {
      float4 v = xv[i];
      s += v.x + v.y + v.z + v.w;
      s2 += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
  } else {
    for (long i = threadIdx.x; i < n; i += blockDim.x) { float v = x[p0+i]; s += v; s2 += v*v; }
  }
  __shared__ float sa[32], sb[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  warp_red2(s, s2);
  if (lane == 0) { sa[wid] = s; sb[wid] = s2; }
  __syncthreads();
  int nw = blockDim.x >> 5;
  if (wid == 0) {
    s = (lane < nw) ? sa[lane] : 0.f; s2 = (lane < nw) ? sb[lane] : 0.f;
    warp_red2(s, s2);
    if (lane == 0) { psum[(long)bg*nchunk+chunk] = s; psum2[(long)bg*nchunk+chunk] = s2; }
  }
}

__global__ void gn_finalize_kernel(const float* __restrict__ psum, const float* __restrict__ psum2,
                                   float* __restrict__ mean, float* __restrict__ rstd,
                                   int nchunk, float inv_n, float eps) {
  const int bg = blockIdx.x;
  float s = 0.f, s2 = 0.f;
  for (int i = threadIdx.x; i < nchunk; i += blockDim.x) {
    s += psum[(long)bg*nchunk+i]; s2 += psum2[(long)bg*nchunk+i];
  }
  __shared__ float sa[32], sb[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  warp_red2(s, s2);
  if (lane == 0) { sa[wid] = s; sb[wid] = s2; }
  __syncthreads();
  int nw = blockDim.x >> 5;
  if (wid == 0) {
    s = (lane < nw) ? sa[lane] : 0.f; s2 = (lane < nw) ? sb[lane] : 0.f;
    warp_red2(s, s2);
    if (lane == 0) {
      float m = s * inv_n; float v = s2 * inv_n - m * m; v = v < 0.f ? 0.f : v;
      mean[bg] = m; rstd[bg] = rsqrtf(v + eps);
    }
  }
}

template <bool HAS_RES>
__global__ void gn_apply_kernel(const float* __restrict__ x, const float* __restrict__ mean,
                                const float* __restrict__ rstd, const float* __restrict__ gamma,
                                const float* __restrict__ beta, const float* __restrict__ res,
                                float* __restrict__ out, long N, int C, int G, int vec4) {
  const int row = blockIdx.x;
  const int c = row % C, b = row / C;
  const int g = (int)((long)c * G / C);
  const float sc = rstd[b*G+g] * gamma[c];
  const float sh = beta[c] - mean[b*G+g] * sc;
  const long base = (long)row * N;
  if (vec4) {
    const float4* xv = reinterpret_cast<const float4*>(x + base);
    const float4* rv = reinterpret_cast<const float4*>(res + base);
    float4* ov = reinterpret_cast<float4*>(out + base);
    long nv = N >> 2;
    for (long i = threadIdx.x; i < nv; i += blockDim.x) {
      float4 v = xv[i]; float4 o;
      o.x = silu(v.x*sc+sh); o.y = silu(v.y*sc+sh);
      o.z = silu(v.z*sc+sh); o.w = silu(v.w*sc+sh);
      if (HAS_RES) { float4 r = rv[i]; o.x+=r.x; o.y+=r.y; o.z+=r.z; o.w+=r.w; }
      ov[i] = o;
    }
  } else {
    for (long i = threadIdx.x; i < N; i += blockDim.x) {
      float o = silu(x[base+i]*sc+sh);
      if (HAS_RES) o += res[base+i];
      out[base+i] = o;
    }
  }
}

torch::Tensor gn_silu(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                      double eps, int64_t groups, c10::optional<torch::Tensor> res_opt) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "float cuda expected");
  TORCH_CHECK(x.is_contiguous(), "contiguous expected");
  const int B = x.size(0), C = x.size(1);
  const long N = x.size(2) * (long)x.size(3);
  const int G = (int)groups;
  const long epg = (long)(C / G) * N;
  auto opts = x.options();
  auto out = torch::empty_like(x);
  int nchunk = (int)std::min<long>(128L, std::max<long>(1L, (epg + 16383) / 16384));
  auto psum = torch::empty({(long)B*G*nchunk}, opts);
  auto psum2 = torch::empty({(long)B*G*nchunk}, opts);
  auto mean = torch::empty({(long)B*G}, opts);
  auto rstd = torch::empty({(long)B*G}, opts);
  auto stream = at::cuda::getCurrentCUDAStream();
  gn_stats_kernel<<<dim3(B*G, nchunk), 256, 0, stream>>>(x.data_ptr<float>(),
      psum.data_ptr<float>(), psum2.data_ptr<float>(), epg, nchunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gn_finalize_kernel<<<B*G, 128, 0, stream>>>(psum.data_ptr<float>(), psum2.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), nchunk, (float)(1.0/(double)epg), (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int vec4 = (N % 4 == 0) ? 1 : 0;
  bool has_res = res_opt.has_value();
  const float* resp = has_res ? res_opt.value().data_ptr<float>() : x.data_ptr<float>();
  if (has_res) {
    gn_apply_kernel<true><<<B*C, 256, 0, stream>>>(x.data_ptr<float>(), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), resp,
        out.data_ptr<float>(), N, C, G, vec4);
  } else {
    gn_apply_kernel<false><<<B*C, 256, 0, stream>>>(x.data_ptr<float>(), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), resp,
        out.data_ptr<float>(), N, C, G, vec4);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

/* ---------------- NHWC specialised path (C=256, G=32) ---------------- */
#define CC 256
#define GG 32
#define CPG 8

// grid (nchunk, B), block 256 threads == one per channel
__global__ void gn_stats_nhwc(const float* __restrict__ x, float* __restrict__ psum,
                              float* __restrict__ psum2, long N, int nchunk) {
  const int chunk = blockIdx.x, b = blockIdx.y;
  const long begin = (long)(chunk) * N / nchunk;
  const long end = (long)(chunk + 1) * N / nchunk;
  const int c = threadIdx.x;
  const float* p = x + (long)b * N * CC + begin * CC + c;
  float s = 0.f, s2 = 0.f;
  for (long i = 0; i < end - begin; ++i) {
    float v = p[i * CC];
    s += v; s2 += v * v;
  }
  // reduce within groups of CPG=8 consecutive lanes
  #pragma unroll
  for (int off = CPG >> 1; off > 0; off >>= 1) {
    s += __shfl_down_sync(0xffffffff, s, off, CPG);
    s2 += __shfl_down_sync(0xffffffff, s2, off, CPG);
  }
  if ((c & (CPG - 1)) == 0) {
    int g = c / CPG;
    long o = ((long)b * nchunk + chunk) * GG + g;
    psum[o] = s; psum2[o] = s2;
  }
}

// grid B*GG
__global__ void gn_finalize_nhwc(const float* __restrict__ psum, const float* __restrict__ psum2,
                                 float* __restrict__ mean, float* __restrict__ rstd,
                                 int nchunk, float inv_n, float eps) {
  const int bg = blockIdx.x;
  const int b = bg / GG, g = bg - b * GG;
  float s = 0.f, s2 = 0.f;
  for (int i = threadIdx.x; i < nchunk; i += blockDim.x) {
    long o = ((long)b * nchunk + i) * GG + g;
    s += psum[o]; s2 += psum2[o];
  }
  __shared__ float sa[32], sb[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  warp_red2(s, s2);
  if (lane == 0) { sa[wid] = s; sb[wid] = s2; }
  __syncthreads();
  int nw = blockDim.x >> 5;
  if (wid == 0) {
    s = (lane < nw) ? sa[lane] : 0.f; s2 = (lane < nw) ? sb[lane] : 0.f;
    warp_red2(s, s2);
    if (lane == 0) {
      float m = s * inv_n; float v = s2 * inv_n - m * m; v = v < 0.f ? 0.f : v;
      mean[bg] = m; rstd[bg] = rsqrtf(v + eps);
    }
  }
}

// NHWC in -> NHWC out, normalize + affine + silu
__global__ void gn_apply_nhwc(const float* __restrict__ x, const float* __restrict__ mean,
                              const float* __restrict__ rstd, const float* __restrict__ gamma,
                              const float* __restrict__ beta, float* __restrict__ out, long N) {
  const int b = blockIdx.y;
  const long total = N * (CC / 4);   // number of float4 for this batch
  const float4* xv = reinterpret_cast<const float4*>(x + (long)b * N * CC);
  float4* ov = reinterpret_cast<float4*>(out + (long)b * N * CC);
  const float4* gv = reinterpret_cast<const float4*>(gamma);
  const float4* bv = reinterpret_cast<const float4*>(beta);
  const long stride = (long)gridDim.x * blockDim.x;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    int c4 = (int)(i & (CC / 4 - 1));
    int g = c4 / (CPG / 4);
    float m = mean[b * GG + g], r = rstd[b * GG + g];
    float4 gm = gv[c4], bt = bv[c4], v = xv[i], o;
    o.x = silu((v.x - m) * r * gm.x + bt.x);
    o.y = silu((v.y - m) * r * gm.y + bt.y);
    o.z = silu((v.z - m) * r * gm.z + bt.z);
    o.w = silu((v.w - m) * r * gm.w + bt.w);
    ov[i] = o;
  }
}

// NHWC in + NCHW residual -> NCHW out, normalize + affine + silu + add
#define TP 64
#define TC 64
__global__ void gn_apply_nhwc2nchw(const float* __restrict__ x, const float* __restrict__ mean,
                                   const float* __restrict__ rstd, const float* __restrict__ gamma,
                                   const float* __restrict__ beta, const float* __restrict__ res,
                                   float* __restrict__ out, long N) {
  __shared__ float tile[TC][TP + 1];
  __shared__ float ssc[TC], ssh[TC];
  const long p0 = (long)blockIdx.x * TP;
  const int c0 = blockIdx.y * TC;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;   // 256 threads

  if (tid < TC) {
    int c = c0 + tid;
    int g = c / CPG;
    float sc = rstd[b * GG + g] * gamma[c];
    ssc[tid] = sc;
    ssh[tid] = beta[c] - mean[b * GG + g] * sc;
  }
  __syncthreads();

  const long xbase = (long)b * N * CC;
  #pragma unroll
  for (int it = 0; it < (TP * TC) / 256; ++it) {
    int t = it * 256 + tid;
    int cl = t & (TC - 1);
    int pl = t >> 6;             // TC == 64
    long p = p0 + pl;
    float v = 0.f;
    if (p < N) {
      v = x[xbase + p * CC + c0 + cl];
      v = silu(v * ssc[cl] + ssh[cl]);
    }
    tile[cl][pl] = v;
  }
  __syncthreads();

  #pragma unroll
  for (int it = 0; it < (TP * TC) / 256; ++it) {
    int t = it * 256 + tid;
    int pl = t & (TP - 1);
    int cl = t >> 6;             // TP == 64
    long p = p0 + pl;
    if (p < N) {
      long idx = ((long)b * CC + c0 + cl) * N + p;
      out[idx] = tile[cl][pl] + res[idx];
    }
  }
}

// same tiled transpose kernel but residual is read in NHWC layout (same
// channel order as x), so it can be sourced from the graph's static NHWC
// input buffer without a layout conversion outside the graph.
__global__ void gn_apply_nhwc2nchw_res(const float* __restrict__ x, const float* __restrict__ mean,
                                   const float* __restrict__ rstd, const float* __restrict__ gamma,
                                   const float* __restrict__ beta, const float* __restrict__ res,
                                   float* __restrict__ out, long N) {
  __shared__ float tile[TC][TP + 1];
  __shared__ float ssc[TC], ssh[TC];
  const long p0 = (long)blockIdx.x * TP;
  const int c0 = blockIdx.y * TC;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;   // 256 threads

  if (tid < TC) {
    int c = c0 + tid;
    int g = c / CPG;
    float sc = rstd[b * GG + g] * gamma[c];
    ssc[tid] = sc;
    ssh[tid] = beta[c] - mean[b * GG + g] * sc;
  }
  __syncthreads();

  const long xbase = (long)b * N * CC;
  #pragma unroll
  for (int it = 0; it < (TP * TC) / 256; ++it) {
    int t = it * 256 + tid;
    int cl = t & (TC - 1);
    int pl = t >> 6;             // TC == 64
    long p = p0 + pl;
    float v = 0.f;
    if (p < N) {
      long off = xbase + p * CC + c0 + cl;
      v = x[off];
      v = silu(v * ssc[cl] + ssh[cl]);
      float r = res[off];
      v += r;
    }
    tile[cl][pl] = v;
  }
  __syncthreads();

  #pragma unroll
  for (int it = 0; it < (TP * TC) / 256; ++it) {
    int t = it * 256 + tid;
    int pl = t & (TP - 1);
    int cl = t >> 6;             // TP == 64
    long p = p0 + pl;
    if (p < N) {
      long idx = ((long)b * CC + c0 + cl) * N + p;
      out[idx] = tile[cl][pl];
    }
  }
}

std::vector<torch::Tensor> gn_stats_nhwc_run(torch::Tensor x, double eps) {
  const int B = x.size(0);
  const long N = x.size(2) * (long)x.size(3);
  auto opts = torch::TensorOptions().dtype(torch::kFloat).device(x.device());
  int nchunk = (int)std::min<long>(256L, std::max<long>(1L, N / 64));
  auto psum = torch::empty({(long)B * nchunk * GG}, opts);
  auto psum2 = torch::empty({(long)B * nchunk * GG}, opts);
  auto mean = torch::empty({(long)B * GG}, opts);
  auto rstd = torch::empty({(long)B * GG}, opts);
  auto stream = at::cuda::getCurrentCUDAStream();
  gn_stats_nhwc<<<dim3(nchunk, B), CC, 0, stream>>>(x.data_ptr<float>(), psum.data_ptr<float>(),
      psum2.data_ptr<float>(), N, nchunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gn_finalize_nhwc<<<B * GG, 256, 0, stream>>>(psum.data_ptr<float>(), psum2.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), nchunk,
      (float)(1.0 / (double)(CPG * N)), (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {mean, rstd};
}

// public export of the stats helper (mean/rstd only, no apply) so the graph
// capture path can compute GN2 stats inside the graph and defer the
// residual+SiLU apply kernel to outside the graph.
std::vector<torch::Tensor> gn_stats_pub(torch::Tensor x, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat);
  TORCH_CHECK(x.size(1) == CC, "channels must be 256");
  return gn_stats_nhwc_run(x, eps);
}

// x: channels-last (B,256,H,W) -> out channels-last
torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat);
  TORCH_CHECK(x.size(1) == CC, "channels must be 256");
  const int B = x.size(0);
  const long N = x.size(2) * (long)x.size(3);
  auto mr = gn_stats_nhwc_run(x, eps);
  auto out = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  long total = N * (CC / 4);
  int nb = (int)std::min<long>(2048L, (total + 255) / 256);
  gn_apply_nhwc<<<dim3(nb, B), 256, 0, stream>>>(x.data_ptr<float>(), mr[0].data_ptr<float>(),
      mr[1].data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
      out.data_ptr<float>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// x: channels-last (B,256,H,W); res: contiguous NCHW -> out contiguous NCHW
torch::Tensor gn_silu_res_nchw(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                               double eps, torch::Tensor res) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat);
  TORCH_CHECK(x.size(1) == CC, "channels must be 256");
  TORCH_CHECK(res.is_contiguous(), "residual must be contiguous NCHW");
  const int B = x.size(0), H = x.size(2), W = x.size(3);
  const long N = (long)H * W;
  auto mr = gn_stats_nhwc_run(x, eps);
  auto out = torch::empty({B, CC, H, W}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)((N + TP - 1) / TP), CC / TC, B);
  gn_apply_nhwc2nchw<<<grid, 256, 0, stream>>>(x.data_ptr<float>(), mr[0].data_ptr<float>(),
      mr[1].data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
      res.data_ptr<float>(), out.data_ptr<float>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Graph-replay path: mean/rstd already computed (inside the captured graph);
// residual (res) is in NHWC layout (same as x); output is a freshly
// allocated contiguous NCHW tensor. Runs outside the CUDA graph.
torch::Tensor gn_silu_res_nchw_from_nhwc(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd,
                               torch::Tensor gamma, torch::Tensor beta, torch::Tensor res,
                               int64_t H, int64_t W) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat);
  TORCH_CHECK(x.size(1) == CC, "channels must be 256");
  const int B = x.size(0);
  const long N = (long)H * W;
  auto out = torch::empty({B, CC, H, W}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)((N + TP - 1) / TP), CC / TC, B);
  gn_apply_nhwc2nchw_res<<<grid, 256, 0, stream>>>(x.data_ptr<float>(), mean.data_ptr<float>(),
      rstd.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
      res.data_ptr<float>(), out.data_ptr<float>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_CPP = (
    "torch::Tensor gn_silu(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, "
    "double eps, int64_t groups, c10::optional<torch::Tensor> res_opt);\n"
    "torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps);\n"
    "torch::Tensor gn_silu_res_nchw(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, "
    "double eps, torch::Tensor res);\n"
    "std::vector<torch::Tensor> gn_stats_pub(torch::Tensor x, double eps);\n"
    "torch::Tensor gn_silu_res_nchw_from_nhwc(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, "
    "torch::Tensor gamma, torch::Tensor beta, torch::Tensor res, int64_t H, int64_t W);"
)

_ext = load_inline(
    name="vae_resblock_ext_v3",
    cpp_sources=_CPP,
    cuda_sources=_SRC,
    functions=["gn_silu", "gn_silu_nhwc", "gn_silu_res_nchw", "gn_stats_pub",
               "gn_silu_res_nchw_from_nhwc"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo",
                       "-gencode=arch=compute_90,code=sm_90", "--use_fast_math"],
)


class ModelNew(nn.Module):
    # ---------------------------------------------------------------------
    # Base fusion (unchanged from prior round): both GroupNorms, both SiLUs,
    # the residual add, and the NHWC<->NCHW layout conversions are fused into
    # gn_silu_nhwc / gn_silu_res_nchw custom kernels; convs stay on cuDNN.
    #
    # THIS ROUND: CUDA-graph capture/replay for the C==256 path.
    # Bottleneck: dispatch/launch bubbles between the 11 per-forward host
    # launches (~19% of forward time is outside any kernel).
    # Method: capture conv1->GN1(stats+apply)->conv2->GN2-stats once per
    # (B,H,W,eps) into a CUDA graph over static channels_last buffers; per
    # call only copy_ the live inputs into those buffers and replay; the
    # final GN2 apply + residual + SiLU + NHWC->NCHW transpose runs OUTSIDE
    # the graph into a freshly allocated NCHW tensor (no copy-out needed).
    # ---------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._graphs = {}
        self._graph_ok = True

    def _build_graph_entry(self, B, H, W, eps, device, dtype):
        cl = torch.channels_last
        static_xc = torch.zeros(B, 256, H, W, device=device, dtype=dtype).to(memory_format=cl)
        static_w1 = torch.zeros(256, 256, 3, 3, device=device, dtype=dtype).to(memory_format=cl)
        static_w2 = torch.zeros(256, 256, 3, 3, device=device, dtype=dtype).to(memory_format=cl)
        static_n1w = torch.zeros(256, device=device, dtype=dtype)
        static_n1b = torch.zeros(256, device=device, dtype=dtype)

        def body():
            o1 = F.conv2d(static_xc, static_w1, bias=None, stride=1, padding=1)
            o1 = _ext.gn_silu_nhwc(o1, static_n1w, static_n1b, eps)
            o2 = F.conv2d(o1, static_w2, bias=None, stride=1, padding=1)
            m, r = _ext.gn_stats_pub(o2, eps)
            return o2, m, r

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            o2, m, r = body()

        return {
            "graph": g, "xc": static_xc, "w1": static_w1, "w2": static_w2,
            "n1w": static_n1w, "n1b": static_n1b, "o2": o2, "mean": m, "rstd": r,
        }

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        G = 32
        x = x if x.is_contiguous() else x.contiguous()
        eps_f = float(eps)

        if (self._graph_ok and x.is_cuda and x.size(1) == 256
                and x.dtype == torch.float32):
            B, C, H, W = x.shape
            key = (B, H, W, eps_f)
            try:
                if key not in self._graphs:
                    self._graphs[key] = self._build_graph_entry(B, H, W, eps_f, x.device, x.dtype)
                entry = self._graphs[key]
                with torch.no_grad():
                    entry["xc"].copy_(x)
                    entry["w1"].copy_(conv1_weight)
                    entry["w2"].copy_(conv2_weight)
                    entry["n1w"].copy_(norm1_weight)
                    entry["n1b"].copy_(norm1_bias)
                    entry["graph"].replay()
                    return _ext.gn_silu_res_nchw_from_nhwc(
                        entry["o2"], entry["mean"], entry["rstd"],
                        norm2_weight, norm2_bias, entry["xc"], H, W)
            except Exception:
                self._graph_ok = False

        if x.size(1) == 256:
            cl = torch.channels_last
            xc = x.contiguous(memory_format=cl)
            w1 = conv1_weight.contiguous(memory_format=cl)
            w2 = conv2_weight.contiguous(memory_format=cl)
            o = F.conv2d(xc, w1, bias=None, stride=1, padding=1)
            o = _ext.gn_silu_nhwc(o, norm1_weight, norm1_bias, eps_f)
            o = F.conv2d(o, w2, bias=None, stride=1, padding=1)
            return _ext.gn_silu_res_nchw(o, norm2_weight, norm2_bias, eps_f, x)
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = _ext.gn_silu(out, norm1_weight, norm1_bias, eps_f, G, None)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        return _ext.gn_silu(out, norm2_weight, norm2_bias, eps_f, G, x)
