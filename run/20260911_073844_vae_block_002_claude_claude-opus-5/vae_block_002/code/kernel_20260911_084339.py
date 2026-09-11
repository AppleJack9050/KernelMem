import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ void block_reduce2(float& s, float& sq, float* smem) {
    // smem must have 2 * 32 floats
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(FULL_MASK, s,  off);
        sq += __shfl_down_sync(FULL_MASK, sq, off);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) { smem[warp] = s; smem[32 + warp] = sq; }
    __syncthreads();
    if (warp == 0) {
        s  = (lane < nwarps) ? smem[lane] : 0.f;
        sq = (lane < nwarps) ? smem[32 + lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            s  += __shfl_down_sync(FULL_MASK, s,  off);
            sq += __shfl_down_sync(FULL_MASK, sq, off);
        }
    }
}

// ---- NCHW -> NHWC (channels_last) shared-memory tiled transpose ---------------
// One block covers a 64(pixels) x 32(channels) region, processed as two 32x32
// sub-tiles so block count is halved vs. a plain 32x32 tiling.
__global__ void nchw_to_nhwc_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int P, int C, int B) {
    __shared__ float tile[32][33];
    int n  = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        int p0 = blockIdx.x * 64 + t * 32;
        int c0 = blockIdx.y * 32;

        // load phase: consecutive tx -> consecutive pixel p (coalesced NCHW read)
        int p = p0 + tx;
        #pragma unroll
        for (int k = 0; k < 32; k += 8) {
            int c = c0 + ty + k;
            float v = 0.f;
            if (p < P && c < C) v = in[((long long)n * C + c) * P + p];
            tile[ty + k][tx] = v;
        }
        __syncthreads();

        // store phase: consecutive tx -> consecutive channel c (coalesced NHWC write)
        int c = c0 + tx;
        #pragma unroll
        for (int k = 0; k < 32; k += 8) {
            int pp = p0 + ty + k;
            if (pp < P && c < C) out[((long long)n * P + pp) * C + c] = tile[tx][ty + k];
        }
        __syncthreads();
    }
}

// ---- conv weight (K,C,3,3) NCHW -> channels_last (K,3,3,C), both weights in
//      a single launch (blockIdx.y selects w1 / w2) ----------------------------
__global__ void weights_nchw_to_nhwc_kernel(const float* __restrict__ w1,
                                             const float* __restrict__ w2,
                                             float* __restrict__ o1,
                                             float* __restrict__ o2,
                                             int C) {
    extern __shared__ float sh[];
    int k   = blockIdx.x;
    int tid = threadIdx.x;
    int n_items = C * 9;

    const float* w = (blockIdx.y == 0) ? w1 : w2;
    float* o       = (blockIdx.y == 0) ? o1 : o2;

    for (int i = tid; i < n_items; i += blockDim.x) {
        sh[i] = w[(long long)k * n_items + i];
    }
    __syncthreads();
    for (int j = tid; j < n_items; j += blockDim.x) {
        int c  = j % C;
        int rs = j / C;
        o[(long long)k * n_items + j] = sh[c * 9 + rs];
    }
}

// ---- stage 1: partial sums over an NHWC tensor, per (n, group) ----------------
__global__ void gn_partial_kernel(const float* __restrict__ in,
                                  float2* __restrict__ partial,
                                  int P, int C, int CPG, int G,
                                  int nparts, int pixPerPart) {
    int bg   = blockIdx.y;
    int n    = bg / G;
    int g    = bg - n * G;
    int part = blockIdx.x;
    int p0   = part * pixPerPart;
    int p1   = p0 + pixPerPart; if (p1 > P) p1 = P;

    const float* base = in + (long long)n * (long long)P * (long long)C + (long long)(g * CPG);

    float s = 0.f, sq = 0.f;
    if ((CPG & 3) == 0 && (C & 3) == 0) {
        // Sector-pair indexing: linearize over (pixel, vec-within-group) pairs so
        // that consecutive threads cover consecutive 16B chunks of the SAME
        // pixel's group slice (adjacent threads => one 32B DRAM sector), instead
        // of each thread striding a full pixel (C floats) apart.
        int vecC = CPG >> 2;
        long long n_items = (long long)(p1 - p0) * (long long)vecC;
        for (long long i = threadIdx.x; i < n_items; i += blockDim.x) {
            long long pi = i / vecC;
            int j = (int)(i - pi * vecC);
            int p = p0 + (int)pi;
            const float4* ptr = reinterpret_cast<const float4*>(
                base + (long long)p * (long long)C + (long long)j * 4);
            float4 t = *ptr;
            s  += t.x + t.y + t.z + t.w;
            sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
        }
    } else {
        for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
            const float* ptr = base + (long long)p * (long long)C;
            for (int j = 0; j < CPG; ++j) {
                float t = ptr[j];
                s += t; sq += t * t;
            }
        }
    }
    __shared__ float smem[64];
    block_reduce2(s, sq, smem);
    if (threadIdx.x == 0) {
        partial[(long long)bg * nparts + part] = make_float2(s, sq);
    }
}

// ---- stage 2: finalize stats -> per (n,c) scale/shift -------------------------
__global__ void gn_finalize_kernel(const float2* __restrict__ partial,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ scale,
                                   float* __restrict__ shift,
                                   int nparts, int C, int CPG, int G,
                                   float eps, float invCount) {
    int bg = blockIdx.x;
    int n  = bg / G;
    int g  = bg - n * G;

    float s = 0.f, sq = 0.f;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) {
        float2 t = partial[(long long)bg * nparts + i];
        s += t.x; sq += t.y;
    }
    __shared__ float smem[64];
    __shared__ float mean_s, rstd_s;
    block_reduce2(s, sq, smem);
    if (threadIdx.x == 0) {
        float mean = s * invCount;
        float var  = sq * invCount - mean * mean;
        if (var < 0.f) var = 0.f;
        mean_s = mean;
        rstd_s = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = mean_s, rstd = rstd_s;
    for (int c = threadIdx.x; c < CPG; c += blockDim.x) {
        int ch = g * CPG + c;
        float sc = gamma[ch] * rstd;
        scale[n * C + ch] = sc;
        shift[n * C + ch] = beta[ch] - mean * sc;
    }
}

// ---- GroupNorm affine + SiLU, NHWC in / NHWC out ------------------------------
__global__ void norm_silu_nhwc_kernel(const float4* __restrict__ in,
                                      float4* __restrict__ out,
                                      const float4* __restrict__ scale,
                                      const float4* __restrict__ shift,
                                      long long totalVec, int vecC, int P) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= totalVec) return;
    int cv       = (int)(v % (long long)vecC);
    long long px = v / (long long)vecC;
    int n        = (int)(px / (long long)P);

    float4 t  = in[v];
    float4 sc = scale[(long long)n * vecC + cv];
    float4 sh = shift[(long long)n * vecC + cv];
    float y0 = t.x * sc.x + sh.x;
    float y1 = t.y * sc.y + sh.y;
    float y2 = t.z * sc.z + sh.z;
    float y3 = t.w * sc.w + sh.w;
    float4 r;
    r.x = y0 / (1.f + __expf(-y0));
    r.y = y1 / (1.f + __expf(-y1));
    r.z = y2 / (1.f + __expf(-y2));
    r.w = y3 / (1.f + __expf(-y3));
    out[v] = r;
}

// scalar fallback (C not divisible by 4)
__global__ void norm_silu_nhwc_scalar_kernel(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             const float* __restrict__ scale,
                                             const float* __restrict__ shift,
                                             long long total, int C, int P) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c        = (int)(i % (long long)C);
    long long px = i / (long long)C;
    int n        = (int)(px / (long long)P);
    float y = in[i] * scale[(long long)n * C + c] + shift[(long long)n * C + c];
    out[i]  = y / (1.f + __expf(-y));
}

// ---- GroupNorm affine + SiLU + residual, NHWC in -> NCHW out ------------------
__global__ void norm_silu_res_t_kernel(const float* __restrict__ in,
                                       const float* __restrict__ res,
                                       float* __restrict__ out,
                                       const float* __restrict__ scale,
                                       const float* __restrict__ shift,
                                       int P, int C) {
    __shared__ float tile[32][33];
    int p0 = blockIdx.x * 32;
    int c0 = blockIdx.y * 32;
    int n  = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c = c0 + tx;
    float sc = 0.f, sh = 0.f;
    if (c < C) { sc = scale[n * C + c]; sh = shift[n * C + c]; }

    const float* ib = in + (long long)n * (long long)P * (long long)C;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int p = p0 + ty + k;
        float v = 0.f;
        if (p < P && c < C) {
            float y = ib[(long long)p * (long long)C + c] * sc + sh;
            v = y / (1.f + __expf(-y));
        }
        tile[ty + k][tx] = v;
    }
    __syncthreads();
    int p = p0 + tx;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        int cc = c0 + ty + k;
        if (p < P && cc < C) {
            long long o = ((long long)n * C + cc) * (long long)P + p;
            out[o] = tile[tx][ty + k] + res[o];
        }
    }
}

// ------------------------------ host helpers ----------------------------------
static void compute_scale_shift(const torch::Tensor& in_nhwc,
                                const torch::Tensor& gamma,
                                const torch::Tensor& beta,
                                double eps, int G, int B, int C, int P,
                                torch::Tensor& scale, torch::Tensor& shift) {
    int CPG = C / G;
    long long bg = (long long)B * G;
    int target = 1024;
    int nparts = (int)((target + bg - 1) / bg);
    int maxparts = (P + 255) / 256;
    if (maxparts < 1) maxparts = 1;
    if (nparts > maxparts) nparts = maxparts;
    if (nparts < 1) nparts = 1;
    int pixPerPart = (P + nparts - 1) / nparts;
    nparts = (P + pixPerPart - 1) / pixPerPart;   // every part holds >=1 pixel

    auto opts = in_nhwc.options();
    auto partial = torch::empty({bg * nparts, 2}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(nparts, (unsigned)bg);
    gn_partial_kernel<<<grid, 256, 0, stream>>>(
        in_nhwc.data_ptr<float>(),
        reinterpret_cast<float2*>(partial.data_ptr<float>()),
        P, C, CPG, G, nparts, pixPerPart);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float invCount = 1.0f / (float)((double)P * (double)CPG);
    gn_finalize_kernel<<<(unsigned)bg, 256, 0, stream>>>(
        reinterpret_cast<const float2*>(partial.data_ptr<float>()),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        nparts, C, CPG, G, (float)eps, invCount);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor to_nhwc(torch::Tensor x) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(x.dim() == 4, "expected 4D NCHW tensor");
    TORCH_CHECK(x.is_contiguous(), "expected contiguous NCHW tensor");
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t P = H * W;
    auto out = torch::empty(x.sizes(), x.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(32, 8);
    dim3 grid((unsigned)((P + 63) / 64), (unsigned)((C + 31) / 32), (unsigned)B);
    nchw_to_nhwc_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), (int)P, (int)C, (int)B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> weights_to_nhwc(torch::Tensor w1, torch::Tensor w2) {
    TORCH_CHECK(w1.scalar_type() == torch::kFloat32 && w2.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(w1.dim() == 4 && w2.dim() == 4, "expected 4D conv weights");
    TORCH_CHECK(w1.is_contiguous() && w2.is_contiguous(), "expected contiguous weights");
    int64_t K  = w1.size(0);
    int64_t Ci = w1.size(1);
    TORCH_CHECK(w1.size(2) == 3 && w1.size(3) == 3, "expected 3x3 kernels");
    TORCH_CHECK(w2.size(0) == K && w2.size(1) == Ci && w2.size(2) == 3 && w2.size(3) == 3,
                "w1/w2 shape mismatch");

    auto o1 = torch::empty(w1.sizes(), w1.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto o2 = torch::empty(w2.sizes(), w2.options().memory_format(at::MemoryFormat::ChannelsLast));

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)K, 2);
    int threads = 256;
    size_t shmem = (size_t)Ci * 9 * sizeof(float);
    weights_nchw_to_nhwc_kernel<<<grid, threads, shmem, stream>>>(
        w1.data_ptr<float>(), w2.data_ptr<float>(),
        o1.data_ptr<float>(), o2.data_ptr<float>(), (int)Ci);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o1, o2};
}

torch::Tensor gn_silu_nhwc(torch::Tensor in_nhwc, torch::Tensor gamma, torch::Tensor beta,
                           double eps, int64_t G, int64_t B, int64_t C, int64_t P) {
    TORCH_CHECK(in_nhwc.scalar_type() == torch::kFloat32, "float32 only");
    auto opts = in_nhwc.options();
    auto scale = torch::empty({B * C}, opts);
    auto shift = torch::empty({B * C}, opts);
    compute_scale_shift(in_nhwc, gamma, beta, eps, (int)G, (int)B, (int)C, (int)P, scale, shift);

    auto out = torch::empty_like(in_nhwc);
    auto stream = at::cuda::getCurrentCUDAStream();
    long long total = (long long)B * C * P;
    if ((C & 3) == 0) {
        int vecC = (int)C >> 2;
        long long totalVec = total >> 2;
        int threads = 256;
        long long blocks = (totalVec + threads - 1) / threads;
        norm_silu_nhwc_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in_nhwc.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            reinterpret_cast<const float4*>(scale.data_ptr<float>()),
            reinterpret_cast<const float4*>(shift.data_ptr<float>()),
            totalVec, vecC, (int)P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        norm_silu_nhwc_scalar_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
            in_nhwc.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(), shift.data_ptr<float>(),
            total, (int)C, (int)P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

torch::Tensor gn_silu_res_nchw(torch::Tensor in_nhwc, torch::Tensor res_nchw,
                               torch::Tensor gamma, torch::Tensor beta,
                               double eps, int64_t G, int64_t B, int64_t C, int64_t P) {
    TORCH_CHECK(in_nhwc.scalar_type() == torch::kFloat32, "float32 only");
    auto opts = in_nhwc.options();
    auto scale = torch::empty({B * C}, opts);
    auto shift = torch::empty({B * C}, opts);
    compute_scale_shift(in_nhwc, gamma, beta, eps, (int)G, (int)B, (int)C, (int)P, scale, shift);

    auto out = torch::empty(res_nchw.sizes(), opts);   // contiguous NCHW
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(32, 8);
    dim3 grid((unsigned)((P + 31) / 32), (unsigned)((C + 31) / 32), (unsigned)B);
    norm_silu_res_t_kernel<<<grid, block, 0, stream>>>(
        in_nhwc.data_ptr<float>(), res_nchw.data_ptr<float>(), out.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(), (int)P, (int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP_SRC = r'''
#include <vector>
torch::Tensor to_nhwc(torch::Tensor x);
std::vector<torch::Tensor> weights_to_nhwc(torch::Tensor w1, torch::Tensor w2);
torch::Tensor gn_silu_nhwc(torch::Tensor in_nhwc, torch::Tensor gamma, torch::Tensor beta,
                           double eps, int64_t G, int64_t B, int64_t C, int64_t P);
torch::Tensor gn_silu_res_nchw(torch::Tensor in_nhwc, torch::Tensor res_nchw,
                               torch::Tensor gamma, torch::Tensor beta,
                               double eps, int64_t G, int64_t B, int64_t C, int64_t P);
'''

_ext = load_inline(
    name="vae_resblock_gn_silu_v2",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["to_nhwc", "weights_to_nhwc", "gn_silu_nhwc", "gn_silu_res_nchw"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-gencode=arch=compute_120,code=sm_120",
    ],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # CUDA-graph capture/replay state (host launch overhead reduction).
        self._graphs = {}        # key -> (CUDAGraph, static_out_tensor)
        self._seen = {}          # key -> int (times this signature was seen)
        self._graph_disabled = False

    def _forward_impl(self, x, conv1_weight, norm1_weight, norm1_bias,
                       conv2_weight, norm2_weight, norm2_bias, eps):
        if isinstance(eps, torch.Tensor):
            eps = float(eps.item())
        num_groups = 32

        assert x.dtype == torch.float32, "float32 only"
        res = x if x.is_contiguous() else x.contiguous()

        B, C, H, W = res.shape
        P = H * W

        w1_c = conv1_weight if conv1_weight.is_contiguous() else conv1_weight.contiguous()
        w2_c = conv2_weight if conv2_weight.is_contiguous() else conv2_weight.contiguous()
        w1_nhwc, w2_nhwc = _ext.weights_to_nhwc(w1_c, w2_c)

        x_nhwc = _ext.to_nhwc(res)

        # conv #1 : cuDNN, channels_last input/weight -> channels_last output
        o = F.conv2d(x_nhwc, w1_nhwc, None, 1, 1)
        if not o.is_contiguous(memory_format=torch.channels_last):
            o = o.contiguous(memory_format=torch.channels_last)

        # fused GroupNorm + SiLU (NHWC -> NHWC)
        o = _ext.gn_silu_nhwc(o, norm1_weight.contiguous(), norm1_bias.contiguous(),
                              eps, num_groups, B, C, P)

        # conv #2
        o = F.conv2d(o, w2_nhwc, None, 1, 1)
        if not o.is_contiguous(memory_format=torch.channels_last):
            o = o.contiguous(memory_format=torch.channels_last)

        # fused GroupNorm + SiLU + residual + NHWC->NCHW restore
        out = _ext.gn_silu_res_nchw(o, res, norm2_weight.contiguous(), norm2_bias.contiguous(),
                                    eps, num_groups, B, C, P)
        return out

    @staticmethod
    def _sig(tensors, eps):
        parts = []
        for t in tensors:
            parts.append((t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype))
        parts.append(float(eps))
        return tuple(parts)

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        tensors = (x, conv1_weight, norm1_weight, norm1_bias,
                   conv2_weight, norm2_weight, norm2_bias)

        eps_f = float(eps.item()) if isinstance(eps, torch.Tensor) else float(eps)

        # Fallback conditions: anything unusual bypasses the graph path entirely.
        eager_only = self._graph_disabled
        if not eager_only:
            if not x.is_cuda or x.dtype != torch.float32:
                eager_only = True
            else:
                for t in tensors:
                    if not t.is_cuda:
                        eager_only = True
                        break
                    if t.requires_grad:
                        eager_only = True
                        break
                if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
                    eager_only = True

        if eager_only:
            return self._forward_impl(x, conv1_weight, norm1_weight, norm1_bias,
                                       conv2_weight, norm2_weight, norm2_bias, eps)

        key = self._sig(tensors, eps_f)

        if key in self._graphs:
            g, out = self._graphs[key]
            g.replay()
            return out

        count = self._seen.get(key, 0) + 1
        self._seen[key] = count

        if count < 3:
            # Warm up cuDNN algo selection / caching allocator before capture.
            with torch.no_grad():
                return self._forward_impl(x, conv1_weight, norm1_weight, norm1_bias,
                                           conv2_weight, norm2_weight, norm2_bias, eps)

        # 3rd sighting of this signature: attempt capture.
        try:
            if len(self._graphs) > 8:
                self._graphs.clear()
                self._seen.clear()
                self._seen[key] = count

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.no_grad():
                    self._forward_impl(x, conv1_weight, norm1_weight, norm1_bias,
                                        conv2_weight, norm2_weight, norm2_bias, eps)
                    self._forward_impl(x, conv1_weight, norm1_weight, norm1_bias,
                                        conv2_weight, norm2_weight, norm2_bias, eps)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                out = self._forward_impl(x, conv1_weight, norm1_weight, norm1_bias,
                                          conv2_weight, norm2_weight, norm2_bias, eps)
            torch.cuda.synchronize()

            self._graphs[key] = (g, out)
            g.replay()
            return out
        except Exception:
            self._graph_disabled = True
            return self._forward_impl(x, conv1_weight, norm1_weight, norm1_bias,
                                       conv2_weight, norm2_weight, norm2_bias, eps)
