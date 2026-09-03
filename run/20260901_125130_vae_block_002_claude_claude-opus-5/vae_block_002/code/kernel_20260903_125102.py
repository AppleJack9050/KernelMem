import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#define TP 32

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + __expf(-v));
}

// ---------------------------------------------------------------- statistics
// x: NHWC contiguous. one block handles (n,g) pixel-segment s.
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ partial,   // [2, BG, S]
                                int HW, int C, int G, int CPG, int S, int BG) {
    const int s  = blockIdx.x;
    const int bg = blockIdx.y;
    const int n  = bg / G;
    const int g  = bg - n * G;

    const long long base = (long long)n * HW * (long long)C + (long long)g * CPG;
    const int p_start = (int)(((long long)HW * s) / S);
    const int p_end   = (int)(((long long)HW * (s + 1)) / S);
    const int cpg4    = CPG >> 2;

    float sum = 0.f, sq = 0.f;
    const long long cnt = (long long)(p_end - p_start) * cpg4;
    for (long long i = threadIdx.x; i < cnt; i += blockDim.x) {
        int p = p_start + (int)(i / cpg4);
        int j = (int)(i - (long long)(p - p_start) * cpg4);
        const float4 v = *reinterpret_cast<const float4*>(x + base + (long long)p * C + (j << 2));
        sum += v.x + v.y + v.z + v.w;
        sq  += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // block reduce
    __shared__ float ws[2][32];
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(mask, sum, off);
        sq  += __shfl_down_sync(mask, sq,  off);
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { ws[0][warp] = sum; ws[1][warp] = sq; }
    __syncthreads();
    int nw = blockDim.x >> 5;
    if (warp == 0) {
        sum = (lane < nw) ? ws[0][lane] : 0.f;
        sq  = (lane < nw) ? ws[1][lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(mask, sum, off);
            sq  += __shfl_down_sync(mask, sq,  off);
        }
        if (lane == 0) {
            partial[(long long)bg * S + s]                 = sum;
            partial[(long long)BG * S + (long long)bg * S + s] = sq;
        }
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int S, int BG, float inv_count, float eps) {
    const int bg = blockIdx.x;
    float sum = 0.f, sq = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        sum += partial[(long long)bg * S + i];
        sq  += partial[(long long)BG * S + (long long)bg * S + i];
    }
    __shared__ float ws[2][32];
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(mask, sum, off);
        sq  += __shfl_down_sync(mask, sq,  off);
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { ws[0][warp] = sum; ws[1][warp] = sq; }
    __syncthreads();
    int nw = blockDim.x >> 5;
    if (warp == 0) {
        sum = (lane < nw) ? ws[0][lane] : 0.f;
        sq  = (lane < nw) ? ws[1][lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(mask, sum, off);
            sq  += __shfl_down_sync(mask, sq,  off);
        }
        if (lane == 0) {
            float m = sum * inv_count;
            float v = sq * inv_count - m * m;
            v = v < 0.f ? 0.f : v;
            mean[bg] = m;
            rstd[bg] = rsqrtf(v + eps);
        }
    }
}

// ------------------------------------------- GN(affine) + SiLU, NHWC -> NHWC
__global__ void gn_silu_nhwc_kernel(const float* __restrict__ x,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ rstd,
                                    float* __restrict__ y,
                                    int HWC4, int C4, int CPG, int G) {
    const int n = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HWC4) return;
    const int c0 = (idx % C4) << 2;
    const int g  = c0 / CPG;
    const int bg = n * G + g;
    const float mu = mean[bg], rs = rstd[bg];

    const long long off = (long long)n * HWC4 + idx;
    const float4 v  = *reinterpret_cast<const float4*>(x + (off << 2));
    const float4 gm = *reinterpret_cast<const float4*>(gamma + c0);
    const float4 bt = *reinterpret_cast<const float4*>(beta + c0);
    float4 o;
    o.x = silu_f((v.x - mu) * rs * gm.x + bt.x);
    o.y = silu_f((v.y - mu) * rs * gm.y + bt.y);
    o.z = silu_f((v.z - mu) * rs * gm.z + bt.z);
    o.w = silu_f((v.w - mu) * rs * gm.w + bt.w);
    *reinterpret_cast<float4*>(y + (off << 2)) = o;
}

// ---------------- GN(affine) + SiLU + residual add, NHWC in -> NHWC out
// Writes into a caller-provided output buffer (needed so pipeline chunks
// can write into disjoint slices of one shared output tensor instead of
// each allocating its own tensor via at::empty_like).
__global__ void gn_silu_res_nhwc_out_kernel(const float* __restrict__ x,
                                            const float* __restrict__ res,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            const float* __restrict__ mean,
                                            const float* __restrict__ rstd,
                                            float* __restrict__ out,
                                            int HWC4, int C4, int CPG, int G) {
    const int n = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HWC4) return;
    const int c0 = (idx % C4) << 2;
    const int g  = c0 / CPG;
    const int bg = n * G + g;
    const float mu = mean[bg], rs = rstd[bg];

    const long long off = (long long)n * HWC4 + idx;
    const float4 v  = *reinterpret_cast<const float4*>(x + (off << 2));
    const float4 r  = *reinterpret_cast<const float4*>(res + (off << 2));
    const float4 gm = *reinterpret_cast<const float4*>(gamma + c0);
    const float4 bt = *reinterpret_cast<const float4*>(beta + c0);
    float4 o;
    o.x = silu_f((v.x - mu) * rs * gm.x + bt.x) + r.x;
    o.y = silu_f((v.y - mu) * rs * gm.y + bt.y) + r.y;
    o.z = silu_f((v.z - mu) * rs * gm.z + bt.z) + r.z;
    o.w = silu_f((v.w - mu) * rs * gm.w + bt.w) + r.w;
    *reinterpret_cast<float4*>(out + (off << 2)) = o;
}

// ------------------------------------------------------------------- launchers
static void compute_stats(const at::Tensor& x, int B, int HW, int C, int G,
                          float eps, at::Tensor& mean, at::Tensor& rstd) {
    const int CPG = C / G;
    const int BG = B * G;
    int S = (456 + BG - 1) / BG;
    int maxS = (HW + 255) / 256;
    if (S > maxS) S = maxS;
    if (S < 1) S = 1;
    auto opts = x.options();
    auto partial = at::empty({2, BG, S}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(S, BG);
    gn_stats_kernel<<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), partial.data_ptr<float>(), HW, C, G, CPG, S, BG);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float inv_count = 1.0f / (float)((long long)HW * CPG);
    gn_finalize_kernel<<<BG, 256, 0, stream>>>(
        partial.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        S, BG, inv_count, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                           int64_t B, int64_t HW, int64_t C, int64_t G, double eps) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    auto opts = x.options();
    auto mean = at::empty({B * G}, opts);
    auto rstd = at::empty({B * G}, opts);
    compute_stats(x, (int)B, (int)HW, (int)C, (int)G, (float)eps, mean, rstd);

    auto y = at::empty_like(x);
    const int C4 = (int)C / 4;
    const int HWC4 = (int)HW * C4;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((HWC4 + 255) / 256, (unsigned)B);
    gn_silu_nhwc_kernel<<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), y.data_ptr<float>(),
        HWC4, C4, (int)(C / G), (int)G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// out-parameter variant: writes GN2+SiLU+residual directly into a
// caller-allocated `out` tensor (must already have the same shape/dtype
// as `x`), so pipeline chunks can target disjoint slices of one buffer.
void gn_silu_res_nhwc_out(torch::Tensor x, torch::Tensor res,
                          torch::Tensor gamma, torch::Tensor beta,
                          torch::Tensor out,
                          int64_t B, int64_t HW, int64_t C, int64_t G, double eps) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(out.sizes() == x.sizes() && out.scalar_type() == at::kFloat,
                "out must match x in shape/dtype");
    auto opts = x.options();
    auto mean = at::empty({B * G}, opts);
    auto rstd = at::empty({B * G}, opts);
    compute_stats(x, (int)B, (int)HW, (int)C, (int)G, (float)eps, mean, rstd);

    const int C4 = (int)C / 4;
    const int HWC4 = (int)HW * C4;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((HWC4 + 255) / 256, (unsigned)B);
    gn_silu_res_nhwc_out_kernel<<<grid, 256, 0, stream>>>(
        x.data_ptr<float>(), res.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), out.data_ptr<float>(),
        HWC4, C4, (int)(C / G), (int)G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Backward-compatible wrapper kept for any external caller expecting a
// returned tensor (allocates internally, then forwards to the out-param path).
torch::Tensor gn_silu_res_nhwc(torch::Tensor x, torch::Tensor res,
                               torch::Tensor gamma, torch::Tensor beta,
                               int64_t B, int64_t HW, int64_t C, int64_t G, double eps) {
    auto out = at::empty_like(x);
    gn_silu_res_nhwc_out(x, res, gamma, beta, out, B, HW, C, G, eps);
    return out;
}
"""

_CPP = r"""
torch::Tensor gn_silu_nhwc(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                           int64_t B, int64_t HW, int64_t C, int64_t G, double eps);
void gn_silu_res_nhwc_out(torch::Tensor x, torch::Tensor res,
                          torch::Tensor gamma, torch::Tensor beta,
                          torch::Tensor out,
                          int64_t B, int64_t HW, int64_t C, int64_t G, double eps);
torch::Tensor gn_silu_res_nhwc(torch::Tensor x, torch::Tensor res,
                               torch::Tensor gamma, torch::Tensor beta,
                               int64_t B, int64_t HW, int64_t C, int64_t G, double eps);
"""

_ext = load_inline(
    name="vae_res_block_pipelined_nhwc",
    cpp_sources=_CPP,
    cuda_sources=_SRC,
    functions=["gn_silu_nhwc", "gn_silu_res_nhwc_out", "gn_silu_res_nhwc"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-use_fast_math",
        "-gencode=arch=compute_90,code=sm_90",
    ],
)

# Lazily-created side stream used to overlap the memory-bound GroupNorm/SiLU
# kernels of one batch-chunk with the compute-bound conv of the next chunk.
_SIDE_STREAM = None


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        global _SIDE_STREAM
        G = 32
        B, C, H, W = x.shape
        HW = H * W

        if x.dtype != torch.float32 or C % (4 * G) != 0:
            # generic fallback (keeps semantics for unsupported configs)
            out = F.conv2d(x, conv1_weight, None, 1, 1)
            out = F.group_norm(out, G, norm1_weight, norm1_bias, eps)
            out = F.silu(out)
            out = F.conv2d(out, conv2_weight, None, 1, 1)
            out = F.group_norm(out, G, norm2_weight, norm2_bias, eps)
            out = F.silu(out)
            return out + x

        xl = x.contiguous(memory_format=torch.channels_last)
        w1 = conv1_weight.contiguous(memory_format=torch.channels_last)
        w2 = conv2_weight.contiguous(memory_format=torch.channels_last)
        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        nch = min(4, B)

        if nch < 2:
            # Single-stream path (unchanged behaviour for B==1).
            y = F.conv2d(xl, w1, None, 1, 1)                       # cuDNN NHWC (vendor)
            y = _ext.gn_silu_nhwc(y, g1, b1, B, HW, C, G, float(eps))
            y = F.conv2d(y, w2, None, 1, 1)                        # cuDNN NHWC (vendor)
            out = torch.empty_like(xl)
            _ext.gn_silu_res_nhwc_out(y, xl, g2, b2, out, B, HW, C, G, float(eps))
            return out

        # Batch-chunked two-stream pipeline: GroupNorm statistics are
        # computed per-(n,g), so the whole residual block is independent
        # per batch element. We split B into `nch` chunks, run all conv2d
        # calls on the main stream (in program order) and all GN/SiLU/
        # residual kernels on a side stream, linked with cuda events, so
        # chunk i's memory-bound GN/SiLU work overlaps chunk i+1's
        # compute-bound conv on the main stream.
        if _SIDE_STREAM is None:
            _SIDE_STREAM = torch.cuda.Stream()
        side = _SIDE_STREAM
        main = torch.cuda.current_stream()

        out = torch.empty_like(xl)
        bounds = [(i * B // nch, (i + 1) * B // nch) for i in range(nch)]
        bounds = [(a, b) for (a, b) in bounds if b > a]

        final_events = []
        keep_alive = []

        for (a, b) in bounds:
            n = b - a
            xl_chunk = xl[a:b]

            c1 = F.conv2d(xl_chunk, w1, None, 1, 1)                 # main stream
            ev_a = torch.cuda.Event()
            ev_a.record(main)

            side.wait_event(ev_a)
            with torch.cuda.stream(side):
                c1.record_stream(side)
                y = _ext.gn_silu_nhwc(c1, g1, b1, n, HW, C, G, float(eps))
                ev_b = torch.cuda.Event()
                ev_b.record(side)

            main.wait_event(ev_b)
            y.record_stream(main)
            c2 = F.conv2d(y, w2, None, 1, 1)                        # main stream
            ev_c = torch.cuda.Event()
            ev_c.record(main)

            side.wait_event(ev_c)
            with torch.cuda.stream(side):
                c2.record_stream(side)
                xl_chunk.record_stream(side)
                out_chunk = out.narrow(0, a, n)
                out_chunk.record_stream(side)
                _ext.gn_silu_res_nhwc_out(c2, xl_chunk, g2, b2, out_chunk,
                                          n, HW, C, G, float(eps))
                ev_d = torch.cuda.Event()
                ev_d.record(side)

            final_events.append(ev_d)
            keep_alive.append((c1, y, c2, xl_chunk))

        for e in final_events:
            main.wait_event(e)

        return out
