import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CH   256
#define NG   32
#define CPG  8

__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                int HW, int nChunks, int chunk)
{
    const int n  = blockIdx.y;
    const int ci = blockIdx.x;
    const int c  = threadIdx.x;
    const int p0 = ci * chunk;
    int p1 = p0 + chunk;
    if (p1 > HW) p1 = HW;

    const float* base = x + (long long)n * HW * CH + c;
    float s = 0.f, sq = 0.f;
    for (int p = p0; p < p1; ++p) {
        float v = base[(long long)p * CH];
        s  += v;
        sq += v * v;
    }
#pragma unroll
    for (int off = 1; off < CPG; off <<= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off);
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if ((c & (CPG - 1)) == 0) {
        const long long o = ((long long)n * nChunks + ci) * NG + (c >> 3);
        psum[o] = s;
        psq[o]  = sq;
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ psum,
                                   const float* __restrict__ psq,
                                   float* __restrict__ mr,
                                   int nChunks, int HW, float eps)
{
    const int n = blockIdx.x;
    const int g = threadIdx.x;
    float s = 0.f, sq = 0.f;
    for (int i = 0; i < nChunks; ++i) {
        const long long o = ((long long)n * nChunks + i) * NG + g;
        s  += psum[o];
        sq += psq[o];
    }
    const float cnt  = (float)HW * (float)CPG;
    const float mean = s / cnt;
    float var = sq / cnt - mean * mean;
    if (!(var > 0.f)) var = 0.f;
    mr[(n * NG + g) * 2 + 0] = mean;
    mr[(n * NG + g) * 2 + 1] = rsqrtf(var + eps);
}

__global__ void gn_silu_nhwc_kernel(float* __restrict__ y,
                                    const float* __restrict__ gam,
                                    const float* __restrict__ bet,
                                    const float* __restrict__ mr,
                                    long long nvec, int HW)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nvec) return;
    const int c4 = (int)(i & (CH / 4 - 1));
    const int c  = c4 * 4;
    const long long pix = i >> 6;           // CH/4 == 64
    const int n = (int)(pix / HW);
    const int g = c >> 3;
    const float mean = mr[(n * NG + g) * 2 + 0];
    const float rstd = mr[(n * NG + g) * 2 + 1];

    float4 v  = reinterpret_cast<float4*>(y)[i];
    float4 gg = reinterpret_cast<const float4*>(gam)[c4];
    float4 bb = reinterpret_cast<const float4*>(bet)[c4];

    float t0 = (v.x - mean) * rstd * gg.x + bb.x;
    float t1 = (v.y - mean) * rstd * gg.y + bb.y;
    float t2 = (v.z - mean) * rstd * gg.z + bb.z;
    float t3 = (v.w - mean) * rstd * gg.w + bb.w;

    v.x = t0 / (1.f + __expf(-t0));
    v.y = t1 / (1.f + __expf(-t1));
    v.z = t2 / (1.f + __expf(-t2));
    v.w = t3 / (1.f + __expf(-t3));
    reinterpret_cast<float4*>(y)[i] = v;
}

__global__ void gn_silu_res_t_kernel(const float* __restrict__ y,
                                     const float* __restrict__ res,
                                     float* __restrict__ out,
                                     const float* __restrict__ gam,
                                     const float* __restrict__ bet,
                                     const float* __restrict__ mr,
                                     int HW)
{
    __shared__ float tile[32][33];
    const int n     = blockIdx.z;
    const int cBase = blockIdx.y * 32;
    const int pBase = blockIdx.x * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const float* yb = y + (long long)n * HW * CH;
    const int c_in = cBase + tx;
    const int g_in = c_in >> 3;
    const float mean = mr[(n * NG + g_in) * 2 + 0];
    const float rstd = mr[(n * NG + g_in) * 2 + 1];
    const float gg = gam[c_in];
    const float bb = bet[c_in];

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = pBase + ty + j * 8;
        float o = 0.f;
        if (p < HW) {
            const float v = yb[(long long)p * CH + c_in];
            const float t = (v - mean) * rstd * gg + bb;
            o = t / (1.f + __expf(-t));
        }
        tile[ty + j * 8][tx] = o;
    }
    __syncthreads();

    const int p = pBase + tx;
    if (p < HW) {
        const float* rb = res + (long long)n * CH * HW;
        float*       ob = out + (long long)n * CH * HW;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = cBase + ty + j * 8;
            ob[(long long)c * HW + p] = tile[tx][ty + j * 8] + rb[(long long)c * HW + p];
        }
    }
}

// ---------------------------------------------------------------------
// NCHW -> NHWC (channels_last-physical) tiled transpose, C fixed to 256.
// Generalises over any logical (B, 256, H, W) tensor: also usable for the
// two 3x3 conv weights viewed as (O, C=256, H=3, W=3) since a
// channels_last-contiguous weight of that shape has the exact same
// physical layout (dim0, dim2, dim3, dim1) as an NHWC activation tensor.
// Both phases are fully 128B-coalesced (read along W within a warp in
// Phase A, write along C within a warp in Phase B); the p<HW guard makes
// the tail-safe for odd HW (e.g. 131*131 = 17161).
// ---------------------------------------------------------------------
__global__ void nchw_to_nhwc_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int HW)
{
    __shared__ float tile[32][33];
    const int n     = blockIdx.z;
    const int cBase = blockIdx.y * 32;
    const int pBase = blockIdx.x * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const float* sb = src + (long long)n * CH * HW;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = pBase + tx;                 // coalesced NCHW read (along W)
        const int c = cBase + ty + j * 8;
        float v = 0.f;
        if (p < HW) {
            v = sb[(long long)c * HW + p];
        }
        tile[ty + j * 8][tx] = v;
    }
    __syncthreads();

    float* db = dst + (long long)n * HW * CH;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = pBase + ty + j * 8;
        const int c = cBase + tx;                 // coalesced NHWC write (along C)
        if (p < HW) {
            db[(long long)p * CH + c] = tile[tx][ty + j * 8];
        }
    }
}

torch::Tensor to_nhwc(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(x.dim() == 4, "expected 4D tensor");
    TORCH_CHECK(x.size(1) == CH, "expected 256 channels");
    TORCH_CHECK(x.is_contiguous(), "expected contiguous (NCHW) input");

    const int B  = (int)x.size(0);
    const int H  = (int)x.size(2);
    const int W  = (int)x.size(3);
    const int HW = H * W;

    auto out = torch::empty({B, CH, H, W}, x.options().memory_format(at::MemoryFormat::ChannelsLast));

    dim3 blk(32, 8);
    dim3 grid((HW + 31) / 32, CH / 32, B);
    auto stream = at::cuda::getCurrentCUDAStream();
    nchw_to_nhwc_kernel<<<grid, blk, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

static inline void compute_stats(const float* xp, int B, int HW, float eps,
                                 torch::Tensor& mr, cudaStream_t stream,
                                 const torch::TensorOptions& opts)
{
    int target = 340;
    int nChunks = (target + B - 1) / B;
    if (nChunks > HW) nChunks = HW;
    if (nChunks < 1)  nChunks = 1;
    const int chunk = (HW + nChunks - 1) / nChunks;
    nChunks = (HW + chunk - 1) / chunk;

    auto psum = torch::empty({(long)B * nChunks * NG}, opts);
    auto psq  = torch::empty({(long)B * nChunks * NG}, opts);

    dim3 grid(nChunks, B);
    gn_stats_kernel<<<grid, CH, 0, stream>>>(xp, psum.data_ptr<float>(),
                                             psq.data_ptr<float>(), HW, nChunks, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gn_finalize_kernel<<<B, NG, 0, stream>>>(psum.data_ptr<float>(), psq.data_ptr<float>(),
                                             mr.data_ptr<float>(), nChunks, HW, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// in-place GroupNorm + SiLU on a channels_last (B,256,H,W) tensor
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gam, torch::Tensor bet, double eps)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    const int B  = (int)y.size(0);
    const int HW = (int)(y.size(2) * y.size(3));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = y.options();
    auto mr = torch::empty({(long)B * NG * 2}, opts);
    compute_stats(y.data_ptr<float>(), B, HW, (float)eps, mr, stream, opts);

    const long long nvec = (long long)B * HW * CH / 4;
    const int blk = 256;
    const long long grid = (nvec + blk - 1) / blk;
    gn_silu_nhwc_kernel<<<(unsigned)grid, blk, 0, stream>>>(
        y.data_ptr<float>(), gam.data_ptr<float>(), bet.data_ptr<float>(),
        mr.data_ptr<float>(), nvec, HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// GroupNorm + SiLU + residual add + NHWC->NCHW, out is contiguous NCHW
torch::Tensor gn_silu_res_nchw(torch::Tensor y, torch::Tensor gam, torch::Tensor bet,
                               torch::Tensor res, double eps)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    const int B  = (int)y.size(0);
    const int H  = (int)y.size(2);
    const int W  = (int)y.size(3);
    const int HW = H * W;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = y.options();
    auto mr = torch::empty({(long)B * NG * 2}, opts);
    compute_stats(y.data_ptr<float>(), B, HW, (float)eps, mr, stream, opts);

    auto out = torch::empty({B, CH, H, W}, opts);
    dim3 blk(32, 8);
    dim3 grid((HW + 31) / 32, CH / 32, B);
    gn_silu_res_t_kernel<<<grid, blk, 0, stream>>>(
        y.data_ptr<float>(), res.data_ptr<float>(), out.data_ptr<float>(),
        gam.data_ptr<float>(), bet.data_ptr<float>(), mr.data_ptr<float>(), HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor gn_silu_nhwc(torch::Tensor y, torch::Tensor gam, torch::Tensor bet, double eps);
torch::Tensor gn_silu_res_nchw(torch::Tensor y, torch::Tensor gam, torch::Tensor bet,
                               torch::Tensor res, double eps);
torch::Tensor to_nhwc(torch::Tensor x);
"""

_ext = load_inline(
    name="vae_resblock_gnsilu",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu_nhwc", "gn_silu_res_nchw", "to_nhwc"],
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
        # CUDA-graph capture cache: key -> (graph, static_output, kept_refs)
        self._cache = {}
        self._captures = 0
        self._pool = None
        self._blacklist = set()

    def _to_nhwc_weight(self, w):
        # Weight (O, 256, 3, 3): our custom tiled kernel views this exactly
        # like an activation (B=O, C=256, HW=9) and produces a
        # channels_last-contiguous (O,256,3,3) tensor -- the same physical
        # layout cuDNN needs for a channels_last conv weight.
        if w.is_contiguous(memory_format=torch.channels_last):
            return w
        if (w.is_contiguous() and w.dim() == 4 and w.size(1) == 256
                and w.size(2) == 3 and w.size(3) == 3):
            return _ext.to_nhwc(w)
        return w.contiguous(memory_format=torch.channels_last)

    def _eager(self, x, conv1_weight, norm1_weight, norm1_bias,
               conv2_weight, norm2_weight, norm2_bias, eps):
        res = x if x.is_contiguous() else x.contiguous()
        # custom coalesced NCHW->NHWC(channels_last) transpose replaces the
        # generic PyTorch contiguous(memory_format=channels_last) staging
        # copy for the activation entering conv1.
        x_cl = _ext.to_nhwc(res)
        w1 = self._to_nhwc_weight(conv1_weight)
        w2 = self._to_nhwc_weight(conv2_weight)
        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        y = F.conv2d(x_cl, w1, bias=None, stride=1, padding=1)
        y = _ext.gn_silu_nhwc(y, g1, b1, float(eps))
        y = F.conv2d(y, w2, bias=None, stride=1, padding=1)
        return _ext.gn_silu_res_nchw(y, g2, b2, res, float(eps))

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        C = x.size(1)
        fast = (
            x.is_cuda and x.dtype == torch.float32 and C == 256 and x.dim() == 4
            and conv1_weight.size(2) == 3 and conv1_weight.size(3) == 3
        )
        if not fast:
            num_groups = 32
            out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
            out = F.silu(out)
            out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
            out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
            out = F.silu(out)
            return out + x

        args = (x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps)

        key = (
            tuple(x.shape), x.stride(), float(eps),
            x.data_ptr(), conv1_weight.data_ptr(), conv2_weight.data_ptr(),
            norm1_weight.data_ptr(), norm1_bias.data_ptr(),
            norm2_weight.data_ptr(), norm2_bias.data_ptr(),
        )

        cached = self._cache.get(key)
        if cached is not None:
            g, out, _refs = cached
            g.replay()
            return out

        if (torch.cuda.is_current_stream_capturing()
                or self._captures >= 8
                or key in self._blacklist):
            return self._eager(*args)

        try:
            # warm-up on a side stream so cuDNN heuristics/workspaces and the
            # load_inline kernels are fully resolved before capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = self._eager(*args)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                out = self._eager(*args)
            self._pool = g.pool()

            # IMPORTANT: cudaStreamBeginCapture only *records* the kernel
            # launches inside the `with torch.cuda.graph(...)` block; it does
            # NOT execute them. `out` is therefore still uninitialized memory
            # at this point. Replay once now so this very first call (the one
            # that triggered capture) also returns a correct, freshly
            # computed result -- subsequent calls hit the cache and replay.
            g.replay()

            self._cache[key] = (g, out, args)
            self._captures += 1
            return out
        except Exception:
            self._blacklist.add(key)
            return self._eager(*args)
