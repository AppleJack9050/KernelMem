# ==========================================================================
# ModelNew — SOL problem 002: Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> +residual
#
# Optimisation applied here: stream_pipeline_overlap.
# The batch is sharded across up to 4 CUDA streams so the DRAM-bound elementwise
# segment (transpose / GN reduce / GN apply, ~90% DRAM but ~5-35% SM) co-resides
# with the tensor-core-bound cuDNN convolutions (~96% of TF32 peak, ~15% DRAM).
# Supporting changes: (a) an _out variant of the fused GN+SiLU+residual epilogue so
# shards write directly into one pre-allocated output tensor, (b) grid-stride
# pixel-tile loops with a ~2-wave grid cap in the two tiled transpose kernels so a
# single shard no longer monopolises all 170 SMs.
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 32
#define TROWS 8
// ~2 waves of 256-thread tile CTAs on 170 SMs (8 resident CTAs/SM assumed)
#define GRID_CAP 2720L

__device__ __forceinline__ float silu_f(float z){
    return z / (1.0f + expf(-z));
}

// ---------------------------------------------------------------- transpose
// grid-stride over pixel tiles: grid.x is capped so other streams' CTAs fit.
__global__ void nchw2nhwc_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int C, long HW, long ntx)
{
    __shared__ float tile[TILE][TILE + 1];
    int  c0 = blockIdx.y * TILE;
    int  n  = blockIdx.z;
    const float* xn = x + (size_t)n * (size_t)C * (size_t)HW;
    float*       yn = y + (size_t)n * (size_t)HW * (size_t)C;
    int tx = threadIdx.x, ty = threadIdx.y;

    for (long bx = blockIdx.x; bx < ntx; bx += gridDim.x) {
        long p0 = bx * TILE;
        #pragma unroll
        for (int i = 0; i < TILE; i += TROWS) {
            int  c = c0 + ty + i;
            long p = p0 + tx;
            float v = 0.f;
            if (c < C && p < HW) v = xn[(size_t)c * (size_t)HW + (size_t)p];
            tile[ty + i][tx] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i += TROWS) {
            long p = p0 + ty + i;
            int  c = c0 + tx;
            if (c < C && p < HW) yn[(size_t)p * (size_t)C + (size_t)c] = tile[tx][ty + i];
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------ GN statistics
// grid = (nch, B), block = G * ppb   (ppb pixel-lanes; a warp reads a full pixel row)
__global__ void gn_reduce_kernel(const float* __restrict__ y,
                                 double* __restrict__ psum,
                                 double* __restrict__ psq,
                                 int C, long HW, int G, int CPG,
                                 int ppb, long chunk, int nch)
{
    extern __shared__ float sh[];
    float* s_s = sh;
    float* s_q = sh + blockDim.x;

    int tid = threadIdx.x;
    int g   = tid % G;
    int dp  = tid / G;
    int n   = blockIdx.y;

    long pstart = (long)blockIdx.x * chunk;
    long pend   = pstart + chunk;
    if (pend > HW) pend = HW;

    const float* base = y + (size_t)n * (size_t)HW * (size_t)C + (size_t)g * (size_t)CPG;

    float s = 0.f, q = 0.f;
    if ((CPG & 3) == 0) {
        int nv = CPG >> 2;
        for (long p = pstart + dp; p < pend; p += ppb) {
            const float4* v4 = reinterpret_cast<const float4*>(base + (size_t)p * (size_t)C);
            for (int j = 0; j < nv; ++j) {
                float4 v = v4[j];
                s += v.x + v.y + v.z + v.w;
                q += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    } else {
        for (long p = pstart + dp; p < pend; p += ppb) {
            const float* v = base + (size_t)p * (size_t)C;
            for (int j = 0; j < CPG; ++j) { float t = v[j]; s += t; q += t * t; }
        }
    }

    s_s[tid] = s; s_q[tid] = q;
    __syncthreads();
    for (int stride = ppb >> 1; stride > 0; stride >>= 1) {
        if (dp < stride) {
            s_s[tid] += s_s[tid + stride * G];
            s_q[tid] += s_q[tid + stride * G];
        }
        __syncthreads();
    }
    if (dp == 0) {
        size_t off = (size_t)(n * G + g) * (size_t)nch + (size_t)blockIdx.x;
        psum[off] = (double)s_s[tid];
        psq[off]  = (double)s_q[tid];
    }
}

// grid = (B*G), block = 128
__global__ void gn_finalize_kernel(const double* __restrict__ psum,
                                   const double* __restrict__ psq,
                                   float* __restrict__ mean,
                                   float* __restrict__ rstd,
                                   int nch, double count, double eps)
{
    __shared__ double sh[256];
    int bg  = blockIdx.x;
    int tid = threadIdx.x;
    double s = 0.0, q = 0.0;
    for (int i = tid; i < nch; i += blockDim.x) {
        s += psum[(size_t)bg * (size_t)nch + i];
        q += psq [(size_t)bg * (size_t)nch + i];
    }
    sh[tid] = s; sh[128 + tid] = q;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) { sh[tid] += sh[tid + st]; sh[128 + tid] += sh[128 + tid + st]; }
        __syncthreads();
    }
    if (tid == 0) {
        double m = sh[0] / count;
        double v = sh[128] / count - m * m;
        if (v < 0.0) v = 0.0;
        mean[bg] = (float)m;
        rstd[bg] = (float)(1.0 / sqrt(v + eps));
    }
}

// ------------------------------------------- GN affine + SiLU (NHWC, in-place, float4)
__global__ void gn_apply_nhwc_fast(float* __restrict__ y,
                                   const float* __restrict__ mean,
                                   const float* __restrict__ rstd,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   int C4, long HW, int G, int CPG, int ppb)
{
    int n  = blockIdx.y;
    int c4 = threadIdx.x % C4;
    int dp = threadIdx.x / C4;
    int c  = c4 << 2;
    int g  = c / CPG;

    float m = mean[n * G + g];
    float r = rstd[n * G + g];
    float a0 = r * w[c],     a1 = r * w[c + 1], a2 = r * w[c + 2], a3 = r * w[c + 3];
    float d0 = b[c]     - m * a0;
    float d1 = b[c + 1] - m * a1;
    float d2 = b[c + 2] - m * a2;
    float d3 = b[c + 3] - m * a3;

    float4* base = reinterpret_cast<float4*>(y) + (size_t)n * (size_t)HW * (size_t)C4;
    for (long p = (long)blockIdx.x * ppb + dp; p < HW; p += (long)gridDim.x * ppb) {
        float4 v = base[(size_t)p * (size_t)C4 + (size_t)c4];
        v.x = silu_f(v.x * a0 + d0);
        v.y = silu_f(v.y * a1 + d1);
        v.z = silu_f(v.z * a2 + d2);
        v.w = silu_f(v.w * a3 + d3);
        base[(size_t)p * (size_t)C4 + (size_t)c4] = v;
    }
}

// generic fallback (any C / CPG), NHWC in-place
__global__ void gn_apply_nhwc_gen(float* __restrict__ y,
                                  const float* __restrict__ mean,
                                  const float* __restrict__ rstd,
                                  const float* __restrict__ w,
                                  const float* __restrict__ b,
                                  int C, long HW, int G, int CPG, long total)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        int  c   = (int)(i % (long)C);
        long pix = i / (long)C;
        int  n   = (int)(pix / HW);
        int  g   = c / CPG;
        float v = y[i];
        float z = (v - mean[n * G + g]) * rstd[n * G + g] * w[c] + b[c];
        y[i] = silu_f(z);
    }
}

// ----------- GN affine + SiLU + residual + NHWC->NCHW transpose (fused epilogue)
// grid-stride over pixel tiles with capped grid.x (stream co-residency).
__global__ void gn_apply_out_kernel(const float* __restrict__ y,   // NHWC
                                    const float* __restrict__ res, // NCHW
                                    float* __restrict__ out,       // NCHW
                                    const float* __restrict__ mean,
                                    const float* __restrict__ rstd,
                                    const float* __restrict__ w,
                                    const float* __restrict__ b,
                                    int C, long HW, int G, int CPG, long ntx)
{
    __shared__ float tile[TILE][TILE + 1];
    int  c0 = blockIdx.y * TILE;
    int  n  = blockIdx.z;
    const float* yn = y + (size_t)n * (size_t)HW * (size_t)C;
    int tx = threadIdx.x, ty = threadIdx.y;

    for (long bx = blockIdx.x; bx < ntx; bx += gridDim.x) {
        long p0 = bx * TILE;
        #pragma unroll
        for (int i = 0; i < TILE; i += TROWS) {
            int  c = c0 + tx;
            long p = p0 + ty + i;
            float t = 0.f;
            if (c < C && p < HW) {
                float v = yn[(size_t)p * (size_t)C + (size_t)c];
                int   g = c / CPG;
                float z = (v - mean[n * G + g]) * rstd[n * G + g] * w[c] + b[c];
                t = silu_f(z);
            }
            tile[ty + i][tx] = t;
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i += TROWS) {
            int  c = c0 + ty + i;
            long p = p0 + tx;
            if (c < C && p < HW) {
                size_t off = ((size_t)n * (size_t)C + (size_t)c) * (size_t)HW + (size_t)p;
                out[off] = tile[tx][ty + i] + res[off];
            }
        }
        __syncthreads();
    }
}

// =========================================================== host side helpers
static inline int pow2_floor(int v) {
    int r = 1;
    while ((r << 1) <= v) r <<= 1;
    return r;
}

static void compute_stats(const float* yptr, int B, int C, long HW, int G, double eps,
                          const torch::TensorOptions& fopts,
                          torch::Tensor& mean, torch::Tensor& rstd)
{
    int CPG = C / G;
    auto stream = at::cuda::getCurrentCUDAStream();

    int ppb = 1;
    if (G <= 256) ppb = pow2_floor(256 / G);
    if (ppb < 1) ppb = 1;
    int block = G * ppb;
    TORCH_CHECK(block <= 1024, "group count too large");

    int nch_target = (int)((1024 + B - 1) / B);
    long nch_max_l = (HW + ppb - 1) / ppb;
    int nch = nch_target;
    if ((long)nch > nch_max_l) nch = (int)nch_max_l;
    if (nch < 1) nch = 1;
    long chunk = (HW + nch - 1) / nch;
    chunk = ((chunk + ppb - 1) / ppb) * ppb;      // align, then RECOMPUTE the grid so that
    if (chunk < 1) chunk = 1;
    nch = (int)((HW + chunk - 1) / chunk);        // full coverage holds for any HW

    auto dopts = fopts.dtype(torch::kDouble);
    auto psum = torch::empty({(long)B * G * nch}, dopts);
    auto psq  = torch::empty({(long)B * G * nch}, dopts);

    dim3 grid(nch, B);
    size_t shmem = (size_t)2 * block * sizeof(float);
    gn_reduce_kernel<<<grid, block, shmem, stream>>>(
        yptr, psum.data_ptr<double>(), psq.data_ptr<double>(),
        C, HW, G, CPG, ppb, chunk, nch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mean = torch::empty({(long)B * G}, fopts);
    rstd = torch::empty({(long)B * G}, fopts);
    double count = (double)HW * (double)CPG;
    gn_finalize_kernel<<<B * G, 128, 0, stream>>>(
        psum.data_ptr<double>(), psq.data_ptr<double>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), nch, count, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------ exported
torch::Tensor nchw_to_nhwc(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(x.dim() == 4 && x.is_contiguous(), "contiguous NCHW expected");
    int B = (int)x.size(0), C = (int)x.size(1);
    long H = x.size(2), W = x.size(3), HW = H * W;
    auto out = torch::empty({B, H, W, C}, x.options());
    long ntx = (HW + TILE - 1) / TILE;
    long gx  = ntx < GRID_CAP ? ntx : GRID_CAP;
    if (gx < 1) gx = 1;
    dim3 block(TILE, TROWS);
    dim3 grid((unsigned)gx, (unsigned)((C + TILE - 1) / TILE), (unsigned)B);
    auto stream = at::cuda::getCurrentCUDAStream();
    nchw2nhwc_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), C, HW, ntx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void gn_silu_nhwc_(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps, int64_t G_)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(y.dim() == 4, "4D expected");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "channels_last expected");
    int B = (int)y.size(0), C = (int)y.size(1), G = (int)G_;
    long HW = y.size(2) * y.size(3);
    TORCH_CHECK(C % G == 0, "C%G");
    int CPG = C / G;
    auto wc = w.is_contiguous() ? w : w.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();

    torch::Tensor mean, rstd;
    compute_stats(y.data_ptr<float>(), B, C, HW, G, eps, y.options(), mean, rstd);

    auto stream = at::cuda::getCurrentCUDAStream();
    bool fast = (C % 4 == 0) && (CPG % 4 == 0);
    int C4 = C / 4;
    int T = 0, ppb = 0;
    if (fast) {
        if (C4 <= 256 && (256 % C4) == 0) { T = 256; ppb = 256 / C4; }
        else if (C4 <= 1024)              { T = C4;  ppb = 1; }
        else fast = false;
    }
    if (fast) {
        long gx = (HW + ppb - 1) / ppb;
        if (gx > 8192) gx = 8192;
        if (gx < 1) gx = 1;
        dim3 grid((unsigned)gx, (unsigned)B);
        gn_apply_nhwc_fast<<<grid, T, 0, stream>>>(
            y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            wc.data_ptr<float>(), bc.data_ptr<float>(), C4, HW, G, CPG, ppb);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        long total = (long)B * HW * (long)C;
        long gx = (total + 255) / 256;
        if (gx > 65535) gx = 65535;
        if (gx < 1) gx = 1;
        gn_apply_nhwc_gen<<<(unsigned)gx, 256, 0, stream>>>(
            y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            wc.data_ptr<float>(), bc.data_ptr<float>(), C, HW, G, CPG, total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// writes into a caller-provided contiguous NCHW destination (enables stream sharding)
void gn_silu_add_nchw_out(torch::Tensor y, torch::Tensor res, torch::Tensor out,
                          torch::Tensor w, torch::Tensor b,
                          double eps, int64_t G_)
{
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32, "fp32 cuda expected");
    TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast), "channels_last expected");
    TORCH_CHECK(res.is_contiguous(), "contiguous residual expected");
    TORCH_CHECK(out.is_contiguous(), "contiguous output expected");
    TORCH_CHECK(res.sizes() == y.sizes(), "shape mismatch");
    TORCH_CHECK(out.sizes() == y.sizes(), "out shape mismatch");
    int B = (int)y.size(0), C = (int)y.size(1), G = (int)G_;
    long H = y.size(2), W = y.size(3), HW = H * W;
    TORCH_CHECK(C % G == 0, "C%G");
    int CPG = C / G;
    auto wc = w.is_contiguous() ? w : w.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();

    torch::Tensor mean, rstd;
    compute_stats(y.data_ptr<float>(), B, C, HW, G, eps, y.options(), mean, rstd);

    long ntx = (HW + TILE - 1) / TILE;
    long gx  = ntx < GRID_CAP ? ntx : GRID_CAP;
    if (gx < 1) gx = 1;
    dim3 block(TILE, TROWS);
    dim3 grid((unsigned)gx, (unsigned)((C + TILE - 1) / TILE), (unsigned)B);
    auto stream = at::cuda::getCurrentCUDAStream();
    gn_apply_out_kernel<<<grid, block, 0, stream>>>(
        y.data_ptr<float>(), res.data_ptr<float>(), out.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        wc.data_ptr<float>(), bc.data_ptr<float>(), C, HW, G, CPG, ntx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_add_nchw(torch::Tensor y, torch::Tensor res,
                               torch::Tensor w, torch::Tensor b,
                               double eps, int64_t G_)
{
    auto out = torch::empty({y.size(0), y.size(1), y.size(2), y.size(3)}, res.options());
    gn_silu_add_nchw_out(y, res, out, w, b, eps, G_);
    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor nchw_to_nhwc(torch::Tensor x);
void gn_silu_nhwc_(torch::Tensor y, torch::Tensor w, torch::Tensor b, double eps, int64_t G_);
torch::Tensor gn_silu_add_nchw(torch::Tensor y, torch::Tensor res, torch::Tensor w,
                               torch::Tensor b, double eps, int64_t G_);
void gn_silu_add_nchw_out(torch::Tensor y, torch::Tensor res, torch::Tensor out,
                          torch::Tensor w, torch::Tensor b, double eps, int64_t G_);
'''

_ext = load_inline(
    name="sol002_gn_silu_res_fused_streams",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["nchw_to_nhwc", "gn_silu_nhwc_", "gn_silu_add_nchw", "gn_silu_add_nchw_out"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-gencode=arch=compute_120,code=sm_120",
    ],
    extra_ldflags=[""],
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

_NUM_GROUPS = 32
_NUM_STREAMS = 4
_STREAMS = None


def _get_streams():
    global _STREAMS
    if _STREAMS is None:
        _STREAMS = [torch.cuda.Stream() for _ in range(_NUM_STREAMS)]
    return _STREAMS


def _rec(t, s):
    try:
        t.record_stream(s)
    except Exception:
        pass


class ModelNew(nn.Module):
    """Granularity (C) fused GN/SiLU/residual/layout CUDA kernels + batch-sharded
    multi-stream pipelining so the DRAM-bound elementwise segment overlaps with the
    tensor-core-bound cuDNN convolutions."""

    def __init__(self):
        super().__init__()

    def _chain(self, xs, w1, w2, n1w, n1b, n2w, n2b, e, out_slice):
        xh = _ext.nchw_to_nhwc(xs)               # (b, H, W, C) contiguous
        xl = xh.permute(0, 3, 1, 2)              # (b, C, H, W), channels_last strides
        o = F.conv2d(xl, w1, None, 1, 1)
        if not o.is_contiguous(memory_format=torch.channels_last):
            o = o.contiguous(memory_format=torch.channels_last)
        _ext.gn_silu_nhwc_(o, n1w, n1b, e, _NUM_GROUPS)
        o2 = F.conv2d(o, w2, None, 1, 1)
        if not o2.is_contiguous(memory_format=torch.channels_last):
            o2 = o2.contiguous(memory_format=torch.channels_last)
        _ext.gn_silu_add_nchw_out(o2, xs, out_slice, n2w, n2b, e, _NUM_GROUPS)

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        # (plan 9) eps pulled to host before any stream work -> no sync inside pipeline
        e = float(eps.item()) if torch.is_tensor(eps) else float(eps)

        # (plan 4) shared read-only tensors materialised on the current stream
        xc = x if x.is_contiguous() else x.contiguous()

        w1 = conv1_weight
        if not w1.is_contiguous(memory_format=torch.channels_last):
            w1 = w1.contiguous(memory_format=torch.channels_last)
        w2 = conv2_weight
        if not w2.is_contiguous(memory_format=torch.channels_last):
            w2 = w2.contiguous(memory_format=torch.channels_last)
        n1w = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        n1b = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        n2w = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        n2b = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        B, C, H, W = xc.shape

        # (plan 5) gate: tiny / single-sample work keeps the original single-stream path
        nsh = 1 if (B < 2 or B * H * W < 2 * 8192) else min(B, _NUM_STREAMS)

        if nsh == 1:
            xh = _ext.nchw_to_nhwc(xc)
            xl = xh.permute(0, 3, 1, 2)
            o = F.conv2d(xl, w1, None, 1, 1)
            if not o.is_contiguous(memory_format=torch.channels_last):
                o = o.contiguous(memory_format=torch.channels_last)
            _ext.gn_silu_nhwc_(o, n1w, n1b, e, _NUM_GROUPS)
            o2 = F.conv2d(o, w2, None, 1, 1)
            if not o2.is_contiguous(memory_format=torch.channels_last):
                o2 = o2.contiguous(memory_format=torch.channels_last)
            return _ext.gn_silu_add_nchw(o2, xc, n2w, n2b, e, _NUM_GROUPS)

        streams = _get_streams()
        cur = torch.cuda.current_stream()

        ev0 = torch.cuda.Event()
        ev0.record(cur)

        # (plan 6) single output allocation on the current stream
        out = torch.empty_like(xc)

        used = []
        evs = []
        # (plan 7) per-shard chain on its own stream
        for i in range(nsh):
            n0 = (i * B) // nsh
            n1 = ((i + 1) * B) // nsh
            if n0 >= n1:
                continue
            s = streams[i]
            used.append(s)
            with torch.cuda.stream(s):
                s.wait_event(ev0)
                xs = xc[n0:n1]
                self._chain(xs, w1, w2, n1w, n1b, n2w, n2b, e, out[n0:n1])
                ev = torch.cuda.Event()
                ev.record(s)
                evs.append(ev)

        # (plan 8) allocator safety for tensors created on the caller stream but used
        # on the side streams, then join back onto the caller stream
        for s in used:
            for t in (xc, w1, w2, n1w, n1b, n2w, n2b, out):
                _rec(t, s)
        for ev in evs:
            cur.wait_event(ev)

        return out
