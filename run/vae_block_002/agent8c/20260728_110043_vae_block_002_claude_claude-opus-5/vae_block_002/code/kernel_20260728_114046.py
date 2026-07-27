import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

#define CPG   8          // channels per group (C=256, G=32)
#define VPP   2          // float4 vectors per pixel per group (CPG/4)
#define NTHR  256
#define PCHUNK 128       // pixels processed per shared-memory tile (NTHR/VPP)

__device__ __forceinline__ void warpReduce2(float &a, float &b) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, off);
        b += __shfl_down_sync(0xffffffffu, b, off);
    }
}

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// Tiled NCHW -> NHWC transpose into a caller-supplied channels_last buffer.
// grid = (ceil(HW/32), ceil(C/32), N), block = (32, 8)
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int C, int HW)
{
    __shared__ float tile[32][33];

    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;

    const float* ibase = in  + (long long)n * (long long)C * HW;
    float*       obase = out + (long long)n * (long long)HW * C;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const int c = c0 + ty + i;
        const int p = p0 + tx;
        float v = 0.f;
        if (c < C && p < HW) v = ibase[(long long)c * HW + p];
        tile[ty + i][tx] = v;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const int p = p0 + ty + i;
        const int c = c0 + tx;
        if (c < C && p < HW) obase[(long long)p * C + c] = tile[tx][ty + i];
    }
}

// ---------------------------------------------------------------------------
// Pass 1: partial sums / sums-of-squares over a pixel chunk of one (n,g) group.
// grid = (K, N*G), block = 256.   Input is channels-last (NHWC).
// ---------------------------------------------------------------------------
__global__ void gn_stats_kernel(const float* __restrict__ inp,
                                float* __restrict__ partial,
                                int HW, int C, int G, int chunk, int K)
{
    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    const float* base = inp + (long long)n * HW * C + (long long)g * CPG;

    const int tid     = threadIdx.x;
    const int j       = tid & (VPP - 1);
    const int p0      = tid / VPP;
    const int pstride = NTHR / VPP;

    float s = 0.f, ss = 0.f;
    for (int p = pstart + p0; p < pend; p += pstride) {
        const float4 v = *reinterpret_cast<const float4*>(base + (long long)p * C + j * 4);
        s  += v.x + v.y + v.z + v.w;
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    __shared__ float sa[NTHR / 32];
    __shared__ float sb[NTHR / 32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    warpReduce2(s, ss);
    if (lane == 0) { sa[wid] = s; sb[wid] = ss; }
    __syncthreads();
    if (wid == 0) {
        const int nw = NTHR / 32;
        s  = (lane < nw) ? sa[lane] : 0.f;
        ss = (lane < nw) ? sb[lane] : 0.f;
        warpReduce2(s, ss);
        if (lane == 0) {
            partial[((long long)ng * K + blk) * 2 + 0] = s;
            partial[((long long)ng * K + blk) * 2 + 1] = ss;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: finalize (mean, rstd) per (n,g).  grid = N*G, block = 32.
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   float* __restrict__ stats,
                                   int K, float invcount, float eps)
{
    const int ng = blockIdx.x;
    float s = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < K; i += 32) {
        s  += partial[((long long)ng * K + i) * 2 + 0];
        ss += partial[((long long)ng * K + i) * 2 + 1];
    }
    warpReduce2(s, ss);
    if (threadIdx.x == 0) {
        const float mean = s * invcount;
        float var = ss * invcount - mean * mean;
        if (var < 0.f) var = 0.f;
        stats[ng * 2 + 0] = mean;
        stats[ng * 2 + 1] = rsqrtf(var + eps);
    }
}

// ---------------------------------------------------------------------------
// Pass 3a: normalize * gamma + beta -> SiLU, NHWC in / NHWC out.
// grid = (K, N*G), block = 256.
// ---------------------------------------------------------------------------
__global__ void gn_silu_apply_nhwc(const float* __restrict__ inp,
                                   float* __restrict__ out,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ stats,
                                   int HW, int C, int G, int chunk)
{
    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const float mean = stats[ng * 2 + 0];
    const float rstd = stats[ng * 2 + 1];

    const int tid     = threadIdx.x;
    const int j       = tid & (VPP - 1);
    const int p0      = tid / VPP;
    const int pstride = NTHR / VPP;

    const float4 gm = *reinterpret_cast<const float4*>(gamma + g * CPG + j * 4);
    const float4 bt = *reinterpret_cast<const float4*>(beta  + g * CPG + j * 4);

    const float* ibase = inp + (long long)n * HW * C + (long long)g * CPG;
    float*       obase = out + (long long)n * HW * C + (long long)g * CPG;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    for (int p = pstart + p0; p < pend; p += pstride) {
        const long long off = (long long)p * C + j * 4;
        const float4 v = *reinterpret_cast<const float4*>(ibase + off);
        float4 o;
        o.x = silu_f((v.x - mean) * rstd * gm.x + bt.x);
        o.y = silu_f((v.y - mean) * rstd * gm.y + bt.y);
        o.z = silu_f((v.z - mean) * rstd * gm.z + bt.z);
        o.w = silu_f((v.w - mean) * rstd * gm.w + bt.w);
        *reinterpret_cast<float4*>(obase + off) = o;
    }
}

// ---------------------------------------------------------------------------
// Pass 3b: normalize -> SiLU -> + residual(NHWC), NHWC in / NCHW contiguous out.
// On-chip transpose through shared memory keeps both loads and stores coalesced.
// grid = (K, N*G), block = 256 (must equal NTHR).
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_apply_nchw(const float* __restrict__ inp,
                                       const float* __restrict__ res,
                                       float* __restrict__ out,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ stats,
                                       int HW, int C, int G, int chunk)
{
    __shared__ float sh[CPG * (PCHUNK + 1)];

    const int blk = blockIdx.x;
    const int ng  = blockIdx.y;
    const int n   = ng / G;
    const int g   = ng - n * G;

    const float mean = stats[ng * 2 + 0];
    const float rstd = stats[ng * 2 + 1];

    const int tid = threadIdx.x;
    const int j   = tid & (VPP - 1);   // which float4 inside the group
    const int pl  = tid / VPP;         // local pixel index (0..127)

    const float4 gm = *reinterpret_cast<const float4*>(gamma + g * CPG + j * 4);
    const float4 bt = *reinterpret_cast<const float4*>(beta  + g * CPG + j * 4);

    const float* ibase = inp + (long long)n * HW * C + (long long)g * CPG;
    const float* rbase = res + (long long)n * HW * C + (long long)g * CPG;

    const int pstart = blk * chunk;
    int pend = pstart + chunk;
    if (pend > HW) pend = HW;

    for (int p0 = pstart; p0 < pend; p0 += PCHUNK) {
        const int p = p0 + pl;
        if (p < pend) {
            const long long off = (long long)p * C + j * 4;
            const float4 v = *reinterpret_cast<const float4*>(ibase + off);
            const float4 r = *reinterpret_cast<const float4*>(rbase + off);
            const int c0 = j * 4;
            sh[(c0 + 0) * (PCHUNK + 1) + pl] = silu_f((v.x - mean) * rstd * gm.x + bt.x) + r.x;
            sh[(c0 + 1) * (PCHUNK + 1) + pl] = silu_f((v.y - mean) * rstd * gm.y + bt.y) + r.y;
            sh[(c0 + 2) * (PCHUNK + 1) + pl] = silu_f((v.z - mean) * rstd * gm.z + bt.z) + r.z;
            sh[(c0 + 3) * (PCHUNK + 1) + pl] = silu_f((v.w - mean) * rstd * gm.w + bt.w) + r.w;
        }
        __syncthreads();

        const int cnt = (pend - p0) < PCHUNK ? (pend - p0) : PCHUNK;
#pragma unroll
        for (int it = 0; it < (CPG * PCHUNK) / NTHR; ++it) {
            const int idx = tid + it * NTHR;
            const int c   = idx / PCHUNK;
            const int lp  = idx - c * PCHUNK;
            if (lp < cnt) {
                const long long oidx =
                    ((long long)(n * C + g * CPG + c)) * HW + (long long)(p0 + lp);
                out[oidx] = sh[c * (PCHUNK + 1) + lp];
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static inline void pick_chunk(long long NG, int HW, int &chunk, int &K)
{
    const long long target_blocks = 2048;
    int k = (int)((target_blocks + NG - 1) / NG);
    if (k < 1) k = 1;
    int per = (HW + k - 1) / k;
    chunk = ((per + PCHUNK - 1) / PCHUNK) * PCHUNK;
    if (chunk < PCHUNK) chunk = PCHUNK;
    K = (HW + chunk - 1) / chunk;
}

static void run_stats(const torch::Tensor &inp, torch::Tensor &stats,
                      int N, int C, int G, int HW, int chunk, int K, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const long long NG = (long long)N * G;
    auto partial = torch::empty({NG * K * 2}, inp.options());

    dim3 grid(K, (unsigned)NG);
    gn_stats_kernel<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), partial.data_ptr<float>(), HW, C, G, chunk, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float invcount = 1.0f / (float)((long long)CPG * HW);
    gn_finalize_kernel<<<(unsigned)NG, 32, 0, stream>>>(
        partial.data_ptr<float>(), stats.data_ptr<float>(), K, invcount, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nchw_to_nhwc_into(torch::Tensor src, torch::Tensor dst)
{
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32 &&
                dst.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(src.dim() == 4 && dst.dim() == 4, "4D only");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous NCHW");
    TORCH_CHECK(dst.is_contiguous(at::MemoryFormat::ChannelsLast), "dst must be channels_last");
    TORCH_CHECK(src.sizes() == dst.sizes(), "size mismatch");

    const int N = (int)src.size(0), C = (int)src.size(1);
    const int HW = (int)(src.size(2) * src.size(3));

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blk(32, 8);
    dim3 grd((unsigned)((HW + 31) / 32), (unsigned)((C + 31) / 32), (unsigned)N);
    nchw2nhwc_kernel<<<grd, blk, 0, stream>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), C, HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gn_silu_nhwc(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, double eps)
{
    TORCH_CHECK(inp.is_cuda(), "input must be CUDA");
    TORCH_CHECK(inp.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(inp.dim() == 4, "4D only");
    TORCH_CHECK(inp.is_contiguous(at::MemoryFormat::ChannelsLast), "need channels_last");

    const int N = (int)inp.size(0), C = (int)inp.size(1);
    const int H = (int)inp.size(2), W = (int)inp.size(3);
    const int G = 32, HW = H * W;
    TORCH_CHECK(C % G == 0 && (C / G) == CPG, "unsupported channel/group config");

    auto out = torch::empty_like(inp);           // preserves channels_last
    auto stats = torch::empty({(long long)N * G * 2}, inp.options());

    int chunk = 0, K = 0;
    pick_chunk((long long)N * G, HW, chunk, K);
    run_stats(inp, stats, N, C, G, HW, chunk, K, eps);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(K, (unsigned)((long long)N * G));
    gn_silu_apply_nhwc<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        stats.data_ptr<float>(), HW, C, G, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Out-variant: writes into a caller-supplied contiguous NCHW tensor.
// Residual is channels_last (NHWC), consumed on the coalesced load path.
void gn_silu_res_nchw_into(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, torch::Tensor res,
                           torch::Tensor out, double eps)
{
    TORCH_CHECK(inp.is_cuda() && res.is_cuda() && out.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(inp.scalar_type() == torch::kFloat32 &&
                res.scalar_type() == torch::kFloat32 &&
                out.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(inp.is_contiguous(at::MemoryFormat::ChannelsLast), "need channels_last input");
    TORCH_CHECK(res.is_contiguous(at::MemoryFormat::ChannelsLast), "residual must be channels_last");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous NCHW");
    TORCH_CHECK(inp.sizes() == res.sizes() && inp.sizes() == out.sizes(), "size mismatch");

    const int N = (int)inp.size(0), C = (int)inp.size(1);
    const int H = (int)inp.size(2), W = (int)inp.size(3);
    const int G = 32, HW = H * W;
    TORCH_CHECK(C % G == 0 && (C / G) == CPG, "unsupported channel/group config");

    auto stats = torch::empty({(long long)N * G * 2}, inp.options());

    int chunk = 0, K = 0;
    pick_chunk((long long)N * G, HW, chunk, K);
    run_stats(inp, stats, N, C, G, HW, chunk, K, eps);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(K, (unsigned)((long long)N * G));
    gn_silu_res_apply_nchw<<<grid, NTHR, 0, stream>>>(
        inp.data_ptr<float>(), res.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        stats.data_ptr<float>(), HW, C, G, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

_CPP_SRC = r'''
void nchw_to_nhwc_into(torch::Tensor src, torch::Tensor dst);
torch::Tensor gn_silu_nhwc(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, double eps);
void gn_silu_res_nchw_into(torch::Tensor inp, torch::Tensor gamma,
                           torch::Tensor beta, torch::Tensor res,
                           torch::Tensor out, double eps);
'''

_ext = load_inline(
    name="vae_resblock_gn_silu_fused_graph2s",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["nchw_to_nhwc_into", "gn_silu_nhwc", "gn_silu_res_nchw_into"],
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

_CL = torch.channels_last


class ModelNew(nn.Module):
    """
    GroupNorm+SiLU (+residual, +NHWC->NCHW transpose) fused in custom CUDA kernels,
    convs on cuDNN (channels_last), whole forward captured as a TWO-STREAM,
    batch-chunked CUDA-Graph DAG so memory-bound GN overlaps compute-bound conv.
    """

    def __init__(self):
        super().__init__()
        self._graph = None
        self._captured = False
        self._graph_disabled = False
        self._recaptures = 0
        self._sig = None
        self._eps = None
        self._nchunk = 1

        # static buffers (allocated at capture time, address-stable)
        self._s_xc = None
        self._s_out = None
        self._s_w1c = None
        self._s_w2c = None
        self._s_g1 = None
        self._s_b1 = None
        self._s_g2 = None
        self._s_b2 = None
        self._ptrs = None

        # streams / events created before capture, never inside it
        self._gstream = None
        self._s1 = None
        self._s2 = None
        self._evA = None
        if torch.cuda.is_available():
            try:
                self._gstream = torch.cuda.Stream()
                self._s1 = torch.cuda.Stream()
                self._s2 = torch.cuda.Stream()
                self._evA = torch.cuda.Event()
            except Exception:
                self._gstream = self._s1 = self._s2 = self._evA = None

    # ------------------------------------------------------------------ eager
    def _eager(self, x, w1, g1, b1, w2, g2, b2, eps_f):
        C = x.size(1)
        G = 32
        if (not x.is_cuda) or x.dtype != torch.float32 or x.dim() != 4 \
           or C % G != 0 or (C // G) != 8:
            res = x if x.is_contiguous() else x.contiguous()
            out = F.conv2d(x, w1, None, 1, 1)
            out = F.silu(F.group_norm(out, G, g1, b1, eps_f))
            out = F.conv2d(out, w2, None, 1, 1)
            out = F.silu(F.group_norm(out, G, g2, b2, eps_f))
            return out + res

        xc = x if x.is_contiguous(memory_format=_CL) else x.contiguous(memory_format=_CL)
        w1c = w1 if w1.is_contiguous(memory_format=_CL) else w1.contiguous(memory_format=_CL)
        w2c = w2 if w2.is_contiguous(memory_format=_CL) else w2.contiguous(memory_format=_CL)
        g1c = g1 if g1.is_contiguous() else g1.contiguous()
        b1c = b1 if b1.is_contiguous() else b1.contiguous()
        g2c = g2 if g2.is_contiguous() else g2.contiguous()
        b2c = b2 if b2.is_contiguous() else b2.contiguous()

        y = F.conv2d(xc, w1c, None, 1, 1)
        if not y.is_contiguous(memory_format=_CL):
            y = y.contiguous(memory_format=_CL)
        y = _ext.gn_silu_nhwc(y, g1c, b1c, eps_f)
        y = F.conv2d(y, w2c, None, 1, 1)
        if not y.is_contiguous(memory_format=_CL):
            y = y.contiguous(memory_format=_CL)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _ext.gn_silu_res_nchw_into(y, g2c, b2c, xc, out, eps_f)
        return out

    # ------------------------------------------------------- captured schedule
    def _run_chunk(self, xs, outs, eps_f, ev=None):
        y = F.conv2d(xs, self._s_w1c, None, 1, 1)
        if ev is not None:
            ev.record()
        if not y.is_contiguous(memory_format=_CL):
            y = y.contiguous(memory_format=_CL)
        y = _ext.gn_silu_nhwc(y, self._s_g1, self._s_b1, eps_f)
        y = F.conv2d(y, self._s_w2c, None, 1, 1)
        if not y.is_contiguous(memory_format=_CL):
            y = y.contiguous(memory_format=_CL)
        _ext.gn_silu_res_nchw_into(y, self._s_g2, self._s_b2, xs, outs, eps_f)

    def _run_pipelined(self, eps_f):
        if self._nchunk == 1:
            self._run_chunk(self._s_xc, self._s_out, eps_f)
            return

        n = self._s_xc.size(0)
        h = n // 2
        cur = torch.cuda.current_stream()
        self._s1.wait_stream(cur)
        self._s2.wait_stream(cur)

        with torch.cuda.stream(self._s1):
            self._run_chunk(self._s_xc[0:h], self._s_out[0:h], eps_f, ev=self._evA)

        self._s2.wait_event(self._evA)
        with torch.cuda.stream(self._s2):
            self._run_chunk(self._s_xc[h:n], self._s_out[h:n], eps_f)

        cur.wait_stream(self._s1)
        cur.wait_stream(self._s2)

    # ------------------------------------------------------------------ capture
    def _capture(self, x, w1, g1, b1, w2, g2, b2, eps_f, sig):
        try:
            if self._s1 is None or self._s2 is None or self._gstream is None:
                self._gstream = torch.cuda.Stream()
                self._s1 = torch.cuda.Stream()
                self._s2 = torch.cuda.Stream()
            if self._evA is None:
                self._evA = torch.cuda.Event()

            N, C, H, W = int(x.size(0)), int(x.size(1)), int(x.size(2)), int(x.size(3))
            dev = x.device

            self._s_xc = torch.empty((N, C, H, W), dtype=torch.float32, device=dev,
                                     memory_format=_CL)
            self._s_out = torch.empty((N, C, H, W), dtype=torch.float32, device=dev)

            self._s_w1c = w1.detach().clone().contiguous(memory_format=_CL)
            self._s_w2c = w2.detach().clone().contiguous(memory_format=_CL)
            self._s_g1 = g1.detach().clone().contiguous()
            self._s_b1 = b1.detach().clone().contiguous()
            self._s_g2 = g2.detach().clone().contiguous()
            self._s_b2 = b2.detach().clone().contiguous()
            self._ptrs = (w1.data_ptr(), w2.data_ptr(), g1.data_ptr(),
                          b1.data_ptr(), g2.data_ptr(), b2.data_ptr())

            self._nchunk = 2 if (N >= 2 and N % 2 == 0) else 1

            # seed static input with real data so warmup runs on valid values
            src = x if x.is_contiguous() else x.contiguous()
            _ext.nchw_to_nhwc_into(src, self._s_xc)

            # warmup on side stream (cuDNN algo selection + allocator caching)
            self._gstream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._gstream):
                for _ in range(3):
                    self._run_pipelined(eps_f)
            torch.cuda.current_stream().wait_stream(self._gstream)
            torch.cuda.synchronize()

            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                self._run_pipelined(eps_f)
            torch.cuda.synchronize()

            if self._captured:
                self._recaptures += 1
            self._graph = gph
            self._captured = True
            self._sig = sig
            self._eps = eps_f
            return True
        except Exception:
            self._graph = None
            self._captured = False
            return False

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self,
                x,
                conv1_weight,
                norm1_weight,
                norm1_bias,
                conv2_weight,
                norm2_weight,
                norm2_bias,
                eps):
        eps_f = eps if isinstance(eps, float) else float(eps)

        usable = (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                  and x.size(1) % 32 == 0 and (x.size(1) // 32) == 8
                  and not self.training)
        if usable:
            try:
                if torch.cuda.is_current_stream_capturing():
                    usable = False
            except Exception:
                pass

        if (not usable) or self._graph_disabled:
            return self._eager(x, conv1_weight, norm1_weight, norm1_bias,
                               conv2_weight, norm2_weight, norm2_bias, eps_f)

        sig = (tuple(x.shape), tuple(conv1_weight.shape), tuple(conv2_weight.shape),
               eps_f, x.device.index)

        if (not self._captured) or sig != self._sig:
            if self._captured and self._recaptures >= 2:
                self._graph_disabled = True
                return self._eager(x, conv1_weight, norm1_weight, norm1_bias,
                                   conv2_weight, norm2_weight, norm2_bias, eps_f)
            ok = self._capture(x, conv1_weight, norm1_weight, norm1_bias,
                               conv2_weight, norm2_weight, norm2_bias, eps_f, sig)
            if not ok:
                self._graph_disabled = True
                return self._eager(x, conv1_weight, norm1_weight, norm1_bias,
                                   conv2_weight, norm2_weight, norm2_bias, eps_f)

        ptrs = (conv1_weight.data_ptr(), conv2_weight.data_ptr(),
                norm1_weight.data_ptr(), norm1_bias.data_ptr(),
                norm2_weight.data_ptr(), norm2_bias.data_ptr())
        if ptrs != self._ptrs:
            self._s_w1c.copy_(conv1_weight)
            self._s_w2c.copy_(conv2_weight)
            self._s_g1.copy_(norm1_weight)
            self._s_b1.copy_(norm1_bias)
            self._s_g2.copy_(norm2_weight)
            self._s_b2.copy_(norm2_bias)
            self._ptrs = ptrs

        src = x if x.is_contiguous() else x.contiguous()
        _ext.nchw_to_nhwc_into(src, self._s_xc)
        self._graph.replay()
        return self._s_out
