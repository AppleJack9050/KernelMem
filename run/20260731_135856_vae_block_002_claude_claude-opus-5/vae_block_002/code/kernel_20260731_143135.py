# =============================================================================
# ModelNew: fused VAE residual block
#   Conv3x3 -> GroupNorm(32) -> SiLU -> Conv3x3 -> GroupNorm(32) -> SiLU -> +x
#
# HEADER (required):
# 1) GRANULARITY: (C) — fuse many ops into one/few custom CUDA kernels.
#
# 2) OPS REPLACED BY CUSTOM CUDA:
#      - the implicit NCHW<->NHWC layout conversions cuDNN performs internally
#      - group_norm  (moments + normalize)  x2
#      - silu                               x2
#      - residual add                       x1
#      - the final NHWC->NCHW layout restore (folded into the last kernel)
#
# 3) FUSION MAP (all custom kernels launched from the load_inline extension):
#      K1 nchw2nhwc_kernel      : NCHW->NHWC tiled transpose of the input once.
#      K2 gn_partial_kernel     : per-(image, pixel-chunk) partial sum/sumsq.
#      K3 gn_finalize_kernel    : partials -> mean/rstd -> per-(n,c) affine pair.
#      K4 gn_silu_inplace_kernel: normalize + affine + SiLU, float4, in place.
#      K5 gn_silu_res_nchw_kernel: normalize + affine + SiLU + residual add +
#                                 NHWC->NCHW transpose in one tiled kernel.
#
# 4) NEW (this revision): batch-chunked two-stream pipelining. The whole block
#    splits exactly along N (GN is per-(n,group), conv is per-image), so each
#    N-chunk runs the full chain on one of two alternating CUDA streams. The
#    DRAM-bound custom kernels of chunk i then co-execute with the compute-bound
#    tensor-core convolution of chunk i+1.
#
# 5) LEFT IN PYTORCH / VENDOR:
#      - the two 3x3 convolutions (at::cudnn_convolution, benchmark=true).
#
# PRECISION: everything stays float32; reductions accumulate in float32.
#            TF32 is used only inside conv, exactly as the reference does.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <exception>

#if defined(__has_include)
#  if __has_include(<torch/version.h>)
#    include <torch/version.h>
#  endif
#endif

#define TILE 32

// ---------------------------------------------------------------------------
// K1: NCHW -> NHWC tiled transpose (32 pixels x 32 channels per block)
// ---------------------------------------------------------------------------
__global__ void nchw2nhwc_kernel(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 int HW, int C)
{
    __shared__ float tile[TILE][TILE + 1];
    const int p0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int n  = blockIdx.z;
    const int tid = threadIdx.x;              // 256 threads

    // read phase: NCHW coalesced along pixels
    const int lp  = tid & 31;
    const int lc0 = tid >> 5;                 // 0..7
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int c = c0 + lc0 + r * 8;
        const int p = p0 + lp;
        if (p < HW) {
            tile[lp][lc0 + r * 8] = src[((long)n * C + c) * (long)HW + p];
        }
    }
    __syncthreads();

    // write phase: NHWC coalesced along channels
    const int lc  = tid & 31;
    const int lp0 = tid >> 5;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int p = p0 + lp0 + r * 8;
        if (p < HW) {
            dst[((long)n * HW + p) * (long)C + c0 + lc] = tile[lp0 + r * 8][lc];
        }
    }
}

// ---------------------------------------------------------------------------
// K2: partial moments over a pixel chunk, for ALL groups at once (NHWC).
// ---------------------------------------------------------------------------
__global__ void gn_partial_kernel(const float* __restrict__ y,
                                  float* __restrict__ partial,   // [B][P][G][2]
                                  int HW, int C, int P, int pixPerBlock)
{
    const int pb = blockIdx.x;
    const int n  = blockIdx.y;
    const int pstart = pb * pixPerBlock;
    if (pstart >= HW) return;
    const int pend = min(pstart + pixPerBlock, HW);

    const int c = threadIdx.x;
    const float* ptr = y + (long)n * HW * C + (long)pstart * C + c;

    float s = 0.f, s2 = 0.f;
    for (int p = pstart; p < pend; ++p) {
        float v = *ptr;
        ptr += C;
        s  += v;
        s2 += v * v;
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off, 8);
        s2 += __shfl_down_sync(0xffffffffu, s2, off, 8);
    }
    if ((c & 7) == 0) {
        const int g = c >> 3;
        const int G = C >> 3;
        float* q = partial + ((((long)n * P + pb) * G) + g) * 2;
        q[0] = s;
        q[1] = s2;
    }
}

__device__ __forceinline__ void block_reduce2(float& s, float& s2)
{
    __shared__ float ws[32], ws2[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off);
        s2 += __shfl_down_sync(0xffffffffu, s2, off);
    }
    if (lane == 0) { ws[wid] = s; ws2[wid] = s2; }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = (int)(blockDim.x >> 5);
        float a = 0.f, b = 0.f;
        for (int i = 0; i < nw; ++i) { a += ws[i]; b += ws2[i]; }
        ws[0] = a; ws2[0] = b;
    }
    __syncthreads();
    s = ws[0]; s2 = ws2[0];
}

// ---------------------------------------------------------------------------
// K3: reduce partials -> per-(n,c) affine pair
// ---------------------------------------------------------------------------
__global__ void gn_finalize_kernel(const float* __restrict__ partial,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ A,
                                   float* __restrict__ Bs,
                                   int P, int C, int G, int CPG,
                                   float eps, float invCount)
{
    const int ng = blockIdx.x;
    const int n  = ng / G;
    const int g  = ng - n * G;

    float s = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const float* q = partial + ((((long)n * P + i) * G) + g) * 2;
        s  += q[0];
        s2 += q[1];
    }
    block_reduce2(s, s2);

    const float mean = s * invCount;
    float var = s2 * invCount - mean * mean;
    var = fmaxf(var, 0.f);
    const float rstd = rsqrtf(var + eps);

    if ((int)threadIdx.x < CPG) {
        const int c = g * CPG + (int)threadIdx.x;
        const float a = gamma[c] * rstd;
        A[(long)n * C + c]  = a;
        Bs[(long)n * C + c] = beta[c] - mean * a;
    }
}

// ---------------------------------------------------------------------------
// K4: normalize + affine + SiLU, in place, NHWC, float4
// ---------------------------------------------------------------------------
__global__ void gn_silu_inplace_kernel(float* __restrict__ y,
                                       const float* __restrict__ A,
                                       const float* __restrict__ Bs,
                                       long HWC4, int C4, int C)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HWC4) return;
    const int n = blockIdx.y;

    float4* yp = reinterpret_cast<float4*>(y) + (long)n * HWC4 + i;
    const int c = (int)(i % (long)C4) * 4;
    const float* a = A  + (long)n * C + c;
    const float* b = Bs + (long)n * C + c;

    float4 v = *yp;
    float t0 = v.x * a[0] + b[0];
    float t1 = v.y * a[1] + b[1];
    float t2 = v.z * a[2] + b[2];
    float t3 = v.w * a[3] + b[3];
    v.x = t0 / (1.f + __expf(-t0));
    v.y = t1 / (1.f + __expf(-t1));
    v.z = t2 / (1.f + __expf(-t2));
    v.w = t3 / (1.f + __expf(-t3));
    *yp = v;
}

// ---------------------------------------------------------------------------
// K5: normalize + affine + SiLU + residual + NHWC->NCHW transpose
// ---------------------------------------------------------------------------
__global__ void gn_silu_res_nchw_kernel(const float* __restrict__ y,   // NHWC
                                        const float* __restrict__ res, // NCHW
                                        const float* __restrict__ A,
                                        const float* __restrict__ Bs,
                                        float* __restrict__ out,       // NCHW
                                        int HW, int C)
{
    __shared__ float tile[TILE][TILE + 1];
    const int p0 = blockIdx.x * TILE;
    const int c0 = blockIdx.y * TILE;
    const int n  = blockIdx.z;
    const int tid = threadIdx.x;

    // load phase: NHWC coalesced along channels
    const int lc  = tid & 31;
    const int lp0 = tid >> 5;
    const int cg  = c0 + lc;
    const float av = A[(long)n * C + cg];
    const float bv = Bs[(long)n * C + cg];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int p = p0 + lp0 + r * 8;
        if (p < HW) {
            float t = y[((long)n * HW + p) * (long)C + cg] * av + bv;
            t = t / (1.f + __expf(-t));
            tile[lp0 + r * 8][lc] = t;
        }
    }
    __syncthreads();

    // store phase: NCHW coalesced along pixels
    const int lp  = tid & 31;
    const int lc0 = tid >> 5;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int c = c0 + lc0 + r * 8;
        const int p = p0 + lp;
        if (p < HW) {
            const long o = ((long)n * C + c) * (long)HW + p;
            out[o] = tile[lp][lc0 + r * 8] + res[o];
        }
    }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------

// ---- one-time global cuDNN benchmark / TF32 switches (plan item 9) ---------
static void init_cudnn_flags()
{
    static bool once = [](){
        at::globalContext().setBenchmarkCuDNN(true);
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        at::globalContext().setBenchmarkLimitCuDNN(10);
#endif
        at::globalContext().setAllowTF32CuDNN(true);
        return true;
    }();
    (void)once;
}

// sticky flag: if the direct cuDNN entry point is unavailable/fails once,
// stop retrying and use at::conv2d for the rest of the process lifetime.
static bool g_cudnn_direct_ok = true;

// optional second variant (plain KCRS filter), env-selected once.
static bool use_kcrs_filter()
{
    static bool v = [](){
        const char* e = std::getenv("VAE_CONV_W_KCRS");
        return (e != nullptr && e[0] == '1');
    }();
    return v;
}

// ---- benchmarked cuDNN convolution with safe fallback (plan item 9) --------
static at::Tensor run_conv(const at::Tensor& in, const at::Tensor& w)
{
    const std::vector<int64_t> pad{1, 1}, str{1, 1}, dil{1, 1};
    if (g_cudnn_direct_ok) {
        try {
            return at::cudnn_convolution(in, w,
                                         at::IntArrayRef(pad),   // padding
                                         at::IntArrayRef(str),   // stride
                                         at::IntArrayRef(dil),   // dilation
                                         /*groups=*/1,
                                         /*benchmark=*/true,
                                         /*deterministic=*/false,
                                         /*allow_tf32=*/true);
        } catch (const std::exception&) {
            g_cudnn_direct_ok = false;   // fall through to the generic path
        }
    }
    return at::conv2d(in, w, at::Tensor(),
                      at::IntArrayRef(str), at::IntArrayRef(pad),
                      at::IntArrayRef(dil), 1);
}

static inline int calc_pix_per_block(int HW, int B)
{
    int target = 1360 / std::max(1, B);
    if (target < 1) target = 1;
    int ppb = (HW + target - 1) / target;
    if (ppb < 16) ppb = 16;
    if (ppb > HW) ppb = HW;
    return ppb;
}

static void run_group_norm_stats(const float* y, float* partial,
                                 const at::Tensor& gamma, const at::Tensor& beta,
                                 float* A, float* Bs,
                                 int B, int C, int HW, int G, int CPG,
                                 int P, int ppb, float eps,
                                 cudaStream_t stream)
{
    dim3 gridp(P, B);
    gn_partial_kernel<<<gridp, C, 0, stream>>>(y, partial, HW, C, P, ppb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float invCount = 1.0f / (float)((long)HW * CPG);
    gn_finalize_kernel<<<B * G, 128, 0, stream>>>(
        partial, gamma.data_ptr<float>(), beta.data_ptr<float>(),
        A, Bs, P, C, G, CPG, eps, invCount);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// plan item 2: the full chain for ONE batch chunk. Everything (ATen conv
// allocations + the 5 custom kernels) runs on the CURRENT stream, so the
// caller only has to install a CUDAStreamGuard.
// ---------------------------------------------------------------------------
static void run_chunk(const at::Tensor& xc_c,        // NCHW slice  [Bc,C,H,W]
                      at::Tensor out_c,              // NCHW slice  [Bc,C,H,W]
                      const at::Tensor& w1c, const at::Tensor& w2c,
                      const at::Tensor& g1, const at::Tensor& b1,
                      const at::Tensor& g2, const at::Tensor& b2,
                      int Bc, int C, int H, int W, int HW,
                      int G, int CPG, int P, int ppb, float eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts   = xc_c.options();

    // ---- K1: layout transform once (plan item 6/8: per-chunk buffer) ----
    auto xn = at::empty({Bc, C, H, W},
                        opts.memory_format(at::MemoryFormat::ChannelsLast));
    {
        dim3 grid((HW + TILE - 1) / TILE, C / TILE, Bc);
        nchw2nhwc_kernel<<<grid, 256, 0, stream>>>(
            xc_c.data_ptr<float>(), xn.data_ptr<float>(), HW, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv1 (vendor cuDNN, NHWC, benchmarked engine) ----------------
    auto y = run_conv(xn, w1c);
    if (!y.is_contiguous(at::MemoryFormat::ChannelsLast))
        y = y.contiguous(at::MemoryFormat::ChannelsLast);

    auto partial = at::empty({(long)Bc * P * G * 2}, opts);
    auto A  = at::empty({(long)Bc * C}, opts);
    auto Bs = at::empty({(long)Bc * C}, opts);

    // ---- K2/K3/K4: groupnorm1 + silu (in place, NHWC) -------------------
    run_group_norm_stats(y.data_ptr<float>(), partial.data_ptr<float>(),
                         g1, b1, A.data_ptr<float>(), Bs.data_ptr<float>(),
                         Bc, C, HW, G, CPG, P, ppb, eps, stream);
    {
        const long HWC4 = (long)HW * (C / 4);
        const int C4 = C / 4;
        const int threads = 256;
        dim3 grid((unsigned)((HWC4 + threads - 1) / threads), Bc);
        gn_silu_inplace_kernel<<<grid, threads, 0, stream>>>(
            y.data_ptr<float>(), A.data_ptr<float>(), Bs.data_ptr<float>(),
            HWC4, C4, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (vendor cuDNN, NHWC, benchmarked engine) ----------------
    auto y2 = run_conv(y, w2c);
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    // ---- K2/K3/K5: groupnorm2 + silu + residual + relayout --------------
    run_group_norm_stats(y2.data_ptr<float>(), partial.data_ptr<float>(),
                         g2, b2, A.data_ptr<float>(), Bs.data_ptr<float>(),
                         Bc, C, HW, G, CPG, P, ppb, eps, stream);
    {
        dim3 grid((HW + TILE - 1) / TILE, C / TILE, Bc);
        gn_silu_res_nchw_kernel<<<grid, 256, 0, stream>>>(
            y2.data_ptr<float>(), xc_c.data_ptr<float>(),
            A.data_ptr<float>(), Bs.data_ptr<float>(),
            out_c.data_ptr<float>(), HW, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// plan item 6: mark a caller-owned / main-stream tensor as used on a side
// stream so the caching allocator cannot recycle it too early.
static inline void rec_stream(const at::Tensor& t, c10::cuda::CUDAStream s)
{
    if (t.defined() && t.has_storage()) {
        c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), s);
    }
}

torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                             torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                             double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    init_cudnn_flags();   // plan item 9

    auto xc = x.is_contiguous() ? x : x.contiguous();
    const int B  = (int)xc.size(0);
    const int C  = (int)xc.size(1);
    const int H  = (int)xc.size(2);
    const int W  = (int)xc.size(3);
    const int HW = H * W;
    const int G   = 32;
    const int CPG = C / G;
    TORCH_CHECK(C == 256 && CPG == 8, "specialised for C=256, groups=32");

    auto main_stream = at::cuda::getCurrentCUDAStream();
    auto opts = xc.options();

    // filter layout (cached channels_last relayout done in Python; the
    // contiguous() call here is a no-op when already in that format)
    at::Tensor w1c, w2c;
    if (use_kcrs_filter()) {
        w1c = w1.contiguous();
        w2c = w2.contiguous();
    } else {
        w1c = w1.contiguous(at::MemoryFormat::ChannelsLast);
        w2c = w2.contiguous(at::MemoryFormat::ChannelsLast);
    }

    // ---- plan item 1: chunk planner -----------------------------------
    int nchunk = 1, cb = B;
    if (B >= 2 && (long)B * (long)HW >= 32768L) {
        cb = std::max(1, B / 4);
        while ((B % cb) != 0) --cb;
        nchunk = B / cb;
        if (nchunk > 4) {
            cb = (B + 3) / 4;
            while (B % cb) ++cb;
            nchunk = B / cb;
        }
    }

    // ppb/P computed from the FULL batch so the per-image reduction order is
    // bit-identical to the single-stream path (plan item 2).
    const int ppb = calc_pix_per_block(HW, B);
    const int P   = (HW + ppb - 1) / ppb;

    auto out = at::empty({B, C, H, W}, opts);

    // ---- plan item 10: guard rails ------------------------------------
    if (nchunk > 1) {
        if (!xc.narrow(0, 0, cb).is_contiguous() ||
            !out.narrow(0, 0, cb).is_contiguous()) {
            nchunk = 1;
            cb = B;
        }
    }

    if (nchunk == 1) {
        // exactly today's single-stream code path
        run_chunk(xc, out, w1c, w2c, g1, b1, g2, b2,
                  B, C, H, W, HW, G, CPG, P, ppb, (float)eps);
        return out;
    }

    // ---- plan item 3: two pool streams, created once -------------------
    static std::array<c10::cuda::CUDAStream, 2> side_streams{
        {c10::cuda::getStreamFromPool(), c10::cuda::getStreamFromPool()}};
    const int nstreams = std::min(2, nchunk);

    // ---- plan item 4: fork ---------------------------------------------
    {
        at::cuda::CUDAEvent start_ev;
        start_ev.record(main_stream);
        for (int k = 0; k < nstreams; ++k) start_ev.block(side_streams[k]);
    }

    // ---- plan item 6: cross-stream lifetime bookkeeping ----------------
    for (int k = 0; k < nstreams; ++k) {
        rec_stream(out,  side_streams[k]);
        rec_stream(xc,   side_streams[k]);
        rec_stream(w1c,  side_streams[k]);
        rec_stream(w2c,  side_streams[k]);
        rec_stream(g1,   side_streams[k]);
        rec_stream(b1,   side_streams[k]);
        rec_stream(g2,   side_streams[k]);
        rec_stream(b2,   side_streams[k]);
    }

    // ---- plan items 2 + 5: per-chunk chain on alternating streams ------
    for (int ci = 0; ci < nchunk; ++ci) {
        c10::cuda::CUDAStreamGuard guard(side_streams[ci & 1]);
        auto xc_c  = xc.narrow(0, (long)ci * cb, cb);
        auto out_c = out.narrow(0, (long)ci * cb, cb);
        run_chunk(xc_c, out_c, w1c, w2c, g1, b1, g2, b2,
                  cb, C, H, W, HW, G, CPG, P, ppb, (float)eps);
    }

    // ---- plan item 7: join --------------------------------------------
    for (int k = 0; k < nstreams; ++k) {
        at::cuda::CUDAEvent e;
        e.record(side_streams[k]);
        e.block(main_stream);
    }

    return out;
}
'''

_CPP_SRC = r'''
torch::Tensor fused_resblock(torch::Tensor x,
                             torch::Tensor w1, torch::Tensor g1, torch::Tensor b1,
                             torch::Tensor w2, torch::Tensor g2, torch::Tensor b2,
                             double eps);
'''

_ext = load_inline(
    name="vae_resblock_fused_v3_pipe",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["fused_resblock"],
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
    """See module-level header comment for granularity / fusion plan (level C)."""

    def __init__(self):
        super().__init__()
        self._ext = _ext
        # identity-keyed channels_last weight cache. The source tensor is stored
        # alongside the relayout so its id() stays unique and its storage cannot
        # be recycled under a different tensor.
        self._cl_cache = {}

    def _cl_weight(self, w):
        key = id(w)
        ent = self._cl_cache.get(key)
        if ent is not None and ent[0] is w:
            return ent[1]
        wc = w.contiguous(memory_format=torch.channels_last)
        self._cl_cache[key] = (w, wc)
        return wc

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
        if torch.is_tensor(eps):
            eps_f = float(eps.item())
        else:
            eps_f = float(eps)

        use_fast = (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.size(1) == 256
            and conv1_weight.dtype == torch.float32
            and conv2_weight.dtype == torch.float32
            and conv1_weight.shape[-2:] == (3, 3)
            and conv2_weight.shape[-2:] == (3, 3)
        )

        if use_fast:
            w1c = self._cl_weight(conv1_weight)
            w2c = self._cl_weight(conv2_weight)
            return self._ext.fused_resblock(
                x, w1c, norm1_weight, norm1_bias,
                w2c, norm2_weight, norm2_bias, eps_f)

        # ---- reference fallback (non-CUDA / unsupported dtype or shape) ----
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=eps_f)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=eps_f)
        out = F.silu(out)
        return out + residual
