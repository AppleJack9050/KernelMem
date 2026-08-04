# ==========================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# HEADER (required):
# 1) GRANULARITY: (C) fuse many ops into one/few kernels, then collapse the
#    whole fused sequence into ONE CUDA-graph replay per forward.
#    The residual block is driven from ONE C++ entry point (`fused_block`)
#    built with load_inline; inside it there are exactly 2 vendor conv calls +
#    1 custom layout kernel + 3 custom CUDA kernels per GroupNorm stage.
#    ModelNew captures that whole call sequence into a per-shape
#    torch.cuda.CUDAGraph over static buffers and replays it once per call,
#    removing ~13 host launches + ATen/cuDNN dispatch + allocator work.
#
# 2) OPS REPLACED (all of them; nothing of the reference forward is executed
#    by torch on the Python side):
#      NCHW->NHWC of x     -> custom `nchw_to_nhwc_tiled` shared-memory tile
#                             transpose (still executes on every graph replay).
#      F.conv2d #1/#2      -> at::conv2d called INSIDE the extension, driven
#                             in NHWC (channels-last) TF32 so that cuDNN never
#                             has to insert nchwToNhwc / nhwcToNchw transposes.
#      F.group_norm #1/#2  -> custom two-stage reduction (gn_stats_partial +
#                             gn_finalize) producing per-(n,c) affine
#                             scale/shift; deterministic (no atomics).
#      F.silu #1/#2        -> folded into the GroupNorm apply kernels.
#      out + residual      -> folded into the 2nd apply kernel.
#      final NHWC->NCHW    -> folded into the 2nd apply kernel via a
#                             shared-memory tile transpose.
#
# 3) FUSION MAP:
#      k0 nchw_to_nhwc_tiled    : 32ch x 128px shared tile, both sides 128B
#                                 coalesced; writes directly into a
#                                 channels-last-allocated buffer.
#      k1 gn_stats_partial      : coalesced float4 pass over NHWC, per-tile
#                                 per-group (sum, sumsq) partials.
#      k2 gn_finalize           : partial reduce -> mean/rstd -> per-(n,c)
#                                 scale = rstd*gamma, shift = beta - mean*scale.
#      k3 gn_apply_silu         : (norm -> affine -> SiLU) in one in-place
#                                 NHWC pass  [stage 1].
#      k3' gn_apply_silu_res_t  : (norm -> affine -> SiLU -> +residual ->
#                                 NHWC->NCHW transpose -> store) in one pass
#                                 [stage 2, writes the final NCHW output].
#      graph                    : all of the above + the 2 cuDNN convs are
#                                 captured once per shape and replayed.
#
# 4) LEFT IN PYTORCH / VENDOR:
#      conv2d itself           : cuDNN TF32 implicit-GEMM is already at/near
#                                roofline; we only own its layout + epilogue.
#      w1/w2 -> channels_last  : done once during warm-up, then baked into the
#                                captured graph's private pool.
#      dtype                   : everything stays float32 (reductions fp32),
#                                TF32 tensor cores only inside conv.
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define CH   256
#define NG   32
#define CPG  8
#define C4   64          // float4 per NHWC row (256 channels)

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// k0: NCHW -> NHWC shared-memory tiled transpose (exact permutation).
//     grid = ((HW+127)/128, CH/32, N), block = 256.
//     tile = 32 channels x 128 pixels, padded row stride 133 floats
//     (gcd(133 % 32 = 5, 32) = 1 -> conflict-free column reads).
//     Phase A: each warp reads 32 consecutive pixels of one channel (128B).
//     Phase B: each warp writes 32 consecutive channels of one pixel (128B).
// ---------------------------------------------------------------------------
#define TT_C      32
#define TT_P      128
#define TT_STRIDE 133

__global__ void nchw_to_nhwc_tiled(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int HW) {
    const int n  = blockIdx.z;
    const int c0 = blockIdx.y * TT_C;
    const int p0 = blockIdx.x * TT_P;

    const float* __restrict__ src = in + ((size_t)n * CH + (size_t)c0) * (size_t)HW;
    float* __restrict__ dst = out + (size_t)n * (size_t)HW * CH;

    int prows = HW - p0;
    if (prows > TT_P) prows = TT_P;

    __shared__ float sm[TT_C * TT_STRIDE];

    // ---- phase A: coalesced NCHW read ----
    const int cl = threadIdx.x >> 7;         // 0..1
    const int pl = threadIdx.x & 127;        // 0..127
#pragma unroll 4
    for (int cc = cl; cc < TT_C; cc += 2) {
        float v = (pl < prows) ? __ldg(&src[(size_t)cc * (size_t)HW + p0 + pl]) : 0.f;
        sm[cc * TT_STRIDE + pl] = v;
    }

    __syncthreads();

    // ---- phase B: coalesced NHWC write ----
    const int lane = threadIdx.x & 31;       // channel within tile
    const int pw   = threadIdx.x >> 5;       // 0..7
#pragma unroll 4
    for (int p = pw; p < TT_P; p += 8) {
        if (p < prows) {
            dst[(size_t)(p0 + p) * CH + c0 + lane] = sm[lane * TT_STRIDE + p];
        }
    }
}

// ---------------------------------------------------------------------------
// k1: per-tile, per-group partial (sum, sumsq) over an NHWC plane.
//     grid = (numTiles, N), block = 256 threads, float4 coalesced loads.
//     thread -> 4 consecutive channels (always inside one group, 4 | 8).
// ---------------------------------------------------------------------------
__global__ void gn_stats_partial(const float* __restrict__ y,
                                 float2* __restrict__ partial,
                                 int HW, int rowsPerTile, int numTiles) {
    const int n    = blockIdx.y;
    const int tile = blockIdx.x;
    const int row0 = tile * rowsPerTile;
    int row1 = row0 + rowsPerTile;
    if (row1 > HW) row1 = HW;

    const int tid   = threadIdx.x;
    const int c4    = tid & (C4 - 1);
    const int rslot = tid >> 6;              // 0..3

    const float4* base =
        reinterpret_cast<const float4*>(y + (size_t)n * (size_t)HW * CH);

    float s = 0.f, q = 0.f;
    for (int r = row0 + rslot; r < row1; r += 4) {
        float4 v = base[(size_t)r * C4 + c4];
        s += v.x + v.y + v.z + v.w;
        q += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    __shared__ float ss[256];
    __shared__ float sq[256];
    ss[tid] = s;
    sq[tid] = q;
    __syncthreads();

    if (tid < NG) {
        float ts = 0.f, tq = 0.f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int i0 = r * C4 + tid * 2;       // group g <-> c4 in {2g, 2g+1}
            ts += ss[i0] + ss[i0 + 1];
            tq += sq[i0] + sq[i0 + 1];
        }
        partial[(size_t)(n * numTiles + tile) * NG + tid] = make_float2(ts, tq);
    }
}

// ---------------------------------------------------------------------------
// k2: reduce partials -> mean/rstd -> per-(n,c) scale/shift.
//     grid = (N), block = 256.
// ---------------------------------------------------------------------------
__global__ void gn_finalize(const float2* __restrict__ partial,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            float* __restrict__ scale,
                            float* __restrict__ shift,
                            int numTiles, float eps, float invCount) {
    const int n = blockIdx.x;
    const int t = threadIdx.x;
    const int g = t >> 3;
    const int j = t & 7;

    float s = 0.f, q = 0.f;
    for (int tl = j; tl < numTiles; tl += 8) {
        float2 p = partial[(size_t)(n * numTiles + tl) * NG + g];
        s += p.x;
        q += p.y;
    }

    __shared__ float ss[256];
    __shared__ float sq[256];
    __shared__ float smean[NG];
    __shared__ float srstd[NG];
    ss[t] = s;
    sq[t] = q;
    __syncthreads();

    if (t < NG) {
        float a = 0.f, b = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a += ss[t * 8 + k];
            b += sq[t * 8 + k];
        }
        float m   = a * invCount;
        float var = b * invCount - m * m;
        if (var < 0.f) var = 0.f;
        smean[t] = m;
        srstd[t] = rsqrtf(var + eps);
    }
    __syncthreads();

    const int c  = t;                       // 256 channels, 256 threads
    const int gg = c >> 3;
    float sc = srstd[gg] * gamma[c];
    scale[n * CH + c] = sc;
    shift[n * CH + c] = beta[c] - smean[gg] * sc;
}

// ---------------------------------------------------------------------------
// k3: in-place NHWC  y = silu(y*scale + shift)
//     grid = (gx, N), block = 256, float4.
// ---------------------------------------------------------------------------
__global__ void gn_apply_silu(float* __restrict__ y,
                              const float* __restrict__ scale,
                              const float* __restrict__ shift,
                              int HW) {
    const int n   = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float4 ssc[C4];
    __shared__ float4 ssh[C4];
    if (tid < C4) {
        ssc[tid] = reinterpret_cast<const float4*>(scale + n * CH)[tid];
        ssh[tid] = reinterpret_cast<const float4*>(shift + n * CH)[tid];
    }
    __syncthreads();

    float4* p = reinterpret_cast<float4*>(y + (size_t)n * (size_t)HW * CH);
    const int total4 = HW * C4;
    const int stride = gridDim.x * blockDim.x;

    for (int idx = blockIdx.x * blockDim.x + tid; idx < total4; idx += stride) {
        const int c4 = idx & (C4 - 1);
        float4 v  = p[idx];
        float4 sc = ssc[c4];
        float4 sh = ssh[c4];
        v.x = silu_f(v.x * sc.x + sh.x);
        v.y = silu_f(v.y * sc.y + sh.y);
        v.z = silu_f(v.z * sc.z + sh.z);
        v.w = silu_f(v.w * sc.w + sh.w);
        p[idx] = v;
    }
}

// ---------------------------------------------------------------------------
// k3': NHWC in -> (norm, affine, SiLU, +residual, transpose) -> NCHW out.
//      grid = (ceil(HW/32), N), block = 256, 32 rows per tile.
// ---------------------------------------------------------------------------
#define TR   32
#define SPAD 260                 // 32 rows * 260 floats, 16B aligned rows

__global__ void gn_apply_silu_res_t(const float* __restrict__ y,
                                    const float* __restrict__ scale,
                                    const float* __restrict__ shift,
                                    const float* __restrict__ res,
                                    float* __restrict__ out,
                                    int HW) {
    const int n    = blockIdx.y;
    const int row0 = blockIdx.x * TR;
    const int tid  = threadIdx.x;

    __shared__ __align__(16) float sm[TR * SPAD];
    __shared__ float4 ssc[C4];
    __shared__ float4 ssh[C4];

    if (tid < C4) {
        ssc[tid] = reinterpret_cast<const float4*>(scale + n * CH)[tid];
        ssh[tid] = reinterpret_cast<const float4*>(shift + n * CH)[tid];
    }
    __syncthreads();

    const int c4    = tid & (C4 - 1);
    const int rslot = tid >> 6;              // 0..3
    const float4* yb =
        reinterpret_cast<const float4*>(y + (size_t)n * (size_t)HW * CH);

    int rows = HW - row0;
    if (rows > TR) rows = TR;

    // phase 1: coalesced NHWC read, epilogue math, staged in shared memory
    for (int rr = rslot; rr < TR; rr += 4) {
        if (rr < rows) {
            float4 v  = yb[(size_t)(row0 + rr) * C4 + c4];
            float4 sc = ssc[c4];
            float4 sh = ssh[c4];
            v.x = silu_f(v.x * sc.x + sh.x);
            v.y = silu_f(v.y * sc.y + sh.y);
            v.z = silu_f(v.z * sc.z + sh.z);
            v.w = silu_f(v.w * sc.w + sh.w);
            *reinterpret_cast<float4*>(&sm[rr * SPAD + 4 * c4]) = v;
        }
    }
    __syncthreads();

    // phase 2: coalesced NCHW write with residual add
    const int cl  = tid >> 5;                // 0..7
    const int row = tid & 31;                // 0..31
    const size_t planeBase = (size_t)n * CH * (size_t)HW;
    if (row < rows) {
        const size_t off = (size_t)(row0 + row);
        for (int cb = 0; cb < CH; cb += 8) {
            const int c = cb + cl;
            const size_t gi = planeBase + (size_t)c * (size_t)HW + off;
            out[gi] = sm[row * SPAD + c] + res[gi];
        }
    }
}

// ---------------------------------------------------------------------------
// host driver: the whole residual block
// ---------------------------------------------------------------------------
static void gn_tiling(int N, int HW, int& rowsPerTile, int& numTiles) {
    rowsPerTile = 64;
    numTiles = (HW + rowsPerTile - 1) / rowsPerTile;
    while ((long long)N * numTiles > 4096 && rowsPerTile < HW) {
        rowsPerTile *= 2;
        numTiles = (HW + rowsPerTile - 1) / rowsPerTile;
    }
    while ((long long)N * numTiles < 132 && rowsPerTile > 8) {
        rowsPerTile /= 2;
        numTiles = (HW + rowsPerTile - 1) / rowsPerTile;
    }
    if (numTiles < 1) numTiles = 1;
}

static void run_group_norm_stats(const at::Tensor& y, const at::Tensor& gamma,
                                 const at::Tensor& beta, at::Tensor& scale,
                                 at::Tensor& shift, int N, int HW, double eps,
                                 cudaStream_t stream) {
    int rowsPerTile, numTiles;
    gn_tiling(N, HW, rowsPerTile, numTiles);

    auto partial = at::empty({(long)N * numTiles * NG * 2}, y.options());

    dim3 g1(numTiles, N), b1(256);
    gn_stats_partial<<<g1, b1, 0, stream>>>(
        y.data_ptr<float>(),
        reinterpret_cast<float2*>(partial.data_ptr<float>()),
        HW, rowsPerTile, numTiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float invCount = (float)(1.0 / ((double)HW * (double)CPG));
    dim3 g2(N), b2(256);
    gn_finalize<<<g2, b2, 0, stream>>>(
        reinterpret_cast<const float2*>(partial.data_ptr<float>()),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        numTiles, (float)eps, invCount);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight,
                          torch::Tensor norm1_weight,
                          torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight,
                          torch::Tensor norm2_weight,
                          torch::Tensor norm2_bias,
                          double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    TORCH_CHECK(C == CH, "channels must be 256");
    const int HW = H * W;

    auto xc   = x.is_contiguous() ? x : x.contiguous();
    auto w1   = conv1_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2   = conv2_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto g1w  = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
    auto g1b  = norm1_bias.is_contiguous()   ? norm1_bias   : norm1_bias.contiguous();
    auto g2w  = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
    auto g2b  = norm2_bias.is_contiguous()   ? norm2_bias   : norm2_bias.contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ---- custom NCHW -> NHWC transpose (replaces the aten elementwise copy) ----
    // Allocate DIRECTLY in channels-last: no aten copy of uninitialized data.
    auto x_cl = at::empty({N, C, H, W},
                          x.options().memory_format(at::MemoryFormat::ChannelsLast));
    {
        dim3 grid((HW + TT_P - 1) / TT_P, CH / TT_C, N);
        dim3 block(256);
        nchw_to_nhwc_tiled<<<grid, block, 0, stream>>>(
            xc.data_ptr<float>(), x_cl.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv1 (NHWC TF32, cuDNN) ----
    auto y1 = at::conv2d(x_cl, w1, at::Tensor(), {1, 1}, {1, 1}, {1, 1}, 1);
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    auto scale = at::empty({N, CH}, x.options());
    auto shift = at::empty({N, CH}, x.options());
    run_group_norm_stats(y1, g1w, g1b, scale, shift, N, HW, eps, stream);

    {
        int total4 = HW * C4;
        int gx = (total4 + 255) / 256;
        if (gx > 2048) gx = 2048;
        if (gx < 1) gx = 1;
        dim3 g(gx, N), b(256);
        gn_apply_silu<<<g, b, 0, stream>>>(
            y1.data_ptr<float>(), scale.data_ptr<float>(),
            shift.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- conv2 (NHWC TF32, cuDNN) ----
    auto y2 = at::conv2d(y1, w2, at::Tensor(), {1, 1}, {1, 1}, {1, 1}, 1);
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    run_group_norm_stats(y2, g2w, g2b, scale, shift, N, HW, eps, stream);

    auto out = at::empty({N, C, H, W}, x.options());
    {
        int gx = (HW + TR - 1) / TR;
        dim3 g(gx, N), b(256);
        gn_apply_silu_res_t<<<g, b, 0, stream>>>(
            y2.data_ptr<float>(), scale.data_ptr<float>(),
            shift.data_ptr<float>(), xc.data_ptr<float>(),
            out.data_ptr<float>(), HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight,
                          torch::Tensor norm1_weight,
                          torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight,
                          torch::Tensor norm2_weight,
                          torch::Tensor norm2_bias,
                          double eps);
"""

_ext = load_inline(
    name="vae_resblock_fused_c",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_block"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-gencode=arch=compute_90,code=sm_90",
    ],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """See file header for granularity (C) / fusion map.

    Steady state: one CUDA-graph replay per forward (per-shape cached).
    """

    def __init__(self):
        super().__init__()
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        self.ext = _ext
        # plan item 2: per-shape graph cache + dedicated warm-up stream
        self._graphs = {}
        try:
            self._warm_stream = torch.cuda.Stream()
        except Exception:
            self._warm_stream = None

    # ---- plan items 5-8: build (warm up + capture) one graph for this key ----
    @torch.no_grad()
    def _build_graph(self, x_c, w_list, eps):
        # 5) OWN static buffers, never aliasing caller tensors
        sx = torch.empty_like(x_c)
        sx.copy_(x_c)
        static_w = []
        for t in w_list:
            s = torch.empty_like(t)
            s.copy_(t)
            static_w.append(s)
        sw1, sn1w, sn1b, sw2, sn2w, sn2b = static_w

        # 6) warm-up on a side stream: cuDNN benchmark, workspace, weight cvt
        cur = torch.cuda.current_stream()
        if self._warm_stream is not None:
            self._warm_stream.wait_stream(cur)
            with torch.cuda.stream(self._warm_stream):
                for _ in range(3):
                    _ = self.ext.fused_block(sx, sw1, sn1w, sn1b,
                                             sw2, sn2w, sn2b, eps)
            cur.wait_stream(self._warm_stream)
        else:
            for _ in range(3):
                _ = self.ext.fused_block(sx, sw1, sn1w, sn1b,
                                         sw2, sn2w, sn2b, eps)
        torch.cuda.synchronize()

        # 7) capture the whole fused_block sequence
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = self.ext.fused_block(sx, sw1, sn1w, sn1b,
                                              sw2, sn2w, sn2b, eps)
        return (g, sx, sw1, sn1w, sn1b, sw2, sn2w, sn2b, static_out)

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        # 3) eligibility guard unchanged
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256):
            epsf = float(eps)
            x_c = x if x.is_contiguous() else x.contiguous()
            w_list = (conv1_weight, norm1_weight, norm1_bias,
                      conv2_weight, norm2_weight, norm2_bias)

            # 4) per-shape / per-constant cache key
            key = (tuple(x.shape), x.dtype, x.device.index, epsf)
            entry = self._graphs.get(key, "MISS")

            if entry == "MISS":
                ok = all(t.is_cuda and t.is_contiguous() for t in w_list)
                if ok:
                    try:
                        entry = self._build_graph(x_c, w_list, epsf)
                    except Exception:
                        entry = None            # 8) permanent eager sentinel
                else:
                    entry = None
                self._graphs[key] = entry

            if entry is not None:
                # 9) steady-state timed path: copies + replay, no allocation
                (g, sx, sw1, sn1w, sn1b, sw2, sn2w, sn2b, static_out) = entry
                sx.copy_(x_c, non_blocking=True)
                sw1.copy_(conv1_weight, non_blocking=True)
                sn1w.copy_(norm1_weight, non_blocking=True)
                sn1b.copy_(norm1_bias, non_blocking=True)
                sw2.copy_(conv2_weight, non_blocking=True)
                sn2w.copy_(norm2_weight, non_blocking=True)
                sn2b.copy_(norm2_bias, non_blocking=True)
                g.replay()
                return static_out

            # eager extension path (capture unavailable for this key)
            return self.ext.fused_block(x, conv1_weight, norm1_weight,
                                        norm1_bias, conv2_weight,
                                        norm2_weight, norm2_bias, epsf)

        # generic fallback (kept only for unsupported shapes/dtypes)
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=eps)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=eps)
        out = F.silu(out)
        return out + x
