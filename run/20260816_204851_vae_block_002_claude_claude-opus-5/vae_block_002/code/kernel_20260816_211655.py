import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <cudaTypedefs.h>
using PFN_cuTensorMapEncodeTiled  = PFN_cuTensorMapEncodeTiled_v12000;
using PFN_cuTensorMapEncodeIm2col = PFN_cuTensorMapEncodeIm2col_v12000;

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#define CH   256
#define CH4  64
#define GRPS 32
#define CPG  8

__device__ __forceinline__ float tf32_rn(float v) {
    unsigned int r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return __int_as_float(r);
}
__device__ __forceinline__ float silu(float z) { return z / (1.f + __expf(-z)); }

// ---------------------------------------------------------- K1: NCHW -> NHWC (+tf32 round)
// tile: 128 pixels x 32 channels
__global__ void nchw2nhwc_round_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst, int HW, int vec) {
    __shared__ float sm[32][132];
    const int p0 = blockIdx.x * 128;
    const int c0 = blockIdx.y * 32;
    const long nb = (long)blockIdx.z * CH * (long)HW;
    const float* sp = src + nb;
    float* dp = dst + nb;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;          // 8 warps
    if (vec && p0 + 128 <= HW) {
        for (int c = wid; c < 32; c += 8) {
            float4 v = *(const float4*)(sp + (long)(c0 + c) * HW + p0 + lane * 4);
            float4 r;
            r.x = tf32_rn(v.x); r.y = tf32_rn(v.y);
            r.z = tf32_rn(v.z); r.w = tf32_rn(v.w);
            *(float4*)&sm[c][lane * 4] = r;
        }
    } else {
        for (int c = wid; c < 32; c += 8)
            for (int j = 0; j < 4; ++j) {
                int p = p0 + lane * 4 + j;
                sm[c][lane * 4 + j] = (p < HW) ? tf32_rn(sp[(long)(c0 + c) * HW + p]) : 0.f;
            }
    }
    __syncthreads();
    const int cv = threadIdx.x & 7;
    const int pr = threadIdx.x >> 3;            // 32 pixels per pass
    for (int p = pr; p < 128; p += 32) {
        int pg = p0 + p;
        if (pg >= HW) break;
        float4 o;
        o.x = sm[cv * 4 + 0][p]; o.y = sm[cv * 4 + 1][p];
        o.z = sm[cv * 4 + 2][p]; o.w = sm[cv * 4 + 3][p];
        *(float4*)(dp + (long)pg * CH + c0 + cv * 4) = o;
    }
}

// ------------------------------------------------- K2: weights (K,C,3,3) -> (K,3,3,C) +round
// one block per filter k, staged in shared memory so both sides are coalesced
__global__ void weight_krsc_round_kernel(const float* __restrict__ w,
                                         float* __restrict__ o) {
    __shared__ float sh[CH * 9];
    const long base = (long)blockIdx.x * (CH * 9);
    for (int i = threadIdx.x; i < CH * 9; i += blockDim.x) sh[i] = w[base + i];
    __syncthreads();
    for (int i = threadIdx.x; i < CH * 9; i += blockDim.x) {
        int c = i & (CH - 1);      // CH is a power of two
        int rs = i >> 8;           // i / CH  in [0,9)
        o[base + i] = tf32_rn(sh[c * 9 + rs]);
    }
}

// ---------------------------------------------------------- K3: per-(n,group) partial moments
__global__ void gn_partial_kernel(const float* __restrict__ y,
                                  float* __restrict__ psum, float* __restrict__ psq,
                                  int HW, int chunks) {
    const int ch = blockIdx.x, n = blockIdx.y;
    const int pstart = (int)(((long)HW * ch) / chunks);
    const int pend   = (int)(((long)HW * (ch + 1)) / chunks);
    const int cv = threadIdx.x & 63;
    const int pr = threadIdx.x >> 6;
    const float4* y4 = (const float4*)(y + (long)n * HW * CH);
    float s = 0.f, q = 0.f;
    for (int p = pstart + pr; p < pend; p += 4) {
        float4 v = y4[(long)p * CH4 + cv];
        s += v.x + v.y + v.z + v.w;
        q += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    __shared__ float ss[256], sq[256];
    ss[threadIdx.x] = s; sq[threadIdx.x] = q;
    __syncthreads();
    if (threadIdx.x < GRPS) {
        int g = threadIdx.x;
        float a = 0.f, b = 0.f;
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            #pragma unroll
            for (int j = 0; j < 2; ++j) { a += ss[r * 64 + g * 2 + j]; b += sq[r * 64 + g * 2 + j]; }
        psum[((long)n * GRPS + g) * chunks + ch] = a;
        psq[((long)n * GRPS + g) * chunks + ch] = b;
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ psum, const float* __restrict__ psq,
                                   const float* __restrict__ w, const float* __restrict__ b,
                                   float* __restrict__ scale, float* __restrict__ shift,
                                   int chunks, float cnt, float eps) {
    const int g = blockIdx.x, n = blockIdx.y, tid = threadIdx.x;
    const float* ps = psum + ((long)n * GRPS + g) * chunks;
    const float* pq = psq  + ((long)n * GRPS + g) * chunks;
    float s = 0.f, q = 0.f;
    for (int i = tid; i < chunks; i += blockDim.x) { s += ps[i]; q += pq[i]; }
    __shared__ float rs[256], rq[256];
    rs[tid] = s; rq[tid] = q;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { rs[tid] += rs[tid + off]; rq[tid] += rq[tid + off]; }
        __syncthreads();
    }
    if (tid < CPG) {
        float mean = rs[0] / cnt;
        float var = rq[0] / cnt - mean * mean;
        var = var > 0.f ? var : 0.f;
        float rstd = rsqrtf(var + eps);
        int c = g * CPG + tid;
        float ww = w[c];
        scale[(long)n * CH + c] = rstd * ww;
        shift[(long)n * CH + c] = b[c] - mean * rstd * ww;
    }
}

// ---------------------------------------------- K4: GN + SiLU + tf32 round, NHWC -> NHWC
__global__ void gn_silu_round_kernel(const float* __restrict__ y,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ shift,
                                     float* __restrict__ out, int HWC4) {
    const int n = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HWC4) return;
    const int cv = i & (CH4 - 1);
    const float4* y4 = (const float4*)(y + (long)n * HWC4 * 4);
    float4* o4 = (float4*)(out + (long)n * HWC4 * 4);
    const float4 sc = ((const float4*)(scale + (long)n * CH))[cv];
    const float4 sh = ((const float4*)(shift + (long)n * CH))[cv];
    float4 v = y4[i], r;
    r.x = tf32_rn(silu(v.x * sc.x + sh.x));
    r.y = tf32_rn(silu(v.y * sc.y + sh.y));
    r.z = tf32_rn(silu(v.z * sc.z + sh.z));
    r.w = tf32_rn(silu(v.w * sc.w + sh.w));
    o4[i] = r;
}

// ------------------------------- K5: GN + SiLU + residual add, NHWC -> NCHW (fused transpose)
// tile: 128 pixels x 32 channels, float4 on both sides when HW % 4 == 0
__global__ void gn_silu_add_nchw_kernel(const float* __restrict__ y,
                                        const float* __restrict__ res,
                                        const float* __restrict__ scale,
                                        const float* __restrict__ shift,
                                        float* __restrict__ out, int HW, int vec) {
    __shared__ float sm[32][133];
    const int p0 = blockIdx.x * 128, c0 = blockIdx.y * 32, n = blockIdx.z;
    const long nb = (long)n * CH * (long)HW;
    const float* yp = y + nb;
    const int cv = threadIdx.x & 7;             // float4 index within the 32-channel tile
    const int pr = threadIdx.x >> 3;            // 32 pixels per pass
    const float4 sc = ((const float4*)(scale + (long)n * CH + c0))[cv];
    const float4 sh = ((const float4*)(shift + (long)n * CH + c0))[cv];
    for (int p = pr; p < 128; p += 32) {
        int pg = p0 + p;
        if (pg >= HW) break;
        float4 v = *(const float4*)(yp + (long)pg * CH + c0 + cv * 4);
        sm[cv * 4 + 0][p] = silu(v.x * sc.x + sh.x);
        sm[cv * 4 + 1][p] = silu(v.y * sc.y + sh.y);
        sm[cv * 4 + 2][p] = silu(v.z * sc.z + sh.z);
        sm[cv * 4 + 3][p] = silu(v.w * sc.w + sh.w);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (vec && p0 + 128 <= HW) {
        for (int c = wid; c < 32; c += 8) {
            long idx = nb + (long)(c0 + c) * HW + p0 + lane * 4;
            float4 r = *(const float4*)(res + idx);
            float4 o;
            o.x = sm[c][lane * 4 + 0] + r.x;
            o.y = sm[c][lane * 4 + 1] + r.y;
            o.z = sm[c][lane * 4 + 2] + r.z;
            o.w = sm[c][lane * 4 + 3] + r.w;
            *(float4*)(out + idx) = o;
        }
    } else {
        for (int c = wid; c < 32; c += 8)
            for (int j = 0; j < 4; ++j) {
                int p = p0 + lane * 4 + j;
                if (p < HW) {
                    long idx = nb + (long)(c0 + c) * HW + p;
                    out[idx] = sm[c][lane * 4 + j] + res[idx];
                }
            }
    }
}

// ---------------------------------------------------------------- CUTLASS conv
using ElementA = cutlass::tfloat32_t;
using ElementB = cutlass::tfloat32_t;
using ElementC = float;
using ElementAcc = float;

template <typename TB, typename WS, int Stages>
struct MakeConv {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementA, cutlass::layout::TensorNHWC,
      ElementB, cutlass::layout::TensorNHWC,
      ElementC, cutlass::layout::TensorNHWC,
      ElementAcc,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TB, WS, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 4, ElementAcc, ElementAcc>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      Stages,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kStrided,
      4, 4>::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

template <typename Conv>
static void run_conv_t(const float* xp, const float* wp, float* yp,
                       int N, int H, int W, int C, int K, const torch::Tensor& proto) {
    cutlass::conv::Conv2dProblemSize problem(
        {N, H, W, C}, {K, 3, 3, C}, {1, 1, 1, 1}, {1, 1}, {1, 1},
        {N, H, W, K}, cutlass::conv::Mode::kCrossCorrelation, 1);
    using E = cutlass::tfloat32_t;
    cutlass::TensorRef<E, cutlass::layout::TensorNHWC> ra(
        reinterpret_cast<E*>(const_cast<float*>(xp)),
        cutlass::layout::TensorNHWC::packed({N, H, W, C}));
    cutlass::TensorRef<E, cutlass::layout::TensorNHWC> rb(
        reinterpret_cast<E*>(const_cast<float*>(wp)),
        cutlass::layout::TensorNHWC::packed({K, 3, 3, C}));
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> rc(
        yp, cutlass::layout::TensorNHWC::packed({N, H, W, K}));
    typename Conv::Arguments args(problem, ra, rb, rc, rc, {1.0f, 0.0f});
    Conv op;
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "conv can_implement failed");
    size_t ws = op.get_workspace_size(args);
    auto wsb = torch::empty({(long)ws + 1}, proto.options().dtype(torch::kUInt8));
    TORCH_CHECK(op.initialize(args, wsb.data_ptr()) == cutlass::Status::kSuccess, "conv init failed");
    TORCH_CHECK(op(at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess, "conv run failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void conv3x3(const float* xp, const float* wp, float* yp,
                    int N, int H, int W, int C, int K, const torch::Tensor& proto) {
    using namespace cutlass::gemm;
    run_conv_t<typename MakeConv<GemmShape<128,64,16>, GemmShape<64,32,16>, 4>::Op>(
        xp, wp, yp, N, H, W, C, K, proto);
}

// ---------------------------------------------------------------- driver
static void gn_stats(const float* y, const float* w, const float* b, double eps,
                     int B, int HW, float* scale, float* shift, const torch::Tensor& proto) {
    int chunks = (680 + B - 1) / B;
    int maxc = (HW + 31) / 32;
    if (chunks > maxc) chunks = maxc;
    if (chunks < 1) chunks = 1;
    if (chunks > 2048) chunks = 2048;
    auto psum = torch::empty({B, GRPS, chunks}, proto.options());
    auto psq  = torch::empty({B, GRPS, chunks}, proto.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    gn_partial_kernel<<<dim3(chunks, B), 256, 0, stream>>>(
        y, psum.data_ptr<float>(), psq.data_ptr<float>(), HW, chunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gn_finalize_kernel<<<dim3(GRPS, B), 256, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(), w, b, scale, shift,
        chunks, (float)((long)HW * CPG), (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor fused_block(torch::Tensor x, torch::Tensor w1, torch::Tensor n1w,
                          torch::Tensor n1b, torch::Tensor w2, torch::Tensor n2w,
                          torch::Tensor n2b, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "x must be cuda float32");
    TORCH_CHECK(x.dim() == 4 && x.size(1) == CH, "expects (B,256,H,W)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    const int B = x.size(0), H = x.size(2), W = x.size(3);
    const int HW = H * W;
    const int vec = (HW % 4 == 0) ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opt = x.options();
    auto opt_cl = opt.memory_format(at::MemoryFormat::ChannelsLast);

    auto w1c = w1.is_contiguous() ? w1 : w1.contiguous();
    auto w2c = w2.is_contiguous() ? w2 : w2.contiguous();
    auto n1wc = n1w.is_contiguous() ? n1w : n1w.contiguous();
    auto n1bc = n1b.is_contiguous() ? n1b : n1b.contiguous();
    auto n2wc = n2w.is_contiguous() ? n2w : n2w.contiguous();
    auto n2bc = n2b.is_contiguous() ? n2b : n2b.contiguous();

    // weights -> KRSC + tf32 round
    auto w1r = torch::empty({CH, 3, 3, CH}, opt);
    auto w2r = torch::empty({CH, 3, 3, CH}, opt);
    weight_krsc_round_kernel<<<CH, 256, 0, stream>>>(w1c.data_ptr<float>(),
                                                     w1r.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    weight_krsc_round_kernel<<<CH, 256, 0, stream>>>(w2c.data_ptr<float>(),
                                                     w2r.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // x -> NHWC + tf32 round
    auto xn = torch::empty({B, CH, H, W}, opt_cl);
    nchw2nhwc_round_kernel<<<dim3((HW + 127) / 128, CH / 32, B), 256, 0, stream>>>(
        x.data_ptr<float>(), xn.data_ptr<float>(), HW, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto y1 = torch::empty({B, CH, H, W}, opt_cl);
    conv3x3(xn.data_ptr<float>(), w1r.data_ptr<float>(), y1.data_ptr<float>(),
            B, H, W, CH, CH, x);

    auto scale = torch::empty({B, CH}, opt);
    auto shift = torch::empty({B, CH}, opt);
    gn_stats(y1.data_ptr<float>(), n1wc.data_ptr<float>(), n1bc.data_ptr<float>(),
             eps, B, HW, scale.data_ptr<float>(), shift.data_ptr<float>(), x);

    auto z1 = torch::empty({B, CH, H, W}, opt_cl);
    {
        long hwc4 = (long)HW * CH4;
        int thr = 256;
        long nb = (hwc4 + thr - 1) / thr;
        gn_silu_round_kernel<<<dim3((unsigned)nb, B), thr, 0, stream>>>(
            y1.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
            z1.data_ptr<float>(), (int)hwc4);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto y2 = torch::empty({B, CH, H, W}, opt_cl);
    conv3x3(z1.data_ptr<float>(), w2r.data_ptr<float>(), y2.data_ptr<float>(),
            B, H, W, CH, CH, x);

    gn_stats(y2.data_ptr<float>(), n2wc.data_ptr<float>(), n2bc.data_ptr<float>(),
             eps, B, HW, scale.data_ptr<float>(), shift.data_ptr<float>(), x);

    auto out = torch::empty({B, CH, H, W}, opt);
    gn_silu_add_nchw_kernel<<<dim3((HW + 127) / 128, CH / 32, B), 256, 0, stream>>>(
        y2.data_ptr<float>(), x.data_ptr<float>(), scale.data_ptr<float>(),
        shift.data_ptr<float>(), out.data_ptr<float>(), HW, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_CPP = ("torch::Tensor fused_block(torch::Tensor x, torch::Tensor w1, torch::Tensor n1w,"
        "torch::Tensor n1b, torch::Tensor w2, torch::Tensor n2w, torch::Tensor n2b, double eps);")

_ext = load_inline(
    name="vae_res_block_d3",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["fused_block"],
    verbose=True,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo",
        "-gencode=arch=compute_120,code=sm_120",
        "-I/home/otter77/git_project/KernelMem/third_party/cutlass/include",
    ],
)


@torch.no_grad()
def _ref(x, c1, n1w, n1b, c2, n2w, n2b, eps):
    out = F.conv2d(x, c1, None, 1, 1)
    out = F.group_norm(out, 32, n1w, n1b, eps)
    out = F.silu(out)
    out = F.conv2d(out, c2, None, 1, 1)
    out = F.group_norm(out, 32, n2w, n2b, eps)
    out = F.silu(out)
    return out + x


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    @torch.no_grad()
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight,
                norm2_weight, norm2_bias, eps):
        if (x.dtype != torch.float32 or x.dim() != 4 or x.size(1) != 256
                or not x.is_cuda):
            return _ref(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight,
                        norm2_weight, norm2_bias, eps)
        xc = x if x.is_contiguous() else x.contiguous()
        return self.ext.fused_block(xc, conv1_weight, norm1_weight, norm1_bias,
                                    conv2_weight, norm2_weight, norm2_bias,
                                    float(eps))
