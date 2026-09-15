import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUTLASS_INC = "/home/otter77/git_project/KernelMem/third_party/cutlass/include"

_CUDA_SRC = r"""
#include <cudaTypedefs.h>
using PFN_cuTensorMapEncodeTiled  = PFN_cuTensorMapEncodeTiled_v12000;
using PFN_cuTensorMapEncodeIm2col = PFN_cuTensorMapEncodeIm2col_v12000;

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#define NCH 256
#define NGROUPS 32
#define CPG 8
#define VPP 64
#define TPB 256
#define PPI 4

// ---------------------------------------------------------------- stats pass
__global__ void gn_stats_kernel(const float* __restrict__ x,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                int npix, int nchunks, int pix_per_chunk) {
    const int b     = blockIdx.y;
    const int chunk = blockIdx.x;
    const int p0    = chunk * pix_per_chunk;
    int p1          = p0 + pix_per_chunk;
    if (p1 > npix) p1 = npix;

    const int t  = threadIdx.x;
    const int v  = t & (VPP - 1);
    const int pl = t >> 6;

    const float4* __restrict__ xv =
        reinterpret_cast<const float4*>(x) + (size_t)b * npix * VPP;

    float s = 0.f, q = 0.f;
    for (int p = p0 + pl; p < p1; p += PPI) {
        float4 a = xv[(size_t)p * VPP + v];
        s += a.x + a.y + a.z + a.w;
        q += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }

    __shared__ float sh_s[TPB];
    __shared__ float sh_q[TPB];
    sh_s[t] = s;
    sh_q[t] = q;
    __syncthreads();

    if (t < NGROUPS) {
        float as = 0.f, aq = 0.f;
#pragma unroll
        for (int j = 0; j < PPI; ++j) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                int idx = j * VPP + t * 2 + k;
                as += sh_s[idx];
                aq += sh_q[idx];
            }
        }
        size_t o = ((size_t)b * NGROUPS + t) * nchunks + chunk;
        psum[o] = as;
        psq[o]  = aq;
    }
}

// --------------------------------------------------------------- finish pass
__global__ void gn_finish_kernel(const float* __restrict__ psum,
                                 const float* __restrict__ psq,
                                 float* __restrict__ mean,
                                 float* __restrict__ rstd,
                                 int nchunks, long long cnt, float eps) {
    const int bg = blockIdx.x;
    const int t  = threadIdx.x;
    const float* ps = psum + (size_t)bg * nchunks;
    const float* pq = psq  + (size_t)bg * nchunks;

    double s = 0.0, q = 0.0;
    for (int i = t; i < nchunks; i += TPB) {
        s += (double)ps[i];
        q += (double)pq[i];
    }
    __shared__ double sh_s[TPB];
    __shared__ double sh_q[TPB];
    sh_s[t] = s;
    sh_q[t] = q;
    __syncthreads();
    for (int stride = TPB / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_s[t] += sh_s[t + stride];
            sh_q[t] += sh_q[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        double n  = (double)cnt;
        double m  = sh_s[0] / n;
        double va = sh_q[0] / n - m * m;
        if (va < 0.0) va = 0.0;
        mean[bg] = (float)m;
        rstd[bg] = (float)(1.0 / sqrt(va + (double)eps));
    }
}

// ---------------------------------------------------------------- apply pass
template <bool HAS_RES>
__global__ void gn_apply_kernel(const float* __restrict__ x,
                                const float* __restrict__ mean,
                                const float* __restrict__ rstd,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                const float* __restrict__ res,
                                float* __restrict__ out,
                                int npix, int pix_per_chunk) {
    const int b     = blockIdx.y;
    const int chunk = blockIdx.x;
    const int p0    = chunk * pix_per_chunk;
    int p1          = p0 + pix_per_chunk;
    if (p1 > npix) p1 = npix;

    const int t  = threadIdx.x;
    const int v  = t & (VPP - 1);
    const int pl = t >> 6;
    const int g  = v >> 1;

    const float4 w4 = reinterpret_cast<const float4*>(gamma)[v];
    const float4 b4 = reinterpret_cast<const float4*>(beta)[v];
    const float m   = mean[b * NGROUPS + g];
    const float r   = rstd[b * NGROUPS + g];

    const size_t base = (size_t)b * npix * VPP;
    const float4* __restrict__ xv = reinterpret_cast<const float4*>(x) + base;
    const float4* __restrict__ rv = reinterpret_cast<const float4*>(res) + base;
    float4* __restrict__ ov = reinterpret_cast<float4*>(out) + base;

    for (int p = p0 + pl; p < p1; p += PPI) {
        size_t o = (size_t)p * VPP + v;
        float4 a = xv[o];
        float y0 = (a.x - m) * r * w4.x + b4.x;
        float y1 = (a.y - m) * r * w4.y + b4.y;
        float y2 = (a.z - m) * r * w4.z + b4.z;
        float y3 = (a.w - m) * r * w4.w + b4.w;
        y0 = y0 / (1.f + expf(-y0));
        y1 = y1 / (1.f + expf(-y1));
        y2 = y2 / (1.f + expf(-y2));
        y3 = y3 / (1.f + expf(-y3));
        if (HAS_RES) {
            float4 rr = rv[o];
            y0 += rr.x; y1 += rr.y; y2 += rr.z; y3 += rr.w;
        }
        float4 res4 = make_float4(y0, y1, y2, y3);
        ov[o] = res4;
    }
}

// ------------------------------------------------------------------- driver
static torch::Tensor gn_silu_impl(torch::Tensor x, torch::Tensor gamma,
                                  torch::Tensor beta, double eps,
                                  c10::optional<torch::Tensor> residual) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "x must be cuda float32");
    TORCH_CHECK(x.dim() == 4 && x.size(1) == NCH, "expect (B,256,H,W)");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "x must be channels_last");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma/beta contiguous");

    const int B    = (int)x.size(0);
    const int H    = (int)x.size(2);
    const int W    = (int)x.size(3);
    const int npix = H * W;

    auto out = torch::empty_like(x);

    long long total = (long long)B * npix;
    int ppc = (int)((total + 679) / 680);
    if (ppc < 8) ppc = 8;
    if (ppc > 256) ppc = 256;
    ppc = ((ppc + PPI - 1) / PPI) * PPI;
    if (ppc > npix) ppc = ((npix + PPI - 1) / PPI) * PPI;
    const int nchunks = (npix + ppc - 1) / ppc;

    auto opts = x.options();
    auto psum = torch::empty({B, NGROUPS, nchunks}, opts);
    auto psq  = torch::empty({B, NGROUPS, nchunks}, opts);
    auto mean = torch::empty({B, NGROUPS}, opts);
    auto rstd = torch::empty({B, NGROUPS}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(nchunks, B);
    gn_stats_kernel<<<grid, TPB, 0, stream>>>(
        x.data_ptr<float>(), psum.data_ptr<float>(), psq.data_ptr<float>(),
        npix, nchunks, ppc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_finish_kernel<<<B * NGROUPS, TPB, 0, stream>>>(
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        nchunks, (long long)npix * CPG, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (residual.has_value()) {
        torch::Tensor rsd = residual.value();
        TORCH_CHECK(rsd.is_cuda() && rsd.scalar_type() == torch::kFloat32, "res f32 cuda");
        TORCH_CHECK(rsd.is_contiguous(at::MemoryFormat::ChannelsLast), "res channels_last");
        TORCH_CHECK(rsd.sizes() == x.sizes(), "res shape");
        gn_apply_kernel<true><<<grid, TPB, 0, stream>>>(
            x.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            rsd.data_ptr<float>(), out.data_ptr<float>(), npix, ppc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        gn_apply_kernel<false><<<grid, TPB, 0, stream>>>(
            x.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            x.data_ptr<float>(), out.data_ptr<float>(), npix, ppc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

torch::Tensor gn_silu(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    return gn_silu_impl(x, gamma, beta, eps, c10::nullopt);
}

torch::Tensor gn_silu_add(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                          double eps, torch::Tensor residual) {
    return gn_silu_impl(x, gamma, beta, eps, residual);
}

// =====================================================================
// CUTLASS TF32 implicit-GEMM fprop convolution (owns the vendor conv):
// fp32 storage, TF32 tensor-op math converted in-register in the mainloop,
// fp32 accumulate.  Deletes cuDNN's convertTensor pre-pass entirely.
// =====================================================================
using ConvKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    float, cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kUnity,
    4, 4
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<ConvKernel>;

torch::Tensor conv3x3_tf32(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "x f32 cuda");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32, "w f32 cuda");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "x channels_last");
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast), "w channels_last");
    TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3, "3x3 only");

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int K = (int)w.size(0);
    TORCH_CHECK(w.size(1) == C, "channel mismatch");

    auto out = torch::empty({N, K, H, W},
                            x.options().memory_format(at::MemoryFormat::ChannelsLast));

    cutlass::conv::Conv2dProblemSize problem(
        cutlass::Tensor4DCoord(N, H, W, C),
        cutlass::Tensor4DCoord(K, 3, 3, C),
        cutlass::Tensor4DCoord(1, 1, 1, 1),   // pad h_begin,h_end,w_begin,w_end
        cutlass::MatrixCoord(1, 1),           // stride
        cutlass::MatrixCoord(1, 1),           // dilation
        cutlass::conv::Mode::kCrossCorrelation,
        1);

    cutlass::layout::TensorNHWC la(C, (long long)W * C, (long long)H * W * C);
    cutlass::layout::TensorNHWC lb(C, (long long)3 * C, (long long)3 * 3 * C);
    cutlass::layout::TensorNHWC lc(K, (long long)W * K, (long long)H * W * K);

    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> ref_a(x.data_ptr<float>(), la);
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> ref_b(w.data_ptr<float>(), lb);
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> ref_c(out.data_ptr<float>(), lc);

    typename ImplicitGemm::EpilogueOutputOp::Params eparams(1.0f, 0.0f);
    typename ImplicitGemm::Arguments args(
        problem, ref_a, ref_b, ref_c, ref_c, eparams);

    ImplicitGemm op;
    cutlass::Status st = op.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass can_implement failed");

    size_t ws = op.get_workspace_size(args);
    torch::Tensor wsbuf;
    void* wsptr = nullptr;
    if (ws > 0) {
        wsbuf = torch::zeros({(long long)ws}, x.options().dtype(torch::kUInt8));
        wsptr = wsbuf.data_ptr();
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    st = op.initialize(args, wsptr, stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass initialize failed");
    st = op(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass run failed");
    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor gn_silu(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor gn_silu_add(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                          double eps, torch::Tensor residual);
torch::Tensor conv3x3_tf32(torch::Tensor x, torch::Tensor w);
"""

_ext = load_inline(
    name="gn_silu_cutlass_conv_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gn_silu", "gn_silu_add", "conv3x3_tf32"],
    verbose=False,
    extra_include_paths=[_CUTLASS_INC],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-gencode=arch=compute_120,code=sm_120",
    ],
)

_CL = torch.channels_last


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        if torch.is_tensor(eps):
            eps = float(eps.item())
        else:
            eps = float(eps)

        xc = x if x.is_contiguous(memory_format=_CL) else x.contiguous(memory_format=_CL)
        w1 = conv1_weight if conv1_weight.is_contiguous(memory_format=_CL) \
            else conv1_weight.contiguous(memory_format=_CL)
        w2 = conv2_weight if conv2_weight.is_contiguous(memory_format=_CL) \
            else conv2_weight.contiguous(memory_format=_CL)

        g1 = norm1_weight if norm1_weight.is_contiguous() else norm1_weight.contiguous()
        b1 = norm1_bias if norm1_bias.is_contiguous() else norm1_bias.contiguous()
        g2 = norm2_weight if norm2_weight.is_contiguous() else norm2_weight.contiguous()
        b2 = norm2_bias if norm2_bias.is_contiguous() else norm2_bias.contiguous()

        out = self.ext.conv3x3_tf32(xc, w1)
        out = self.ext.gn_silu(out, g1, b1, eps)

        out = self.ext.conv3x3_tf32(out, w2)
        out = self.ext.gn_silu_add(out, g2, b2, eps, xc)
        return out
