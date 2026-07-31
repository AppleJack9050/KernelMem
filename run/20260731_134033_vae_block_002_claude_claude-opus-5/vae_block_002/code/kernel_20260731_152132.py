import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cstring>

namespace {

constexpr int CC  = 256;          // channels (const in the SOL definition)
constexpr int GG  = 32;           // groups   (const in the SOL definition)
constexpr int CPG = CC / GG;      // 8 channels per group

// ============ plan item 2: __constant__ I/O base pointers + GN parameter bank ======
struct IOPtrs { const float* x; float* out; };
__constant__ IOPtrs c_io;
__constant__ float  c_gn[4 * CC];   // n1w @0, n1b @256, n2w @512, n2b @768 (floats)

// ---------------------------------------------------------------- K1: NCHW -> NHWC
// plan item 3: source pointer now comes from c_io.x + elem_off (no pointer args)
__global__ void nchw2nhwc_kernel(float* __restrict__ dst,
                                 long elem_off,
                                 int HW) {
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const float* s = c_io.x + elem_off + (size_t)n * (size_t)CC * (size_t)HW;
    float*       d = dst             + (size_t)n * (size_t)HW * (size_t)CC;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + ty + 8 * k;
        const int p = p0 + tx;
        float v = 0.f;
        if (p < HW) v = s[(size_t)c * (size_t)HW + p];
        tile[ty + 8 * k][tx] = v;            // tile[c_local][p_local]
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = p0 + ty + 8 * k;
        const int c = c0 + tx;
        if (p < HW) d[(size_t)p * (size_t)CC + c] = tile[tx][ty + 8 * k];
    }
}

// ------------------------------------------------- K2: per-(n,group) partial moments
__global__ void gn_partial_kernel(const float* __restrict__ y,
                                  double2* __restrict__ part,
                                  int HW, int nchunks, int ppc) {
    const int n  = blockIdx.y;
    const int ch = blockIdx.x;
    const int c  = threadIdx.x;

    const int pstart = ch * ppc;
    int pend = pstart + ppc;
    if (pend > HW) pend = HW;

    const float* base = y + (size_t)n * (size_t)HW * (size_t)CC + c;

    double s = 0.0, ss = 0.0;
    int p = pstart;
    for (; p + 3 < pend; p += 4) {
        float v0 = base[(size_t)(p    ) * CC];
        float v1 = base[(size_t)(p + 1) * CC];
        float v2 = base[(size_t)(p + 2) * CC];
        float v3 = base[(size_t)(p + 3) * CC];
        s  += (double)v0 + (double)v1 + (double)v2 + (double)v3;
        ss += (double)v0 * (double)v0 + (double)v1 * (double)v1
            + (double)v2 * (double)v2 + (double)v3 * (double)v3;
    }
    for (; p < pend; ++p) {
        float v = base[(size_t)p * CC];
        s  += (double)v;
        ss += (double)v * (double)v;
    }

#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  off);
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    if ((c & (CPG - 1)) == 0) {
        const int g = c >> 3;
        part[(size_t)(n * GG + g) * (size_t)nchunks + ch] = make_double2(s, ss);
    }
}

// ------------------------------------------- K3: partials -> per-(n,c) scale / shift
// plan item 5: gamma/beta now selected from the __constant__ bank via `which`
__global__ void gn_finalize_kernel(const double2* __restrict__ part,
                                   int which,
                                   float* __restrict__ scale,
                                   float* __restrict__ shift,
                                   int nchunks, double M, double eps) {
    __shared__ double sm_s[128];
    __shared__ double sm_ss[128];
    const int ng  = blockIdx.x;          // n * GG + g
    const int tid = threadIdx.x;

    double s = 0.0, ss = 0.0;
    for (int i = tid; i < nchunks; i += 128) {
        double2 v = part[(size_t)ng * (size_t)nchunks + i];
        s  += v.x;
        ss += v.y;
    }
    sm_s[tid] = s; sm_ss[tid] = ss;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) { sm_s[tid] += sm_s[tid + st]; sm_ss[tid] += sm_ss[tid + st]; }
        __syncthreads();
    }
    if (tid < CPG) {
        const float* gamma = c_gn + (which ? 512 : 0);
        const float* beta  = c_gn + (which ? 768 : 256);
        const double mean = sm_s[0] / M;
        double var = sm_ss[0] / M - mean * mean;
        if (!(var > 0.0)) var = 0.0;
        const double rstd = 1.0 / sqrt(var + eps);
        const int g = ng % GG;
        const int n = ng / GG;
        const int c = g * CPG + tid;
        const double gm = (double)gamma[c];
        const double bt = (double)beta[c];
        scale[n * CC + c] = (float)(rstd * gm);
        shift[n * CC + c] = (float)(bt - mean * rstd * gm);
    }
}

// ------------------------------------------------ K4: GN-affine + SiLU (NHWC -> NHWC)
__global__ void gn_silu_kernel(const float* __restrict__ y,
                               const float* __restrict__ scale,
                               const float* __restrict__ shift,
                               float* __restrict__ out,
                               long HW) {
    __shared__ float ssc[CC];
    __shared__ float ssh[CC];
    const int n = blockIdx.y;
    const int tid = threadIdx.x;
    ssc[tid] = scale[n * CC + tid];
    ssh[tid] = shift[n * CC + tid];
    __syncthreads();

    const float4* yp = reinterpret_cast<const float4*>(y   + (size_t)n * (size_t)HW * CC);
    float4*       op = reinterpret_cast<float4*>      (out + (size_t)n * (size_t)HW * CC);
    const long nq = HW * (long)(CC / 4);

    for (long q = (long)blockIdx.x * blockDim.x + tid; q < nq;
         q += (long)gridDim.x * blockDim.x) {
        const int cq = (int)(q & (long)(CC / 4 - 1));
        const float4 v  = yp[q];
        const float sc0 = ssc[cq * 4 + 0], sh0 = ssh[cq * 4 + 0];
        const float sc1 = ssc[cq * 4 + 1], sh1 = ssh[cq * 4 + 1];
        const float sc2 = ssc[cq * 4 + 2], sh2 = ssh[cq * 4 + 2];
        const float sc3 = ssc[cq * 4 + 3], sh3 = ssh[cq * 4 + 3];
        float t0 = fmaf(v.x, sc0, sh0);
        float t1 = fmaf(v.y, sc1, sh1);
        float t2 = fmaf(v.z, sc2, sh2);
        float t3 = fmaf(v.w, sc3, sh3);
        float4 r;
        r.x = t0 / (1.0f + expf(-t0));
        r.y = t1 / (1.0f + expf(-t1));
        r.z = t2 / (1.0f + expf(-t2));
        r.w = t3 / (1.0f + expf(-t3));
        op[q] = r;
    }
}

// ------------ K5: GN-affine + SiLU + residual add + NHWC -> NCHW transpose (one pass)
// plan items 4 + 12: residual/out bases from c_io, register cap via __launch_bounds__
__global__ __launch_bounds__(256, 8)
void gn_silu_res_t_kernel(const float* __restrict__ y,
                          const float* __restrict__ scale,
                          const float* __restrict__ shift,
                          long elem_off,
                          int HW) {
    __shared__ float tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const float* __restrict__ xres = c_io.x   + elem_off;   // NCHW contiguous
    float* __restrict__       outp = c_io.out + elem_off;   // NCHW contiguous

    const float* yb = y + (size_t)n * (size_t)HW * (size_t)CC;
    const float* sc_b = scale + n * CC;
    const float* sh_b = shift + n * CC;

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = p0 + ty + 8 * k;
        const int c = c0 + tx;
        float t = 0.f;
        if (p < HW) {
            const float v = yb[(size_t)p * (size_t)CC + c];
            t = fmaf(v, sc_b[c], sh_b[c]);
            t = t / (1.0f + expf(-t));
        }
        tile[tx][ty + 8 * k] = t;          // tile[c_local][p_local]
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + ty + 8 * k;
        const int p = p0 + tx;
        if (p < HW) {
            const size_t off = ((size_t)n * CC + c) * (size_t)HW + p;
            outp[off] = tile[ty + 8 * k][tx] + xres[off];
        }
    }
}

} // namespace

// ------------------------------------------------------------- cudnn_algo_autotune
namespace detail {

template <class... Args>
auto cudnn_conv_dispatch(int, Args&&... args)
    -> decltype(at::cudnn_convolution(std::forward<Args>(args)...)) {
    return at::cudnn_convolution(std::forward<Args>(args)...);
}

template <class... Args>
at::Tensor cudnn_conv_dispatch(long, Args&&...) {
    throw std::runtime_error("at::cudnn_convolution unavailable with this signature");
}

template <class Ctx>
auto set_bm_limit(Ctx& c, int v, int) -> decltype(c.setBenchmarkLimitCuDNN(v), void()) {
    c.setBenchmarkLimitCuDNN(v);
}
template <class Ctx>
void set_bm_limit(Ctx&, int, long) {}

inline void ensure_bm() {
    static const bool _bm_init = [](){
        at::globalContext().setBenchmarkCuDNN(true);
        set_bm_limit(at::globalContext(), 16, 0);
        return true;
    }();
    (void)_bm_init;
}

} // namespace detail

// ================= plan item 9: chunk / reduction-schedule planning (UNCHANGED) =====
std::tuple<int64_t, int64_t, int64_t> plan_chunk(int64_t N, int64_t C,
                                                 int64_t H, int64_t W) {
    detail::ensure_bm();
    const int64_t HW = H * W;
    const int64_t per_sample_bytes = C * HW * 4;
    const int64_t target_live      = (int64_t)40 << 20;   // ~80% of H100's 50 MB L2

    int64_t chunk = target_live / (2 * per_sample_bytes); // two intermediates live
    if (chunk < 1) chunk = 1;
    if (chunk > N) chunk = N;
    while (N % chunk) --chunk;                            // snap DOWN to a divisor of N
    while (chunk * HW < 16384 && chunk < N) {             // conv-efficiency guard
        ++chunk;
        while (N % chunk) ++chunk;                        // next divisor of N
    }
    if (N * per_sample_bytes <= target_live) chunk = N;
    TORCH_CHECK(N % chunk == 0, "chunk must divide N");

    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int min_ppc = 32;
    long nc = ((long)8 * sms + chunk - 1) / chunk;
    const long max_nc = (HW + min_ppc - 1) / min_ppc;
    if (nc > max_nc) nc = max_nc;
    if (nc < 1) nc = 1;
    const int ppc     = (int)((HW + nc - 1) / nc);
    const int nchunks = (int)((HW + ppc - 1) / ppc);

    return std::make_tuple(chunk, (int64_t)nchunks, (int64_t)ppc);
}

// ================= plan item 9: persisting-L2 window on the CURRENT stream ==========
bool set_l2_window(torch::Tensor z) {
    static const size_t max_win = [](){
        const auto* p = at::cuda::getCurrentDeviceProperties();
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           (size_t)p->persistingL2CacheMaxSize);
        cudaGetLastError();
        return (size_t)p->accessPolicyMaxWindowSize;
    }();
    const size_t win_bytes = (size_t)z.numel() * (size_t)z.element_size();
    if (win_bytes > max_win || win_bytes > ((size_t)30 << 20)) return false;
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStreamAttrValue av;
    memset(&av, 0, sizeof(av));
    av.accessPolicyWindow.base_ptr  = z.data_ptr();
    av.accessPolicyWindow.num_bytes = win_bytes;
    av.accessPolicyWindow.hitRatio  = 1.0f;
    av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    const bool ok = (cudaStreamSetAttribute(stream,
                        cudaStreamAttributeAccessPolicyWindow, &av) == cudaSuccess);
    cudaGetLastError();
    return ok;
}

// ================= plan item 6: I/O base pointers -> __constant__ (outside graph) ===
namespace detail {
constexpr int IO_SLOTS = 32;
inline void* io_host_buf() {
    static void* p = [](){
        void* q = nullptr;
        if (cudaHostAlloc(&q, (size_t)IO_SLOTS * 16, cudaHostAllocDefault) != cudaSuccess)
            q = nullptr;
        cudaGetLastError();
        return q;
    }();
    return p;
}
} // namespace detail

void set_io_ptrs(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(),
                "set_io_ptrs: contiguous NCHW tensors expected");
    static int slot = 0;
    void* base = detail::io_host_buf();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (base == nullptr) {
        // extremely defensive: fall back to a synchronous stack copy
        void* tmp[2];
        tmp[0] = (void*)x.data_ptr<float>();
        tmp[1] = (void*)out.data_ptr<float>();
        C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_io, tmp, 16, 0,
                                               cudaMemcpyHostToDevice, stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    void** h = reinterpret_cast<void**>(reinterpret_cast<char*>(base) + (size_t)slot * 16);
    slot = (slot + 1) % detail::IO_SLOTS;
    h[0] = (void*)x.data_ptr<float>();
    h[1] = (void*)out.data_ptr<float>();
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_io, h, 16, 0,
                                           cudaMemcpyHostToDevice, stream));
}

// ================= plan item 7: GN parameter vectors -> __constant__ (DMA) =========
void set_gn_params(torch::Tensor n1w, torch::Tensor n1b,
                   torch::Tensor n2w, torch::Tensor n2b) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t nb = (size_t)CC * sizeof(float);
    TORCH_CHECK(n1w.numel() == CC && n1b.numel() == CC &&
                n2w.numel() == CC && n2b.numel() == CC,
                "set_gn_params: expected 256-element GN vectors");
    TORCH_CHECK(n1w.is_contiguous() && n1b.is_contiguous() &&
                n2w.is_contiguous() && n2b.is_contiguous(),
                "set_gn_params: contiguous vectors expected");
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_gn, n1w.data_ptr<float>(), nb,
                                           0 * nb, cudaMemcpyDeviceToDevice, stream));
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_gn, n1b.data_ptr<float>(), nb,
                                           1 * nb, cudaMemcpyDeviceToDevice, stream));
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_gn, n2w.data_ptr<float>(), nb,
                                           2 * nb, cudaMemcpyDeviceToDevice, stream));
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_gn, n2b.data_ptr<float>(), nb,
                                           3 * nb, cudaMemcpyDeviceToDevice, stream));
}

// =========================== K1 for one chunk (interior-graph fallback) ============
void k1_chunk(int64_t elem_off, torch::Tensor x_cl) {
    at::NoGradGuard no_grad;
    const int64_t C = x_cl.size(1), H = x_cl.size(2), W = x_cl.size(3), HW = H * W;
    const int64_t chunk = x_cl.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 t_blk(32, 8);
    const dim3 t_grd((unsigned)((HW + 31) / 32), (unsigned)(C / 32), (unsigned)chunk);
    nchw2nhwc_kernel<<<t_grd, t_blk, 0, stream>>>(
        x_cl.data_ptr<float>(), (long)elem_off, (int)HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ================= graph-captured interior (fallback path, conv1..gn_finalize2) ====
torch::Tensor interior(torch::Tensor x_cl,
                       torch::Tensor w1c, torch::Tensor w2c,
                       double eps,
                       torch::Tensor part, torch::Tensor scale, torch::Tensor shift,
                       torch::Tensor z,
                       int64_t nchunks, int64_t ppc, double M) {
    at::NoGradGuard no_grad;
    detail::ensure_bm();

    const int64_t chunk = x_cl.size(0);
    const int64_t H = x_cl.size(2), W = x_cl.size(3), HW = H * W;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    std::vector<int64_t> stride{1, 1}, pad{1, 1}, dil{1, 1};

    const long nq = HW * (long)(CC / 4);
    int silu_blocks = (int)std::min<long>((nq + 255) / 256, (long)(6 * sms));
    if (silu_blocks < 1) silu_blocks = 1;

    at::Tensor y1;
    try {
        y1 = detail::cudnn_conv_dispatch(0, x_cl, w1c, pad, stride, dil,
                                         (int64_t)1, true, false, true);
    } catch (const std::exception&) {
        y1 = at::conv2d(x_cl, w1c, {}, stride, pad, dil, 1);
    }
    if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
        y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

    gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)chunk), 256, 0, stream>>>(
        y1.data_ptr<float>(), reinterpret_cast<double2*>(part.data_ptr<double>()),
        (int)HW, (int)nchunks, (int)ppc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gn_finalize_kernel<<<(unsigned)(chunk * GG), 128, 0, stream>>>(
        reinterpret_cast<const double2*>(part.data_ptr<double>()), 0,
        scale.data_ptr<float>(), shift.data_ptr<float>(), (int)nchunks, M, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gn_silu_kernel<<<dim3((unsigned)silu_blocks, (unsigned)chunk), 256, 0, stream>>>(
        y1.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
        z.data_ptr<float>(), (long)HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor y2;
    try {
        y2 = detail::cudnn_conv_dispatch(0, z, w2c, pad, stride, dil,
                                         (int64_t)1, true, false, true);
    } catch (const std::exception&) {
        y2 = at::conv2d(z, w2c, {}, stride, pad, dil, 1);
    }
    if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
        y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

    gn_partial_kernel<<<dim3((unsigned)nchunks, (unsigned)chunk), 256, 0, stream>>>(
        y2.data_ptr<float>(), reinterpret_cast<double2*>(part.data_ptr<double>()),
        (int)HW, (int)nchunks, (int)ppc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gn_finalize_kernel<<<(unsigned)(chunk * GG), 128, 0, stream>>>(
        reinterpret_cast<const double2*>(part.data_ptr<double>()), 1,
        scale.data_ptr<float>(), shift.data_ptr<float>(), (int)nchunks, M, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y2;
}

// =========================== K5 for one chunk (interior-graph fallback) ============
void k5_chunk(torch::Tensor y2, int64_t elem_off,
              torch::Tensor scale, torch::Tensor shift, int64_t C) {
    at::NoGradGuard no_grad;
    const int64_t H = y2.size(2), W = y2.size(3), HW = H * W;
    const int64_t chunk = y2.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 t_blk(32, 8);
    const dim3 t_grd((unsigned)((HW + 31) / 32), (unsigned)(C / 32), (unsigned)chunk);
    gn_silu_res_t_kernel<<<t_grd, t_blk, 0, stream>>>(
        y2.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
        (long)elem_off, (int)HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ================= plan item 8: whole-forward single-stream pipeline ===============
void full_pipeline(torch::Tensor w1c, torch::Tensor w2c, double eps,
                   torch::Tensor part, torch::Tensor scale, torch::Tensor shift,
                   torch::Tensor x_cl, torch::Tensor z,
                   int64_t nchunks, int64_t ppc, double M, int64_t N) {
    at::NoGradGuard no_grad;
    detail::ensure_bm();

    const int64_t chunk = x_cl.size(0);
    const int64_t C = x_cl.size(1), H = x_cl.size(2), W = x_cl.size(3), HW = H * W;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    std::vector<int64_t> stride{1, 1}, pad{1, 1}, dil{1, 1};

    const long nq = HW * (long)(CC / 4);
    int silu_blocks = (int)std::min<long>((nq + 255) / 256, (long)(6 * sms));
    if (silu_blocks < 1) silu_blocks = 1;

    const dim3 t_blk(32, 8);
    const dim3 t_grd((unsigned)((HW + 31) / 32), (unsigned)(C / 32), (unsigned)chunk);
    const dim3 p_grd((unsigned)nchunks, (unsigned)chunk);
    const dim3 s_grd((unsigned)silu_blocks, (unsigned)chunk);

    for (int64_t off = 0; off < N; off += chunk) {
        const long elem_off = (long)off * (long)C * (long)HW;

        // ---- K1: NCHW -> NHWC for this chunk (source from c_io.x) --------------
        nchw2nhwc_kernel<<<t_grd, t_blk, 0, stream>>>(
            x_cl.data_ptr<float>(), elem_off, (int)HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // ---- conv1 (cuDNN v8, NHWC, TF32, autotuned) --------------------------
        at::Tensor y1;
        try {
            y1 = detail::cudnn_conv_dispatch(0, x_cl, w1c, pad, stride, dil,
                                             (int64_t)1, true, false, true);
        } catch (const std::exception&) {
            y1 = at::conv2d(x_cl, w1c, {}, stride, pad, dil, 1);
        }
        if (!y1.is_contiguous(at::MemoryFormat::ChannelsLast))
            y1 = y1.contiguous(at::MemoryFormat::ChannelsLast);

        gn_partial_kernel<<<p_grd, 256, 0, stream>>>(
            y1.data_ptr<float>(), reinterpret_cast<double2*>(part.data_ptr<double>()),
            (int)HW, (int)nchunks, (int)ppc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        gn_finalize_kernel<<<(unsigned)(chunk * GG), 128, 0, stream>>>(
            reinterpret_cast<const double2*>(part.data_ptr<double>()), 0,
            scale.data_ptr<float>(), shift.data_ptr<float>(), (int)nchunks, M, eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        gn_silu_kernel<<<s_grd, 256, 0, stream>>>(
            y1.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
            z.data_ptr<float>(), (long)HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // ---- conv2 -------------------------------------------------------------
        at::Tensor y2;
        try {
            y2 = detail::cudnn_conv_dispatch(0, z, w2c, pad, stride, dil,
                                             (int64_t)1, true, false, true);
        } catch (const std::exception&) {
            y2 = at::conv2d(z, w2c, {}, stride, pad, dil, 1);
        }
        if (!y2.is_contiguous(at::MemoryFormat::ChannelsLast))
            y2 = y2.contiguous(at::MemoryFormat::ChannelsLast);

        gn_partial_kernel<<<p_grd, 256, 0, stream>>>(
            y2.data_ptr<float>(), reinterpret_cast<double2*>(part.data_ptr<double>()),
            (int)HW, (int)nchunks, (int)ppc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        gn_finalize_kernel<<<(unsigned)(chunk * GG), 128, 0, stream>>>(
            reinterpret_cast<const double2*>(part.data_ptr<double>()), 1,
            scale.data_ptr<float>(), shift.data_ptr<float>(), (int)nchunks, M, eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // ---- K5: GN + SiLU + residual + NHWC->NCHW (dest from c_io.out) --------
        gn_silu_res_t_kernel<<<t_grd, t_blk, 0, stream>>>(
            y2.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
            elem_off, (int)HW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// ------------------------------- monolithic eager path (final fallback) -----------
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight,
                          torch::Tensor norm1_weight,
                          torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight,
                          torch::Tensor norm2_weight,
                          torch::Tensor norm2_bias,
                          double eps) {
    at::NoGradGuard no_grad;
    detail::ensure_bm();

    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "expected NCHW");
    TORCH_CHECK(x.size(1) == CC, "specialised for C=256");

    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t HW = H * W;

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto opts = xc.options();
    auto stream = at::cuda::getCurrentCUDAStream();

    int64_t chunk, nchunks64, ppc64;
    std::tie(chunk, nchunks64, ppc64) = plan_chunk(N, C, H, W);
    const int nchunks = (int)nchunks64;
    const int ppc     = (int)ppc64;
    const double M    = (double)CPG * (double)HW;

    auto w1c = conv1_weight.contiguous(at::MemoryFormat::ChannelsLast);
    auto w2c = conv2_weight.contiguous(at::MemoryFormat::ChannelsLast);

    auto x_cl  = at::empty({chunk, C, H, W},
                           opts.memory_format(at::MemoryFormat::ChannelsLast));
    auto z     = at::empty({chunk, C, H, W},
                           opts.memory_format(at::MemoryFormat::ChannelsLast));
    auto part  = at::empty({(int64_t)chunk * GG * nchunks, 2}, opts.dtype(at::kDouble));
    auto scale = at::empty({chunk, C}, opts);
    auto shift = at::empty({chunk, C}, opts);
    auto out   = at::empty({N, C, H, W}, opts);

    set_gn_params(norm1_weight.contiguous(), norm1_bias.contiguous(),
                  norm2_weight.contiguous(), norm2_bias.contiguous());
    set_io_ptrs(xc, out);

    bool win_set = false;
    if (chunk < N) win_set = set_l2_window(z);

    full_pipeline(w1c, w2c, eps, part, scale, shift, x_cl, z,
                  (int64_t)nchunks, (int64_t)ppc, M, N);

    if (win_set) {
        cudaStreamAttrValue av;
        memset(&av, 0, sizeof(av));
        av.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &av);
        cudaGetLastError();
    }
    return out;
}
'''

_CPP_SRC = r'''
#include <tuple>

std::tuple<int64_t, int64_t, int64_t> plan_chunk(int64_t N, int64_t C,
                                                 int64_t H, int64_t W);
bool set_l2_window(torch::Tensor z);
void set_io_ptrs(torch::Tensor x, torch::Tensor out);
void set_gn_params(torch::Tensor n1w, torch::Tensor n1b,
                   torch::Tensor n2w, torch::Tensor n2b);
void k1_chunk(int64_t elem_off, torch::Tensor x_cl);
torch::Tensor interior(torch::Tensor x_cl,
                       torch::Tensor w1c, torch::Tensor w2c,
                       double eps,
                       torch::Tensor part, torch::Tensor scale, torch::Tensor shift,
                       torch::Tensor z,
                       int64_t nchunks, int64_t ppc, double M);
void k5_chunk(torch::Tensor y2, int64_t elem_off,
              torch::Tensor scale, torch::Tensor shift, int64_t C);
void full_pipeline(torch::Tensor w1c, torch::Tensor w2c, double eps,
                   torch::Tensor part, torch::Tensor scale, torch::Tensor shift,
                   torch::Tensor x_cl, torch::Tensor z,
                   int64_t nchunks, int64_t ppc, double M, int64_t N);
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight,
                          torch::Tensor norm1_weight,
                          torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight,
                          torch::Tensor norm2_weight,
                          torch::Tensor norm2_bias,
                          double eps);
'''

_ext = load_inline(
    name="vae_resblock_fused_v5_fullgraph",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["plan_chunk", "set_l2_window", "set_io_ptrs", "set_gn_params",
               "k1_chunk", "interior", "k5_chunk", "full_pipeline", "fused_block"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-gencode=arch=compute_90,code=sm_90",
    ],
)


class ModelNew(nn.Module):
    """Whole-forward CUDA-graph capture; caller I/O + GN params via __constant__."""

    def __init__(self):
        super().__init__()
        self._ext = _ext
        self._state = {}          # cap 8 entries

    # ---------------- plan item 10: build static buffers + capture the FULL forward --
    def _build_full(self, x, w1, n1w, n1b, w2, n2w, n2b, eps):
        try:
            N, C, H, W = int(x.size(0)), int(x.size(1)), int(x.size(2)), int(x.size(3))
            dev = x.device
            chunk, nchunks, ppc = self._ext.plan_chunk(N, C, H, W)
            chunk, nchunks, ppc = int(chunk), int(nchunks), int(ppc)
            M = 8.0 * float(H * W)

            f32 = torch.float32
            x_cl = torch.empty((chunk, C, H, W), dtype=f32, device=dev,
                               memory_format=torch.channels_last)
            z = torch.empty((chunk, C, H, W), dtype=f32, device=dev,
                            memory_format=torch.channels_last)
            part = torch.empty((chunk * 32 * nchunks, 2), dtype=torch.float64, device=dev)
            scale = torch.empty((chunk, C), dtype=f32, device=dev)
            shift = torch.empty((chunk, C), dtype=f32, device=dev)

            w1c = w1.contiguous(memory_format=torch.channels_last).clone()
            w2c = w2.contiguous(memory_format=torch.channels_last).clone()

            x_cl.zero_()
            z.zero_()

            # scratch output for the warmup executions only
            warm_out = torch.empty((N, C, H, W), dtype=f32, device=dev)

            self._ext.set_gn_params(n1w.contiguous(), n1b.contiguous(),
                                    n2w.contiguous(), n2b.contiguous())
            self._ext.set_io_ptrs(x, warm_out)

            args = (w1c, w2c, float(eps), part, scale, shift, x_cl, z,
                    nchunks, ppc, M, N)

            # ---- persisting-L2 window, set BEFORE capture ------------------------
            cap_s = torch.cuda.Stream(device=dev)
            if chunk < N:
                self._ext.set_l2_window(z)
                with torch.cuda.stream(cap_s):
                    self._ext.set_l2_window(z)

            # ---- warmup x3 on a side stream (locks the cuDNN plan) ---------------
            cur = torch.cuda.current_stream(dev)
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._ext.full_pipeline(*args)
            cur.wait_stream(s)
            torch.cuda.synchronize(dev)

            # ---- fresh per-key pool + capture of the WHOLE forward ---------------
            pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            try:
                ctx = torch.cuda.graph(g, pool=pool, stream=cap_s)
            except TypeError:
                ctx = torch.cuda.graph(g, pool=pool)
            with ctx:
                self._ext.full_pipeline(*args)
            torch.cuda.synchronize(dev)

            del warm_out

            return {
                "mode": "full", "g": g, "pool": pool, "chunk": chunk,
                "x_cl": x_cl, "z": z, "part": part,
                "scale": scale, "shift": shift,
                "w1c": w1c, "w2c": w2c,
            }
        except Exception:
            return None

    # ---------------- fallback: capture only the interior (per-chunk replay) --------
    def _build_interior(self, x, w1, n1w, n1b, w2, n2w, n2b, eps):
        try:
            N, C, H, W = int(x.size(0)), int(x.size(1)), int(x.size(2)), int(x.size(3))
            dev = x.device
            chunk, nchunks, ppc = self._ext.plan_chunk(N, C, H, W)
            chunk, nchunks, ppc = int(chunk), int(nchunks), int(ppc)
            M = 8.0 * float(H * W)

            f32 = torch.float32
            x_cl = torch.empty((chunk, C, H, W), dtype=f32, device=dev,
                               memory_format=torch.channels_last)
            z = torch.empty((chunk, C, H, W), dtype=f32, device=dev,
                            memory_format=torch.channels_last)
            part = torch.empty((chunk * 32 * nchunks, 2), dtype=torch.float64, device=dev)
            scale = torch.empty((chunk, C), dtype=f32, device=dev)
            shift = torch.empty((chunk, C), dtype=f32, device=dev)

            w1c = w1.contiguous(memory_format=torch.channels_last).clone()
            w2c = w2.contiguous(memory_format=torch.channels_last).clone()
            x_cl.zero_()

            self._ext.set_gn_params(n1w.contiguous(), n1b.contiguous(),
                                    n2w.contiguous(), n2b.contiguous())

            args = (x_cl, w1c, w2c, float(eps), part, scale, shift, z,
                    nchunks, ppc, M)

            cap_s = torch.cuda.Stream(device=dev)
            if chunk < N:
                self._ext.set_l2_window(z)
                with torch.cuda.stream(cap_s):
                    self._ext.set_l2_window(z)

            cur = torch.cuda.current_stream(dev)
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = self._ext.interior(*args)
            cur.wait_stream(s)
            torch.cuda.synchronize(dev)

            pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            try:
                ctx = torch.cuda.graph(g, pool=pool, stream=cap_s)
            except TypeError:
                ctx = torch.cuda.graph(g, pool=pool)
            with ctx:
                y2_s = self._ext.interior(*args)
            torch.cuda.synchronize(dev)

            return {
                "mode": "interior", "g": g, "pool": pool, "chunk": chunk,
                "x_cl": x_cl, "z": z, "part": part,
                "scale": scale, "shift": shift,
                "w1c": w1c, "w2c": w2c, "y2": y2_s,
            }
        except Exception:
            return None

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
        e = float(eps)
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
                and x.size(1) == 256 and conv1_weight.dtype == torch.float32
                and conv2_weight.dtype == torch.float32):
            xc = x if x.is_contiguous() else x.contiguous()

            dev_idx = xc.device.index if xc.device.index is not None else 0
            key = (tuple(xc.shape), xc.dtype, dev_idx, e)
            if key in self._state:
                st = self._state[key]
            else:
                st = self._build_full(xc, conv1_weight, norm1_weight, norm1_bias,
                                      conv2_weight, norm2_weight, norm2_bias, e)
                if st is None:
                    st = self._build_interior(xc, conv1_weight, norm1_weight,
                                              norm1_bias, conv2_weight,
                                              norm2_weight, norm2_bias, e)
                if len(self._state) >= 8:
                    self._state.clear()
                self._state[key] = st

            if st is None:
                return self._ext.fused_block(xc, conv1_weight, norm1_weight,
                                             norm1_bias, conv2_weight,
                                             norm2_weight, norm2_bias, e)

            # ---- steady state: 2 weight copies + 4 DMA symbol writes + 1 replay ---
            st["w1c"].copy_(conv1_weight)
            st["w2c"].copy_(conv2_weight)
            self._ext.set_gn_params(norm1_weight.contiguous(),
                                    norm1_bias.contiguous(),
                                    norm2_weight.contiguous(),
                                    norm2_bias.contiguous())

            out = torch.empty(xc.shape, dtype=xc.dtype, device=xc.device)
            self._ext.set_io_ptrs(xc, out)

            if st["mode"] == "full":
                st["g"].replay()
                return out

            # interior-graph fallback: per-chunk enqueue around the captured interior
            C = int(xc.size(1))
            HW = int(xc.size(2)) * int(xc.size(3))
            chunk = st["chunk"]
            N = int(xc.size(0))
            g = st["g"]
            for off in range(0, N, chunk):
                elem_off = off * C * HW
                self._ext.k1_chunk(elem_off, st["x_cl"])
                g.replay()
                self._ext.k5_chunk(st["y2"], elem_off, st["scale"], st["shift"], C)
            return out

        # ---- fallback: exact reference semantics for unsupported configurations ----
        residual = x
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm1_weight, bias=norm1_bias, eps=e)
        out = F.silu(out)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, 32, weight=norm2_weight, bias=norm2_bias, eps=e)
        out = F.silu(out)
        return out + residual
