# =============================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# Round change: CUDA_Graph_Capture_Replay_StaticBuffers (refined).
#   ALL 11 kernels of the forward now live in ONE explicitly-constructed
#   cudaGraph (cudaGraphAddKernelNode, not stream capture), instantiated once
#   per (B,C,H,W,eps) key over module-owned persistent scratch. Caller-owned
#   pointers (x, w1, w2, gamma/beta, out) are rebound HOST-SIDE per call with
#   cudaGraphExecKernelNodeSetParams on stored node handles -- no device
#   pointer table, no extra kernel, no memcpy, no sync. The explicit DAG makes
#   wt1 / wt2 / nchw2nhwc three concurrent roots so the wt2 weight transform
#   overlaps conv1.
#
#   Every __global__ kernel body, signature, block size and grid computation is
#   byte-identical to the BASE kernel.
#
# 1) GRANULARITY: (D) fully rewrite forward.
#    The whole reference forward (2x [Conv3x3 -> GroupNorm -> SiLU] + residual)
#    runs in kernels built by one load_inline extension. The vendor conv is
#    OWNED here: a tiled implicit GEMM in NHWC with TF32 wmma fragments and
#    fp32 accumulate. No cuDNN / at::conv2d anywhere.
#
# 2) OPS REPLACED: F.conv2d x2, F.group_norm x2, F.silu x2, residual add, plus
#    the NCHW<->NHWC conversions cuDNN was doing internally.
#
# 3) FUSION MAP: wt_kernel, nchw2nhwc, conv_gemm_kernel, gstats_partial/final,
#    norm_silu, final_epilogue (GN2 + SiLU + residual + NHWC->NCHW in one write).
#
# PRECISION: fp32 storage everywhere, fp32 wmma accumulate, TF32 tensor cores
# for the GEMM only (matches cudnn.allow_tf32 default).
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <map>
#include <tuple>
#include <mutex>
#include <memory>
#include <cstring>

using namespace nvcuda;

#define BM 128
#define BN 64
#define BK 32
#define LDA 36
#define LDB 68
#define NTHREADS 256

// ---------------------------------------------------------------- weights ---
// (plan item 1) BASE signature / BASE body, no pointer-table indirection.
__global__ void wt_kernel(const float* __restrict__ w, float* __restrict__ wt,
                          int C, int N, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int n  = idx % N;
    int k  = idx / N;
    int c  = k % C;
    int rs = k / C;
    wt[idx] = w[((long)n * C + c) * 9 + rs];
}

// -------------------------------------------------------------- transpose ---
__global__ void nchw2nhwc(const float* __restrict__ src, float* __restrict__ dst,
                          int C, int HW) {
    __shared__ float tile[32][33];
    int p0 = blockIdx.x * 32;
    int c0 = blockIdx.y * 32;
    int b  = blockIdx.z;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c = c0 + threadIdx.y + i * 8;
        int p = p0 + threadIdx.x;
        float v = 0.f;
        if (c < C && p < HW) v = src[((long)b * C + c) * HW + p];
        tile[threadIdx.y + i * 8][threadIdx.x] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int p = p0 + threadIdx.y + i * 8;
        int c = c0 + threadIdx.x;
        if (c < C && p < HW)
            dst[((long)b * HW + p) * C + c] = tile[threadIdx.x][threadIdx.y + i * 8];
    }
}

// ------------------------------------------------------- groupnorm moments ---
__global__ void gstats_partial(const float* __restrict__ y,
                               float* __restrict__ psum, float* __restrict__ psq,
                               int HW, int C, int G, int cpg, int nchunk) {
    __shared__ float shm[64];
    int chunk = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
    int per   = (HW + nchunk - 1) / nchunk;
    int start = chunk * per;
    int end   = start + per; if (end > HW) end = HW;

    float s = 0.f, sq = 0.f;
    int vpp = cpg >> 2;
    int nv  = (end > start) ? (end - start) * vpp : 0;
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        int pi = i / vpp;
        int j  = (i - pi * vpp) * 4;
        const float4 v = *(const float4*)(y + ((long)(b * HW + start + pi)) * C + g * cpg + j);
        s  += v.x + v.y + v.z + v.w;
        sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s, off);
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    }
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nw   = blockDim.x >> 5;
    if (lane == 0) { shm[warp] = s; shm[32 + warp] = sq; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, c2 = 0.f;
        for (int i = 0; i < nw; ++i) { a += shm[i]; c2 += shm[32 + i]; }
        long o = (long)(b * G + g) * nchunk + chunk;
        psum[o] = a; psq[o] = c2;      // always written, even for empty chunks
    }
}

__global__ void gstats_final(const float* __restrict__ psum, const float* __restrict__ psq,
                             float* __restrict__ mean, float* __restrict__ rstd,
                             int nchunk, float count, float eps) {
    int idx = blockIdx.x;
    float s = 0.f, sq = 0.f;
    for (int i = threadIdx.x; i < nchunk; i += 32) {
        s  += psum[(long)idx * nchunk + i];
        sq += psq[(long)idx * nchunk + i];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s, off);
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (threadIdx.x == 0) {
        float m = s / count;
        float var = sq / count - m * m;
        if (!(var > 0.f)) var = 0.f;
        mean[idx] = m;
        rstd[idx] = rsqrtf(var + eps);
    }
}

// ------------------------------------------- groupnorm affine + SiLU (NHWC) ---
__global__ void norm_silu(float* __restrict__ y,
                          const float* __restrict__ gamma, const float* __restrict__ beta,
                          const float* __restrict__ mean, const float* __restrict__ rstd,
                          int HW, int C, int G, int cpg, long nvec4) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvec4) return;
    long e   = idx * 4;
    int  ch  = (int)(e % C);
    long row = e / C;
    int  b   = (int)(row / HW);
    int  g   = ch / cpg;
    float m = mean[b * G + g], rs = rstd[b * G + g];
    float4 v = *(float4*)(y + e);
    float t[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float z = (t[i] - m) * rs * gamma[ch + i] + beta[ch + i];
        t[i] = z / (1.f + expf(-z));
    }
    *(float4*)(y + e) = make_float4(t[0], t[1], t[2], t[3]);
}

// ------------------- groupnorm + SiLU + residual + NHWC->NCHW (fused write) ---
__global__ void final_epilogue(const float* __restrict__ y, const float* __restrict__ xres,
                               const float* __restrict__ gamma, const float* __restrict__ beta,
                               const float* __restrict__ mean, const float* __restrict__ rstd,
                               float* __restrict__ out, int C, int HW, int G, int cpg) {
    __shared__ float tile[32][33];
    int p0 = blockIdx.x * 32;
    int c0 = blockIdx.y * 32;
    int b  = blockIdx.z;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int p = p0 + threadIdx.y + i * 8;
        int c = c0 + threadIdx.x;
        float v = 0.f;
        if (p < HW && c < C) {
            v = y[((long)b * HW + p) * C + c];
            int g = c / cpg;
            float z = (v - mean[b * G + g]) * rstd[b * G + g] * gamma[c] + beta[c];
            v = z / (1.f + expf(-z));
        }
        tile[threadIdx.y + i * 8][threadIdx.x] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c = c0 + threadIdx.y + i * 8;
        int p = p0 + threadIdx.x;
        if (p < HW && c < C) {
            long o = ((long)b * C + c) * HW + p;
            out[o] = tile[threadIdx.x][threadIdx.y + i * 8] + xres[o];
        }
    }
}

// ------------------------------------------- implicit GEMM conv3x3 (NHWC) ---
// M = B*H*W (rows), N = out channels, K = C*9 with k = ((r*3+s)*C + c).
// UNTOUCHED: 80 regs/thread, 29696 B static shared memory.
__global__ __launch_bounds__(NTHREADS) void conv_gemm_kernel(
        const float* __restrict__ A, const float* __restrict__ WT,
        float* __restrict__ Y, int M, int H, int W, int C, int N) {
    __shared__ __align__(16) float smem[BM * LDA + BK * LDB];
    __shared__ int sh_b[BM], sh_oh[BM], sh_ow[BM];
    float* As = smem;
    float* Bs = smem + BM * LDA;

    const int tid = threadIdx.x;
    const int m0  = blockIdx.x * BM;
    const int n0  = blockIdx.y * BN;
    const int HW  = H * W;

    for (int i = tid; i < BM; i += NTHREADS) {
        int p = m0 + i;
        if (p < M) {
            int b   = p / HW;
            int rem = p - b * HW;
            int oh  = rem / W;
            sh_b[i] = b; sh_oh[i] = oh; sh_ow[i] = rem - oh * W;
        } else {
            sh_b[i] = -1; sh_oh[i] = 0; sh_ow[i] = 0;
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    const int warp = tid >> 5;
    const int wm   = warp >> 1;
    const int wn   = warp & 1;
    const int arow = tid >> 3, avec = tid & 7;
    const int brow = tid >> 4, bvec = tid & 15;

    __syncthreads();

    const int numK = C * 9;
    for (int kt = 0; kt < numK; kt += BK) {
        int rs = kt / C;
        int c0 = kt - rs * C;
        int r  = rs / 3;
        int s  = rs - r * 3;

        #pragma unroll
        for (int pass = 0; pass < BM / 32; ++pass) {
            int rowl = arow + pass * 32;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            int bb = sh_b[rowl];
            if (bb >= 0) {
                int ih = sh_oh[rowl] + r - 1;
                int iw = sh_ow[rowl] + s - 1;
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                    const float* src = A + ((long)(bb * HW + ih * W + iw)) * C + c0 + avec * 4;
                    v = *(const float4*)src;
                }
            }
            *(float4*)&As[rowl * LDA + avec * 4] = v;
        }
        #pragma unroll
        for (int pass = 0; pass < BK / 16; ++pass) {
            int kl = brow + pass * 16;
            float4 w = *(const float4*)(WT + (long)(kt + kl) * N + n0 + bvec * 4);
            *(float4*)&Bs[kl * LDB + bvec * 4] = w;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK / 8; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(af[i], As + (wm * 32 + i * 16) * LDA + kk * 8, LDA);
                #pragma unroll
                for (int t = 0; t < af[i].num_elements; ++t)
                    af[i].x[t] = wmma::__float_to_tf32(af[i].x[t]);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(bf[j], Bs + (kk * 8) * LDB + wn * 32 + j * 16, LDB);
                #pragma unroll
                for (int t = 0; t < bf[j].num_elements; ++t)
                    bf[j].x[t] = wmma::__float_to_tf32(bf[j].x[t]);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Y has ceil(M/BM)*BM rows, so full 16-row fragment stores are always legal.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int row = m0 + wm * 32 + i * 16;
            int col = n0 + wn * 32 + j * 16;
            wmma::store_matrix_sync(Y + (long)row * N + col, acc[i][j], N, wmma::mem_row_major);
        }
    }
}

// ============================================================================
// Plan item 3: POD argument storage. cudaKernelNodeParams::kernelParams holds
// POINTERS to these fields, so the storage MUST live inside the (address-
// stable) Ctx, never on the stack.
// ============================================================================
struct Args {
    // caller-owned (rebound every forward)
    const float* w1   = nullptr;
    const float* w2   = nullptr;
    const float* xsrc = nullptr;
    const float* g1   = nullptr;
    const float* b1   = nullptr;
    const float* g2   = nullptr;
    const float* b2   = nullptr;
    float*       out  = nullptr;
    // module-owned scratch (frozen for the process lifetime)
    float* xn   = nullptr;
    float* y1   = nullptr;
    float* y2   = nullptr;
    float* wt1  = nullptr;
    float* wt2  = nullptr;
    float* psum = nullptr;
    float* psq  = nullptr;
    float* mean1 = nullptr; float* rstd1 = nullptr;
    float* mean2 = nullptr; float* rstd2 = nullptr;
    // scalars
    int  C = 0, N = 0, total = 0, HW = 0, M = 0, H = 0, W = 0;
    int  G = 32, cpg = 0, nchunk = 1;
    long nvec4 = 0;
    float count = 0.f, eps = 0.f;
};

// ==================== persistent scratch + CUDA-graph cache =================
// Plan item 2/3: file-static Ctx holding every scratch tensor for one
// (B,C,H,W,eps) key, the 11 node handles, the 11 kernel-node param blocks and
// the argument-pointer arrays.
struct Ctx {
    torch::Tensor xn, y1, y2, wt1, wt2, psum, psq, mean1, rstd1, mean2, rstd2;

    cudaGraph_t          g    = nullptr;
    cudaGraphExec_t      exec = nullptr;
    cudaGraphNode_t      node[11];
    cudaKernelNodeParams kp[11];
    void*                argp[11][12];

    Args a;
    bool graph_tried = false;

    int B = 0, C = 0, H = 0, W = 0, HW = 0, G = 32, cpg = 0;
    int M = 0, gridm = 0, nchunk = 1;
    long nvec4 = 0;
    float eps = 0.f;

    dim3 gw, gt, gc, gp, gf, gn;   // base grid formulas, precomputed once

    Ctx() {
        std::memset(node, 0, sizeof(node));
        std::memset(kp,   0, sizeof(kp));
        std::memset(argp, 0, sizeof(argp));
    }
};

using Key = std::tuple<int, int, int, int, int>;   // B,C,H,W,eps-bits

// leaked on purpose: avoids static-destruction order issues with the CUDA ctx
static std::map<Key, Ctx>& g_cache() {
    static std::map<Key, Ctx>* m = new std::map<Key, Ctx>();
    return *m;
}
static std::mutex g_mu;

// Plan item 3/10: single allocation site; shapes identical to the base kernel.
static void alloc_ctx(Ctx& c, int B, int C, int H, int W, float eps,
                      const at::TensorOptions& opts) {
    c.B = B; c.C = C; c.H = H; c.W = W; c.eps = eps;
    c.HW  = H * W;
    c.G   = 32;
    c.cpg = C / c.G;
    const long Ml = (long)B * c.HW;
    c.M     = (int)Ml;
    c.gridm = (c.M + BM - 1) / BM;
    const long Mpad = (long)c.gridm * BM;
    c.nvec4 = Ml * (long)C / 4;

    int nchunk = c.HW / 2048; if (nchunk < 1) nchunk = 1; if (nchunk > 64) nchunk = 64;
    c.nchunk = nchunk;

    c.xn    = torch::empty({Ml, (long)C}, opts);
    c.y1    = torch::empty({Mpad, (long)C}, opts);
    c.y2    = torch::empty({Mpad, (long)C}, opts);
    c.wt1   = torch::empty({(long)C * 9, (long)C}, opts);
    c.wt2   = torch::empty({(long)C * 9, (long)C}, opts);
    c.psum  = torch::empty({(long)B * c.G * nchunk}, opts);
    c.psq   = torch::empty({(long)B * c.G * nchunk}, opts);
    c.mean1 = torch::empty({(long)B * c.G}, opts);
    c.rstd1 = torch::empty({(long)B * c.G}, opts);
    c.mean2 = torch::empty({(long)B * c.G}, opts);
    c.rstd2 = torch::empty({(long)B * c.G}, opts);

    Args& a = c.a;
    a.xn = c.xn.data_ptr<float>();  a.y1 = c.y1.data_ptr<float>();
    a.y2 = c.y2.data_ptr<float>();  a.wt1 = c.wt1.data_ptr<float>();
    a.wt2 = c.wt2.data_ptr<float>();
    a.psum = c.psum.data_ptr<float>(); a.psq = c.psq.data_ptr<float>();
    a.mean1 = c.mean1.data_ptr<float>(); a.rstd1 = c.rstd1.data_ptr<float>();
    a.mean2 = c.mean2.data_ptr<float>(); a.rstd2 = c.rstd2.data_ptr<float>();

    a.C = C; a.N = C; a.total = C * 9 * C; a.HW = c.HW;
    a.M = c.M; a.H = H; a.W = W; a.G = c.G; a.cpg = c.cpg;
    a.nchunk = nchunk; a.nvec4 = c.nvec4;
    a.count = (float)((long)c.HW * c.cpg);
    a.eps = eps;

    // Plan item 4: BASE grid/block formulas, computed once.
    c.gw = dim3((unsigned)((a.total + 255) / 256));
    c.gt = dim3((unsigned)((c.HW + 31) / 32), (unsigned)((C + 31) / 32), (unsigned)B);
    c.gc = dim3((unsigned)c.gridm, (unsigned)(C / BN));
    c.gp = dim3((unsigned)nchunk, (unsigned)c.G, (unsigned)B);
    c.gf = dim3((unsigned)(B * c.G));
    c.gn = dim3((unsigned)((c.nvec4 + 255) / 256));
}

// Plan item 9: eager fallback -- the 11 kernels in BASE order on one stream.
static void launch_all(Ctx& c, cudaStream_t s) {
    Args& a = c.a;
    const dim3 tb(32, 8);
    wt_kernel<<<c.gw, 256, 0, s>>>(a.w1, a.wt1, a.C, a.N, a.total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wt_kernel<<<c.gw, 256, 0, s>>>(a.w2, a.wt2, a.C, a.N, a.total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nchw2nhwc<<<c.gt, tb, 0, s>>>(a.xsrc, a.xn, a.C, a.HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    conv_gemm_kernel<<<c.gc, NTHREADS, 0, s>>>(a.xn, a.wt1, a.y1, a.M, a.H, a.W, a.C, a.N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gstats_partial<<<c.gp, 128, 0, s>>>(a.y1, a.psum, a.psq, a.HW, a.C, a.G, a.cpg, a.nchunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gstats_final<<<c.gf, 32, 0, s>>>(a.psum, a.psq, a.mean1, a.rstd1, a.nchunk, a.count, a.eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    norm_silu<<<c.gn, 256, 0, s>>>(a.y1, a.g1, a.b1, a.mean1, a.rstd1,
                                   a.HW, a.C, a.G, a.cpg, a.nvec4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    conv_gemm_kernel<<<c.gc, NTHREADS, 0, s>>>(a.y1, a.wt2, a.y2, a.M, a.H, a.W, a.C, a.N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gstats_partial<<<c.gp, 128, 0, s>>>(a.y2, a.psum, a.psq, a.HW, a.C, a.G, a.cpg, a.nchunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gstats_final<<<c.gf, 32, 0, s>>>(a.psum, a.psq, a.mean2, a.rstd2, a.nchunk, a.count, a.eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    final_epilogue<<<c.gt, tb, 0, s>>>(a.y2, a.xsrc, a.g2, a.b2, a.mean2, a.rstd2,
                                       a.out, a.C, a.HW, a.G, a.cpg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Plan item 3: bind argp[i][*] to the Ctx-resident Args fields.
static void bind_args(Ctx& c) {
    Args& a = c.a;
    void** p;
    p = c.argp[0];  p[0] = (void*)&a.w1;  p[1] = (void*)&a.wt1; p[2] = (void*)&a.C;
                    p[3] = (void*)&a.N;   p[4] = (void*)&a.total;
    p = c.argp[1];  p[0] = (void*)&a.w2;  p[1] = (void*)&a.wt2; p[2] = (void*)&a.C;
                    p[3] = (void*)&a.N;   p[4] = (void*)&a.total;
    p = c.argp[2];  p[0] = (void*)&a.xsrc; p[1] = (void*)&a.xn; p[2] = (void*)&a.C;
                    p[3] = (void*)&a.HW;
    p = c.argp[3];  p[0] = (void*)&a.xn;  p[1] = (void*)&a.wt1; p[2] = (void*)&a.y1;
                    p[3] = (void*)&a.M;   p[4] = (void*)&a.H;   p[5] = (void*)&a.W;
                    p[6] = (void*)&a.C;   p[7] = (void*)&a.N;
    p = c.argp[4];  p[0] = (void*)&a.y1;  p[1] = (void*)&a.psum; p[2] = (void*)&a.psq;
                    p[3] = (void*)&a.HW;  p[4] = (void*)&a.C;    p[5] = (void*)&a.G;
                    p[6] = (void*)&a.cpg; p[7] = (void*)&a.nchunk;
    p = c.argp[5];  p[0] = (void*)&a.psum; p[1] = (void*)&a.psq; p[2] = (void*)&a.mean1;
                    p[3] = (void*)&a.rstd1; p[4] = (void*)&a.nchunk;
                    p[5] = (void*)&a.count; p[6] = (void*)&a.eps;
    p = c.argp[6];  p[0] = (void*)&a.y1;  p[1] = (void*)&a.g1;  p[2] = (void*)&a.b1;
                    p[3] = (void*)&a.mean1; p[4] = (void*)&a.rstd1;
                    p[5] = (void*)&a.HW;  p[6] = (void*)&a.C;   p[7] = (void*)&a.G;
                    p[8] = (void*)&a.cpg; p[9] = (void*)&a.nvec4;
    p = c.argp[7];  p[0] = (void*)&a.y1;  p[1] = (void*)&a.wt2; p[2] = (void*)&a.y2;
                    p[3] = (void*)&a.M;   p[4] = (void*)&a.H;   p[5] = (void*)&a.W;
                    p[6] = (void*)&a.C;   p[7] = (void*)&a.N;
    p = c.argp[8];  p[0] = (void*)&a.y2;  p[1] = (void*)&a.psum; p[2] = (void*)&a.psq;
                    p[3] = (void*)&a.HW;  p[4] = (void*)&a.C;    p[5] = (void*)&a.G;
                    p[6] = (void*)&a.cpg; p[7] = (void*)&a.nchunk;
    p = c.argp[9];  p[0] = (void*)&a.psum; p[1] = (void*)&a.psq; p[2] = (void*)&a.mean2;
                    p[3] = (void*)&a.rstd2; p[4] = (void*)&a.nchunk;
                    p[5] = (void*)&a.count; p[6] = (void*)&a.eps;
    p = c.argp[10]; p[0] = (void*)&a.y2;  p[1] = (void*)&a.xsrc; p[2] = (void*)&a.g2;
                    p[3] = (void*)&a.b2;  p[4] = (void*)&a.mean2; p[5] = (void*)&a.rstd2;
                    p[6] = (void*)&a.out; p[7] = (void*)&a.C;    p[8] = (void*)&a.HW;
                    p[9] = (void*)&a.G;   p[10] = (void*)&a.cpg;
}

static inline void set_kp(cudaKernelNodeParams& k, void* func, dim3 g, dim3 b, void** params) {
    std::memset(&k, 0, sizeof(k));
    k.func           = func;
    k.gridDim        = g;
    k.blockDim       = b;
    k.sharedMemBytes = 0;   // all kernels use STATIC __shared__ memory
    k.kernelParams   = params;
    k.extra          = nullptr;
}

// Plan items 4/5/6: explicit graph construction + DAG + instantiation.
static bool build_graph(Ctx& c) {
    const dim3 tb(32, 8);
    bind_args(c);

    set_kp(c.kp[0],  (void*)wt_kernel,       c.gw, dim3(256), c.argp[0]);
    set_kp(c.kp[1],  (void*)wt_kernel,       c.gw, dim3(256), c.argp[1]);
    set_kp(c.kp[2],  (void*)nchw2nhwc,       c.gt, tb,        c.argp[2]);
    set_kp(c.kp[3],  (void*)conv_gemm_kernel,c.gc, dim3(NTHREADS), c.argp[3]);
    set_kp(c.kp[4],  (void*)gstats_partial,  c.gp, dim3(128), c.argp[4]);
    set_kp(c.kp[5],  (void*)gstats_final,    c.gf, dim3(32),  c.argp[5]);
    set_kp(c.kp[6],  (void*)norm_silu,       c.gn, dim3(256), c.argp[6]);
    set_kp(c.kp[7],  (void*)conv_gemm_kernel,c.gc, dim3(NTHREADS), c.argp[7]);
    set_kp(c.kp[8],  (void*)gstats_partial,  c.gp, dim3(128), c.argp[8]);
    set_kp(c.kp[9],  (void*)gstats_final,    c.gf, dim3(32),  c.argp[9]);
    set_kp(c.kp[10], (void*)final_epilogue,  c.gt, tb,        c.argp[10]);

    cudaError_t e = cudaGraphCreate(&c.g, 0);
    if (e != cudaSuccess) { c.g = nullptr; return false; }

    cudaGraphNode_t dep[2];
    // three concurrent roots: wt1, wt2, nchw2nhwc
    if (cudaGraphAddKernelNode(&c.node[0], c.g, nullptr, 0, &c.kp[0]) != cudaSuccess) return false;
    if (cudaGraphAddKernelNode(&c.node[1], c.g, nullptr, 0, &c.kp[1]) != cudaSuccess) return false;
    if (cudaGraphAddKernelNode(&c.node[2], c.g, nullptr, 0, &c.kp[2]) != cudaSuccess) return false;
    // conv1 needs wt1 + the NHWC input
    dep[0] = c.node[0]; dep[1] = c.node[2];
    if (cudaGraphAddKernelNode(&c.node[3], c.g, dep, 2, &c.kp[3]) != cudaSuccess) return false;
    dep[0] = c.node[3];
    if (cudaGraphAddKernelNode(&c.node[4], c.g, dep, 1, &c.kp[4]) != cudaSuccess) return false;
    dep[0] = c.node[4];
    if (cudaGraphAddKernelNode(&c.node[5], c.g, dep, 1, &c.kp[5]) != cudaSuccess) return false;
    dep[0] = c.node[5];
    if (cudaGraphAddKernelNode(&c.node[6], c.g, dep, 1, &c.kp[6]) != cudaSuccess) return false;
    // conv2 needs norm_silu(y1) + wt2  -> wt2 overlaps conv1
    dep[0] = c.node[6]; dep[1] = c.node[1];
    if (cudaGraphAddKernelNode(&c.node[7], c.g, dep, 2, &c.kp[7]) != cudaSuccess) return false;
    dep[0] = c.node[7];
    if (cudaGraphAddKernelNode(&c.node[8], c.g, dep, 1, &c.kp[8]) != cudaSuccess) return false;
    dep[0] = c.node[8];
    if (cudaGraphAddKernelNode(&c.node[9], c.g, dep, 1, &c.kp[9]) != cudaSuccess) return false;
    dep[0] = c.node[9];
    if (cudaGraphAddKernelNode(&c.node[10], c.g, dep, 1, &c.kp[10]) != cudaSuccess) return false;

    if (cudaGraphInstantiateWithFlags(&c.exec, c.g, 0) != cudaSuccess) { c.exec = nullptr; return false; }
    return true;
}

static inline void fill_caller_args(Ctx& c,
                                    const float* x, const float* w1, const float* w2,
                                    const float* g1, const float* b1,
                                    const float* g2, const float* b2, float* out) {
    Args& a = c.a;
    a.xsrc = x; a.w1 = w1; a.w2 = w2;
    a.g1 = g1; a.b1 = b1; a.g2 = g2; a.b2 = b2; a.out = out;
}

// ------------------------------------------------------------ orchestrator ---
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight, torch::Tensor norm1_weight, torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight, torch::Tensor norm2_weight, torch::Tensor norm2_bias,
                          double eps) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.dim() == 4, "NCHW expected");
    TORCH_CHECK(conv1_weight.size(2) == 3 && conv1_weight.size(3) == 3, "3x3 conv only");

    auto xc = x.is_contiguous()  ? x : x.contiguous();
    auto w1 = conv1_weight.is_contiguous() ? conv1_weight : conv1_weight.contiguous();
    auto w2 = conv2_weight.is_contiguous() ? conv2_weight : conv2_weight.contiguous();
    auto g1 = norm1_weight.is_contiguous() ? norm1_weight : norm1_weight.contiguous();
    auto b1 = norm1_bias.is_contiguous()   ? norm1_bias   : norm1_bias.contiguous();
    auto g2 = norm2_weight.is_contiguous() ? norm2_weight : norm2_weight.contiguous();
    auto b2 = norm2_bias.is_contiguous()   ? norm2_bias   : norm2_bias.contiguous();

    const int B = (int)xc.size(0), C = (int)xc.size(1);
    const int H = (int)xc.size(2), W = (int)xc.size(3);
    const int HW = H * W;
    const int G = 32;
    const int cpg = C / G;
    TORCH_CHECK(C % BN == 0 && C % BK == 0 && cpg % 4 == 0, "unsupported channel count");
    const long Ml = (long)B * HW;
    TORCH_CHECK(Ml < (1L << 30), "problem too large");
    (void)G;

    const float epsf = (float)eps;
    int epsbits; std::memcpy(&epsbits, &epsf, sizeof(int));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts   = xc.options();
    auto out    = torch::empty_like(xc);   // plan item 8: only allocation

    std::lock_guard<std::mutex> lk(g_mu);

    // Plan item 9: never build/launch a graph while the caller's stream is
    // itself being captured by someone else.
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap_st);
    const bool outer_capture = (cap_st != cudaStreamCaptureStatusNone);

    Key key = std::make_tuple(B, C, H, W, epsbits);
    std::unique_ptr<Ctx> localp;      // used only when the 8-key cache is full
    Ctx* ctxp = nullptr;
    bool fresh = false;

    auto& cache = g_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        ctxp = &it->second;
    } else if (cache.size() < 8) {
        Ctx& nc = cache[key];                            // in-place, address-stable
        alloc_ctx(nc, B, C, H, W, epsf, opts);           // plan items 2/3/10
        ctxp = &nc;
        fresh = true;
    } else {
        localp.reset(new Ctx());
        alloc_ctx(*localp, B, C, H, W, epsf, opts);      // plain eager path
        ctxp = localp.get();
    }
    Ctx& ctx = *ctxp;

    // Plan item 7: fresh caller pointers into the Ctx-resident arg storage.
    fill_caller_args(ctx,
                     xc.data_ptr<float>(), w1.data_ptr<float>(), w2.data_ptr<float>(),
                     g1.data_ptr<float>(), b1.data_ptr<float>(),
                     g2.data_ptr<float>(), b2.data_ptr<float>(),
                     out.data_ptr<float>());

    // ---------------- one-time eager warmup + explicit graph build ----------
    if (fresh && !outer_capture && !ctx.graph_tried) {
        ctx.graph_tried = true;
        launch_all(ctx, stream);                 // plan item 6: warmup on real data
        cudaStreamSynchronize(stream);
        cudaGetLastError();
        if (!build_graph(ctx)) {                 // plan item 9: any failure -> eager
            if (ctx.exec) { cudaGraphExecDestroy(ctx.exec); ctx.exec = nullptr; }
            if (ctx.g)    { cudaGraphDestroy(ctx.g);        ctx.g    = nullptr; }
        }
        cudaGetLastError();
    }

    if (ctx.exec != nullptr && !outer_capture) {
        // Plan item 7: rebind ONLY the five caller-pointer nodes, host-side.
        const int rebind[5] = {0, 1, 2, 6, 10};
        bool ok = true;
        for (int r = 0; r < 5; ++r) {
            int i = rebind[r];
            if (cudaGraphExecKernelNodeSetParams(ctx.exec, ctx.node[i], &ctx.kp[i]) != cudaSuccess) {
                ok = false; break;
            }
        }
        if (ok) {
            // Plan item 8: exactly ONE host launch for the whole forward.
            if (cudaGraphLaunch(ctx.exec, stream) != cudaSuccess) {
                cudaGetLastError();
                launch_all(ctx, stream);
            }
        } else {
            cudaGetLastError();
            launch_all(ctx, stream);             // plan item 9
        }
    } else {
        launch_all(ctx, stream);                 // plan item 9 fallback
    }

    return out;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_block(torch::Tensor x,
                          torch::Tensor conv1_weight, torch::Tensor norm1_weight, torch::Tensor norm1_bias,
                          torch::Tensor conv2_weight, torch::Tensor norm2_weight, torch::Tensor norm2_bias,
                          double eps);
"""

_ext = load_inline(
    name="vae_resblock_implicit_gemm_graph_full",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
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
)


class ModelNew(nn.Module):
    """Full forward-level replacement of the SOL 002 residual block.

    Granularity (D): the conv3x3 is our own TF32 implicit-GEMM kernel; the
    GroupNorm/SiLU/residual/layout work is fused into its surrounding passes.
    All 11 kernels are placed into ONE explicitly-built cudaGraph (kernel nodes
    + explicit DAG, three concurrent roots) per (B,C,H,W,eps) key over
    module-owned persistent scratch; caller-owned pointers are rebound
    host-side with cudaGraphExecKernelNodeSetParams, so a forward costs exactly
    one host launch and allocates only the output tensor.

    Stateless module (the reference has no parameters), so parameter parity is
    trivially preserved: all weights arrive as forward arguments and are used
    verbatim by the extension.
    """

    def __init__(self):
        super().__init__()
        self._ext = _ext

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        return self._ext.fused_block(
            x, conv1_weight, norm1_weight, norm1_bias,
            conv2_weight, norm2_weight, norm2_bias, float(eps)
        )
