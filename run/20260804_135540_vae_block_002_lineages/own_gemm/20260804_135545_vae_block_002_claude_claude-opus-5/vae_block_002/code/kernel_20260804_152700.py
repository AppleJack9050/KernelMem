# =============================================================================
# ModelNew — SOL 002_vae_conv3x3_groupnorm_silu_residual_fused
#
# Round change: CUDA_Graph_Capture_Replay_StaticBuffers, extended to S
# INDEPENDENT PER-BATCH-CHUNK CHAINS inside ONE explicitly-built cudaGraph.
#
#   The forward is expressed as an explicit graph DAG:
#       roots:  wt_kernel(w1) , wt_kernel(w2)
#       chain k (k = 0..S-1, no cross-chunk edges):
#           nchw2nhwc_k -> conv_gemm_k -> gstats_partial_k -> gstats_final_k
#                       -> norm_silu_k -> conv_gemm_k(2) -> gstats_partial_k(2)
#                       -> gstats_final_k(2) -> final_epilogue_k
#   Batches are fully independent (conv padding, GN stats, residual are all
#   per-b), so the DRAM-bound transpose/GN/epilogue nodes of one chunk
#   co-schedule into the idle warp slots / idle DRAM of another chunk's
#   compute-bound conv node, all from a SINGLE host launch.
#
#   Kernel bodies are byte-identical to the base kernel except two added
#   trailing int arguments (conv_gemm_kernel: mbase, gstats_partial: boff).
#
# 1) GRANULARITY: (D) fully rewrite forward. The vendor conv is OWNED here:
#    tiled implicit GEMM in NHWC with TF32 wmma fragments, fp32 accumulate.
#    No cuDNN / at::conv2d anywhere.
# 2) OPS REPLACED: F.conv2d x2, F.group_norm x2, F.silu x2, residual add,
#    plus the NCHW<->NHWC conversions cuDNN was doing internally.
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
#include <memory>
#include <tuple>
#include <vector>
#include <mutex>
#include <cstring>

using namespace nvcuda;

#define BM 128
#define BN 64
#define BK 32
#define LDA 36
#define LDB 68
#define NTHREADS 256
#define MAXS 4

// ---------------------------------------------------------------- weights ---
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
// Plan item 2: trailing `int boff`; b = boff + blockIdx.z. Body otherwise identical.
__global__ void gstats_partial(const float* __restrict__ y,
                               float* __restrict__ psum, float* __restrict__ psq,
                               int HW, int C, int G, int cpg, int nchunk, int boff) {
    __shared__ float shm[64];
    int chunk = blockIdx.x, g = blockIdx.y;
    int b = boff + blockIdx.z;
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
// Plan item 2: trailing `int mbase`; m0 = (blockIdx.x + mbase) * BM. Nothing else changes.
__global__ __launch_bounds__(NTHREADS) void conv_gemm_kernel(
        const float* __restrict__ A, const float* __restrict__ WT,
        float* __restrict__ Y, int M, int H, int W, int C, int N, int mbase) {
    __shared__ __align__(16) float smem[BM * LDA + BK * LDB];
    __shared__ int sh_b[BM], sh_oh[BM], sh_ow[BM];
    float* As = smem;
    float* Bs = smem + BM * LDA;

    const int tid = threadIdx.x;
    const int m0  = (blockIdx.x + mbase) * BM;
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

// ================= persistent scratch + explicit CUDA-graph DAG =============
// Plan item 6: every kernel argument lives in Ctx-owned storage (never on the
// stack) so cudaGraphExecKernelNodeSetParams can be re-issued per call.

struct WtArgs {
    const float* w; float* wt; int C, N, total; void* p[5];
    void bind() { p[0]=&w; p[1]=&wt; p[2]=&C; p[3]=&N; p[4]=&total; }
};
struct TrArgs {
    const float* src; float* dst; int C, HW; void* p[4];
    void bind() { p[0]=&src; p[1]=&dst; p[2]=&C; p[3]=&HW; }
};
struct ConvArgs {
    const float* A; const float* WT; float* Y; int M, H, W, C, N, mbase; void* p[9];
    void bind() { p[0]=&A; p[1]=&WT; p[2]=&Y; p[3]=&M; p[4]=&H; p[5]=&W; p[6]=&C; p[7]=&N; p[8]=&mbase; }
};
struct PartArgs {
    const float* y; float* psum; float* psq; int HW, C, G, cpg, nchunk, boff; void* p[9];
    void bind() { p[0]=&y; p[1]=&psum; p[2]=&psq; p[3]=&HW; p[4]=&C; p[5]=&G; p[6]=&cpg; p[7]=&nchunk; p[8]=&boff; }
};
struct FinArgs {
    const float* psum; const float* psq; float* mean; float* rstd; int nchunk; float count, eps; void* p[7];
    void bind() { p[0]=&psum; p[1]=&psq; p[2]=&mean; p[3]=&rstd; p[4]=&nchunk; p[5]=&count; p[6]=&eps; }
};
struct NsArgs {
    float* y; const float* gamma; const float* beta; const float* mean; const float* rstd;
    int HW, C, G, cpg; long nvec4; void* p[10];
    void bind() { p[0]=&y; p[1]=&gamma; p[2]=&beta; p[3]=&mean; p[4]=&rstd;
                  p[5]=&HW; p[6]=&C; p[7]=&G; p[8]=&cpg; p[9]=&nvec4; }
};
struct EpiArgs {
    const float* y; const float* xres; const float* gamma; const float* beta;
    const float* mean; const float* rstd; float* out; int C, HW, G, cpg; void* p[11];
    void bind() { p[0]=&y; p[1]=&xres; p[2]=&gamma; p[3]=&beta; p[4]=&mean; p[5]=&rstd;
                  p[6]=&out; p[7]=&C; p[8]=&HW; p[9]=&G; p[10]=&cpg; }
};

struct Ctx {
    torch::Tensor xn, y1, y2, wt1, wt2, psum, psq, mean1, rstd1, mean2, rstd2;
    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t exec  = nullptr;
    int B = 0, C = 0, H = 0, W = 0, HW = 0, G = 32, cpg = 0;
    int M = 0, gridm = 0, nchunk = 1, S = 1;
    int cb0[MAXS], cb1[MAXS];
    float eps = 0.f;

    WtArgs   a_wt[2];
    TrArgs   a_t[MAXS];
    ConvArgs a_c1[MAXS], a_c2[MAXS];
    PartArgs a_p1[MAXS], a_p2[MAXS];
    FinArgs  a_f1[MAXS], a_f2[MAXS];
    NsArgs   a_ns[MAXS];
    EpiArgs  a_e[MAXS];

    std::vector<cudaKernelNodeParams> knp;   // topological launch order
    std::vector<cudaGraphNode_t>      nodes; // parallel to knp
    std::vector<int>                  upd;   // nodes holding caller pointers
};

using Key = std::tuple<int, int, int, int, int>;   // B,C,H,W,eps-bits
static std::map<Key, std::unique_ptr<Ctx>> g_cache;
static std::mutex g_mu;

// Plan items 1/3: single allocation site (shapes identical to base kernel) plus
// the batch-chunk split S.
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

    // ---- Plan item 3: S = (HW%BM==0 && B>=2) ? min(B,4) : 1, plus a guard so
    //      every conv node still launches >= 114 blocks (one wave per SM).
    int S = 1;
    if ((c.HW % BM) == 0 && B >= 2) {
        S = (B < MAXS) ? B : MAXS;
        while (S > 1) {
            long minb   = B / S;
            long blocks = minb * (long)(c.HW / BM) * (long)(C / BN);
            if (blocks >= 114) break;
            --S;
        }
    }
    c.S = S;
    for (int k = 0; k < S; ++k) {
        c.cb0[k] = (int)((long)k * B / S);
        c.cb1[k] = (int)((long)(k + 1) * B / S);
    }
}

// Plan item 4: fixed launch shapes / scalar args for every node (per chunk).
static void build_nodes(Ctx& c) {
    c.knp.clear(); c.nodes.clear(); c.upd.clear();
    c.knp.reserve(2 + 9 * MAXS);
    c.nodes.reserve(2 + 9 * MAXS);

    auto add = [&](const void* func, dim3 grid, dim3 block, void** params) {
        cudaKernelNodeParams np;
        std::memset(&np, 0, sizeof(np));
        np.func = (void*)func;
        np.gridDim = grid; np.blockDim = block;
        np.sharedMemBytes = 0;               // all smem is static
        np.kernelParams = params; np.extra = nullptr;
        c.knp.push_back(np);
    };

    const int C = c.C, HW = c.HW, G = c.G, cpg = c.cpg;
    const int wtot = C * 9 * C;

    // ---- roots: weight relayout (caller-owned w1/w2) -> indices 0,1
    for (int i = 0; i < 2; ++i) {
        c.a_wt[i].C = C; c.a_wt[i].N = C; c.a_wt[i].total = wtot;
        c.a_wt[i].w = nullptr; c.a_wt[i].wt = nullptr;
        c.a_wt[i].bind();
        add((const void*)wt_kernel, dim3((wtot + 255) / 256), dim3(256), c.a_wt[i].p);
        c.upd.push_back(i);
    }

    dim3 tb(32, 8);
    for (int k = 0; k < c.S; ++k) {
        const int b0 = c.cb0[k], b1 = c.cb1[k], nb = b1 - b0;
        const long rows = (long)nb * HW;
        const int  gm   = (int)((rows + BM - 1) / BM);
        const int base  = (int)c.knp.size();

        // t_k : nchw2nhwc  (reads caller x)
        c.a_t[k].C = C; c.a_t[k].HW = HW; c.a_t[k].src = nullptr; c.a_t[k].dst = nullptr;
        c.a_t[k].bind();
        add((const void*)nchw2nhwc, dim3((HW + 31) / 32, (C + 31) / 32, nb), tb, c.a_t[k].p);

        // c1_k
        c.a_c1[k].A = nullptr; c.a_c1[k].WT = nullptr; c.a_c1[k].Y = nullptr;
        c.a_c1[k].M = c.M; c.a_c1[k].H = c.H; c.a_c1[k].W = c.W; c.a_c1[k].C = C; c.a_c1[k].N = C;
        c.a_c1[k].mbase = (int)((long)b0 * HW / BM);
        c.a_c1[k].bind();
        add((const void*)conv_gemm_kernel, dim3(gm, C / BN), dim3(NTHREADS), c.a_c1[k].p);

        // p1_k
        c.a_p1[k].y = nullptr; c.a_p1[k].psum = nullptr; c.a_p1[k].psq = nullptr;
        c.a_p1[k].HW = HW; c.a_p1[k].C = C; c.a_p1[k].G = G; c.a_p1[k].cpg = cpg;
        c.a_p1[k].nchunk = c.nchunk; c.a_p1[k].boff = b0;
        c.a_p1[k].bind();
        add((const void*)gstats_partial, dim3(c.nchunk, G, nb), dim3(128), c.a_p1[k].p);

        // f1_k
        c.a_f1[k].psum = nullptr; c.a_f1[k].psq = nullptr;
        c.a_f1[k].mean = nullptr; c.a_f1[k].rstd = nullptr;
        c.a_f1[k].nchunk = c.nchunk;
        c.a_f1[k].count  = (float)((long)HW * cpg);
        c.a_f1[k].eps    = c.eps;
        c.a_f1[k].bind();
        add((const void*)gstats_final, dim3(nb * G), dim3(32), c.a_f1[k].p);

        // ns_k (reads caller gamma1/beta1)
        c.a_ns[k].y = nullptr; c.a_ns[k].gamma = nullptr; c.a_ns[k].beta = nullptr;
        c.a_ns[k].mean = nullptr; c.a_ns[k].rstd = nullptr;
        c.a_ns[k].HW = HW; c.a_ns[k].C = C; c.a_ns[k].G = G; c.a_ns[k].cpg = cpg;
        c.a_ns[k].nvec4 = (long)nb * HW * (long)C / 4;
        c.a_ns[k].bind();
        add((const void*)norm_silu, dim3((int)((c.a_ns[k].nvec4 + 255) / 256)), dim3(256), c.a_ns[k].p);

        // c2_k
        c.a_c2[k] = c.a_c1[k];
        c.a_c2[k].bind();
        add((const void*)conv_gemm_kernel, dim3(gm, C / BN), dim3(NTHREADS), c.a_c2[k].p);

        // p2_k
        c.a_p2[k] = c.a_p1[k];
        c.a_p2[k].bind();
        add((const void*)gstats_partial, dim3(c.nchunk, G, nb), dim3(128), c.a_p2[k].p);

        // f2_k
        c.a_f2[k] = c.a_f1[k];
        c.a_f2[k].bind();
        add((const void*)gstats_final, dim3(nb * G), dim3(32), c.a_f2[k].p);

        // e_k (reads caller x + gamma2/beta2, writes caller out)
        c.a_e[k].y = nullptr; c.a_e[k].xres = nullptr; c.a_e[k].gamma = nullptr;
        c.a_e[k].beta = nullptr; c.a_e[k].mean = nullptr; c.a_e[k].rstd = nullptr;
        c.a_e[k].out = nullptr;
        c.a_e[k].C = C; c.a_e[k].HW = HW; c.a_e[k].G = G; c.a_e[k].cpg = cpg;
        c.a_e[k].bind();
        add((const void*)final_epilogue, dim3((HW + 31) / 32, (C + 31) / 32, nb), tb, c.a_e[k].p);

        c.upd.push_back(base + 0);   // t_k
        c.upd.push_back(base + 4);   // ns_k
        c.upd.push_back(base + 8);   // e_k
    }
}

// Plan item 7: refresh only the caller-owned pointers (+ the Ctx-relative
// per-chunk offsets they imply) in the persistent arg storage.
static void set_call_ptrs(Ctx& c,
                          const float* xp, const float* w1p, const float* w2p,
                          const float* g1p, const float* b1p,
                          const float* g2p, const float* b2p, float* outp) {
    const int C = c.C, HW = c.HW, G = c.G;

    c.a_wt[0].w = w1p; c.a_wt[0].wt = c.wt1.data_ptr<float>();
    c.a_wt[1].w = w2p; c.a_wt[1].wt = c.wt2.data_ptr<float>();

    float* xn = c.xn.data_ptr<float>();
    float* y1 = c.y1.data_ptr<float>();
    float* y2 = c.y2.data_ptr<float>();
    float* ps = c.psum.data_ptr<float>();
    float* pq = c.psq.data_ptr<float>();
    float* m1 = c.mean1.data_ptr<float>(); float* r1 = c.rstd1.data_ptr<float>();
    float* m2 = c.mean2.data_ptr<float>(); float* r2 = c.rstd2.data_ptr<float>();

    for (int k = 0; k < c.S; ++k) {
        const long b0 = c.cb0[k];
        const long offN = b0 * (long)HW * C;   // NHWC row offset
        const long offC = b0 * (long)C * HW;   // NCHW plane offset
        const long offG = b0 * G;
        const long offP = b0 * (long)G * c.nchunk;

        c.a_t[k].src = xp + offC;      c.a_t[k].dst = xn + offN;

        c.a_c1[k].A = xn;              c.a_c1[k].WT = c.wt1.data_ptr<float>(); c.a_c1[k].Y = y1;
        c.a_p1[k].y = y1;              c.a_p1[k].psum = ps;  c.a_p1[k].psq = pq;
        c.a_f1[k].psum = ps + offP;    c.a_f1[k].psq = pq + offP;
        c.a_f1[k].mean = m1 + offG;    c.a_f1[k].rstd = r1 + offG;

        c.a_ns[k].y = y1 + offN;       c.a_ns[k].gamma = g1p; c.a_ns[k].beta = b1p;
        c.a_ns[k].mean = m1 + offG;    c.a_ns[k].rstd = r1 + offG;

        c.a_c2[k].A = y1;              c.a_c2[k].WT = c.wt2.data_ptr<float>(); c.a_c2[k].Y = y2;
        c.a_p2[k].y = y2;              c.a_p2[k].psum = ps;  c.a_p2[k].psq = pq;
        c.a_f2[k].psum = ps + offP;    c.a_f2[k].psq = pq + offP;
        c.a_f2[k].mean = m2 + offG;    c.a_f2[k].rstd = r2 + offG;

        c.a_e[k].y = y2 + offN;        c.a_e[k].xres = xp + offC;
        c.a_e[k].gamma = g2p;          c.a_e[k].beta = b2p;
        c.a_e[k].mean = m2 + offG;     c.a_e[k].rstd = r2 + offG;
        c.a_e[k].out = outp + offC;
    }
}

// Plan item 8: identical node sequence launched eagerly (warmup / fallback).
static void launch_eager(Ctx& c, cudaStream_t stream) {
    for (size_t i = 0; i < c.knp.size(); ++i) {
        const cudaKernelNodeParams& np = c.knp[i];
        AT_CUDA_CHECK(cudaLaunchKernel(np.func, np.gridDim, np.blockDim,
                                       np.kernelParams, 0, stream));
    }
}

// Plan item 5: explicit DAG — two shared roots + S independent chains.
static bool build_graph(Ctx& c) {
    cudaGraph_t g = nullptr;
    if (cudaGraphCreate(&g, 0) != cudaSuccess) { cudaGetLastError(); return false; }

    c.nodes.assign(c.knp.size(), nullptr);
    auto addnode = [&](int idx, cudaGraphNode_t* deps, int nd) -> bool {
        cudaError_t e = cudaGraphAddKernelNode(&c.nodes[idx], g, deps, (size_t)nd, &c.knp[idx]);
        return e == cudaSuccess;
    };

    bool ok = addnode(0, nullptr, 0) && addnode(1, nullptr, 0);
    for (int k = 0; ok && k < c.S; ++k) {
        const int b = 2 + 9 * k;
        cudaGraphNode_t d2[2];
        ok = ok && addnode(b + 0, nullptr, 0);                       // t_k
        d2[0] = c.nodes[b + 0]; d2[1] = c.nodes[0];
        ok = ok && addnode(b + 1, d2, 2);                            // c1_k
        ok = ok && addnode(b + 2, &c.nodes[b + 1], 1);               // p1_k
        ok = ok && addnode(b + 3, &c.nodes[b + 2], 1);               // f1_k
        ok = ok && addnode(b + 4, &c.nodes[b + 3], 1);               // ns_k
        if (!ok) break;
        d2[0] = c.nodes[b + 4]; d2[1] = c.nodes[1];
        ok = ok && addnode(b + 5, d2, 2);                            // c2_k
        ok = ok && addnode(b + 6, &c.nodes[b + 5], 1);               // p2_k
        ok = ok && addnode(b + 7, &c.nodes[b + 6], 1);               // f2_k
        ok = ok && addnode(b + 8, &c.nodes[b + 7], 1);               // e_k
    }
    if (!ok) { cudaGetLastError(); cudaGraphDestroy(g); return false; }

    cudaGraphExec_t ex = nullptr;
    if (cudaGraphInstantiateWithFlags(&ex, g, 0) != cudaSuccess || ex == nullptr) {
        cudaGetLastError(); cudaGraphDestroy(g); return false;
    }
    c.graph = g; c.exec = ex;
    return true;
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
    const int cpg = C / 32;
    TORCH_CHECK(C % BN == 0 && C % BK == 0 && cpg % 4 == 0, "unsupported channel count");
    const long Ml = (long)B * H * W;
    TORCH_CHECK(Ml < (1L << 30), "problem too large");

    const float epsf = (float)eps;
    int epsbits; std::memcpy(&epsbits, &epsf, sizeof(int));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts   = xc.options();
    auto out    = torch::empty_like(xc);   // only allocation on the replay path

    std::lock_guard<std::mutex> lk(g_mu);

    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap_st);
    const bool outer_capture = (cap_st != cudaStreamCaptureStatusNone);

    Key key = std::make_tuple(B, C, H, W, epsbits);
    auto it = g_cache.find(key);
    bool fresh = false;
    Ctx* ctxp = nullptr;

    if (it != g_cache.end()) {
        ctxp = it->second.get();
    } else {
        if (g_cache.size() >= 16) {                 // plan item 8: bounded cache
            cudaDeviceSynchronize();
            for (auto& kv : g_cache) {
                if (kv.second->exec)  cudaGraphExecDestroy(kv.second->exec);
                if (kv.second->graph) cudaGraphDestroy(kv.second->graph);
            }
            g_cache.clear();
            cudaGetLastError();
        }
        auto up = std::unique_ptr<Ctx>(new Ctx());
        ctxp = up.get();
        g_cache.emplace(key, std::move(up));
        alloc_ctx(*ctxp, B, C, H, W, epsf, opts);   // plan items 1/3
        build_nodes(*ctxp);                         // plan items 4/6
        fresh = true;
    }
    Ctx& c = *ctxp;

    set_call_ptrs(c, xc.data_ptr<float>(), w1.data_ptr<float>(), w2.data_ptr<float>(),
                  g1.data_ptr<float>(), b1.data_ptr<float>(),
                  g2.data_ptr<float>(), b2.data_ptr<float>(), out.data_ptr<float>());

    bool used_graph = false;
    if (!fresh && c.exec != nullptr && !outer_capture) {
        bool ok = true;
        for (int idx : c.upd) {                     // plan item 7
            if (cudaGraphExecKernelNodeSetParams(c.exec, c.nodes[idx], &c.knp[idx]) != cudaSuccess) {
                cudaGetLastError(); ok = false; break;
            }
        }
        if (ok && cudaGraphLaunch(c.exec, stream) == cudaSuccess) used_graph = true;
        else cudaGetLastError();
    }

    if (!used_graph) {
        launch_eager(c, stream);                    // plan items 6/8
        if (fresh && !outer_capture) {
            cudaStreamSynchronize(stream);
            cudaGetLastError();
            if (!build_graph(c)) { c.graph = nullptr; c.exec = nullptr; }
        }
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
    name="vae_resblock_igemm_graph_chunked",
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
    The whole forward is expressed as ONE explicitly built cudaGraph containing
    S independent per-batch-chunk chains over module-owned persistent scratch,
    so the DRAM-bound transpose/GN/epilogue nodes of one chunk co-schedule with
    the compute-bound conv node of another. Per forward there is exactly one
    host graph launch and one allocation (the output tensor).

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
