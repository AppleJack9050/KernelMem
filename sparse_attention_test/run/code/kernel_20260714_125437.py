import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sliding_window_attn_kernel(
    Q, K, V, Out,
    N, WINDOW, scale,
    stride_bh, stride_n, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh * stride_bh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    q_mask = offs_m[:, None] < N
    q = tl.load(Q + q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    start_m = pid_m * BLOCK_M
    # key positions relevant: j > i-WINDOW and j <= i, over the query block
    lo = start_m - WINDOW + 1
    lo = tl.maximum(lo, 0)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start_m + BLOCK_M, N)

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        n_mask = offs_n[:, None] < N
        k = tl.load(K + k_ptrs, mask=n_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_M, BLOCK_N)

        rel = offs_m[:, None] - offs_n[None, :]
        valid = (rel >= 0) & (rel < WINDOW) & (offs_n[None, :] < N)
        qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # Guard against inf-minus-inf: when m_new is -inf (no valid keys seen yet),
        # alpha must be 0 (empty accumulator) and p must be 0 for invalid positions.
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(valid, tl.exp(qk - m_new[:, None]), 0.0)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V + k_ptrs, mask=n_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]

    o_ptrs = base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    tl.store(Out + o_ptrs, out.to(Out.dtype.element_ty), mask=q_mask)


def sliding_window_attention(q, k, v, window, scale):
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()

    B, H, N, D = q.shape
    BH = B * H
    out = torch.empty_like(q)

    qf = q.view(BH, N, D)
    kf = k.view(BH, N, D)
    vf = v.view(BH, N, D)
    of = out.view(BH, N, D)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(N, BLOCK_M), BH)
    sliding_window_attn_kernel[grid](
        qf, kf, vf, of,
        N, window, scale,
        qf.stride(0), qf.stride(1), qf.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return sliding_window_attention(q, k, v, self.window_size, self.scale)


# ---- benchmark configuration ----
batch_size = 4
num_heads = 8
seq_len = 1024
head_dim = 64
window_size = 256


def get_inputs():
    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = torch.rand(batch_size, num_heads, seq_len, head_dim)
    v = torch.rand(batch_size, num_heads, seq_len, head_dim)
    return [q, k, v]


def get_init_inputs():
    return [num_heads, head_dim, window_size]
