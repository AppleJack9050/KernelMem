import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_partial_kernel(
    x_ptr, partial_ptr,
    stride_row,
    K: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    cols = tile * BLOCK_COL + tl.arange(0, BLOCK_COL)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_COL), BLOCK_COL)

    x = tl.load(x_ptr + row * stride_row + cols).to(tl.float32)
    psum = tl.sum(x * x, axis=0)
    tl.store(partial_ptr + row * K + tile, psum)


@triton.jit
def rmsnorm_apply_kernel(
    x_ptr, w_ptr, partial_ptr, out_ptr,
    stride_row,
    K: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    # read the K partial sum-of-squares for this row (tiny, from L2)
    kidx = tl.arange(0, K)
    partials = tl.load(partial_ptr + row * K + kidx).to(tl.float32)
    sumsq = tl.sum(partials, axis=0)
    inv_rms = tl.rsqrt(sumsq / n_cols + eps)

    cols = tile * BLOCK_COL + tl.arange(0, BLOCK_COL)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_COL), BLOCK_COL)

    x = tl.load(x_ptr + row * stride_row + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = x * inv_rms * w
    tl.store(out_ptr + row * stride_row + cols, y.to(out_ptr.dtype.element_ty))


def rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()

    batch_size, hidden_size = x.shape
    out = torch.empty_like(x)

    # Choose K so batch_size*K fills the SMs (>2 waves).
    K = 8
    while hidden_size % K != 0:
        K //= 2
    if K < 1:
        K = 1
    BLOCK_COL = hidden_size // K

    partial = torch.empty((batch_size, K), device=x.device, dtype=torch.float32)

    grid = (batch_size, K)

    rmsnorm_partial_kernel[grid](
        x, partial,
        x.stride(0),
        K=K,
        BLOCK_COL=BLOCK_COL,
        num_warps=4,
        num_stages=2,
    )

    rmsnorm_apply_kernel[grid](
        x, w, partial, out,
        x.stride(0),
        K=K,
        BLOCK_COL=BLOCK_COL,
        n_cols=hidden_size,
        eps=1e-5,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, weight):
        return rmsnorm(hidden_states, weight)
