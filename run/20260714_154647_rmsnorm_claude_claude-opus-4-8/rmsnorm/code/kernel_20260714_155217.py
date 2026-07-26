import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_partial_kernel(
    x_ptr, sumsq_ptr,
    stride_row,
    COL_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    col_base = split * COL_TILE
    cols = col_base + tl.arange(0, COL_TILE)
    cols = tl.max_contiguous(tl.multiple_of(cols, COL_TILE), COL_TILE)

    x = tl.load(x_ptr + row * stride_row + cols).to(tl.float32)
    partial = tl.sum(x * x, axis=0)
    tl.atomic_add(sumsq_ptr + row, partial)


@triton.jit
def rmsnorm_apply_kernel(
    x_ptr, w_ptr, sumsq_ptr, out_ptr,
    stride_row,
    COL_TILE: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    col_base = split * COL_TILE
    cols = col_base + tl.arange(0, COL_TILE)
    cols = tl.max_contiguous(tl.multiple_of(cols, COL_TILE), COL_TILE)

    sumsq = tl.load(sumsq_ptr + row)
    inv_rms = tl.rsqrt(sumsq / n_cols + eps)

    x = tl.load(x_ptr + row * stride_row + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = x * inv_rms * w
    tl.store(out_ptr + row * stride_row + cols, y.to(out_ptr.dtype.element_ty))


def rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()

    batch_size, hidden_size = x.shape
    out = torch.empty_like(x)

    # choose a split count that keeps blocks per SM high and evenly divides hidden_size
    NUM_SPLITS = 4
    if hidden_size % NUM_SPLITS != 0:
        # fall back to a divisor-safe split
        NUM_SPLITS = 1
        for cand in (8, 4, 2):
            if hidden_size % cand == 0:
                NUM_SPLITS = cand
                break
    COL_TILE = hidden_size // NUM_SPLITS

    sumsq = torch.zeros(batch_size, dtype=torch.float32, device=x.device)

    grid = (batch_size, NUM_SPLITS)

    rmsnorm_partial_kernel[grid](
        x, sumsq,
        x.stride(0),
        COL_TILE=COL_TILE,
        num_warps=4,
        num_stages=2,
    )

    rmsnorm_apply_kernel[grid](
        x, w, sumsq, out,
        x.stride(0),
        COL_TILE=COL_TILE,
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
