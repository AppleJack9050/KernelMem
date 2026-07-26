import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row

    cols = tl.arange(0, BLOCK_SIZE)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_SIZE), BLOCK_SIZE)

    x = tl.load(x_row + cols).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sumsq / n_cols + eps)

    w = tl.load(w_ptr + cols).to(tl.float32)
    y = x * inv_rms * w
    tl.store(out_row + cols, y.to(out_ptr.dtype.element_ty))


def rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()

    batch_size, hidden_size = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    grid = (batch_size,)

    rmsnorm_kernel[grid](
        x, w, out,
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        n_cols=hidden_size,
        eps=1e-5,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, weight):
        return rmsnorm(hidden_states, weight)
