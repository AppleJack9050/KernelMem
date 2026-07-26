import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_reduce_kernel(
    x_ptr, sumsq_ptr,
    stride_row,
    CHUNK: tl.constexpr,
    n_cols: tl.constexpr,
    EVEN: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    x_row = x_ptr + row * stride_row
    cols = split * CHUNK + tl.arange(0, CHUNK)
    if EVEN:
        x = tl.load(x_row + cols).to(tl.float32)
    else:
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(x * x, axis=0)
    tl.atomic_add(sumsq_ptr + row, partial)


@triton.jit
def rms_apply_kernel(
    x_ptr, w_ptr, out_ptr, sumsq_ptr,
    stride_row,
    CHUNK: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    EVEN: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row
    cols = split * CHUNK + tl.arange(0, CHUNK)

    s = tl.load(sumsq_ptr + row)
    inv_rms = tl.rsqrt(s / n_cols + eps)

    if EVEN:
        x = tl.load(x_row + cols).to(tl.float32)
        w = tl.load(w_ptr + cols).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_row + cols, y.to(out_ptr.dtype.element_ty))
    else:
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()

    batch_size, hidden_size = x.shape
    out = torch.empty_like(x)
    sumsq = torch.zeros(batch_size, dtype=torch.float32, device=x.device)

    # Choose SPLITS as a divisor of hidden_size for mask-free contiguous chunks.
    SPLITS = 8
    while SPLITS > 1 and (hidden_size % SPLITS != 0):
        SPLITS //= 2
    CHUNK = triton.next_power_of_2(triton.cdiv(hidden_size, SPLITS))
    EVEN = (hidden_size % SPLITS == 0) and (CHUNK == hidden_size // SPLITS)

    grid = (batch_size, SPLITS)

    rms_reduce_kernel[grid](
        x, sumsq,
        x.stride(0),
        CHUNK=CHUNK,
        n_cols=hidden_size,
        EVEN=EVEN,
        num_warps=4,
    )
    rms_apply_kernel[grid](
        x, w, out, sumsq,
        x.stride(0),
        CHUNK=CHUNK,
        n_cols=hidden_size,
        eps=1e-5,
        EVEN=EVEN,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, weight):
        return rmsnorm(hidden_states, weight)
