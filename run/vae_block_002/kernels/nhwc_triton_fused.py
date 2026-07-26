import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gn_partial(
    y_ptr, sum_ptr, sqsum_ptr,
    HW, C: tl.constexpr, CPG: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_bg = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    G: tl.constexpr = C // CPG
    b = pid_bg // G
    g = pid_bg % G
    offs_p = pid_chunk * BLOCK + tl.arange(0, BLOCK)
    mask_p = offs_p < HW
    offs_c = tl.arange(0, CPG)
    ptrs = y_ptr + b * HW * C + offs_p[:, None] * C + g * CPG + offs_c[None, :]
    v = tl.load(ptrs, mask=mask_p[:, None], other=0.0)
    tl.atomic_add(sum_ptr + pid_bg, tl.sum(v))
    tl.atomic_add(sqsum_ptr + pid_bg, tl.sum(v * v))


@triton.jit
def _gn_silu_apply(
    y_ptr, out_ptr, res_ptr, sum_ptr, sqsum_ptr, w_ptr, b_ptr,
    HW, eps,
    C: tl.constexpr, CPG: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_bg = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    G: tl.constexpr = C // CPG
    b = pid_bg // G
    g = pid_bg % G
    n = HW * CPG
    s = tl.load(sum_ptr + pid_bg)
    sq = tl.load(sqsum_ptr + pid_bg)
    mean = s / n
    var = sq / n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    offs_p = pid_chunk * BLOCK + tl.arange(0, BLOCK)
    mask_p = offs_p < HW
    offs_c = tl.arange(0, CPG)
    gamma = tl.load(w_ptr + g * CPG + offs_c)
    beta = tl.load(b_ptr + g * CPG + offs_c)
    idx = b * HW * C + offs_p[:, None] * C + g * CPG + offs_c[None, :]
    v = tl.load(y_ptr + idx, mask=mask_p[:, None], other=0.0)
    vn = (v - mean) * rstd * gamma[None, :] + beta[None, :]
    out = vn * tl.sigmoid(vn)
    if HAS_RES:
        out = out + tl.load(res_ptr + idx, mask=mask_p[:, None], other=0.0)
    tl.store(out_ptr + idx, out, mask=mask_p[:, None])


def _fused_gn_silu(y, gamma, beta, eps, residual=None):
    B, C, H, W = y.shape
    G = 32
    CPG = C // G
    HW = H * W
    BLOCK = 128
    y = y.contiguous(memory_format=torch.channels_last)
    out = torch.empty_like(y)
    sums = torch.zeros(B * G, device=y.device, dtype=torch.float32)
    sqsums = torch.zeros(B * G, device=y.device, dtype=torch.float32)
    grid = (B * G, triton.cdiv(HW, BLOCK))
    _gn_partial[grid](y, sums, sqsums, HW, C=C, CPG=CPG, BLOCK=BLOCK)
    _gn_silu_apply[grid](
        y, out, residual if residual is not None else y,
        sums, sqsums, gamma, beta, HW, eps,
        C=C, CPG=CPG, HAS_RES=residual is not None, BLOCK=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    """channels_last cuDNN convs + Triton-fused GroupNorm+SiLU(+residual)."""

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        xl = x.contiguous(memory_format=torch.channels_last)
        w1 = conv1_weight.contiguous(memory_format=torch.channels_last)
        w2 = conv2_weight.contiguous(memory_format=torch.channels_last)
        y = F.conv2d(xl, w1, None, 1, 1)
        y = _fused_gn_silu(y, norm1_weight, norm1_bias, eps)
        y = F.conv2d(y, w2, None, 1, 1)
        return _fused_gn_silu(y, norm2_weight, norm2_bias, eps, residual=xl)
