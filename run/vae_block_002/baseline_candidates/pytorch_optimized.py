"""Optimized PyTorch implementation of problem 002 — scoring-baseline candidate.

This is what upstream's sol_score() docstring means by T_b ("an optimized PyTorch
implementation of the reference solution"): same math as reference.py, but using
channels_last and torch.compile, which is what a competent PyTorch user would write.
Exposes run(...) with the reference signature for `solbench_bridge make-baselines`.
"""

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 64

_compiled = None


def _block(x, conv1_weight, norm1_weight, norm1_bias,
           conv2_weight, norm2_weight, norm2_bias, eps):
    xl = x.contiguous(memory_format=torch.channels_last)
    w1 = conv1_weight.contiguous(memory_format=torch.channels_last)
    w2 = conv2_weight.contiguous(memory_format=torch.channels_last)
    out = F.conv2d(xl, w1, None, 1, 1)
    out = F.group_norm(out, 32, norm1_weight, norm1_bias, eps)
    out = F.silu(out)
    out = F.conv2d(out, w2, None, 1, 1)
    out = F.group_norm(out, 32, norm2_weight, norm2_bias, eps)
    out = F.silu(out)
    return out + xl


@torch.no_grad()
def run(x, conv1_weight, norm1_weight, norm1_bias,
        conv2_weight, norm2_weight, norm2_bias, eps):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_block, dynamic=False)
    return _compiled(x, conv1_weight, norm1_weight, norm1_bias,
                     conv2_weight, norm2_weight, norm2_bias, eps)
