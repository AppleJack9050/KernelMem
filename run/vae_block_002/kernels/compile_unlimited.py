import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 64


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


class ModelNew(nn.Module):
    """torch.compile with raised dynamo cache limit (20 distinct shapes > default 8)."""

    def __init__(self):
        super().__init__()
        self._fn = torch.compile(_block, dynamic=False)

    def forward(self, *inputs):
        return self._fn(*inputs)
