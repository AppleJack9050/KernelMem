import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

# 20 distinct workload shapes > default dynamo cache limit of 8; raise it so
# every shape gets its own max-autotune'd graph instead of falling back to eager.
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
    """torch.compile mode='max-autotune' over the channels_last block.

    max-autotune benchmarks Triton/cuDNN conv templates per shape and turns on
    CUDA graphs; the raised dynamo cache limit lets all 20 shapes compile.
    """

    def __init__(self):
        super().__init__()
        self._fn = torch.compile(_block, mode="max-autotune", dynamic=False)

    def forward(self, *inputs):
        return self._fn(*inputs)
