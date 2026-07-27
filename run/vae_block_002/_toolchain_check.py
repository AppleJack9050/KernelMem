"""Minimal CUDA extension build check for this host.

Verifies that torch.utils.cpp_extension can build and run a trivial kernel for
the local GPU. Used to validate toolchain fixes before spending agent rounds.
"""
import os

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
__global__ void addk(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] + 1.f;
}
torch::Tensor addone(torch::Tensor a) {
    auto b = torch::empty_like(a);
    int n = a.numel();
    addk<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), n);
    return b;
}
'''
cpp_src = "torch::Tensor addone(torch::Tensor a);"

m = load_inline(
    name=os.environ.get("EXTNAME", "toolchain_check"),
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["addone"],
    extra_cuda_cflags=["-O3", "-std=c++20"],
    verbose=False,
)
a = torch.ones(1024, device="cuda")
print("BUILD OK, result mean =", m.addone(a).mean().item())
