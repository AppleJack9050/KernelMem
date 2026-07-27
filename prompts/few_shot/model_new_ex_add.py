import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    auto stream = at::cuda::getDefaultCUDAStream();
    elementwise_add_kernel<<<num_blocks, block_size, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), (int)size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_src = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[
        "-O3",                          # High optimization
        "-std=c++20",                    # Use C++20 standard
        # "-U__CUDA_NO_HALF_OPERATORS__",  # Enable half precision operations
        # "-U__CUDA_NO_HALF_CONVERSIONS__", # Enable conversions for half precision
        # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", # Enable bfloat16 operations
        # "-U__CUDA_NO_HALF2_OPERATORS__", # Enable half2 precision
        ],
    extra_ldflags=[""],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",  # nvcc 13.x miscompiles the ATen headers in C++17 mode
        "-gencode=arch=compute_80,code=sm_80",  # Correct CUDA target arch
        "--expt-relaxed-constexpr",  # CUDA specific flag
        "-lineinfo",  # Line information for debugging
        ],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
