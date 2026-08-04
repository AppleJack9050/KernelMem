"""
A List of GPU Specs to include in the prompt

"""


GPU_SPEC_INFO = {
    # Measured on this host: 170 SMs, 2.42 GHz boost, sm_120.
    # Tensor-core figures are dense; NVIDIA's marketing numbers for this part are
    # quoted with 2:4 sparsity (2x) and, for FP16/FP8, with FP16 accumulate (2x).
    "RTX 5090": {
        "GPU Architecture": "Blackwell (sm_120)",
        "GPU Memory": "32GB GDDR7",
        "Memory Bandwidth": "1792 GB/s",
        "Streaming Multiprocessors": "170",
        "FP32 TFLOPS": "104.8",
        "TF32 Tensor Core TFLOPS": "104.8 (209.5 with sparsity)",
        "FP16 Tensor Core TFLOPS": "209.5 with FP32 accumulate (419 with FP16 accumulate)",
        "BFLOAT16 Tensor Core TFLOPS": "209.5 with FP32 accumulate",
        "FP8 Tensor Core TFLOPS": "419 (838 with sparsity)",
        "FP4 Tensor Core TFLOPS": "838 (1676 with sparsity)",
        "Peak INT8 Tensor TOPS": "419 (838 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },

    "L40S": {
        "GPU Architecture": "Ada",
        "GPU Memory": "48GB GDDR6 with ECC",
        "Memory Bandwidth": "864 GB/s",
        "RT Core Performance TFLOPS": "212",
        "FP32 TFLOPS": "91.6",
        "TF32 Tensor Core TFLOPS": "183.2 (366 with sparsity)",
        "FP16 Tensor Core TFLOPS": "362.05 (733 with sparsity)",
        "FP8 Tensor Core TFLOPS": "733 (1466 with sparsity)",
        "Peak INT8 Tensor TOPS": "733 (1466 with sparsity)",
        "Peak INT4 Tensor TOPS": "733 (1466 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    
    # "H100" describes the card actually installed in this machine: H100 PCIe
    # 80GB (114 SMs). The previous entry here held H100 *SXM* figures quoted
    # *with sparsity* -- 3.35 TB/s and 989 TF32 TFLOPS -- neither of which any
    # dense kernel can reach on any H100. The error is not cosmetic: it is the
    # denominator of every "% of peak" the seed and judge reason with. A vendor
    # GEMM/conv measured here at ~244 TFLOP/s reads as 25% of roofline against
    # 989 -- 4x headroom apparently left on the table -- and as 64% against the
    # real 378, which is about as good as a convolution gets. The first reading
    # makes "rewrite the vendor kernel" look attractive when it is not.
    # Sparsity is listed separately because it requires 2:4 structured-sparse
    # weights, which none of these kernels have.
    # Keep per-problem measurements OUT of this table: it is emitted into every
    # prompt for every task, so a number from one workload becomes a false
    # statement about the next.
    # See "H100-SXM" below for the datasheet SXM part.
    "H100": {
        "GPU Architecture": "Hopper",
        "Board": "H100 PCIe 80GB (NOT SXM -- lower clocks, HBM2e, 114 SMs)",
        "GPU Memory": "80GB HBM2e",
        "Memory Bandwidth": "2.0 TB/s peak, 1.84 TB/s measured (1 GiB d2d copy, 92% of peak)",
        "Streaming Multiprocessors": "114",
        "FP64 TFLOPS": "26",
        "FP64 Tensor Core TFLOPS": "51",
        "FP32 TFLOPS": "51 peak, 36.5 measured (SGEMM)",
        "TF32 Tensor Core TFLOPS": "378 dense (756 with 2:4 sparsity); 223 measured on a 8192^3 GEMM",
        "BFLOAT16 Tensore Core TFLOPS": "756 dense (1513 with 2:4 sparsity)",
        "FP16 Tensor Core TFLOPS": "756 dense (1513 with 2:4 sparsity)",
        "FP8 Tensor Core TFLOPS": "1513 dense (3026 with 2:4 sparsity)",
        "INT8 Tensor Core TOPS": "1513 dense (3026 with 2:4 sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },

    # The datasheet SXM part, for runs on a different machine. Dense figures;
    # the sparsity numbers are double these and apply only to 2:4 sparse weights.
    "H100-SXM": {
        "GPU Architecture": "Hopper",
        "Board": "H100 SXM5 80GB (132 SMs)",
        "GPU Memory": "80GB HBM3",
        "Memory Bandwidth": "3.35 TB/s",
        "Streaming Multiprocessors": "132",
        "FP64 TFLOPS": "34",
        "FP64 Tensor Core TFLOPS": "67",
        "FP32 TFLOPS": "67",
        "TF32 Tensor Core TFLOPS": "495 dense (989 with 2:4 sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "989 dense (1979 with 2:4 sparsity)",
        "FP16 Tensor Core TFLOPS": "989 dense (1979 with 2:4 sparsity)",
        "FP8 Tensor Core TFLOPS": "1979 dense (3958 with 2:4 sparsity)",
        "INT8 Tensor Core TOPS": "1979 dense (3958 with 2:4 sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },
    # this is 40GB (Standard)
    "A100": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "40GB",
        "Memory Bandwidth": "1555 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A100-80GB": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "L4": {
        "GPU Architecture": "Ada",
        "GPU Memory": "24GB",
        "Memory Bandwidth": "300 GB/s",
        "FP32 TFLOPS": "30.3",
        "TF32 Tensor Core TFLOPS": "120 with sparsity",
        "BFLOAT16 Tensore Core TFLOPS": "242 with sparsity",
        "FP8 Tensor Core TFLOPS": "485 with sparsity",
        "INT8 Tensor Core TOPS": "485 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    }, 
    "Quadro RTX 6000": {
        "GPU Architecture": "Turing",
        "GPU Memory": "24GB GDDR6 with ECC",
        "Memory Bandwidth": "624 GB/s",
        "FP32 TFLOPS": "16.3",
        "TF32 Tensor Core TFLOPS": "— (Turing does not support TF32)",
        "FP16 Tensor Core TFLOPS": "261 (522 with sparsity)*",
        "FP8 Tensor Core TFLOPS": "— (Not supported for FP8)",
        "INT8 Tensor Core TOPS": "261 (522 with sparsity)*",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "64 KB",
        "Maximum shared memory per thread block": "64 KB"
    },
    "T4": {
        "GPU Architecture": "Turing",
        "GPU Memory": "16 GB GDDR6",
        "Memory Bandwidth": "300 GB/s",
        "Single-Precision TFLOPS": "8.1",
        "Mixed-Precision (FP16/FP32) TFLOPS": "65",
        "INT8 TOPS": "130",
        "INT4 TOPS": "260",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "16",
        "Shared memory capacity per SM": "64 KB",
    },
    "A10G": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "24GB GDDR6",
        "Memory Bandwidth": "600 GB/s",
        "FP32 TFLOPS": "31.2",
        "TF32 Tensor Core TFLOPS": "62.5 (125 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "125 (250 with sparsity)",
        "FP16 Tensor Core TFLOPS": "125 (250 with sparsity)",
        "INT8 Tensor Core TOPS": "250 (500 with sparsity)",
        "INT4 Tensor Core TOPS": "500 (1000 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    }
}

# Basic GPU concept definitions
GPU_DEFINITIONS = {
    "Thread": "A thread is a single execution unit that can run a single instruction at a time.",
    "Thread Block": "A thread block is a group of threads that can cooperate with each other.",
    "Warp": "A warp is a group of threads that are scheduled together and execute in parallel.",
    "Shared Memory": "Shared memory is a memory space that can be accessed by all threads in a thread block.",
    "Register": "A register is a small memory space that can be accessed by a single thread.",
    "Memory Hierarchy": "Memory hierarchy is a pyramid of memory types with different speeds and sizes.",
    "Memory Bandwidth": "Memory bandwidth is the rate at which data can be read from or stored into memory.",
    "Cache": "Cache is a small memory space that stores frequently accessed data.",
    "HBM": "HBM is a high-bandwidth memory technology that uses 3D-stacked DRAM.",
}



GPU_BEST_PRACTICES = [
    # From https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
    # CUDA Best Practices Section
    "Find ways to parallelize sequential code.",
    "Minimize data transfers between the host and the device.",
    "Adjust kernel launch configuration to maximize device utilization.",
    "Ensure that global memory accesses are coalesced.",
    "Minimize redundant accesses to global memory whenever possible.",
    "Avoid long sequences of diverged execution by threads within the same warp.",
    # we added this to reference the specific GPU architecture
    "Use specialized instructions based on the specific GPU architecture",
]