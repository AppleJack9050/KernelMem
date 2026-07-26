# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path

import torch.utils.cpp_extension as ext

from sol_execbench.core import Solution

HERE = Path.cwd().resolve()

# Parse solution — validates sources (e.g. forbidden keywords) at compile time.
solution = Solution(**json.loads((HERE / "solution.json").read_text()))
compile_options = solution.spec.compile_options

# set flags
cuda_cflags = compile_options.cuda_cflags if compile_options else []
cflags = compile_options.cflags if compile_options else []
ld_flags = compile_options.ld_flags if compile_options else []

# Collect C/C++/CUDA source files from current directory
sources = [
    str(p)
    for p in HERE.iterdir()
    if p.suffix in (".cu", ".cpp", ".cc", ".cxx", ".c") and p.is_file()
]
if not sources:
    raise RuntimeError("No CUDA/C++ source files found in working directory")

cutlass_dir = os.environ.get("CUTLASS_DIR", "/usr/local/cutlass")
extra_include_paths = [
    str(HERE),
    f"{cutlass_dir}/include",
    f"{cutlass_dir}/tools/util/include",
]

ext.load(
    name="benchmark_kernel",
    sources=sources,
    extra_cuda_cflags=cuda_cflags,
    extra_cflags=cflags,
    extra_ldflags=ld_flags,
    extra_include_paths=extra_include_paths,
    build_directory=str(HERE),
    verbose=True,
)

# Rename platform-suffixed .so → benchmark_kernel.so
so_files = [
    f for f in HERE.glob("benchmark_kernel*.so") if f.name != "benchmark_kernel.so"
]
if so_files:
    so_files[0].rename("benchmark_kernel.so")
elif not (HERE / "benchmark_kernel.so").exists():
    raise FileNotFoundError("benchmark_kernel.so not produced by compilation")

print("benchmark_kernel.so ready", flush=True)
