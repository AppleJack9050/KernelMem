#!/usr/bin/env python3
"""Benchmark driver used by NCU/NSYS profiling (template for subproc 0).

Reconstructed template: this file was referenced by main_memory_latest.py (which
clones it to bench_ref_inputs_{subproc_id}.py by literally replacing "ref_101.py"
and "test_kernel_101.py") but was missing from the repository.

It loads the reference task from ref_101.py (get_inputs / get_init_inputs) and the
candidate Triton kernel module (--test, default test_kernel_101.py, must define
ModelNew), then runs ModelNew.forward `--repeat` times so ncu/nsys can profile
the kernel launches. NCU is configured with --launch-skip=2 --launch-count=6
(metrics pass), so each Triton kernel should be launched at least 8 times;
the first launches also absorb Triton JIT compilation.

Invoked as:
    <python> bench_ref_inputs_0.py --device-idx <int> --test <kernel.py> [--repeat <int>]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REF_PY = "ref_101.py"


def _load_module(path: str | Path, name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (cwd={Path.cwd()})")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ModelNew forward repeatedly for profiling")
    parser.add_argument("--device-idx", type=int, default=0, help="CUDA device index")
    parser.add_argument("--test", default="test_kernel_101.py", help="Candidate kernel .py defining ModelNew")
    parser.add_argument("--repeat", type=int, default=10, help="Number of profiled forward passes")
    parser.add_argument("--warmup", type=int, default=2, help="Un-profiled warm-up passes (JIT/autotune)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for profiling")
    device = torch.device(f"cuda:{args.device_idx}")
    torch.cuda.set_device(device)
    # --- load reference task + candidate kernel ---
    ref_mod = _load_module(REF_PY, "bench_ref_module")
    test_mod = _load_module(args.test, "bench_test_module")

    ModelNew = getattr(test_mod, "ModelNew", None)
    get_inputs = getattr(ref_mod, "get_inputs", None)
    if ModelNew is None:
        raise RuntimeError(f"{args.test} must define ModelNew")
    if get_inputs is None:
        raise RuntimeError(f"{REF_PY} must define get_inputs()")

    init_args, init_kwargs = [], {}
    get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
    if callable(get_init_inputs):
        init_obj = get_init_inputs()
        if isinstance(init_obj, dict):
            init_kwargs = dict(init_obj)
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)
    # --- inputs on the profiling device ---
    torch.manual_seed(100)
    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    inputs = [x.to(device) if hasattr(x, "to") else x for x in inputs]

    torch.manual_seed(100)
    model = ModelNew(*init_args, **init_kwargs).to(device).eval()
    # --- profiled window ---
    with torch.no_grad():
        for _ in range(max(0, args.warmup)):
            model(*inputs)
        torch.cuda.synchronize(device)
        for _ in range(max(1, args.repeat)):
            model(*inputs)
        torch.cuda.synchronize(device)

    print(f"[bench] Completed {args.repeat} profiled forward passes on {device}", flush=True)


if __name__ == "__main__":
    main()
