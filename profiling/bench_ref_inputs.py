#!/usr/bin/env python3
"""Benchmark driver used by NCU/NSYS profiling.

This is a template. main_memory_latest.py copies it, unchanged, into the run's
scratch directory (run/<batch>/<task>/scratch/bench_ref_inputs.py) beside
ref.py (a copy of the task file) and test_kernel.py (the kernel being profiled),
and hands ncu/nsys that copy by absolute path. By default it loads both modules
from its own directory, so the copy works from any working directory; --ref and
--test override that for standalone use.

It loads the reference task (get_inputs / get_init_inputs) and the candidate
kernel module (must define ModelNew), then runs ModelNew.forward `--repeat`
times so ncu/nsys can profile the kernel launches. NCU is configured with
--launch-skip=2 --launch-count=6 (metrics pass), so each kernel should be
launched at least 8 times; the first launches also absorb JIT compilation.

Invoked as:
    <python> bench_ref_inputs.py --device-idx <int> --test <kernel.py> [--ref <task.py>] [--repeat <int>]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

# Siblings in the scratch directory this template is copied into.
_HERE = Path(__file__).resolve().parent
DEFAULT_REF_PY = _HERE / "ref.py"
DEFAULT_TEST_PY = _HERE / "test_kernel.py"


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


def _load_kernel_module(path: str | Path, name: str):
    """Import the candidate kernel under the harness's build policy.

    ncu and nsys must import the very binary the preload worker warmed and the
    bench measured, or the kernel compiles UNDER THE PROFILER: in
    20260911_214651 the plain import here rebuilt the bench's ``-Xptxas -v``
    cuda.o (~31 s first ncu invocation, rounds 3-5) and same-named kernels
    rebuilt each other (round 6 nsys 29.8 s). ``kernel_build_context`` gives the
    same flags and the same content-hashed build folder as every other import
    site. ncu.py/nsys.py put the repo root on PYTHONPATH; standalone use without
    it still works, but says loudly that it is not sharing the harness's build.
    """
    try:
        from utils.ext_naming import kernel_build_context
    except ImportError as exc:
        print("\n" + "!" * 78 + "\n"
              f"[bench] WARNING: utils.ext_naming is not importable ({exc}).\n"
              "[bench] Loading the kernel PLAINLY: its own extension name and flags, NOT\n"
              "[bench] the harness build policy. It will not reuse the bench/preload build\n"
              "[bench] and may compile under the profiler. Put the KernelMem repo root on\n"
              "[bench] PYTHONPATH (profiling/ncu.py and nsys.py do).\n" + "!" * 78 + "\n",
              file=sys.stderr, flush=True)
        return _load_module(path, name)
    with kernel_build_context(force_verbose=False):
        return _load_module(path, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ModelNew forward repeatedly for profiling")
    parser.add_argument("--device-idx", type=int, default=0, help="CUDA device index")
    parser.add_argument("--ref", default=str(DEFAULT_REF_PY),
                        help="Reference task .py defining get_inputs() (default: ref.py beside this script)")
    parser.add_argument("--test", default=str(DEFAULT_TEST_PY),
                        help="Candidate kernel .py defining ModelNew (default: test_kernel.py beside this script)")
    parser.add_argument("--repeat", type=int, default=10, help="Number of profiled forward passes")
    parser.add_argument("--warmup", type=int, default=2, help="Un-profiled warm-up passes (JIT/autotune)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for profiling")
    device = torch.device(f"cuda:{args.device_idx}")
    torch.cuda.set_device(device)
    # --- load reference task + candidate kernel ---
    ref_mod = _load_module(args.ref, "bench_ref_module")
    test_mod = _load_kernel_module(args.test, "bench_test_module")

    ModelNew = getattr(test_mod, "ModelNew", None)
    get_inputs = getattr(ref_mod, "get_inputs", None)
    if ModelNew is None:
        raise RuntimeError(f"{args.test} must define ModelNew")
    if get_inputs is None:
        raise RuntimeError(f"{args.ref} must define get_inputs()")

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
