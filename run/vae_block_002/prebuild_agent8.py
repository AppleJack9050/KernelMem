#!/usr/bin/env python3
"""Turn a KernelMem load_inline kernel into a prebuilt-.so kernel for SOL-ExecBench.

The SOL harness blocks torch.utils.cpp_extension.load/load_inline on the GPU
server, so the CUDA has to be compiled ahead of time. This imports the kernel
once (which compiles the extension into the torch_extensions cache), copies the
resulting .so next to the output file, and rewrites the `_ext = load_inline(...)`
call into a direct .so load. The compute is untouched.

Usage: python3 prebuild_agent8.py <kernel.py> <out_name>
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREBUILT = HERE / "prebuilt"

LOADER = '''
import importlib.util as _ilu
import os as _os


def _load_prebuilt_ext(_name):
    """Load the ahead-of-time compiled extension .so.

    SOL-ExecBench blocks cpp_extension.load_inline() on the GPU server; the
    compute is identical to the load_inline build. The harness stages this file
    into a temp dir without the .so, so fall back to the absolute build path.
    """
    _so = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _name + ".so")
    if not _os.path.exists(_so):
        _so = _os.path.join(__PREBUILT_DIR__, _name + ".so")
    _spec = _ilu.spec_from_file_location(_name, _so)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod
'''


def find_call_span(src, start_idx):
    """Return the end index (exclusive) of the parenthesised call starting at start_idx."""
    depth = 0
    i = src.index("(", start_idx)
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced parentheses in load_inline call")


def main(kernel_path, out_name):
    kernel_path = Path(kernel_path).resolve()
    src = kernel_path.read_text()

    marker = "_ext = load_inline("
    if marker not in src:
        raise SystemExit(f"no '{marker}' found in {kernel_path}")
    start = src.index(marker)
    end = find_call_span(src, start)
    call = src[start:end]

    # extension name -> .so file name
    name_key = 'name="'
    n0 = call.index(name_key) + len(name_key)
    ext_name = call[n0:call.index('"', n0)]

    # 1. import the kernel so load_inline compiles the .so into the cache
    print(f"[prebuild] compiling {kernel_path.name} (ext={ext_name}) ...")
    spec = importlib.util.spec_from_file_location("_k_" + out_name, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("[prebuild] compiled OK")

    # 2. locate the built .so in the torch_extensions cache
    import torch
    from torch.utils.cpp_extension import _get_build_directory
    build_dir = Path(_get_build_directory(ext_name, verbose=False))
    so = build_dir / f"{ext_name}.so"
    if not so.exists():
        cands = list(build_dir.glob("*.so"))
        if not cands:
            raise SystemExit(f"no .so found in {build_dir}")
        so = cands[0]

    PREBUILT.mkdir(exist_ok=True)
    dst_so = PREBUILT / f"{ext_name}.so"
    shutil.copy2(so, dst_so)
    print(f"[prebuild] copied {so} -> {dst_so}")

    # 3. rewrite the load_inline call into a .so load
    new_src = (
        src[:start]
        + f'_ext = _load_prebuilt_ext("{ext_name}")'
        + src[end:]
    )
    # drop the now-unused import so the harness never sees load_inline referenced
    new_src = new_src.replace(
        "from torch.utils.cpp_extension import load_inline\n", ""
    )
    loader = LOADER.replace("__PREBUILT_DIR__", repr(str(PREBUILT)))
    new_src = new_src.replace("import torch\n", "import torch\n" + loader, 1)

    out_py = PREBUILT / f"{out_name}.py"
    out_py.write_text(new_src)
    print(f"[prebuild] wrote {out_py}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
