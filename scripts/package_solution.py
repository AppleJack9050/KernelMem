#!/usr/bin/env python
"""Package a KernelMem kernel as a SOL-ExecBench solution.json.

Replaces solbench_bridge.kernel_to_solution, which exists in this checkout only
as Python 3.13 bytecode (magic 3571) and therefore cannot be imported by the
3.11 env that owns torch. The schema and the appended entry-point shim are
copied verbatim from a solution.json the bridge itself produced
(run/vae_block_002/out8_maxautotune/solution.json), so packaging stays
byte-compatible with previously scored solutions.

    python package_solution.py <kernel.py> <out_dir> [--name-suffix TAG]
"""
import argparse
import json
from pathlib import Path

DEFINITION = "002_vae_conv3x3_groupnorm_silu_residual_fused"

# Verbatim from the bridge's own output.
SHIM = '''

# ===== appended by solbench_bridge.kernel_to_solution (SOL entry point) =====
import torch as _sb_torch  # noqa: E402

_SB_MODEL = None
_SB_INIT_ARGS = []


def run(*args):
    """SOL-ExecBench entry point (value-returning; destination_passing_style=false)."""
    global _SB_MODEL
    if _SB_MODEL is None:
        _m = ModelNew(*_SB_INIT_ARGS)
        if isinstance(_m, _sb_torch.nn.Module):
            if _sb_torch.cuda.is_available():
                _m = _m.cuda()
            _m.eval()
        _SB_MODEL = _m
    with _sb_torch.no_grad():
        return _SB_MODEL(*args)
'''

SPEC = {
    "languages": ["pytorch"],
    "target_hardware": ["LOCAL"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch"],
    "destination_passing_style": False,
    "binding": None,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("out_dir")
    ap.add_argument("--name-suffix", default="kernelmem")
    ap.add_argument("--description", default="")
    args = ap.parse_args()

    src = Path(args.kernel).read_text(encoding="utf-8")
    if "ModelNew" not in src:
        raise SystemExit(f"{args.kernel} defines no ModelNew; not a KernelMem kernel")
    if "def run(" in src:
        raise SystemExit(f"{args.kernel} already defines run(); refusing to shadow it")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    solution = {
        "name": f"{DEFINITION}__{args.name_suffix}",
        "definition": DEFINITION,
        "author": "kernelmem",
        "description": args.description or f"packaged from {Path(args.kernel).name}",
        "spec": SPEC,
        "sources": [{"path": "kernel.py", "content": src + SHIM}],
    }
    dest = out / "solution.json"
    dest.write_text(json.dumps(solution, indent=1), encoding="utf-8")
    print(f"wrote {dest}  ({len(solution['sources'][0]['content'])} chars of kernel.py)")


if __name__ == "__main__":
    main()
