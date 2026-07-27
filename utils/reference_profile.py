"""Profile the *reference* model before the seed kernel is generated.

Rationale
---------
The seed prompt asks the LLM to commit to a granularity level (A/B/C/D) that
determines what it is allowed to rewrite, and that choice is never revisited in
later rounds. Without profiling data the model picks blind, so it reliably
chooses a conservative level and leaves the dominant operator on a library call
it can then never touch — capping the whole run before round 1.

This module runs the reference once under torch.profiler and reports where GPU
time actually goes, separating kernels the model could own from library kernels
(cuDNN / cuBLAS / CUTLASS / ATen) that can only be replaced by rewriting the op.

Runs in a subprocess so the parent never initialises a CUDA context.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

__all__ = ["profile_reference", "build_reference_profile_block"]

# Vendor GEMM/convolution kernels. These are tuned tensor-core implementations;
# matching one from scratch is a major undertaking, so time spent here is
# effectively fixed unless the whole operator is reimplemented.
_VENDOR_GEMM_MARKERS = (
    "implicit_gemm", "cutlass", "cublas", "xmma", "sgemm", "hgemm",
    "gemvx", "fprop", "dgrad", "wgrad", "winograd",
)
# Everything else the reference launches (elementwise, reductions, norms, layout
# conversions) is ordinary library code that a custom CUDA kernel can replace —
# this is exactly what KernelMem normally does.
# 20 iterations gave a 70-75% spread on the dominant kernel across repeat runs;
# 50 tightens that at negligible cost (a few hundred ms even on large shapes).
_DEFAULT_ITERS = 50


def _is_vendor_gemm(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in _VENDOR_GEMM_MARKERS)


def _shorten(name: str, width: int = 58) -> str:
    """Trim a mangled kernel name down to something readable."""
    n = name.strip()
    if n.startswith("void "):
        n = n[5:]
    # Cut template/arg lists only when the remaining prefix still names something.
    # "at::native::" alone is not informative, so require a real trailing symbol.
    for cut in ("<", "("):
        idx = n.find(cut)
        if 0 < idx < width:
            head = n[:idx].strip()
            if len(head) >= 16 and not head.endswith(":"):
                n = head
                break
    n = n.strip()
    return n if len(n) <= width else n[: width - 1] + "…"


# --------------------------------------------------------------------------
# child process: profile and emit JSON on stdout
# --------------------------------------------------------------------------
def _child_main(task_py: str, device_idx: int, iters: int) -> int:
    import importlib.util

    import torch
    from torch.profiler import ProfilerActivity, profile

    spec = importlib.util.spec_from_file_location("_ref_task_prof", task_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dev = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(dev)
    model = mod.Model(*(mod.get_init_inputs() or [])).to(dev).eval()
    inputs = [x.to(dev) if torch.is_tensor(x) else x for x in mod.get_inputs()]

    with torch.no_grad():
        for _ in range(5):  # warmup
            model(*inputs)
        torch.cuda.synchronize(dev)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(*inputs)
            torch.cuda.synchronize(dev)

    rows = []
    for e in prof.key_averages():
        # torch >=2.1 renamed self_cuda_time_total -> self_device_time_total
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0)
        if t and t > 0:
            rows.append({"name": e.key, "us": float(t) / iters, "count": e.count // iters})

    total = sum(r["us"] for r in rows)
    rows.sort(key=lambda r: -r["us"])
    print("@@JSON@@" + json.dumps({"total_us": total, "kernels": rows}))
    return 0


# --------------------------------------------------------------------------
# parent side
# --------------------------------------------------------------------------
def profile_reference(
    task_py: Path | str,
    device_idx: int = 0,
    iters: int = _DEFAULT_ITERS,
    timeout: int = 600,
) -> Optional[dict[str, Any]]:
    """Return {'total_us', 'kernels': [...]} for the reference, or None on failure."""
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(task_py),
             str(device_idx), str(iters)],
            capture_output=True, text=True, timeout=timeout,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("@@JSON@@"):
                return json.loads(line[len("@@JSON@@"):])
        print(f"[ref_profile] no result from profiler (rc={proc.returncode}); "
              f"skipping reference profile")
        if proc.stderr:
            print(f"[ref_profile] stderr tail: {proc.stderr.strip().splitlines()[-1:]}")
    except Exception as exc:  # profiling is advisory - never break the run
        print(f"[ref_profile] failed ({type(exc).__name__}: {exc}); skipping")
    return None


def build_reference_profile_block(
    task_py: Path | str,
    device_idx: int = 0,
    iters: int = _DEFAULT_ITERS,
    top_n: int = 8,
) -> Optional[str]:
    """Render the reference profile as a prompt block, or None if unavailable."""
    data = profile_reference(task_py, device_idx, iters)
    if not data or not data.get("kernels") or data.get("total_us", 0) <= 0:
        return None

    total = data["total_us"]
    kernels = data["kernels"][:top_n]
    gemm_us = sum(k["us"] for k in data["kernels"] if _is_vendor_gemm(k["name"]))
    gemm_pct = gemm_us / total * 100.0
    # Amdahl: if fraction f is left untouched, the best achievable speedup is 1/f
    # even when every other kernel is made completely free.
    bound = 1.0 / max(1e-6, gemm_pct / 100.0) if gemm_pct > 0 else float("inf")

    lines = [
        "REFERENCE PROFILE (measured on this GPU, not estimated)",
        f"The unmodified reference takes {total:.1f} us of GPU time per forward.",
        "Where that time actually goes:",
        "",
        f"  {'% time':>7}  {'us':>9}  {'calls':>5}  kernel",
    ]
    for k in kernels:
        tag = "   <-- vendor GEMM/conv" if _is_vendor_gemm(k["name"]) else ""
        lines.append(
            f"  {k['us'] / total * 100:>6.1f}%  {k['us']:>9.1f}  {k['count']:>5}  "
            f"{_shorten(k['name'])}{tag}"
        )

    if gemm_pct > 0:
        lines += [
            "",
            f"Vendor GEMM/convolution kernels are {gemm_pct:.1f}% of GPU time.",
            "These are tuned tensor-core implementations. Ordinary library kernels",
            "(elementwise, reductions, norms, layout conversions) are straightforward",
            "to replace with your own CUDA; a vendor GEMM/conv is not.",
            "",
            "USE THIS WHEN CHOOSING GRANULARITY:",
            f"- Leaving the vendor GEMM/conv in place fixes {gemm_pct:.1f}% of the runtime.",
            f"  By Amdahl's law your speedup is then capped at 1/{gemm_pct / 100:.3f} = "
            f"{bound:.2f}x,",
            "  and only if you drive ALL other work to zero. Judge whether that cap is",
            "  worth the effort before committing to a granularity.",
            "- Do NOT assume the vendor kernel is slow. It is often at or near the",
            "  hardware roofline, in which case reimplementing it wins nothing by itself.",
            "  The gain from owning it comes from FUSING the surrounding work into its",
            "  prologue/epilogue so intermediates are never written to global memory.",
            "- Owning that operator requires granularity (D). If you choose (A)/(B)/(C),",
            f"  you are accepting the {bound:.2f}x cap for the entire run — later rounds",
            "  cannot revisit this decision.",
        ]
    else:
        lines += [
            "",
            "No vendor GEMM/convolution kernel dominates: every kernel above is",
            "ordinary library code that custom CUDA can replace directly.",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
