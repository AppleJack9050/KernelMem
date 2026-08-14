from __future__ import annotations
"""
compare_and_bench.py – single-GPU benchmark (**full compile + runtime traceback**).

Key features
------------
* Dynamically imports two PyTorch models (reference & candidate) and **captures
  every byte** printed by Python *and* child processes (ninja / nvcc).
  - On any *build* failure, raises `CompilationError(full_log)`.
* On **runtime failure** (forward, benchmark, accuracy), re-raises
  `RuntimeError(traceback.format_exc())` so callers get the *entire*
  traceback – not just `str(exc)`.
* Benchmarks on CUDA (default) or CPU (`--cpu`).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import signal
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch

from utils import clock_lock, device_state

# ---------------------------------------------------------------------------

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------


class CompilationError(RuntimeError):
    """Raised when dynamic import / nvcc build fails.

    The *first* argument is the full build log (Python + ninja/nvcc).
    """


class CompilationTimeoutError(CompilationError):
    """Raised when compilation exceeds the timeout limit."""


class AccuracyError(RuntimeError):
    """Raised when outputs do not meet the accuracy tolerance."""


# =========================== dynamic import ===============================
def _timeout_handler(signum, frame):
    """Handler for compilation timeout."""
    raise CompilationTimeoutError("Compilation exceeded timeout limit (10 minutes)")


def _capture_import(path: Path, timeout: int = 600):
    """Import *path* dynamically and capture **all** build logs.

    Parameters
    ----------
    path : Path
        Path to the Python file to import.
    timeout : int, optional
        Compilation timeout in seconds (default: 600 = 10 minutes).

    Returns
    -------
    (module, full_log : str)

    Raises
    ------
    FileNotFoundError
        *path* does not exist.
    CompilationTimeoutError
        Compilation exceeded the timeout limit.
    CompilationError
        Any Python / ninja / nvcc error during import.  The exception's first
        argument is the concatenated log.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)                     # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    # ---- Python-level stdout/stderr to StringIO --------------------------
    py_buf = io.StringIO()

    # Set up compilation timeout (Unix-only)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    
    # ---- OS-level FD 1/2 (stdout/stderr) to a temp file -----------------
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
         contextlib.redirect_stdout(py_buf), \
         contextlib.redirect_stderr(py_buf):

        # Save current FDs so we can restore later
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)     # redirect FD 1 → temp file
            os.dup2(fd_buf.fileno(), 2)     # redirect FD 2 → temp file

            # ------------ REAL IMPORT (build/compile) with timeout --------------------
            signal.alarm(timeout)  # Start the timeout timer
            spec.loader.exec_module(module)                             # pyright: ignore[attr-defined]
            signal.alarm(0)  # Cancel the alarm if compilation succeeds

            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()

        except CompilationTimeoutError as exc:
            # Timeout occurred
            fd_buf.flush(); fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([
                py_buf.getvalue(), 
                subproc_log, 
                f"\n[TIMEOUT] Compilation exceeded {timeout}s limit. "
                "This may indicate issues with the kernel itself, such as "
                "infinite loops in template expansion, or compiler bugs. "
                "Please investigate potential causes and fix them."
            ]).strip()
            raise CompilationTimeoutError(full_log) from None

        except Exception as exc:  # ← build / link / import failed
            # Combine StringIO + temp-file logs + Exception str
            signal.alarm(0)  # Cancel alarm on error
            fd_buf.flush(); fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
            raise CompilationError(full_log) from None

        finally:
            # Always restore original FDs and signal handler
            signal.alarm(0)  # Ensure alarm is cancelled
            signal.signal(signal.SIGALRM, old_handler)
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    # ---------------- SUCCESS --------------------------------------------
    return module, py_buf.getvalue() + subproc_log


# =========================== timing helpers ===============================
def _run_once(model: torch.nn.Module,
              inp: List[torch.Tensor],
              dev: torch.device) -> Tuple[torch.Tensor, float]:
    model.to(dev).eval()
    # Some tasks (e.g., matrix + scalar) pass Python scalars alongside tensors.
    # Only move objects that actually support `.to()`.
    moved_inp = []
    for x in inp:
        if hasattr(x, "to"):
            moved_inp.append(x.to(dev))
        else:
            moved_inp.append(x)
    inp = moved_inp

    if TORCH_DEVICE == "cpu":
        t0 = datetime.now()
        out = model(*inp)
        ms = (datetime.now() - t0).total_seconds() * 1_000
        return out, ms

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(dev)
    start.record()
    out = model(*inp)
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end)


def _bench(model: torch.nn.Module,
           inp: List[torch.Tensor],
           dev: torch.device,
           warm: int,
           rep: int) -> List[float]:
    model.to(dev).eval()
    moved_inp = []
    for x in inp:
        if hasattr(x, "to"):
            moved_inp.append(x.to(dev))
        else:
            moved_inp.append(x)
    inp = moved_inp

    for _ in range(warm):
        model(*inp)

    if TORCH_DEVICE == "cpu":
        res = []
        for _ in range(rep):
            t0 = datetime.now()
            model(*inp)
            res.append((datetime.now() - t0).total_seconds() * 1_000)
        return res

    torch.cuda.synchronize(dev)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    times: List[float] = []
    for _ in range(rep):
        s.record()
        model(*inp)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return times


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for t in x:
            if isinstance(t, torch.Tensor):
                return t
    raise TypeError("Model forward did not return a Tensor (or a Tensor inside a sequence).")


def _shape_tag(inp) -> str:
    """Human-readable tag for an input tuple, used to name a shape in messages."""
    for x in inp:
        if torch.is_tensor(x) and x.dim() >= 2:
            return "x".join(str(d) for d in x.shape)
    return "?"


# ================ uninitialized-memory detection (allocator poisoning) ======
# A kernel that reads a buffer it allocated with at::empty() but never fully
# wrote is a real defect, yet it hides from a plain correctness check: pages
# handed out fresh by the CUDA driver arrive zeroed, so the wrong kernel looks
# right on a clean allocator and only misbehaves later, once the caching
# allocator starts recycling blocks that held live data. Whether it is caught
# then depends on allocation history, which makes the failure look random.
#
# Poisoning the allocator's free blocks with NaN before the kernel runs turns
# that into a deterministic, immediate failure. NaN is used deliberately: it
# propagates through arithmetic, so *any* read of unwritten memory reaches the
# output regardless of how small the stray values would otherwise have been.
# The two runs are made symmetric on purpose: the SAME size ladder is filled
# both times, and only the fill VALUE differs (0.0 for the reference run, NaN
# for the adversarial one). Any block the poison can reach was therefore also
# reached by the zero-fill, so a difference between the two runs can only come
# from memory the kernel read without writing.
#
# Two things that look like they should work, do not, and both were tried:
#   * torch.cuda.empty_cache() does NOT hand back zeroed memory. The driver
#     scrubs pages between PROCESSES, not within one, so cudaFree + cudaMalloc
#     returns the same physical pages with the previous contents intact.
#   * A descending (largest-first) fill misses the small pool entirely: PyTorch
#     serves sub-1MB requests from separate segments, and the budget is gone
#     before the ladder ever gets down there -- which is exactly where the
#     small scratch buffers this bug class lives in are found.
_POISON_MIN_ELEMS = 256                # 1 KB   - smallest size class filled
_POISON_MAX_ELEMS = (128 << 20) // 4   # 128 MB - largest
_POISON_PER_CLASS = 4                  # blocks per size class
_POISON_FREE_FRACTION = 4              # take at most 1/N of driver-free VRAM


def _fill_free_blocks(dev: torch.device, value: float) -> int:
    """Write *value* into the caching allocator's free blocks; return bytes written.

    Blocks are released back to *PyTorch's pool*, not to the driver, so the next
    at::empty() of a matching size is handed one of them. The ladder ascends so
    that the small pool is filled before the budget is spent.
    """
    if TORCH_DEVICE != "cuda":
        return 0
    try:
        driver_free, _ = torch.cuda.mem_get_info(dev)
    except Exception:
        return 0

    budget = driver_free // _POISON_FREE_FRACTION
    junk: List[torch.Tensor] = []
    used = 0
    n = _POISON_MIN_ELEMS
    while n <= _POISON_MAX_ELEMS:
        for _ in range(_POISON_PER_CLASS):
            nbytes = n * 4
            if used + nbytes > budget:
                n = _POISON_MAX_ELEMS  # out of budget; stop after this class
                break
            try:
                junk.append(torch.full((n,), value, device=dev, dtype=torch.float32))
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                n = _POISON_MAX_ELEMS
                break
            used += nbytes
        n *= 2

    del junk  # -> returned to PyTorch's pool, NOT to the driver
    return used


def _poison_allocator(dev: torch.device) -> int:
    """Fill free blocks with NaN; returns bytes poisoned (0 = check is inert)."""
    return _fill_free_blocks(dev, float("nan"))


def _fresh_allocator(dev: torch.device) -> int:
    """Fill free blocks with 0.0 - the benign counterpart of the poisoned run.

    Used before every reference run, and again after the poisoned probe so the
    timing loop and the following shape do not inherit NaN.
    """
    return _fill_free_blocks(dev, 0.0)


def _allocator_dependence_note(test_model: torch.nn.Module,
                               inp: List[torch.Tensor],
                               dev: torch.device,
                               out_clean: torch.Tensor,
                               tol: float,
                               shape_tag: str) -> str | None:
    """Re-run *test_model* against a poisoned allocator and report if it changed.

    Returns a diagnosis string when the kernel's output depends on what the
    allocator happened to hand it, or ``None`` when the output is stable (the
    normal case). Purely advisory: any failure inside this check returns None
    rather than masking or inventing a result.

    Limitation, stated because it is easy to assume otherwise: the reference leg
    is not guaranteed to run on pristine memory. Neither empty_cache() nor a
    size-ladder zero-fill can promise that, because the driver does not scrub
    pages within a process and a kernel launched between fill and probe
    redistributes the pool. So the reference leg may itself carry NaN from an
    earlier shape's probe.

    This costs specificity, not soundness. A kernel that never reads unwritten
    memory is identical in both legs and is never flagged; a kernel that does is
    rejected either here or by the plain tolerance check that follows, which sees
    the NaN directly. For an exact answer rather than a differential one, run
    ``compute-sanitizer --tool initcheck`` on the candidate.
    """
    if not _poison_allocator(dev):
        return None
    try:
        out, _ = _run_once(test_model, inp, dev)
        if TORCH_DEVICE == "cuda":
            torch.cuda.synchronize(dev)
        out = _first_tensor(out).contiguous()
        if out.dtype != out_clean.dtype:
            out = out.to(out_clean.dtype)
        if out.device != out_clean.device:
            out = out.to(out_clean.device)
        if out.shape != out_clean.shape:
            return None

        nan_now, nan_before = torch.isnan(out), torch.isnan(out_clean)
        fresh_nan = int((nan_now & ~nan_before).sum().item())
        differs = (~torch.isclose(out, out_clean, atol=tol, rtol=tol)) | (nan_now ^ nan_before)
        n_differs = int(differs.sum().item())
        if n_differs == 0:
            return None
        total = out.numel()
        # Report the largest *finite* disagreement; NaN-vs-number is counted
        # above and would otherwise render the whole line as "nan".
        drift = (out - out_clean).abs()
        drift = drift[torch.isfinite(drift)]
        drift_line = (f"  largest finite disagreement : {drift.max().item():.6e}\n"
                      if drift.numel() else
                      "  largest finite disagreement : n/a (every difference involves NaN)\n")
    except Exception:
        return None
    finally:
        # Never leave NaN in the pool for the timing loop or the next shape.
        _fresh_allocator(dev)

    return (
        "UNINITIALIZED GPU MEMORY: this kernel's output depends on what the CUDA "
        f"caching allocator previously stored, on shape {shape_tag}.\n"
        "Run twice on identical inputs, differing only in what was left in the "
        "allocator's free blocks (the second run had them filled with NaN), it "
        "produced different results.\n"
        f"  output elements that changed: {n_differs} of {total} "
        f"({100.0 * n_differs / max(1, total):.2f}%)\n"
        f"  of those, newly NaN          : {fresh_nan}\n"
        f"{drift_line}"
        "A buffer allocated with at::empty()/torch::empty() is being READ at "
        "positions that were never WRITTEN. This most often happens when a grid is "
        "sized from one rounding of a division and the per-block work from another, "
        "so trailing blocks exit early without storing their slot, while the "
        "consumer kernel still sums every slot.\n"
        "Fix the producer/consumer to agree on the exact element count, or "
        "zero-initialise the buffer. Note that this bug is INVISIBLE on a clean "
        "allocator, so it may reproduce only for some input shapes."
    )


# ======================= RNG & determinism settings =======================
def _seed_everything(seed: int | None, device_idx: int | None = None):
    """Set the random seeds and (optionally) enable deterministic backends."""
    import os, random
    import numpy as np
    import torch

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if device_idx is not None:
            torch.cuda.set_device(device_idx)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Stronger reproducibility (comment out if not needed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # or ":16:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some ops have no deterministic implementation: warn instead of erroring
        torch.use_deterministic_algorithms(True, warn_only=True)


# ======= Parameter alignment (generic + class/export-name specific) =======
import torch
import torch.nn as nn
from collections import defaultdict

def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    named: dict[str, torch.Tensor] = {}
    for k, p in model.named_parameters(recurse=True):
        named[f"param::{k}"] = p
    for k, b in model.named_buffers(recurse=True):
        named[f"buffer::{k}"] = b
    return named

@torch.no_grad()
def _safe_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.shape != src.shape:
        return False
    dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    return True

@torch.no_grad()
def _try_map_shape_and_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """
    Supported shape mappings:
      - depthwise 2D:   (C,1,Kh,1)<->(C,Kh), (C,1,Kh,Kw)<->(C,Kh,Kw)
      - PW/Linear:      (Out,In,1,1)<->(Out,In)
      - Conv/ConvT 3D:  (Out,In,kD,kH,kW) <-> (In,Out,kD,kH,kW) (first two dims swapped)
      - depthwise 3D:   (C,1,kD,kH,kW) <-> (C,kD,kH,kW)
    """
    s = tuple(src.shape)
    d = tuple(dst.shape)

    # --- depthwise 2D: (C,1,Kh,1) <-> (C,Kh)
    if len(s) == 4 and s[1] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[2] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[1] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- depthwise 2D: (C,1,Kh,Kw) -> (C,Kh,Kw) and the reverse
    if len(s) == 4 and s[1] == 1 and len(d) == 3 and s[0] == d[0] and s[2] == d[1] and s[3] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 3 and len(d) == 4 and d[1] == 1 and s[0] == d[0] and s[1] == d[2] and s[2] == d[3]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    # --- PW/Linear: (Out,In,1,1) <-> (Out,In)
    if len(s) == 4 and s[2] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[2] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- Conv/ConvTranspose 3D: swap the first two dims of the 5D weight
    #     (Out, In, kD, kH, kW)  <->  (In, Out, kD, kH, kW)
    if len(s) == 5 and len(d) == 5 and s[0] == d[1] and s[1] == d[0] and s[2:] == d[2:]:
        dst.copy_(src.permute(1, 0, 2, 3, 4).contiguous().to(dtype=dst.dtype, device=dst.device))
        return True

    # --- depthwise 3D: (C,1,kD,kH,kW) -> (C,kD,kH,kW) and the reverse
    if len(s) == 5 and s[1] == 1 and len(d) == 4 and s[0] == d[0] and s[2:] == d[1:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 4 and len(d) == 5 and d[1] == 1 and s[0] == d[0] and s[1:] == d[2:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    return False

@torch.no_grad()
def align_params_generic(ref_model: nn.Module, test_model: nn.Module) -> dict[str, int]:
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    copied_same, unique_shape_copied, mapped, skipped = 0, 0, 0, 0
    aligned_test: set[str] = set()

    # 1) Same name, same shape
    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None and _safe_copy_(t_dst, t_src):
            copied_same += 1
            aligned_test.add(name)

    # 2) Unique shape match
    shape2ref: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    shape2test: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for n, t in ref_named.items():
        shape2ref[tuple(t.shape)].append((n, t))
    for n, t in test_named.items():
        if n in aligned_test: 
            continue
        shape2test[tuple(t.shape)].append((n, t))

    for shp, items in shape2test.items():
        if len(items) == 1 and len(shape2ref.get(shp, [])) == 1:
            tname, t_dst = items[0]
            _, t_src = shape2ref[shp][0]
            if _safe_copy_(t_dst, t_src):
                unique_shape_copied += 1
                aligned_test.add(tname)

    # 3) Shape mapping
    for name, t_dst in test_named.items():
        if name in aligned_test:
            continue
        ok = False
        for _, t_src in ref_named.items():
            if _try_map_shape_and_copy_(t_dst, t_src):
                mapped += 1
                aligned_test.add(name)
                ok = True
                break
        if not ok:
            skipped += 1

    return {
        "copied_same_shape": copied_same,
        "unique_shape_copied": unique_shape_copied,
        "mapped_shape": mapped,
        "skipped": skipped,
    }

# (Optional) Register a "dedicated aligner" by class name/export name: Model → ModelNew
_PAIR_ALIGNERS: dict[tuple[str, str], callable] = {}

def register_pair_aligner(ref_key: str, test_key: str):
    def deco(fn):
        _PAIR_ALIGNERS[(ref_key, test_key)] = fn
        return fn
    return deco

@register_pair_aligner("Model", "ModelNew")
@torch.no_grad()
def _align_Model_to_ModelNew(ref_model: nn.Module, test_model: nn.Module) -> dict[str, int]:
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    def pick(named: dict[str, torch.Tensor], dims: int):
        cand = [(n, t) for n, t in named.items()
                if n.startswith("param::") and "weight" in n and t.ndim == dims]
        if not cand:
            cand = [(n, t) for n, t in named.items()
                    if n.startswith("param::") and t.ndim == dims]
        return cand

    # ---- 2D: Conv / ConvTranspose (4D same shape or first two dims swapped) ----
    r4 = pick(ref_named, 4); t4 = pick(test_named, 4)
    if len(r4) == 1 and len(t4) == 1:
        w_ref, w_tst = r4[0][1], t4[0][1]
        if tuple(w_ref.shape) == tuple(w_tst.shape):
            w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device))
            pass_bias = True
        elif (w_ref.shape[0] == w_tst.shape[1] and w_ref.shape[1] == w_tst.shape[0]
              and w_ref.shape[2:] == w_tst.shape[2:]):
            w_tst.copy_(w_ref.permute(1, 0, 2, 3).contiguous().to(dtype=w_tst.dtype, device=w_tst.device))
            pass_bias = True
        else:
            pass_bias = False

        if pass_bias:
            rb = [(n,t) for n,t in ref_named.items() if "bias" in n and n.startswith("param::") and t.ndim==1]
            tb = [(n,t) for n,t in test_named.items() if "bias" in n and n.startswith("param::") and t.ndim==1]
            if len(rb)==1 and len(tb)==1 and tuple(rb[0][1].shape)==tuple(tb[0][1].shape):
                tb[0][1].copy_(rb[0][1].to(dtype=tb[0][1].dtype, device=tb[0][1].device))
            return {"pair_aligner": 1, "copied_same_shape": int(tuple(w_ref.shape)==tuple(w_tst.shape)),
                    "mapped_shape": int(tuple(w_ref.shape)!=tuple(w_tst.shape)), "skipped": 0}

    # ---- 3D: Conv3d / ConvTranspose3d (5D same shape or first two dims swapped) ----
    r5 = pick(ref_named, 5); t5 = pick(test_named, 5)
    if len(r5) == 1 and len(t5) == 1:
        w_ref, w_tst = r5[0][1], t5[0][1]
        if tuple(w_ref.shape) == tuple(w_tst.shape):
            w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device))
            return {"pair_aligner": 1, "copied_same_shape": 1, "mapped_shape": 0, "skipped": 0}
        if (w_ref.shape[0] == w_tst.shape[1] and w_ref.shape[1] == w_tst.shape[0]
                and w_ref.shape[2:] == w_tst.shape[2:]):
            w_tst.copy_(w_ref.permute(1, 0, 2, 3, 4).contiguous().to(dtype=w_tst.dtype, device=w_tst.device))
            return {"pair_aligner": 1, "copied_same_shape": 0, "mapped_shape": 1, "skipped": 0}

    # ---- depthwise-3D: (C,1,kD,kH,kW) ↔ (C,kD,kH,kW) ----
    if len(r5) == 1:
        w_ref = r5[0][1]
        t4 = pick(test_named, 4)
        if len(t4) == 1:
            w_tst = t4[0][1]
            if w_ref.size(1) == 1 and tuple(w_tst.shape) == (w_ref.size(0), w_ref.size(2), w_ref.size(3), w_ref.size(4)):
                w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device).squeeze(1).contiguous())
                return {"pair_aligner": 1, "copied_same_shape": 0, "mapped_shape": 1, "skipped": 0}

    # Fall back to the generic aligner for everything else
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    return stats

@torch.no_grad()
def try_align_params(ref_model: nn.Module, test_model: nn.Module,
                     ref_mod=None, test_mod=None) -> dict[str, int]:
    """
    Priority:
      0) Dispatch by export name (_export_symbol), e.g. ("Model","ModelNew")
      0b) Dispatch by instance class name
      1) Task-specific map_ref_to_test_params / align_params
      2) Generic automatic alignment
    """
    # 0) Export name (if compare_and_bench already set it)
    key_export = (getattr(ref_model, "_export_symbol", None),
                  getattr(test_model, "_export_symbol", None))
    if key_export in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_export](ref_model, test_model)
        stats["pair_key"] = f"{key_export[0]}->{key_export[1]}"
        return stats

    # 0b) Instance class name
    key_class = (ref_model.__class__.__name__, test_model.__class__.__name__)
    if key_class in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_class](ref_model, test_model)
        stats["pair_key"] = f"{key_class[0]}->{key_class[1]}"
        return stats

    # 1) Task-specific
    for mod in (test_mod, ref_mod):
        if mod is None:
            continue
        for fn_name in ("map_ref_to_test_params", "align_params"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(ref_model, test_model)
                return {"pair_aligner": 0, "copied_same_shape": -1, "mapped_shape": -1,
                        "skipped": -1, "pair_key": "custom_fn"}

    # 2) Generic
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    stats["pair_key"] = "generic"
    return stats



# ========= compare_and_bench (with generic alignment and seeding) =========
def compare_and_bench(
    ref_py: Path,
    test_py: Path,
    *,
    device_idx: int = 0,
    warmup: int = 5,
    repeat: int = 20,
    tol: float = 1e-4,
    log_dir: str | Path | None = "run/debug",
    seed: int = 100,  # Fixed default seed; set it to None to read the seed from the env instead
    reject_on_state_leak: bool = True,
    time_ref: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark *test_py* against *ref_py*.

    Reads get_init_inputs() only from the reference script, and uses the same init args for ref/test.
    Also: fixed randomness + parameter alignment (supports the Model→ModelNew dedicated aligner & generic alignment).

    *time_ref* controls whether the REFERENCE is timed. It must stay True for any
    caller that reads ``score`` or ``speedup`` -- those are ``T_ref / T_test`` and
    have no meaning without it. Pass False when only the candidate's absolute time
    is wanted, and ``ref_ms``/``speedup``/``score`` come back None rather than
    fabricated.

    Why the switch exists: the reference is timed at the full warmup+repeat count
    on every shape, and it is the SLOWER side by construction (that is what a
    speedup above 1 means), so it is the majority of the benchmark's GPU time.
    ``utils.paired_bench`` consumes only ``test_ms`` and threw all of it away --
    on a 1.2x kernel over this repo's four shapes that is ~55% of every rep, paid
    3-8 times per ratchet decision. Correctness still runs the reference, but that
    needs ONE forward pass for the output comparison, not warmup+repeat of them.
    """
    import os
    import contextlib
    from datetime import datetime

    # ------------ Device setup ------------
    dev = torch.device(f"cuda:{device_idx}") if TORCH_DEVICE == "cuda" else torch.device("cpu")
    if TORCH_DEVICE == "cuda":
        torch.cuda.set_device(dev)
        # Nothing is timed until the GPU clock is pinned. This is the choke point
        # every measurement passes through -- the seed loop, the repair path, the
        # optimization path, paired_bench, the noise probes and each spawned
        # benchmark subprocess -- so validating here covers them all by
        # construction, the same reason gpu_section() is wrapped around this
        # function rather than around its callers. Already pinned by the parent
        # run (the normal case) costs one nvidia-smi query to re-verify; not
        # pinned at all locks it now; unlockable stops the run rather than
        # producing an unreproducible number.
        clock_lock.ensure_locked(device_idx, what="benchmark",
                                 verbose=clock_lock.owner_pid() is None)
        # Set CUDA_LAUNCH_BLOCKING=1 to get detailed error messages for "unspecified launch failure"
        # This makes CUDA operations synchronous and provides better error reporting
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # In case the seed needs to be controlled through an environment variable
    if seed is None:
        env_seed = os.environ.get("KERNELBENCH_SEED")
        seed = int(env_seed) if env_seed is not None else None

    def _build_mismatch_debug_msg(*,
                                  reason: str,
                                  tol: float,
                                  max_err: float,
                                  mean_err: float,
                                  ref_out: torch.Tensor,
                                  test_out: torch.Tensor,
                                  diff: torch.Tensor,
                                  inputs: List[Any],
                                  ref_model: torch.nn.Module) -> str:
        """Build a rich debug string for mismatch cases."""
        lines: List[str] = []
        lines.append(f"{reason} (atol={tol}, rtol={tol}). "
                     f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}")

        # How much of the tensor is wrong separates a boundary/indexing slip from
        # a wholesale-wrong result; the max alone cannot distinguish them.
        try:
            bad = diff > (tol + tol * ref_out.abs())
            n_bad = int(bad.sum().item())
            n_tot = int(bad.numel())
            pct = (100.0 * n_bad / n_tot) if n_tot else 0.0
            lines.append(f"mismatched_elements={n_bad}/{n_tot} ({pct:.2f}%)")
        except Exception as _e:
            lines.append(f"[debug] failed to count mismatched elements: {_e}")

        # Input tensor info
        for i, x in enumerate(inputs):
            if isinstance(x, torch.Tensor):
                lines.append(
                    f"in[{i}]: shape={tuple(x.shape)}, dtype={x.dtype}, contiguous={x.is_contiguous()}, device={x.device}"
                )
            else:
                lines.append(f"in[{i}]: type={type(x).__name__} (non-tensor)")

        # Model parameter info (best-effort)
        weight = getattr(ref_model, "weight", None)
        if isinstance(weight, torch.Tensor):
            lines.append(
                f"ref.weight: shape={tuple(weight.shape)}, dtype={weight.dtype}, contiguous={weight.is_contiguous()}, device={weight.device}"
            )
        bias = getattr(ref_model, "bias", None)
        if isinstance(bias, torch.Tensor):
            lines.append(
                f"ref.bias: shape={tuple(bias.shape)}, dtype={bias.dtype}, contiguous={bias.is_contiguous()}, device={bias.device}"
            )
        # Common convolution attributes (best-effort)
        for name in ("stride", "padding", "output_padding", "groups", "dilation", "kernel_size"):
            if hasattr(ref_model, name):
                lines.append(f"ref.{name}={getattr(ref_model, name)}")

        # Output shapes/dtypes
        lines.append(f"ref_out: shape={tuple(ref_out.shape)}, dtype={ref_out.dtype}, device={ref_out.device}")
        lines.append(f"test_out: shape={tuple(test_out.shape)}, dtype={test_out.dtype}, device={test_out.device}")

        # Max error index and values (on CPU for logging safety)
        try:
            diff_cpu = diff.detach().to("cpu")
            max_idx_flat = diff_cpu.argmax()
            coord = torch.unravel_index(max_idx_flat, diff_cpu.shape)
            ref_cpu = ref_out.detach().to("cpu")
            test_cpu = test_out.detach().to("cpu")
            ref_val = ref_cpu[coord].item()
            test_val = test_cpu[coord].item()
            lines.append(f"max_err_index={tuple(int(c) for c in coord)}, ref_val={ref_val}, cand_val={test_val}")
        except Exception as _e:
            lines.append(f"[debug] failed to extract max_err index/value: {_e}")

        # State the evaluation contract explicitly. Every shape is scored with
        # freshly allocated tensors and earlier ones are freed first, so the
        # caching allocator routinely hands a new tensor the address of a dead
        # one. A cache keyed on data_ptr() then returns another tensor's data,
        # which reads as an arithmetic bug and sends the repair after the math.
        lines.append(
            "NOTE ON EVALUATION: this candidate is scored on several input shapes, "
            "each with FRESHLY ALLOCATED tensors, and tensors from earlier shapes are "
            "freed first. PyTorch's caching allocator therefore reuses addresses across "
            "calls. Any cache keyed on data_ptr() MUST hold a strong reference to the "
            "source tensor and verify identity (cached_src is src), or it will silently "
            "return a stale entry belonging to a freed tensor."
        )

        return "\n".join(lines)

    # ------------ Dynamic import ------------
    ref_mod, _ = _capture_import(ref_py)
    test_mod, _ = _capture_import(test_py)

    RefModel   = getattr(ref_mod,  "Model",       None)
    get_inputs = getattr(ref_mod,  "get_inputs",  None)
    ModelNew   = getattr(test_mod, "ModelNew",    None)

    if None in (RefModel, get_inputs):
        raise RuntimeError(f"Reference '{ref_py}' must define Model and get_inputs().")
    if ModelNew is None:
        raise RuntimeError(f"Candidate '{test_py}' must define a ModelNew class.")

    # ------------ Init args come from ref only ------------
    init_args: List[Any] = []
    init_kwargs: Dict[str, Any] = {}
    get_init_inputs_ref = getattr(ref_mod, "get_init_inputs", None)

    if callable(get_init_inputs_ref):
        init_obj = get_init_inputs_ref()
        if isinstance(init_obj, dict):
            init_kwargs = dict(init_obj)
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)
        elif init_obj is not None:
            raise TypeError("get_init_inputs() must return a list/tuple (used as *args) or a dict (used as **kwargs).")

    # ------------ Run & benchmark ------------
    _leaks: List[str] = []
    _clean_state: Dict[str, Any] = {}
    try:
        ctx = torch.cuda.device(dev) if TORCH_DEVICE == "cuda" else contextlib.nullcontext()
        with ctx:
            # Pin down input randomness
            _seed_everything(seed, device_idx)
            inp = get_inputs()
            if not isinstance(inp, (list, tuple)):
                inp = [inp]

            # Pin down parameter initialization: reseed before constructing each side
            _seed_everything(seed, device_idx)
            ref_model  = RefModel(*init_args, **init_kwargs)
            # # torch.compile speedup
            # ref_model = torch.compile(ref_model)
            
            _seed_everything(seed, device_idx)

            # Baseline the device-global state HERE: the harness has finished
            # configuring itself (_seed_everything sets the cudnn/determinism
            # flags just above) but the candidate has not been constructed or run
            # yet. Snapshotting any earlier attributes the harness's own settings
            # to the candidate, which false-positives on every kernel.
            # Every reference timing below is forced back to this, so T_ref stays
            # a constant of (task, shape, hardware) rather than something the
            # candidate can move. See utils/device_state.py for the incident.
            _clean_state = device_state.snapshot()

            test_model = ModelNew(*init_args, **init_kwargs)
            # # torch.compile speedup
            # test_model = torch.compile(test_model)   
            
            # ★ Parameter alignment (Model→ModelNew dedicated aligner first, then task-specific, then generic)
            align_stats = try_align_params(ref_model, test_model, ref_mod=ref_mod, test_mod=test_mod)

            # Forward pass (synchronize so errors surface where they happen)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)
            ref_out,  _ = _run_once(ref_model,  inp, dev)
            # Give the candidate the cleanest allocator it could ever see, so the
            # comparison against the poisoned run below is a controlled two-point
            # test rather than a function of allocation history.
            _fresh_allocator(dev)
            test_out, _ = _run_once(test_model, inp, dev)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            # Normalize to a Tensor and make sure it is contiguous
            ref_out  = _first_tensor(ref_out).contiguous()
            test_out = _first_tensor(test_out).contiguous()
            if ref_out.dtype != test_out.dtype:
                test_out = test_out.to(ref_out.dtype)

            # Make sure both tensors are on the same device (fixes device mismatch issues)
            if ref_out.device != test_out.device:
                # Move both tensors onto ref_out's device
                test_out = test_out.to(ref_out.device)

            # Check memory usage
            ref_out_bytes = ref_out.element_size() * ref_out.nelement()
            check_precision = True  # Initialize the flag

            if ref_out_bytes * 8 > 40 * 1024**3:
                import psutil
                from utils.print_utils import print_warning
                
                # Estimate CPU memory needed (3x safety factor for copy + diff)
                needed_cpu_mem = ref_out_bytes * 3
                avail_cpu_mem = psutil.virtual_memory().available
                
                if avail_cpu_mem < needed_cpu_mem:
                    print_warning(f"Skipping precision check: Tensor too large for both GPU and CPU RAM (Need ~{needed_cpu_mem/1024**3:.1f}GB, Avail {avail_cpu_mem/1024**3:.1f}GB)")
                    check_precision = False
                    # Release tensors to free memory for benchmarking
                    del ref_out, test_out
                    if TORCH_DEVICE == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print_warning(f"Warning: Output tensor size is too large ({ref_out_bytes / 1024**3:.2f} GB). Moving to CPU for comparison to avoid OOM.")
                    ref_out = ref_out.cpu()
                    test_out = test_out.cpu()
                    # Make sure both tensors are on the same device again
                    if ref_out.device != test_out.device:
                        test_out = test_out.to(ref_out.device)

            # Error & allclose (only run when check_precision is True)
            if check_precision:
                # Compute the diff first (keep the original dtype to preserve precision)
                diff = (test_out - ref_out).abs()
                max_err = diff.max().item()
                
                # mean() requires a floating/complex dtype, so cast diff to float if it is integral
                # NOTE: the cast only happens for the mean, it does not affect the precision of diff itself
                if not torch.is_floating_point(diff):
                    mean_err = diff.float().mean().item()
                else:
                    mean_err = diff.mean().item()

                # For integral outputs (e.g. argmax), use torch.equal to check for exact equality
                # For floating-point outputs, use torch.allclose to check they are within tolerance
                if not torch.is_floating_point(ref_out) and not torch.is_floating_point(test_out):
                    # Integral outputs: must be exactly equal
                    if not torch.equal(ref_out, test_out):
                        raise ValueError(_build_mismatch_debug_msg(
                            reason="Integer outputs are not equal",
                            tol=tol,
                            max_err=max_err,
                            mean_err=mean_err,
                            ref_out=ref_out,
                            test_out=test_out,
                            diff=diff,
                            inputs=inp,
                            ref_model=ref_model,
                        ))
                else:
                    # Floating-point outputs: check with allclose
                    close = torch.allclose(ref_out, test_out, atol=tol, rtol=tol)

                    # Regardless of the verdict, establish whether the result is
                    # even reproducible. A kernel that reads uninitialized memory
                    # answers differently depending on allocation history, and
                    # saying so is far more actionable than "outputs are not
                    # close", which sends the repair round after the arithmetic.
                    alloc_note = _allocator_dependence_note(
                        test_model, inp, dev, test_out, tol, _shape_tag(inp))

                    if not close:
                        msg = _build_mismatch_debug_msg(
                            reason="Outputs are not close",
                            tol=tol,
                            max_err=max_err,
                            mean_err=mean_err,
                            ref_out=ref_out,
                            test_out=test_out,
                            diff=diff,
                            inputs=inp,
                            ref_model=ref_model,
                        )
                        if alloc_note:
                            msg = f"{msg}\n\n{alloc_note}"
                        raise ValueError(msg)

                    # Matched the reference, but only by luck of allocation.
                    if alloc_note:
                        raise ValueError(alloc_note)

            # Timing. The candidate has already executed once (correctness check
            # above), so any state it leaked is in effect right now -- restore
            # before measuring the reference, never after.
            # Record the leak BEFORE restoring -- restoring erases the evidence,
            # so an end-of-round diff would always come back clean.
            _leaks = device_state.diff(_clean_state, device_state.snapshot())
            device_state.restore(_clean_state)
            # The restore above must precede BOTH timings, so skipping the
            # reference cannot let the candidate's leaked state reach its own
            # measurement -- the ordering, not the reference run, is what guards
            # that. See time_ref in the docstring.
            ref_t  = _bench(ref_model, inp, dev, warmup, repeat) if time_ref else []
            test_t = _bench(test_model, inp, dev, warmup, repeat)

            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            # ---- optional multi-shape scoring --------------------------------
            # A task may expose get_inputs_extra() -> list of ADDITIONAL input
            # tuples beyond get_inputs(). The get_inputs() shape stays the sole
            # profiling/NCU target; these only widen the score so that a kernel
            # specialised on one shape cannot win on that shape alone. Tasks
            # without the hook behave exactly as before.
            def _avg(ts):
                return sum(ts) / len(ts)

            per_shape = [{
                "shape": _shape_tag(inp),
                "ref_ms": _avg(ref_t) if time_ref else None,
                "test_ms": _avg(test_t),
                "speedup": (_avg(ref_t) / _avg(test_t)) if time_ref else None,
                "max_abs_err": max_err,
                "primary": True,
            }]

            get_inputs_extra = getattr(ref_mod, "get_inputs_extra", None)
            if get_inputs_extra is not None:
                for extra in get_inputs_extra():
                    if not isinstance(extra, (list, tuple)):
                        extra = [extra]
                    e_ref, _ = _run_once(ref_model, extra, dev)
                    _fresh_allocator(dev)   # see the primary run above
                    e_tst, _ = _run_once(test_model, extra, dev)
                    if TORCH_DEVICE == "cuda":
                        torch.cuda.synchronize(dev)
                    e_ref = _first_tensor(e_ref).contiguous()
                    e_tst = _first_tensor(e_tst).contiguous().to(e_ref.dtype)
                    e_diff = (e_ref - e_tst).abs()
                    e_err = e_diff.max().item()
                    e_mean = e_diff.mean().item()
                    e_note = _allocator_dependence_note(
                        test_model, extra, dev, e_tst, tol, _shape_tag(extra))
                    if not torch.allclose(e_ref, e_tst, atol=tol, rtol=tol):
                        # Same diagnostic depth as the primary shape. This path used
                        # to report only max_abs_err, which is the least actionable
                        # number available and left the repair round guessing.
                        msg = _build_mismatch_debug_msg(
                            reason=f"Outputs are not close on extra shape {_shape_tag(extra)}",
                            tol=tol,
                            max_err=e_err,
                            mean_err=e_mean,
                            ref_out=e_ref,
                            test_out=e_tst,
                            diff=e_diff,
                            inputs=extra,
                            ref_model=ref_model,
                        )
                        if e_note:
                            msg = f"{msg}\n\n{e_note}"
                        raise ValueError(msg)
                    if e_note:
                        raise ValueError(e_note)
                    device_state.restore(_clean_state)   # same rule per shape
                    e_rt = _bench(ref_model, extra, dev, warmup, repeat) if time_ref else []
                    e_tt = _bench(test_model, extra, dev, warmup, repeat)
                    per_shape.append({
                        "shape": _shape_tag(extra),
                        "ref_ms": _avg(e_rt) if time_ref else None,
                        "test_ms": _avg(e_tt),
                        "speedup": (_avg(e_rt) / _avg(e_tt)) if time_ref else None,
                        "max_abs_err": e_err,
                        "primary": False,
                    })
                    del e_ref, e_tst
                    if TORCH_DEVICE == "cuda":
                        torch.cuda.synchronize(dev)
                        # Deliberately NOT torch.cuda.empty_cache(): returning
                        # blocks to the driver means the next shape is handed
                        # freshly-mapped (zeroed) pages, which is exactly what
                        # hides a kernel that reads uninitialized memory. Keeping
                        # the blocks in PyTorch's pool preserves whatever the
                        # previous shape left there, and _allocator_dependence_note
                        # above relies on being able to poison that pool.

            # Geometric mean over shapes: each shape weighs equally, so a large
            # shape cannot dominate the score simply by taking longer.
            # Undefined without a reference timing -- None, never a stand-in, so a
            # caller that wanted a score and forgot time_ref fails loudly here
            # rather than ranking on a fabricated number.
            if time_ref:
                _sp = [s["speedup"] for s in per_shape]
                score = math.exp(sum(math.log(v) for v in _sp) / len(_sp))
            else:
                score = None

            # The scores above are already honest -- every reference was timed
            # from _clean_state. But a candidate that leaks device state is still
            # not a valid solution: it degrades every other tenant of the process.
            # _leaks was captured above, at the moment the leak was still visible.
            _leaks += [c for c in device_state.diff(_clean_state, device_state.snapshot())
                       if c not in _leaks]
            if _leaks:
                device_state.restore(_clean_state)
                print(f"[device_state] candidate leaked: {'; '.join(_leaks)}", flush=True)
                if reject_on_state_leak and device_state.is_fatal(_leaks):
                    raise ValueError(device_state.leak_message(_leaks))

    except Exception:
        # Raise the full traceback (caught by the caller)
        import traceback as _tb
        raise RuntimeError(_tb.format_exc()) from None
    finally:
        # Leave the process exactly as it was found, on EVERY exit path. Without
        # this an accuracy failure (which throws before the leak check) would let
        # one candidate's device state reach the next kernel measured in this
        # process -- which is precisely how paired_bench, below, could be fooled.
        device_state.restore(_clean_state)

    # ------------ Result summary ------------
    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_file": str(ref_py),
        "candidate_file": str(test_py),
        "tolerance": tol,
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "ref_latency_ms": {
            "avg": sum(ref_t) / len(ref_t),
            "min": min(ref_t),
            "max": max(ref_t),
            "all": ref_t,
        } if ref_t else None,
        "test_latency_ms": {
            "avg": sum(test_t) / len(test_t),
            "min": min(test_t),
            "max": max(test_t),
            "all": test_t,
        },
        "num_runs": repeat,
        "state_leak": _leaks,
        # The frequency these times were taken at, recorded WITH them. Latencies
        # from two different clocks are not comparable, and REPORT_002:762 traces
        # the "a trace cannot be audited on its own" problem to exactly this
        # field being absent from every artifact this repo had written.
        "clock": clock_lock.state(),
        # Geometric-mean speedup across every benchmarked shape. Equals the
        # single-shape ref/test ratio when the task has no get_inputs_extra().
        "score": score,
        "per_shape": per_shape,
        "model_init_args": init_args,
        "model_init_kwargs": init_kwargs,
        "seed": seed,
        "align_stats": align_stats,  # Alignment info (including whether the Model→ModelNew dedicated aligner was hit)
    }
    return result





# =========================== CLI wrapper ==================================
def _cli():
    p = argparse.ArgumentParser(description="Compare & bench two model files.")
    p.add_argument("reference", type=Path, help="Path to reference .py")
    p.add_argument("candidate", type=Path, help="Path to candidate .py")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Benchmark runs")
    p.add_argument("--tol", type=float, default=1e-4, help="Max abs error tolerance")
    p.add_argument("--dump", type=Path, help="If set, write JSON results here")
    args = p.parse_args()

    res = compare_and_bench(
        args.reference,
        args.candidate,
        device_idx=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
    )
    print(json.dumps(res, indent=2))

    if args.dump:
        args.dump.write_text(json.dumps(res, indent=2))
        print(f"\nSaved ⇒ {args.dump}")


if __name__ == "__main__":
    _cli()

