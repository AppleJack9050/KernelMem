"""Snapshot / restore / diff of the device-global state a candidate can mutate.

Why this exists
---------------
``compare_and_bench`` times the reference and the candidate in the SAME process,
and computes ``speedup = T_ref / T_k``. That is only meaningful if ``T_ref`` is a
constant of (task, shape, hardware). It is not: a candidate can change
device-global state that outlives its own execution, and the reference is timed
afterwards in whatever state the candidate left behind.

This was not hypothetical. On 2026-07-31, vae_block_002 round 4 shipped

    static const size_t max_win = [](){
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           p->persistingL2CacheMaxSize);   // never reset
        ...
    }();

inside a static initializer. It raises the persisting-L2 reservation from the
default 9.375 MB to 31.25 MB of the H100's 50 MB L2, so every *streaming* access
in the process -- i.e. the whole eager reference -- is squeezed into 18.75 MB.
Measured effect on T_ref alone, with no kernel loaded and only this one limit
flipped: 1.6703 -> 2.7409 ms (1.64x) on 8x256x64x128, 6.5787 -> 10.9244 ms
(1.66x) on 4x256x256x256, returning to 1.7067 / 6.6002 ms when released.

The loop therefore recorded a score of 1.8972 for a kernel that was, in absolute
time, 1.28x SLOWER than the round-3 kernel scoring 1.6168. Reported score rose
while the kernel regressed, and the ratchet locked onto the polluted branch.

The rule this module enforces: the reference is always timed in the state that
was in effect BEFORE the candidate was ever imported or run.

Note that ``restore`` is the load-bearing half. A detector alone is not enough
here: the candidate's first execution happens during the correctness check,
which runs *before* any timing, so by the time T_ref is measured the damage is
already done and a before/after-timing comparison sees nothing wrong.
"""
from __future__ import annotations

import ctypes
import glob
import os
from typing import Any, Dict, List, Optional

import torch

# cudaLimit / cudaDeviceAttr values (driver ABI, stable across CUDA versions)
_CUDA_LIMIT_PERSISTING_L2 = 0x06

_cudart: Optional[ctypes.CDLL] = None
_cudart_tried = False


def _libcudart() -> Optional[ctypes.CDLL]:
    """The libcudart torch itself is linked against, or None if unavailable.

    Best-effort by design: on a CPU-only box, or if the symbol layout ever
    changes, we degrade to covering only the torch-level knobs rather than
    breaking every benchmark.
    """
    global _cudart, _cudart_tried
    if _cudart_tried:
        return _cudart
    _cudart_tried = True
    if not torch.cuda.is_available():
        return None
    try:
        cand = glob.glob(os.path.join(os.path.dirname(torch.__file__),
                                      "lib", "libcudart*.so*"))
        _cudart = ctypes.CDLL(cand[0] if cand else "libcudart.so")
    except OSError:
        _cudart = None
    return _cudart


def _get_persisting_l2() -> Optional[int]:
    lib = _libcudart()
    if lib is None:
        return None
    try:
        v = ctypes.c_size_t(0)
        if lib.cudaDeviceGetLimit(ctypes.byref(v), _CUDA_LIMIT_PERSISTING_L2) != 0:
            return None
        return v.value
    except Exception:
        return None


def _set_persisting_l2(n: int) -> bool:
    lib = _libcudart()
    if lib is None:
        return False
    try:
        return lib.cudaDeviceSetLimit(_CUDA_LIMIT_PERSISTING_L2,
                                      ctypes.c_size_t(n)) == 0
    except Exception:
        return False


# Each entry: key -> (getter, setter). Anything a candidate can flip that
# changes how UNRELATED work on the same device performs belongs here.
_KNOBS = {
    "cudnn.benchmark": (
        lambda: torch.backends.cudnn.benchmark,
        lambda v: setattr(torch.backends.cudnn, "benchmark", v),
    ),
    "cudnn.deterministic": (
        lambda: torch.backends.cudnn.deterministic,
        lambda v: setattr(torch.backends.cudnn, "deterministic", v),
    ),
    "cudnn.allow_tf32": (
        lambda: torch.backends.cudnn.allow_tf32,
        lambda v: setattr(torch.backends.cudnn, "allow_tf32", v),
    ),
    "matmul.allow_tf32": (
        lambda: torch.backends.cuda.matmul.allow_tf32,
        lambda v: setattr(torch.backends.cuda.matmul, "allow_tf32", v),
    ),
    "float32_matmul_precision": (
        torch.get_float32_matmul_precision,
        torch.set_float32_matmul_precision,
    ),
    "deterministic_algorithms": (
        torch.are_deterministic_algorithms_enabled,
        lambda v: torch.use_deterministic_algorithms(v, warn_only=True),
    ),
    "persisting_l2_bytes": (_get_persisting_l2, _set_persisting_l2),
}

# Knobs whose leak is EXPLOITABLE, i.e. can make the reference SLOWER and so
# inflate T_ref/T_k. Only these fail the candidate; everything else warns.
#
# The distinction matters and was learned the hard way. `cudnn.benchmark` also
# leaks -- every kernel here that calls at::cudnn_convolution with
# /*benchmark=*/true flips the global context flag as a side effect, with no
# Python assignment anywhere in the generated source. But that leak makes cuDNN
# autotune, which makes the REFERENCE faster and understates the candidate. It
# is untidy, not exploitable. Marking it fatal rejected round 3 -- the best
# kernel in the 2026-07-31 run -- on the very first test.
#
# `persisting_l2_bytes` is the exploitable one: it shrinks the L2 available to
# every other tenant of the process, which is exactly how a 28%-slower kernel
# came to score 1.8972 against a 1.6168 rival.
_FATAL = ("persisting_l2_bytes",)


def snapshot() -> Dict[str, Any]:
    """Current value of every tracked knob. Never raises."""
    snap: Dict[str, Any] = {}
    for key, (get, _set) in _KNOBS.items():
        try:
            snap[key] = get()
        except Exception:
            snap[key] = None
    return snap


def restore(snap: Dict[str, Any]) -> List[str]:
    """Force every tracked knob back to *snap*. Returns the keys it had to change."""
    changed: List[str] = []
    for key, want in (snap or {}).items():
        if want is None or key not in _KNOBS:
            continue
        get, setter = _KNOBS[key]
        try:
            if get() != want:
                setter(want)
                changed.append(key)
        except Exception:
            continue
    return changed


def diff(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    """Human-readable 'key: before -> after' for every knob that moved."""
    out: List[str] = []
    for key in (before or {}):
        b, a = before.get(key), after.get(key)
        if b is None or a is None or b == a:
            continue
        out.append(f"{key}: {b} -> {a}")
    return out


def is_fatal(changes: List[str]) -> bool:
    return any(c.split(":", 1)[0] in _FATAL for c in changes)


def leak_message(changes: List[str]) -> str:
    """Actionable rejection text; this reaches the repair prompt via the error log."""
    return (
        "Candidate leaked device-global state.\n"
        "Changed after the candidate ran:\n  "
        + "\n  ".join(changes)
        + "\n\nThis state outlives the candidate's own execution and changes how "
          "UNRELATED work on the device performs, including the reference the "
          "candidate is scored against. A kernel that raises the persisting-L2 "
          "reservation and never releases it does not get faster -- it makes "
          "everything else slower, which inflates T_ref/T_k without improving "
          "the kernel.\n"
          "Fix: set such state and restore it INSIDE forward (save the previous "
          "value, restore before returning; for persisting L2 also call "
          "cudaCtxResetPersistingL2Cache()). Do not configure it from a static "
          "initializer or module scope, which runs once and never unwinds."
    )
