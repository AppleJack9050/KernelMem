"""Append-only duration log for a run (``timing.csv``, beside ``usage.csv``).

Nothing in this harness used to record how long anything took. ``run_ncu_memory``
printed "Completed for kernel i/N" with no elapsed time, ``lineage.log`` carried no
timestamps, and manual ``--resume`` invocations wrote their stdout to a terminal that
was never captured. So the only way to recover a duration was to subtract two artifact
mtimes -- and that subtraction silently includes any period when the process was not
running.

That bit us concretely. On 2026-08-06 the own_gemm lineage was stopped gracefully after
round 7 (08:20:59, the post-loop writer's summary/figures prove a clean exit) and
resumed by hand 51 minutes later (09:12:11, the mtime of the ``ref_0.py`` written at
process startup). The next ncu CSV landed at 09:13:10. Subtracting mtimes charged the
whole 52.5 minutes to the ncu pass, and it read as a profiler stall that had to be
explained. Re-running that exact profile later took 1.05 minutes. There was never a
stall; there was a stopped run, and no record said so.

Two properties make the rows here trustworthy where an mtime delta is not:

* durations are measured with ``time.perf_counter()`` around the work itself, so a
  process that is not running cannot contribute to one; and
* ``process_start`` / ``process_exit`` rows bound each session, so a reader can tell a
  gap between rows apart from work -- the distinction that was unrecoverable before.

Rows are appended and flushed the moment a phase ends, so a hard kill keeps everything
already written. Every function here swallows its own errors: timing is diagnostics and
must never be the reason a round dies.

Columns: ``timestamp,round_idx,phase,seconds,detail``
  round_idx  -1 when the row is not attributable to a round (startup, exit).
  phase      'llm:optimization', 'ncu:rejected', 'bench:opt', 'round_total',
             'process_start', 'process_exit', 'stop_signal', ...
  seconds    wall-clock for the phase; 0.0 for instantaneous lifecycle events.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

__all__ = ["set_timing_log", "timing_log_path", "set_round", "record", "event",
           "phase_timer"]

_LOG_PATH: Optional[Path] = None
_ROUND_IDX: int = -1
_HEADER = "timestamp,round_idx,phase,seconds,detail\n"


def set_timing_log(path: Path) -> None:
    """Point the module at *path*, creating it with a header if absent.

    Called once per run, right after the task's output directory is known. Safe to
    call again; an existing file is appended to, never truncated, so a ``--resume``
    keeps the previous session's rows.
    """
    global _LOG_PATH
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(_HEADER)
        _LOG_PATH = path
    except Exception as exc:  # never let diagnostics break a run
        print(f"[timing] disabled ({exc.__class__.__name__}: {exc})", flush=True)
        _LOG_PATH = None


def timing_log_path() -> Optional[Path]:
    return _LOG_PATH


def set_round(round_idx: int) -> None:
    """Set the round every subsequent row is attributed to.

    Called once at the top of each round so that callers deep in the stack --
    ``run_ncu_memory.profile_bench`` in particular, which has no idea what a round is
    -- do not have to thread a round index through their signatures.
    """
    global _ROUND_IDX
    try:
        _ROUND_IDX = int(round_idx)
    except Exception:
        _ROUND_IDX = -1


def _clean(text: str) -> str:
    """Flatten a detail string into one CSV-safe field (no quoting needed)."""
    return str(text).replace(",", ";").replace("\n", " ").replace("\r", " ").strip()[:200]


def record(phase: str, seconds: float, *, round_idx: Optional[int] = None,
           detail: str = "") -> None:
    """Append one duration row and flush it. ``round_idx`` defaults to the current round."""
    if _LOG_PATH is None:
        return
    try:
        ridx = _ROUND_IDX if round_idx is None else int(round_idx)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{stamp},{ridx},{_clean(phase)},{float(seconds):.3f},"
                    f"{_clean(detail)}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass  # diagnostics only


def event(kind: str, *, round_idx: Optional[int] = None, detail: str = "") -> None:
    """Append a zero-duration lifecycle row (process_start, stop_signal, ...)."""
    record(kind, 0.0, round_idx=round_idx, detail=detail)


@contextmanager
def phase_timer(phase: str, *, round_idx: Optional[int] = None, detail: str = ""):
    """Time a block and record it, whether or not the block raises.

    A phase that raises is still recorded, with ``failed:`` prefixed to its detail --
    an ncu pass that died after 600 s cost those 600 s and the log should say so.
    """
    t0 = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        dt = time.perf_counter() - t0
        record(phase, dt, round_idx=round_idx,
               detail=(f"failed:{detail}" if failed else detail))
