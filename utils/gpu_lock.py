"""A cross-process mutex around every GPU section, so lineages can run in parallel.

Why this exists
---------------
Parallel lineages are worth having because the wall clock is dominated by LLM
generation, not by the GPU: on the 2026-08-04 granularity-D run a seed draw took
16-20 minutes of generation against roughly a minute of benchmarking. Running
lineages concurrently therefore converts almost linearly into wall-clock saving.

But the GPU part must NOT overlap, and that is not a tuning preference -- it is
the whole basis on which this project's numbers mean anything. ``compare_and_bench``
computes ``speedup = T_ref / T_k`` and the ratchet resolves differences of 0.5%.
A second process running a kernel on the same device during that measurement
perturbs clocks, L2 residency and SM occupancy by far more than 0.5%, so
concurrent benchmarking does not make the numbers noisier in a way more
repetitions would fix -- it makes them mean nothing. The same applies with more
force to ``ncu``, which serializes and replays kernels, and to ``nsys``.

So: generate in parallel, measure one at a time.

Opt-in by construction
----------------------
If ``KERNELMEM_GPU_LOCK`` is unset -- which is the case for every existing
single-process invocation -- ``gpu_section()`` is a no-op context manager and
costs one env lookup. Nothing about the default path changes. The coordinator
sets the variable for its children, and only then does locking engage.

Uses ``fcntl.flock``, so the lock is released by the kernel if a holder is
killed (including ``kill -9``), and a crashed lineage cannot wedge the others.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Iterator, Optional

_ENV_VAR = "KERNELMEM_GPU_LOCK"
_ENV_LABEL = "KERNELMEM_LINEAGE_LABEL"

# Held across a whole benchmark or an ncu sweep, and ncu on a 5-kernel candidate
# takes minutes, so this has to be generous. It is a deadlock backstop, not a
# scheduling parameter: expiring it means proceeding UNSERIALIZED, which
# silently corrupts a measurement, so it must only fire when something is truly
# wedged.
_DEFAULT_TIMEOUT_S = 3600.0


def lock_path() -> Optional[Path]:
    """The configured lock file, or None when locking is disabled."""
    raw = os.environ.get(_ENV_VAR, "").strip()
    return Path(raw) if raw else None


def enabled() -> bool:
    return lock_path() is not None


def _label() -> str:
    return os.environ.get(_ENV_LABEL, "") or f"pid{os.getpid()}"


@contextlib.contextmanager
def gpu_section(what: str = "gpu", verbose: bool = True) -> Iterator[bool]:
    """Serialize a GPU-touching block against every other lineage.

    Yields True when the lock was actually held, False when locking is disabled
    (the single-process default) or unavailable. Never raises on lock failure --
    a benchmark that runs unserialized is bad, but a run that dies because it
    could not take a lock is worse, so failures degrade to a warning.
    """
    path = lock_path()
    if path is None:
        yield False
        return

    try:
        import fcntl
    except ImportError:  # non-POSIX; degrade rather than break
        yield False
        return

    fh = None
    t0 = time.monotonic()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            waited = 0.0
        except BlockingIOError:
            if verbose:
                print(f"[gpu_lock] {_label()}: waiting for the GPU to free up ({what}) ...",
                      flush=True)
            deadline = t0 + _DEFAULT_TIMEOUT_S
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        print(f"[gpu_lock] WARNING: {_label()} waited "
                              f"{_DEFAULT_TIMEOUT_S:.0f}s for '{what}' and gave up. "
                              f"Proceeding WITHOUT the lock -- timings from this "
                              f"measurement may overlap another lineage and should "
                              f"not be trusted.", flush=True)
                        yield False
                        return
                    time.sleep(0.25)
            waited = time.monotonic() - t0
            if verbose:
                print(f"[gpu_lock] {_label()}: acquired after {waited:.1f}s ({what})",
                      flush=True)
    except Exception as exc:
        print(f"[gpu_lock] WARNING: could not acquire the GPU lock ({exc}); "
              f"proceeding unserialized.", flush=True)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        yield False
        return

    try:
        yield True
    finally:
        try:
            import fcntl as _f
            _f.flock(fh.fileno(), _f.LOCK_UN)
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
