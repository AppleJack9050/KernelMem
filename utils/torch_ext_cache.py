"""Clear orphaned torch extension build locks before a run starts.

Why this exists
---------------
torch's JIT extension loader guards each build directory with a ``FileBaton``
(``torch/utils/file_baton.py``): the winner creates ``<build_dir>/lock`` with
``O_CREAT|O_EXCL`` and every loser spins in ``wait()`` until it disappears.
That ``wait()`` has no timeout, no staleness check and no holder-liveness check,
``cpp_extension`` constructs the baton with ``warn_after_seconds=None`` so the
spin is silent, and ``release()`` runs only in the winner's ``finally``.

Any path that skips stack unwinding therefore orphans the lock permanently:

* ``main_memory_latest._bench_and_score`` -- ``p.terminate()`` / ``p.kill()``
  after the 20-minute ``p.join`` budget expires (SIGTERM has no handler in a
  multiprocessing child, so no ``finally`` runs),
* ``run_lineages`` -- SIGTERM then SIGKILL when a lineage is cut.

The 600s SIGALRM in ``utils.compile_and_run._capture_import`` is NOT one of
them, which is worth stating because it looks like the obvious candidate: it
raises ``CompilationTimeoutError``, an ordinary Python exception, which unwinds
straight through ``cpp_extension``'s ``finally: baton.release()`` and removes
the lock. Verified by reproduction -- a build alarmed mid-ninja leaves
``['build.ninja', 'main.cpp']`` and no ``lock``, while the same build killed
with SIGTERM or SIGKILL leaves the lock behind. That alarm still matters here,
just in the other direction: it is what BOUNDS a live holder's lifetime, which
is what makes the age threshold below safe.

The damage is silent and lasts forever. Build directories are keyed on the
extension NAME the model chose, and independently generated kernels reuse names
routinely (``utils/verify_chain.py`` documents two rounds both emitting
``vae_resblock_fused_ms_graph_ext``). Once a name is poisoned, any later kernel
picking it hangs until the 600s compile alarm fires and is then reported through
main_memory_latest's timeout diagnostic as an *"illegal memory access ...
infinite loops or deadlocks in the kernel code"*. A perfectly good kernel is
scored -inf and sent to repair for a bug that does not exist.

Found in the wild on 2026-08-06: ``cudnn_conv3x3_nhwc_v7_a/`` held a 0-byte
``lock`` from 2026-07-31 17:23 with no ``.so`` and no ``.o``, while ``cuda.cu``
carried a 2026-08-02 mtime -- a later process had written its sources into the
shared directory, spun on the dead baton and built nothing. That name appears in
three generated kernels of the 2026-07-31 run.

Note the irony this fixes: ``utils/gpu_lock`` deliberately uses ``fcntl.flock``
precisely so "a crashed lineage cannot wedge the others", and the project then
sat on a non-crash-safe lock covering the same processes on the same path.

Two gates, because deleting a live lock is worse than the bug
-------------------------------------------------------------
Unlinking a lock a build still holds is not a smaller version of the problem,
it is a worse one: ``FileBaton.release()`` has no ``try``, so the holder dies
with ``FileNotFoundError`` out of ``load_inline``'s ``finally``, and a second
process can ``try_acquire()`` the same directory immediately -- a hang traded
for silent corruption. So a lock must clear BOTH gates before it is removed.

1. **Age.** A baton's mtime is set once at ``O_CREAT`` and never refreshed, so
   age is how long the build has been running. No legitimate holder in this
   project outlives its own compile budget (``_capture_import`` alarms at 600s,
   ``_bench_and_score`` joins for 1200s), so the 3600s default cannot catch one
   even on a loaded box running several lineages. ``KERNELMEM_BATON_MAX_AGE``
   tunes it and 0 disables the sweep, but a positive value is clamped up to
   ``_MIN_MAX_AGE_S`` -- without that floor, setting it below the compile budget
   would destroy perfectly normal builds, which is exactly the failure this
   module exists to prevent.

2. **Liveness.** Age alone would be enough if this repo owned the cache, but
   ``~/.cache/torch_extensions`` is user-global and shared with every other
   torch process on the box -- a notebook, another checkout, a training script
   -- none of which are bounded by the budgets above. So we also ask ``/proc``
   what is still in use, once per sweep.

   The obvious form of that question is wrong, and it is worth recording why.
   "Does any process hold the lock file open?" tracks only the Python holder:
   the baton fd is ``O_CLOEXEC``, and the compile itself is
   ``subprocess.run(['ninja'], cwd=build_directory)``, which has no PDEATHSIG
   and is in no separate process group. Kill the Python worker -- which is
   precisely what this harness does when a compile blows its join budget -- and
   the fd dies while ninja and nvcc carry on writing into that directory. The
   lock reads as unheld at the exact moment the build is most alive.

   What stays true is that those orphans have the build directory as their cwd.
   So the liveness set is open fds AND cwds, and a lock is spared if either it
   or its parent directory appears in it. Where ``/proc`` is unavailable the
   startup sweep falls back to age alone and says so; the per-round sweep, whose
   floor is far too low to stand on its own, removes nothing.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

_ENV_MAX_AGE = "KERNELMEM_BATON_MAX_AGE"
_DEFAULT_MAX_AGE_S = 3600.0
# Floor for any positive KERNELMEM_BATON_MAX_AGE. Sits above the largest compile
# budget in the repo (the 1200s p.join in _bench_and_score) with room to spare,
# so no tuning of this knob can make the sweep delete a lock a normal build is
# still holding. 0 is honoured as-is -- that means "disabled", not "impatient".
_MIN_MAX_AGE_S = 1800.0
# Age floor for the liveness-gated per-round sweep. Note what this does and does
# NOT buy, because the obvious reading is wrong: age is measured from the lock's
# mtime, which FileBaton stamps once at O_CREAT, so it is ELAPSED BUILD TIME, not
# time since the holder let go. It therefore gives no margin at all on the
# close->remove window inside release() -- any build longer than a minute has
# already cleared this floor by the time it gets there. That window is closed by
# _CONFIRM_DELAY_S below instead. What the floor does buy is the acquisition
# side: a lock created after the one-shot /proc snapshot was taken has age ~0 and
# must not be judged by it.
_RACE_FLOOR_S = 60.0
# A lock that looks unheld is observed twice, this far apart, before it is
# removed. A holder inside FileBaton.release() completes os.remove within
# microseconds of os.close, so it cannot still be there a second later; and a
# freshly-orphaned compile gets a second chance to show a live descendant.
_CONFIRM_DELAY_S = 1.0


def extensions_root() -> Path:
    """The directory torch builds JIT extensions into, honouring the env var."""
    override = os.environ.get("TORCH_EXTENSIONS_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "torch_extensions"


def _clamp_max_age(value: float) -> float:
    """0 (or less) disables the sweep; any positive value is floored."""
    if value <= 0:
        return 0.0
    return max(value, _MIN_MAX_AGE_S)


def _max_age() -> float:
    raw = os.environ.get(_ENV_MAX_AGE, "").strip()
    if not raw:
        return _DEFAULT_MAX_AGE_S
    try:
        return _clamp_max_age(float(raw))
    except ValueError:
        return _DEFAULT_MAX_AGE_S


def _open_inodes(proc_root: Path = Path("/proc")) -> Optional[Set[Tuple[int, int]]]:
    """(st_dev, st_ino) of everything a live process is using: open fds AND cwds.

    The cwd half is not incidental, it is the load-bearing half. An earlier
    version of this module scanned only open descriptors and concluded that an
    unheld lock meant a dead builder. That is false, and provably so:

    * ``FileBaton.try_acquire`` opens the lock with ``os.open``, which under
      PEP 446 sets ``O_CLOEXEC``. The fd is therefore NOT inherited across
      ``exec``, so ninja and nvcc never hold it -- only the Python process does.
    * ``cpp_extension._run_ninja_build`` is ``subprocess.run(['ninja', ...],
      cwd=build_directory)``. ninja is in no separate process group and has no
      PDEATHSIG, so ``p.terminate()`` / ``p.kill()`` of the Python worker (which
      is exactly what main_memory_latest does when a compile blows its join
      budget) kills the fd holder and leaves ninja and its nvcc children alive,
      reparented to init, still writing into that directory.

    So immediately after the kill the lock is "unheld" while the build is
    genuinely still running. Removing it there would admit a second builder into
    a directory an orphan is mid-write in -- clobbered objects, a half-linked
    ``.so``, or an ``undefined symbol`` blamed on the kernel under test. That is
    the "hang traded for silent corruption" this module's header calls strictly
    worse than the bug it fixes.

    The orphans do still declare themselves: their cwd IS the build directory
    (verified by reproduction -- after SIGKILL of the Python parent, a surviving
    descendant's ``/proc/<pid>/cwd`` still resolves to it). Collecting cwds makes
    them visible, and the caller checks the lock's PARENT directory against this
    set as well as the lock file itself.

    ``None`` means "cannot tell" -- no ``/proc``. Processes owned by other users
    are unreadable and simply do not contribute; acceptable because the cache
    being swept lives under this user's ``$HOME``.

    *proc_root* is a parameter only so the no-``/proc`` branch is reachable from
    a test on a host that has one.
    """
    proc = Path(proc_root)
    if not proc.is_dir():
        return None
    live: Set[Tuple[int, int]] = set()
    try:
        for pid_dir in proc.iterdir():
            if not pid_dir.name.isdigit():
                continue
            try:
                st = (pid_dir / "cwd").stat()   # the build dir of an orphaned ninja
                live.add((st.st_dev, st.st_ino))
            except OSError:
                pass                            # exited, or another user's
            try:
                for fd in (pid_dir / "fd").iterdir():
                    try:
                        st = fd.stat()          # follows the fd symlink to the file
                    except OSError:
                        continue                # fd closed under us, or not ours
                    live.add((st.st_dev, st.st_ino))
            except OSError:
                continue
    except OSError:
        return None
    return live


def _in_use(lock: Path, lock_stat: os.stat_result,
            live: Set[Tuple[int, int]]) -> Optional[str]:
    """Why *lock* must not be removed, or None if nothing is using it."""
    if (lock_stat.st_dev, lock_stat.st_ino) in live:
        return "a live process still holds it open"
    try:
        parent = lock.parent.stat()
    except OSError:
        return "its build directory could not be checked"
    if (parent.st_dev, parent.st_ino) in live:
        return "a live process is still building in that directory"
    return None


def _sweep(root: Path, *, min_age_s: float, liveness_required: bool,
           verbose: bool) -> List[Path]:
    """Shared walk. Removes a lock only if it is at least *min_age_s* old and
    nothing live is using it -- neither an open fd on the lock nor a process
    whose cwd is the build directory (see ``_open_inodes`` for why the second
    check is the one that matters).

    *liveness_required* decides what happens when ``/proc`` cannot be read.
    False (the startup sweep) falls back to the age gate, which is safe because
    that gate is 3600s. True (the per-round sweep) removes nothing, because its
    age floor is 60s and would be meaningless without evidence of death. It also
    turns on the confirm pass: every candidate is observed twice, so a holder
    caught mid-``release()`` is not mistaken for a dead one.
    """
    candidates: List[Tuple[Path, os.stat_result, float]] = []
    live: Optional[Set[Tuple[int, int]]] = None
    probed = False
    try:
        if not root.is_dir():
            return []
        now = time.time()
        for lock in root.glob("**/lock"):
            try:
                if not lock.is_file():
                    continue
                stat = lock.stat()
                age = now - stat.st_mtime
                if age < min_age_s:
                    continue
                # Probed lazily and once: /proc is only walked if something
                # actually cleared the age gate, and never per-lock. In a
                # healthy cache there are no locks at all, so it never runs.
                if not probed:
                    live, probed = _open_inodes(), True
                    if live is None and verbose:
                        consequence = ("no lock can be shown dead, so none will be removed"
                                       if liveness_required else
                                       "stale locks are identified by age alone")
                        print(f"[baton] NOTE: /proc is unavailable, so {consequence}.",
                              flush=True)
                # Deliberately redundant with the same check in the confirm pass
                # below. Removing either one changes no behaviour, which is the
                # point: this is a destructive operation, and "removes nothing
                # without evidence of death" should not rest on a single line.
                if live is None and liveness_required:
                    continue
                reason = _in_use(lock, stat, live) if live is not None else None
                if reason:
                    if verbose:
                        print(f"[baton] Keeping {lock.parent.name}: {age / 60:.0f} min old "
                              f"but {reason}.", flush=True)
                    continue
                candidates.append((lock, stat, age))
            except FileNotFoundError:
                # Vanished between is_file() and stat(). A microsecond window
                # that no test can force, unlike the unlink race below.
                continue
            except OSError as exc:
                if verbose:
                    print(f"[baton] WARNING: could not examine {lock}: {exc}", flush=True)
    except OSError as exc:
        if verbose:
            print(f"[baton] WARNING: could not scan {root}: {exc}", flush=True)
        return []

    if not candidates:
        return []

    # Second observation. Age cannot cover the close->remove window inside
    # release() -- it is measured from O_CREAT, so a long build clears any floor
    # long before it gets there -- and a compile orphaned moments ago may not
    # have had its descendants scheduled yet. Looking twice covers both: a
    # holder mid-release() finishes os.remove in microseconds, so by now the
    # path is gone, and a live orphan has had a second to show its cwd.
    if liveness_required:
        time.sleep(_CONFIRM_DELAY_S)
        live = _open_inodes()
        if live is None:
            return []

    removed: List[Path] = []
    for lock, first, age in candidates:
        try:
            stat = lock.stat()
            # A cheap first filter, not the guarantee: ext4 hands the same inode
            # straight back on unlink+create, so a lock re-acquired between the
            # two observations can present the identical (dev, ino). What
            # actually protects that case is the liveness re-check below -- a
            # real new builder is holding the fd it just opened.
            if (stat.st_dev, stat.st_ino) != (first.st_dev, first.st_ino):
                continue
            if liveness_required and live is not None:
                reason = _in_use(lock, stat, live)
                if reason:
                    if verbose:
                        print(f"[baton] Keeping {lock.parent.name} after a second look: "
                              f"{reason}.", flush=True)
                    continue
            built = any(lock.parent.glob("*.so"))
            lock.unlink()
            removed.append(lock)
            if verbose:
                print(f"[baton] Removed an orphaned build lock ({age / 60:.0f} min old, "
                      f"{'a .so is present' if built else 'NO .so was ever built'}): "
                      f"{lock.parent.name}", flush=True)
        except FileNotFoundError:
            # Released between the two observations -- exactly what we hoped for.
            continue
        except OSError as exc:
            if verbose:
                print(f"[baton] WARNING: could not remove {lock}: {exc}", flush=True)
    return removed


def sweep_stale_batons(root: Optional[Path] = None,
                       max_age_s: Optional[float] = None,
                       verbose: bool = True) -> List[Path]:
    """Startup sweep: remove ``lock`` files older than *max_age_s* and unheld.

    Returns the paths removed. Never raises: a cache that cannot be read or
    written must not take a run down with it, since the run works fine without
    the sweep -- it merely stays exposed to whatever poisoned names are there.
    """
    root = extensions_root() if root is None else Path(root)
    max_age_s = _max_age() if max_age_s is None else _clamp_max_age(max_age_s)
    if max_age_s <= 0:
        return []

    removed = _sweep(root, min_age_s=max_age_s, liveness_required=False, verbose=verbose)
    if verbose and removed:
        print(f"[baton] Cleared {len(removed)} stale torch-extension lock(s) under {root}. "
              f"A kernel reusing one of those extension names would otherwise have hung "
              f"until the compile alarm and been misreported as a memory fault.", flush=True)
    return removed


def sweep_unheld_batons(root: Optional[Path] = None,
                        min_age_s: float = _RACE_FLOOR_S,
                        verbose: bool = True) -> List[Path]:
    """Per-round sweep: remove locks nothing live is using any more.

    The startup sweep cannot help a run that poisons its own cache. A lineage
    killed mid-build leaves a lock whose mtime is minutes old, far inside the
    3600s age gate, so every later round of THAT run is exposed to the name it
    stranded -- and rounds are 10-25 minutes each.

    Age cannot be the gate here, so evidence of death is. Note carefully what
    that evidence is NOT: an unheld lock does not mean the builder is gone. The
    baton fd is ``O_CLOEXEC`` and so belongs to the Python process alone, while
    the actual compile is a ``subprocess.run(['ninja'], cwd=build_directory)``
    that survives its parent being killed. "Nobody holds the lock" is routinely
    true while nvcc is still writing into the directory. ``_open_inodes`` and
    ``_in_use`` therefore also check whether any live process has the BUILD
    DIRECTORY as its cwd, which is how an orphaned toolchain stays visible.

    Three conservatisms, each earning its place:

    * A candidate is observed twice, ``_CONFIRM_DELAY_S`` apart, and removed
      only if both observations agree AND the inode has not changed. This is
      what covers the ``os.close``/``os.remove`` window inside ``release()``;
      *min_age_s* does not, because age runs from ``O_CREAT``.
    * *min_age_s* covers the acquisition side instead: a lock created after the
      one-shot ``/proc`` snapshot has age ~0 and must not be judged by it.
    * Without ``/proc`` this removes NOTHING. Falling back to age would mean
      deleting minute-old locks on no evidence, which is the corruption this
      module exists to avoid.

    Assumes the cache belongs to the invoking user, which it does -- it lives
    under ``$HOME``. Another user's build would be invisible in ``/proc`` and
    could be collected; that requires a deliberately shared cache directory.

    ``KERNELMEM_BATON_MAX_AGE=0`` disables this sweep too, so the documented off
    switch turns off all lock removal rather than just the startup half.
    """
    root = extensions_root() if root is None else Path(root)
    if _max_age() <= 0:                 # the documented off switch covers both
        return []
    removed = _sweep(root, min_age_s=max(0.0, min_age_s),
                     liveness_required=True, verbose=verbose)
    if verbose and removed:
        print(f"[baton] Cleared {len(removed)} lock(s) left behind by a killed build. "
              f"Later rounds can now reuse those extension names instead of hanging "
              f"on them.", flush=True)
    return removed


if __name__ == "__main__":
    import sys
    n = sweep_stale_batons()
    print(f"{len(n)} stale lock(s) removed.")
    sys.exit(0)
