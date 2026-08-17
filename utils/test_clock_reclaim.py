"""Checks for stale-pin reclamation after a run is killed.

    python -m utils.test_clock_reclaim

Releasing the pin was already handled for every exit a process can observe:
atexit for a normal return, SIGINT/SIGTERM handlers for a stop. SIGKILL, an OOM
kill and a power cut are not observable, and before this the card simply stayed
pinned with nothing left to say who pinned it -- ENV_OWNER lives in the
environment and dies with the process.

The next run then took the "pre-existing external lock" branch and adopted the
dead run's frequency as though an operator had chosen it. Measured 2026-08-16: a
SIGKILLed run left 2482 MHz, the next run adopted it instead of applying the
2407 MHz target, and its reference profile read 2224 us against ~2526 us at
target -- every score inflated against a T_SOL that assumes 2.41 GHz.

These checks never touch a real GPU: `unlock` is stubbed, so they can run while
a real run holds the real pin.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.clock_lock as cl  # noqa: E402

IDX = 991                      # a device index no real card will have


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


class _StubUnlock:
    """Record unlock calls instead of touching hardware."""

    def __enter__(self):
        self.calls = []
        self._real = cl.unlock

        def fake(smi_idx=0, *, keep_claim=False):
            self.calls.append(smi_idx)
            if not keep_claim:
                cl._clear_owner(smi_idx)
        cl.unlock = fake
        return self

    def __exit__(self, *a):
        cl.unlock = self._real


def _dead_pid() -> int:
    """A pid that is certainly not running: fork a child and reap it."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


cl._clear_owner(IDX)

print("[reclaim] a live owner is left alone")
cl._write_owner(IDX, 2407)
rec = cl._read_owner(IDX)
_check(rec is not None and rec["pid"] == os.getpid(), "the claim records this pid")
_check(rec.get("start_ticks") is not None, "and this process's start time")
_check(cl._owner_alive(rec), "the owner reads as alive")
with _StubUnlock() as st:
    _check(cl.reclaim_stale(IDX, verbose=False) is False, "a live pin is NOT reclaimed")
    _check(st.calls == [], "and unlock was never called")
_check(cl._read_owner(IDX) is not None, "the claim survives")

print("[reclaim] a dead owner's pin is released")
dead = _dead_pid()
cl._write_owner(IDX, 2482)
rec = cl._read_owner(IDX)
rec["pid"] = dead
rec["start_ticks"] = None
cl._lockfile(IDX).write_text(__import__("json").dumps(rec), encoding="utf-8")
_check(not cl._owner_alive(cl._read_owner(IDX)), "a reaped pid reads as dead")
with _StubUnlock() as st:
    _check(cl.reclaim_stale(IDX, verbose=False) is True, "the stale pin IS reclaimed")
    _check(st.calls == [IDX], "unlock was called exactly once, on the right device")
_check(cl._read_owner(IDX) is None, "and the stale claim is cleared")

print("[reclaim] pid reuse does not make a stale claim look live")
# The recycled-pid case: the pid exists (it is us) but started at a different
# time. Without start_ticks this reads as alive and the pin is never released.
cl._write_owner(IDX, 2482)
rec = cl._read_owner(IDX)
rec["start_ticks"] = (rec["start_ticks"] or 0) + 99999
cl._lockfile(IDX).write_text(__import__("json").dumps(rec), encoding="utf-8")
_check(not cl._owner_alive(cl._read_owner(IDX)),
       "same pid, different start time -> treated as a DIFFERENT process")
with _StubUnlock():
    _check(cl.reclaim_stale(IDX, verbose=False) is True, "so its pin is reclaimed")

print("[reclaim] no claim at all is not an error")
cl._clear_owner(IDX)
with _StubUnlock() as st:
    _check(cl.reclaim_stale(IDX, verbose=False) is False, "nothing to reclaim")
    _check(st.calls == [], "and nothing is unlocked -- a genuine operator pin is safe")

print("[reclaim] a corrupt claim is ignored rather than fatal")
cl._lockfile(IDX).write_text("{not json", encoding="utf-8")
_check(cl._read_owner(IDX) is None, "unparseable claim reads as absent")
with _StubUnlock():
    _check(cl.reclaim_stale(IDX, verbose=False) is False, "and reclaim is a no-op")
cl._clear_owner(IDX)

print("[reclaim] unlock drops the claim, so a released pin is never 'stale'")
cl._write_owner(IDX, 2407)
real = cl.unlock
cl.unlock = lambda smi_idx=0, *, keep_claim=False: (
    None if keep_claim else cl._clear_owner(smi_idx))
try:
    cl.unlock(IDX)
finally:
    cl.unlock = real
_check(cl._read_owner(IDX) is None, "claim gone after unlock")

print("[reclaim] the claim is machine-scoped and per-GPU")
p0, p1 = cl._lockfile(0), cl._lockfile(1)
_check(p0 != p1, "different GPUs get different claim files")
_check(str(p0).startswith(__import__("tempfile").gettempdir()),
       "and they live in the temp dir, which a reboot clears -- as does a reboot "
       "reset the clocks")

cl._clear_owner(IDX)
print("\n[reclaim] all checks passed")
