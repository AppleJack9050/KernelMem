#!/usr/bin/env python
"""Checks for the clock-lock preflight.

Run:  python -m utils.test_clock_lock

This gate decides whether a run is allowed to measure anything at all, so its
failure modes are worse than a wrong number: fail open and every artifact from
that run is silently unreproducible, fail closed wrongly and no run starts. The
cases below are the ones where that distinction is decided.

Nothing here pins a real clock -- the privileged calls are stubbed, so this runs
on a machine with no GPU and no sudo.
"""
from __future__ import annotations

import json
import os
import tempfile

from utils import clock_lock as cl


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _Env:
    """Restore every clock-lock env var, whatever the test did to them."""

    KEYS = (cl.ENV_POLICY, cl.ENV_OWNER, cl.ENV_STATE, cl.ENV_GPU_MHZ,
            cl.ENV_DRAM_MHZ, cl.ENV_KEEP, cl.ENV_AUTOCAL)

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Stub:
    """Replace the nvidia-smi-facing functions with a scripted fake device."""

    def __init__(self, *, name="NVIDIA GeForce RTX 5090", sm=2700, max_sm=3105,
                 gr=(3090, 2700, 2610, 2407, 1800, 400), mem=(14001, 13801, 810, 405),
                 can_lock=True, lock_succeeds=True, ext_locked=False):
        self.name, self.sm, self.max_sm = name, sm, max_sm
        self.gr, self.mem = list(gr), list(mem)
        self._can_lock, self._lock_succeeds, self._ext = can_lock, lock_succeeds, ext_locked
        self.locked_to = None
        self.unlocks = 0

    def __enter__(self):
        self._orig = {n: getattr(cl, n) for n in
                      ("smi_index", "device_name", "current_clocks", "supported_clocks",
                       "can_lock", "lock", "unlock", "detect_external_lock",
                       "_release_on_exit")}
        cl.smi_index = lambda device_idx=0: 0
        cl.device_name = lambda smi_idx=0: self.name
        cl.current_clocks = lambda smi_idx=0: {
            "sm_mhz": self.sm, "mem_mhz": self.mem[0], "max_sm_mhz": self.max_sm,
            "temp_c": 60, "power_w": "400", "util_pct": 90, "throttle": ""}
        cl.supported_clocks = lambda smi_idx=0: (self.mem, self.gr)
        cl.can_lock = lambda smi_idx=0: self._can_lock
        cl.detect_external_lock = lambda smi_idx=0: {
            "locked": self._ext, "mhz": self.sm if self._ext else None,
            "evidence": "stub", "observed": cl.current_clocks(0)}
        cl._release_on_exit = lambda smi_idx: None

        def _lock(smi_idx=0, gpu_mhz=None, dram_mhz=None):
            if gpu_mhz is None:
                gpu_mhz, dram_mhz, why = cl.resolve_target(smi_idx)
            else:
                why = "explicit argument"
            self.locked_to = gpu_mhz
            return {"locked": bool(self._lock_succeeds), "gpu_name": self.name,
                    "smi_index": smi_idx, "target_gpu_mhz": gpu_mhz,
                    "target_dram_mhz": dram_mhz, "max_gpu_mhz": self.max_sm,
                    "chosen_by": why, "dram_note": "", "observed": cl.current_clocks(0),
                    "error": "" if self._lock_succeeds else "stub refused to lock"}

        def _unlock(smi_idx=0):
            self.unlocks += 1

        cl.lock, cl.unlock = _lock, _unlock
        return self

    def __exit__(self, *exc):
        for n, fn in self._orig.items():
            setattr(cl, n, fn)


def test_unlocked_run_is_refused() -> None:
    """The whole point: no lock, no measurement."""
    with _Env(), _Stub(can_lock=False):
        try:
            cl.ensure_locked(0, what="test", verbose=False)
        except cl.ClockLockError as exc:
            _check("install_clock_lock_sudoers" in str(exc),
                   "the refusal must say how to fix it, not just that it failed")
            print("PASS  a run that cannot pin the clock is refused, with the fix")
            return
        raise AssertionError("ensure_locked returned instead of refusing")


def test_lock_is_taken_when_absent() -> None:
    """'Make sure it is locked; if not, lock it' -- the second half."""
    with _Env(), _Stub() as stub:
        st = cl.ensure_locked(0, what="test", verbose=False)
        _check(st["locked"], "should report locked")
        _check(stub.locked_to == 2407, f"should pin the 5090 preset, pinned {stub.locked_to}")
        _check(os.environ[cl.ENV_OWNER] == str(os.getpid()),
               "the locking process must record itself as owner")
        _check(json.loads(os.environ[cl.ENV_STATE])["locked"] is True,
               "state must be exported for children and artifacts")
        print(f"PASS  an unpinned clock is pinned at run start ({stub.locked_to} MHz)")


def test_child_verifies_but_never_relocks() -> None:
    """A child that re-locked would also unlock on exit, un-pinning its parent."""
    with _Env(), _Stub() as stub:
        os.environ[cl.ENV_OWNER] = "999999"
        os.environ[cl.ENV_STATE] = json.dumps(
            {"locked": True, "gpu_name": "stub", "target_gpu_mhz": 2700,
             "chosen_by": "parent"})
        st = cl.ensure_locked(0, what="child", verbose=False)
        _check(st["locked"], "child should confirm the inherited lock")
        _check(stub.locked_to is None, "child must not re-lock")
        _check(stub.unlocks == 0, "child must not unlock")
        print("PASS  a child verifies the inherited pin and touches nothing")


def test_mid_run_release_is_caught() -> None:
    """Something else moved the clock: the run must not keep quietly measuring."""
    with _Env(), _Stub(sm=1200):
        os.environ[cl.ENV_OWNER] = "999999"
        os.environ[cl.ENV_STATE] = json.dumps(
            {"locked": True, "gpu_name": "stub", "target_gpu_mhz": 2700,
             "chosen_by": "parent"})
        try:
            cl.ensure_locked(0, what="child", verbose=False)
        except cl.ClockLockError as exc:
            _check("1200" in str(exc), "the error should name the clock it found")
            print("PASS  a lock released mid-run is caught, not measured through")
            return
        raise AssertionError("drift off the pinned clock was not detected")


def test_external_lock_is_adopted_not_fought() -> None:
    """A machine whose operator pins clocks centrally already satisfies the rule."""
    with _Env(), _Stub(ext_locked=True, sm=2400) as stub:
        st = cl.ensure_locked(0, what="test", verbose=False)
        _check(st["locked"], "a pre-existing pin satisfies the requirement")
        _check(st["target_gpu_mhz"] == 2400, "should adopt the frequency in force")
        _check(stub.locked_to is None, "must not re-pin what someone else pinned")
        _check(st["owned_by_this_run"] is False,
               "and must not claim ownership, or it would unlock someone else's pin")
        print("PASS  a pre-existing external lock is adopted, not overridden")


def test_opt_out_is_explicit_and_recorded() -> None:
    with _Env(), _Stub(can_lock=False):
        os.environ[cl.ENV_POLICY] = "0"
        st = cl.ensure_locked(0, what="test", verbose=False)
        _check(st["locked"] is False, "opt-out must not claim to be locked")
        _check(json.loads(os.environ[cl.ENV_STATE])["locked"] is False,
               "the opt-out must reach the artifacts, not just the console")
        print("PASS  opting out is allowed, and is stamped 'locked: false'")


def test_target_is_per_device() -> None:
    """One frequency per device, never one frequency for every device."""
    with _Env(), _Stub(name="NVIDIA A100-SXM4-80GB", gr=(1410, 1275, 1065, 600),
                       mem=(1593, 1215)):
        gpu, _dram, why = cl.resolve_target(0)
        _check(gpu == 1065, f"A100 should resolve to its own preset, got {gpu}")
        _check("A100" in why, f"and say so: {why}")
    with _Env(), _Stub(name="NVIDIA GeForce RTX 5090"):
        gpu, dram, _why = cl.resolve_target(0)
        # 2407 rather than the ~2763 this card sustains: the SOL peak constant
        # this repo scores against has 2.41 GHz baked into it, so the pin is set
        # for comparability with T_SOL, not for throughput.
        _check(gpu == 2407, f"5090 should resolve to its own preset, got {gpu}")
        _check(dram == 13801, f"and to the memory bin it actually runs, got {dram}")
    print("PASS  each device resolves to its own measured frequency")


def test_unknown_device_falls_back_by_class_and_says_so() -> None:
    with _Env(), _Stub(name="NVIDIA GeForce RTX 9090", gr=(3000, 2550, 2400, 900),
                       max_sm=3000):
        gpu, _dram, why = cl.resolve_target(0)
        _check(why.startswith("UNCALIBRATED"),
               f"an unknown card must be labelled a guess: {why}")
        _check(gpu == 2550, f"consumer fallback is 85% of 3000 -> 2550, got {gpu}")
        _check("--calibrate" in why, "and must point at the way to measure it")
    with _Env(), _Stub(name="NVIDIA H200", gr=(2000, 1500, 1000), max_sm=2000):
        gpu, _dram, why = cl.resolve_target(0)
        _check(cl.device_class("NVIDIA H200") == "datacenter", "H200 is datacenter class")
        _check(gpu == 1500, f"datacenter fallback is 75% of 2000 -> 1500, got {gpu}")
    print("PASS  an unknown card falls back per class, snapped to a supported bin")


def test_override_beats_everything() -> None:
    with _Env(), _Stub():
        os.environ[cl.ENV_GPU_MHZ] = "2610"
        gpu, _dram, why = cl.resolve_target(0)
        _check(gpu == 2610, f"explicit override must win, got {gpu}")
        _check(cl.ENV_GPU_MHZ in why, f"and be recorded as the reason: {why}")
    print("PASS  an operator override outranks preset and measurement")


def test_measured_cache_outranks_builtin_preset() -> None:
    """The same model behaves differently box to box; this box's number wins."""
    with _Env(), _Stub() as stub, tempfile.TemporaryDirectory() as tmp:
        orig = cl.MEASURED_PRESETS
        cl.MEASURED_PRESETS = os.path.join(tmp, "clock_presets.json")
        try:
            cl._save_measured(stub.name, {"gpu_mhz": 2610, "dram_mhz": 13801,
                                          "settled_median_mhz": 2670,
                                          "measured_on": "2026-08-14"})
            gpu, dram, why = cl.resolve_target(0)
            _check(gpu == 2610, f"the measured value must win over the preset, got {gpu}")
            _check(dram == 13801, "and carry its memory clock with it")
            _check("measured on this machine" in why, f"and say where it came from: {why}")
        finally:
            cl.MEASURED_PRESETS = orig
    print("PASS  a value measured on this machine outranks the built-in preset")


def test_snap_to_supported_bins() -> None:
    _check(cl._snap(2705, [3090, 2700, 2610]) == 2700, "should snap to the nearest bin")
    _check(cl._snap(13900, [14001, 13801, 810]) == 13801, "memory clocks snap too")
    _check(cl._snap(2700, []) == 2700, "with no list, leave the value alone")
    print("PASS  targets snap to frequencies the driver actually offers")


def main() -> None:
    test_unlocked_run_is_refused()
    test_lock_is_taken_when_absent()
    test_child_verifies_but_never_relocks()
    test_mid_run_release_is_caught()
    test_external_lock_is_adopted_not_fought()
    test_opt_out_is_explicit_and_recorded()
    test_target_is_per_device()
    test_unknown_device_falls_back_by_class_and_says_so()
    test_override_beats_everything()
    test_measured_cache_outranks_builtin_preset()
    test_snap_to_supported_bins()
    print("\nAll clock-lock checks passed.")


if __name__ == "__main__":
    main()
