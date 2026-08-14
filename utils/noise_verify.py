#!/usr/bin/env python
"""Does the harness resolve a 1% difference? Measure it, do not assume it.

Run:
    python -m utils.noise_verify --ref tasks/vae_block_002.py \
        --kernel run/vae_block_002/kernels/agent_best.py \
        --processes 8 --calls 3 --band 1.0

What this answers
-----------------
The optimization loop accepts or rejects a candidate by comparing its score with
an incumbent's. That comparison is only meaningful if re-measuring an UNCHANGED
kernel moves the score by less than the difference being judged. So: run the same
kernel, from the same file, against the same reference, at the same shapes and
seed, many times, and report how far apart the answers land.

The number that matters is the **+/-2 sd band**: the interval a re-measurement of
an unchanged kernel falls in ~95% of the time. An "improvement" smaller than that
band is indistinguishable from having measured the same kernel twice. The gate
here is on that band, not on the standard deviation, because the band is what a
decision rule actually has to clear.

Two levels, separated on purpose
--------------------------------
* **L1, within one process** -- repeated calls in one interpreter. Excludes
  extension load, allocator state and context setup.
* **L2, across fresh processes** -- one call per interpreter, which is what the
  loop actually sees: ``main_memory_latest.py`` spawns a fresh subprocess per
  candidate, so every candidate is measured in a process the previous one never
  touched. **L2 is the level the 1% claim has to hold at**, because every
  candidate-vs-incumbent comparison the loop makes is a comparison across
  processes.

Reading the two against each other
----------------------------------
They are NOT directly comparable as printed. Each L2 point is the *mean* of one
process's ``--calls`` measurements, so within-process jitter is already averaged
down by sqrt(calls), while each L1 point is a single call. Measured here, L2
(0.46%) therefore came out *below* L1 (0.72%), which says nothing about process
setup being free. Separate the two terms instead:

    between-process sd = sqrt( sd_L2^2 - (sd_L1 / sqrt(calls))^2 )

On the 2026-08-14 run that is sqrt(0.229^2 - (0.361/sqrt3)^2) ~= 0.10% -- the
genuine cost of being measured in a fresh process, small but not zero. The
printed summary reports this term so the comparison is not misread.

Two traps this is built to avoid
--------------------------------
1. **A kernel that autotunes per instance is not a fixed subject.**
   ``kernel_autotune_splitk.py`` keeps its choice in ``self._choice``, so two
   instances can select different configs and the spread that produces (8.1%
   across processes) is the kernel re-rolling, not the harness measuring. Use a
   kernel with a fixed config, or pin the config, or this measures the lottery.
2. **An unpinned clock makes the answer a property of the room.** The clock state
   is recorded in the output and printed with the verdict, so a passing run at
   boost clock cannot be quoted as if it were a passing run at a pinned clock.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _geo(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _band(xs: List[float]) -> Dict[str, float]:
    """Spread of a metric, in the terms a decision rule cares about."""
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else 0.0
    cv = 100.0 * sd / m if m else float("nan")
    return {"n": n, "mean": m, "sd": sd, "cv_pct": cv,
            "band_2sd_pct": 2.0 * cv,
            "min": min(xs), "max": max(xs),
            "peak_to_peak_pct": 100.0 * (max(xs) / min(xs) - 1.0) if min(xs) else float("nan")}


def _fmt(name: str, b: Dict[str, float], unit: str = "") -> str:
    return (f"  {name:<26} n={b['n']:<3} mean={b['mean']:.4f}{unit}  "
            f"cv={b['cv_pct']:.3f}%  +/-2sd={b['band_2sd_pct']:.2f}%  "
            f"peak-to-peak={b['peak_to_peak_pct']:.2f}%")


def _ppid(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _is_ours(pid: int, roots: set) -> bool:
    """Is *pid* this verifier or one of its descendants?"""
    seen = 0
    cur: Optional[int] = pid
    while cur and cur > 1 and seen < 40:
        if cur in roots:
            return True
        cur = _ppid(cur)
        seen += 1
    return False


def _other_gpu_pids(roots: Optional[set] = None) -> List[int]:
    """PIDs using the GPU that are not us or our children.

    Walks the parent chain rather than comparing against a single pid: the probe
    runs in a subprocess which may itself spawn, and counting our own worker as
    an interloper would make the verifier wait for itself forever.
    """
    roots = roots or {os.getpid()}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in res.stdout.strip().splitlines():
        line = line.strip()
        if line.isdigit() and not _is_ours(int(line), roots):
            out.append(int(line))
    return out


def _wait_for_free_gpu(timeout_s: float = 1800.0, quiet_for_s: float = 3.0) -> bool:
    """Block until nothing else is on the GPU.

    Not politeness -- correctness. A second process running kernels during a
    measurement moves clocks, L2 residency and SM occupancy by far more than the
    1% being resolved, so an overlapped sample is not a noisier sample, it is a
    meaningless one. This box runs several sessions at once, so the check is
    repeated before every probe rather than once at the start.
    """
    t0 = time.monotonic()
    quiet_since: Optional[float] = None
    warned = False
    while time.monotonic() - t0 < timeout_s:
        busy = _other_gpu_pids()
        if not busy:
            quiet_since = quiet_since if quiet_since is not None else time.monotonic()
            if time.monotonic() - quiet_since >= quiet_for_s:
                return True
        else:
            quiet_since = None
            if not warned:
                print(f"[verify] another process is on the GPU (pids {busy}); "
                      f"waiting -- overlapping would invalidate the measurement",
                      flush=True)
                warned = True
        time.sleep(2.0)
    print("[verify] gave up waiting for a free GPU", flush=True)
    return False


def _run_probe(ref: Path, kernel: Path, out: Path, tag: str, calls: int,
               device: int, warmup: int, repeat: int,
               env: Dict[str, str]) -> Optional[List[int]]:
    """Run one probe process, watching for anyone else on the GPU while it runs.

    Checking only before the probe starts is not enough on a shared box: a
    neighbour that starts one second in overlaps the whole measurement, and the
    result looks like harness noise rather than like two processes sharing a
    card. Returns the foreign PIDs seen during the window (empty = clean),
    or None if the probe failed.
    """
    import threading

    cmd = [sys.executable, "-u", "-m", "utils.noise_probe",
           "--ref", str(ref), "--kernel", str(kernel), "--calls", str(calls),
           "--device", str(device), "--warmup", str(warmup), "--repeat", str(repeat),
           "--time_ref", "--tag", tag, "--out", str(out)]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    intruders: set = set()
    stop = threading.Event()
    roots = {os.getpid(), proc.pid}

    def _watch() -> None:
        while not stop.is_set():
            for pid in _other_gpu_pids(roots):
                intruders.add(pid)
            stop.wait(1.0)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    stdout, _ = proc.communicate()
    stop.set()
    watcher.join(timeout=3)

    if proc.returncode != 0:
        print(f"[verify] probe {tag} FAILED (rc={proc.returncode}):\n"
              f"{(stdout or '')[-2000:]}", flush=True)
        return None
    return sorted(intruders)


def _load_calls(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "call":
                rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="tasks/vae_block_002.py")
    ap.add_argument("--kernel", default="run/vae_block_002/kernels/agent_best.py")
    ap.add_argument("--processes", type=int, default=8,
                    help="Fresh interpreters (L2). This is the level that matters.")
    ap.add_argument("--calls", type=int, default=3,
                    help="Calls inside each process (L1)")
    ap.add_argument("--band", type=float, default=1.0,
                    help="Pass if the L2 +/-2sd band is within this %%")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--out", default=None, help="JSONL path (default: alongside a summary)")
    ap.add_argument("--label", default="", help="Tag for the summary, e.g. 'locked'")
    args = ap.parse_args()

    ref, kernel = Path(args.ref), Path(args.kernel)
    for p in (ref, kernel):
        if not p.is_file():
            print(f"[verify] not a file: {p}")
            return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = args.label or "run"
    out = Path(args.out) if args.out else Path(f"noise_verify_{label}_{stamp}.jsonl")
    env = dict(os.environ)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # tasks/*.py hardcodes an absolute SOLBENCH_SRC default from the machine it
    # was generated on, so importing the reference dies here without this. It
    # fails at IMPORT, which reads as a compilation error rather than a missing
    # path, so set it rather than leaving the next person to decode that.
    env.setdefault("SOLBENCH_SRC",
                   str(Path(__file__).resolve().parent.parent /
                       "third_party" / "SOL-ExecBench" / "src"))

    from utils import clock_lock
    clock = clock_lock.state()
    print(f"[verify] {label}: {args.processes} processes x {args.calls} calls, "
          f"kernel={kernel.name}")
    print(f"[verify] {clock_lock.describe(clock)}")
    print(f"[verify] writing {out}\n")

    t0 = time.time()
    dirty_tags: Dict[str, List[int]] = {}
    for i in range(args.processes):
        if not _wait_for_free_gpu():
            return 2
        tag = f"L2_p{i}"
        intruders = _run_probe(ref, kernel, out, tag, args.calls,
                               args.device, args.warmup, args.repeat, env)
        if intruders is None:
            print("[verify] aborting: a probe process failed")
            return 2
        note = ""
        if intruders:
            dirty_tags[tag] = intruders
            note = f"  CONTAMINATED (shared the GPU with pids {intruders})"
        print(f"[verify] process {i + 1}/{args.processes} done "
              f"({time.time() - t0:.0f}s elapsed){note}", flush=True)

    if dirty_tags:
        print(f"\n[verify] discarding {len(dirty_tags)} of {args.processes} "
              f"processes that shared the GPU: {sorted(dirty_tags)}")

    rows = [r for r in _load_calls(out) if r.get("tag") not in dirty_tags]
    if len(rows) < 2:
        print("[verify] not enough clean measurements to say anything")
        return 2

    # L2: one number per PROCESS (the mean of its own calls), so within-process
    # jitter does not leak into the between-process figure.
    by_pid: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_pid.setdefault(r["pid"], []).append(r)

    l2_score = [st.mean([c["score"] for c in cs if c.get("score")])
                for cs in by_pid.values()
                if any(c.get("score") for c in cs)]
    l2_time = [st.mean([c["geo_test_ms"] for c in cs]) for cs in by_pid.values()]
    l1_time = [c["geo_test_ms"] for c in rows]
    l1_score = [c["score"] for c in rows if c.get("score")]

    print(f"\n=== noise over {len(by_pid)} processes x {args.calls} calls "
          f"({time.time() - t0:.0f}s) ===")
    print("L1  within one process (excludes process setup):")
    print(_fmt("kernel time (geo)", _band(l1_time), " ms"))
    if len(l1_score) > 1:
        print(_fmt("score = T_ref/T_test", _band(l1_score)))
    print("L2  across fresh processes -- what a round actually sees:")
    print(_fmt("kernel time (geo)", _band(l2_time), " ms"))
    verdict_band = None
    between_cv = None
    if len(l2_score) > 1:
        b = _band(l2_score)
        print(_fmt("score = T_ref/T_test", b))
        verdict_band = b["band_2sd_pct"]
        # An L2 point is a mean of `calls` measurements, so part of its spread is
        # just L1 averaged down. Subtract that in quadrature to isolate what being
        # a FRESH PROCESS actually costs -- otherwise L2 < L1 reads as a paradox.
        l1_cv = _band(l1_score)["cv_pct"]
        expected = l1_cv / math.sqrt(max(1, args.calls))
        resid = b["cv_pct"] ** 2 - expected ** 2
        between_cv = math.sqrt(resid) if resid > 0 else 0.0
        print(f"  {'-> of which fresh-process':<26} "
              f"cv={between_cv:.3f}%  (L2 {b['cv_pct']:.3f}% vs "
              f"{expected:.3f}% expected from averaging {args.calls} calls)")
    else:
        verdict_band = _band(l2_time)["band_2sd_pct"]

    # The clock each measurement actually ran at: a spread explained by the clock
    # having moved is not the harness's noise, and it is the first thing to check.
    sms = [int(g["sm_mhz"]) for r in rows for g in (r.get("gpu", {}).get("gpus") or [])
           if str(g.get("sm_mhz", "")).strip().isdigit()]
    if sms:
        print(f"\n  SM clock across all calls: {min(sms)}-{max(sms)} MHz "
              f"(spread {100 * (max(sms) / min(sms) - 1):.1f}%)")

    ok = verdict_band is not None and verdict_band <= args.band
    print(f"\n[verify] L2 +/-2sd band = {verdict_band:.2f}%  vs target {args.band:.2f}%"
          f"  -> {'PASS' if ok else 'FAIL'}")
    print(f"[verify] clock: {'PINNED at ' + str(clock.get('target_gpu_mhz')) + ' MHz' if clock.get('locked') else 'NOT PINNED -- this figure is a property of this room, not of the harness'}")

    summary = {"label": label, "kernel": str(kernel), "ref": str(ref),
               "processes": len(by_pid), "calls_per_process": args.calls,
               "warmup": args.warmup, "repeat": args.repeat,
               "clock": clock, "target_band_pct": args.band,
               "l1_time": _band(l1_time), "l2_time": _band(l2_time),
               "l1_score": _band(l1_score) if len(l1_score) > 1 else None,
               "l2_score": _band(l2_score) if len(l2_score) > 1 else None,
               "verdict_band_pct": verdict_band, "pass": ok,
               "between_process_cv_pct": between_cv,
               "discarded_contaminated": {k: v for k, v in dirty_tags.items()},
               "sm_clock_min": min(sms) if sms else None,
               "sm_clock_max": max(sms) if sms else None,
               "seconds": round(time.time() - t0, 1)}
    sp = out.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[verify] summary -> {sp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
