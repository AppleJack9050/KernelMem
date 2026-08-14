#!/usr/bin/env python
"""Run several kernel lineages in parallel, each with its own base and ratchet.

    python run_lineages.py tasks/vae_block_002.py --gpu "RTX 5090" \
        --plan keep_vendor,own_gemm,own_winograd \
        --round 10 --incumbent 1.2039

Why lineages
------------
`main_memory_latest.py` hill-climbs from ONE seed and never revisits the
granularity that seed chose. On vae_block_002 that seed decided 93.5% of the
final score and 21 optimization rounds decided 7%, while the spread between seed
draws was 10-41%. So the budget was going almost entirely into the small term,
and the large term was a single unrepeated sample.

Each lineage here is a separate `main_memory_latest.py` process with its own
`--subproc_id`, its own batch folder, and therefore its own base kernel and
ratchet -- which is exactly the "per-lineage ratchet" the single-process loop
cannot express. A structurally new kernel is compared only against its own
history, so it is not killed on evaluation one for being worse than an incumbent
it has not been tuned to beat yet. Process-level isolation also means the
existing, working round loop is reused unchanged rather than refactored.

Parallel where it pays, serial where it must
--------------------------------------------
Generation is ~95% of the wall clock (16-20 min per draw versus roughly a minute
of GPU), so lineages generate concurrently. But every GPU section takes a single
cross-process mutex (`utils/gpu_lock`), because two processes benchmarking the
same device do not merely add noise to a 0.5%-margin comparison -- they void it.
Parallelism buys wall clock here without costing measurement validity.

Two gates keep this from being a token bonfire
----------------------------------------------
* fund by ceiling: a lineage whose best conceivable outcome loses to the
  incumbent is never started (`utils.lineage.ceilings` / `fund`);
* stop by trajectory: a funded lineage is killed once its own score history
  says it cannot arrive in the rounds remaining (`trajectory_verdict`).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils import clock_lock
from utils.lineage import (LineageSpec, LineageState, ceilings, fund, vendor_overhead,
                           read_lineage_progress, read_stop_reason,
                           trajectory_verdict, vendor_split)

ROOT = Path(__file__).resolve().parent

# The stock plans. `keep_vendor` is the behaviour every previous run had; the
# other two own the vendor operator and differ only in the algorithm they use to
# replace it -- which is the term that actually moves the ceiling.
PLANS: Dict[str, LineageSpec] = {
    "keep_vendor": LineageSpec(
        id="keep_vendor", granularity="C", algorithm="implicit_gemm",
        owns_vendor_op=False,
        note="Fuse around the vendor GEMM/conv. What every run so far did."),
    "own_gemm": LineageSpec(
        id="own_gemm", granularity="D", algorithm="implicit_gemm",
        owns_vendor_op=True,
        note="Own the conv with the same algorithm; ceiling is vendor parity."),
    "own_winograd": LineageSpec(
        id="own_winograd", granularity="D", algorithm="winograd_f2x3",
        owns_vendor_op=True,
        note="Own the conv with Winograd F(2x2,3x3): 2.25x fewer multiplies."),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parallel multi-lineage kernel search")
    p.add_argument("arch_py", type=Path, help="Task .py (single task only)")
    p.add_argument("--plan", default="keep_vendor,own_gemm,own_winograd",
                   help=f"Comma-separated lineage ids from: {','.join(PLANS)}")
    p.add_argument("--gpu", default="RTX 5090")
    p.add_argument("--model_name", default="claude-opus-5")
    p.add_argument("--round", type=int, default=10, help="Rounds per lineage")
    p.add_argument("--num_seeds", type=int, default=3, help="Seed draws per lineage")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--work_dir", type=Path, default=Path("run"))
    p.add_argument("--incumbent", type=float, default=None,
                   help="Best speedup already achieved for this task. Lineages "
                        "whose CEILING cannot beat it by --min_gain are never "
                        "started, and it is the target the trajectory test "
                        "projects against.")
    p.add_argument("--min_gain", type=float, default=0.10,
                   help="Required headroom over --incumbent before a lineage is "
                        "funded. The ceiling is an optimistic bound, so a "
                        "lineage that only just clears it is not worth running.")
    p.add_argument("--residual_us", type=float, default=0.0,
                   help="Work that survives perfect fusion (us), e.g. a final "
                        "elementwise pass that must read and write the tensor. "
                        "Tightens the ceiling; without it a keep-the-vendor "
                        "lineage is bounded by raw Amdahl and looks fundable "
                        "when it is already exhausted. CALIBRATE IT, do not "
                        "guess: the ceiling is computed from nsys GPU time while "
                        "--incumbent is a wall-clock ratio, so the two are not on "
                        "the same basis and a guessed value can put the gate off "
                        "by several percent. Pick the value that makes the "
                        "ceiling of the lineage you ALREADY ran come out at the "
                        "score it actually reached, then reuse it. On "
                        "vae_block_002 (total 2181.8us, vendor 1615.3us, "
                        "incumbent 1.2039) that is about 200us.")
    p.add_argument("--max_parallel", type=int, default=0,
                   help="Lineages generating at once (0 = all). GPU sections are "
                        "serialized regardless, so this caps LLM concurrency and "
                        "host memory, not device contention.")
    p.add_argument("--grace", type=int, default=5,
                   help="OPTIMIZATION rounds a lineage runs before the trajectory "
                        "test may kill it (the seed draw does not count). A "
                        "structural change is worse before it is tuned; this is "
                        "the room it gets to prove otherwise.")
    p.add_argument("--child_patience", type=int, default=0,
                   help="--patience handed to each child (0 = disabled, the "
                        "default here). main_memory_latest.py has its own "
                        "plateau stopper, defaulting to 4, and leaving it on "
                        "gives a lineage TWO stoppers that cannot see each "
                        "other: the trajectory test is kill-only, so a child "
                        "can shut itself down on a 4-round drought even after "
                        "the coordinator has ruled 'already at X >= target, "
                        "keep'. Both rules also extrapolate from droughts, and "
                        "stacking two mechanisms with the same blind spot "
                        "roughly doubles the chance of ending a lineage right "
                        "before it pays -- on vae_block_002 a lineage killed at "
                        "1.6216 scored 1.7098 the very next round. The "
                        "coordinator is the only component with the "
                        "cross-lineage view, so it holds sole stopping "
                        "authority unless you set this back to 4.")
    p.add_argument("--poll_s", type=float, default=60.0,
                   help="Seconds between trajectory checks.")
    p.add_argument("--dry_run", action="store_true",
                   help="Profile, compute ceilings, print funding decisions, exit.")
    return p.parse_args()


def _profile_reference(task: Path, device: int) -> Optional[dict]:
    from utils.reference_profile import profile_reference
    return profile_reference(task, device)


def _spawn(spec: LineageSpec, state: LineageState, args, lock_file: Path) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u", str(ROOT / "main_memory_latest.py"), str(args.arch_py),
        "--gpu", args.gpu, "--model_name", args.model_name,
        "--round", str(args.round), "--num_seeds", str(args.num_seeds),
        "--device", str(args.device),
        # Per-lineage work_dir, NOT the shared root: the child names its batch
        # folder "<stamp>_<task>_<tag>" from a 1-second timestamp, and lineages
        # are spawned in a tight loop, so under the shared root two of them land
        # in the SAME folder and overwrite each other's checkpoint and code.
        "--work_dir", str(state.batch_dir),
        "--subproc_id", str(state.subproc_id),
        "--seed_granularity", spec.granularity,
        # Sole stopping authority lives in the coordinator -- see --child_patience.
        "--patience", str(args.child_patience),
    ]
    # The algorithm is what separates own_gemm from own_winograd; without it
    # they are the same lineage run twice.
    if spec.owns_vendor_op:
        cmd += ["--seed_algorithm", spec.algorithm]
    env = dict(os.environ)
    # Children serialize their GPU sections against each other through this.
    env["KERNELMEM_GPU_LOCK"] = str(lock_file)
    env["KERNELMEM_LINEAGE_LABEL"] = spec.label
    state.batch_dir.mkdir(parents=True, exist_ok=True)
    log = open(state.batch_dir / "lineage.log", "a")
    print(f"[lineage] starting {spec.label} -> {state.batch_dir}", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log,
                            stderr=subprocess.STDOUT)


def _child_batch_dir(lineage_dir: Path, task_stem: str) -> Path:
    """Where the child actually writes, resolved rather than assumed.

    The child names its own batch folder "<stamp>_<task>_<tag>" under the
    work_dir it is given, so the coordinator cannot know that name up front.
    Reading the lineage folder itself finds no checkpoint and therefore no
    scores -- which trajectory_verdict cannot distinguish from a lineage that
    has genuinely not finished a round, so it reports "within grace" forever and
    the stop-by-trajectory gate never fires.
    """
    cands = sorted((p for p in lineage_dir.glob(f"*_{task_stem}_*") if p.is_dir()),
                   key=lambda p: p.name)
    return cands[-1] if cands else lineage_dir


def main() -> None:
    args = _parse_args()
    if not args.arch_py.is_file():
        print(f"[ERROR] not a task file: {args.arch_py}")
        return

    wanted = [s.strip() for s in args.plan.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in PLANS]
    if unknown:
        print(f"[ERROR] unknown plan id(s): {unknown}. Known: {list(PLANS)}")
        return

    # Pin the clock once, here, before any lineage starts. Lineages are compared
    # against each other and funded on those comparisons, so they must all be
    # measured at the same frequency -- and since the coordinator's environment
    # is inherited by every child, one lock at the top covers all of them. A
    # child locking for itself would also unlock when it finished, un-pinning the
    # lineages still running.
    try:
        clock_lock.ensure_locked(args.device, what="lineage run")
    except clock_lock.ClockLockError as exc:
        print(f"\n[clock] {exc}\n")
        raise SystemExit(2)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.work_dir / f"{stamp}_{args.arch_py.stem}_lineages").resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_file = root / "gpu.lock"
    print(f"[lineage] root: {root}")

    # ---- profile once, share the numbers with every funding decision ----
    print("[lineage] profiling the reference to compute ceilings ...", flush=True)
    prof = _profile_reference(args.arch_py, args.device)
    if not prof:
        print("[ERROR] reference profiling failed; cannot compute ceilings.")
        return
    total_us, vendor_us = vendor_split(prof)
    # Vendor-internal satellites (convertTensor etc): fixed for keep_vendor,
    # deletable by a lineage that owns the op. Without this term own_gemm's
    # ceiling equals keep_vendor's and it can never be funded.
    overhead_us = vendor_overhead(prof)
    print(f"[lineage] reference {total_us:.1f} us/forward, vendor GEMM/conv "
          f"{vendor_us:.1f} us ({vendor_us/max(total_us,1e-9)*100:.1f}%)")

    # ---- fund by ceiling ----
    states: List[LineageState] = []
    for i, pid in enumerate(wanted):
        spec = PLANS[pid]
        ceil = ceilings(total_us, vendor_us, spec, residual_us=args.residual_us,
                        vendor_overhead_us=overhead_us)
        ok, why = fund(ceil, args.incumbent, args.min_gain)
        st = LineageState(spec=spec, subproc_id=100 + i,
                          batch_dir=root / spec.id, ceiling=ceil,
                          status="pending" if ok else "unfunded", reason=why)
        states.append(st)
        mark = "FUND " if ok else "SKIP "
        print(f"[lineage] {mark}{spec.label:<34} ceiling {ceil:>6.2f}x  -- {why}")

    funded = [s for s in states if s.status == "pending"]
    if not funded:
        print("[lineage] nothing clears the funding gate; nothing to run.")
        return
    if args.dry_run:
        print("[lineage] --dry_run: stopping before launch.")
        return

    # ---- launch ----
    cap = args.max_parallel if args.max_parallel > 0 else len(funded)
    procs: Dict[str, subprocess.Popen] = {}
    queue = list(funded)
    running: List[LineageState] = []

    def _start_more() -> None:
        while queue and len(running) < cap:
            st = queue.pop(0)
            procs[st.spec.id] = _spawn(st.spec, st, args, lock_file)
            st.status = "running"
            running.append(st)

    _start_more()
    task_stem = args.arch_py.stem
    target = (args.incumbent or 0.0) * (1.0 + args.min_gain) if args.incumbent else 0.0

    try:
        while running:
            time.sleep(args.poll_s)
            for st in list(running):
                p = procs[st.spec.id]
                st.scores, st.rounds_done = read_lineage_progress(
                    _child_batch_dir(st.batch_dir, task_stem), task_stem)
                st.best = max(st.scores) if st.scores else None

                if p.poll() is not None:
                    # rc=0 alone cannot separate "ran every round" from "stopped
                    # itself early": both exit clean. Use the round count, and
                    # the reason the child recorded, so the summary says which.
                    stop_reason = read_stop_reason(
                        _child_batch_dir(st.batch_dir, task_stem), task_stem)
                    if p.returncode != 0:
                        st.status = "failed"
                        st.reason = (f"exited rc={p.returncode} after "
                                     f"{st.rounds_done}/{args.round} rounds")
                    elif st.rounds_done >= args.round:
                        st.status = "done"
                        st.reason = f"completed all {args.round} rounds"
                    else:
                        st.status = "stopped_early"
                        st.reason = (f"self-stopped after {st.rounds_done}/{args.round} "
                                     f"rounds ({stop_reason or 'reason not recorded'}), "
                                     f"not a trajectory kill")
                    running.remove(st)
                    print(f"[lineage] {st.spec.label} finished "
                          f"(best {st.best if st.best is None else round(st.best,4)})",
                          flush=True)
                    _start_more()
                    continue

                if target > 0:
                    keep, why = trajectory_verdict(
                        st.scores, target, args.round - st.rounds_done, args.grace)
                    if not keep:
                        print(f"[lineage] killing {st.spec.label}: {why}", flush=True)
                        st.status, st.reason = "killed", why
                        p.send_signal(signal.SIGTERM)   # graceful: writes artifacts
                        running.remove(st)
                        _start_more()
    except KeyboardInterrupt:
        print("[lineage] interrupted; terminating children ...", flush=True)
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)

    for p in procs.values():
        try:
            p.wait(timeout=600)
        except Exception:
            p.kill()

    # ---- report ----
    for st in states:
        st.scores, st.rounds_done = read_lineage_progress(
            _child_batch_dir(st.batch_dir, task_stem), task_stem)
        st.best = max(st.scores) if st.scores else None
    ranked = sorted([s for s in states if s.best is not None],
                    key=lambda s: -s.best)
    summary = {
        "task": str(args.arch_py),
        "reference_us": total_us,
        "vendor_us": vendor_us,
        "incumbent": args.incumbent,
        "lineages": [{
            "id": s.spec.id, "granularity": s.spec.granularity,
            "algorithm": s.spec.algorithm, "ceiling": s.ceiling,
            "status": s.status, "reason": s.reason,
            "rounds_done": s.rounds_done, "best": s.best,
            "batch_dir": str(s.batch_dir),
        } for s in states],
        "winner": ranked[0].spec.id if ranked else None,
        "timestamp": datetime.now().isoformat(),
    }
    (root / "lineages.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== lineage summary ===")
    for s in states:
        b = "n/a" if s.best is None else f"{s.best:.4f}"
        print(f"  {s.spec.label:<34} ceiling {s.ceiling:>6.2f}x  best {b:>8}  "
              f"{s.status}  ({s.reason})")
    if ranked:
        w = ranked[0]
        print(f"\n[lineage] winner: {w.spec.label} at {w.best:.4f}")
        if args.incumbent:
            d = (w.best / args.incumbent - 1.0) * 100.0
            print(f"[lineage] vs incumbent {args.incumbent:.4f}: {d:+.2f}%")
            print("[lineage] NOTE: cross-lineage scores come from separate "
                  "sessions and carry drift. Confirm the winner against the "
                  "incumbent with a paired re-measure before believing the gap.")
    print(f"[lineage] wrote {root/'lineages.json'}")


if __name__ == "__main__":
    main()
