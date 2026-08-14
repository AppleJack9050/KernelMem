#!/usr/bin/env python
"""Run the loop's own accept/reject statistic on a kernel against ITSELF.

``adaptive_paired_verdict`` is what decides whether a round advances. Feeding it
two byte-identical copies of one kernel makes the true effect exactly zero, so
everything it reports -- ``rel_pct``, ``t``, ``beats_margin`` -- is noise by
construction. Repeating that gives the false-positive rate of the gate and the
size of the improvement it can hallucinate, measured rather than assumed.

This is the number that matters operationally: L1/L2 spreads describe the raw
timings, but the loop never reads a raw timing, it reads this verdict.

    python -m utils.noise_null_verdict --ref tasks/vae_block_002.py \
        --kernel run/vae_block_002/kernels/agent_best.py --trials 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import os
from datetime import datetime
from pathlib import Path

from utils.paired_bench import adaptive_paired_verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--tmpdir", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # The false-positive rate this measures is a property of the verdict function
    # at a FIXED clock. Leave the clock free and it also folds in whatever the
    # driver did between trials, which is not what the number is read as.
    from utils import clock_lock
    try:
        clock_lock.ensure_locked(args.device, what="null-verdict trials")
    except clock_lock.ClockLockError as exc:
        print(f"\n[clock] {exc}\n")
        return 2

    # Two distinct paths holding identical bytes: the verdict function takes two
    # files, and giving it the same path twice would let any per-path caching
    # make the comparison artificially clean.
    src = Path(args.kernel)
    a = Path(args.tmpdir) / "null_base.py"
    b = Path(args.tmpdir) / "null_cand.py"
    shutil.copyfile(src, a)
    shutil.copyfile(src, b)
    assert a.read_bytes() == b.read_bytes() == src.read_bytes()

    out = Path(args.out)
    rel, ts, flagged, resolved = [], [], 0, 0
    for i in range(args.trials):
        v = adaptive_paired_verdict(
            Path(args.ref), a, b, device=args.device,
            warmup=args.warmup, repeat=args.repeat, tol=1e-2,
            margin=args.margin, sigma=args.sigma, log=lambda m: print(m, flush=True),
        )
        if v is None:
            print(f"  trial {i + 1}: measurement failed", flush=True)
            continue
        rel.append(v["rel_pct"])
        ts.append(v["t"])
        flagged += int(bool(v["beats_margin"]) and bool(v["sigma_ok"]))
        resolved += int(bool(v["resolved"]))
        with out.open("a") as fh:
            fh.write(json.dumps({"trial": i, "ts": datetime.now().isoformat(timespec="seconds"),
                                 **{k: v[k] for k in
                                    ("rel_pct", "se_pct", "t", "dof", "reps",
                                     "beats_margin", "sigma_ok", "resolved",
                                     "base_ms", "cand_ms", "base_ms_all",
                                     "cand_ms_all")}}) + "\n")
        print(f"  trial {i + 1}/{args.trials}: rel={v['rel_pct']:+.3f}% "
              f"se={v['se_pct']:.3f}% t={v['t']:+.2f} reps={v['reps']} "
              f"beats_margin={v['beats_margin']} sigma_ok={v['sigma_ok']}", flush=True)

    if rel:
        print(f"\n[null] n={len(rel)}  true effect = 0 by construction")
        print(f"  rel_pct: mean={st.mean(rel):+.3f}%  sd={st.stdev(rel) if len(rel) > 1 else 0:.3f}%  "
              f"min={min(rel):+.3f}%  max={max(rel):+.3f}%")
        print(f"  |rel| >= margin({100 * args.margin:.1f}%) in {sum(1 for r in rel if abs(r) >= 100 * args.margin)}/{len(rel)} trials")
        print(f"  ACCEPTED (beats margin AND passes {args.sigma} sigma): {flagged}/{len(rel)}")
        print(f"  resolved: {resolved}/{len(rel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
