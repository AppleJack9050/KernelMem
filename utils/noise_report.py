#!/usr/bin/env python
"""Summarise noise_probe output: spread of an unchanged kernel, per level.

Reports each metric the optimization loop can rank on, because they do NOT have
the same noise: ``test_ms`` has no denominator, while ``score`` = T_ref/T_test
carries a separately measured reference and therefore roughly doubles the
variance it inherits.

Also splits within-call jitter (the 100 CUDA-event reps of one call) from
call-to-call offset, since only the second kind survives averaging and can be
mistaken for a real change.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Dict, List


def _stats(xs: List[float], name: str, unit: str = "") -> str:
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else 0.0
    lo, hi = min(xs), max(xs)
    cv = 100 * sd / m if m else float("nan")
    spread = 100 * (hi / lo - 1) if lo else float("nan")
    # +/-2 sd as a percentage of the mean: the band inside which a re-measurement
    # of an UNCHANGED kernel lands ~95% of the time. Any "improvement" smaller
    # than this is indistinguishable from having measured the same kernel twice.
    band = 2 * cv
    return (f"  {name:<28} n={n:<3} mean={m:.4f}{unit}  sd={sd:.4f}{unit}  "
            f"cv={cv:.3f}%  min={lo:.4f} max={hi:.4f}  "
            f"peak-to-peak={spread:.2f}%  +/-2sd={band:.2f}%")


def _drift(xs: List[float]) -> str:
    """Least-squares slope over call index, as % of the mean per 10 calls.

    Separates a monotone ramp (clock/thermal) from symmetric jitter: a ramp is
    not reduced by averaging more reps and biases whichever kernel ran later.
    """
    n = len(xs)
    if n < 3:
        return "  (too few points for a trend)"
    mx = (n - 1) / 2
    my = st.mean(xs)
    num = sum((i - mx) * (x - my) for i, x in enumerate(xs))
    den = sum((i - mx) ** 2 for i in range(n))
    slope = num / den
    first_half = st.mean(xs[: n // 2])
    second_half = st.mean(xs[n - n // 2:])
    return (f"  trend: {100 * slope * 10 / my:+.3f}% per 10 calls   "
            f"first-half {first_half:.4f} -> second-half {second_half:.4f} "
            f"({100 * (second_half / first_half - 1):+.3f}%)")


def load(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") == "call":
                out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    for f in args.files:
        calls = load(Path(f))
        if not calls:
            print(f"\n=== {f}: no calls ===")
            continue
        tag = calls[0].get("tag", "?")
        print(f"\n=== {f}   tag={tag}   calls={len(calls)} ===")

        geo = [c["geo_test_ms"] for c in calls]
        print(_stats(geo, "geomean test_ms (4 shapes)", " ms"))
        print(_drift(geo))

        scores = [c["score"] for c in calls if c.get("score") is not None]
        if scores:
            print(_stats(scores, "score (T_ref/T_test)"))
            print(_drift(scores))

        shapes = [s["shape"] for s in calls[0]["per_shape"]]
        for sh in shapes:
            v = [next(s["test_ms"] for s in c["per_shape"] if s["shape"] == sh)
                 for c in calls]
            print(_stats(v, f"  test_ms {sh}", " ms"))
        for sh in shapes:
            v = [next(s["ref_ms"] for s in c["per_shape"] if s["shape"] == sh)
                 for c in calls if next(s["ref_ms"] for s in c["per_shape"]
                                        if s["shape"] == sh) is not None]
            if v:
                print(_stats(v, f"  ref_ms  {sh}", " ms"))

        # Within-call jitter: the spread of the 100 event-timed reps that get
        # averaged into ONE test_ms. Large here is harmless; large across calls
        # is what fools a comparison.
        cvs, mins, meds = [], [], []
        for c in calls:
            reps = c.get("primary_reps_ms") or []
            if len(reps) > 2:
                cvs.append(100 * st.stdev(reps) / st.mean(reps))
                mins.append(min(reps))
                meds.append(st.median(reps))
        if cvs:
            print(f"  within-call rep CV (primary shape): median {st.median(cvs):.3f}%  "
                  f"max {max(cvs):.3f}%   (averaged away into one test_ms)")
            print(_stats(mins, "  primary MIN-of-reps", " ms"))
            print(_stats(meds, "  primary MEDIAN-of-reps", " ms"))

        clks = []
        for c in calls:
            g = (c.get("gpu") or {}).get("gpus") or []
            for row in g:
                if row.get("index") == "0":
                    try:
                        clks.append(float(row["sm_mhz"]))
                    except Exception:
                        pass
        if clks:
            print(f"  gpu0 SM clock after call: min={min(clks):.0f} max={max(clks):.0f} MHz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
