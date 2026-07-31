#!/usr/bin/env python
"""Interleaved paired benchmarking for comparing kernels without drift.

Why this exists
---------------
``compare_and_bench`` measures one kernel in one process. The optimization loop
compares a challenger measured *now* against a base measured several rounds ago,
and GPU state drifts between those measurements. Measured on an RTX 5090 with
vae_block_002, three unchanged kernels re-benchmarked minutes after their run
all read high by +0.94%, +1.04% and +1.73% -- a shared multiplicative offset,
larger than any real per-round improvement the optimizer produced.

That drift hides real signal. A kernel verified here at +1.26% over its seed
(~8 sigma) showed up as only +0.57% when the two were measured rounds apart.

Interleaving base and challenger inside one session -- b, c, b, c, b, c -- makes
the drift hit both equally so it cancels in the difference. Running all of one
kernel's repeats before the other does NOT work: a clock ramp part-way through
the sequence reads as a difference between the kernels.

Usage
-----
    python -m utils.paired_bench ref_0.py seed.py candidate.py --reps 3

The first kernel is the baseline; every other kernel is compared against it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Dict, List

from utils.compile_and_run import compare_and_bench


def _welch(a: List[float], b: List[float]) -> Dict[str, float]:
    """Welch's t-test for two samples of possibly unequal variance.

    Returns the raw difference, its standard error, and t. With the small rep
    counts used here (3-5) the dof is tiny, so t is a rough guide rather than a
    precise p-value -- read |t| >= 3 as "clearly real", |t| <= 1 as "no signal".
    """
    na, nb = len(a), len(b)
    ma, mb = st.mean(a), st.mean(b)
    va = st.variance(a) if na > 1 else 0.0
    vb = st.variance(b) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb) if (na and nb) else float("nan")
    return {
        "diff": ma - mb,
        "rel_diff_pct": (ma / mb - 1.0) * 100.0 if mb else float("nan"),
        "se": se,
        "t": (ma - mb) / se if se > 0 else float("inf") if ma != mb else 0.0,
    }


def run(reference: Path, kernels: List[Path], reps: int, device: int,
        warmup: int, repeat: int, tol: float) -> Dict[str, Any]:
    # Interleave: rep-major, so every kernel is sampled once before any is
    # sampled twice. This is what makes the drift common-mode.
    samples: Dict[str, List[Dict[str, Any]]] = {k.stem: [] for k in kernels}
    for r in range(reps):
        for k in kernels:
            res = compare_and_bench(reference, k, device_idx=device,
                                    warmup=warmup, repeat=repeat, tol=tol)
            samples[k.stem].append(res)
            sc = res.get("score")
            print(f"  rep {r + 1}/{reps}  {k.stem:<28} score={sc:.4f}", flush=True)

    shapes = [s["shape"] for s in samples[kernels[0].stem][0]["per_shape"]]
    out: Dict[str, Any] = {"reference": str(reference), "reps": reps,
                           "shapes": shapes, "kernels": {}}

    for k in kernels:
        name = k.stem
        scores = [r["score"] for r in samples[name]]
        per_shape = {sh: [next(p["speedup"] for p in r["per_shape"] if p["shape"] == sh)
                          for r in samples[name]] for sh in shapes}
        mean = st.mean(scores)
        sd = st.stdev(scores) if len(scores) > 1 else 0.0
        out["kernels"][name] = {
            "path": str(k),
            "scores": scores,
            "mean": mean,
            "stdev": sd,
            "stdev_pct": sd / mean * 100.0 if mean else float("nan"),
            "sem_pct": (sd / math.sqrt(len(scores))) / mean * 100.0 if mean else float("nan"),
            "per_shape_mean": {sh: st.mean(v) for sh, v in per_shape.items()},
            "per_shape_stdev_pct": {
                sh: (st.stdev(v) / st.mean(v) * 100.0 if len(v) > 1 else 0.0)
                for sh, v in per_shape.items()
            },
        }

    # Every kernel after the first is compared against the first.
    base = kernels[0].stem
    base_scores = out["kernels"][base]["scores"]
    for k in kernels[1:]:
        name = k.stem
        cmp = _welch(out["kernels"][name]["scores"], base_scores)
        # Sign test across shapes. Per-shape noise is independent while drift
        # moves all shapes together, so "won on N of M shapes" survives drift
        # that any magnitude comparison on the geomean would fail.
        wins = [sh for sh in shapes
                if out["kernels"][name]["per_shape_mean"][sh]
                > out["kernels"][base]["per_shape_mean"][sh]]
        cmp["shape_wins"] = len(wins)
        cmp["shape_total"] = len(shapes)
        cmp["shapes_won"] = wins
        out["kernels"][name]["vs_base"] = cmp

    out["baseline"] = base
    return out


def _report(out: Dict[str, Any]) -> None:
    shapes = out["shapes"]
    base = out["baseline"]
    print("\n" + "=" * 78)
    print(f"PAIRED INTERLEAVED BENCHMARK   reps={out['reps']}   baseline={base}")
    print("=" * 78)
    print(f"{'kernel':<30}{'mean':>9}{'stdev':>8}{'sem':>8}")
    for name, d in out["kernels"].items():
        print(f"{name:<30}{d['mean']:>9.4f}{d['stdev_pct']:>7.2f}%{d['sem_pct']:>7.2f}%")

    print(f"\nPer-shape means (noise in parentheses, as stdev %):")
    print(f"{'shape':<20}" + "".join(f"{n[-13:]:>16}" for n in out["kernels"]))
    for sh in shapes:
        row = f"{sh:<20}"
        for d in out["kernels"].values():
            row += f"{d['per_shape_mean'][sh]:>10.4f}({d['per_shape_stdev_pct'][sh]:>4.1f})"
        print(row)

    print(f"\nComparisons vs {base}:")
    for name, d in out["kernels"].items():
        c = d.get("vs_base")
        if not c:
            continue
        verdict = ("REAL" if abs(c["t"]) >= 3 else
                   "likely" if abs(c["t"]) >= 2 else "NOT RESOLVED")
        print(f"  {name:<28} {c['rel_diff_pct']:>+7.2f}%  "
              f"se={c['se'] / d['mean'] * 100:>5.2f}%  t={c['t']:>+6.2f}  "
              f"shapes {c['shape_wins']}/{c['shape_total']}  -> {verdict}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare kernels with interleaved repeats so GPU drift cancels.")
    p.add_argument("reference", type=Path, help="Reference .py")
    p.add_argument("kernels", type=Path, nargs="+",
                   help="Kernels to compare; the FIRST is the baseline")
    p.add_argument("--reps", type=int, default=3, help="Interleaved repeats per kernel")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--dump", type=Path, default=None, help="Write JSON results here")
    a = p.parse_args()

    if len(a.kernels) < 2:
        p.error("need at least two kernels (baseline + candidate)")

    out = run(a.reference, a.kernels, a.reps, a.device, a.warmup, a.repeat, a.tol)
    _report(out)
    if a.dump:
        a.dump.write_text(json.dumps(out, indent=2))
        print(f"Saved -> {a.dump}")


if __name__ == "__main__":
    main()
