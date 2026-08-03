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

Rank on ABSOLUTE time, never on ``score``
-----------------------------------------
``score`` is ``T_ref / T_k``, and each kernel's ``T_ref`` is measured separately.
A kernel that slows the reference down therefore outranks a kernel that is
genuinely faster. This is not theoretical: on 2026-07-31 this tool reported
vae_block_002 round 6 as "+9.35%, t=+7.33, REAL" over round 3, when round 6 was
in fact 1.28x SLOWER in absolute time. Round 6 raised the persisting-L2
reservation to 31.25 MB of the H100's 50 MB L2 and never released it, so its
reference lost two thirds of the cache -- while round 3, measured afterwards in
the same process, lost the cache with no access-policy window to win it back.
The polluter was flattered twice and its rival penalised.

So: ``test_ms`` is the metric. It has no denominator to corrupt. ``score`` is
still recorded, for reading only, never for ranking.

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
from typing import Any, Callable, Dict, List, Optional

from utils.compile_and_run import compare_and_bench


def _geo(xs: List[float]) -> float:
    """Geometric mean, so no single shape dominates by being large."""
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


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


def adaptive_paired_verdict(
    reference: Path,
    base_py: Path,
    cand_py: Path,
    *,
    device: int = 0,
    warmup: int = 25,
    repeat: int = 100,
    tol: float = 1e-2,
    margin: float = 0.005,
    min_reps: int = 3,
    max_reps: int = 8,
    escalate_k: float = 3.0,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Decide whether ``cand_py`` beats ``base_py`` by ``margin``, measuring BOTH now.

    Why this exists
    ---------------
    The ratchet compares a candidate measured this minute against ``base_score``,
    a float stored when the base last advanced -- possibly hours earlier and never
    re-taken. That number goes stale fast. Measured here on an RTX 5090, one
    byte-identical kernel read 1.2014 ms and then 1.2141 ms half an hour later:
    **+1.06%, twice the 0.5% margin that decides whether the base advances**. So
    a meaningful share of "did this improve" was really "which way did the GPU
    drift since the base was measured".

    Re-running the base in the same session removes that term. The base and the
    candidate are interleaved -- base, cand, base, cand -- rather than run in
    blocks, because a clock ramp part-way through a block reads as a difference
    between the kernels rather than as drift. The reps in one of today's runs
    climbed 1.2065 -> 1.2210 monotonically inside a single session; blocked
    sampling would have charged all of that to whichever kernel ran second.

    Adaptive reps
    -------------
    Most rounds are not close and do not need precision: of the seven measured
    candidates in rounds 13-20 of the exp3 run, six landed at -0.66% to -11.06%
    and one at +0.49%. So sample ``min_reps`` first and stop as soon as the
    DECISION is safe -- what matters is distance from the margin, not from zero,
    since +5% and -5% are both unambiguous against a 0.5% gate while +0.49% sits
    on the line. Only the genuinely close calls pay for ``max_reps``.

    Returns None if measurement fails, so the caller can fall back to the stored
    score rather than lose the round.

    The returned ``rel_pct`` is the candidate's advantage in ABSOLUTE kernel time,
    never in ``score``: ``score`` is ``T_ref / T_k`` with a separately measured
    ``T_ref``, so a candidate that slows the reference down outranks one that is
    genuinely faster (see the module docstring for the case where that reported
    "+9.35%, REAL" for a kernel 1.28x slower).
    """
    say = log or (lambda _m: None)
    base_ms: List[float] = []
    cand_ms: List[float] = []

    def _one(path: Path) -> float:
        # Diagnostic path: measure and report a leaking kernel rather than refuse
        # it. compare_and_bench still restores device state after every call, so
        # one kernel's leak cannot reach the next one measured here.
        res = compare_and_bench(reference, path, device_idx=device, warmup=warmup,
                                repeat=repeat, tol=tol, reject_on_state_leak=False)
        return _geo([p["test_ms"] for p in res["per_shape"]])

    def _stats() -> tuple:
        # Welch on speed (1/time) so "positive" keeps meaning "candidate better".
        sb = [1.0 / t for t in base_ms]
        sc = [1.0 / t for t in cand_ms]
        w = _welch(sc, sb)
        mean_b = st.mean(sb)
        return w["rel_diff_pct"], (w["se"] / mean_b * 100.0 if mean_b else float("nan")), w["t"]

    try:
        while len(base_ms) < max_reps:
            base_ms.append(_one(base_py))
            cand_ms.append(_one(cand_py))
            if len(base_ms) < min_reps:
                continue
            rel, se, _t = _stats()
            if not (se > 0):        # zero variance -- more reps cannot help
                break
            if abs(rel - margin * 100.0) >= escalate_k * se:
                break
    except Exception as exc:
        say(f"[base] paired re-measure failed ({exc.__class__.__name__}: {exc}); "
            f"falling back to the stored base score")
        return None

    if len(base_ms) < min_reps:
        return None

    rel, se, t = _stats()
    return {
        "rel_pct": rel,
        "se_pct": se,
        "t": t,
        "reps": len(base_ms),
        "escalated": len(base_ms) > min_reps,
        "base_ms": st.mean(base_ms),
        "cand_ms": st.mean(cand_ms),
        "base_ms_all": base_ms,
        "cand_ms_all": cand_ms,
        "margin_pct": margin * 100.0,
        "beats_margin": rel >= margin * 100.0,
        # "Resolved" is about the DECISION, not the effect: a +0.49% that cannot
        # be told apart from a 0.50% gate is unresolved even though the kernel
        # plainly changed something. Rounds 13-20 reported every such case to the
        # next round as a failure, which is why one of them re-ran a mechanism
        # that had in fact been the run's best kernel.
        "resolved": abs(rel - margin * 100.0) >= 2.0 * se if se > 0 else True,
    }


def run(reference: Path, kernels: List[Path], reps: int, device: int,
        warmup: int, repeat: int, tol: float) -> Dict[str, Any]:
    # Interleave: rep-major, so every kernel is sampled once before any is
    # sampled twice. This is what makes the drift common-mode.
    samples: Dict[str, List[Dict[str, Any]]] = {k.stem: [] for k in kernels}
    for r in range(reps):
        for k in kernels:
            # Diagnostic tool: measure and REPORT a leaking kernel rather than
            # refuse it. compare_and_bench still restores state after each call,
            # so one kernel's leak cannot reach the next one measured here.
            res = compare_and_bench(reference, k, device_idx=device,
                                    warmup=warmup, repeat=repeat, tol=tol,
                                    reject_on_state_leak=False)
            samples[k.stem].append(res)
            ms = _geo([p["test_ms"] for p in res["per_shape"]])
            leak = res.get("state_leak")
            print(f"  rep {r + 1}/{reps}  {k.stem:<28} {ms:8.4f} ms"
                  + (f"   [leaked: {'; '.join(leak)}]" if leak else ""), flush=True)

    shapes = [s["shape"] for s in samples[kernels[0].stem][0]["per_shape"]]
    out: Dict[str, Any] = {"reference": str(reference), "reps": reps,
                           "shapes": shapes, "kernels": {}}

    for k in kernels:
        name = k.stem
        # ABSOLUTE candidate time per shape -- never `speedup`, which carries a
        # separately-measured T_ref in its denominator. See the module docstring.
        per_shape = {sh: [next(p["test_ms"] for p in r["per_shape"] if p["shape"] == sh)
                          for r in samples[name]] for sh in shapes}
        # One number per rep, so reps are the sampling unit for the t-test.
        times = [_geo([p["test_ms"] for p in r["per_shape"]]) for r in samples[name]]
        # Welch runs on speed (1/time) so that "positive" keeps meaning "better".
        speeds = [1.0 / t for t in times]
        mean_ms, mean_sp = st.mean(times), st.mean(speeds)
        sd = st.stdev(times) if len(times) > 1 else 0.0
        leaks = sorted({c for r in samples[name] for c in (r.get("state_leak") or [])})
        out["kernels"][name] = {
            "path": str(k),
            "times_ms": times,
            "speeds": speeds,
            "mean_ms": mean_ms,
            "mean_speed": mean_sp,
            "stdev_pct": sd / mean_ms * 100.0 if mean_ms else float("nan"),
            "sem_pct": (sd / math.sqrt(len(times))) / mean_ms * 100.0 if mean_ms else float("nan"),
            "state_leak": leaks,
            "scores": [r["score"] for r in samples[name]],   # recorded, NOT ranked on
            "per_shape_ms": {sh: st.mean(v) for sh, v in per_shape.items()},
            "per_shape_stdev_pct": {
                sh: (st.stdev(v) / st.mean(v) * 100.0 if len(v) > 1 else 0.0)
                for sh, v in per_shape.items()
            },
        }

    # Every kernel after the first is compared against the first.
    base = kernels[0].stem
    base_speeds = out["kernels"][base]["speeds"]
    for k in kernels[1:]:
        name = k.stem
        cmp = _welch(out["kernels"][name]["speeds"], base_speeds)
        # Sign test across shapes. Per-shape noise is independent while drift
        # moves all shapes together, so "won on N of M shapes" survives drift
        # that any magnitude comparison on the geomean would fail.
        # A win is now a SHORTER time, hence '<'.
        wins = [sh for sh in shapes
                if out["kernels"][name]["per_shape_ms"][sh]
                < out["kernels"][base]["per_shape_ms"][sh]]
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
    print("Ranked on ABSOLUTE kernel time (geomean over shapes) -- LOWER is better.")
    print(f"{'kernel':<30}{'geomean ms':>12}{'stdev':>8}{'sem':>8}")
    for name, d in out["kernels"].items():
        print(f"{name:<30}{d['mean_ms']:>12.4f}{d['stdev_pct']:>7.2f}%{d['sem_pct']:>7.2f}%")

    print(f"\nPer-shape mean time in ms (noise in parentheses, as stdev %):")
    print(f"{'shape':<20}" + "".join(f"{n[-13:]:>16}" for n in out["kernels"]))
    for sh in shapes:
        row = f"{sh:<20}"
        for d in out["kernels"].values():
            row += f"{d['per_shape_ms'][sh]:>10.4f}({d['per_shape_stdev_pct'][sh]:>4.1f})"
        print(row)

    leaky = {n: d["state_leak"] for n, d in out["kernels"].items() if d.get("state_leak")}
    if leaky:
        print("\nDevice-state leaks (a leaking kernel can distort its RIVAL's timing):")
        for n, cs in leaky.items():
            print(f"  {n:<28} {'; '.join(cs)}")

    print(f"\nComparisons vs {base}   (positive = FASTER than baseline):")
    for name, d in out["kernels"].items():
        c = d.get("vs_base")
        if not c:
            continue
        verdict = ("REAL" if abs(c["t"]) >= 3 else
                   "likely" if abs(c["t"]) >= 2 else "NOT RESOLVED")
        print(f"  {name:<28} {c['rel_diff_pct']:>+7.2f}%  "
              f"se={c['se'] / d['mean_speed'] * 100:>5.2f}%  t={c['t']:>+6.2f}  "
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
