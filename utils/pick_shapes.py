#!/usr/bin/env python
"""Pick the extra scored shapes for any SOL-ExecBench problem: smallest, middle, largest.

The rule
--------
`get_inputs()` stays whatever the problem packager chose -- it is the sole
ncu/nsys profiling target and moving it would invalidate every profile. This
picks `get_inputs_extra()`, and the rule is fixed for every problem:

    smallest   the least total work in the suite
    middle     the workload nearest the GEOMETRIC midpoint of the size range
    largest    the most total work in the suite

Geometric, not arithmetic. Suite size ranges span two orders of magnitude (128x
on problem 002), so the arithmetic mean of smallest and largest lands
three-quarters of the way up and "middle" stops meaning middle. On a log axis
the three picks sit at 0%, ~50% and 100% of the range by construction.

Why coverage rather than difficulty
-----------------------------------
Difficulty cannot be known before a kernel exists, and the obvious proxies
actively mislead: on 002 the workload furthest from speed-of-light (b1 768x768,
2.02x off) turned out to be one the kernel handled easily, while the hardest
(b4 256x256, closing 23% of its gap against 59-68% elsewhere) sat at below-median
distance. Correlation between PyTorch's own optimiser-responsiveness and measured
difficulty was r=+0.15, i.e. nothing. So the set is chosen to span the space
evenly and let the search discover difficulty itself.

The awkward-extent constraint
-----------------------------
One shape must have a spatial extent divisible by no tile size kernels reach for
(8/16/32/64). This is a correctness trap, not a coverage slot: kernels derive a
grid from one rounding of a division and per-block work from another, and the
two agree only when the tile divides the extent exactly. Problem 002 shipped a
GroupNorm whose partial-sum buffer had an unwritten tail whenever H*W % 16 != 0,
and every shape scored during that search was a multiple of 16.

So `middle` prefers an awkward workload when one lies near the midpoint --
folding the trap into a slot that was needed anyway. If none is close, the tool
says so loudly rather than silently shipping a set that cannot see tail bugs.

Usage
-----
    python -m utils.pick_shapes solbench_problems/L1/002_vae_conv3x3_groupnorm_silu_residual_fused
    python -m utils.pick_shapes <problem_dir> --emit     # ready-to-paste _WORKLOAD_EXTRA

Changing a scored set makes scores incomparable across the change -- on 002 a
single swap moved the score +6.4% with no kernel change. Do it once, early, then
freeze.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TILE_SIZES = 64          # awkward := spatial extent not divisible by this
AWKWARD_WINDOW = 20.0    # how far from the midpoint an awkward shape may sit (log %)


def _resolve_axes(defn: Dict[str, Any], wkl: Dict[str, Any]) -> Dict[str, int]:
    """Axis name -> value, taking `var` axes from the workload and `const` from the definition."""
    out: Dict[str, int] = {}
    for name, spec in (defn.get("axes") or {}).items():
        if spec.get("type") == "const":
            out[name] = int(spec["value"])
    for name, val in (wkl.get("axes") or {}).items():
        out[name] = int(val)
    return out


def _output_elems(defn: Dict[str, Any], axes: Dict[str, int]) -> Optional[int]:
    """Element count of the primary output, resolved through the axis table."""
    outs = defn.get("outputs") or {}
    if not outs:
        return None
    shape = (list(outs.values())[0] or {}).get("shape")
    if not shape:
        return None
    n = 1
    for dim in shape:
        if isinstance(dim, int):
            n *= dim
        elif dim in axes:
            n *= axes[dim]
        else:
            return None
    return n


def _batch_axis(defn: Dict[str, Any]) -> Optional[str]:
    for name in (defn.get("axes") or {}):
        if "batch" in name.lower():
            return name
    return None


def _spatial_extent(defn: Dict[str, Any], axes: Dict[str, int]) -> Optional[int]:
    """Product of the var axes that are not batch -- i.e. the per-sample spatial size.

    Channels are excluded deliberately: they are const and usually a multiple of
    64, which would mask a ragged spatial extent behind a tidy total.
    """
    batch = _batch_axis(defn)
    prod, seen = 1, False
    for name, spec in (defn.get("axes") or {}).items():
        if spec.get("type") != "var" or name == batch:
            continue
        if name not in axes:
            return None
        prod *= axes[name]
        seen = True
    return prod if seen else None


def load(problem_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    defn = json.loads((problem_dir / "definition.json").read_text())
    wkls = [json.loads(l) for l in (problem_dir / "workload.jsonl").read_text().splitlines() if l.strip()]
    return defn, wkls


def find_t_sol(wkls: List[Dict[str, Any]], explicit: Optional[Path] = None) -> Dict[str, float]:
    """uuid -> t_sol_ms, the analytic compute roofline for each workload.

    This is the size metric we actually want. `t_sol` is FLOPs divided by the
    device's peak rate -- on problem 002 it comes out to exactly 104.8 TFLOPS for
    all 20 workloads -- so ranking by it ranks by COMPUTATIONAL WORK. Output
    element count is only a proxy, and an unreliable one: it coincides with FLOPs
    only when every axis that scales the arithmetic also scales the output. A
    GEMM with a var reduction axis breaks that (FLOPs ~ M*N*K, output ~ M*N), and
    so does any problem with a variable channel count.

    Searched rather than required, because t_sol lives with the run artifacts
    (run/<task>/t_sol.json) rather than with the problem definition.
    """
    want = {w["uuid"] for w in wkls}
    cands = [explicit] if explicit else sorted(Path("run").glob("*/t_sol.json"))
    for p in cands:
        if not p or not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        hit = {k: v["t_sol_ms"] for k, v in d.items()
               if k in want and isinstance(v, dict) and "t_sol_ms" in v}
        if len(hit) == len(want):          # only accept a table covering every workload
            return hit
    return {}


def analyse(defn: Dict[str, Any], wkls: List[Dict[str, Any]],
            t_sol: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Rank by computational work: t_sol when available, output elements otherwise."""
    rows = []
    batch = _batch_axis(defn)
    t_sol = t_sol or {}
    for w in wkls:
        axes = _resolve_axes(defn, w)
        n = t_sol.get(w["uuid"]) or _output_elems(defn, axes)
        if not n:
            continue
        spatial = _spatial_extent(defn, axes)
        label = "".join(
            [f"b{axes[batch]} " if batch and batch in axes else ""]
            + ["x".join(str(axes[a]) for a, s in (defn.get("axes") or {}).items()
                        if s.get("type") == "var" and a != batch and a in axes)]
        ) or w["uuid"][:8]
        rows.append({"uuid": w["uuid"], "label": label, "elems": n, "raw": w,
                     "spatial": spatial,
                     "awkward": (spatial is not None and spatial % TILE_SIZES != 0)})
    rows.sort(key=lambda r: r["elems"])
    lo, hi = rows[0]["elems"], rows[-1]["elems"]
    span = math.log(hi) - math.log(lo) if hi > lo else 1.0
    for r in rows:
        r["pos"] = (math.log(r["elems"]) - math.log(lo)) / span * 100.0
    return rows


def pick(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    smallest, largest = rows[0], rows[-1]

    # Middle = nearest the geometric midpoint, preferring an awkward shape when one
    # sits close enough -- that folds the correctness trap into a slot already needed.
    cands = [r for r in rows if r is not smallest and r is not largest]
    if not cands:
        return [smallest, largest], ["fewer than 3 workloads; returned what exists"]
    awk_near = [r for r in cands if r["awkward"] and abs(r["pos"] - 50.0) <= AWKWARD_WINDOW]
    if awk_near:
        middle = min(awk_near, key=lambda r: abs(r["pos"] - 50.0))
        notes.append(f"middle: chose the awkward '{middle['label']}' at {middle['pos']:.0f}% "
                     f"(spatial {middle['spatial']} not divisible by {TILE_SIZES}) -- "
                     f"folds the tail-bug trap into the middle slot")
    else:
        middle = min(cands, key=lambda r: abs(r["pos"] - 50.0))
        notes.append(f"middle: nearest the geometric midpoint at {middle['pos']:.0f}%")

    chosen = [smallest, middle, largest]
    if not any(r["awkward"] for r in chosen):
        awk = [r for r in rows if r["awkward"]]
        if awk:
            best = min(awk, key=lambda r: abs(r["pos"] - 50.0))
            notes.append(f"WARNING: no chosen shape has an awkward extent. Nearest is "
                         f"'{best['label']}' at {best['pos']:.0f}%, outside the "
                         f"+-{AWKWARD_WINDOW:.0f}pt window. Consider adding it as a 4th: a "
                         f"set of tile-aligned shapes cannot see ragged-tail bugs.")
        else:
            notes.append("note: no workload in this suite has an awkward extent; "
                         "the tail-bug trap is unavailable here.")
    return chosen, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("problem_dir", type=Path)
    ap.add_argument("--emit", action="store_true", help="print a ready-to-paste _WORKLOAD_EXTRA block")
    ap.add_argument("--t-sol", type=Path, default=None,
                    help="path to a t_sol.json; auto-searched under run/*/ when omitted")
    a = ap.parse_args()

    defn, wkls = load(a.problem_dir)
    tsol = find_t_sol(wkls, a.t_sol)
    rows = analyse(defn, wkls, tsol)
    metric = "t_sol ms (compute roofline)" if tsol else "output elements (PROXY -- no t_sol found)"

    if not rows:
        raise SystemExit("could not resolve any workload sizes from this problem")
    chosen, notes = pick(rows)
    ids = {r["uuid"] for r in chosen}

    print(f"problem : {defn.get('name')}")
    print(f"ranked by: {metric}")
    print(f"workloads: {len(rows)}   range {rows[0]['elems']:,.4g} .. {rows[-1]['elems']:,.4g} "
          f"({rows[-1]['elems']/rows[0]['elems']:.0f}x)\n")
    if not tsol:
        print("  WARNING: no t_sol table covering every workload was found, so this ranks by")
        print("  output elements. That equals computational work only when every axis scaling")
        print("  the arithmetic also scales the output -- false for a var reduction axis.\n")
    print(f"{'workload':<16} {'work':>13} {'log-pos':>8} {'awkward':>8}  pick")
    for r in rows:
        tag = ("SMALLEST" if r is chosen[0] else "MIDDLE" if r is chosen[1]
               else "LARGEST" if r is chosen[-1] else "")
        print(f"{r['label']:<16} {r['elems']:>13,.4g} {r['pos']:>7.0f}% "
              f"{'YES' if r['awkward'] else '':>8}  {tag}")
    print("\n" + "\n".join(f"* {n}" for n in notes))
    cov = sorted(r["pos"] for r in chosen)
    print(f"\ncoverage at {', '.join(f'{c:.0f}%' for c in cov)} of the log-size range")

    if a.emit:
        print("\n# ---- paste into ref_*.py, replacing _WORKLOAD_EXTRA ----")
        print("_WORKLOAD_EXTRA = [")
        for r, tag in zip(chosen, ("smallest", "middle", "largest")):
            extra = " (awkward extent: divisible by no tile size)" if r["awkward"] else ""
            print(f"    # {tag}: {r['label']}, {r['pos']:.0f}% of the log-size range{extra}")
            print(f"    json.loads('{json.dumps(r['raw'])}'),")
        print("]")


if __name__ == "__main__":
    main()
