#!/usr/bin/env python3
"""Summarize SOL-ExecBench traces for arbitrary candidate out-dirs.

Usage: python3 summarize_agent8.py out_dir [out_dir ...]

Reports, per candidate, the geometric-mean speedup anchored on the
*same-session* reference latency carried in each trace (the stored
baselines.json t_b drifts across sessions and inflates results), plus the
per-shape breakdown so a schedule that only wins on the bound workload is
visible.
"""
import json
import pathlib
import statistics
import sys

BASE = pathlib.Path(__file__).parent
baselines = json.loads((BASE / "baselines.json").read_text())


def load(out_dir):
    rows = []
    p = pathlib.Path(out_dir) / "traces.jsonl"
    if not p.exists():
        raise SystemExit(f"no traces.jsonl in {out_dir}")
    for line in p.read_text().splitlines():
        t = json.loads(line)
        ev = t["evaluation"]
        perf = ev.get("performance") or {}
        rows.append({
            "uuid": t["workload"]["uuid"],
            "status": ev["status"],
            "lat": perf.get("latency_ms"),
            "ref": perf.get("reference_latency_ms"),
            "axes": t["workload"]["axes"],
        })
    return rows


def main(out_dirs):
    for d in out_dirs:
        rows = load(d)
        name = pathlib.Path(d).name
        ok = [r for r in rows if r["status"] == "PASSED" and r["lat"] and r["ref"]]
        sp = [r["ref"] / r["lat"] for r in ok]
        sol = []
        for r in ok:
            b = baselines.get(r["uuid"])
            if b:
                sol.append(min(b["t_sol_ms"] / r["lat"], 1.0))
        print(f"\n=== {name} ===")
        print(f"pass {len(ok)}/{len(rows)}")
        if not ok:
            bad = {r["status"] for r in rows}
            print(f"  statuses: {bad}")
            continue
        print(f"geomean speedup (same-session anchor): "
              f"{statistics.geometric_mean(sp):.4f}x")
        print(f"min {min(sp):.3f}x  max {max(sp):.3f}x")
        if sol:
            print(f"mean SOL (indicative, local t_sol): {statistics.mean(sol):.3f}")
        print(f"\n  {'axes':<46}{'ref ms':>10}{'ker ms':>10}{'speedup':>10}")
        for r in sorted(ok, key=lambda r: r["ref"]):
            ax = json.dumps(r["axes"], separators=(",", ":"))
            print(f"  {ax:<46}{r['ref']:>10.3f}{r['lat']:>10.3f}"
                  f"{r['ref'] / r['lat']:>9.3f}x")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
