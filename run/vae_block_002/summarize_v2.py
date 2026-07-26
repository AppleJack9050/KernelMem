#!/usr/bin/env python3
"""Compare candidates on problem 002 using the SAME-SESSION reference anchor.

The stored baselines.json t_b came from a separate session; on unlocked WSL2
clocks it drifts (~7% here), inflating every SOL score. Each trace carries its
own co-measured reference_latency_ms, so anchor on that and report the drift.
"""
import json
import pathlib
import statistics

BASE = pathlib.Path(__file__).parent
CANDS = ["nhwc_eager", "nhwc_triton_fused", "compile_default",
         "compile_cudagraph", "compile_unlimited", "agent_best"]

baselines = json.loads((BASE / "baselines.json").read_text())


def load(cand):
    rows = {}
    p = BASE / f"out_{cand}" / "traces.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        t = json.loads(line)
        ev = t["evaluation"]
        perf = ev.get("performance") or {}
        rows[t["workload"]["uuid"]] = {
            "status": ev["status"],
            "lat": perf.get("latency_ms"),
            "ref": perf.get("reference_latency_ms"),
            "axes": t["workload"]["axes"],
        }
    return rows


all_rows = {c: load(c) for c in CANDS}

print("Anchor comparison — speedup and SOL vs stored t_b (cross-session) "
      "and vs same-session reference\n")
hdr = f"{'candidate':<20}{'pass':>6}{'geo spd (stored)':>19}{'geo spd (same-sess)':>21}{'SOL (stored)':>14}{'SOL (same-sess)':>17}"
print(hdr)
print("-" * len(hdr))

drift_flags = []
for c in CANDS:
    sp_stored, sp_same, sol_stored, sol_same, n = [], [], [], [], 0
    for u, r in all_rows[c].items():
        if r["status"] != "PASSED" or not r["lat"] or not r["ref"]:
            continue
        n += 1
        b = baselines[u]
        sp_stored.append(b["t_b_ms"] / r["lat"])
        sp_same.append(r["ref"] / r["lat"])
        sol_stored.append(min(b["t_sol_ms"] / r["lat"], 1.0))
        # same-session anchor: scale t_sol comparison by measured reference
        sol_same.append(min(b["t_sol_ms"] / r["lat"], 1.0))
        drift = r["ref"] / b["t_b_ms"] - 1.0
        if abs(drift) > 0.05:
            drift_flags.append((c, json.dumps(r["axes"]), drift))
    if not n:
        continue
    print(f"{c:<20}{n:>4}/20"
          f"{statistics.geometric_mean(sp_stored):>18.3f}x"
          f"{statistics.geometric_mean(sp_same):>20.3f}x"
          f"{statistics.mean(sol_stored):>14.3f}"
          f"{statistics.mean(sol_same):>17.3f}")

print("\nNote: SOL columns share a numerator (analytic t_sol) so they coincide; "
      "the meaningful\ncorrection is the speedup anchor. Stored-anchor speedups "
      "are inflated where drift > 0.")

print(f"\nWorkloads where same-session reference deviates >5% from stored t_b: "
      f"{len(drift_flags)}")
by_cand = {}
for c, ax, d in drift_flags:
    by_cand.setdefault(c, []).append(d)
for c, ds in by_cand.items():
    print(f"  {c:<20} {len(ds):>2} workloads, median drift {statistics.median(ds)*100:+.1f}%")
