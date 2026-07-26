#!/usr/bin/env python3
"""Aggregate per-candidate traces into one comparison table for problem 002."""
import json
import pathlib
import statistics

BASE = pathlib.Path(__file__).parent
CANDS = ["nhwc_eager", "nhwc_triton_fused", "compile_default", "compile_cudagraph", "compile_unlimited", "agent_best"]

baselines = json.loads((BASE / "baselines.json").read_text())


def load(cand):
    rows = {}
    p = BASE / f"out_{cand}" / "traces.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        t = json.loads(line)
        ev = t["evaluation"]
        uuid = t["workload"]["uuid"]
        lat = (ev.get("performance") or {}).get("latency_ms")
        rows[uuid] = {
            "status": ev["status"],
            "lat_ms": lat,
            "axes": t["workload"]["axes"],
        }
    return rows


all_rows = {c: load(c) for c in CANDS}
uuids = list(baselines.keys())

print(f"{'axes':<38}{'t_b':>8}{'t_sol':>8}", end="")
for c in CANDS:
    print(f"{c[:16]:>18}", end="")
print()

geo = {c: [] for c in CANDS}
sol = {c: [] for c in CANDS}
npass = {c: 0 for c in CANDS}
for u in uuids:
    b = baselines[u]
    ax = None
    line = ""
    for c in CANDS:
        r = all_rows[c].get(u)
        if r and ax is None:
            ax = r["axes"]
        if not r or r["status"] != "PASSED" or not r["lat_ms"]:
            line += f"{'FAIL' if r else '-':>18}"
            continue
        npass[c] += 1
        sp = b["t_b_ms"] / r["lat_ms"]
        s = min(b["t_sol_ms"] / r["lat_ms"], 1.0)
        geo[c].append(sp)
        sol[c].append(s)
        line += f"{r['lat_ms']:>8.3f} {sp:>4.2f}x {s:>4.2f}".rjust(18)
    axs = json.dumps(ax or {})
    print(f"{axs:<38}{b['t_b_ms']:>8.3f}{b['t_sol_ms']:>8.3f}{line}")

print()
for c in CANDS:
    if geo[c]:
        g = statistics.geometric_mean(geo[c])
        ms = statistics.mean(sol[c])
        print(f"{c:<22} passed {npass[c]:>2}/20  geomean speedup vs T_b: {g:.3f}x   mean SOL score: {ms:.3f}   median SOL: {statistics.median(sol[c]):.3f}")
    else:
        print(f"{c:<22} passed {npass[c]:>2}/20")
