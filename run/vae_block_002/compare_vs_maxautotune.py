#!/usr/bin/env python3
"""Compare candidates against the torch.compile max-autotune baseline.

Each evaluate run co-measures the eager reference in its own session, so we
normalize every candidate by its own session reference before taking the ratio.
That cancels cross-session clock drift on an unlocked-clock desktop GPU:

    speedup_vs_MA = (t_MA / ref_MA) / (t_cand / ref_cand)

The raw ratio t_MA / t_cand is printed too, as a drift sanity check.
"""
import json
import pathlib
import statistics

BASE = pathlib.Path(__file__).parent

BASELINE = ("max-autotune", "out8_maxautotune")
# run A's fp16 candidate was retracted and its traces deleted: it computed both
# convs on cuDNN half tensor-core engines, exceeding the workload's max_atol
# (2.8e-3) on 18/20 shapes. It only ever passed because the framework's own gate
# ran at tol=1e-2. Not a valid candidate at this problem's declared fp32 precision.
CANDS = [
    ("agent8  fp32 (run A seed)", "out8_agent_seed"),
    ("agent8b fp32 (run B best)", "out8b_agent_fp32"),
    ("agent8c fp32 (run C best)", "out8c_agent_fp32"),
]


def load(out_dir):
    rows = {}
    p = BASE / out_dir / "traces.jsonl"
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


base = load(BASELINE[1])
print(f"Baseline: torch.compile(mode='max-autotune')   [{BASELINE[1]}]")
print(f"All speedups are RELATIVE TO max-autotune (>1 means the candidate wins)\n")

results = {}
for name, d in CANDS:
    cand = load(d)
    norm, raw, per_shape = [], [], []
    for u, b in base.items():
        c = cand.get(u)
        if not c or b["status"] != "PASSED" or c["status"] != "PASSED":
            continue
        if not (b["lat"] and c["lat"] and b["ref"] and c["ref"]):
            continue
        n = (b["lat"] / b["ref"]) / (c["lat"] / c["ref"])
        norm.append(n)
        raw.append(b["lat"] / c["lat"])
        per_shape.append((b["axes"], b["lat"], c["lat"], n))
    results[name] = (norm, raw, per_shape)
    print(f"{name:<22} n={len(norm):>2}  "
          f"geomean {statistics.geometric_mean(norm):.3f}x  "
          f"(raw {statistics.geometric_mean(raw):.3f}x)  "
          f"min {min(norm):.3f}x  max {max(norm):.3f}x")

name = CANDS[0][0]
print(f"\nPer-shape: {name} vs max-autotune (drift-normalized)\n")
print(f"  {'batch':>6}{'H':>6}{'W':>6}{'MA ms':>11}{'cand ms':>11}{'vs MA':>10}")
rows = sorted(results[name][2], key=lambda r: (r[0]["batch_size"], r[0]["height"]))
for ax, bl, cl, n in rows:
    print(f"  {ax['batch_size']:>6}{ax['height']:>6}{ax['width']:>6}"
          f"{bl:>11.3f}{cl:>11.3f}{n:>9.3f}x")
