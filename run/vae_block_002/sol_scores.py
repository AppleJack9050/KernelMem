#!/usr/bin/env python3
"""Per-candidate SOL scores using the harness's real formula.

S(T_k) = 1 / (1 + (T_k - T_SOL)/(T_b - T_SOL));  S=0.5 at T_k=T_b, S=1 at T_k=T_SOL.

Reported under two anchors for T_b:
  stored      - baselines.json (measured in an earlier session; drifts on WSL2)
  same-sess   - the reference latency co-measured inside each candidate's own run
"""
import json
import pathlib
import statistics

BASE = pathlib.Path(__file__).parent
CANDS = ["nhwc_eager", "nhwc_triton_fused", "compile_default",
         "compile_cudagraph", "compile_unlimited", "agent_best"]

baselines = json.loads((BASE / "baselines.json").read_text())


def sol_score(t_k, t_b, t_sol):
    gap = t_b - t_sol
    if gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / gap)


hdr = f"{'candidate':<20}{'pass':>7}{'SOL (stored t_b)':>19}{'SOL (same-session)':>21}{'worst shape':>28}"
print(hdr)
print("-" * len(hdr))

for c in CANDS:
    p = BASE / f"out_{c}" / "traces.jsonl"
    if not p.exists():
        continue
    stored, same, worst = [], [], (2.0, None)
    n = 0
    for line in p.read_text().splitlines():
        t = json.loads(line)
        ev = t["evaluation"]
        perf = ev.get("performance") or {}
        lat, ref = perf.get("latency_ms"), perf.get("reference_latency_ms")
        if ev["status"] != "PASSED" or not lat or not ref:
            continue
        n += 1
        b = baselines[t["workload"]["uuid"]]
        s_stored = sol_score(lat, b["t_b_ms"], b["t_sol_ms"])
        s_same = sol_score(lat, ref, b["t_sol_ms"])
        stored.append(s_stored)
        same.append(s_same)
        if s_same < worst[0]:
            worst = (s_same, t["workload"]["axes"])
    if not n:
        continue
    ax = worst[1]
    axs = f"b{ax['batch_size']} {ax['height']}x{ax['width']}" if ax else "-"
    print(f"{c:<20}{n:>4}/20{statistics.mean(stored):>19.3f}"
          f"{statistics.mean(same):>21.3f}{f'{worst[0]:.3f} @ {axs}':>28}")

print("\nS = 0.5 means 'matched the PyTorch reference'. Above 0.5 = faster, below = slower.")
print("Not leaderboard-comparable: real SOL scoring uses NVIDIA's private T_b and T_SOL")
print("on locked-clock B200, and the T_SOL here is a direct-convolution bound.")
