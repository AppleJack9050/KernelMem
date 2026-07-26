#!/usr/bin/env python3
"""Layered SOL scores for problem 002.

S(T_k) = 1/(1 + (T_k - T_SOL)/(T_b - T_SOL)); S=0.5 at T_k=T_b, S=1 at T_k=T_SOL.

Two baselines, because the answer depends entirely on which one you mean:
  naive  - the problem's own reference.py (baselines.json)
  strong - fastest of {reference, channels_last eager, channels_last+torch.compile},
           measured in ONE session (baselines_strong.json). This is what upstream's
           sol_score docstring means by "an optimized PyTorch implementation".
"""
import json
import pathlib
import statistics

BASE = pathlib.Path(__file__).parent
CANDS = ["nhwc_eager", "nhwc_triton_fused", "compile_default",
         "compile_cudagraph", "compile_unlimited", "agent_best"]

# compile_unlimited is essentially the same code as the strong baseline's winning
# candidate, so scoring it against that baseline is circular (~0.5 by construction).
CIRCULAR_VS_STRONG = {"compile_unlimited", "nhwc_eager", "compile_default",
                      "compile_cudagraph"}

naive = json.loads((BASE / "baselines.json").read_text())
strong = json.loads((BASE / "baselines_strong.json").read_text())


def sol_score(t_k, t_b, t_sol):
    gap = t_b - t_sol
    if gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / gap)


hdr = (f"{'candidate':<20}{'pass':>7}{'SOL vs naive':>15}{'SOL vs strong':>16}"
       f"{'speedup vs strong':>20}")
print(hdr)
print("-" * len(hdr))

for c in CANDS:
    p = BASE / f"out_{c}" / "traces.jsonl"
    if not p.exists():
        continue
    s_naive, s_strong, spd = [], [], []
    n = 0
    for line in p.read_text().splitlines():
        t = json.loads(line)
        ev = t["evaluation"]
        lat = (ev.get("performance") or {}).get("latency_ms")
        if ev["status"] != "PASSED" or not lat:
            continue
        u = t["workload"]["uuid"]
        n += 1
        s_naive.append(sol_score(lat, naive[u]["t_b_ms"], naive[u]["t_sol_ms"]))
        s_strong.append(sol_score(lat, strong[u]["t_b_ms"], strong[u]["t_sol_ms"]))
        spd.append(strong[u]["t_b_ms"] / lat)
    if not n:
        continue
    mark = " *" if c in CIRCULAR_VS_STRONG else "  "
    print(f"{c:<20}{n:>4}/20{statistics.mean(s_naive):>15.3f}"
          f"{statistics.mean(s_strong):>14.3f}{mark}"
          f"{statistics.geometric_mean(spd):>19.3f}x")

print("\n* = circular against the strong baseline: this candidate is (near-)identical")
print("  code to the implementation that won the baseline, so ~0.5 is by construction.")
print("  Only nhwc_triton_fused and agent_best are independent of it.")
print("\nS=0.5 means matching the baseline. Not leaderboard-comparable: real SOL")
print("scoring uses NVIDIA's private T_b/T_SOL on locked-clock B200, and this T_SOL")
print("is a direct-convolution bound that Winograd/fp16 could legitimately beat.")
