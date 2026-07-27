#!/usr/bin/env python3
"""SOL Score with torch.compile(max-autotune) as the scoring baseline T_b.

Implements the metric from SOL-ExecBench (arXiv:2603.19173), Section 4.2-4.3:

    T_SOL = max( FLOPs / compute_throughput , fused_bytes / memory_bandwidth )   (Eq. 1)

    S(Tk) = (Tb - T_SOL) / ( (Tk - T_SOL) + (Tb - T_SOL) )                       (Eq. 3)

Anchors:  Tk = Tb  -> S = 0.5      (matches the scoring baseline)
          Tk = SOL -> S = 1.0      (hardware speed-of-light)
          Tk -> inf -> S -> 0

NVIDIA holds the official T_b internal (paper Sec 4.5) and generates it with agents
"restricted to producing solutions using only PyTorch and standard Python packages" —
the same class as torch.compile max-autotune, so using it as T_b here is consistent
with their methodology, though absolute scores are NOT leaderboard-comparable
(different GPU, unlocked clocks).

Per-problem suite score is the arithmetic mean of per-workload S (paper Eq. 4).
"""
import json
import pathlib
import statistics

BASE = pathlib.Path(__file__).parent

# ---- RTX 5090 hardware constants -------------------------------------------
BW_BYTES_S = 1792e9        # 1792 GB/s GDDR7
PEAK_TF32 = 104.8e12       # dense TF32/FP32 tensor  (what an fp32 kernel may use)
PEAK_FP16 = 209.5e12       # dense FP16 w/ FP32 accumulate

# The problem is declared float32 end-to-end and the harness flags fp32->fp16
# downcasting as reward hacking (paper Table 3), so TF32 is the honest ceiling:
# it is what the reference itself runs (torch.backends.cudnn.allow_tf32 = True).
PEAK = PEAK_TF32

C, KS, GROUPS = 256, 3, 32
BASELINE_DIR = "out8_maxautotune"
# run A's fp16 candidate was retracted and its traces deleted (see the audit
# guards in sol_score below, which is what flagged it: measured faster than the
# fp32 SOL bound because it was not computing in fp32). The guards are kept so
# any future precision-downgrade candidate is caught the same way.
CANDIDATES = [
    ("run A fp32 seed", "out8_agent_seed"),
    ("run B fp32 best", "out8b_agent_fp32"),
    ("run C fp32 best", "out8c_agent_fp32"),
]


def sol_ms(B, H, W):
    """T_SOL for one workload, per Eq. 1."""
    flops = 2 * 2 * B * C * C * KS * KS * H * W        # two 3x3 convs, 2 FLOP/MAC
    # Fused byte count: with perfect fusion the block reads x once, writes out
    # once, and reads both conv weights. Intermediates stay on-chip.
    act = B * C * H * W * 4
    weights = 2 * (C * C * KS * KS * 4) + 4 * C * 4
    fused_bytes = 2 * act + weights
    return max(flops / PEAK, fused_bytes / BW_BYTES_S) * 1e3


def load(out_dir):
    rows = {}
    for line in (BASE / out_dir / "traces.jsonl").read_text().splitlines():
        t = json.loads(line)
        ev = t["evaluation"]
        perf = ev.get("performance") or {}
        a = t["workload"]["axes"]
        rows[t["workload"]["uuid"]] = {
            "axes": a,
            "status": ev["status"],
            "lat": perf.get("latency_ms"),
            "ref": perf.get("reference_latency_ms"),
        }
    return rows


def sol_score(tk, tb, tsol):
    """Eq. 3, with the paper's Sec 4.3 audit guards.

    The metric assumes Tb > T_SOL and Tk >= T_SOL. A kernel measured FASTER than
    the speed-of-light bound cannot be physically valid at the problem's declared
    precision, so the paper treats it as an audit signal for SOLAR bound review
    and reward-hacking inspection rather than awarding a score. Returning a number
    here would be meaningless: the denominator crosses zero and S explodes.
    """
    if tk is None or tb is None:
        return None, "missing"
    if tb <= tsol:
        return None, "AUDIT: baseline at/below SOL"
    if tk < tsol:
        return None, "AUDIT: faster than SOL"
    return gap_score(tk, tb, tsol), None


def gap_score(tk, tb, tsol):
    gap = tb - tsol
    return gap / ((tk - tsol) + gap)


base = load(BASELINE_DIR)

print(f"SOL Score with T_b = torch.compile(mode='max-autotune')")
print(f"T_SOL = max(FLOPs/{PEAK/1e12:.1f}e12, fused_bytes/{BW_BYTES_S/1e9:.0f}e9)")
print(f"S=0.5 means matching max-autotune; S=1.0 means hardware speed-of-light\n")

results = {}
for name, d in CANDIDATES:
    cand = load(d)
    scores, rows, audits = [], [], []
    for uuid, b in base.items():
        c = cand.get(uuid)
        if not c or b["status"] != "PASSED" or c["status"] != "PASSED":
            continue
        a = b["axes"]
        tsol = sol_ms(a["batch_size"], a["height"], a["width"])
        # normalize the candidate into the baseline's session using each run's own
        # co-measured eager reference, cancelling cross-session clock drift
        tk = c["lat"] * (b["ref"] / c["ref"])
        s, flag = sol_score(tk, b["lat"], tsol)
        if flag:
            audits.append((a, tk, tsol, flag))
            continue
        scores.append(s)
        rows.append((a, b["lat"], tk, tsol, s))
    results[name] = (scores, rows)
    if audits:
        print(f"{name:<34} *** {len(audits)}/{len(audits)+len(scores)} workloads "
              f"FLAGGED: {audits[0][3]} ***")
        worst = min(audits, key=lambda r: r[1] / r[2])
        print(f"{'':<34}     worst: {worst[1]:.3f} ms vs SOL {worst[2]:.3f} ms "
              f"= {worst[2]/worst[1]:.2f}x faster than the fp32 bound")
    if scores:
        print(f"{name:<34} S = {statistics.fmean(scores):.3f}   "
              f"(min {min(scores):.3f}, max {max(scores):.3f}, n={len(scores)})")

print(f"\n{'max-autotune (T_b, by definition)':<34} S = 0.500")

name = CANDIDATES[-1][0]
print(f"\nPer-workload: {name}\n")
print(f"  {'batch':>5}{'H':>5}{'W':>5}{'T_b ms':>9}{'T_k ms':>9}{'T_SOL':>9}{'S':>8}")
for a, tb, tk, tsol, s in sorted(results[name][1],
                                 key=lambda r: (r[0]["batch_size"], r[0]["height"])):
    print(f"  {a['batch_size']:>5}{a['height']:>5}{a['width']:>5}"
          f"{tb:>9.3f}{tk:>9.3f}{tsol:>9.3f}{s:>8.3f}")
