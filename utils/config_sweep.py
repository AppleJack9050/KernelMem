#!/usr/bin/env python
"""Enumerate every (variant, split_k) config the split-K autotuner can pick.

`ModelNew._tune` sweeps 16 candidates taking ONE 5-rep sample each and keeps the
raw minimum. Min-of-16-noisy-draws is the winner's curse: it selects the luckiest
reading, not the fastest config. That is the suspected cause of the 8.1% spread
seen across six fresh processes at a pinned clock.

This measures what the autotuner is trying to estimate, but properly:

* every config timed in EVERY round, round-robin, so thermal drift is common-mode
* paired log-ratio against the shipped (0,1) baseline, per round -> t-stat and CI
* numerical check of each config's output against (0,1), because split_k changes
  the reduction order

Output is the true ranking. Compare it against what `_tune` actually picks.

    python -m utils.config_sweep --rounds 60
    python -m utils.config_sweep --shape <uuid> --rounds 60
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("SOLBENCH_SRC", str(REPO / "third_party" / "SOL-ExecBench" / "src"))
KERNEL = REPO / "run/vae_block_002/kernel_autotune_splitk.py"
SHAPE = "cdb231f0-8b76-5b89-a93b-21af0627e037"      # b8 64x128, as in clock_drift_test
BASE = (0, 1)


def _load(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=SHAPE)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--reps", type=int, default=5, help="matches _tune's rep count")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--settle", type=int, default=1,
                    help="untimed reps before each timed block. >1 lets the "
                         "two-stream chunk pipeline reach steady state, which "
                         "round-robin switching otherwise prevents.")
    args_ns = ap.parse_args()

    import torch
    sys.path.insert(0, str(REPO))
    ref = _load(REPO / "ref_0.py", "_ref0")
    from sol_execbench.core import Workload
    from sol_execbench.core.bench.io import gen_inputs

    wkl_d = next(w for w in [ref._WORKLOAD] + ref._WORKLOAD_EXTRA
                 if w["uuid"] == args_ns.shape)
    wkl = Workload(**wkl_d)
    inp = gen_inputs(ref._DEFN, wkl, device="cpu", custom_inputs_fn=ref._CUSTOM_FN)
    dev = torch.device("cuda:0")
    inp = tuple(x.to(dev) if torch.is_tensor(x) else x for x in inp)

    kmod = _load(KERNEL, "_cand")
    m = kmod.ModelNew().to(dev).eval()
    ext, cands = m.ext, list(kmod.ModelNew._CANDIDATES)
    margin = kmod.ModelNew._MARGIN

    x = inp[0]
    print(f"shape {tuple(x.shape)}  uuid {args_ns.shape}")
    print(f"{len(cands)} configs x {args_ns.rounds} rounds x {args_ns.reps} reps, "
          f"round-robin interleaved")

    def time_cfg(v, sk, reps):
        ext.set_conv_override(v, sk)
        for _ in range(args_ns.settle):                # warm this config
            ext.fused_resblock(*inp)
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(reps):
            ext.fused_resblock(*inp)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / reps

    # ---- correctness: does split_k change the answer? -------------------------
    ext.set_conv_override(*BASE)
    ref_out = ext.fused_resblock(*inp).clone()
    err, alive = {}, []
    for (v, sk) in cands:
        try:
            ext.set_conv_override(v, sk)
            o = ext.fused_resblock(*inp)
            err[(v, sk)] = (o - ref_out).abs().max().item()
            alive.append((v, sk))
        except RuntimeError as exc:
            err[(v, sk)] = float("nan")
            print(f"  config {(v, sk)} unsupported: {str(exc)[:70]}")

    # ---- warmup, then round-robin --------------------------------------------
    for _ in range(args_ns.warmup):
        for (v, sk) in alive:
            time_cfg(v, sk, 2)

    samples = {c: [] for c in alive}
    t0 = time.perf_counter()
    for r in range(args_ns.rounds):
        for c in alive:                                # same order every round
            samples[c].append(time_cfg(c[0], c[1], args_ns.reps))
        if (r + 1) % 20 == 0:
            print(f"  round {r + 1}/{args_ns.rounds}  ({time.perf_counter() - t0:.0f}s)")

    # ---- paired stats vs the shipped baseline --------------------------------
    def paired(cfg):
        rs = [math.log(b / t) for b, t in zip(samples[BASE], samples[cfg])]
        n = len(rs)
        mu = st.mean(rs)
        sd = st.stdev(rs) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        return mu, se, (mu - 1.96 * se, mu + 1.96 * se)

    rows = []
    for c in alive:
        mu, se, ci = paired(c)
        rows.append((c, st.median(samples[c]), st.mean(samples[c]),
                     st.stdev(samples[c]) / st.mean(samples[c]) * 100,
                     min(samples[c]), mu * 100, ci[0] * 100, ci[1] * 100, err[c]))
    rows.sort(key=lambda r: r[1])

    print(f"\n{'cfg':>9}{'median':>9}{'mean':>9}{'CV%':>7}{'min':>9}"
          f"{'gain%':>8}{'  95% CI':>18}{'maxdiff':>10}")
    for c, med, mean, cv, mn, g, lo, hi, e in rows:
        tag = "  <-- shipped default" if c == BASE else ""
        print(f"{str(c):>9}{med:>9.4f}{mean:>9.4f}{cv:>7.2f}{mn:>9.4f}"
              f"{g:>8.2f}  [{lo:>6.2f},{hi:>6.2f}]{e:>10.2e}{tag}")

    best = rows[0][0]
    b_med = next(r[1] for r in rows if r[0] == BASE)
    print(f"\ntrue best by median: {best}  {rows[0][1]:.4f} ms "
          f"vs baseline {b_med:.4f} ms  ({(b_med / rows[0][1] - 1) * 100:+.2f}%)")

    # ---- what would _tune have done? -----------------------------------------
    # Replay its exact rule (min of one sample each, then the margin gates) on
    # every round independently: each round is a synthetic "fresh process".
    picks = {}
    for r in range(args_ns.rounds):
        base_t = samples[BASE][r]
        bt, bc = base_t, BASE
        for c in alive:
            if c == BASE:
                continue
            if samples[c][r] < bt:
                bt, bc = samples[c][r], c
        if bt > base_t * (1.0 - margin):
            bc = BASE
        picks[bc] = picks.get(bc, 0) + 1
    print(f"\nreplaying _tune's rule on each round independently "
          f"({args_ns.rounds} synthetic fresh processes, margin={margin}):")
    for c, n in sorted(picks.items(), key=lambda kv: -kv[1]):
        med = next(r[1] for r in rows if r[0] == c)
        print(f"  picks {str(c):>9}  {n:>3}/{args_ns.rounds} rounds "
              f"({n / args_ns.rounds * 100:>5.1f}%)   true median {med:.4f} ms")


if __name__ == "__main__":
    main()
