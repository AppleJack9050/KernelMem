#!/usr/bin/env python
"""A/A test: how often does the harness call noise an improvement?

Times ONE kernel against ITSELF, interleaved, for a fixed wall-clock window.
Every "improvement" it finds is a false positive by construction -- the two
sides are the same code on the same inputs. That makes three things measurable
that nothing else measures:

* drift    -- slope of latency against wall time. `verify_chain` records
              +0.9..+1.7% over 30 min on this box; this quantifies it directly.
* spread   -- coefficient of variation between measurements.
* FPR      -- the fraction of 5-rep windows where `_paired_stats` clears the
              same sigma gate `main_memory_latest.py` uses to accept a new base.
              This is the number that matters: it is the rate at which the
              search promotes nothing.

Run it once with clocks free and once with them pinned
(`sudo nvidia-smi -lgc 2407,2407`) and the delta is what clock locking buys.
2407 MHz is chosen because it is the supported clock nearest the 2.41 GHz the
SOL model assumes, so pinning there also makes measured times directly
comparable to t_sol instead of drifting relative to it.

    python -m utils.clock_drift_test --tag unlocked --seconds 240
    sudo nvidia-smi -lgc 2407,2407
    python -m utils.clock_drift_test --tag locked   --seconds 240
    python -m utils.clock_drift_test --compare
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("SOLBENCH_SRC", str(REPO / "third_party" / "SOL-ExecBench" / "src"))
OUT = REPO / "run" / "clock_drift"
KERNEL = REPO / "run/vae_block_002/kernel_autotune_splitk.py"
# b8 64x128: ~1.5ms, so a 4-minute window yields hundreds of samples. Big enough
# to be a real workload, small enough that the sample count carries the statistics.
SHAPE = "cdb231f0-8b76-5b89-a93b-21af0627e037"


def _load(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def collect(tag: str, seconds: int, inner: int) -> None:
    import torch
    sys.path.insert(0, str(REPO))
    ref = _load(REPO / "ref_0.py", "_ref0")
    from sol_execbench.core import Workload
    from sol_execbench.core.bench.io import gen_inputs

    wkl = Workload(**next(w for w in [ref._WORKLOAD] + ref._WORKLOAD_EXTRA
                          if w["uuid"] == SHAPE))
    inp = gen_inputs(ref._DEFN, wkl, device="cpu", custom_inputs_fn=ref._CUSTOM_FN)
    dev = torch.device("cuda:0")
    inp = tuple(x.to(dev) if torch.is_tensor(x) else x for x in inp)

    kmod = _load(KERNEL, "_cand")
    # TWO independent instances of the SAME class. Byte-identical work, so any
    # difference between them is measurement error and nothing else.
    a = kmod.ModelNew().to(dev).eval()
    b = kmod.ModelNew().to(dev).eval()

    def once(m):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            m(*inp)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / inner

    with torch.no_grad():
        for _ in range(20):                     # absorb the split-K autotune search
            a(*inp); b(*inp)
        torch.cuda.synchronize()
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{tag}.jsonl"
        t_start = time.time()
        n = 0
        with path.open("w") as f:
            while time.time() - t_start < seconds:
                # Interleaved A,B so drift is common-mode within a pair, exactly
                # as paired_bench does it.
                ta, tb = once(a), once(b)
                f.write(json.dumps({"t": time.time() - t_start, "a": ta, "b": tb}) + "\n")
                n += 1
                if n % 25 == 0:
                    f.flush()
                    print(f"  {tag}: {n} pairs, {time.time()-t_start:.0f}s, "
                          f"last {ta:.4f}/{tb:.4f} ms")
    clk = os.popen("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader").read().strip()
    print(f"{tag}: {n} pairs written to {path}   (SM clock now {clk})")


def _p_to_sigma(p: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(1.0 - p)


def analyse(tag: str, sigma: float, window: int) -> dict | None:
    from utils.paired_bench import _paired_stats, _t_sf, _sigma_to_p
    path = OUT / f"{tag}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) < window * 2:
        return None
    A = [r["a"] for r in rows]
    B = [r["b"] for r in rows]
    T = [r["t"] for r in rows]
    allv = A + B

    # drift: least-squares slope of latency vs wall time, as %/minute
    mt, mv = st.mean(T), st.mean(A)
    num = sum((t - mt) * (v - mv) for t, v in zip(T, A))
    den = sum((t - mt) ** 2 for t in T)
    slope = (num / den) if den else 0.0                      # ms per second
    drift_pct_min = slope * 60.0 / mv * 100.0
    total_drift = (slope * (max(T) - min(T))) / mv * 100.0

    # false positives: how many disjoint windows clear the acceptance gate
    p_target = _sigma_to_p(sigma)
    fp = tot = 0
    for i in range(0, len(rows) - window, window):
        s = _paired_stats(A[i:i + window], B[i:i + window])
        tot += 1
        if s["se_log"] > 0 and s["dof"] >= 1:
            # one-sided: candidate faster than base by any margin
            if s["mean_log"] > 0 and _t_sf(s["t"], s["dof"]) <= p_target:
                fp += 1
    return {"tag": tag, "n_pairs": len(rows), "mean_ms": mv,
            "cv_pct": st.stdev(allv) / st.mean(allv) * 100.0,
            "drift_pct_per_min": drift_pct_min, "total_drift_pct": total_drift,
            "range_pct": (max(allv) - min(allv)) / st.mean(allv) * 100.0,
            "fp": fp, "windows": tot, "fpr_pct": fp / tot * 100.0 if tot else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seconds", type=int, default=240)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=5, help="reps per verdict, matching --base_reps")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.tag:
        collect(a.tag, a.seconds, a.inner)
    if a.compare or a.tag:
        print(f"\n{'condition':<12}{'pairs':>7}{'mean ms':>9}{'CV %':>7}{'range %':>9}"
              f"{'drift %/min':>13}{'total drift %':>14}{'FALSE POSITIVES':>17}")
        for tag in ("unlocked", "locked"):
            r = analyse(tag, a.sigma, a.window)
            if not r:
                continue
            print(f"{tag:<12}{r['n_pairs']:>7}{r['mean_ms']:>9.4f}{r['cv_pct']:>7.2f}"
                  f"{r['range_pct']:>9.2f}{r['drift_pct_per_min']:>13.3f}"
                  f"{r['total_drift_pct']:>14.2f}"
                  f"{r['fp']}/{r['windows']} = {r['fpr_pct']:.1f}%".rjust(17))
        print(f"\nFALSE POSITIVES = windows of {a.window} where the same code beat itself at "
              f"{a.sigma} sigma.\nEvery one is the search accepting a kernel that changed nothing.")


if __name__ == "__main__":
    main()
