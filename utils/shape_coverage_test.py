#!/usr/bin/env python
"""Does the best kernel hold up on the workloads it was never scored on?

Why this exists
---------------
The search optimises four shapes -- b8 64x128 (the profiling target) plus
b2 64x64, b4 256x256 and b1 131x131 from `get_inputs_extra()`. But the problem
defines twenty workloads, and the two furthest from speed-of-light are not among
the four:

    b1 768x768     t_sol 13.278ms   t_b 26.874ms   2.02x off SOL
    b1 1024x1024   t_sol 23.606ms   t_b 46.529ms   1.97x off SOL

Those are the largest gaps in the whole set. If the kernel generalises, it
should score on them roughly as it does on the four it was tuned for. If it does
not, the search has been grinding on shapes where the conv is already near peak
while the real headroom sat outside its objective.

The scoring function makes this the right question to ask, because it is
SOL-relative, not PyTorch-relative:

    score = 1 / (1 + (t_k - t_sol) / (t_b - t_sol))

Baseline parity scores 0.5; reaching SOL scores 1.0. So "how far from SOL" is
literally the objective, and a ratio against PyTorch cannot see it -- both sides
lose the same wave quantisation and it cancels.

Method
------
Latencies are measured interleaved (candidate, reference, candidate, ...) in one
session so GPU drift is common-mode and cancels in the ratio, the same reason
`paired_bench.py` exists. Two numbers come back per shape:

* `score_stored` -- against the stored t_b/t_sol calibration. Comparable to the
  run's own reported scores, but t_b was measured in another session so it
  carries drift.
* `score_live`   -- t_b replaced by the eager reference measured here, right
  next to the candidate. Drift-free, but eager is a weaker baseline than the
  compiled/cudagraph t_b, so it flatters the kernel. The two bracket the truth.

What would falsify the hypothesis: the two unscored shapes coming back with
scores in line with the four scored ones. That is a real possible outcome and
the reason this is worth running rather than arguing about.

Run:  SOLBENCH_SRC=<repo>/third_party/SOL-ExecBench/src python -m utils.shape_coverage_test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# ref_*.py hardcodes a stale default (/home/elek/...) and dies at import without
# this, so set it before the import rather than relying on the caller's env.
os.environ.setdefault("SOLBENCH_SRC", str(REPO / "third_party" / "SOL-ExecBench" / "src"))

BEST = REPO / "run/20260804_best19_vae_block_002_claude_claude-opus-5/vae_block_002/code/kernel_20260804_101722.py"

# (label, uuid, in the search's objective?)
SHAPES = [
    ("b8 64x128",    "cdb231f0-8b76-5b89-a93b-21af0627e037", True),
    ("b2 64x64",     "8d631edd-1bc9-5142-9253-b0378a890e67", True),
    ("b4 256x256",   "952fec71-f323-5dab-9340-dad59ad7a3f1", True),
    ("b1 131x131",   "f1b799bf-831f-5434-98be-68e897f6a219", True),
    ("b1 768x768",   "8de1dc63-b34a-5618-ba36-abaa79de54cf", False),   # never scored
    ("b1 1024x1024", "38357eec-a997-567f-a5e4-07cb993c02f9", False),   # never scored
]


def sol_score(t_k: float, t_b: float, t_sol: float) -> float:
    gap = t_b - t_sol
    if gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / gap)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bench_paired(fn_a, fn_b, warmup: int, reps: int):
    """Interleaved A,B,A,B... so session drift is common-mode."""
    import torch
    for _ in range(warmup):
        fn_a(); fn_b()
    torch.cuda.synchronize()
    ta, tb = [], []
    for _ in range(reps):
        for fn, acc in ((fn_a, ta), (fn_b, tb)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            acc.append((time.perf_counter() - t0) * 1000.0)
    return ta, tb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=str(BEST))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()

    import torch

    # Timing the unscored shapes against stored scores only means something if
    # both were taken at the same clock.
    from utils import clock_lock
    try:
        clock_lock.ensure_locked(0, what="shape coverage")
    except clock_lock.ClockLockError as exc:
        print(f"\n[clock] {exc}\n")
        raise SystemExit(2)

    sys.path.insert(0, str(REPO))
    ref_mod = _load(REPO / "ref_0.py", "_ref0")
    from sol_execbench.core import Workload
    from sol_execbench.core.bench.io import gen_inputs

    tsol = json.loads((REPO / "run/vae_block_002/t_sol.json").read_text())
    base = json.loads((REPO / "run/vae_block_002/baselines.json").read_text())
    wl_by_uuid = {}
    for path in (REPO / "run/vae_block_002/baselines_fastest_pytorch.json",):
        pass

    kmod = _load(Path(a.kernel), "_cand")
    dev = torch.device("cuda:0")
    print(f"kernel: {Path(a.kernel).name}")
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    print(f"{'workload':<14} {'scored?':>8} {'t_sol':>8} {'t_b':>8} {'t_ref':>8} "
          f"{'t_kern':>8} {'vs eager':>9} {'score_stored':>13} {'score_live':>11}")

    rows = []
    for label, uuid, scored in SHAPES:
        t_sol = tsol[uuid]["t_sol_ms"]
        t_b = base[uuid]["t_b_ms"]
        # Rebuild the workload dict from the reference definition; only the axes
        # differ between workloads, and gen_inputs needs the full Workload.
        axes = None
        for w in [ref_mod._WORKLOAD] + ref_mod._WORKLOAD_EXTRA:
            if w["uuid"] == uuid:
                axes = w
                break
        if axes is None:
            b, hw = label.split()
            h, w_ = hw.split("x")
            axes = dict(ref_mod._WORKLOAD)
            axes = json.loads(json.dumps(axes))
            axes["uuid"] = uuid
            axes["axes"] = {"batch_size": int(b[1:]), "height": int(h), "width": int(w_)}
        wkl = Workload(**axes)

        try:
            inp = gen_inputs(ref_mod._DEFN, wkl, device="cpu",
                             custom_inputs_fn=ref_mod._CUSTOM_FN)
            inp = tuple(x.to(dev) if torch.is_tensor(x) else x for x in inp)
            ref = ref_mod.Model().to(dev).eval()
            cand = kmod.ModelNew().to(dev).eval()
            with torch.no_grad():
                out_r = ref(*inp)
                out_c = cand(*inp)
                err = (out_r - out_c).abs().max().item()
                tk, tr = _bench_paired(lambda: cand(*inp), lambda: ref(*inp),
                                       a.warmup, a.reps)
            t_kern, t_ref = st.median(tk), st.median(tr)
            s_stored = sol_score(t_kern, t_b, t_sol)
            s_live = sol_score(t_kern, t_ref, t_sol)
            mark = "yes" if scored else "NO"
            print(f"{label:<14} {mark:>8} {t_sol:>8.3f} {t_b:>8.3f} {t_ref:>8.3f} "
                  f"{t_kern:>8.3f} {t_ref/t_kern:>8.3f}x {s_stored:>13.3f} {s_live:>11.3f}")
            rows.append((label, scored, s_stored, s_live, t_ref / t_kern, err))
        except torch.OutOfMemoryError as e:
            print(f"{label:<14} {'NO':>8}  OOM: {str(e)[:60]}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{label:<14} {'?':>8}  FAILED {type(e).__name__}: {str(e)[:80]}")
            torch.cuda.empty_cache()
        finally:
            for n in ("inp", "ref", "cand", "out_r", "out_c"):
                if n in dir():
                    pass
            torch.cuda.empty_cache()

    if rows:
        sc = [r for r in rows if r[1]]
        un = [r for r in rows if not r[1]]
        print("\n=== VERDICT ===")
        if sc:
            print(f"scored shapes   n={len(sc)}  mean score_stored {st.mean([r[2] for r in sc]):.3f}"
                  f"   mean speedup vs eager {st.mean([r[4] for r in sc]):.3f}x")
        if un:
            print(f"UNSCORED shapes n={len(un)}  mean score_stored {st.mean([r[3-1] for r in un]):.3f}"
                  f"   mean speedup vs eager {st.mean([r[4] for r in un]):.3f}x")
        if sc and un:
            d = st.mean([r[2] for r in sc]) - st.mean([r[2] for r in un])
            print(f"\ngap (scored - unscored) = {d:+.3f} score")
            print("A large positive gap means the kernel is tuned to its objective and the"
                  "\ntwo big-headroom workloads were left on the table. Near zero falsifies"
                  "\nthe hypothesis: the kernel generalises and the scoring set is fine.")
        print(f"\nmax abs err per shape: " +
              "  ".join(f"{r[0]}={r[5]:.2e}" for r in rows))


if __name__ == "__main__":
    main()
