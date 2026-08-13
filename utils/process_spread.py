#!/usr/bin/env python
"""How much do two fresh processes disagree when NOTHING differs?

Same kernel source, same input tensors, same pinned (variant, split_k) -- so the
autotune lottery is removed -- same locked clock. Any spread left is pure
measurement variance attributable to process launch: CUDA context, allocator
state, extension load, whatever the scheduler feels like.

This separates two things that both look like "the number moved":
  * WITHIN-process CV  -- scatter between consecutive timings in one process
  * BETWEEN-process CV -- scatter between the settled means of separate processes

Run one process per invocation, then aggregate:

    for i in $(seq 8); do python -m utils.process_spread --tag p$i; done
    python -m utils.process_spread --report
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("SOLBENCH_SRC", str(REPO / "third_party" / "SOL-ExecBench" / "src"))
OUT = REPO / "run" / "process_spread"
KERNEL = REPO / "run/vae_block_002/kernel_autotune_splitk.py"

# all four SCORED shapes, cheapest first
SHAPES = [
    ("b2 64x64",     "8d631edd-1bc9-5142-9253-b0378a890e67", 400),
    ("b1 131x131",   "f1b799bf-831f-5434-98be-68e897f6a219", 400),
    ("b8 64x128",    "cdb231f0-8b76-5b89-a93b-21af0627e037", 250),
    ("b1 1024x1024", "38357eec-a997-567f-a5e4-07cb993c02f9",  25),
]
CONFIG = (0, 1)          # pinned: the autotuner never runs


def _load(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def measure(tag: str, blocks: int) -> None:
    import torch
    sys.path.insert(0, str(REPO))
    ref = _load(REPO / "ref_0.py", "_ref0")
    from sol_execbench.core import Workload
    from sol_execbench.core.bench.io import gen_inputs

    kmod = _load(KERNEL, "_cand")
    m = kmod.ModelNew().to(torch.device("cuda:0")).eval()
    ext = m.ext
    dev = torch.device("cuda:0")
    rec = {"tag": tag, "config": list(CONFIG), "shapes": {}}

    for label, uuid, reps in SHAPES:
        wkl = Workload(**next(w for w in [ref._WORKLOAD] + ref._WORKLOAD_EXTRA
                              if w["uuid"] == uuid))
        inp = gen_inputs(ref._DEFN, wkl, device="cpu", custom_inputs_fn=ref._CUSTOM_FN)
        inp = tuple(x.to(dev) if torch.is_tensor(x) else x for x in inp)
        ext.set_conv_override(*CONFIG)

        for _ in range(reps):                          # settle
            ext.fused_resblock(*inp)
        torch.cuda.synchronize()

        vals = []
        for _ in range(blocks):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            for _ in range(reps):
                ext.fused_resblock(*inp)
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) / reps)
        rec["shapes"][label] = vals
        print(f"  {tag} {label:<13} median {st.median(vals):8.4f} ms  "
              f"CV {st.stdev(vals) / st.mean(vals) * 100:5.2f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(rec))


def report() -> None:
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not recs:
        print("no runs found")
        return
    print(f"\n{len(recs)} fresh processes, config pinned to {tuple(recs[0]['config'])}, "
          f"identical inputs\n")
    print(f"{'shape':<14}{'within-CV%':>12}{'between-CV%':>13}{'spread%':>10}"
          f"{'min':>10}{'max':>10}")
    for label, _, _ in SHAPES:
        per = [r["shapes"][label] for r in recs if label in r["shapes"]]
        if len(per) < 2:
            continue
        within = st.mean([st.stdev(v) / st.mean(v) * 100 for v in per])
        meds = [st.median(v) for v in per]
        between = st.stdev(meds) / st.mean(meds) * 100
        print(f"{label:<14}{within:>12.2f}{between:>13.2f}"
              f"{(max(meds) / min(meds) - 1) * 100:>10.2f}{min(meds):>10.4f}{max(meds):>10.4f}")
    print("\nper-process medians")
    for label, _, _ in SHAPES:
        meds = [st.median(r["shapes"][label]) for r in recs if label in r["shapes"]]
        print(f"  {label:<14}" + " ".join(f"{v:8.4f}" for v in meds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        measure(a.tag or f"p{int(time.time())}", a.blocks)
