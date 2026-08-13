#!/usr/bin/env python
"""Measure the harness's own measurement noise: SAME kernel file, SAME workload.

Nothing here varies except time. The kernel source, the reference, the shapes,
the seed and the warmup/repeat counts are byte-identical across every call, so
any spread in the reported numbers is measurement noise and nothing else.

Three nested levels are worth telling apart, because the optimization loop is
exposed to different ones in different places:

  L1  call-to-call inside ONE process   -- what ``adaptive_paired_verdict``
                                           samples when it interleaves reps
  L2  process-to-process                -- adds extension load, allocator and
                                           context setup; this is what a fresh
                                           round sees
  L3  the same, spread over time        -- adds clock/thermal drift, the term
                                           that makes a stored base score stale

Every call records the raw per-rep event times for the primary shape as well as
the aggregates, plus the SM clock / power / temperature sampled immediately
after, so a spread can be attributed rather than merely reported.

Usage
-----
    python -m utils.noise_probe --ref tasks/vae_block_002.py \
        --kernel run/vae_block_002/kernels/agent_best.py \
        --calls 20 --out noise_L1.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _geo(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _gpu_state(bus_filter: str | None = None) -> Dict[str, Any]:
    """SM clock / power / temperature / utilisation, straight from nvidia-smi.

    Sampled per call so that a slow call can be checked against the clock it ran
    at instead of being written off as unexplained noise.
    """
    try:
        q = ("index,pci.bus_id,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
             "utilization.gpu,clocks_throttle_reasons.active")
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        rows = []
        for line in out.strip().splitlines():
            f = [c.strip() for c in line.split(",")]
            if bus_filter and bus_filter not in f[1]:
                continue
            rows.append({"index": f[0], "bus": f[1], "sm_mhz": f[2], "mem_mhz": f[3],
                         "power_w": f[4], "temp_c": f[5], "util_pct": f[6],
                         "throttle": f[7] if len(f) > 7 else ""})
        return {"gpus": rows}
    except Exception as exc:                     # never let telemetry kill a run
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--time_ref", action="store_true",
                    help="Also time the reference, so score/speedup are populated. "
                         "Doubles the GPU time; needed to see score noise.")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to idle between calls (L3: let the clock relax).")
    ap.add_argument("--tag", default="L1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import torch
    from utils.compile_and_run import compare_and_bench

    props = torch.cuda.get_device_properties(args.device)
    # 114 SMs = H100 PCIe, 132 = H100 NVL. Printed so a run can never be
    # silently attributed to the wrong card.
    header = {"tag": args.tag, "pid": os.getpid(), "device": args.device,
              "gpu_name": props.name, "sm_count": props.multi_processor_count,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "kernel": args.kernel, "ref": args.ref,
              "warmup": args.warmup, "repeat": args.repeat,
              "time_ref": bool(args.time_ref),
              "started": datetime.now().isoformat(timespec="seconds")}
    out_path = Path(args.out)
    with out_path.open("a") as fh:
        fh.write(json.dumps({"type": "header", **header}) + "\n")
    print(f"[probe] {props.name} ({props.multi_processor_count} SM) "
          f"pid={os.getpid()} tag={args.tag}", flush=True)

    t_start = time.time()
    for i in range(args.calls):
        if i and args.sleep:
            time.sleep(args.sleep)
        t0 = time.time()
        res = compare_and_bench(
            Path(args.ref), Path(args.kernel), device_idx=args.device,
            warmup=args.warmup, repeat=args.repeat, tol=args.tol,
            reject_on_state_leak=False, time_ref=bool(args.time_ref),
        )
        wall = time.time() - t0
        per_shape = [{"shape": s["shape"], "test_ms": s["test_ms"],
                      "ref_ms": s["ref_ms"], "speedup": s["speedup"]}
                     for s in res["per_shape"]]
        rec = {
            "type": "call", "tag": args.tag, "pid": os.getpid(), "call": i,
            "t_since_start_s": round(time.time() - t_start, 3),
            "wall_s": round(wall, 3),
            "geo_test_ms": _geo([s["test_ms"] for s in res["per_shape"]]),
            "score": res["score"],
            "per_shape": per_shape,
            # Raw per-rep CUDA-event times for the PRIMARY shape: separates
            # within-measurement jitter from call-to-call offset.
            "primary_reps_ms": res["test_latency_ms"]["all"],
            "max_abs_err": res["max_abs_err"],
            "gpu": _gpu_state(),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        msg = (f"  call {i + 1}/{args.calls}  geo_test={rec['geo_test_ms']:.4f} ms"
               f"  wall={wall:.1f}s")
        if rec["score"] is not None:
            msg += f"  score={rec['score']:.4f}"
        print(msg, flush=True)

    # A quick summary so the run is readable without post-processing.
    with out_path.open() as fh:
        vals = [json.loads(l)["geo_test_ms"] for l in fh
                if json.loads(l).get("type") == "call"
                and json.loads(l).get("tag") == args.tag]
    if len(vals) > 1:
        m, sd = st.mean(vals), st.stdev(vals)
        print(f"[probe] n={len(vals)} mean={m:.4f} ms sd={sd:.4f} ms "
              f"cv={100 * sd / m:.3f}%  range={min(vals):.4f}-{max(vals):.4f} ms "
              f"({100 * (max(vals) / min(vals) - 1):.3f}% spread)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
