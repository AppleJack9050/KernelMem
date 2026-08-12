#!/usr/bin/env python
"""Fit the MCGS mechanism prior from saved optimization trees.

    python scripts/build_mechanism_prior.py --task vae_block_002 \
        --out priors/vae_block_002.json --max_round 9

    python main_memory_latest.py tasks/vae_block_002.py --search mcgs \
        --mcgs_prior priors/vae_block_002.json

What it fits
------------
One number per optimization mechanism: the mean relative gain of the edges that
applied it, minus the global mean. That is the whole model. A state-conditioned
version (JitRL Eq. 4-7, kNN over the code-feature vector) was measured and buys
nothing on this data -- top-third win rate 18% against a 19% base -- because the
feature vector carries about two bits.

Measured leave-one-run-out at the defaults (86 edges, 10 runs, rounds 0-9):
rho = +0.324, top-third win rate 58% vs a 47% base. For comparison, a node's own
score predicts the gain it yields at rho = +0.02, which is why selection had no
signal to work with before this.

Why --max_round matters
-----------------------
Win rate over the saved runs is 42% in rounds 0-9 and 0% from round 10 on. Fit
over all rounds and the prior partly re-learns that cliff: rho = +0.485 with the
late rounds in, +0.324 with only rounds 0-9. MCGS already handles the cliff with
--mcgs_max_depth, so fitting over everything credits it twice and inflates what
the prior looks like it is contributing. Default is 9 for that reason.

Honest scope
------------
The vocabulary is per-task -- l2_cache_blocking's value on a VAE conv block says
nothing about a reduction kernel -- so fit one prior per task and do NOT reuse
one across tasks. Coverage on vae_block_002 is 67%: 40 distinct names over 86
edges in the fitted window, 26 seen exactly once. Where the prior has no estimate
it returns 0.0, which leaves selection where it was rather than guessing.

--report runs the leave-one-RUN-out evaluation that justified the default
parameters, so a refit can be checked instead of trusted.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mcgs import MechanismPrior  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _finite_pos(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def collect(run_dir: Path, task: str, max_round: int) -> List[dict]:
    """(run, mechanism, gain%, round) for every scored edge that names a method."""
    out: List[dict] = []
    for t in sorted(run_dir.rglob("optimization_tree.json")):
        try:
            d = json.loads(t.read_text(encoding="utf-8"))
        except Exception:
            continue
        K = d.get("kernels") or {}
        if not K or Path(str(d.get("task") or "")).stem != task:
            continue
        for name, v in K.items():
            p = v.get("parent")
            if not p or p not in K:
                continue
            if not (_finite_pos(v.get("speedup")) and _finite_pos(K[p].get("speedup"))):
                continue
            strat = v.get("strategy") if isinstance(v.get("strategy"), dict) else None
            mech = (strat or {}).get("method_name")
            if not mech or not str(mech).strip():
                continue
            rnd = v.get("round") if isinstance(v.get("round"), int) else -1
            if max_round >= 0 and rnd > max_round:
                continue
            # The run is the first path component under run_dir. Indexing from the
            # END instead (t.parts[-4]) collapses every flat run/<name>/<task>/
            # path onto the literal "run" and merges unrelated sessions into one
            # holdout group -- which made leave-one-run-out hold out most of the
            # corpus at once, dropping coverage to 29% and the correlation to
            # -0.171. Lineage children under one <stamp>_lineages folder stay
            # grouped, which is right: they are one session sharing a GPU and a
            # seed decision.
            try:
                run_id = t.relative_to(run_dir).parts[0]
            except ValueError:
                run_id = t.parts[1] if len(t.parts) > 1 else str(t)
            out.append({"run": run_id,
                        "mech": str(mech).strip(),
                        "gain": (v["speedup"] / K[p]["speedup"] - 1.0) * 100.0,
                        "round": rnd})
    return out


def _spearman(X: List[float], Y: List[float]) -> float:
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for m in range(i, j + 1):
                r[o[m]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(X), rank(Y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def report(edges: List[dict], kappa: float, min_support: int) -> None:
    """Leave-one-RUN-out. Not leave-one-edge-out: edges inside a run share a
    lineage, so holding out only the edge leaks its own ancestry."""
    runs = sorted({e["run"] for e in edges})
    if len(runs) < 2:
        print("[prior] only one run available; cannot cross-validate.")
        return
    preds, truth = [], []
    for held in runs:
        mem = [(e["mech"], e["gain"]) for e in edges if e["run"] != held]
        # min_support=1 for the CV memory on purpose. The deployed prior drops
        # n=1 mechanisms because one sample says nothing about a mean; but
        # applying that filter to a leave-one-run-out MEMORY also silences every
        # mechanism that happens to appear once in the other nine runs, which
        # measures the filter's coverage rather than the prior's accuracy. Report
        # both so the split is visible.
        pr = MechanismPrior.fit(mem, kappa=kappa, min_support=1)
        for e in (x for x in edges if x["run"] == held):
            if pr.support(e["mech"]) <= 0:
                continue                       # silent, not wrong
            preds.append(pr.advantage(e["mech"]))
            truth.append(e["gain"])
    if len(preds) < 10:
        print(f"[prior] too few scorable edges to evaluate ({len(preds)}).")
        return
    order = sorted(range(len(preds)), key=lambda i: -preds[i])
    top = order[: max(1, len(order) // 3)]
    base = sum(1 for g in truth if g > 1) / len(truth)
    prec = sum(1 for i in top if truth[i] > 1) / len(top)
    print(f"[prior] leave-one-run-out: n={len(preds)} of {len(edges)} edges "
          f"(coverage {len(preds) / len(edges) * 100:.0f}%)")
    print(f"[prior]   rho(advantage, held-out gain) = {_spearman(preds, truth):+.3f}")
    print(f"[prior]   top-third win rate {prec * 100:.0f}% vs base {base * 100:.0f}%   "
          f"median gain {st.median([truth[i] for i in top]):+.2f}% vs "
          f"{st.median(truth):+.2f}%")
    if _spearman(preds, truth) < 0.1:
        print("[prior]   WARNING: little or no signal. Do not enable --mcgs_prior "
              "on this fit; it would add machinery that predicts nothing.")


def main() -> None:
    ap = argparse.ArgumentParser("Fit the MCGS mechanism prior from run history")
    ap.add_argument("--task", default="vae_block_002",
                    help="Task stem to fit on. The mechanism vocabulary is "
                         "task-specific; never reuse a prior across tasks.")
    ap.add_argument("--run_dir", type=Path, default=ROOT / "run")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write the prior JSON (default priors/<task>.json)")
    ap.add_argument("--max_round", type=int, default=9,
                    help="Ignore edges past this round. Rounds 10+ have a 0%% win "
                         "rate, so including them makes the prior re-learn the "
                         "exhaustion cliff that --mcgs_max_depth already handles. "
                         "Set -1 to keep every round.")
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="Shrinkage: an estimate keeps n/(n+kappa) of its raw "
                         "advantage. Defaults to 0 (off) because the sweep says so: "
                         "leave-one-run-out rho is +0.324 at kappa=0 and falls "
                         "monotonically to +0.118 at kappa=10. Shrinking by support "
                         "damages this prior because the n=3 mechanisms are genuinely "
                         "good (3/3 wins) while the n=18 one is genuinely mediocre, so "
                         "support is not a proxy for reliability here. --min_support "
                         "is the correction that does help.")
    ap.add_argument("--tau", type=float, default=2.0,
                    help="Softmax temperature over advantages, in percent units.")
    ap.add_argument("--min_support", type=int, default=2,
                    help="Drop mechanisms seen fewer times than this. One "
                         "observation says nothing about a mean when the "
                         "per-edge spread is several percent.")
    ap.add_argument("--repeat_penalty", type=float, default=0.25)
    ap.add_argument("--report", action="store_true",
                    help="Cross-validate the fit and print the numbers.")
    args = ap.parse_args()

    edges = collect(args.run_dir, args.task, args.max_round)
    if not edges:
        print(f"[prior] no scored, method-named edges found for task {args.task} "
              f"under {args.run_dir}. Nothing to fit.")
        return

    runs = sorted({e["run"] for e in edges})
    counts = Counter(e["mech"] for e in edges)
    print(f"[prior] {len(edges)} edges from {len(runs)} run(s), "
          f"{len(counts)} distinct mechanisms "
          f"({sum(1 for c in counts.values() if c == 1)} seen once)")
    print(f"[prior] win rate (>+1%): "
          f"{sum(1 for e in edges if e['gain'] > 1) / len(edges) * 100:.0f}%   "
          f"median gain {st.median(e['gain'] for e in edges):+.2f}%")

    if args.report:
        report(edges, args.kappa, args.min_support)

    prior = MechanismPrior.fit(
        [(e["mech"], e["gain"]) for e in edges],
        kappa=args.kappa, tau=args.tau, min_support=args.min_support,
        repeat_penalty=args.repeat_penalty,
        note=(f"task={args.task} runs={len(runs)} edges={len(edges)} "
              f"max_round={args.max_round} kappa={args.kappa} "
              f"min_support={args.min_support}"))

    print(f"[prior] fitted {len(prior.table)} mechanisms "
          f"(global mean gain {prior.global_mean:+.2f}%)")
    print(f"[prior] top by shrunk advantage:")
    for m, a, n in prior.ranked(8):
        print(f"           {m[:46]:<46} {a:+6.2f}%  n={n}")
    worst = sorted(((m, a, n) for m, (a, n) in prior.table.items()), key=lambda t: t[1])[:5]
    if worst:
        print(f"[prior] bottom:")
        for m, a, n in worst:
            print(f"           {m[:46]:<46} {a:+6.2f}%  n={n}")

    out = args.out or (ROOT / "priors" / f"{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prior.to_dict(), indent=2), encoding="utf-8")
    print(f"[prior] wrote {out}")
    print(f"[prior] use it with:  --search mcgs --mcgs_prior {out}")


if __name__ == "__main__":
    main()
