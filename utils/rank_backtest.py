#!/usr/bin/env python
"""Backtest: can a model rank sibling proposals by their measured outcome?

Why this exists
---------------
The open question is whether a cheap model can triage branches -- look at two
proposed optimisations and say which one is worth Opus's time -- before either
is implemented. That is a *ranking* task, not a prediction task: it needs the
ordering right, not the magnitude. Which matters, because the magnitude channel
is already known to be broken. `verify_chain.py` records three mechanisms Opus 5
claimed at +0.90%, +1.12% and +1.18% that measured 0.09-0.17% on paired
re-ablation, i.e. the frontier model mis-sized its own proposals by ~10x.

Ordering is a much lower bar, and it is answerable from data already on disk.
Every `optimization_tree.json` stores the proposal (`strategy`) that produced
each kernel AND the speedup that kernel went on to measure. Wherever one parent
has two or more children, a branch choice was actually made and its outcome is
recorded. Show a model the two proposals with the outcomes withheld, ask which
wins, and score it.

No GPU, no compilation, no CUDA. This is replay: JSON reads plus API calls.
It runs while the card is busy.

Three things this is careful about
----------------------------------
1. POSITION BIAS. Every pair is asked twice, in both orders. Accuracy is
   averaged over the two, which cancels a constant preference for whichever
   option is shown first. The disagreement rate between the two orders is
   reported as `consistency`, and it is the single most useful number here: it
   needs no ground truth at all, and a model at ~50% consistency is flipping a
   coin no matter what its accuracy looks like.

2. THE GROUND TRUTH IS ALSO NOISY. Siblings were measured in different rounds,
   so the recorded gap between them carries the same +0.9..+1.7% drift that
   `verify_chain.py` documents. A quarter of the pairs here sit at or below that
   drift, which means their "winner" is partly a coin flip and no ranker can be
   scored against them. So the headline metric is restricted to pairs whose gap
   is comfortably above drift (`--decisive`, default 3%), and the full
   accuracy-vs-margin curve is printed so the cut can be inspected rather than
   trusted.

3. THE PAIRS ARE NOT INDEPENDENT. A parent with 7 children yields 21 pairs from
   7 underlying kernels. Treating those as 21 independent trials would roughly
   halve the error bars. Confidence intervals come from a bootstrap that
   resamples whole sibling groups, which is the honest unit. The naive binomial
   SE is printed alongside purely to show how much it would have flattered us.

Free baselines
--------------
A model arm is only interesting if it beats what costs nothing:

* `headroom`     -- the proposer's own low/medium/high self-assessment, already
                    stored on every proposal. Beating this is the minimum bar:
                    if it ranks as well as a model, the ranking is already free.
* `claimed_gain` -- believe the largest percentage the proposal claims for
                    itself in `expected_metric_change`. Given the verify_chain
                    finding this is expected to be weak, which is the point.
* `longer`       -- the longer proposal wins. A degenerate-signal control: any
                    arm that fails to beat prose length is not reasoning about
                    CUDA, it is reasoning about verbosity.

`ceilings()` from utils/lineage.py would be the strongest free baseline, but it
needs ncu-derived inputs and only 23 of the 32 nodes with a recorded profile
still have the CSV on disk. Re-profiling those 9 needs the GPU. Left as a
follow-up rather than scored on a third of the data.

Usage
-----
    python -m utils.rank_backtest build
    python -m utils.rank_backtest run --arm haiku
    python -m utils.rank_backtest run --arm sonnet
    python -m utils.rank_backtest run --arm opus
    python -m utils.rank_backtest score

`run` caches every answer to disk keyed by (arm, pair, order) and skips what it
already has, so it is interruptible and resumable at no cost.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import statistics as st
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "run" / "rank_backtest"

# An arm is a (model, effort) pair, not just a model. Effort is a real variable
# here: it decides whether the model thinks at all, which drove a 10x latency
# spread across models and is the knob most likely to move accuracy. Arms
# suffixed `_max` re-run the same model at max effort so the two can be compared
# on identical pairs -- otherwise "would more thinking help?" is unanswerable.
ARMS = {
    "haiku":      "claude-haiku-4-5",
    "sonnet":     "claude-sonnet-5",
    "opus":       "claude-opus-5",
    "sonnet_max": "claude-sonnet-5",
    "opus_max":   "claude-opus-5",
}

# query_server reads effort from this env var at call time (not import time), so
# setting it per-run is enough; there is no per-call override -- its
# `reasoning_effort` parameter is accepted and then ignored.
EFFORT_ENV = "KERNELMEM_CLAUDE_EFFORT"


def _arm_effort(arm: str) -> str:
    return "max" if arm.endswith("_max") else "high"

# Fields shown to the ranker. `headroom`/`confidence` are deliberately withheld
# -- they are the proposer's own verdict, so including them would let the ranker
# read the answer off the page instead of reasoning. They are scored separately
# as the `headroom` baseline.
SHOW_FIELDS = [
    ("bottleneck", "Bottleneck identified"),
    ("primary_optimisation_method", "Optimisation method"),
    ("optimisation method", "Optimisation method"),
    ("method_name", "Method name"),
    ("modification_plan", "Modification plan"),
    ("modification plan", "Modification plan"),
    ("evidence", "Evidence"),
    ("expected_metric_change", "Expected metric change"),
    ("structural_rewrite", "Structural rewrite"),
]
HIDE_FIELDS = {"headroom", "confidence"}

_KERNEL_ID = re.compile(r"kernel_\d{8}_\d{6}")
_PCT = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def _load_trees() -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for f in sorted(glob.glob(str(REPO / "run" / "*" / "*" / "optimization_tree.json"))):
        try:
            t = json.load(open(f))
        except Exception as e:                       # a half-written tree from a killed run
            print(f"  skip unreadable {f}: {e}")
            continue
        ks = t.get("kernels") or {}
        if not isinstance(ks, dict):
            ks = {k.get("kernel_name", f"n{i}"): k for i, k in enumerate(ks)}
        for name, k in ks.items():
            k.setdefault("kernel_name", name)
        out.append((f, ks))
    return out


def _strategy_text(strategy: Dict[str, Any]) -> str:
    """Render a proposal for the ranker, minus its own self-assessment."""
    seen, parts = set(), []
    for key, label in SHOW_FIELDS:
        if key in HIDE_FIELDS or label in seen:
            continue
        v = strategy.get(key)
        if not v:
            continue
        seen.add(label)
        parts.append(f"{label}:\n{str(v).strip()}")
    # Repair-phase proposals use a different schema; carry anything left over so
    # they are not rendered blank if they are ever included via --include-repair.
    known = {k for k, _ in SHOW_FIELDS} | HIDE_FIELDS
    for key, v in strategy.items():
        if key not in known and v:
            parts.append(f"{key.replace('_', ' ').title()}:\n{str(v).strip()}")
    return _KERNEL_ID.sub("<kernel>", "\n\n".join(parts))


def _claimed_gain(strategy: Dict[str, Any]) -> Optional[float]:
    """Largest percentage the proposal claims anywhere in its own forecast."""
    blob = " ".join(str(strategy.get(k) or "") for k in
                    ("expected_metric_change", "primary_optimisation_method"))
    vals = [abs(float(m)) for m in _PCT.findall(blob)]
    return max(vals) if vals else None


_HEADROOM_RANK = {"low": 0.0, "medium": 1.0, "high": 2.0}


def _headroom(strategy: Dict[str, Any]) -> Optional[float]:
    v = strategy.get("headroom") or strategy.get("confidence")
    return _HEADROOM_RANK.get(str(v).strip().lower()) if v else None


def build(include_repair: bool = False) -> Dict[str, Any]:
    trees = _load_trees()
    groups: List[Dict[str, Any]] = []
    dropped = collections.Counter()

    for tree_path, ks in trees:
        by_parent: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for k in ks.values():
            parent = k.get("parent")
            if not parent:
                dropped["root (no parent)"] += 1
                continue
            if not include_repair and k.get("phase") == "repair":
                dropped["repair phase"] += 1
                continue
            if not isinstance(k.get("strategy"), dict) or not k["strategy"]:
                dropped["no strategy text"] += 1
                continue
            if not k.get("runnable") or not k.get("speedup"):
                dropped["not runnable / no speedup"] += 1
                continue
            by_parent[parent].append(k)

        for parent, kids in by_parent.items():
            if len(kids) < 2:
                dropped["only child"] += 1
                continue
            kids.sort(key=lambda k: k["kernel_name"])
            groups.append({
                "group_id": f"{Path(tree_path).parent.parent.name}::{parent}",
                "tree": tree_path,
                "parent": parent,
                "parent_speedup": (ks.get(parent) or {}).get("speedup"),
                "task": (ks.get(parent) or {}).get("phase"),
                "children": kids,
            })

    pairs = []
    for g in groups:
        kids = g["children"]
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                a, b = kids[i], kids[j]
                sa, sb = float(a["speedup"]), float(b["speedup"])
                margin = abs(sa / sb - 1.0) * 100.0
                pairs.append({
                    "pair_id": f"{g['group_id']}::{a['kernel_name']}|{b['kernel_name']}",
                    "group_id": g["group_id"],
                    "parent": g["parent"],
                    "parent_speedup": g["parent_speedup"],
                    "a_name": a["kernel_name"], "b_name": b["kernel_name"],
                    "a_text": _strategy_text(a["strategy"]),
                    "b_text": _strategy_text(b["strategy"]),
                    "a_speedup": sa, "b_speedup": sb,
                    "winner": "a" if sa > sb else "b",
                    "margin_pct": margin,
                    "a_headroom": _headroom(a["strategy"]),
                    "b_headroom": _headroom(b["strategy"]),
                    "a_claimed": _claimed_gain(a["strategy"]),
                    "b_claimed": _claimed_gain(b["strategy"]),
                    "a_len": len(a["strategy"] and _strategy_text(a["strategy"])),
                    "b_len": len(b["strategy"] and _strategy_text(b["strategy"])),
                    "a_round": a.get("round"), "b_round": b.get("round"),
                })

    OUT.mkdir(parents=True, exist_ok=True)
    blob = {"pairs": pairs,
            "n_groups": len(groups),
            "group_sizes": sorted(len(g["children"]) for g in groups),
            "dropped": dict(dropped)}
    (OUT / "pairs.json").write_text(json.dumps(blob, indent=1))

    print(f"trees            {len(trees)}")
    print(f"sibling groups   {len(groups)}   sizes {blob['group_sizes']}")
    print(f"pairs            {len(pairs)}")
    print(f"dropped          {dict(dropped)}")
    if pairs:
        m = sorted(p["margin_pct"] for p in pairs)
        q = lambda p: m[int(p * (len(m) - 1))]
        print(f"margin %%        p25 {q(.25):.2f}  med {q(.5):.2f}  p75 {q(.75):.2f}  max {m[-1]:.1f}")
        for thr in (1, 2, 3, 5, 10):
            print(f"  margin > {thr:2d}%   {sum(x > thr for x in m):3d} pairs")
    print(f"\nwrote {OUT / 'pairs.json'}")
    return blob


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
SYSTEM = """You are a senior CUDA performance engineer triaging optimisation proposals.

You will see the current state of a kernel and two proposed next optimisations
for it. Both were written before either was implemented. Exactly one of them,
once implemented and benchmarked, measured faster than the other.

Your job is to say which one. You are ranking, not forecasting: you do not need
to estimate how much either gains, only which ends up ahead.

Reply with ONE line of JSON and nothing else:
{"winner": "A" or "B", "confidence": 0.0-1.0, "reason": "under 200 chars"}"""


def _prompt(pair: Dict[str, Any], flip: bool) -> str:
    first, second = ("b", "a") if flip else ("a", "b")
    ps = pair.get("parent_speedup")
    head = (f"Current kernel speedup over the PyTorch reference: {ps:.4f}x\n"
            if isinstance(ps, (int, float)) else "")
    return (f"{head}Two proposals were generated from this same state. "
            f"Which measured faster after implementation?\n\n"
            f"===== PROPOSAL A =====\n{pair[first + '_text']}\n\n"
            f"===== PROPOSAL B =====\n{pair[second + '_text']}\n\n"
            f"Which proposal produced the faster kernel? JSON only.")


_JSON = re.compile(r"\{[^{}]*\}")


def _parse(text: str) -> Dict[str, Any]:
    """Last JSON object wins; fall back to a bare A/B mention."""
    for m in reversed(_JSON.findall(text or "")):
        try:
            d = json.loads(m)
        except Exception:
            continue
        w = str(d.get("winner", "")).strip().upper()
        if w in ("A", "B"):
            conf = d.get("confidence")
            return {"winner": w,
                    "confidence": float(conf) if isinstance(conf, (int, float)) else None,
                    "reason": str(d.get("reason", ""))[:300]}
    m = re.search(r"\b(?:winner|answer)\b\D{0,20}\b([AB])\b", text or "", re.I)
    if m:
        return {"winner": m.group(1).upper(), "confidence": None, "reason": "<recovered>"}
    return {"winner": None, "confidence": None, "reason": "<unparsed>", "raw": (text or "")[:400]}


def run_arm(arm: str, workers: int = 4, limit: Optional[int] = None) -> None:
    if arm not in ARMS:
        sys.exit(f"unknown arm {arm!r}; pick one of {sorted(ARMS)}")
    sys.path.insert(0, str(REPO))
    os.environ[EFFORT_ENV] = _arm_effort(arm)      # read per call, so this takes effect
    from agents.query_server import query_server   # subscription auth, no API key

    pairs = json.loads((OUT / "pairs.json").read_text())["pairs"]
    if limit:
        pairs = pairs[:limit]

    cache_path = OUT / f"preds_{arm}.jsonl"
    done = set()
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("winner"):                 # retry anything that failed to parse
                    done.add((r["pair_id"], r["flip"]))
            except Exception:
                pass

    jobs = [(p, flip) for p in pairs for flip in (False, True)
            if (p["pair_id"], flip) not in done]
    print(f"arm {arm} ({ARMS[arm]}): {len(jobs)} calls to make, {len(done)} cached")
    if not jobs:
        return

    lock = threading.Lock()
    fh = cache_path.open("a")
    n_done = [0]

    def one(job):
        pair, flip = job
        try:
            txt = query_server(
                prompt=_prompt(pair, flip),
                system_prompt=SYSTEM,
                model_name=ARMS[arm],
                # NOT in _TOOL_CALL_TYPES, so this stays a one-shot call with no
                # tools and max_turns=1 -- the ranker must not go compile anything.
                call_type="rank_backtest",
                log_path=str(OUT / "usage.csv"),
                round_idx=-1,
            )
            rec = _parse(txt)
        except Exception as e:
            rec = {"winner": None, "confidence": None,
                   "reason": f"<error {type(e).__name__}: {e}>"}
        rec.update({"pair_id": pair["pair_id"], "flip": flip, "arm": arm})
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done[0] += 1
            if n_done[0] % 10 == 0 or n_done[0] == len(jobs):
                print(f"  {n_done[0]}/{len(jobs)}")

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, jobs))
    finally:
        fh.close()
    print(f"wrote {cache_path}")


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
def _baseline_pick(pair: Dict[str, Any], kind: str) -> Optional[str]:
    """Which side a free baseline picks, or None when it cannot separate them."""
    if kind == "headroom":
        a, b = pair["a_headroom"], pair["b_headroom"]
    elif kind == "claimed_gain":
        a, b = pair["a_claimed"], pair["b_claimed"]
    elif kind == "longer":
        a, b = pair["a_len"], pair["b_len"]
    else:
        raise ValueError(kind)
    if a is None or b is None or a == b:
        return None
    return "a" if a > b else "b"


def _arm_scores(pairs: List[Dict[str, Any]], arm: str) -> Dict[str, Dict[str, Any]]:
    """Per-pair score in [0,1] averaged over both orders, plus consistency."""
    path = OUT / f"preds_{arm}.jsonl"
    if not path.exists():
        return {}
    preds: Dict[Tuple[str, bool], str] = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("winner"):
            preds[(r["pair_id"], bool(r["flip"]))] = r["winner"]

    out = {}
    for p in pairs:
        trials, picks = [], []
        for flip in (False, True):
            w = preds.get((p["pair_id"], flip))
            if w is None:
                continue
            # In a flipped prompt, "A" is the pair's b-side.
            picked = ("b" if w == "A" else "a") if flip else ("a" if w == "A" else "b")
            picks.append(picked)
            trials.append(1.0 if picked == p["winner"] else 0.0)
        if not trials:
            continue
        out[p["pair_id"]] = {
            "score": st.mean(trials),
            "n_trials": len(trials),
            "consistent": (len(picks) == 2 and picks[0] == picks[1]),
            "both_orders": len(picks) == 2,
        }
    return out


def _bootstrap(pairs: List[Dict[str, Any]], score_of, n_boot: int = 4000,
               seed: int = 20260812) -> Optional[Tuple[float, float]]:
    """CI resampling whole sibling GROUPS -- pairs inside a group are not
    independent, so the group is the honest unit."""
    by_group: Dict[str, List[float]] = collections.defaultdict(list)
    for p in pairs:
        s = score_of(p)
        if s is not None:
            by_group[p["group_id"]].append(s)
    groups = [v for v in by_group.values() if v]
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pool: List[float] = []
        for _ in range(len(groups)):
            pool.extend(groups[rng.randrange(len(groups))])
        if pool:
            means.append(st.mean(pool))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _report_row(name: str, pairs: List[Dict[str, Any]], score_of) -> Optional[Dict[str, Any]]:
    scored = [(p, score_of(p)) for p in pairs]
    scored = [(p, s) for p, s in scored if s is not None]
    if not scored:
        return None
    vals = [s for _, s in scored]
    acc = st.mean(vals)
    naive_se = math.sqrt(max(acc * (1 - acc), 1e-12) / len(vals))
    ci = _bootstrap([p for p, _ in scored], score_of)
    return {"name": name, "n": len(vals), "acc": acc, "naive_se": naive_se,
            "ci": ci, "groups": len({p["group_id"] for p, _ in scored})}


def score(decisive: float = 3.0) -> None:
    blob = json.loads((OUT / "pairs.json").read_text())
    allp = blob["pairs"]
    dec = [p for p in allp if p["margin_pct"] > decisive]

    print(f"pairs {len(allp)} in {len({p['group_id'] for p in allp})} sibling groups")
    print(f"decisive subset (margin > {decisive}%): {len(dec)} pairs in "
          f"{len({p['group_id'] for p in dec})} groups\n")
    print(f"The decisive cut exists because measured drift is +0.9..+1.7%; below "
          f"that the recorded winner is\npartly noise, so no ranker can be scored "
          f"against it. Headline = decisive subset.\n")

    arm_cache = {a: _arm_scores(allp, a) for a in ARMS}

    for label, subset in (("DECISIVE (headline)", dec), ("ALL PAIRS", allp)):
        print(f"===== {label} =====")
        print(f"{'arm/baseline':<16} {'n':>4} {'groups':>6} {'acc':>7} "
              f"{'95% CI (clustered)':>22} {'naive SE':>9}")
        rows = []
        for a in ARMS:
            sc = arm_cache[a]
            if sc:
                rows.append(_report_row(a, subset,
                                        lambda p, sc=sc: (sc.get(p["pair_id"]) or {}).get("score")))
        for b in ("headroom", "claimed_gain", "longer"):
            rows.append(_report_row(b, subset, lambda p, b=b: (
                None if _baseline_pick(p, b) is None
                else (1.0 if _baseline_pick(p, b) == p["winner"] else 0.0))))
        for r in rows:
            if not r:
                continue
            ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r["ci"] else "n/a"
            print(f"{r['name']:<16} {r['n']:>4} {r['groups']:>6} {r['acc']:>7.3f} "
                  f"{ci:>22} {r['naive_se']:>9.3f}")
        print(f"{'coin flip':<16} {'':>4} {'':>6} {0.5:>7.3f}\n")

    print("===== CONSISTENCY (no ground truth needed) =====")
    print("Same pair asked in both orders. A model near 0.50 here is guessing,")
    print("whatever its accuracy says -- accuracy that high with consistency this")
    print("low is position bias plus luck, not signal.\n")
    print(f"{'arm':<16} {'pairs':>6} {'consistent':>11} {'unparsed':>9}")
    for a in ARMS:
        sc = arm_cache[a]
        if not sc:
            continue
        both = [v for v in sc.values() if v["both_orders"]]
        if not both:
            continue
        cons = st.mean([1.0 if v["consistent"] else 0.0 for v in both])
        missing = len(allp) - len(sc)
        print(f"{a:<16} {len(both):>6} {cons:>11.3f} {missing:>9}")

    print("\n===== ACCURACY vs MARGIN =====")
    print("If an arm has real signal it should rise to the right: wide gaps are")
    print("both easier to call and more reliably labelled.\n")
    bands = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]
    hdr = "  ".join(f"{lo}-{hi if hi < 1e9 else '+'}%".rjust(8) for lo, hi in bands)
    print(f"{'arm':<16} {hdr}")
    for a in ARMS:
        sc = arm_cache[a]
        if not sc:
            continue
        cells = []
        for lo, hi in bands:
            v = [sc[p["pair_id"]]["score"] for p in allp
                 if lo <= p["margin_pct"] < hi and p["pair_id"] in sc]
            cells.append(f"{st.mean(v):.2f}({len(v)})".rjust(8) if v else "-".rjust(8))
        print(f"{a:<16} {'  '.join(cells)}")

    if not any(arm_cache.values()):
        print("\nNo model arms run yet.  python -m utils.rank_backtest run --arm haiku")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="walk the trees into pairs.json (no API, no GPU)")
    b.add_argument("--include-repair", action="store_true",
                   help="also pair repair-phase proposals (different schema and goal)")

    r = sub.add_parser("run", help="query one model arm (resumable)")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=None, help="first N pairs only, for a smoke test")

    s = sub.add_parser("score", help="accuracy, clustered CIs, baselines")
    s.add_argument("--decisive", type=float, default=3.0,
                   help="margin %% above which the recorded winner is trusted (default 3)")

    a = ap.parse_args()
    if a.cmd == "build":
        build(include_repair=a.include_repair)
    elif a.cmd == "run":
        run_arm(a.arm, workers=a.workers, limit=a.limit)
    elif a.cmd == "score":
        score(decisive=a.decisive)


if __name__ == "__main__":
    main()
