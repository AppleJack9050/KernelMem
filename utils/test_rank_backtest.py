#!/usr/bin/env python
"""Checks for the ranking backtest's scoring logic.

Run:  python -m utils.test_rank_backtest

The whole conclusion rests on two pieces of bookkeeping that are easy to get
silently wrong and impossible to eyeball afterwards:

* THE FLIP. Each pair is asked twice, and in the flipped prompt "A" refers to
  the pair's b-side. Invert that mapping and a perfect ranker scores 0.5 while a
  useless one also scores 0.5, so the bug is invisible in the output.
* THE POSITION-BIAS CANCELLATION. A model that always answers "A" must land at
  exactly 0.5 with consistency 0.0. If it can score above chance by ignoring the
  content entirely, the experiment measures nothing.

So this drives the scorer with synthetic arms whose true accuracy is known by
construction -- oracle, anti-oracle, always-A, always-B -- and asserts it
recovers them. It also checks the clustered bootstrap is actually clustering,
since the group correction is the difference between an honest CI and a
flattering one.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import tempfile
from pathlib import Path

from utils import rank_backtest as rb


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _fake_pairs():
    """Two groups: one of 3 children (3 pairs), one of 2 (1 pair)."""
    pairs = []
    for gid, names in (("g1", "xyz"), ("g2", "pq")):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sa, sb = 1.0 + 0.1 * i, 1.0 + 0.1 * j     # b always faster
                pairs.append({
                    "pair_id": f"{gid}::{a}|{b}", "group_id": gid,
                    "a_name": a, "b_name": b,
                    "a_speedup": sa, "b_speedup": sb,
                    "winner": "a" if sa > sb else "b",
                    "margin_pct": abs(sa / sb - 1) * 100,
                    "a_headroom": None, "b_headroom": None,
                    "a_claimed": None, "b_claimed": None,
                    "a_len": 10, "b_len": 20,
                })
    return pairs


def _write_arm(tmp: Path, arm: str, pairs, policy) -> None:
    """policy(pair, flip) -> "A"/"B" as the model would answer that prompt."""
    with (tmp / f"preds_{arm}.jsonl").open("w") as f:
        for p in pairs:
            for flip in (False, True):
                f.write(json.dumps({"pair_id": p["pair_id"], "flip": flip,
                                    "arm": arm, "winner": policy(p, flip)}) + "\n")


def _oracle(p, flip):
    """Always names the true winner, accounting for the flip."""
    shown_as_A = "b" if flip else "a"
    return "A" if p["winner"] == shown_as_A else "B"


def test_scoring_recovers_known_arms() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        old, rb.OUT = rb.OUT, tmp
        try:
            pairs = _fake_pairs()
            _write_arm(tmp, "opus",   pairs, _oracle)
            _write_arm(tmp, "sonnet", pairs, lambda p, f: "B" if _oracle(p, f) == "A" else "A")
            _write_arm(tmp, "haiku",  pairs, lambda p, f: "A")

            o = rb._arm_scores(pairs, "opus")
            _check(len(o) == len(pairs), "oracle must score every pair")
            _check(all(v["score"] == 1.0 for v in o.values()), "oracle must be 1.0")
            _check(all(v["consistent"] for v in o.values()), "oracle must be self-consistent")

            a = rb._arm_scores(pairs, "sonnet")
            _check(all(v["score"] == 0.0 for v in a.values()), "anti-oracle must be 0.0")
            _check(all(v["consistent"] for v in a.values()),
                   "anti-oracle is wrong but still consistent -- consistency is not accuracy")

            p = rb._arm_scores(pairs, "haiku")
            _check(all(v["score"] == 0.5 for v in p.values()),
                   "always-A must average to exactly 0.5 once both orders are pooled")
            _check(not any(v["consistent"] for v in p.values()),
                   "always-A must be 0% consistent -- that is how position bias is caught")
        finally:
            rb.OUT = old
    print("PASS  oracle=1.0 consistent, anti-oracle=0.0 consistent, always-A=0.5 at 0% consistency")


def test_flip_mapping_is_not_symmetric() -> None:
    """Guard the exact bug the flip invites: dropping the inversion.

    A ranker that is right in the unflipped order and wrong in the flipped one
    is a position-biased ranker, and must score 0.5 -- not 1.0.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        old, rb.OUT = rb.OUT, tmp
        try:
            pairs = _fake_pairs()
            # Answers "A" is-the-winner only when unflipped; ignores the flip.
            _write_arm(tmp, "opus", pairs,
                       lambda p, f: ("A" if p["winner"] == "a" else "B"))
            s = rb._arm_scores(pairs, "opus")
            _check(all(v["score"] == 0.5 for v in s.values()),
                   "ignoring the flip must cost exactly half the credit")
            _check(not any(v["consistent"] for v in s.values()),
                   "such an arm names a different kernel each order, so it is inconsistent")
        finally:
            rb.OUT = old
    print("PASS  flip inversion is applied: an order-blind answer scores 0.5, not 1.0")


def test_partial_and_unparsed_are_handled() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        old, rb.OUT = rb.OUT, tmp
        try:
            pairs = _fake_pairs()
            with (tmp / "preds_opus.jsonl").open("w") as f:
                p0 = pairs[0]
                f.write(json.dumps({"pair_id": p0["pair_id"], "flip": False,
                                    "arm": "opus", "winner": _oracle(p0, False)}) + "\n")
                f.write(json.dumps({"pair_id": pairs[1]["pair_id"], "flip": False,
                                    "arm": "opus", "winner": None}) + "\n")
            s = rb._arm_scores(pairs, "opus")
            _check(len(s) == 1, "a pair with only unparsed answers must be dropped, not counted")
            v = s[pairs[0]["pair_id"]]
            _check(v["n_trials"] == 1 and not v["both_orders"],
                   "a one-order pair must be flagged so it is excluded from consistency")
        finally:
            rb.OUT = old
    print("PASS  unparsed answers drop out; single-order pairs are excluded from consistency")


def test_bootstrap_clusters_by_group() -> None:
    """The clustered CI must be wider than a naive per-pair one.

    Build a set where outcomes are perfectly correlated inside each group and
    split across groups: resampling pairs would call this well-determined,
    resampling groups correctly reports near-total uncertainty.
    """
    pairs = []
    for g in range(6):
        for k in range(5):
            pairs.append({"pair_id": f"g{g}::{k}", "group_id": f"g{g}",
                          "margin_pct": 10.0})
    val = {f"g{g}": (1.0 if g % 2 == 0 else 0.0) for g in range(6)}
    ci = rb._bootstrap(pairs, lambda p: val[p["group_id"]], n_boot=3000)
    _check(ci is not None, "bootstrap must return an interval")
    lo, hi = ci
    _check(hi - lo > 0.5, f"clustered CI must stay wide under group-correlated data, got {hi-lo:.2f}")
    naive = 2 * 1.96 * (0.5 / (len(pairs) ** 0.5))
    _check(hi - lo > naive, f"clustered CI {hi-lo:.2f} must exceed the naive {naive:.2f}")
    print(f"PASS  clustered bootstrap CI [{lo:.2f}, {hi:.2f}] is wider than the naive "
          f"+-{naive/2:.2f} it replaces")


def test_parse_handles_real_shapes() -> None:
    _check(rb._parse('{"winner": "A", "confidence": 0.7, "reason": "x"}')["winner"] == "A", "plain")
    _check(rb._parse('thinking...\n{"winner":"B","confidence":0.3}')["winner"] == "B", "prefixed")
    _check(rb._parse('{"winner":"A"}\n{"winner":"B"}')["winner"] == "B", "last object wins")
    _check(rb._parse('```json\n{"winner": "B"}\n```')["winner"] == "B", "fenced")
    _check(rb._parse("The winner is A.")["winner"] == "A", "prose fallback")
    _check(rb._parse("no answer here")["winner"] is None, "unparseable must be None, not a guess")
    _check(rb._parse('{"winner":"C"}')["winner"] is None, "an invalid label must not be accepted")
    print("PASS  parser handles fenced/prefixed/multiple JSON and refuses to guess")


def test_baselines_abstain_on_ties() -> None:
    p = {"a_headroom": 1.0, "b_headroom": 1.0, "a_claimed": None, "b_claimed": 5.0,
         "a_len": 10, "b_len": 20, "winner": "b"}
    _check(rb._baseline_pick(p, "headroom") is None, "equal headroom must abstain, not pick")
    _check(rb._baseline_pick(p, "claimed_gain") is None, "a missing claim must abstain")
    _check(rb._baseline_pick(p, "longer") == "b", "longer must pick the longer side")
    print("PASS  free baselines abstain on ties/missing rather than scoring a coin flip")


def test_real_pairs_are_wellformed() -> None:
    path = rb.OUT / "pairs.json"
    if not path.exists():
        print("SKIP  no pairs.json yet; run `python -m utils.rank_backtest build`")
        return
    pairs = json.loads(path.read_text())["pairs"]
    for p in pairs:
        _check(p["winner"] in ("a", "b"), "winner must be a or b")
        faster = "a" if p["a_speedup"] > p["b_speedup"] else "b"
        _check(p["winner"] == faster, f"winner disagrees with speedups in {p['pair_id']}")
        _check(p["a_text"] and p["b_text"], f"blank proposal text in {p['pair_id']}")
        _check(p["a_name"] != p["b_name"], "a pair must be two distinct kernels")
        for side in ("a", "b"):
            t = p[f"{side}_text"].lower()
            _check("headroom:" not in t, f"headroom field leaked into {p['pair_id']}")
            _check("confidence:" not in t, f"confidence field leaked into {p['pair_id']}")
    ids = [p["pair_id"] for p in pairs]
    _check(len(ids) == len(set(ids)), "pair_ids must be unique")
    print(f"PASS  {len(pairs)} real pairs are well-formed, deduplicated and unleaked")


def main() -> None:
    test_scoring_recovers_known_arms()
    test_flip_mapping_is_not_symmetric()
    test_partial_and_unparsed_are_handled()
    test_bootstrap_clusters_by_group()
    test_parse_handles_real_shapes()
    test_baselines_abstain_on_ties()
    test_real_pairs_are_wellformed()
    print("\nAll ranking-backtest checks passed.")


if __name__ == "__main__":
    main()
