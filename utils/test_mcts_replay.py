"""Replay test for the MCTS search loop -- mechanics only, no GPU, no LLM.

Scope, stated up front: this checks that the search BEHAVES, not that it wins.
Whether MCTS beats the ratchet cannot be answered by replay, because the gain a
round produces depends on the parent that round was given, and the counterfactual
parent was never measured. That question needs a live A/B
(``--search mcts`` vs ``--search ratchet`` at matched budget on one task).

What is checked here is everything that could silently break the loop:
  * selection never returns a node with no runnable code to hand the model
  * the search revisits non-incumbent nodes -- the entire reason it exists
  * the value chain stays on the paired basis and stays finite
  * max_depth is honoured, so the round-10 cliff is respected
  * every rollout makes exactly one node, and every node keeps one parent
  * the tree survives a checkpoint round trip mid-run with Q intact

The graph version of this file also replayed the three ``state_key`` modes and
asserted that ``features`` over-pools while ``code`` degenerates to a tree. There
are no key modes left to compare, and the tree is what ``code`` mode was, so
those three sections are gone rather than rewritten.

Run: python -m utils.test_mcts_replay
"""
from __future__ import annotations

import json
import random
from typing import List, Optional

from utils.mcts import MonteCarloTreeSearch, reward_from_gain

# The measured gain distribution on vae_block_002 (112 parent->child edges):
# 42% regress past -1%, 33% win past +1%, 25% inside the +-1% band, and the shape
# is bimodal rather than centred. Sampled here so the replay exercises the same
# regime the real loop sees, including the heavy tails that make max-backup right.
_BANDS = [
    (-12.0, -5.0, 0.116),
    (-5.0, -2.0, 0.232),
    (-2.0, -1.0, 0.071),
    (-1.0, 0.0, 0.116),
    (0.0, 1.0, 0.134),
    (1.0, 5.0, 0.232),
    (5.0, 12.0, 0.098),
]
_METHODS = ["vectorize_io", "shared_tile", "split_reduction", "channels_last",
            "cuda_graph", "fuse_epilogue", None]     # None = an unnamed change


def _sample_gain(rng: random.Random) -> float:
    r, acc = rng.random(), 0.0
    for lo, hi, p in _BANDS:
        acc += p
        if r <= acc:
            return rng.uniform(lo, hi)
    return rng.uniform(-1.0, 1.0)


def replay(rounds: int = 25, seed: int = 7, *, max_depth: int = 10,
           verbose: bool = False):
    rng = random.Random(seed)
    g = MonteCarloTreeSearch(max_depth=max_depth)
    g.observe(kernel_name="seed", kernel_path="/tmp/seed.py", value=1.0)

    switches = 0
    prev_parent: Optional[str] = None
    depths: List[int] = []
    values: List[float] = []

    for r in range(rounds):
        sel = g.select()
        assert sel is not None, "selection returned nothing"
        assert sel.node.kernel is not None, "selected a node with no kernel"
        assert sel.node.runnable, "selected an unrunnable node"
        assert sel.node.depth <= max_depth, f"selection exceeded max_depth: {sel.node.depth}"
        assert len(sel.path) == len(set(sel.path)), f"trajectory repeats: {sel.path}"
        assert sel.path == g.path_to(sel.node.key), \
            f"the trajectory is not the node's lineage: {sel.path} vs {g.path_to(sel.node.key)}"
        depths.append(sel.node.depth)
        if prev_parent is not None and sel.node.key != prev_parent:
            switches += 1
        prev_parent = sel.node.key

        # one round: the model proposes a change, it is measured, paired-verified
        gain = _sample_gain(rng)
        failed = rng.random() < 0.12          # compile/run failures do happen
        mech = rng.choice(_METHODS)
        child_value = sel.node.value * (1.0 + gain / 100.0)
        # No key to compute and no landing to find: observe() mints the id and
        # hands the node back. The graph version had to prefix-search for a "/s"
        # split key here, because the merge guard could rehome the kernel under a
        # key the caller had not asked for.
        child = g.observe(kernel_name=f"k{r}", kernel_path=f"/tmp/k{r}.py",
                          value=child_value, parent_key=sel.node.key,
                          runnable=not failed, mechanism=mech, note=f"round {r}")
        path = list(sel.path) + [child.key]
        g.backup(path, reward_from_gain(gain, failed=failed), failed=failed)
        values.append(child_value)
        if verbose:
            print(f"  r{r:>2} depth={sel.node.depth} N={sel.node.N} "
                  f"gain={gain:+6.2f}%{' FAILED' if failed else ''} -> {child_value:.4f}")

    return g, dict(switches=switches, depths=depths, values=values)


def main() -> None:
    def _check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        assert cond, msg

    print("[replay] 25 rounds")
    g, info = replay(rounds=25, seed=7)
    st = g.stats()
    print(f"  tree: {json.dumps(st)}")

    _check(st["nodes"] > 1, "the tree grew beyond the root")
    _check(info["switches"] > 0,
           f"selection revisited non-incumbent nodes ({info['switches']} parent switches "
           f"in 24 transitions) -- the ratchet would score 0 here")
    _check(max(info["depths"]) <= 10, "max_depth was never exceeded")
    _check(all(v > 0 and v == v for v in info["values"]), "the value chain stayed finite and positive")
    _check(st["total_visits"] == 25, "one backup per rollout")

    print("\n[replay] the shape is a tree, exactly")
    _check(st["nodes"] == 26, f"one node per rollout plus the seed (got {st['nodes']})")
    _check(sum(1 for n in g.nodes.values() if n.parent is None) == 1,
           "exactly one node has no parent")
    _check(all(n.parent in g.nodes for n in g.nodes.values() if n.parent is not None),
           "every parent resolves")
    child_of, doubled = {}, []
    for k, n in g.nodes.items():
        for c in n.children:
            if c in child_of:
                doubled.append(f"{c} under both {child_of[c]} and {k}")
            child_of[c] = k
    _check(not doubled, f"no node is a child of two parents (got {doubled[:3]})")
    _check(len(child_of) == st["nodes"] - 1, "every node but the root is someone's child")
    _check(all(g.nodes[c].parent == pk for c, pk in child_of.items()),
           "the parent pointer and the child list agree everywhere")
    # Reachability: a node the root cannot reach is invisible to select() while
    # best() and stats() still count it -- the failure the graph could produce.
    seen, stack = set(), [g.root]
    while stack:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k)
        stack.extend(g.nodes[k].children)
    _check(len(seen) == st["nodes"], "every node is reachable from the root")
    _check(st["mean_N"] > 1.0, f"mean visits per node is above 1 (got {st['mean_N']:.2f})")

    print("\n[replay] checkpoint round trip mid-run")
    blob = json.loads(json.dumps(g.to_dict()))
    g2 = MonteCarloTreeSearch.from_dict(blob)
    _check(len(g2.nodes) == len(g.nodes), "every node survived")
    _check(g2.total_visits == g.total_visits, "the visit total survived")
    _check(all(abs(g2.nodes[k].q() - g.nodes[k].q()) < 1e-12 for k in g.nodes),
           "every Q is reproduced exactly")
    _check(all(g2.nodes[k].parent == g.nodes[k].parent for k in g.nodes),
           "every parent pointer survived")
    s1 = g.select(); s2 = g2.select()
    _check(s1.node.key == s2.node.key,
           "the restored tree selects the same node -- a resume continues, not restarts")
    _check(g2.observe(kernel_name="post", kernel_path="/p", value=1.0,
                      parent_key=g2.root).key not in blob["nodes"],
           "and a node minted after the restore cannot collide with a restored id")

    print("\n[replay] stability across seeds")
    for sd in (1, 2, 3, 11, 42):
        gg, ii = replay(rounds=25, seed=sd)
        sg = gg.stats()
        assert sg["nodes"] == 26 and max(ii["depths"]) <= 10
        assert sg["max_depth_seen"] <= 11        # a child of a node at the cap
        print(f"  seed {sd:>2}: {sg['nodes']:>2} nodes, "
              f"{ii['switches']:>2} switches, mean N {sg['mean_N']:.2f}, "
              f"depth {sg['max_depth_seen']}, best {gg.best().value:.4f}")

    print("\n[replay] the shipped prior defaults still reproduce their claimed signal")
    # The one number that justifies --mcts_prior existing. Guarded rather than
    # trusted: a defaults change or a refit that quietly drops rho would otherwise
    # leave the flag recommending itself on evidence it no longer has.
    try:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from scripts.build_mechanism_prior import collect as _collect, _spearman
        from utils.mcts import MechanismPrior as _MP
        _edges = _collect(_P("run"), "vae_block_002", 9)
    except Exception as _exc:
        _edges = []
        print(f"  skipped: run history unavailable ({_exc.__class__.__name__})")
    if len(_edges) >= 40:
        _runs = sorted({e["run"] for e in _edges})
        _pr, _tr = [], []
        for _held in _runs:
            _m = _MP.fit([(e["mech"], e["gain"]) for e in _edges if e["run"] != _held],
                         min_support=1)
            for _e in (x for x in _edges if x["run"] == _held):
                if _m.support(_e["mech"]) > 0:
                    _pr.append(_m.advantage(_e["mech"])); _tr.append(_e["gain"])
        _rho = _spearman(_pr, _tr)
        _ord = sorted(range(len(_pr)), key=lambda i: -_pr[i])
        _top = _ord[: max(1, len(_ord) // 3)]
        _prec = sum(1 for i in _top if _tr[i] > 1) / len(_top)
        _base = sum(1 for g in _tr if g > 1) / len(_tr)
        print(f"  leave-one-run-out: n={len(_pr)}/{len(_edges)} runs={len(_runs)} "
              f"rho={_rho:+.3f} top-third {_prec*100:.0f}% vs base {_base*100:.0f}%")
        _check(_rho > 0.20,
               f"the prior still predicts held-out gain (rho={_rho:+.3f} > 0.20)")
        _check(_prec > _base,
               f"and still beats the base win rate ({_prec*100:.0f}% > {_base*100:.0f}%)")
        _check(len(_runs) >= 5,
               f"cross-validated across enough distinct runs ({len(_runs)}) -- a "
               f"run-grouping bug once collapsed these to 6 and flipped rho to -0.171")
    elif _edges:
        print(f"  skipped: only {len(_edges)} edges available")

    print("\n[replay] all checks passed")
    print("NOTE: this proves the mechanics, not that MCTS wins. Run the live A/B "
          "(--search mcts vs --search ratchet, matched --round, one task) for that.")


if __name__ == "__main__":
    main()
