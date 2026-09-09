"""Checks for the live tree view.

    python -m tests.test_mcts_view

The renderer walks the same structure `select()` walks. On a graph that meant it
inherited `select()`'s hazard -- a back-edge is a cycle, and a recursive descent
that does not guard against one does not return -- and most of this file used to
be about that. `observe()` cannot build a cycle on a tree, so what is worth
testing now is different: that every node is drawn exactly once under its only
parent, that the header agrees with the picture, that a graph-era checkpoint is
flattened rather than refused, and that the not-ready paths still explain
themselves instead of crashing a watcher that was started too early.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from utils.mcts import MonteCarloTreeSearch
from utils.mcts_view import find_tree_file, load_tree, render, render_file


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


def _tree() -> MonteCarloTreeSearch:
    """A seed, two branches, a re-derived recipe on each, and a dead node.

    `both` and `both2` apply the same two mechanisms in opposite orders. On the
    graph they merged into one state reachable by two routes -- the transposition
    the renderer drew once and referenced once. Here they are simply two nodes.
    """
    g = MonteCarloTreeSearch()
    g.observe(key="S", kernel_name="seed", kernel_path="/s", value=1.00)
    g.observe(key="V", kernel_name="vec", kernel_path="/v", value=1.04,
              parent_key="S", mechanism="vectorize")
    g.observe(key="T", kernel_name="tile", kernel_path="/t", value=1.09,
              parent_key="S", mechanism="tiling")
    g.observe(key="B", kernel_name="both", kernel_path="/b", value=1.12,
              parent_key="V", mechanism="tiling")
    g.observe(key="B2", kernel_name="both2", kernel_path="/b2", value=1.11,
              parent_key="T", mechanism="vectorize")
    g.observe(key="K", kernel_name="splitk", kernel_path="/k", value=1.19,
              parent_key="B", mechanism="split_k")
    g.observe(key="X", kernel_name="broke", kernel_path=None, value=0.0,
              parent_key="B", mechanism="warp_spec", runnable=False)
    for n in g.nodes.values():
        n.N, n.W, n.M = 3, 1.5, 0.6
    return g


print("[view] every node is drawn exactly once, under its only parent")
g = _tree()
txt = render(g)
_check(isinstance(txt, str) and txt, "render() returns")
lines = txt.splitlines()
for k in g.nodes:
    stat_lines = [ln for ln in lines if "N=" in ln and f" {k} " in ln + " "]
    _check(len(stat_lines) == 1,
           f"node {k} carries statistics on exactly one line (got {len(stat_lines)})")

print("[view] the graph-era markers are gone because they cannot occur")
_check("!!" not in txt and "back-edge" not in txt,
       "no back-edge marker: observe() cannot create an edge into an existing node")
_check("transposition" not in txt and "==" not in txt,
       "no transposition marker: a re-derived recipe is its own node")
_check("parents=" not in txt, "no multi-parent annotation")

print("[view] the re-derived recipe is two separate nodes on two branches")
_check(g.nodes["B"].parent == "V" and g.nodes["B2"].parent == "T",
       "each keeps the lineage it was actually produced on")
_check("B" in txt and "B2" in txt, "and both are drawn")

print("[view] a dead node is marked, not silently dropped")
_check(" x " in txt, "a node with no runnable kernel is flagged")
_check("X" in txt, "and it still appears")

print("[view] the header agrees with the picture")
st = g.stats()
_check(f"nodes {st['nodes']}" in txt, f"the node count is reported ({st['nodes']})")
_check(f"max-depth {st['max_depth_seen']}" in txt, "so is the depth")
_check("best" in txt and "1.1900" in txt, "the best value is named")
_check("not reachable" not in txt, "nothing is orphaned in a rooted tree")

print("[view] file resolution and the not-ready paths")
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    task = root / "vae_block_002"
    task.mkdir()
    ck = task / "checkpoint.json"
    ck.write_text(json.dumps({"version": 1, "mcts": _tree().to_dict()}))
    _check(find_tree_file(root) == ck, "a batch folder resolves to its task checkpoint")
    _check(find_tree_file(task) == ck, "a task folder resolves too")
    _check(find_tree_file(ck) == ck, "an explicit file is accepted")
    _check(find_tree_file(root / "nope") is None, "a missing path resolves to None")
    _check("nodes" in render_file(ck), "render_file works end to end")

    ratchet = task / "tree.json"
    ratchet.write_text(json.dumps({"optimization_tree": {}}))
    t, note = load_tree(ratchet)
    _check(t is None and "no MCTS tree" in note,
           "a ratchet-era checkpoint is explained, not crashed on")

    empty = task / "tree.json"
    empty.write_text(json.dumps({"mcts": {"root": None, "nodes": {}}}))
    t, note = load_tree(empty)
    _check(t is None and "empty" in note, "a tree with no rounds yet says so")

    torn = task / "tree.json"
    torn.write_text('{"mcts": {"nodes": {"a"')
    t, note = load_tree(torn)
    _check(t is None and "mid-write" in note, "a torn read is retried, not fatal")

    # The graph-era key and schema: read, flattened, and reported -- never refused.
    legacy = task / "tree.json"
    legacy.write_text(json.dumps({"mcgs": {
        "version": 1, "root": "R", "total_visits": 3, "splits": 0,
        "params": {"state_key_mode": "mechanisms", "merge_tolerance": 0.15},
        "nodes": {
            "R": {"key": "R", "depth": 0, "members": ["seed"], "rep": "seed",
                  "rep_path": "/s", "rep_value": 1.0, "N": 2, "W": 1.0, "M": 0.6,
                  "parents": [], "children": ["A"]},
            "A": {"key": "A", "depth": 1, "members": ["k1", "k1b"], "rep": "k1",
                  "rep_path": "/a", "rep_value": 1.2, "N": 1, "W": 0.7, "M": 0.7,
                  "parents": ["R"], "children": ["R"], "via": "vectorize"},
        }}}))
    t, note = load_tree(legacy)
    _check(t is not None and note == "", "an 'mcgs' blob loads rather than erroring")
    _check(set(t.nodes) == {"R", "A"} and t.nodes["A"].parent == "R",
           "it is flattened to a tree")
    _check("R" not in t.nodes["A"].children, "the back-edge A -> R is dropped")
    ltxt = render(t)
    _check("merged kernel" in ltxt and "extra parent" in ltxt,
           "and the render says what the flatten dropped rather than hiding it")

print("[view] orphans are reported rather than hidden")
go = _tree()
go.nodes["ORPHAN"] = go.nodes["K"].__class__(key="ORPHAN", depth=0)
to = render(go)
_check("not reachable from the root" in to and "ORPHAN" in to,
       "a node unreachable from the root is still listed")

print("\n[view] all checks passed")
