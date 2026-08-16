"""Checks for the live graph view.

    python -m utils.test_mcgs_view

The renderer walks the same DAG `select()` walks, so it inherits the same
hazard: a back-edge is a cycle, and a recursive descent that does not guard
against one does not return. That is the first thing tested here.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from utils.mcgs import MonteCarloGraphSearch
from utils.mcgs_view import find_graph_file, load_graph, render, render_file


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


def _graph(with_cycle: bool = True) -> MonteCarloGraphSearch:
    g = MonteCarloGraphSearch()
    g.observe(key="S", kernel_name="seed", kernel_path="/s", value=1.00)
    g.observe(key="V", kernel_name="vec", kernel_path="/v", value=1.04,
              parent_key="S", mechanism="vectorize")
    g.observe(key="T", kernel_name="tile", kernel_path="/t", value=1.09,
              parent_key="S", mechanism="tiling")
    # one state, two routes -- the transposition MCGS exists to exploit
    g.observe(key="B", kernel_name="both", kernel_path="/b", value=1.12,
              parent_key="V", mechanism="tiling")
    g.observe(key="B", kernel_name="both2", kernel_path="/b2", value=1.11,
              parent_key="T", mechanism="vectorize")
    g.observe(key="K", kernel_name="splitk", kernel_path="/k", value=1.19,
              parent_key="B", mechanism="split_k")
    g.observe(key="X", kernel_name="broke", kernel_path=None, value=0.0,
              parent_key="B", mechanism="warp_spec", runnable=False)
    if with_cycle:
        g.observe(key="V", kernel_name="revert", kernel_path="/r", value=1.04,
                  parent_key="K", mechanism="revert")
    for n in g.nodes.values():
        n.N, n.W, n.M = 3, 1.5, 0.6
    return g


print("[view] a back-edge does not trap the renderer")
gc_ = _graph(with_cycle=True)
_check("V" in gc_.nodes["K"].children, "fixture really has the K->V back-edge")
_check("K" in gc_.nodes["B"].children and "B" in gc_.nodes["V"].children,
       "and the forward path V->B->K that closes it into a cycle")
txt = render(gc_)          # before the ancestor guard this would not return
_check(isinstance(txt, str) and txt, "render() returns on a cyclic graph")
_check("back-edge" in txt, "the back-edge is labelled rather than followed")
_check(txt.count("!!") == 1, "the cycle is reported exactly once")

print("[view] a merged state is drawn once, at its shallowest route")
_check(txt.count("transposition") >= 1, "the second route to B is a reference")
# B is a child of both V and T; it must be EXPANDED once and referenced once.
# An expanded line is the one carrying statistics; a reference line has none.
expanded_B = [ln for ln in txt.splitlines() if "N=" in ln and " B " in ln.replace("==", "  ")]
_check(len(expanded_B) == 1, f"B expands exactly once (got {len(expanded_B)})")

print("[view] every node appears somewhere")
for k in gc_.nodes:
    _check(k in txt, f"state {k} is present in the render")
_check("not reachable" not in txt, "nothing is orphaned in a rooted graph")

print("[view] a dead state is marked, not silently dropped")
_check(" x " in txt, "a state with no runnable representative is flagged")

print("[view] acyclic graphs render without any cycle marker")
ta = render(_graph(with_cycle=False))
_check("!!" not in ta and "back-edge" not in ta, "no false cycle report")

print("[view] file resolution and the not-ready paths")
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    task = root / "vae_block_002"
    task.mkdir()
    ck = task / "checkpoint.json"
    ck.write_text(json.dumps({"version": 1, "mcgs": _graph().to_dict()}))
    _check(find_graph_file(root) == ck, "a batch folder resolves to its task checkpoint")
    _check(find_graph_file(task) == ck, "a task folder resolves too")
    _check(find_graph_file(ck) == ck, "an explicit file is accepted")
    _check(find_graph_file(root / "nope") is None, "a missing path resolves to None")
    _check("states" in render_file(ck), "render_file works end to end")

    legacy = task / "graph.json"
    legacy.write_text(json.dumps({"optimization_tree": {}}))
    g, note = load_graph(legacy)
    _check(g is None and "no MCGS graph" in note,
           "a ratchet-era checkpoint is explained, not crashed on")

    empty = task / "graph.json"
    empty.write_text(json.dumps({"mcgs": {"root": None, "nodes": {}}}))
    g, note = load_graph(empty)
    _check(g is None and "empty" in note, "a graph with no rounds yet says so")

    torn = task / "graph.json"
    torn.write_text('{"mcgs": {"nodes": {"a"')
    g, note = load_graph(torn)
    _check(g is None and "mid-write" in note, "a torn read is retried, not fatal")

print("[view] orphans are reported rather than hidden")
go = _graph(with_cycle=False)
go.nodes["ORPHAN"] = go.nodes["K"].__class__(key="ORPHAN", depth=0)
to = render(go)
_check("not reachable from the root" in to and "ORPHAN" in to,
       "a state unreachable from the root is still listed")

print("\n[view] all checks passed")
