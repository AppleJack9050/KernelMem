"""Checks for the broadcast-credit path memory.

    python -m tests.test_pathmemory

The three things that can go wrong here, in the order they would hurt:

1. **It does not terminate, or blows the stack.** The pass walks the whole tree,
   and recursion depth is bounded by `--mcts_max_depth` only in a healthy run --
   a tree restored from a checkpoint written under a raised cap can be
   arbitrarily deep, and a RecursionError in a memory helper must never be able
   to take down a round. Tested first, on a spine far deeper than any recursive
   implementation would survive. (The graph version was also tested against a
   back-edge installed by a transposition merge; `observe` cannot build one on a
   tree, so that hazard is gone rather than guarded.)
2. **It credits the wrong line.** The whole point is to disagree with Q where Q
   is wrong -- a branch holding the record among duds. If credit merely tracked Q
   the feature would be an expensive no-op, so that disagreement is asserted
   directly rather than assumed.
3. **It changes the search when it was supposed to be off.** `--mcts_pv_bonus`
   defaults to 0.0, and a default-off policy knob that silently perturbs
   selection is worse than one that does nothing.
"""
from __future__ import annotations

from utils.mcts import MonteCarloTreeSearch
from utils.pathmemory import (broadcast_credit, pathway_lesson,
                              principal_variation, render_pathway)


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


def _linear() -> MonteCarloTreeSearch:
    """Seed -> A -> B, plus a dead sibling C off the seed."""
    g = MonteCarloTreeSearch()
    g.observe(key="S", kernel_name="seed", kernel_path="/s", value=1.00)
    g.observe(key="A", kernel_name="a", kernel_path="/a", value=1.05,
              parent_key="S", mechanism="vectorize")
    g.observe(key="B", kernel_name="b", kernel_path="/b", value=1.20,
              parent_key="A", mechanism="tiling")
    g.observe(key="C", kernel_name="c", kernel_path="/c", value=0.90,
              parent_key="S", mechanism="shared-mem")
    return g


print("[credit] the seed learns what it eventually led to")
g = _linear()
broadcast_credit(g)
_check(abs(g.nodes["S"].credit - 1.20) < 1e-9,
       f"root credit is the best in the whole tree (got {g.nodes['S'].credit:.4f})")
_check(g.nodes["S"].credit_key == "B", "root knows WHICH node holds the record")
_check(abs(g.nodes["C"].credit - 0.90) < 1e-9,
       "a dead sibling is credited with its own value, not the tree's best")
_check(abs(g.nodes["S"].value - 1.00) < 1e-9,
       "broadcasting does not overwrite a node's own measured value")

print("[pv] the principal variation is the chain that reached the record")
pv = principal_variation(g)
_check(pv == ["S", "A", "B"], f"PV is seed->A->B (got {pv})")
_check(g.pv == {"S", "A", "B"}, "broadcast_credit refills tree.pv for selection")

print("[credit] credit disagrees with Q where the two are denominated differently")
# The disagreement is NOT "one record among duds" -- backup propagates the
# record's reward up the selected path, so Q's max term already sees it. The real
# gap is the denominator: reward_from_gain is a function of the PERCENTAGE GAIN
# OVER THE PARENT, so a line climbing hard off a bad seed beats, on Q, a line
# inching forward from a good one -- while being worse on absolute score, which
# is the only number the run reports.
#   CLIMBER: 1.00 -> 0.60 -> 0.90   huge relative gain, poor absolute value
#   TOPPER:  1.00 -> 1.25 -> 1.30   small relative gain, holds the record
d = MonteCarloTreeSearch()
d.observe(key="S", kernel_name="seed", kernel_path="/s", value=1.00)
d.observe(key="CLIMBER", kernel_name="c", kernel_path="/c", value=0.60,
          parent_key="S", mechanism="rescue")
d.observe(key="CLIMBER2", kernel_name="c2", kernel_path="/c2", value=0.90,
          parent_key="CLIMBER", mechanism="rescue-2")
d.observe(key="TOPPER", kernel_name="t", kernel_path="/t", value=1.25,
          parent_key="S", mechanism="fuse")
d.observe(key="RECORD", kernel_name="r", kernel_path="/r", value=1.30,
          parent_key="TOPPER", mechanism="fuse-2")
d.backup(["S", "CLIMBER"], 0.20)                  # -40%: a bad step
d.backup(["S", "CLIMBER", "CLIMBER2"], 0.95)      # +50%: a huge relative gain
d.backup(["S", "TOPPER"], 0.70)                   # +25%
d.backup(["S", "TOPPER", "RECORD"], 0.60)         # +4%: modest, but it is the top
broadcast_credit(d)
_check(d.nodes["CLIMBER"].q(0.7) > d.nodes["TOPPER"].q(0.7),
       f"Q prefers the big climber ({d.nodes['CLIMBER'].q(0.7):.3f} > "
       f"{d.nodes['TOPPER'].q(0.7):.3f})")
_check(d.nodes["TOPPER"].credit > d.nodes["CLIMBER"].credit,
       "credit prefers the line that actually holds the record")
_check(principal_variation(d) == ["S", "TOPPER", "RECORD"],
       "the PV follows absolute score, not the steeper climb")
_check(d.nodes["S"].credit_key == "RECORD",
       "and the seed names the record holder, which a scalar Q cannot do")

print("[depth] a deep spine does not blow the stack")
# 4000 is far past sys.getrecursionlimit()'s default 1000, so a recursive
# post-order would die here. --mcts_max_depth defaults to 10, but the cap is a
# SELECTION rule: nothing stops a checkpoint written at a raised cap, or a
# hand-edited one, from carrying a spine this long into a resume.
DEEP = 4000
deep = MonteCarloTreeSearch()
prev = deep.observe(kernel_name="seed", kernel_path="/s", value=1.00)
root_key, tip_key = prev.key, None
for i in range(DEEP):
    prev = deep.observe(kernel_name=f"k{i}", kernel_path=f"/{i}",
                        value=1.00 + i * 0.001, parent_key=prev.key,
                        mechanism=f"m{i % 7}")
tip_key = prev.key
broadcast_credit(deep)                     # must return, not RecursionError
_check(abs(deep.nodes[root_key].credit - deep.nodes[tip_key].value) < 1e-9,
       f"credit reaches the root across {DEEP} levels")
_check(deep.nodes[root_key].credit_key == tip_key,
       "and the root names the deepest node as the record holder")
_check(len(principal_variation(deep)) == DEEP + 1,
       f"the PV spans the whole spine (got {len(principal_variation(deep))})")

print("[tree] every node has one parent, so the PV is the lineage the run walked")
one = MonteCarloTreeSearch()
r0 = one.observe(kernel_name="seed", kernel_path="/s", value=1.00)
a0 = one.observe(kernel_name="a", kernel_path="/a", value=1.10,
                 parent_key=r0.key, mechanism="m1")
b0 = one.observe(kernel_name="b", kernel_path="/b", value=1.15,
                 parent_key=a0.key, mechanism="m2")
# The graph-era transposition: the same recipe reached by another order. It is a
# separate node now, and it does NOT become a second parent of anything.
b1 = one.observe(kernel_name="b_again", kernel_path="/b2", value=1.14,
                 parent_key=r0.key, mechanism="m2")
broadcast_credit(one)
_check(b0.key != b1.key, "re-deriving a recipe makes a new node, never a merge")
_check(one.nodes[b0.key].parent == a0.key and one.nodes[b1.key].parent == r0.key,
       "each keeps its own single parent")
_check(principal_variation(one) == [r0.key, a0.key, b0.key],
       f"the PV is the exact lineage (got {principal_variation(one)})")
_check(one.path_to(b0.key) == principal_variation(one),
       "and walking up the parents gives the same answer as descending credit")

print("[render] the block carries each step's own context")
block = render_pathway(g, current_key="B")
_check("SEED" in block and "vectorize" in block and "tiling" in block,
       "every mechanism on the pathway is named")
_check("1.2000" in block, "the record value appears")
_check("YOU ARE HERE" in block, "the node being optimized is marked")
_check("shared-mem" in block,
       "a dead end tried off the pathway is reported, so it is not re-proposed")

print("[render] a kernel off the pathway is told so, with both numbers")
off = render_pathway(g, current_key="C")
_check("NOT on this pathway" in off, "an off-pathway kernel is told plainly")
_check("0.9000" in off and "1.2000" in off,
       "both its own value and the pathway's are quoted, so the gap is legible")

print("[render] silence rather than an empty heading")
empty = MonteCarloTreeSearch()
_check(render_pathway(empty) == "", "an empty graph renders nothing at all")
seed_only = MonteCarloTreeSearch()
seed_only.observe(key="S", kernel_name="s", kernel_path="/s", value=1.0)
broadcast_credit(seed_only)
_check("SEED" in render_pathway(seed_only),
       "a seed-only graph still reports the seed as the pathway")

print("[policy] the PV bonus is off by default and inert when off")
plain = _linear()
broadcast_credit(plain)
_check(plain.pv_bonus == 0.0, "pv_bonus defaults to 0.0")
before = plain.select()
plain.pv = set()                                   # as if never broadcast
after = plain.select()
_check(before is not None and after is not None
       and before.node.key == after.node.key,
       "with the bonus off, selection is identical with and without a PV")

print("[policy] the bonus moves selection only when switched on")
# Two children of the root, both selectable, the PV one deliberately given the
# WORSE Q so a change of winner can only come from the bonus.
b = MonteCarloTreeSearch(widen_k=0.5, widen_alpha=0.0)
b.observe(key="S", kernel_name="seed", kernel_path="/s", value=1.00)
b.observe(key="LOW", kernel_name="l", kernel_path="/l", value=1.40,
          parent_key="S", mechanism="m1")
b.observe(key="HIGH", kernel_name="h", kernel_path="/h", value=1.01,
          parent_key="S", mechanism="m2")
b.backup(["S", "LOW"], 0.10)
b.backup(["S", "HIGH"], 0.50)
broadcast_credit(b)
_check(b.nodes["S"].credit_key == "LOW", "LOW holds the record despite the worse Q")
picked_off = b.select()
b.pv_bonus = 0.75
picked_on = b.select()
_check(picked_off is not None and picked_on is not None, "both selections returned")
_check(picked_off.node.key == "HIGH", "off: UCT takes the better-Q child")
_check(picked_on.node.key == "LOW", "on: the bonus takes the record-holding child")

print("[persist] credit and the bonus survive a checkpoint round trip")
rt = MonteCarloTreeSearch.from_dict(b.to_dict())
_check(abs(rt.nodes["S"].credit - b.nodes["S"].credit) < 1e-9, "credit round-trips")
_check(rt.nodes["S"].credit_key == b.nodes["S"].credit_key, "credit_key round-trips")
_check(rt.pv_bonus == b.pv_bonus, "pv_bonus round-trips, so a resume keeps the policy")

print("[persist] a checkpoint written before path memory existed still loads")
old = b.to_dict()
del old["params"]["pv_bonus"]
for nd in old["nodes"].values():
    nd.pop("credit", None)
    nd.pop("credit_key", None)
legacy = MonteCarloTreeSearch.from_dict(old)
_check(legacy.pv_bonus == 0.0, "a legacy checkpoint resumes with the bonus off")
_check(legacy.nodes["S"].credit == 0.0, "a legacy node starts uncredited")
broadcast_credit(legacy)
_check(abs(legacy.nodes["S"].credit - 1.40) < 1e-9, "and the next broadcast refills it")

print("[lesson] a pathway worth recording becomes a lesson; a bare seed does not")
les = pathway_lesson(g, "vae_block_002")
_check(les is not None and les["confidence"] == "measured", "a real pathway distils")
_check("vectorize" in les["evidence"] and "tiling" in les["evidence"],
       "the lesson's evidence names the actual chain")
_check(pathway_lesson(seed_only, "t") is None,
       "a seed with no surviving edit teaches nothing and is not recorded")

print("\n[pathmemory] all checks passed")
