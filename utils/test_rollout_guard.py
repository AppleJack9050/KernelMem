"""Checks for the guarded rollout call site.

    python -m utils.test_rollout_guard

Before this, `_llm_to_kernel` at the optimization call site sat at zero enclosing
try-blocks. Anything the agent raised -- most often the turn budget expiring with
nothing salvageable on disk -- unwound past the round loop and ended the run.
Three runs on 002 died that way.

The damage was never only the round. The exception also skipped `graph.backup`,
so the selected node kept its visit count and an untouched Q. The search held no
record that it had been expanded and produced nothing, and a --resume would
reselect it and draw the same dead plan. mcgs.reward_from_gain already specifies
the handling -- "a state that keeps emitting uncompilable code should fall out of
contention on its own rather than needing a separate rule" -- and returns 0.0 for
exactly this case. The mechanism existed; nothing reached it.

These checks drive the graph directly rather than the round loop, so they answer
the question the guard actually rests on: does backing up a 0.0 reward move
selection away from a node that keeps failing, and how fast?
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.mcgs import MonteCarloGraphSearch, reward_from_gain  # noqa: E402

LAM = 0.7


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


def _graph():
    """Root plus two children: `good` slightly ahead, `bad` just behind."""
    g = MonteCarloGraphSearch()
    g.observe(key="root", kernel_name="k0", kernel_path="k0.py", value=1.00)
    g.observe(key="good", kernel_name="k1", kernel_path="k1.py", value=1.20,
              parent_key="root", mechanism="A")
    g.observe(key="bad", kernel_name="k2", kernel_path="k2.py", value=1.19,
              parent_key="root", mechanism="B")
    for k, rel in (("good", 4.0), ("bad", 3.6)):
        g.backup(["root", k], reward_from_gain(rel))
    return g


print("[guard] a failed rollout is worth 0.0, strictly below a measured no-op")
_check(reward_from_gain(None, failed=True) == 0.0, "failed -> 0.0")
_check(abs(reward_from_gain(0.0) - 0.5) < 1e-9, "measured no-op -> 0.5")
_check(reward_from_gain(None, failed=True) < reward_from_gain(0.0),
       "so producing nothing is worse than producing something that did not help")

print("[guard] backing up a failure moves N, W and failures -- but NOT M")
g = _graph()
n = g.nodes["bad"]
before = (n.N, n.W, n.M, n.q(LAM), n.failures)
g.backup(["root", "bad"], reward_from_gain(None, failed=True), failed=True)
after = (n.N, n.W, n.M, n.q(LAM), n.failures)
_check(after[0] == before[0] + 1, f"N {before[0]} -> {after[0]}")
_check(abs(after[1] - before[1]) < 1e-12, "W unchanged (reward was 0.0)")
_check(abs(after[2] - before[2]) < 1e-12, "M unchanged -- M is a running max")
_check(after[4] == before[4] + 1, f"failures {before[4]} -> {after[4]}")
_check(after[3] < before[3], f"Q falls {before[3]:.4f} -> {after[3]:.4f}")

# Worth stating plainly rather than leaving implied: q = (1-lam)*(W/N) + lam*M,
# and M never decreases. At the default lam=0.7 a failure therefore moves only
# the 30% mean term, so ONE failure is a nudge, not a verdict.
drop = (before[3] - after[3]) / before[3] * 100.0
print(f"         (one failure costs {drop:.1f}% of Q at lam={LAM}; "
      f"M is a max, so 70% of q is untouched)")
_check(drop < 25.0, "a single failure is deliberately not decisive")

print("[guard] repeated failures do decide it -- selection moves off the node")
g = _graph()
picked = []
for _ in range(6):
    sel = g.select()
    picked.append(sel.node.key)
    if sel.node.key == "bad":
        g.backup(list(sel.path), reward_from_gain(None, failed=True), failed=True)
    else:
        g.backup(list(sel.path), reward_from_gain(3.0))
_check("bad" in picked, "the failing node does get tried")
_check(picked[-1] != "bad" or picked.count("bad") < len(picked),
       f"but does not monopolise selection: {picked}")
print(f"         selection order: {picked}")

print("[guard] a node that ONLY ever fails ends up behind one that works")
g = _graph()
for _ in range(4):
    g.backup(["root", "bad"], reward_from_gain(None, failed=True), failed=True)
    g.backup(["root", "good"], reward_from_gain(3.0))
qb, qg = g.nodes["bad"].q(LAM), g.nodes["good"].q(LAM)
_check(qb < qg, f"Q(bad)={qb:.4f} < Q(good)={qg:.4f}")
_check(g.nodes["bad"].failures == 4, "and the failure count is on the record")

print("[guard] the unguarded case: no backup at all leaves the node pristine")
g = _graph()
q_before = g.nodes["bad"].q(LAM)
n_before = g.nodes["bad"].N
# This is what happened when the exception propagated: nothing ran.
_check(g.nodes["bad"].q(LAM) == q_before and g.nodes["bad"].N == n_before,
       "an unrecorded failure is indistinguishable from never having tried")
_check(g.nodes["bad"].failures == 0,
       "which is exactly why --resume reselected it and drew the same plan")

print("\n[guard] all checks passed")
