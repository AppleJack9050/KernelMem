"""Just-in-time long-term memory: the winning pathway, broadcast back to its seed.

    python -m utils.pathmemory run/<stamp>_<task>_<tag>          # show the pathway
    python -m utils.pathmemory run/<stamp>_<task>_<tag> --prompt # show the injected block

What this adds that ``backup()`` does not
-----------------------------------------
MCTS ``backup()`` credits the selected path with the reward of ONE rollout, at
the moment that rollout is measured. It answers *"how did this edit score?"*. It
cannot answer *"which chain of edits is the run actually winning with?"*, because
a node whose descendant eventually produced the record kernel gets no more credit
at its ancestors than a sibling that happened to score the same on its own visit.
The information exists in the tree and nothing reads it.

``broadcast_credit()`` runs the other direction: a reverse pass over the tree that
gives every node the value of the BEST kernel anywhere in its subtree. The seed
root therefore ends up holding the best score the whole run has reached, and every
node on the way holds the best score reachable *from it*. Two numbers per node,
answering different questions:

* ``Q``      -- how much this line has IMPROVED, mixing the mean and the max of
  its rollout rewards. What UCT already selects on.
* ``credit`` -- how high this line has ever REACHED, in absolute score.

They disagree because they are denominated differently, and this is the whole
reason the pass is worth running. ``backup`` stores a REWARD, and
``reward_from_gain`` is ``tanh`` of the rollout's *percentage gain over its
parent* -- so a line that climbed +50% off a bad seed outranks, on Q, a line that
climbed +4% off a good one, while being far worse on the only number the run
reports. The node holding the record can therefore lose the argmax to a line that
has never come near it. Q is the right signal for "is this line still yielding?";
it was never a claim about absolute standing, and nothing else in the tree was
making that claim either.

There is a second gap credit closes: ``Q`` is a scalar in [0, 1] with no link back
to a kernel, so it cannot tell a prompt WHICH kernel a line reached. ``credit_key``
can, and that is what makes the pathway renderable at all.

The three consumers
-------------------
1. ``render_pathway()`` -- the prompt block. Full context for every node on the
   principal variation: which mechanism produced it, what it scored, the delta it
   bought over its parent, and what was tried and FAILED off it. The model is
   told where the run actually is on the map rather than being handed one kernel
   with no lineage.
2. ``MonteCarloTreeSearch.pv`` + ``--mcts_pv_bonus`` -- a selection bonus for
   staying on that pathway. OFF by default; see the note on convention below.
3. ``pathway_lesson()`` -- distils the pathway into a printable summary of the
   chain that won this run, for a human reading a finished run.

On defaults, and this repo's convention
---------------------------------------
The prompt block is ON by default: it is context, and context that is measured
this run cannot mislead the way an unsourced claim can. ``--mcts_pv_bonus`` is OFF by default, matching ``--mcts_prior`` and
``--mcts_epsilon``: it changes the search policy, and the convention here is that
such a change is opt-in until it has won its own A/B. Set ``KERNELMEM_PATHWAY=0``
to drop the block (e.g. for that A/B).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.mcts import MonteCarloTreeSearch, TreeNode  # noqa: E402

_ENV = "KERNELMEM_PATHWAY"

# A node whose kernel never ran holds no value of its own to contribute. It is
# not skipped -- its children may still carry credit through it -- it simply
# starts from "nothing here".
_NOTHING = float("-inf")


def enabled() -> bool:
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


# --------------------------------------------------------------------- credit
def broadcast_credit(tree: MonteCarloTreeSearch) -> Dict[str, float]:
    """Give every node the best value reachable from it, and refill ``tree.pv``.

    Iterative post-order rather than recursion: depth is bounded by ``max_depth``
    in a healthy run, but a tree loaded from a checkpoint written under a raised
    cap can be arbitrarily deep, and a RecursionError in a memory helper must
    never be able to take down a round.

    The graph version needed a three-colour DFS here, because a transposition
    merge could install a back-edge and this was a general digraph: a child still
    on the stack had to be skipped when its parent was finalised, since its credit
    was not yet known and waiting for it would deadlock. On a tree there are no
    back-edges and every child is finalised before its parent, so the colours are
    gone and the pass is exact rather than merely safe.

    Returns the credit map, and as a side effect writes ``credit``/``credit_key``
    onto every node and sets ``tree.pv`` to the principal variation's members.
    """
    nodes = tree.nodes
    credit: Dict[str, float] = {}
    credit_key: Dict[str, Optional[str]] = {}
    seen: Set[str] = set()

    # Roots first, then anything a malformed checkpoint left disconnected, so a
    # node that is not under `tree.root` still gets a credit rather than silently
    # keeping a stale one.
    starts = [k for k, n in nodes.items() if n.parent is None or n.parent not in nodes]
    for start in starts + list(nodes):
        if start in seen:
            continue
        stack: List[Tuple[str, bool]] = [(start, False)]
        while stack:
            k, finalising = stack.pop()
            n = nodes.get(k)
            if n is None:
                continue
            if finalising:
                # Own value first, then the best any child can offer.
                best_v = n.value if n.kernel is not None else _NOTHING
                best_k: Optional[str] = k if n.kernel is not None else None
                for c in n.children:
                    cv = credit.get(c)
                    if cv is not None and cv > best_v:
                        best_v, best_k = cv, credit_key.get(c)
                credit[k] = best_v
                credit_key[k] = best_k
                continue
            if k in seen:
                continue
            seen.add(k)
            stack.append((k, True))
            for c in n.children:
                if c not in seen:
                    stack.append((c, False))

    for k, n in nodes.items():
        v = credit.get(k, _NOTHING)
        n.credit = 0.0 if v == _NOTHING else float(v)
        n.credit_key = credit_key.get(k)

    tree.pv = set(principal_variation(tree))
    return {k: v for k, v in credit.items() if v != _NOTHING}


def principal_variation(tree: MonteCarloTreeSearch) -> List[str]:
    """The chain of nodes from the seed root to the one holding the best kernel.

    Walks UP from the node that holds the record, via `parent`. On a tree that is
    both exact and the shortest statement of it: there is exactly one route from
    the root to any node, so the lineage reported is necessarily the one the
    search actually walked.

    The graph version could not do this. A merged state had several parents, so
    "the parent" was not well defined and walking up would sometimes report a
    lineage the run never took; it had to descend from the root by ``credit_key``
    instead, and give up when the only route forward was a back-edge.

    Requires ``broadcast_credit()`` to have run; returns just the root otherwise.
    """
    root = tree.root
    if not root or root not in tree.nodes:
        return []
    target = tree.nodes[root].credit_key
    if not target or target not in tree.nodes:
        return [root]
    path = tree.path_to(target) or [root]
    # `path_to` walks parents, so it terminates at whatever node has none. If the
    # record is not under `root` (a checkpoint with a detached subtree), report
    # the lineage that exists rather than inventing a link to the root.
    return path


# --------------------------------------------------------------------- render
def _pct(new: float, old: float) -> str:
    if old <= 0:
        return "n/a"
    return f"{(new / old - 1.0) * 100.0:+.2f}%"


def _dead_ends(node: TreeNode, limit: int = 4) -> List[str]:
    """What was tried FROM this node and did not work, worst news first.

    Failures before mere underperformance: "this does not compile here" saves the
    model a whole round, "this compiled and was slower" only saves it a choice.
    """
    failed = [t for t in node.tried if not t.get("runnable")]
    weak = [t for t in node.tried
            if t.get("runnable") and float(t.get("value") or 0.0) < node.value]
    out = []
    for t in (failed + weak)[:limit]:
        mech = t.get("mechanism") or "(unnamed change)"
        if not t.get("runnable"):
            out.append(f"{mech} -> failed to compile/run")
        else:
            out.append(f"{mech} -> ran but scored {float(t.get('value') or 0.0):.4f}")
    return out


def render_pathway(tree: MonteCarloTreeSearch, *,
                   current_key: Optional[str] = None,
                   lam: float = 0.7) -> str:
    """The block injected into the optimization prompt.

    Returns "" when there is nothing to say or the feature is off. An empty
    heading is worse than silence, because the model will try to honour it.

    Deliberately reports the pathway as measurement, not instruction. The
    ``allowed_methods`` precedent in this repo is that the catalog's 26 entries
    did not contain the method which produced the best kernel to date, so binding
    the model to recorded knowledge would have forbidden the win. The closing
    line therefore says the pathway is where the run IS, and that leaving it is
    allowed with a reason -- not that it must be continued.
    """
    if not enabled() or not tree.nodes:
        return ""
    pv = principal_variation(tree)
    if len(pv) < 1:
        return ""
    root = tree.nodes[pv[0]]
    best_val = root.credit
    if best_val <= 0.0:
        return ""

    out = [
        "### THE WINNING PATHWAY SO FAR -- measured on this task, this run",
        "",
        "Every kernel this run has produced sits in a search tree. Below is the",
        "chain of edits that actually reached the best kernel, from the seed down.",
        "`reached` is the best score anywhere below that step, so it tells you what a",
        "step eventually led to -- not merely what it scored on the day it was made.",
        "",
    ]

    prev_val: Optional[float] = None
    for i, k in enumerate(pv):
        n = tree.nodes.get(k)
        if n is None:
            continue
        label = "SEED" if i == 0 else f"step {i}"
        via = n.via or ("initial kernel" if i == 0 else "(unnamed change)")
        head = f"- **{label}** via `{via}` -- scored {n.value:.4f}"
        if prev_val is not None:
            head += f" ({_pct(n.value, prev_val)} vs the step above)"
        head += f", reached {n.credit:.4f}"
        if k == current_key:
            head += "   <-- YOU ARE HERE"
        out.append(head)
        out.append(f"  - tree depth {n.depth}, visits N={n.N}, Q={n.q(lam):.3f}"
                   + (f", {n.failures} child(ren) never ran" if n.failures else ""))
        for d in _dead_ends(n):
            out.append(f"  - already tried from here: {d}")
        prev_val = n.value

    tip = tree.nodes.get(pv[-1])
    out += [
        "",
        f"The record kernel is {best_val:.4f}, held by the last step above.",
    ]
    if current_key and current_key not in set(pv):
        cur = tree.nodes.get(current_key)
        if cur is not None:
            out.append(
                f"The kernel you are being asked to optimize is NOT on this pathway: it "
                f"scored {cur.value:.4f} and its line has reached {cur.credit:.4f}, "
                f"against the pathway's {best_val:.4f}. Either beat that, or say plainly "
                f"which step above you would rather build on and why."
            )
    elif tip is not None and current_key == tip.key:
        out.append(
            "You are being asked to extend the record itself, so the bar is the number "
            "above and every edit below it has already been spent."
        )
    out += [
        "",
        "Use this as evidence, not as an order. The pathway is where the run HAS got to;",
        "it is not proof that the next win lies along it. If your profiling this round",
        "says the pathway is exhausted, say so and go elsewhere -- but say which step you",
        "are leaving and what measurement sent you.",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------- long term
def pathway_lesson(tree: MonteCarloTreeSearch, task: str) -> Optional[Dict[str, Any]]:
    """Distil the pathway into a summary of the chain that won this run.

    Returns None when the run has no pathway worth recording -- a single seed with
    no surviving edit teaches nothing, and an entry that says "the seed was the
    best" is prompt budget spent to tell the model something it will discover in
    round one anyway.

    Printed by the CLI for a human to read; nothing consumes it automatically.
    """
    pv = principal_variation(tree)
    if len(pv) < 2:
        return None
    steps = []
    for i, k in enumerate(pv[1:], 1):
        n = tree.nodes.get(k)
        if n is None:
            continue
        steps.append(f"{i}. {n.via or '(unnamed)'} -> {n.value:.4f}")
    if not steps:
        return None
    root = tree.nodes[pv[0]]
    tip = tree.nodes[pv[-1]]
    return {
        "id": f"pathway-{Path(task).stem}",
        "confidence": "measured",
        "claim": (f"The best chain of edits found so far runs {len(pv) - 1} step(s) from the "
                  f"seed and ends at {tip.value:.4f}.\n"),
        "evidence": ("Principal variation of the MCTS tree, by broadcast credit: seed "
                     f"{root.value:.4f} -> " + " -> ".join(steps) + ".\n"),
        "action": (f"Start from `{tip.via or 'the recorded tip'}` rather than rediscovering "
                   f"the chain above; spend this run's budget past step {len(pv) - 1}.\n"),
        "source": "utils/pathmemory.py principal_variation",
    }


# --------------------------------------------------------------------- loading
def load_tree_from_run(path: Path) -> Tuple[Optional[MonteCarloTreeSearch], Optional[Path]]:
    """Find a search tree under a batch folder, a task folder, or a file, and load it.

    Reuses ``utils.mcts_view.find_tree_file`` when it is importable so the two
    tools cannot disagree about which file holds the tree; falls back to the same
    search itself so this module stands alone if that one is ever removed.

    Both the current ``"mcts"`` key and the graph-era ``"mcgs"`` key are accepted:
    `MonteCarloTreeSearch.from_dict` flattens the older schema, so a run recorded
    before the tree existed still renders its pathway.
    """
    try:
        from utils.mcts_view import find_tree_file, load_tree  # type: ignore
        f = find_tree_file(path)
        if f is None:
            return None, None
        # load_tree returns (tree, note): a run that has not reached its first
        # round boundary has no tree, and that is a normal state, not an error.
        g, _note = load_tree(f)
        return g, f
    except Exception:
        pass
    cands: List[Path] = []
    p = Path(path)
    if p.is_file():
        cands = [p]
    else:
        cands = sorted(p.glob("**/checkpoint.json")) + sorted(p.glob("**/tree.json")) \
            + sorted(p.glob("**/graph.json"))
    for c in cands:
        try:
            d = json.loads(c.read_text(encoding="utf-8"))
        except Exception:
            continue
        g = (d.get("mcts") or d.get("mcgs")) if isinstance(d, dict) else None
        if g:
            return MonteCarloTreeSearch.from_dict(g), c
    return None, None


# ------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="batch folder, task folder, or tree file")
    ap.add_argument("--prompt", action="store_true",
                    help="print the block exactly as the optimization prompt sees it")
    ap.add_argument("--lesson", action="store_true",
                    help="print the memorybank lesson distilled from the pathway")
    ap.add_argument("--lam", type=float, default=0.7, help="Q mixing weight, for display")
    a = ap.parse_args(argv)

    tree, src = load_tree_from_run(a.path)
    if tree is None:
        print(f"no MCTS tree found under {a.path}", file=sys.stderr)
        return 1
    if tree.migration_note:
        print(f"[mcts] {tree.migration_note}", file=sys.stderr)
    broadcast_credit(tree)
    task = Path(src).parent.name if src else "unknown"

    if a.lesson:
        lesson = pathway_lesson(tree, task)
        if lesson is None:
            print("no pathway worth recording (fewer than two nodes on the PV)")
            return 0
        for k, v in lesson.items():
            print(f"{k}: {str(v).strip()}")
        return 0

    if a.prompt:
        block = render_pathway(tree, lam=a.lam)
        print(block or "(empty -- no pathway yet, or KERNELMEM_PATHWAY=0)")
        return 0

    pv = principal_variation(tree)
    print(f"tree: {src}")
    print(f"nodes: {len(tree.nodes)}  root: {tree.root}  visits: {tree.total_visits}")
    print(f"principal variation: {len(pv)} node(s)")
    for i, k in enumerate(pv):
        n = tree.nodes[k]
        tag = "SEED" if i == 0 else f"  +{i}"
        print(f"  {tag}  {k[:18]:20} via {str(n.via or '-'):28} "
              f"score={n.value:.4f}  reached={n.credit:.4f}  N={n.N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
