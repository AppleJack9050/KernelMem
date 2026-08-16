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
The information exists in the graph and nothing reads it.

``broadcast_credit()`` runs the other direction: a reverse pass over the DAG that
gives every state the value of the BEST kernel anywhere in its subtree. The seed
root therefore ends up holding the best score the whole run has reached, and every
state on the way holds the best score reachable *from it*. Two numbers per state,
answering different questions:

* ``Q``      -- how much this line has IMPROVED, mixing the mean and the max of
  its rollout rewards. What UCT already selects on.
* ``credit`` -- how high this line has ever REACHED, in absolute score.

They disagree because they are denominated differently, and this is the whole
reason the pass is worth running. ``backup`` stores a REWARD, and
``reward_from_gain`` is ``tanh`` of the rollout's *percentage gain over its
parent* -- so a line that climbed +50% off a bad seed outranks, on Q, a line that
climbed +4% off a good one, while being far worse on the only number the run
reports. The state holding the record can therefore lose the argmax to a line that
has never come near it. Q is the right signal for "is this line still yielding?";
it was never a claim about absolute standing, and nothing else in the graph was
making that claim either.

There is a second gap credit closes: ``Q`` is a scalar in [0, 1] with no link back
to a kernel, so it cannot tell a prompt WHICH kernel a line reached. ``credit_key``
can, and that is what makes the pathway renderable at all.

The three consumers
-------------------
1. ``render_pathway()`` -- the prompt block. Full context for every state on the
   principal variation: which mechanism produced it, what it scored, the delta it
   bought over its parent, and what was tried and FAILED off it. The model is
   told where the run actually is on the map rather than being handed one kernel
   with no lineage.
2. ``MonteCarloGraphSearch.pv`` + ``--mcgs_pv_bonus`` -- a selection bonus for
   staying on that pathway. OFF by default; see the note on convention below.
3. ``pathway_lesson()`` -- distils the pathway into a ``memorybank/lessons``
   entry, so the NEXT run starts knowing which chain won this one. This is the
   part that makes it long-term rather than merely within-run.

On defaults, and this repo's convention
---------------------------------------
The prompt block is ON by default, matching ``MEMORYBANK_LESSONS``: it is context,
and context that is measured this run cannot mislead the way an unsourced claim
can. ``--mcgs_pv_bonus`` is OFF by default, matching ``--mcgs_prior`` and
``--mcgs_epsilon``: it changes the search policy, and the convention here is that
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

from utils.mcgs import MonteCarloGraphSearch, StateNode  # noqa: E402

_ENV = "KERNELMEM_PATHWAY"

# A state with no runnable member holds no kernel, so it has no value of its own
# to contribute. It is not skipped -- its children may still carry credit through
# it -- it simply starts from "nothing here".
_NOTHING = float("-inf")


def enabled() -> bool:
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


# --------------------------------------------------------------------- credit
def broadcast_credit(graph: MonteCarloGraphSearch) -> Dict[str, float]:
    """Give every state the best value reachable from it, and refill ``graph.pv``.

    Iterative post-order rather than recursion: depth is bounded by ``max_depth``
    in a healthy run, but a graph loaded from a checkpoint written under
    ``features`` keying can be arbitrarily deep, and a RecursionError in a memory
    helper must never be able to take down a round.

    Cycles are tolerated, not assumed away. Under ``features`` keying a merge can
    install a back-edge, so this is a general digraph. A child still on the stack
    (grey) is skipped when its parent is finalised: its credit is not yet known,
    and waiting for it would deadlock. That under-counts only in the case where
    the ONLY route to a high value runs through a back-edge -- and a back-edge
    introduces no new kernel by construction, so the value it would carry is
    already reachable by the forward path that created it.

    Returns the credit map, and as a side effect writes ``credit``/``credit_key``
    onto every node and sets ``graph.pv`` to the principal variation's members.
    """
    nodes = graph.nodes
    credit: Dict[str, float] = {}
    credit_key: Dict[str, Optional[str]] = {}
    WHITE, GREY, BLACK = 0, 1, 2
    color: Dict[str, int] = {k: WHITE for k in nodes}

    for start in list(nodes):
        if color.get(start, WHITE) != WHITE:
            continue
        stack: List[Tuple[str, bool]] = [(start, False)]
        while stack:
            k, finalising = stack.pop()
            n = nodes.get(k)
            if n is None:
                continue
            if finalising:
                # Own value first, then the best any child can offer.
                best_v = n.rep_value if n.rep is not None else _NOTHING
                best_k: Optional[str] = k if n.rep is not None else None
                for c in n.children:
                    cv = credit.get(c)
                    if cv is not None and cv > best_v:
                        best_v, best_k = cv, credit_key.get(c)
                credit[k] = best_v
                credit_key[k] = best_k
                color[k] = BLACK
                continue
            if color.get(k, WHITE) != WHITE:
                continue
            color[k] = GREY
            stack.append((k, True))
            for c in n.children:
                if color.get(c, WHITE) == WHITE:
                    stack.append((c, False))

    for k, n in nodes.items():
        v = credit.get(k, _NOTHING)
        n.credit = 0.0 if v == _NOTHING else float(v)
        n.credit_key = credit_key.get(k)

    graph.pv = set(principal_variation(graph))
    return {k: v for k, v in credit.items() if v != _NOTHING}


def principal_variation(graph: MonteCarloGraphSearch) -> List[str]:
    """The chain of states from the seed root to the one holding the best kernel.

    Walks DOWN from the root rather than up from the best node, because a merged
    state has several parents and "the parent" is not well defined -- walking up
    would have to pick one and would sometimes report a lineage the run never took.
    Descending by ``credit_key`` is exact: at every step there is at least one
    child whose subtree contains the target, and following it reconstructs a route
    the search really walked.

    Requires ``broadcast_credit()`` to have run; returns just the root otherwise.
    """
    root = graph.root
    if not root or root not in graph.nodes:
        return []
    target = graph.nodes[root].credit_key
    path = [root]
    if not target or target == root:
        return path
    seen = {root}
    cur = root
    while cur != target:
        nxt = None
        for c in graph.nodes[cur].children:
            cn = graph.nodes.get(c)
            if cn is None or c in seen:
                continue
            if cn.credit_key == target:
                nxt = c
                break
        if nxt is None:      # target unreachable forward (back-edge only route)
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


# --------------------------------------------------------------------- render
def _pct(new: float, old: float) -> str:
    if old <= 0:
        return "n/a"
    return f"{(new / old - 1.0) * 100.0:+.2f}%"


def _dead_ends(node: StateNode, limit: int = 4) -> List[str]:
    """What was tried FROM this state and did not work, worst news first.

    Failures before mere underperformance: "this does not compile here" saves the
    model a whole round, "this compiled and was slower" only saves it a choice.
    """
    failed = [t for t in node.tried if not t.get("runnable")]
    weak = [t for t in node.tried
            if t.get("runnable") and float(t.get("value") or 0.0) < node.rep_value]
    out = []
    for t in (failed + weak)[:limit]:
        mech = t.get("mechanism") or "(unnamed change)"
        if not t.get("runnable"):
            out.append(f"{mech} -> failed to compile/run")
        else:
            out.append(f"{mech} -> ran but scored {float(t.get('value') or 0.0):.4f}")
    return out


def render_pathway(graph: MonteCarloGraphSearch, *,
                   current_key: Optional[str] = None,
                   lam: float = 0.7) -> str:
    """The block injected into the optimization prompt.

    Returns "" when there is nothing to say or the feature is off. An empty
    heading is worse than silence, because the model will try to honour it --
    the same reasoning as ``memorybank_lessons.render``.

    Deliberately reports the pathway as measurement, not instruction. The
    ``allowed_methods`` precedent in this repo is that the catalog's 26 entries
    did not contain the method which produced the best kernel to date, so binding
    the model to recorded knowledge would have forbidden the win. The closing
    line therefore says the pathway is where the run IS, and that leaving it is
    allowed with a reason -- not that it must be continued.
    """
    if not enabled() or not graph.nodes:
        return ""
    pv = principal_variation(graph)
    if len(pv) < 1:
        return ""
    root = graph.nodes[pv[0]]
    best_val = root.credit
    if best_val <= 0.0:
        return ""

    out = [
        "### THE WINNING PATHWAY SO FAR -- measured on this task, this run",
        "",
        "Every kernel this run has produced sits in a graph of states. Below is the",
        "chain of edits that actually reached the best kernel, from the seed down.",
        "`reached` is the best score anywhere below that step, so it tells you what a",
        "step eventually led to -- not merely what it scored on the day it was made.",
        "",
    ]

    prev_val: Optional[float] = None
    for i, k in enumerate(pv):
        n = graph.nodes.get(k)
        if n is None:
            continue
        label = "SEED" if i == 0 else f"step {i}"
        via = n.via or ("initial kernel" if i == 0 else "(unnamed change)")
        head = f"- **{label}** via `{via}` -- scored {n.rep_value:.4f}"
        if prev_val is not None:
            head += f" ({_pct(n.rep_value, prev_val)} vs the step above)"
        head += f", reached {n.credit:.4f}"
        if k == current_key:
            head += "   <-- YOU ARE HERE"
        out.append(head)
        out.append(f"  - state depth {n.depth}, visits N={n.N}, Q={n.q(lam):.3f}"
                   + (f", {n.failures} child(ren) never ran" if n.failures else ""))
        for d in _dead_ends(n):
            out.append(f"  - already tried from here: {d}")
        prev_val = n.rep_value

    tip = graph.nodes.get(pv[-1])
    out += [
        "",
        f"The record kernel is {best_val:.4f}, held by the last step above.",
    ]
    if current_key and current_key not in set(pv):
        cur = graph.nodes.get(current_key)
        if cur is not None:
            out.append(
                f"The kernel you are being asked to optimize is NOT on this pathway: it "
                f"scored {cur.rep_value:.4f} and its line has reached {cur.credit:.4f}, "
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
def pathway_lesson(graph: MonteCarloGraphSearch, task: str) -> Optional[Dict[str, Any]]:
    """Distil the pathway into a ``memorybank/lessons`` entry for the NEXT run.

    Returns None when the run has no pathway worth recording -- a single seed with
    no surviving edit teaches nothing, and an entry that says "the seed was the
    best" is prompt budget spent to tell the model something it will discover in
    round one anyway.

    Not written automatically. ``memorybank/lessons/<task>.yaml`` is hand-curated
    and its stated bar is that every entry carries its evidence; an automated
    writer appending to it every run is how a curated file becomes a log. The CLI
    prints the entry and ``--write`` commits it.
    """
    pv = principal_variation(graph)
    if len(pv) < 2:
        return None
    steps = []
    for i, k in enumerate(pv[1:], 1):
        n = graph.nodes.get(k)
        if n is None:
            continue
        steps.append(f"{i}. {n.via or '(unnamed)'} -> {n.rep_value:.4f}")
    if not steps:
        return None
    root = graph.nodes[pv[0]]
    tip = graph.nodes[pv[-1]]
    return {
        "id": f"pathway-{Path(task).stem}",
        "confidence": "measured",
        "claim": (f"The best chain of edits found so far runs {len(pv) - 1} step(s) from the "
                  f"seed and ends at {tip.rep_value:.4f}.\n"),
        "evidence": ("Principal variation of the MCGS graph, by broadcast credit: seed "
                     f"{root.rep_value:.4f} -> " + " -> ".join(steps) + ".\n"),
        "action": (f"Start from `{tip.via or 'the recorded tip'}` rather than rediscovering "
                   f"the chain above; spend this run's budget past step {len(pv) - 1}.\n"),
        "source": "utils/pathmemory.py principal_variation",
    }


# --------------------------------------------------------------------- loading
def load_graph_from_run(path: Path) -> Tuple[Optional[MonteCarloGraphSearch], Optional[Path]]:
    """Find a graph under a batch folder, a task folder, or a file, and load it.

    Reuses ``utils.mcgs_view.find_graph_file`` when it is importable so the two
    tools cannot disagree about which file is the graph; falls back to the same
    search itself so this module stands alone if that one is ever removed.
    """
    try:
        from utils.mcgs_view import find_graph_file, load_graph  # type: ignore
        f = find_graph_file(path)
        if f is None:
            return None, None
        # load_graph returns (graph, note): a run that has not reached its first
        # round boundary has no graph, and that is a normal state, not an error.
        g, _note = load_graph(f)
        return g, f
    except Exception:
        pass
    cands: List[Path] = []
    p = Path(path)
    if p.is_file():
        cands = [p]
    else:
        cands = sorted(p.glob("**/checkpoint.json")) + sorted(p.glob("**/graph.json"))
    for c in cands:
        try:
            d = json.loads(c.read_text(encoding="utf-8"))
        except Exception:
            continue
        g = d.get("mcgs") if isinstance(d, dict) else None
        if g:
            return MonteCarloGraphSearch.from_dict(g), c
    return None, None


# ------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="batch folder, task folder, or graph file")
    ap.add_argument("--prompt", action="store_true",
                    help="print the block exactly as the optimization prompt sees it")
    ap.add_argument("--lesson", action="store_true",
                    help="print the memorybank lesson distilled from the pathway")
    ap.add_argument("--write", action="store_true",
                    help="with --lesson, commit it to memorybank/lessons/<task>.yaml")
    ap.add_argument("--lam", type=float, default=0.7, help="Q mixing weight, for display")
    a = ap.parse_args(argv)

    graph, src = load_graph_from_run(a.path)
    if graph is None:
        print(f"no MCGS graph found under {a.path}", file=sys.stderr)
        return 1
    broadcast_credit(graph)
    task = Path(src).parent.name if src else "unknown"

    if a.lesson:
        lesson = pathway_lesson(graph, task)
        if lesson is None:
            print("no pathway worth recording (fewer than two states on the PV)")
            return 0
        if a.write:
            from utils.memorybank_lessons import load as _load, save as _save
            data = _load(task)
            data["lessons"] = [l for l in (data.get("lessons") or [])
                               if l.get("id") != lesson["id"]] + [lesson]
            print(f"wrote {lesson['id']} to {_save(task, data)}")
        else:
            for k, v in lesson.items():
                print(f"{k}: {str(v).strip()}")
        return 0

    if a.prompt:
        block = render_pathway(graph, lam=a.lam)
        print(block or "(empty -- no pathway yet, or KERNELMEM_PATHWAY=0)")
        return 0

    pv = principal_variation(graph)
    print(f"graph: {src}")
    print(f"states: {len(graph.nodes)}  root: {graph.root}  visits: {graph.total_visits}")
    print(f"principal variation: {len(pv)} state(s)")
    for i, k in enumerate(pv):
        n = graph.nodes[k]
        tag = "SEED" if i == 0 else f"  +{i}"
        print(f"  {tag}  {k[:18]:20} via {str(n.via or '-'):28} "
              f"score={n.rep_value:.4f}  reached={n.credit:.4f}  N={n.N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
