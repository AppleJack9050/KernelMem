"""Live view of the MCTS tree as a run builds it.

    python -m utils.mcts_view run/<stamp>_<task>_<tag> --watch

Reads the tree a run has already written to disk and renders it. It NEVER
imports or touches the running loop: the tree is persisted every round boundary
(``checkpoint.json["mcts"]``) and every rollout (``tree.json["mcts"]``,
utils/mcts_quick.py), so a watcher polling that file gets the same object the
search is using, one round behind. That is the whole reason this is a separate
module rather than a print inside the loop -- it can be started, stopped and
restarted against a run that is already hours in, and a bug in the renderer
cannot take the run down with it.

What the shape tells you, which the score curve cannot:

* **wide and shallow** -- progressive widening is spending its budget on new
  children of the root instead of developing any line. Expected early; if it
  persists, the edits are not compounding.
* **one deep spine** -- the search has committed. Good if the spine is gaining,
  a local optimum if `Q` has gone flat along it.
* **`x` markers** -- a node whose kernel never ran. It keeps its visits, so its
  parent is charged for having produced it, but it is not a destination: there
  is no code to hand the model.
* **many nodes at `N=1`** -- the budget is being spread over first visits and no
  node has enough statistics for Q to mean anything. The measured regime here is
  ~30 evaluations per run, so this is the failure mode to watch for.

The graph-era version of this renderer also drew `==` transposition markers and
`!!` back-edges, because a merged state could be reached by several routes and
could close a cycle. Neither exists on a tree, so neither is drawn: every node
appears exactly once, under its only parent. A `"mcgs"` blob is still readable --
it is flattened on load and the note says what that dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.mcts import MonteCarloTreeSearch, TreeNode

# checkpoint.json is the main loop's; tree.json is mcts_quick's. graph.json is
# the quick path's graph-era filename, still read so an old run still renders.
_SOURCES = ("checkpoint.json", "tree.json", "graph.json")


def find_tree_file(path: Path) -> Optional[Path]:
    """Resolve *path* to the file holding the tree.

    Accepts the file itself, a task folder, or a batch folder (in which case the
    single task inside it is found). Prefers the most recently modified match, so
    pointing at a batch folder with several tasks follows the active one.
    """
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    hits: List[Path] = []
    for name in _SOURCES:
        hits.extend(path.glob(name))
        hits.extend(path.glob(f"*/{name}"))
    hits = [h for h in hits if h.is_file()]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def load_tree(f: Path) -> Tuple[Optional[MonteCarloTreeSearch], str]:
    """Return (tree, note). A partial write is a normal event, not an error.

    The writer renames a temp file into place, so a torn read should be
    impossible -- but a run that has not reached its first round boundary has no
    tree at all, and that is the common case when a watcher starts.
    """
    try:
        blob = json.loads(f.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no tree file yet"
    except json.JSONDecodeError as exc:
        return None, f"file is mid-write ({exc.__class__.__name__}); will retry"
    except OSError as exc:
        return None, f"unreadable: {exc}"
    # "mcgs" is the graph-era key; from_dict flattens that schema to a tree.
    m = blob.get("mcts") or blob.get("mcgs")
    if not m:
        if "optimization_tree" in blob:
            return None, ("this checkpoint carries no MCTS tree -- it is from a "
                          "--search ratchet run, or predates --search mcts")
        return None, "no 'mcts' key in this file"
    if not (m.get("nodes")):
        return None, "tree is present but still empty (no round has completed)"
    return MonteCarloTreeSearch.from_dict(m), ""


def _short(key: str, width: int = 10) -> str:
    """Node ids are `nNNNN`; graph-era keys are `m:<sha1>` and need truncating."""
    if len(key) <= width:
        return key
    return key[:width - 1] + "…"


def _bar(q: float, width: int = 10) -> str:
    filled = max(0, min(width, int(round(q * width))))
    return "█" * filled + "·" * (width - filled)


def render(tree: MonteCarloTreeSearch, *, lam: float = 0.7,
           max_width: int = 200, show_mechanism: bool = True) -> str:
    """The tree as an indented outline, root first."""
    out: List[str] = []
    st = tree.stats()
    best = tree.best()
    out.append(
        f"nodes {st['nodes']}  leaves {st['leaves']}  at-N<=1 {st['nodes_at_N<=1']}"
        f"  visits {st['total_visits']}  max-depth {st['max_depth_seen']}"
        f"  mean-N {st['mean_N']:.2f}")
    if best is not None:
        out.append(f"best   {_short(best.key, 14)}  value {best.value:.4f}  via {best.kernel}")
    if tree.migration_note:
        out.append(f"note   {tree.migration_note}")
    out.append("")

    # Every node appears exactly once, under its only parent. No canonicalisation
    # pass, no ancestor set, no already-expanded set: on a tree a depth-first walk
    # from the root visits each node once by construction, which is the whole
    # bookkeeping the DAG version needed to fake.
    # Explicit stack, not recursion. `--mcts_max_depth` defaults to 10 and caps
    # SELECTION, not the tree: a checkpoint written at a raised cap, or one
    # hand-edited, can carry a spine far past sys.getrecursionlimit(), and a
    # renderer that dies on the run it is meant to be watching is worse than no
    # renderer. Same reason utils.pathmemory.broadcast_credit is iterative.
    drawn: set = set()
    seen_root = bool(tree.root and tree.root in tree.nodes)
    if seen_root:
        stack: List[Tuple[str, str, bool, Optional[str]]] = [(tree.root, "", True, None)]
        while stack:
            key, prefix, is_last, via_edge = stack.pop()
            n = tree.nodes.get(key)
            if n is None or key in drawn:
                continue
            drawn.add(key)
            elbow = "" if not prefix and not via_edge else ("└─ " if is_last else "├─ ")
            # `kernel is None`, not `not runnable`: a node whose kernel failed keeps
            # runnable=False AND kernel=None, and having no kernel is what actually
            # makes it unexpandable. Same test _selectable_children uses.
            mark = " " if (n.kernel is not None and n.runnable) else "x"
            best_mark = "*" if best is not None and n.key == best.key else " "
            label = (f"{best_mark}{mark} {_short(key)}"
                     f"  N={n.N:<3d} Q={n.q(lam):.3f} [{_bar(n.q(lam))}]"
                     f"  v={n.value:.4f}"
                     f"  d={n.depth}")
            if n.failures:
                label += f"  fail={n.failures}"
            if show_mechanism and via_edge:
                label += f"  <- {via_edge}"
            out.append((prefix + elbow + label)[:max_width])

            kids = [k for k in n.children if k in tree.nodes and k not in drawn]
            # Best first: the interesting line should be readable without
            # scrolling. Pushed in REVERSE so the stack pops them best-first, which
            # keeps the output identical to the recursive version's.
            kids.sort(key=lambda k: -tree.nodes[k].q(lam))
            child_prefix = prefix + ("" if not via_edge and not prefix
                                     else ("   " if is_last else "│  "))
            for i in range(len(kids) - 1, -1, -1):
                stack.append((kids[i], child_prefix, i == len(kids) - 1,
                              tree.nodes[kids[i]].via))
    else:
        out.append("(no root yet)")

    # Anything not under the root -- a node observed with no parent, or one whose
    # parent went missing. `from_dict` drops these when it flattens a graph-era
    # checkpoint, so this should only ever fire on a live tree that was built
    # oddly; silently omitting them would make the picture disagree with the
    # node count in the header. `drawn` is filled by walk() itself rather than by
    # a second traversal: the walk already visits exactly the reachable set, and
    # a separate recursive pass would be one more thing to blow the stack on a
    # deep tree.
    orphans = [k for k in tree.nodes if k not in drawn]
    if orphans:
        out.append("")
        out.append(f"not reachable from the root ({len(orphans)}):")
        for k in sorted(orphans)[:20]:
            n = tree.nodes[k]
            out.append(f"   {_short(k)}  N={n.N} v={n.value:.4f} d={n.depth}")
        if len(orphans) > 20:
            out.append(f"   ... and {len(orphans) - 20} more")
    return "\n".join(out)


def render_file(f: Path, **kw) -> str:
    tree, note = load_tree(f)
    if tree is None:
        return f"[mcts-view] {note}"
    return render(tree, **kw)


def watch(f: Path, interval: float = 5.0, *, clear: bool = True, **kw) -> None:
    """Re-render whenever the file changes. Ctrl-C to stop.

    Polls mtime+size rather than inotify: the file is renamed into place, which
    some inotify setups report as a delete of the watched inode.
    """
    last: Optional[Tuple[float, int]] = None
    first = True
    try:
        while True:
            try:
                stat = f.stat()
                sig = (stat.st_mtime, stat.st_size)
            except OSError:
                sig = None
            if sig != last or first:
                last, first = sig, False
                body = render_file(f, **kw)
                if clear and sys.stdout.isatty():
                    sys.stdout.write("\033[H\033[2J")
                stamp = time.strftime("%H:%M:%S")
                print(f"[mcts-view] {f}  @{stamp}\n")
                print(body, flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[mcts-view] stopped.", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render the MCTS tree a run is building.")
    ap.add_argument("path", type=Path,
                    help="a batch folder, a task folder, or a checkpoint.json / tree.json")
    ap.add_argument("--watch", action="store_true",
                    help="re-render whenever the file changes (Ctrl-C to stop)")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between polls in --watch (default 5)")
    ap.add_argument("--lam", type=float, default=0.7,
                    help="lambda for Q = (1-lam)*mean + lam*max; match the run's --mcts_lam")
    ap.add_argument("--no_clear", action="store_true",
                    help="do not clear the screen between renders")
    ap.add_argument("--no_mechanism", action="store_true",
                    help="omit the edge mechanism labels")
    a = ap.parse_args(argv)

    f = find_tree_file(a.path)
    if f is None:
        # In --watch this is worth waiting out: pointing the viewer at a run that
        # has not finished its first round is the normal way to start it.
        if a.watch and a.path.is_dir():
            print(f"[mcts-view] no {' or '.join(_SOURCES)} under {a.path} yet; waiting ...",
                  flush=True)
            while f is None:
                time.sleep(a.interval)
                f = find_tree_file(a.path)
        else:
            print(f"[mcts-view] found no {' or '.join(_SOURCES)} under {a.path}",
                  file=sys.stderr)
            return 2

    kw = dict(lam=a.lam, show_mechanism=not a.no_mechanism)
    if a.watch:
        watch(f, a.interval, clear=not a.no_clear, **kw)
        return 0
    print(render_file(f, **kw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
