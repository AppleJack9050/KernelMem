"""Live view of the MCGS graph as a run builds it.

    python -m utils.mcgs_view run/<stamp>_<task>_<tag> --watch

Reads the graph a run has already written to disk and renders it. It NEVER
imports or touches the running loop: the graph is persisted every round boundary
(``checkpoint.json["mcgs"]``, main_memory_latest.py:3369) and every rollout
(``graph.json["mcgs"]``, utils/mcgs_quick.py), so a watcher polling that file
gets the same object the search is using, one round behind. That is the whole
reason this is a separate module rather than a print inside the loop -- it can be
started, stopped and restarted against a run that is already hours in, and a bug
in the renderer cannot take the run down with it.

What the shape tells you, which the score curve cannot:

* **wide and shallow** -- progressive widening is spending its budget on new
  children of the root instead of developing any line. Expected early; if it
  persists, the edits are not compounding.
* **one deep spine** -- the search has committed. Good if the spine is gaining,
  a local optimum if `Q` has gone flat along it.
* **`==` markers** -- transpositions: two different edit orders reached one
  state. This is the merging MCGS exists to exploit, and the measured note in
  `--mcgs_state_key`'s help is that genuine commuting transpositions are ~absent
  here, so several of these is itself a finding.
* **`!!` markers** -- a back-edge, i.e. a cycle in the DAG. Under `mechanisms`
  keying this cannot happen (the multiset grows by one per edge); under
  `features` keying it can, and `select()`'s trajectory mask is what keeps the
  descent from looping on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.mcgs import MonteCarloGraphSearch, StateNode

# checkpoint.json is the main loop's; graph.json is mcgs_quick's. Both store the
# graph under the same "mcgs" key, so one reader serves both.
_SOURCES = ("checkpoint.json", "graph.json")


def find_graph_file(path: Path) -> Optional[Path]:
    """Resolve *path* to the file holding the graph.

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


def load_graph(f: Path) -> Tuple[Optional[MonteCarloGraphSearch], str]:
    """Return (graph, note). A partial write is a normal event, not an error.

    The writer renames a temp file into place, so a torn read should be
    impossible -- but a run that has not reached its first round boundary has no
    graph at all, and that is the common case when a watcher starts.
    """
    try:
        blob = json.loads(f.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no graph file yet"
    except json.JSONDecodeError as exc:
        return None, f"file is mid-write ({exc.__class__.__name__}); will retry"
    except OSError as exc:
        return None, f"unreadable: {exc}"
    m = blob.get("mcgs")
    if not m:
        if "optimization_tree" in blob:
            return None, ("this checkpoint carries no MCGS graph -- it is from a "
                          "--search ratchet run, or predates --search mcgs")
        return None, "no 'mcgs' key in this file"
    if not (m.get("nodes")):
        return None, "graph is present but still empty (no round has completed)"
    return MonteCarloGraphSearch.from_dict(m), ""


def _short(key: str, width: int = 10) -> str:
    """Keys are `m:<sha1>`; the prefix carries the mode, the hash is noise."""
    if len(key) <= width:
        return key
    return key[:width - 1] + "…"


def _bar(q: float, width: int = 10) -> str:
    filled = max(0, min(width, int(round(q * width))))
    return "█" * filled + "·" * (width - filled)


def render(graph: MonteCarloGraphSearch, *, lam: float = 0.7,
           max_width: int = 200, show_mechanism: bool = True) -> str:
    """The graph as an indented DAG, root first."""
    out: List[str] = []
    st = graph.stats()
    best = graph.best()
    out.append(
        f"states {st['states']}  kernels {st['kernels']}  merged {st['merged_states']}"
        f"  splits-refused {st['splits_refused']}  visits {st['total_visits']}"
        f"  max-depth {st['max_depth_seen']}")
    if best is not None:
        out.append(f"best   {_short(best.key, 14)}  value {best.rep_value:.4f}  via {best.rep}")
    out.append("")

    # A node with several parents is expanded ONCE and referenced elsewhere;
    # otherwise a merged state's whole subtree is duplicated under every parent,
    # which is the picture a DAG exists to stop you drawing.
    #
    # Where it expands is decided by a BFS from the root, not by whichever route
    # the render happens to reach first: children are printed best-Q-first, so a
    # depth-first "first arrival" rule can expand a shallow node deep inside an
    # unrelated branch -- e.g. a root child appearing under a back-edge four
    # levels down. BFS gives every node its shallowest position.
    from collections import deque
    canon: Dict[str, Optional[str]] = {}
    if graph.root and graph.root in graph.nodes:
        canon[graph.root] = None
        dq = deque([graph.root])
        while dq:
            k = dq.popleft()
            for c in graph.nodes[k].children:
                if c in graph.nodes and c not in canon:
                    canon[c] = k
                    dq.append(c)
    expanded: Set[str] = set()

    def walk(key: str, prefix: str, is_last: bool, ancestors: Set[str],
             via_edge: Optional[str], parent: Optional[str] = None) -> None:
        n = graph.nodes.get(key)
        if n is None:
            return
        elbow = "" if not prefix and not via_edge else ("└─ " if is_last else "├─ ")
        # A back-edge: this node is its own ancestor on the current branch. Print
        # and STOP -- recursing here is the loop select() is guarded against.
        if key in ancestors:
            out.append(f"{prefix}{elbow}!! {_short(key)}  back-edge (cycle)")
            return
        if key in expanded or canon.get(key, parent) != parent:
            where = "shown above" if key in expanded else "shown at its shallowest route"
            out.append(f"{prefix}{elbow}== {_short(key)}  transposition, {where}")
            return
        expanded.add(key)

        # `rep is None`, not `not runnable`: observe() only ever ORs `runnable`
        # to True, so a state whose every member failed still reads runnable.
        # Having no representative is what actually makes a state unexpandable,
        # and it is the same test _selectable_children uses.
        mark = " " if n.rep is not None else "x"
        best_mark = "*" if best is not None and n.key == best.key else " "
        label = (f"{best_mark}{mark} {_short(key)}"
                 f"  N={n.N:<3d} Q={n.q(lam):.3f} [{_bar(n.q(lam))}]"
                 f"  v={n.rep_value:.4f}"
                 f"  d={n.depth}"
                 f"  m={len(n.members)}")
        if n.failures:
            label += f"  fail={n.failures}"
        if len(n.parents) > 1:
            label += f"  parents={len(n.parents)}"
        if show_mechanism and via_edge:
            label += f"  <- {via_edge}"
        out.append((prefix + elbow + label)[:max_width])

        kids = [k for k in n.children if k in graph.nodes]
        # Best first: the interesting line should be readable without scrolling.
        kids.sort(key=lambda k: -graph.nodes[k].q(lam))
        child_prefix = prefix + ("" if not via_edge and not prefix
                                 else ("   " if is_last else "│  "))
        for i, k in enumerate(kids):
            walk(k, child_prefix, i == len(kids) - 1,
                 ancestors | {key}, graph.nodes[k].via, key)

    if graph.root and graph.root in graph.nodes:
        walk(graph.root, "", True, set(), None)
    else:
        out.append("(no root yet)")

    # Anything unreachable from the root -- a state observed with no parent, or
    # whose parent was never recorded. Silently dropping these would make the
    # picture disagree with the node count in the header.
    orphans = [k for k in graph.nodes if k not in expanded]
    if orphans:
        out.append("")
        out.append(f"not reachable from the root ({len(orphans)}):")
        for k in sorted(orphans)[:20]:
            n = graph.nodes[k]
            out.append(f"   {_short(k)}  N={n.N} v={n.rep_value:.4f} d={n.depth}")
        if len(orphans) > 20:
            out.append(f"   ... and {len(orphans) - 20} more")
    return "\n".join(out)


def render_file(f: Path, **kw) -> str:
    graph, note = load_graph(f)
    if graph is None:
        return f"[mcgs-view] {note}"
    return render(graph, **kw)


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
                print(f"[mcgs-view] {f}  @{stamp}\n")
                print(body, flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[mcgs-view] stopped.", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render the MCGS graph a run is building.")
    ap.add_argument("path", type=Path,
                    help="a batch folder, a task folder, or a checkpoint.json / graph.json")
    ap.add_argument("--watch", action="store_true",
                    help="re-render whenever the file changes (Ctrl-C to stop)")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between polls in --watch (default 5)")
    ap.add_argument("--lam", type=float, default=0.7,
                    help="lambda for Q = (1-lam)*mean + lam*max; match the run's --mcgs_lam")
    ap.add_argument("--no_clear", action="store_true",
                    help="do not clear the screen between renders")
    ap.add_argument("--no_mechanism", action="store_true",
                    help="omit the edge mechanism labels")
    a = ap.parse_args(argv)

    f = find_graph_file(a.path)
    if f is None:
        # In --watch this is worth waiting out: pointing the viewer at a run that
        # has not finished its first round is the normal way to start it.
        if a.watch and a.path.is_dir():
            print(f"[mcgs-view] no {' or '.join(_SOURCES)} under {a.path} yet; waiting ...",
                  flush=True)
            while f is None:
                time.sleep(a.interval)
                f = find_graph_file(a.path)
        else:
            print(f"[mcgs-view] found no {' or '.join(_SOURCES)} under {a.path}",
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
