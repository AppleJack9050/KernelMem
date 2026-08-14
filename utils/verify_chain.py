#!/usr/bin/env python
"""Re-measure a run's ancestor chain, paired, to separate real gains from drift.

Why this exists
---------------
`optimization_tree.json` records each kernel's gain over its parent as
``speedup_child / speedup_parent - 1``. Those two speedups were taken in
different rounds, often hours apart, so every recorded gain carries GPU drift.
On an RTX 5090 that drift is +0.9..+1.7%: one byte-identical kernel measured
1.2014 ms and then 1.2141 ms thirty minutes later, +1.06%. The gains being
compared against it are 0.9-1.8%, so the noise is the same size as the signal.

That is not hypothetical. Ablating the five mechanisms of the exp3 round-9 base
found three of them -- claimed +0.90%, +1.12% and +1.18% -- measuring 0.09-0.17%
when removed and re-measured side by side, i.e. indistinguishable from zero.

Every kernel in the chain is still on disk, so the fix needs no code generation:
measure each child against its own parent, interleaved in ONE session, and the
drift becomes common-mode and cancels. What comes back is a same-session number
with an error bar for the exact claim the judge prompt makes ("+1.12% over its
parent").

What this does and does not establish
-------------------------------------
Each round's kernel is a full LLM rewrite, not a surgical patch, so a child
differs from its parent by the named mechanism PLUS whatever else changed in the
rewrite. This therefore answers "was this ROUND a real improvement", not "was the
named MECHANISM real" -- only a surgical ablation answers the latter. It is still
the right correction for the claim as stated, because that claim is about the
round, not the mechanism.

This is a ONE-TIME backfill. Once the ratchet records a paired verdict on every
advance, new edges are measured this way automatically; this only fills in edges
created before that existed.

Usage
-----
    export SOLBENCH_SRC=/path/to/SOL-ExecBench/src      # if the task needs it
    python -m utils.verify_chain run/<stamp>/<task>/ --reps 3
    python -m utils.verify_chain run/<stamp>/<task>/ --write    # persist verdicts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import lazily inside the worker so the parent process never initialises CUDA.


def _ext_name(src: str) -> Optional[str]:
    """The load_inline extension name, or None if the file does not use one."""
    i = src.find("load_inline(")
    if i < 0:
        return None
    m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', src[i:i + 800])
    return m.group(1) if m else None


def _chain(kernels: Dict[str, Any], head: str) -> List[str]:
    """Ancestor names oldest-first, following `parent` links from `head`."""
    out: List[str] = []
    cur: Optional[str] = head
    while cur and cur in kernels and cur not in out:
        out.append(cur)
        cur = kernels[cur].get("parent")
    out.reverse()
    return out


def _prep_pair(parent: Path, child: Path, tmp: Path, idx: int) -> Tuple[Path, Path, bool]:
    """Copy both kernels somewhere safe, renaming the extension if they collide.

    This is not cosmetic. Paired measurement requires both kernels live in ONE
    process, and load_inline keys its build cache -- and Python its module cache
    -- on the extension name. In the exp3 run rounds 4 and 5 BOTH emit
    `vae_resblock_fused_ms_graph_ext`, so importing the second would hand back
    the first one's module and the pair would measure exactly 0.00%: the tool
    would "confirm" the round did nothing, for entirely the wrong reason.

    Only rename on an actual collision. A fresh name forces a full rebuild
    (~1-3 min), while the original names are already in torch's cache and load
    in seconds, so renaming everything unconditionally would be far slower for
    no benefit.
    """
    p_src, c_src = parent.read_text(), child.read_text()
    p_ext, c_ext = _ext_name(p_src), _ext_name(c_src)
    collide = bool(p_ext) and p_ext == c_ext

    p_out, c_out = tmp / f"p{idx}_{parent.stem}.py", tmp / f"c{idx}_{child.stem}.py"
    p_out.write_text(p_src)
    c_out.write_text(c_src.replace(c_ext, f"{c_ext}_vfy{idx}") if collide else c_src)
    return p_out, c_out, collide


def _worker(reference: str, base_py: str, cand_py: str, device: int, warmup: int,
            repeat: int, tol: float, margin: float, min_reps: int, max_reps: int,
            conn) -> None:
    """Subprocess entry: one pair, measured start to finish in this process."""
    try:
        import torch
        from utils.paired_bench import adaptive_paired_verdict
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        conn.send(("ok", adaptive_paired_verdict(
            Path(reference), Path(base_py), Path(cand_py), device=device,
            warmup=warmup, repeat=repeat, tol=tol, margin=margin,
            min_reps=min_reps, max_reps=max_reps)))
    except Exception as e:
        conn.send(("err", f"{e.__class__.__name__}: {e}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _measure(reference: Path, parent: Path, child: Path, *, device: int, warmup: int,
             repeat: int, tol: float, margin: float, min_reps: int, max_reps: int,
             timeout: int = 2400) -> Optional[Dict[str, Any]]:
    """Measure one pair in its own process. Returns None on any failure.

    One process per pair, rather than one for the whole chain: a CUDA context
    with ten kernel extensions loaded into it is its own source of trouble, and
    a single bad kernel then takes down every remaining edge.
    """
    from multiprocessing import get_context

    ctx = get_context("spawn")
    rx, tx = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker, args=(str(reference), str(parent), str(child),
                                          device, warmup, repeat, tol, margin,
                                          min_reps, max_reps, tx))
    p.start()
    try:
        tx.close()
    except Exception:
        pass
    p.join(timeout=timeout)
    if p.is_alive():
        print(f"    timed out after {timeout}s", flush=True)
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
        return None
    payload = rx.recv() if rx.poll() else None
    if not payload or payload[0] != "ok" or payload[1] is None:
        if payload and payload[0] == "err":
            print(f"    failed: {payload[1]}", flush=True)
        return None
    return payload[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-measure a run's ancestor chain with paired, same-session timing.")
    ap.add_argument("task_dir", type=Path,
                    help="Per-task folder containing optimization_tree.json and code/")
    ap.add_argument("--head", default=None,
                    help="Kernel to walk back from (default: the tree's base_kernel)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="Reference .py (default: the tree's `task` field)")
    ap.add_argument("--reps", type=int, default=3, help="Minimum interleaved repeats")
    ap.add_argument("--max_reps", type=int, default=8, help="Cap when the call is close")
    ap.add_argument("--margin", type=float, default=0.005,
                    help="Margin the verdict is judged against. This deliberately does NOT "
                         "track main_memory_latest.py --base_margin, which is now a hard 5%% "
                         "adoption gate. It was briefly set to 0.05 to match and that voids "
                         "this tool: adaptive_paired_verdict stops adding reps as soon as the "
                         "DECISION is safe, and against a 5%% margin a ~1%% effect is "
                         "unambiguously not a 5%% improvement on the first check -- so it "
                         "breaks at --reps and returns resolved=True having never resolved "
                         "whether the edge is +1%% or 0%%, which is the only question this "
                         "tool exists to answer. The adoption gate asks 'is it big enough to "
                         "adopt'; this asks 'how big is it, and is it distinguishable from "
                         "zero'. Keep them separate.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--write", action="store_true",
                    help="Persist each verdict onto its tree node. Off by default: "
                         "this edits a run artifact, so look at the report first.")
    a = ap.parse_args()

    tree_path = a.task_dir / "optimization_tree.json"
    if not tree_path.exists():
        print(f"no optimization_tree.json in {a.task_dir}", file=sys.stderr)
        return 2
    tree = json.loads(tree_path.read_text())
    kernels = tree.get("kernels") or {}
    head = a.head or tree.get("base_kernel")
    if not head or head not in kernels:
        print(f"base_kernel {head!r} not present in the tree", file=sys.stderr)
        return 2

    reference = a.reference or Path(tree.get("task", ""))
    if not reference.exists():
        print(f"reference {reference} not found; pass --reference", file=sys.stderr)
        return 2

    chain = _chain(kernels, head)
    code = a.task_dir / "code"
    print(f"chain of {len(chain)} kernels, {len(chain) - 1} edges to measure "
          f"(reference {reference})\n")

    results = []
    with tempfile.TemporaryDirectory(prefix="verify_chain_") as td:
        tmp = Path(td)
        for i in range(len(chain) - 1):
            pn, cn = chain[i], chain[i + 1]
            pf, cf = code / f"{pn}.py", code / f"{cn}.py"
            node = kernels[cn]
            strat = node.get("strategy") or {}
            method = (strat.get("method_name") if isinstance(strat, dict) else None) or "?"
            if not pf.exists() or not cf.exists():
                print(f"  [{i+1}/{len(chain)-1}] {method}: SKIPPED, kernel file missing")
                continue

            pp, cc, collided = _prep_pair(pf, cf, tmp, i)
            note = "  (renamed: shared extension name)" if collided else ""
            print(f"  [{i+1}/{len(chain)-1}] round {node.get('round')} {method}{note}", flush=True)
            v = _measure(reference, pp, cc, device=a.device, warmup=a.warmup,
                         repeat=a.repeat, tol=a.tol, margin=a.margin,
                         min_reps=a.reps, max_reps=a.max_reps)
            if v is None:
                continue

            sp, psp = node.get("speedup"), kernels[pn].get("speedup")
            claimed = ((sp / psp - 1.0) * 100.0) if (sp and psp) else None
            results.append((cn, method, node.get("round"), claimed, v))
            if a.write:
                node["paired_verdict"] = {k: v[k] for k in
                                          ("rel_pct", "se_pct", "t", "reps", "resolved")}

    if not results:
        print("\nnothing measured")
        return 1

    print(f"\n{'round':>5} {'method':<30}{'claimed':>9}{'measured':>10}{'se':>7}"
          f"{'sigma':>7}{'reps':>5}  verdict")
    for _cn, method, rnd, claimed, v in results:
        sig = v["rel_pct"] / v["se_pct"] if v["se_pct"] else float("inf")
        # sigma is measured-gain-over-zero: "is this an improvement at all",
        # which is the question --base_sigma exists to answer.
        verdict = ("REAL" if sig >= 3 else "likely" if sig >= 2 else "NOT RESOLVED")
        print(f"{str(rnd):>5} {method:<30}"
              f"{(f'{claimed:+.2f}%' if claimed is not None else '-'):>9}"
              f"{v['rel_pct']:>+9.2f}%{v['se_pct']:>6.2f}%{sig:>7.1f}{v['reps']:>5}  {verdict}")

    drifted = [r for r in results if r[3] is not None
               and abs(r[3] - r[4]["rel_pct"]) > 2 * r[4]["se_pct"]]
    print(f"\n{len(drifted)}/{len(results)} recorded gains differ from the paired "
          f"measurement by more than 2 se -- those carried drift, not signal.")
    real = [r[4]["rel_pct"] / r[4]["se_pct"] for r in results
            if r[4]["se_pct"] and r[4]["rel_pct"] / r[4]["se_pct"] >= 3]
    if real:
        print(f"genuine advances cluster at {min(real):.1f}-{max(real):.1f} sigma; "
              f"set --base_sigma below {min(real):.1f} to keep admitting them.")

    if a.write:
        tree_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False))
        print(f"\nwrote {len(results)} verdicts -> {tree_path}")
    else:
        print("\n(dry run; pass --write to persist these onto the tree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
