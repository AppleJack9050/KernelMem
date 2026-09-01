"""A quick path for the MCTS rollout: no ncu, no judge, everything else intact.

What this is for
----------------
The normal pathway (``main_memory_latest.py --search mcts``) costs 9.5 min per
round, measured as the median over 122 clean consecutive rounds:

    ncu profile of the base kernel .... 1.1 min  (12 subprocess launches:
                                                  6 kernels x metrics+section)
    judge_optimization LLM ........... 2.8 min  (opus at effort=high; ~18k tok
                                                  median, up to 85k = 13 min, of
                                                  which 94-98% is thinking that
                                                  is discarded after the JSON is
                                                  extracted)
    optimization LLM (THE ROLLOUT) ... 2.6 min  (claude-sonnet-5, effort=high)
    build + bench + paired verdict .... 2.7 min  (cold nvcc alone is 30.5 s)

Only the rollout grows the tree. The other ~6.6 min is overhead the search does
not consume: MCTS needs a parent, a candidate, and a paired gain between them.
Dropping ncu and the judge leaves ~5.3 min per rollout -- ~11 rollouts/hour
against ~6 -- and the search gets 1.8x more edges per wall-clock hour.

What is deliberately KEPT, and why
----------------------------------
`_paired_base_verdict`. Between-process CV on this machine is 0.3-1.4% on score,
and one unchanged kernel drifted +1.06% in 30 minutes -- twice the accept margin.
An unverified single-shot score therefore enters the tree as a fake gain, and
the tree is a ledger of gains: a fake +1.4% at depth 2 is multiplied into every
descendant's chained value forever. The verdict is the one part of the 2.7-min
bench block that cannot be cut without making the numbers this path produces
incomparable with the numbers a normal run produces -- which is the entire point
of the merge below.

What replaces the judge
-----------------------
The judge's only product that reaches the rollout is ``strategy_json``, and its
only load-bearing field is ``method_name``: it becomes ``node.via``, and through
that ``path_mechanisms`` and the PUCT prior's repeat damping.
Here the SEARCH picks the mechanism before the rollout -- a prior-weighted draw
over mechanisms not already spent on this path -- and the same string goes into
both the prompt and ``observe(mechanism=...)``. That makes the mechanism a
decision of the search rather than a report from a profile, which is where
`MechanismPrior.hint`'s own docstring says the prior has more leverage than it
can reach from selection alone.

The rest of the strategy dict is built from measured tree evidence
(``prior.hint`` advantages with n=, ``siblings_context`` outcomes,
``path_mechanisms``). ``bottleneck`` says plainly that no profile was taken. No
metric number is ever fabricated: the optimization template tells the model those
fields carry evidence, so inventing one is worse than admitting there is none.

State, and concurrency
----------------------
This owns its own run directory and its own ``tree.json``. It NEVER writes a
normal run's ``checkpoint.json`` during the rollout loop -- folding results back
is a separate, explicit ``merge`` invocation with its own preconditions. That is
what makes it safe to run beside a live normal run, provided both processes have
``KERNELMEM_GPU_LOCK`` pointing at the same lock file (see the startup gate; this
file refuses to start without it, but it cannot retrofit it onto a normal run
that was launched without it).

Run:
    python -m utils.mcts_quick run --from_run run/<batch>/<task> --rounds 20
    python -m utils.mcts_quick run --resume run/quick_20260814_101500_vae_block_002
    python -m utils.mcts_quick merge --quick_run <quick dir> --host <task_root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# `import main_memory_latest` resolves only when the repo root is on sys.path, and
# sys.path[0] for `python -m utils.X` is the CWD -- so a quick run launched from
# anywhere but the repo root dies at the first lazy import. Inserted rather than
# asserted because the alternative is a ModuleNotFoundError three minutes into a
# rollout that has already spent its LLM call.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# tasks/*.py and every ref_*.py default SOLBENCH_SRC to
# /home/elek/KernelMem/third_party/SOL-ExecBench/src, which does not exist on this
# machine. The failure surfaces INSIDE the spawned bench worker as a
# ModuleNotFoundError landing in ind.metrics["message"], so the whole rollout is
# charged to the kernel and the log blames code that was never given a chance to
# run. Set at MODULE scope, following utils/shape_coverage_test.py:61, because
# multiprocessing's spawn re-executes this file in every bench and verdict child
# before that child imports the reference -- setting it inside main() would be too
# late for the child.
os.environ.setdefault("SOLBENCH_SRC",
                      str(_REPO / "third_party" / "SOL-ExecBench" / "src"))

# utils.mcts pulls in only json/math/hashlib/re/dataclasses. Everything heavier --
# main_memory_latest, and through it torch and matplotlib -- is imported lazily
# inside cmd_run. Module scope here is re-executed as __mp_main__ in EVERY spawned
# bench child and EVERY spawned paired-verdict child, i.e. twice per rollout, so
# anything expensive at module scope is paid twice per rollout for nothing.
from utils.mcts import (MechanismPrior, MonteCarloTreeSearch,  # noqa: E402
                        Selection, _norm_code, reward_from_gain)

_QUICK_VERSION = 1
_TREE_NAME = "tree.json"
# The graph era's filename for the same file. Read, never written: a quick run
# made before the refactor still resumes and still merges, and the first
# _write_tree of a resumed run moves it to _TREE_NAME.
_TREE_NAME_LEGACY = "graph.json"
_JOURNAL_NAME = "journal.jsonl"
_LEDGER_NAME = "mcts_quick_merged.json"    # lives beside the HOST checkpoint.json
_CHECKPOINT_NAME = "checkpoint.json"

# Set by the SIGINT/SIGTERM handler, read only at the top of the rollout loop.
# A module-local dict rather than main_memory_latest._STOP_REQUESTED: that is a
# private cross-module global, and reaching into it would make this file's stop
# behaviour depend on whether the host module happened to be imported yet.
_STOP: Dict[str, bool] = {"requested": False}

# Mechanisms the model has actually produced on this task, as a floor under
# --mcts_prior. Measured 2026-08-14 with
#   scripts.build_mechanism_prior.collect(Path("run"), "vae_block_002", 30)
# -> 108 parent->child edges naming a method, 47 distinct method_name strings, of
# which exactly these 17 reached n>=3; the other 30 include 27 that appear exactly
# once, i.e. names the model invented and never reused.
#
# This list exists because a None mechanism is NOT a neutral default. With
# mechanisms=[] the key is "x:" + sha1("m-empty:" + kernel_name) (utils/mcts.py
# :172-176), so every child gets a unique key: no transposition ever merges, every
# node stays at N=1, path_mechanisms returns [] forever, PUCT degenerates to plain
# UCT, and the tree quietly becomes a list. The measured cost of that bucket when
# it was shared instead was 37 kernels spanning a 457% speedup range.
_FALLBACK_MECHANISMS: Tuple[str, ...] = (
    "CUDA_Graph_Capture_Replay_StaticBuffers", "l2_cache_blocking",
    "stream_pipeline_overlap", "mem_vectorize", "cudnn_plan_select",
    "multistream_pipeline", "cuda_graph_capture", "cudnn_conv_engine_autotune",
    "stream_stage_interlock_pipelining", "reduction_kernel_fusion_lastblock",
    "l2_evict_policy_hints", "compute_mem_stream_stagger", "reduction_kernel_fusion",
    "stream_specialize_pipeline", "sm_budget_grid_cap", "smallB_multistream_gate",
    "group_tile_l2_blocking",
)

# The nine keys prompts/optimization_memory_latest.py:_format_problem actually
# reads (:169-201). Anything else in the dict is silently dropped before the
# prompt is rendered, so emitting it is a note to the reader of this file, not a
# message to the model -- and a note that looks like a message is a defect.
_STRATEGY_KEYS: Tuple[str, ...] = (
    "bottleneck", "optimisation method", "primary_optimisation_method",
    "method_name", "modification plan", "modification_plan", "evidence",
    "expected_metric_change", "headroom",
)


# ===========================================================================
# lazy host-helper access
# ===========================================================================
def _load_host_helpers():
    """Import main_memory_latest and hand back the module.

    Deferred to call time, never module scope. Two reasons, both measured:
    (a) spawn re-executes THIS file in every bench and verdict child, and
    main_memory_latest drags in torch, matplotlib and the whole prompt package --
    twice per rollout, for a child that only needs `_bench_worker_entry`; and
    (b) utils/test_mcts_quick.py has to run with no GPU and no torch, and it
    asserts `"main_memory_latest" not in sys.modules` right after importing this
    module. That assertion is the tripwire for anyone who later hoists this to
    the top of the file.

    main_memory_latest is import-safe: its only module-level side effect is
    matplotlib.use("Agg") at :20, and it guards its entry point with
    `if __name__ == "__main__"` at :3594.
    """
    import main_memory_latest  # noqa: WPS433 (deliberate late import)
    return main_memory_latest


# ===========================================================================
# local copies of closures that main_memory_latest nests inside _run_single_task
# ===========================================================================
# _register / _resolve below are copies of main_memory_latest.py:1613-1630. They
# are closures over `_kernel_registry` inside `_run_single_task`, so
# hasattr(main_memory_latest, "_resolve") is False and they cannot be imported.
# This ~20 lines of duplication is unavoidable without editing the host file,
# which this work is explicitly forbidden from doing. If you are here to "fix the
# duplication": hoisting them in main_memory_latest.py is a behaviour-visible edit
# to a 3400-line file that is the only thing producing this project's numbers.
# Report it, do not do it.
def _register(registry: Dict[str, Any], ind: Optional[Any]) -> None:
    if ind is not None and getattr(ind, "code_path", None):
        registry[Path(ind.code_path).stem] = ind


def _resolve(registry: Dict[str, Any], name: Optional[str], path: Optional[str],
             *, kernel_cls) -> Optional[Any]:
    """A tree node's kernel as a live KernelIndividual.

    The tree stores kernel NAMES (it has to be JSON-serialisable), and the
    prompt needs `.code`/`.code_path`, so names resolve through the registry and
    fall back to reading the file when a resume has no live object.
    """
    if not name:
        return None
    got = registry.get(name)
    if got is not None:
        return got
    if path and Path(path).exists():
        ind = kernel_cls(Path(path).read_text(encoding="utf-8"))
        ind.code_path = Path(path)
        ind.metrics = {"runnable": True}
        registry[name] = ind
        return ind
    return None


# ===========================================================================
# mechanism selection -- the judge's job, done by the search
# ===========================================================================
def _mechanism_candidates(tree: MonteCarloTreeSearch) -> List[str]:
    """Every mechanism name the search may propose, most-informed first.

    The union of three sources, because none alone is sufficient: the prior's
    table is empty until --mcts_prior points at a fitted file; the tree's own
    `via` labels do not exist until a few rollouts have landed; and the measured
    fallback list is the floor that guarantees the returned name is never empty.

    Never returns an empty list. That is the whole contract -- see the comment on
    _FALLBACK_MECHANISMS for what an empty mechanism does to the key space.
    """
    out: List[str] = []
    seen = set()

    def _add(name: Optional[str]) -> None:
        if not name:
            return
        m = str(name).strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)

    if tree.prior is not None and tree.prior.table:
        # Ranked, so the highest-advantage names come first; only matters for
        # `argmax` ties and for how the list reads in the log.
        for m, _adv, _n in tree.prior.ranked(10_000):
            _add(m)
    for node in tree.nodes.values():
        _add(node.via)
    for m in _FALLBACK_MECHANISMS:
        _add(m)
    return out


def _choose_mechanism(tree: MonteCarloTreeSearch, sel: Selection,
                      rng: random.Random, policy: str,
                      candidates: Optional[Sequence[str]] = None) -> str:
    """Pick the mechanism this rollout will be asked for, and keyed by.

    Two dampings, not one. `MechanismPrior.weights` damps mechanisms already on
    the PATH (root->parent), which stops the prior recommending the same edit all
    the way down a branch. It says nothing about mechanisms already tried FROM
    this node and rejected -- and that is the failure `siblings_context` exists
    to warn the model about, which means it is an observed failure mode and the
    prompt-side warning alone did not stop it. So the node's own `tried` list
    damps by the same repeat_penalty (default 0.25) before the draw.

    `sample` rather than `argmax` by default: argmax hands every child of a node
    the same mechanism, so progressive widening spends its k-th child slot
    re-asking for the edit that was just refused.
    """
    cands = list(candidates) if candidates else _mechanism_candidates(tree)
    if not cands:
        # Unreachable while _FALLBACK_MECHANISMS is non-empty, and asserted rather
        # than tolerated because the failure it guards against is silent.
        raise RuntimeError("no mechanism candidates; the key space would collapse "
                           "into the x:m-empty bucket")

    applied = tree.path_mechanisms(sel.path)
    if tree.prior is not None and tree.prior.table:
        w = list(tree.prior.weights(cands, already_applied=applied))
    else:
        w = [1.0 / len(cands)] * len(cands)

    penalty = float(tree.prior.repeat_penalty) if tree.prior is not None else 0.25
    tried = {str(t.get("mechanism")).strip()
             for t in sel.node.tried if t.get("mechanism")}
    if tried and penalty >= 0.0:
        w = [x * penalty if c in tried else x for c, x in zip(cands, w)]
    total = sum(w)
    w = [x / total for x in w] if total > 0 else [1.0 / len(cands)] * len(cands)

    if policy == "argmax":
        return cands[max(range(len(cands)), key=lambda i: w[i])]
    return rng.choices(cands, weights=w, k=1)[0]


def _build_strategy(tree: MonteCarloTreeSearch, sel: Selection,
                    mech: str) -> Dict[str, str]:
    """The judge substitute: a strategy dict built from measured tree evidence.

    Only keys `_format_problem` reads are emitted. Three the judge emits are
    dropped on purpose:
      * expected_metric_change -- it exists to be checked against a NEXT ncu
        profile, which this path never takes. A number nobody will check is a
        claim, not evidence.
      * structural_rewrite -- _format_problem drops it anyway, and the only
        consumer (main_memory_latest.py:2770-2777, the --structural_grace
        ratchet) is inert under MCTS because the base gate requires not _mcts_base_frozen.
      * the spaced legacy aliases "optimisation method" / "modification plan" --
        pure back-compat, and emitting both spellings duplicates the field in the
        rendered JSON the model reads.
    """
    applied = tree.path_mechanisms(sel.path)
    siblings = tree.siblings_context(sel.node, limit=8)
    hint = (tree.prior.hint(already_applied=applied)
            if tree.prior is not None else "(no mechanism prior loaded)")

    plan = [f"1. Implement {mech} in this kernel. It is the primary optimisation "
            f"method for this round and every other change must serve it."]
    plan.append("2. Do NOT re-propose anything in this list -- it has already been "
                "tried from exactly this kernel, with these outcomes:\n"
                + _indent(siblings))
    if applied:
        plan.append("3. These mechanisms are already carried by this kernel and must "
                    "be PRESERVED, not undone or re-derived: " + ", ".join(applied) + ".")
    else:
        plan.append("3. This kernel is the search root; nothing has been applied to it "
                    "yet, so there is nothing to preserve beyond its own structure.")
    plan.append("4. Keep the numeric class of the reference. Do not introduce fp16, "
                "bf16, fp8 or int8 anywhere on the compute path: the harness compares "
                "against a reference at the original precision, and a precision change "
                "is a different problem, not a faster solution to this one.")
    plan.append("5. Keep ModelNew a drop-in replacement -- same public API, same "
                "parameter structure, same device handling as the base kernel file.")

    # The template treats modification_plan as a hard checklist (§(0.1) at
    # prompts/optimization_memory_latest.py:63-70, "You MUST NOT silently drop
    # plan items") and Section A of the reply is a plan-item -> code-location
    # map. So an empty or vacuous plan degrades the reply STRUCTURALLY, not just
    # informationally: there is nothing for Section A to enumerate.
    evidence = "\n".join([
        hint,
        "",
        "Already tried from this exact kernel (measured, paired where marked):",
        _indent(siblings),
        "",
        (f"Parent node {sel.node.key} at N={sel.node.N}, "
         f"Q={sel.node.q(tree.lam):.4f}, chained value {sel.node.value:.4f}. "
         f"The search selected it because: {sel.reason}"),
    ])

    return {
        # method_name is load-bearing far beyond the prompt: it becomes
        # observe(mechanism=) -> node.via -> path_mechanisms -> the PUCT prior and
        # the PUCT prior. The prompt and the tree MUST see the same string, which
        # is why it is chosen before the rollout instead of parsed back out of the
        # reply (a parse failure here would have no judge to blame and would
        # silently key the child into the x:m-empty bucket).
        "method_name": mech,
        "primary_optimisation_method": (
            f"Apply {mech} to this kernel. This mechanism was selected by the search "
            f"from the measured mechanism prior and from what has already been tried "
            f"from this node -- not from a profile of this kernel."),
        "modification_plan": "\n".join(plan),
        "evidence": evidence,
        # No ncu profile was taken, so there is no measured bottleneck. Saying so
        # is the only honest option: the template's preamble
        # (prompts/optimization_memory_latest.py:16-20) tells the model these
        # fields carry evidence, so a plausible-sounding invented metric is read
        # as a measurement.
        "bottleneck": (
            "No ncu profile was taken for this rollout (quick path: ncu costs 1.1 min "
            "of the 9.5-min round and the search does not consume it). The bottleneck "
            "is NOT measured. Infer it from the kernel source and from the sibling "
            "outcomes in `evidence` below, and do not assume a profile said anything."),
        # Consumed at prompts/optimization_memory_latest.py:101-104 to set how
        # aggressive the edit should be. With no profile there is no headroom
        # estimate, and "medium" is the only value that is not a claim.
        "headroom": "medium",
    }


def _indent(text: str, prefix: str = "   ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


# ===========================================================================
# tree plumbing shared by the live loop, resume replay and merge
# ===========================================================================
def _host_path(tree: MonteCarloTreeSearch, target: str) -> Optional[List[str]]:
    """Root-to-*target* route, or None if *target* is not in the tree.

    Was a BFS over `children`, because on a DAG a node could be reached by
    several routes and the shortest had to be found; `depth` there was "shortest
    route known when the node was last observed" and a recorded `sel_path` could
    be strictly longer than the live one. On a tree there is exactly one route
    and `MonteCarloTreeSearch.path_to` walks it up the parents in O(depth).

    `_reaches` went with it: it existed only to refuse a merge edge that would
    close a cycle, and `observe` cannot create an edge into an existing node.
    """
    return tree.path_to(target)


def _apply_record(tree: MonteCarloTreeSearch, rec: Dict[str, Any],
                  applied_obs: set, rep_sha: Dict[str, str],
                  id_map: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
    """Fold one journal record into *tree*. Returns (landed_key or None, reason).

    *id_map* translates the record's LOCAL node ids into *tree*'s ids, and is the
    difference between folding a journal into the tree that produced it and
    folding it into a different tree.

    Pass None when the record's ids ARE this tree's ids -- the live loop and the
    --resume replay, which both apply into the very tree the ids were minted in.

    Pass a map for merge, where they are not. Node ids are minted per tree
    (`MonteCarloTreeSearch._mint`), so `n0002` in a quick run and `n0002` in the
    host it forked from are the same node ONLY for nodes that existed at the
    fork; after that both trees mint from the same counter independently and the
    id means different kernels in each. The graph era had no such problem because
    a key was a content hash of the path's mechanism multiset and was therefore
    portable, which is why nothing translated. Looking a local id up directly in
    the host silently grafts a whole quick-run subtree onto an unrelated kernel:
    the id exists, so `parent-not-in-target` cannot fire, the chained value is
    computed off the wrong parent, and the reward is backed up the wrong lineage.
    An id the map does not carry is REFUSED rather than guessed at.

    ONE function for three callers -- the live loop, --resume replay, and merge --
    on purpose. Three copies of this would be three chances for the replay to
    reconstruct a tree the live loop did not build, and the crash-recovery test
    would then be testing the copy rather than the thing that runs.

    Idempotent by content address: `obs_id` hashes
    (parent_key, mechanism, code_sha, rel_pct), so replaying a journal twice is a
    no-op. That is not a nicety. `backup` has no observation identity
    (utils/mcts.py:550-559), so a double-apply doubles N and W together, which
    leaves q() unchanged and therefore LOOKS harmless -- while
    ceil(widen_k * N**alpha) takes a node from N=4 (budget 2) to N=8 (budget 3)
    and hands it a child it never paid for.
    """
    obs_id = rec.get("obs_id")
    if not obs_id:
        return (None, "no-obs-id")
    if obs_id in applied_obs:
        return (None, "duplicate")

    kernel_name = rec.get("kernel_name")
    if not kernel_name:
        # An LLM error or an extraction failure: there is no kernel, so there is
        # nothing to observe. Marked applied so a replay does not re-examine it
        # every time; the record stays in the journal so the 2.6 min it cost is
        # visible instead of being a silent gap in the timeline.
        applied_obs.add(obs_id)
        return (None, "no-kernel-produced")

    parent_local = rec.get("parent_key")
    if id_map is None:
        parent_key = parent_local
    else:
        # An unmapped local id is a parent this journal produced whose own record
        # did not land here (refused, or from a fork this map does not describe).
        # Its children have no parent in the target either.
        parent_key = id_map.get(parent_local) if parent_local else None
        if parent_local and parent_key is None:
            return (None, "parent-not-in-target")
    parent = tree.nodes.get(parent_key) if parent_key else None
    if parent is None:
        # NEVER call observe() with a parent_key absent from nodes. Verified: it
        # creates a depth-0 node with no parent, does not promote it to root, and
        # leaves it permanently invisible to select() (which descends `children`
        # from the root only) while best() and stats() still count it. The run
        # then announces a best node it can never branch from.
        return (None, "parent-not-in-target")

    # Belt and braces on top of the map: the journal records the parent's KERNEL
    # NAME as well as its id, so the graft can be checked against something that
    # is not an id at all. Written since the quick path existed and never read
    # until now. A mismatch means the map is wrong, and a wrong map is exactly
    # the silent corruption this function must not commit -- refuse, do not fold.
    _parent_rep = rec.get("parent_rep")
    if _parent_rep and parent.kernel and str(_parent_rep) != str(parent.kernel):
        return (None, "parent-identity-mismatch")

    runnable = bool(rec.get("runnable"))
    if not runnable and not bool(rec.get("count_failures")):
        # Faithful to main_memory_latest.py: its observe/backup block sits under
        # `if this_score is not None:` (:2966) and this_score is None whenever
        # runnable is False (:2951), so a rollout that will not compile leaves the
        # parent's N, its `tried` list and its widening budget untouched. Diverging
        # by default would make a merged N incomparable with the host's, which is
        # the one thing the merge exists to preserve.
        applied_obs.add(obs_id)
        return (None, "failed-rollout-not-counted")

    host_path = _host_path(tree, parent_key)
    if host_path is None:
        return (None, "parent-unreachable-from-root")

    # Three refusals stood here and are all gone with the state key. The child
    # key was recomputed in HOST space from the path's mechanism multiset, which
    # could (a) fall through to an "x:" file-stem hash and collide with an
    # unrelated kernel that happened to be saved in the same wall-clock second,
    # (b) land on a node the merge tolerance then rehomed under a split key, or
    # (c) close a cycle when the host already held the reverse edge. A node id is
    # minted per observation now: it cannot collide, cannot be rehomed, and
    # cannot point at an existing node, so there is nothing to refuse.
    mech = rec.get("mechanism") or None

    rel = rec.get("rel_pct")
    rel = 0.0 if rel is None else float(rel)
    child_value = float(parent.value) * (1.0 + rel / 100.0)

    node = tree.observe(
        kernel_name=kernel_name,
        kernel_path=rec.get("kernel_path"), value=child_value,
        parent_key=parent_key, runnable=runnable, mechanism=mech,
        note=f"{rec.get('origin', 'quick')} r{rec.get('rollout_idx')}, "
             f"{rel:+.2f}% {rec.get('basis', '?')}")

    # observe()'s RETURN value, never a key computed ahead of the call. The id is
    # minted inside observe() and is not predictable from here.
    landed = node.key

    reward = reward_from_gain(rel, scale=tree.reward_scale, failed=not runnable)
    # Always fresh: observe() mints an id that is not in `nodes`, and every key on
    # host_path is. The graph needed a membership test here because a merge could
    # land the child ON the path, double-counting one evaluation.
    path = list(host_path) + [landed]
    # Exactly one backup per record, so total_visits advances by one per rollout
    # that actually landed -- never by the quick tree's total_visits, which counts
    # rollouts a merge may have refused.
    tree.backup(path, reward, failed=not runnable)

    if rec.get("code_sha"):
        rep_sha[landed] = str(rec["code_sha"])
    applied_obs.add(obs_id)
    return (landed, "applied")


# ===========================================================================
# persistence
# ===========================================================================
def find_tree_path(quick_dir: Path) -> Optional[Path]:
    """The quick run's tree file, under either its current or its graph-era name.

    utils.mcts_view and utils.pathmemory both already fall back to graph.json, so
    without this the refactor is internally inconsistent about whether a
    pre-refactor quick run is readable: the viewer would render it while --resume
    reported "no tree.json" and merge died on an uncaught FileNotFoundError.
    """
    for name in (_TREE_NAME, _TREE_NAME_LEGACY):
        f = Path(quick_dir) / name
        if f.exists():
            return f
    return None


def _write_tree(path: Path, tree: MonteCarloTreeSearch,
                 meta: Dict[str, Any]) -> None:
    """Atomically replace tree.json with the current state.

    tmp + Path.replace, the same idiom as _save_checkpoint
    (main_memory_latest.py:1307-1310): a kill during the write leaves the previous
    tree.json intact rather than a truncated one.

    The tree blob is nested under "mcts" and is byte-identical in SHAPE to
    checkpoint["mcts"], so merge can feed it straight to
    MonteCarloTreeSearch.from_dict with nothing translating. Everything
    quick-path-specific is a SIBLING of that key, never inside it, because
    to_dict() rebuilds its dict from scratch and destroys any extra key stuffed
    into it.
    """
    blob = dict(meta)
    blob["updated"] = datetime.now().isoformat(timespec="seconds")
    # Stamp the current kind rather than copying a graph-era one forward: the
    # payload written below IS the current schema, so a blob that still called
    # itself "mcgs_quick_graph" would be describing a file that no longer exists.
    blob["kind"] = "mcts_quick_tree"
    blob["mcts"] = tree.to_dict()
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_tree(path: Path) -> Tuple[MonteCarloTreeSearch, Dict[str, Any], set]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if blob.get("kind") not in ("mcts_quick_tree", "mcgs_quick_graph"):
        raise ValueError(f"{path} is not a quick-path tree "
                         f"(kind={blob.get('kind')!r})")
    if int(blob.get("quick_version") or 0) != _QUICK_VERSION:
        raise ValueError(f"{path} was written by quick_version "
                         f"{blob.get('quick_version')}, this build is {_QUICK_VERSION}")
    # "mcgs" is the graph-era nesting; from_dict flattens that schema, so a
    # quick run started before the tree existed still resumes.
    tree = MonteCarloTreeSearch.from_dict(blob.get("mcts") or blob.get("mcgs"))
    meta = {k: v for k, v in blob.items() if k not in ("mcts", "mcgs")}
    applied = set(meta.get("applied_obs") or [])
    return tree, meta, applied


def _append_journal(path: Path, rec: Dict[str, Any]) -> None:
    """Append one record and fsync it BEFORE the tree is written.

    Order matters and is the reverse of the intuitive one. A crash between the
    journal write and the tree write leaves a record the tree lacks, which
    --resume replays idempotently (_apply_record is content-addressed). A crash
    the other way round loses the observation with no trace that it happened, and
    the 5.3 minutes it cost are unrecoverable.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_journal(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn final line is the expected shape of a kill mid-append. Every
            # earlier line is intact because each is written and fsync'd whole.
            print(f"[quick] WARNING: dropping an unparseable journal line in {path} "
                  f"(a kill during the final append looks exactly like this).",
                  flush=True)
    return out


# ===========================================================================
# the rollout
# ===========================================================================
def _one_rollout(tree: MonteCarloTreeSearch, *, rollout_idx: int,
                 rng: random.Random, policy: str, gpu_name: Optional[str],
                 registry: Dict[str, Any], dirs: Dict[str, Path],
                 count_failures: bool, stamp: Dict[str, Any],
                 resolve_fn: Callable[[Optional[str], Optional[str]], Optional[Any]],
                 prompt_fn: Callable[[Path, Optional[str], Dict[str, str]], str],
                 rollout_fn: Callable[[str, int], Any],
                 bench_fn: Callable[[Any], None],
                 verdict_fn: Callable[[Path, Path], Optional[Dict[str, Any]]],
                 sanitize_fn: Callable[[BaseException], str] = None,
                 now: Callable[[], datetime] = None) -> Optional[Dict[str, Any]]:
    """One select -> prompt -> generate -> bench -> paired-verify cycle.

    Mutates NOTHING persistent: it writes artifacts (prompt, reply, strategy,
    metrics) and returns a journal record. The tree is read here and written only
    by _apply_record, one level up, after the record has been fsync'd. That split
    is what makes the crash-recovery replay meaningful -- if this function could
    also mutate the tree there would be a window in which the two disagreed with
    nothing on disk to reconcile them from.

    Every side-effecting dependency is injected so utils/test_mcts_quick.py can
    drive the whole cycle with no GPU, no LLM and no subprocess.
    """
    now = now or datetime.now
    ts = now().isoformat(timespec="seconds")

    sel = tree.select()
    if sel is None:
        # select() returns None iff the root is missing or not in nodes
        # (utils/mcts.py:578-579). There is no other None path, so this is not a
        # skippable round -- it means the tree has no anchor and every later
        # rollout would fail identically.
        raise RuntimeError("tree.select() returned None: the root is missing. "
                           "Every rollout from here would fail the same way.")

    parent = resolve_fn(sel.node.kernel, sel.node.kernel_path)
    if parent is None:
        # The representative's file is gone (hand-cleaned run dir, or a resume
        # against moved artifacts). main_memory_latest.py:2164-2172 prints and
        # falls back to the incumbent, which produces no observe and no backup;
        # here there is no incumbent to fall back to, so the round produces
        # nothing at all -- which is the same outcome, stated honestly.
        print(f"[quick] WARNING: could not resolve the code for node {sel.node.key} "
              f"(rep {sel.node.kernel} at {sel.node.kernel_path}); this rollout produces "
              f"nothing. Check that the quick run's code/ directory is intact.",
              flush=True)
        return None

    parent_key = sel.node.key
    parent_value = float(sel.node.value)
    g = tree.stats()
    print(f"[quick] Rollout {rollout_idx}: branching from node {parent_key} "
          f"(depth {sel.node.depth}, N={sel.node.N}, "
          f"Q={sel.node.q(tree.lam):.3f}, value {parent_value:.4f}) "
          f"via {sel.node.kernel}", flush=True)
    print(f"[quick]   why: {sel.reason}", flush=True)
    print(f"[quick]   tree: {g['nodes']} nodes ({g['leaves']} leaves), "
          f"mean N={g['mean_N']:.2f}, depth seen "
          f"{g['max_depth_seen']}", flush=True)

    mech = _choose_mechanism(tree, sel, rng, policy)
    strategy = _build_strategy(tree, sel, mech)
    io_dir = dirs["io"]
    io_dir.mkdir(parents=True, exist_ok=True)
    (io_dir / f"{rollout_idx:03d}_strategy.json").write_text(
        json.dumps(strategy, indent=2, ensure_ascii=False), encoding="utf-8")
    if tree.prior is not None and tree.prior.table:
        prior_txt = (f"prior advantage {tree.prior.advantage(mech):+.2f}%, "
                     f"n={tree.prior.support(mech)}")
    else:
        prior_txt = "no prior loaded, so the draw is uniform over the candidates"
    print(f"[quick]   mechanism: {mech} (policy={policy}; {prior_txt})", flush=True)

    rec: Dict[str, Any] = {
        "ts": ts, "rollout_idx": rollout_idx, "origin": "quick",
        "parent_key": parent_key, "parent_rep": sel.node.kernel,
        "parent_rep_value": parent_value,
        "sel_path": list(sel.path), "sel_reason": sel.reason,
        "mechanism": mech, "strategy": strategy,
        "count_failures": bool(count_failures),
        "kernel_name": None, "kernel_path": None, "code_sha": None,
        "runnable": False, "score": None, "rel_pct": None, "basis": "unmeasurable",
        "beats_margin": None, "sigma_ok": None, "verdict": None,
        "error_type": None, "message": None,
    }
    rec.update(stamp)

    # ---- generate -----------------------------------------------------------
    arch_path = Path(parent.code_path)
    opt_prompt = prompt_fn(arch_path, gpu_name, strategy)
    (io_dir / f"{rollout_idx:03d}_opt_prompt.txt").write_text(opt_prompt,
                                                             encoding="utf-8")
    try:
        ind = rollout_fn(opt_prompt, rollout_idx)
    except Exception as exc:
        # _llm_to_kernel propagates whatever query_server raises. Journalled as a
        # record rather than swallowed, so the 2.6 min the call burned is visible
        # in the timeline instead of being an unexplained gap between rollouts.
        import traceback
        (io_dir / f"{rollout_idx:03d}_rollout_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8")
        rec["error_type"] = exc.__class__.__name__
        rec["message"] = (sanitize_fn(exc) if sanitize_fn else str(exc))[:2000]
        rec["obs_id"] = hashlib.sha1(
            f"{stamp.get('quick_run_id')}|{rollout_idx}|llm-error".encode()).hexdigest()
        print(f"[quick] Rollout {rollout_idx} FAILED in the model call: "
              f"{rec['error_type']}: {rec['message'][:200]}", flush=True)
        return rec

    rec["kernel_name"] = Path(ind.code_path).stem if getattr(ind, "code_path", None) else None
    rec["kernel_path"] = str(ind.code_path) if getattr(ind, "code_path", None) else None
    rec["code_sha"] = hashlib.sha1(_norm_code(ind.code or "").encode()).hexdigest()

    # ---- bench --------------------------------------------------------------
    bench_fn(ind)
    runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
    this_score = ind.score if (ind.score is not None and runnable) else None
    rec["runnable"] = runnable
    rec["score"] = float(this_score) if this_score is not None else None
    if not runnable:
        m = getattr(ind, "metrics", {}) or {}
        rec["error_type"] = m.get("error_type")
        rec["message"] = str(m.get("message") or "")[:2000]
        print(f"[quick] Rollout {rollout_idx} produced an unrunnable kernel: "
              f"{rec['error_type']}", flush=True)

    # ---- paired verdict -----------------------------------------------------
    # Guarded exactly as main_memory_latest.py:3009-3011. This is the one fidelity
    # knob the quick path keeps: between-process CV is 0.3-1.4% on score, so
    # without it a candidate's "gain" is partly the drift between two processes.
    verdict = None
    base_path = getattr(parent, "code_path", None)
    cand_path = getattr(ind, "code_path", None)
    if (this_score is not None and base_path and cand_path
            and Path(base_path).exists() and Path(cand_path).exists()):
        verdict = verdict_fn(Path(base_path), Path(cand_path))

    if verdict is not None:
        rel = float(verdict["rel_pct"])
        basis = "paired"
        # Read sigma_ok, NEVER rel_pct/se_pct >= sigma. That ratio is a t
        # statistic on dof = reps-1, and comparing it to 3.0 as though it were a z
        # score was ~6x more permissive than advertised: at 3 reps (dof 2) a true
        # 3-sigma tail needs |t| >= 19.2. adaptive_paired_verdict evaluates the t
        # distribution at the right dof and reports sigma_ok itself.
        sig_ok = bool(verdict.get("sigma_ok", True))
        rec["sigma_ok"] = sig_ok
        rec["beats_margin"] = bool(verdict.get("beats_margin", False))
        rec["verdict"] = {k: verdict[k] for k in
                          ("rel_pct", "se_pct", "t", "dof", "reps", "resolved",
                           "p_one_sided", "sigma_equiv", "sigma_ok", "method")
                          if k in verdict}
        print(f"[quick]   paired {rel:+.2f}% +/-{verdict.get('se_pct', 0.0):.2f}% "
              f"on {verdict.get('dof', '?')} dof, {verdict.get('reps', '?')} reps, "
              f"{'resolved' if verdict.get('resolved') else 'UNRESOLVED'}; "
              f"beats_margin={rec['beats_margin']} sigma_ok={sig_ok}", flush=True)
    elif parent_value > 0 and this_score is not None and this_score > 0:
        # Note this compares against parent_value -- the CHAINED rep_value, not a
        # base measured now. Same as main_memory_latest.py:3111-3113, and it is
        # exactly why the label says contaminated: the number carries the drift
        # the paired path removes, and it is stamped on the edge forever so a
        # later reader can tell the two bases apart.
        rel = (this_score / parent_value - 1.0) * 100.0
        basis = "blocked (drift-contaminated; no paired verdict)"
        print(f"[quick]   NO paired verdict; falling back to the blocked delta "
              f"{rel:+.2f}% against the stored chain value {parent_value:.4f}. "
              f"This number carries cross-process drift (measured CV 0.3-1.4%).",
              flush=True)
    else:
        rel, basis = 0.0, "unmeasurable"

    rec["rel_pct"] = float(rel)
    rec["basis"] = basis
    rec["obs_id"] = hashlib.sha1(
        f"{parent_key}|{mech}|{rec['code_sha']}|{rel:.6f}".encode()).hexdigest()
    return rec


# ===========================================================================
# stop handling
# ===========================================================================
def _install_stop_handler() -> None:
    """SIGINT/SIGTERM become a graceful stop at the next rollout boundary.

    A local copy of main_memory_latest.py:1173-1204, writing into this module's
    _STOP dict. The signal almost always lands inside the 2.6-min model call or
    the 2.7-min bench, so the flag is only CHECKED at the top of the loop: the
    in-flight rollout finishes, its journal record is fsync'd and its tree write
    lands, and the run exits with tree.json current as of the last completed
    rollout. Signalling twice restores the default handler so a wedged run can
    still be killed outright -- at the cost of that rollout.
    """
    from utils import run_timing

    def _handler(signum, _frame):
        if _STOP["requested"]:
            run_timing.event("abort_signal", detail=f"signum={signum} rollout_abandoned")
            print("\n[stop] Second signal - aborting immediately. The in-flight "
                  "rollout is lost; tree.json is current as of the last completed "
                  "one.", flush=True)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        _STOP["requested"] = True
        run_timing.event("stop_signal", detail=f"signum={signum} graceful")
        print(f"\n[stop] Signal {signum} received. Finishing the current rollout, "
              f"then stopping. Signal again to abort now.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not the main thread, or the platform lacks this signal


# ===========================================================================
# CLI
# ===========================================================================
def _build_parser(suppress: bool = False) -> argparse.ArgumentParser:
    """Build the CLI. With *suppress*, unpassed options vanish from the namespace.

    The suppressed parse is how "did the user pass this explicitly" is answered
    without maintaining a second list of defaults. It matters in exactly one
    place and it matters a lot: when forking a live search blob, an --mcts_* flag
    that disagrees with the blob is a hard error, and telling "the user asked for
    0.9" apart from "the default is 0.8" cannot be done by comparing values.

    Note the suppression is applied by rewriting each action's `default` AFTER the
    parser is built, not by passing ``argument_default=SUPPRESS`` to add_parser:
    argparse's parser-level argument_default is only consulted for arguments that
    declare no `default=` of their own, and every argument here declares one. The
    first version of this did pass argument_default and silently reported every
    flag as explicitly passed, which would have made --mcts_* fork checking fire
    on defaults the user never typed.
    """
    p = argparse.ArgumentParser(
        "mcts_quick",
        description="MCTS rollouts without ncu and without the judge LLM.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------ run
    r = sub.add_parser("run", help="Run quick rollouts into a private tree.")
    entry = r.add_mutually_exclusive_group(required=True)
    entry.add_argument(
        "--from_run", type=Path, default=None,
        help="Fork an existing run: pass its TASK ROOT (the directory holding "
             "checkpoint.json) or the batch directory above it. Read once, "
             "read-only, and never written to -- the fork records the "
             "checkpoint's sha1 so `merge` can tell how far the host has moved "
             "since. If that checkpoint carries a search blob the quick tree IS "
             "that tree, so keys, depths, the value-chain anchor and the prior "
             "all match by construction and merge has nothing to translate. If it "
             "does not (measured 2026-08-14: NONE of the 8 checkpoint.json files "
             "under run/ carry one), a fresh root is seeded from its best kernel "
             "and merging back needs --adopt.")
    entry.add_argument(
        "--from_kernel", type=Path, default=None,
        help="Bench this kernel .py and make it the search root. Requires --task. "
             "Use when there is no run to fork -- e.g. starting a quick search "
             "from a hand-written or hand-picked kernel. The seed bench is a full "
             "_bench_and_score, so an unrunnable kernel is refused at startup "
             "rather than after the first rollout: a root the tree cannot branch "
             "from is not a tree.")
    entry.add_argument(
        "--resume", type=Path, default=None,
        help="Continue an existing quick run directory. Replays any journal "
             "record the tree is missing (the crash window between the journal "
             "fsync and the tree write) and carries on at rollouts_done. "
             "Idempotent: replaying records already folded in changes nothing, "
             "because each is addressed by a sha1 of "
             "(parent_key, mechanism, code_sha, rel_pct).")
    r.add_argument(
        "--task", type=Path, default=None,
        help="Task file, e.g. tasks/vae_block_002.py. Required with "
             "--from_kernel. With --from_run it defaults to the checkpoint's "
             "`task` field and only needs passing if that path has moved.")

    r.add_argument(
        "--rounds", "-G", type=int, default=20,
        help="Number of ROLLOUTS -- not rounds. There is no seed phase and no "
             "repair phase here, so one iteration is one select+generate+measure "
             "cycle. At the measured 5.3 min/rollout (9.5 min median round over "
             "122 clean consecutive rounds, minus 1.1 min of ncu and 2.8 min of "
             "judge) 20 rollouts is about 1.8 h. Default 20 rather than the "
             "normal path's 10 because a rollout here costs 56%% of a round "
             "there.")
    r.add_argument(
        "--work_dir", type=Path, default=Path("run"),
        help="Root for the quick run directory, created as "
             "<work_dir>/quick_<timestamp>_<task stem>/. Deliberately NOT inside a "
             "normal run's tree: the host may clean or rewrite its own code/ "
             "directory, and the tree's rep_path entries are read from disk at "
             "selection time.")
    r.add_argument(
        "--allow_unserialized", action="store_true",
        help="Start even when KERNELMEM_GPU_LOCK is unset. Off by default because "
             "gpu_section() is a NO-OP without that variable (utils/gpu_lock.py"
             ":50-58), so 'safe to run beside a normal run' would be a lie: the "
             "two processes' benches would interleave, and interleaving does not "
             "make the paired verdict noisier, it destroys the common-mode "
             "cancellation the verdict is built on. Pass this only when nothing "
             "else is touching the GPU.")

    r.add_argument("--gpu", default=None,
                   help="GPU name for the prompt's hardware block "
                        "(default: auto-detect via torch and normalise through "
                        "resolve_gpu_name). Wrong here means the model reasons "
                        "about the wrong roofline.")
    r.add_argument("--model_name", default="claude-opus-5",
                   help="Default Claude model. Unused by the rollout itself "
                        "(see --rollout_model); kept because _make_llm_caller's "
                        "closure reads it.")
    r.add_argument("--server_type", default="claude",
                   help="Label only; all calls go through the Claude Agent SDK.")
    r.add_argument("--server_address", default="localhost",
                   help="Unused (kept for compatibility with the shared caller).")
    r.add_argument("--server_port", type=int, default=8000,
                   help="Unused (kept for compatibility with the shared caller).")
    r.add_argument("--temperature", type=float, default=1,
                   help="Sampling temperature for the rollout call.")
    r.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p.")
    r.add_argument(
        "--rollout_model", default="claude-sonnet-5",
        help="Model for the rollout -- the only LLM call this path makes. Same "
             "default as the normal path's --rollout_model. Note the call goes in "
             "as call_type='optimization', which puts it in "
             "agents/query_server.py's TOOL mode (_TOOL_CALL_TYPES at :43, "
             "max_turns = KERNELMEM_AGENT_MAX_TURNS, default 30 at :45): that "
             "30-turn loop with nvcc inside it is where the long rollouts go, and "
             "the quick path does not change it.")
    r.add_argument(
        "--rollout_effort", default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort for the rollout call. High on purpose: the measured "
             "failure mode is candidates that do not build, not candidates that "
             "were under-tuned, and a rollout that fails to compile costs the full "
             "5.3 min and (by default) does not even charge its parent a visit.")

    r.add_argument("--device", type=int, default=0,
                   help="CUDA device index for benchmarking.")
    r.add_argument("--warmup", type=int, default=25, help="Warm-up iterations.")
    r.add_argument("--repeat", type=int, default=100,
                   help="Timed iterations per benchmark.")
    r.add_argument("--tol", type=float, default=1e-2, help="Max |err| tolerated.")
    r.add_argument("--base_margin", type=float, default=0.05,
                   help="Relative margin fed to the paired verdict's beats_margin "
                        "gate. Informational here -- MCTS never ratchets a base, "
                        "tree.observe picks each state's representative by value "
                        "-- but it is recorded on every edge.")
    r.add_argument("--base_reps", type=int, default=5,
                   help="Minimum interleaved rep pairs in the paired re-measure.")
    r.add_argument(
        "--base_max_reps", type=int, default=8,
        help="Maximum interleaved rep pairs; 0 DISABLES the paired re-measure. "
             "This is the one fidelity knob the quick path keeps, and turning it "
             "off is the one change that makes the resulting tree unmergeable in "
             "spirit: between-process CV on score is 0.3-1.4%% and one unchanged "
             "kernel drifted +1.06%% in 30 minutes, so an unverified single-shot "
             "score enters the tree as a fake gain and is then multiplied into "
             "every descendant's chained value forever.")
    r.add_argument("--base_sigma", type=float, default=3.0,
                   help="Significance target for the verdict's sigma_ok gate, "
                        "evaluated on the t distribution at the measured dof (not "
                        "as a z score -- at dof 2 a true 3-sigma tail needs "
                        "|t| >= 19.2).")

    # Retired with the graph. Accepted and ignored so a saved command line still
    # starts; there is no state abstraction to key on and nothing merges.
    r.add_argument("--mcgs_state_key", "--mcts_state_key", default=None,
                   help=argparse.SUPPRESS)
    r.add_argument("--mcgs_merge_tol", "--mcts_merge_tol", type=float, default=None,
                   help=argparse.SUPPRESS)
    r.add_argument("--mcts_c_puct", "--mcgs_c_puct", type=float, default=0.8,
                   help="UCT exploration weight, used only when no prior is "
                        "loaded. Low because the budget is tens of evaluations.")
    r.add_argument("--mcts_lam", "--mcgs_lam", type=float, default=0.7,
                   help="Weight on the MAX term in Q = (1-lam)*mean + lam*max. "
                        "High because the measured gain distribution is bimodal: "
                        "42%% of edges regress past -1%%, 33%% win past +1%%, only "
                        "25%% land inside the +-1%% band.")
    r.add_argument("--mcts_widen_k", "--mcgs_widen_k", type=float, default=1.0,
                   help="Progressive widening: a state may have "
                        "ceil(k * N**alpha) children.")
    r.add_argument("--mcts_widen_alpha", "--mcgs_widen_alpha", type=float, default=0.5,
                   help="Widening exponent. At k=1, alpha=0.5 the budget is "
                        "N=1->1, 4->2, 9->3, 16->4.")
    r.add_argument("--mcts_max_depth", "--mcgs_max_depth", type=int, default=10,
                   help="Edits from the seed after which selection expands "
                        "sideways instead of deeper. Measured on vae_block_002: "
                        "win rate 41%% for rounds 0-4 and 5-9, then 0 wins in 22 "
                        "edges past round 10.")
    r.add_argument("--mcts_prior", "--mcgs_prior", default="",
                   help="Path to a mechanism prior fitted by "
                        "scripts/build_mechanism_prior.py (e.g. "
                        "priors/vae_block_002.json). It does more here than in the "
                        "normal path: selection uses it for PUCT AS WELL AS the "
                        "per-rollout mechanism draw that replaces the judge. "
                        "Ignored with a warning when forking a checkpoint that "
                        "already carries a tree -- that blob's prior is "
                        "authoritative, because a resumed search must select with "
                        "the policy it started with.")
    r.add_argument("--mcts_c_prior", "--mcgs_c_prior", type=float, default=1.0,
                   help="PUCT exploration weight, used only with --mcts_prior: "
                        "U = Q + c_prior * P(a) * sqrt(N_parent) / (1 + N_child).")
    r.add_argument("--mcts_reward_scale", "--mcgs_reward_scale", type=float, default=3.0,
                   help="Percent gain mapping to a near-saturated reward via "
                        "tanh(rel/scale). At 3.0: 0%% -> 0.50, +1%% -> 0.66, "
                        "+3%% -> 0.88. (utils/mcts.py:191's docstring says 0.58 "
                        "for +1%%; that is the value for +0.5%%.)")

    r.add_argument(
        "--mechanism_policy", default="sample", choices=["sample", "argmax"],
        help="How the per-rollout mechanism is drawn from the prior-weighted "
             "distribution. `argmax` hands every child of a node the same "
             "mechanism, so progressive widening spends its next child slot "
             "re-asking for the edit that state just tried; `sample` (default) "
             "draws, with mechanisms already on the path or already in this "
             "state's `tried` list damped by the prior's repeat_penalty (0.25).")
    r.add_argument(
        "--mechanism_seed", type=int, default=0,
        help="Seed for the mechanism draw; 0 derives one from the run id. Fixed so "
             "a replayed or resumed run reproduces its mechanism sequence -- "
             "without it, --resume would silently explore a different branch of "
             "the action space than the interrupted session was on.")
    r.add_argument(
        "--count_failures", action="store_true",
        help="Charge the parent a visit at reward 0.0 when a rollout will not "
             "compile. OFF by default to stay faithful to main_memory_latest.py, "
             "where the observe/backup block sits under `if this_score is not "
             "None:` (:2966) and this_score is None whenever runnable is False "
             "(:2951) -- so a rollout that fails to build costs a full evaluation "
             "and leaves the parent's N, `tried` and widening budget untouched, "
             "and reward_from_gain(..., failed=True) is dead code from that call "
             "site. Turning this ON is the behaviour the existing code plainly "
             "intended but never reaches; it is opt-in because it makes merged N "
             "and W incomparable with a host run's, and it is stamped into the "
             "tree meta and into every journal record so a later reader can tell.")

    r.add_argument("--no_clock_lock", action="store_true",
                   help="Measure without pinning the GPU clock. The result is not "
                        "comparable with any locked run and will be REFUSED by "
                        "`merge` (precondition R7).")
    r.add_argument("--gpu_clock_mhz", type=int, default=0,
                   help="Override the target core clock in MHz (0 = the measured "
                        "preset for this card; the 5090 pins at 2407 MHz).")

    # ---------------------------------------------------------------- merge
    m = sub.add_parser(
        "merge", help="Fold a quick run's observations into a host checkpoint.")
    m.add_argument("--quick_run", type=Path, required=True,
                   help="The quick run directory (holds tree.json and "
                        "journal.jsonl).")
    m.add_argument("--host", type=Path, required=True,
                   help="The host TASK ROOT holding checkpoint.json.")
    m.add_argument(
        "--in_place", action="store_true",
        help="Overwrite the host's checkpoint.json. Without it the merge writes "
             "checkpoint.json.merged beside it and prints the mv, which is the "
             "safe default because the host rewrites checkpoint.json at the end of "
             "every round and would clobber a merge landing mid-round.")
    m.add_argument(
        "--adopt", action="store_true",
        help="Permit merging into a host whose checkpoint has no search blob, by "
             "installing the quick tree wholesale. Needed for every run currently "
             "on disk (measured 2026-08-14: 0 of 8 checkpoint.json files carry "
             "one). The host's base_kernel and best_kernel are NOT touched, and the "
             "installed tree's rep_path entries point into the quick run "
             "directory -- which must not be moved or deleted, because _resolve "
             "reads them from disk at selection time.")
    m.add_argument(
        "--host_idle_s", type=int, default=900,
        help="A checkpoint.json touched more recently than this counts as LIVE and "
             "the merge refuses. 900 s because the median normal round is 9.5 min "
             "= 570 s, so this clears one full round with margin. Set 0 to skip "
             "the mtime test (the /proc liveness scan still runs).")
    m.add_argument(
        "--force_reward_scale", action="store_true",
        help="Proceed when the two trees disagree on reward_scale, recomputing "
             "every reward at the HOST's scale. Legitimate only because the merge "
             "recomputes reward from the journal's rel_pct and never copies a "
             "stored reward -- but it does change what the host's existing W and M "
             "mean relative to the new rows, so say so in the run notes.")
    m.add_argument("--host_device", type=int, default=0,
                   help="The device the HOST measured on. Asserted against the "
                        "journal because checkpoint.json records no device field, "
                        "so this is the only way to catch a cross-card merge.")
    m.add_argument("--dry_run", action="store_true",
                   help="Do everything, write nothing, print the report.")

    if suppress:
        for sp in (r, m):
            for action in sp._actions:
                if action.dest != "help":
                    action.default = argparse.SUPPRESS
    return p


def _explicit_dests(argv: Sequence[str]) -> set:
    """Which options the user actually typed, as dest names."""
    try:
        ns = _build_parser(suppress=True).parse_args(list(argv))
    except SystemExit:
        return set()
    return set(vars(ns).keys())


def _llm_args(a: argparse.Namespace) -> argparse.Namespace:
    """The six attributes main_memory_latest._make_llm_caller's closure reads.

    Deliberately minimal rather than handing our own namespace through. If that
    closure grows a seventh attribute we want an AttributeError on the first call,
    not a silent fallback to whatever our parser happened to name the same thing.
    Verified against main_memory_latest.py:494-533.
    """
    return argparse.Namespace(
        model_name=a.model_name, server_type=a.server_type,
        server_address=a.server_address, server_port=a.server_port,
        temperature=a.temperature, top_p=a.top_p)


# ===========================================================================
# seeding
# ===========================================================================
def _resolve_task_root(p: Path) -> Path:
    """Accept either a task root or the batch directory above it."""
    p = Path(p).resolve()
    if (p / _CHECKPOINT_NAME).exists():
        return p
    found = sorted(p.glob(f"*/{_CHECKPOINT_NAME}"))
    if len(found) == 1:
        return found[0].parent
    if not found:
        raise SystemExit(
            f"[quick] No {_CHECKPOINT_NAME} under {p} or its immediate children. "
            f"Pass the task root (the directory that holds {_CHECKPOINT_NAME}).")
    raise SystemExit(
        f"[quick] {p} holds several task roots: "
        f"{', '.join(sorted(f.parent.name for f in found))}. Pass one of them.")


def _load_prior(path_str: str) -> Optional[MechanismPrior]:
    """Load a fitted mechanism prior, warning exactly as the host does.

    Same guard pattern as main_memory_latest.py:1580-1602: a missing, unreadable
    or empty prior degrades to plain UCT with a printed warning rather than
    killing the run -- but here it also costs the mechanism draw its ranking, so
    the warning matters more.
    """
    if not path_str:
        return None
    pp = Path(path_str)
    if not pp.exists():
        print(f"[quick] WARNING: --mcts_prior {pp} does not exist; continuing with "
              f"plain UCT and an unranked mechanism draw. Fit one with "
              f"scripts/build_mechanism_prior.py.", flush=True)
        return None
    try:
        prior = MechanismPrior.from_dict(json.loads(pp.read_text(encoding="utf-8")))
    except Exception as exc:
        print(f"[quick] WARNING: could not read the prior ({exc}); continuing with "
              f"plain UCT.", flush=True)
        return None
    if prior is None or not prior.table:
        print(f"[quick] WARNING: the prior at {pp} is empty; plain UCT it is.",
              flush=True)
        return None
    top = ", ".join(f"{m} {a:+.2f}%(n={n})" for m, a, n in prior.ranked(4))
    print(f"[quick] PUCT prior loaded from {pp}: {len(prior.table)} mechanisms, "
          f"fitted on {prior.fitted_on} edges. Top: {top}", flush=True)
    if prior.note:
        print(f"[quick]   fit: {prior.note}", flush=True)
    return prior


def _graph_from_flags(a: argparse.Namespace) -> MonteCarloTreeSearch:
    g = MonteCarloTreeSearch(
        c_puct=a.mcts_c_puct, lam=a.mcts_lam, widen_k=a.mcts_widen_k,
        widen_alpha=a.mcts_widen_alpha, max_depth=a.mcts_max_depth,
        reward_scale=a.mcts_reward_scale, c_prior=a.mcts_c_prior)
    g.prior = _load_prior(a.mcts_prior)
    return g


_SEARCH_FLAG_TO_ATTR = {
    "mcts_c_puct": "c_puct", "mcts_lam": "lam", "mcts_widen_k": "widen_k",
    "mcts_widen_alpha": "widen_alpha", "mcts_max_depth": "max_depth",
    "mcts_reward_scale": "reward_scale", "mcts_c_prior": "c_prior",
}


def _check_forked_params(tree: MonteCarloTreeSearch, a: argparse.Namespace,
                         explicit: set) -> None:
    """A search parameter passed on the CLI must agree with a forked blob.

    Hard error, naming both values, rather than silently picking one. Honouring
    the blob would ignore the operator; honouring the flag would change the search
    policy of a tree mid-life, which is exactly what to_dict()'s prior-persistence
    comment (utils/mcts.py:697-700) exists to prevent -- a policy change halfway
    through makes every earlier visit statistic mean something different from every
    later one, and nothing on disk records where the switch happened.
    """
    bad = []
    for flag, attr in _SEARCH_FLAG_TO_ATTR.items():
        if flag not in explicit:
            continue
        want = getattr(a, flag)
        have = getattr(tree, attr)
        if isinstance(have, float) and isinstance(want, (int, float)):
            same = abs(float(have) - float(want)) < 1e-12
        else:
            same = have == want
        if not same:
            bad.append(f"--{flag} {want!r} but the forked tree has {attr}={have!r}")
    if bad:
        raise SystemExit(
            "[quick] REFUSING to fork: the search parameters you passed disagree "
            "with the tree you are forking.\n  " + "\n  ".join(bad) +
            "\nDrop the flag to continue with the tree's policy, or fork with "
            "--from_kernel to start a fresh tree under the new policy.")


# ===========================================================================
# run
# ===========================================================================
def cmd_run(a: argparse.Namespace, explicit: set) -> int:
    from utils import clock_lock, gpu_lock, run_timing
    from utils.torch_ext_cache import sweep_stale_batons, sweep_unheld_batons

    # ---- gate 0: the retired flags ----------------------------------------
    # Named on stderr rather than dropped in silence, matching
    # main_memory_latest._normalise_retired_search_flags. Ignoring a flag quietly
    # is how a run ends up not being the run someone thought they launched, and
    # these two were load-bearing under the graph: one chose the state
    # abstraction, the other bounded how wrong it was allowed to be.
    for _flag, _why in (("mcgs_state_key",
                         "node identity is minted per kernel; there is no state "
                         "abstraction left to key on"),
                        ("mcgs_merge_tol",
                         "nothing merges, so there is no merge for a tolerance to "
                         "refuse")):
        if getattr(a, _flag, None) is not None:
            print(f"[quick] NOTE: --{_flag} is retired and IGNORED -- {_why}.",
                  flush=True)

    # ---- gate 1: the clock -------------------------------------------------
    # Copied from main_memory_latest.main():3481-3489 with what="quick". Every run
    # in this repo pins the clock or refuses to start, and a quick path that
    # skipped it would produce numbers that are not comparable with a normal run's
    # -- which is the entire premise of the merge below. ensure_locked is
    # inheritance-aware: launched under a normal run's environment it verifies and
    # returns without re-locking, and it cannot un-pin the parent.
    if a.no_clock_lock:
        os.environ[clock_lock.ENV_POLICY] = "0"
    if a.gpu_clock_mhz:
        os.environ[clock_lock.ENV_GPU_MHZ] = str(a.gpu_clock_mhz)
    try:
        clock_lock.ensure_locked(a.device, what="quick")
    except clock_lock.ClockLockError as exc:
        print(f"\n[clock] {exc}\n", flush=True)
        return 2
    print(clock_lock.describe(), flush=True)

    # ---- gate 2: GPU serialization ----------------------------------------
    # Refused, not warned, by analogy with the clock lock: an unserialized number
    # entering the tree is worse than a run that did not start, because the tree
    # keeps it forever and multiplies it into every descendant.
    if not gpu_lock.enabled() and not a.allow_unserialized:
        print("[gpu] KERNELMEM_GPU_LOCK is unset, so gpu_section() is a no-op "
              "(utils/gpu_lock.py:50-58) and this run's benches will NOT be "
              "serialized against a concurrent normal run. Export it to the SAME "
              "path in both processes, or pass --allow_unserialized if nothing "
              "else is touching the GPU.", flush=True)
        return 2

    # ---- run directory -----------------------------------------------------
    if a.resume:
        quick_dir = Path(a.resume).resolve()
        if find_tree_path(quick_dir) is None:
            print(f"[quick] --resume {quick_dir} has no {_TREE_NAME} "
                  f"(or {_TREE_NAME_LEGACY}).", flush=True)
            return 3
    else:
        # The directory name is chosen before the checkpoint is parsed, so with
        # --from_run and no --task it has to peek at the checkpoint's `task` field
        # itself. Worth the extra read: a tree full of directories all named
        # quick_<stamp>_quick is unnavigable, and the stamp alone does not say
        # which workload a run measured.
        task_hint = "quick"
        if a.task:
            task_hint = Path(a.task).stem
        elif a.from_run:
            try:
                _peek = json.loads(
                    (_resolve_task_root(a.from_run) / _CHECKPOINT_NAME
                     ).read_text(encoding="utf-8"))
                task_hint = Path(str(_peek.get("task") or "")).stem or "quick"
            except (OSError, ValueError, SystemExit):
                pass
        elif a.from_kernel:
            task_hint = Path(a.from_kernel).stem
        stamp_txt = datetime.now().strftime("%Y%m%d_%H%M%S")
        quick_dir = (Path(a.work_dir).resolve()
                     / f"quick_{stamp_txt}_{task_hint}")
        quick_dir.mkdir(parents=True, exist_ok=False)
    run_id = quick_dir.name
    code_dir, eval_dir = quick_dir / "code", quick_dir / "evaluation"
    io_dir = eval_dir / "llm_io"
    for d in (code_dir, eval_dir, io_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Read from wherever it is, write back under the current name -- so a
    # resumed graph-era run migrates its filename on its first rollout instead of
    # keeping a name that no longer describes it.
    _found = find_tree_path(quick_dir)
    tree_path = _found if (_found is not None and a.resume) else quick_dir / _TREE_NAME
    journal_path = quick_dir / _JOURNAL_NAME
    usage_csv = quick_dir / "usage.csv"

    # timing.csv lands inside the quick run, so it cannot collide with a
    # concurrent normal run's file even when both are writing every few minutes.
    run_timing.set_timing_log(quick_dir / "timing.csv")
    run_timing.event("process_start", detail=f"mcts_quick {run_id}")
    _install_stop_handler()
    os.environ.setdefault("KERNELMEM_LINEAGE_LABEL", f"quick:{run_id}")

    # Clear orphaned torch extension build locks before anything compiles: a baton
    # left by a killed process never expires on its own, and the next kernel that
    # picks that extension name hangs until the compile alarm and is then
    # misreported as an illegal memory access.
    sweep_stale_batons()

    mml = _load_host_helpers()
    from utils.kernel_io import save_kernel_code

    # ---- seed --------------------------------------------------------------
    registry: Dict[str, Any] = {}
    rep_sha: Dict[str, str] = {}
    applied_obs: set = set()

    if a.resume:
        tree, meta, applied_obs = _read_tree(tree_path)
        rep_sha = dict(meta.get("rep_sha") or {})
        task_path = Path(meta["task"])
        records = _read_journal(journal_path)
        replayed = 0
        for rec in records:
            landed, why = _apply_record(tree, rec, applied_obs, rep_sha)
            if why not in ("duplicate",):
                replayed += 1
                meta.setdefault("outcomes", {})[rec.get("obs_id", "?")] = {
                    "landed_key_local": landed, "apply_reason": why}
        if replayed:
            print(f"[quick] --resume replayed {replayed} journal record(s) the tree "
                  f"was missing -- this is the crash window between the journal "
                  f"fsync and the tree write.", flush=True)
        meta["rollouts_done"] = max(int(meta.get("rollouts_done") or 0), len(records))
        meta["applied_obs"] = sorted(applied_obs)
        meta["rep_sha"] = rep_sha
        _write_tree(tree_path, tree, meta)
        # The stored run configuration wins on resume: re-deriving it from CLI
        # defaults would silently change the bench parameters, and a tree whose
        # edges were measured at two different --repeat values is one chain over
        # two bases.
        bench_cfg, verdict_cfg = meta["bench"], meta["verdict"]
        policy_cfg = meta["policy"]
        for src, dst in ((bench_cfg, ("device", "warmup", "repeat", "tol")),
                         (verdict_cfg, ("base_margin", "base_reps", "base_max_reps",
                                        "base_sigma"))):
            for k in dst:
                if k not in explicit:
                    setattr(a, k, src[k])
        a.mechanism_policy = policy_cfg["mechanism_policy"]
        a.mechanism_seed = policy_cfg["mechanism_seed"]
        a.count_failures = policy_cfg["count_failures"]
        print(f"[quick] Resuming {run_id} at rollout {meta['rollouts_done']}"
              f"/{a.rounds}.", flush=True)
    else:
        try:
            tree, meta, task_path = _seed(a, explicit, quick_dir, code_dir,
                                           eval_dir, mml, save_kernel_code,
                                           registry, rep_sha, applied_obs, run_id)
        except SystemExit:
            # Seeding refuses for several reasons that are the operator's fault
            # (a conflicting --mcts_* against a forked tree, a missing task, an
            # unrunnable seed kernel), and every one of them leaves an empty
            # quick_<stamp>_<task> directory behind. Ten of those in run/ are ten
            # directories a later reader has to open to discover they are empty,
            # so remove ours if nothing MEANINGFUL was written into it. timing.csv
            # is deliberately not meaningful here: set_timing_log creates it at
            # startup, before seeding runs, so testing for "any file" never fires.
            keep = ([tree_path, journal_path] + list(code_dir.glob("*"))
                    + list(eval_dir.glob("eval_*.json")))
            if not any(p.exists() for p in keep):
                import shutil
                shutil.rmtree(quick_dir, ignore_errors=True)
            raise
        _write_tree(tree_path, tree, meta)

    if tree.root is None or tree.root not in tree.nodes:
        print("[quick] The tree has no root; nothing can be selected. Seed it with "
              "--from_kernel.", flush=True)
        return 3

    # ---- bound the injected helpers ---------------------------------------
    call_llm = mml._make_llm_caller(_llm_args(a))

    def _prompt_fn(arch_path: Path, gpu_name, strategy) -> str:
        # arch_path is the PARENT KERNEL, not the task file: the template drops it
        # into [BASE KERNEL FILE] as the thing to edit. history_block is left ""
        # because prompts/optimization_memory_latest.py:241 hardcodes it away and
        # the template carries no $history_block placeholder -- siblings_context
        # inside `evidence` is what replaces it.
        return mml.build_optimization_prompt(
            arch_path=arch_path, gpu_name=gpu_name, history_block="",
            optimization_suggestion=strategy)

    def _rollout_fn(prompt: str, idx: int):
        # call_type="optimization" is not cosmetic: it selects
        # _extract_kernel_from_optimization_reply, which splits on
        # "=== KERNEL CODE STARTS BELOW ===". Any other value takes the FIRST code
        # block, which is the Section-A checklist, not the kernel.
        #
        # save_kernel_code names files at one-second resolution
        # (utils/kernel_io.py:194), so two kernels saved inside the same second
        # overwrite each other. Harmless at ~5 min/rollout; a best-of-N loop added
        # here later would walk straight into it.
        return mml._llm_to_kernel(
            prompt, code_dir, call_llm, io_dir, idx, log_path=usage_csv,
            call_type="optimization", model_name=a.rollout_model,
            reasoning_effort=a.rollout_effort)

    def _bench_fn(ind) -> None:
        mml._bench_and_score(ind, ref_py=task_path, device_idx=a.device,
                             warmup=a.warmup, repeat=a.repeat, tol=a.tol,
                             phase="quick", metrics_dir=eval_dir)

    def _verdict_fn(base_p: Path, cand_p: Path):
        if a.base_max_reps <= 0:
            return None
        print(f"[quick] Re-measuring parent and candidate side by side "
              f"({a.base_reps}-{a.base_max_reps} interleaved reps)...", flush=True)
        return mml._paired_base_verdict(
            task_path, base_p, cand_p, device_idx=a.device, warmup=a.warmup,
            repeat=a.repeat, tol=a.tol, margin=a.base_margin,
            min_reps=a.base_reps, max_reps=a.base_max_reps, sigma=a.base_sigma)

    def _resolve_fn(name, path):
        return _resolve(registry, name, path, kernel_cls=mml.KernelIndividual)

    dirs = {"code": code_dir, "eval": eval_dir, "io": io_dir}
    clock_state = clock_lock.state()
    stamp = {
        "quick_run_id": run_id, "task": str(task_path), "ref_py": str(task_path),
        "device": int(a.device),
        "clock_locked": bool(clock_state.get("locked")),
        "clock_mhz": clock_state.get("target_gpu_mhz"),
        "gpu_lock_enabled": bool(gpu_lock.enabled()),
    }
    seed_base = a.mechanism_seed or (
        int(hashlib.sha1(run_id.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF)
    meta["policy"]["mechanism_seed"] = seed_base

    # ---- the loop ----------------------------------------------------------
    llm_errors_in_a_row = 0
    start = int(meta.get("rollouts_done") or 0)
    for i in range(start, a.rounds):
        if _STOP["requested"]:
            print(f"[stop] Stopping before rollout {i}. Completed {i} of {a.rounds}. "
                  f"Resume with: python -m utils.mcts_quick run --resume {quick_dir}",
                  flush=True)
            break
        run_timing.set_round(i)
        # The startup sweep cannot help a run that poisons its own cache: a build
        # killed mid-ninja leaves a lock only minutes old, far inside the age gate.
        sweep_unheld_batons()
        # Not called anywhere in main_memory_latest.py, and it should be: the
        # startup check cannot catch another user, another tool or a driver reset
        # moving the clock during a 2-hour run. Warn-only (fatal=False); a False
        # return stops the loop gracefully, because every remaining rollout would
        # produce numbers that fail merge precondition R7 anyway.
        if not a.no_clock_lock and not clock_lock.assert_still_locked(a.device,
                                                                     what="quick"):
            print("[quick] Stopping: the clock has drifted off its lock, so further "
                  "rollouts would not be comparable with the ones already in the "
                  "tree (and merge would refuse them). tree.json is current.",
                  flush=True)
            break

        t0 = time.perf_counter()
        rec = _one_rollout(
            tree, rollout_idx=i, rng=random.Random(seed_base ^ i),
            policy=a.mechanism_policy, gpu_name=a.gpu, registry=registry,
            dirs=dirs, count_failures=a.count_failures, stamp=stamp,
            resolve_fn=_resolve_fn, prompt_fn=_prompt_fn, rollout_fn=_rollout_fn,
            bench_fn=_bench_fn, verdict_fn=_verdict_fn,
            sanitize_fn=mml._sanitize_error_message)
        run_timing.record("quick:rollout", time.perf_counter() - t0, round_idx=i)
        if rec is None:
            # Selection produced a state whose code is gone. Nothing to journal.
            continue

        if rec.get("error_type") and rec.get("kernel_name") is None:
            llm_errors_in_a_row += 1
        else:
            llm_errors_in_a_row = 0

        _append_journal(journal_path, rec)
        landed, why = _apply_record(tree, rec, applied_obs, rep_sha)
        meta.setdefault("outcomes", {})[rec["obs_id"]] = {
            "landed_key_local": landed, "apply_reason": why}
        meta["rollouts_done"] = i + 1
        meta["applied_obs"] = sorted(applied_obs)
        meta["rep_sha"] = rep_sha
        _write_tree(tree_path, tree, meta)

        if landed is not None:
            node = tree.nodes[landed]
            print(f"[quick] Child node {landed} (depth {node.depth}): "
                  f"{rec['rel_pct']:+.2f}% ({rec['basis']}) -> value "
                  f"{node.value:.4f}, N={node.N}", flush=True)
            best = tree.best()
            if best is not None:
                print(f"[quick] Best node so far: {best.key} at "
                      f"{best.value:.4f} via {best.kernel} "
                      f"(N={best.N}, Q={best.q(tree.lam):.3f})", flush=True)
        else:
            print(f"[quick] Rollout {i} did not enter the tree: {why}", flush=True)

        if llm_errors_in_a_row >= 3:
            # Three in a row is not a bad prompt, it is a dead credential or a
            # dead server. Further rollouts only burn wall clock at 2.6 min each.
            print("[quick] Three consecutive model-call failures; aborting. Fix the "
                  "credential or the server and resume with "
                  f"--resume {quick_dir}", flush=True)
            run_timing.event("process_exit", detail="llm_errors")
            return 4

    st = tree.stats()
    print(f"\n[quick] Done. {meta['rollouts_done']} rollout(s). tree: "
          f"{json.dumps(st)}", flush=True)
    best = tree.best()
    if best is not None:
        print(f"[quick] Best: {best.key} at {best.value:.4f} via {best.kernel} "
              f"({best.kernel_path})", flush=True)
    print(f"[quick] Graph:   {tree_path}")
    print(f"[quick] Journal: {journal_path}")
    print(f"[quick] Merge with: python -m utils.mcts_quick merge "
          f"--quick_run {quick_dir} --host <task_root>")
    run_timing.event("process_exit", detail=f"rollouts={meta['rollouts_done']}")
    return 0


def _seed(a, explicit, quick_dir: Path, code_dir: Path, eval_dir: Path, mml,
          save_kernel_code, registry, rep_sha, applied_obs,
          run_id: str) -> Tuple[MonteCarloTreeSearch, Dict[str, Any], Path]:
    """Build the initial tree and meta for a fresh quick run."""
    from utils import clock_lock, gpu_lock

    fork_meta: Optional[Dict[str, Any]] = None
    seed_source: Optional[Path] = None
    seed_mode: str

    if a.from_run:
        seed_mode = "from_run"
        host_root = _resolve_task_root(a.from_run)
        ckpt_path = host_root / _CHECKPOINT_NAME
        raw = ckpt_path.read_bytes()
        ckpt = json.loads(raw.decode("utf-8"))
        if int(ckpt.get("version") or 0) != 1:
            raise SystemExit(f"[quick] {ckpt_path} is checkpoint version "
                             f"{ckpt.get('version')}, expected 1.")
        host_ckpt_sha = hashlib.sha1(raw).hexdigest()
        task_path = Path(a.task).resolve() if a.task else Path(ckpt["task"])
        if not task_path.exists():
            raise SystemExit(f"[quick] The task file {task_path} does not exist. "
                             f"Pass --task if it has moved.")

        blob = ckpt.get("mcts") or ckpt.get("mcgs")
        if isinstance(blob, dict) and blob.get("nodes"):
            tree = MonteCarloTreeSearch.from_dict(blob)
            if not tree.root or tree.root not in tree.nodes:
                raise SystemExit(
                    f"[quick] {ckpt_path}'s tree has no usable root "
                    f"({tree.root!r}); select() would return None forever.")
            if tree.migration_note:
                print(f"[quick] {tree.migration_note}", flush=True)
            _check_forked_params(tree, a, explicit)
            if a.mcts_prior:
                print("[quick] NOTE: ignoring --mcts_prior. The forked tree carries "
                      "its own prior, and a resumed search must select with the "
                      "policy it started with (utils/mcts.py:697-700).", flush=True)
            fork_meta = {
                "host_task_root": str(host_root),
                "host_checkpoint_sha": host_ckpt_sha,
                "fork_mcts_sha": hashlib.sha1(
                    json.dumps(blob, sort_keys=True).encode()).hexdigest(),
                "forked_from_mcts": True,
                "fork_root": tree.root,
                "fork_node_keys": sorted(tree.nodes),
            }
            print(f"[quick] Forked {len(tree.nodes)} states / "
                  f"{tree.total_visits} visits from {ckpt_path}.", flush=True)
        else:
            # The measured common case: 0 of the 8 checkpoint.json files under
            # run/ carry a search blob, because every existing run predates
            # --search mcts or ran --search ratchet.
            tree = _graph_from_flags(a)
            which, entry = None, None
            for name in ("best", "base", "current"):
                e = ckpt.get(name)
                if (e and e.get("code_path") and Path(e["code_path"]).exists()
                        and isinstance(e.get("score"), (int, float))):
                    which, entry = name, e
                    break
            if entry is None:
                raise SystemExit(
                    f"[quick] {ckpt_path} has no best/base/current entry with a "
                    f"score and a kernel file still on disk; nothing to seed from.")
            # Copied into the quick run rather than referenced in place: the host
            # may rewrite or clean its own code/ directory mid-run, and _resolve
            # reads rep_path from disk at every selection.
            src = Path(entry["code_path"])
            dst = save_kernel_code(src.read_text(encoding="utf-8"), code_dir)
            seed_source = src
            root = tree.observe(kernel_name=dst.stem, kernel_path=str(dst),
                                value=float(entry["score"]), parent_key=None,
                                runnable=True,
                                note=f"seed from {host_root.name} {which}")
            root_key = root.key
            rep_sha[root_key] = hashlib.sha1(
                _norm_code(dst.read_text(encoding="utf-8")).encode()).hexdigest()
            fork_meta = {
                "host_task_root": str(host_root),
                "host_checkpoint_sha": host_ckpt_sha,
                "fork_mcts_sha": None,
                "forked_from_mcts": False,
                "fork_root": root_key,
                "fork_node_keys": [root_key],
            }
            print(f"\n[quick] ============================================\n"
                  f"[quick] {ckpt_path} carries NO search tree, so this is a "
                  f"BOOTSTRAP, not a fork: the root was seeded from its "
                  f"'{which}' kernel at {float(entry['score']):.4f}.\n"
                  f"[quick] The resulting keys are not in that host's key space by "
                  f"construction, so `merge` will refuse it unless you pass "
                  f"--adopt and the host still has no tree of its own.\n"
                  f"[quick] ============================================\n",
                  flush=True)

    else:
        seed_mode = "from_kernel"
        if not a.task:
            raise SystemExit("[quick] --from_kernel requires --task.")
        task_path = Path(a.task).resolve()
        if not task_path.exists():
            raise SystemExit(f"[quick] The task file {task_path} does not exist.")
        src = Path(a.from_kernel).resolve()
        dst = save_kernel_code(src.read_text(encoding="utf-8"), code_dir)
        seed_source = src
        ind = mml.KernelIndividual(dst.read_text(encoding="utf-8"))
        ind.code_path = dst
        print(f"[quick] Benching the seed kernel {dst.name} ...", flush=True)
        mml._bench_and_score(ind, ref_py=task_path, device_idx=a.device,
                             warmup=a.warmup, repeat=a.repeat, tol=a.tol,
                             phase="quick_seed", metrics_dir=eval_dir)
        if not (getattr(ind, "metrics", {}) or {}).get("runnable") or ind.score is None \
                or ind.score == float("-inf"):
            m = getattr(ind, "metrics", {}) or {}
            raise SystemExit(
                f"[quick] The seed kernel is not runnable "
                f"({m.get('error_type')}: {str(m.get('message'))[:400]}). A root "
                f"the tree cannot branch from is not a tree.")
        tree = _graph_from_flags(a)
        _register(registry, ind)
        root = tree.observe(kernel_name=dst.stem, kernel_path=str(dst),
                            value=float(ind.score), parent_key=None, runnable=True,
                            note="seed (--from_kernel)")
        root_key = root.key
        rep_sha[root_key] = hashlib.sha1(
            _norm_code(ind.code or "").encode()).hexdigest()
        print(f"[quick] Root node {root_key} <- {dst.stem} at {ind.score:.4f}",
              flush=True)

    meta = {
        "quick_version": _QUICK_VERSION,
        "kind": "mcts_quick_tree",
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": str(task_path),
        "seed_mode": seed_mode,
        "seed_source_path": str(seed_source) if seed_source else None,
        "fork": fork_meta,
        "bench": {"device": int(a.device), "warmup": int(a.warmup),
                  "repeat": int(a.repeat), "tol": float(a.tol)},
        "verdict": {"base_margin": float(a.base_margin), "base_reps": int(a.base_reps),
                    "base_max_reps": int(a.base_max_reps),
                    "base_sigma": float(a.base_sigma)},
        "llm": {"rollout_model": a.rollout_model, "rollout_effort": a.rollout_effort,
                "model_name": a.model_name, "temperature": a.temperature,
                "top_p": a.top_p},
        "policy": {"mechanism_policy": a.mechanism_policy,
                   "mechanism_seed": int(a.mechanism_seed),
                   "count_failures": bool(a.count_failures)},
        "clock": clock_lock.state(),
        "gpu_lock": str(gpu_lock.lock_path()) if gpu_lock.lock_path() else None,
        "rollouts_done": 0,
        "applied_obs": sorted(applied_obs),
        "rep_sha": dict(rep_sha),
        "outcomes": {},
    }
    return tree, meta, task_path


# ===========================================================================
# merge
# ===========================================================================
def _host_is_live(host_root: Path, idle_s: int) -> Optional[str]:
    """Reason the host looks live, or None.

    Two independent tests because neither alone is sufficient: a run parked
    between rounds has a stale mtime but is still going to write, and a run
    launched with an unusual command line may not name its own task root.
    """
    ckpt = host_root / _CHECKPOINT_NAME
    if idle_s > 0 and ckpt.exists():
        age = time.time() - ckpt.stat().st_mtime
        if age < idle_s:
            return (f"{ckpt} was written {age:.0f}s ago, inside --host_idle_s "
                    f"{idle_s}; the median normal round is 570s, so this run is "
                    f"probably mid-round and will overwrite the merge")
    needles = {str(host_root), str(host_root.parent)}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", "replace")
            except OSError:
                continue
            if "main_memory_latest" not in cmd:
                continue
            if any(n in cmd for n in needles):
                return f"pid {entry.name} looks like a live host run: {cmd.strip()[:200]}"
    except OSError:
        pass
    return None


def _code_sha_of(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha1(_norm_code(p.read_text(encoding="utf-8")).encode()).hexdigest()


def _orphans(tree: MonteCarloTreeSearch) -> List[str]:
    if not tree.root or tree.root not in tree.nodes:
        return sorted(tree.nodes)
    seen = {tree.root}
    q = deque([tree.root])
    while q:
        for k in tree.nodes[q.popleft()].children:
            if k in tree.nodes and k not in seen:
                seen.add(k)
                q.append(k)
    return sorted(set(tree.nodes) - seen)


def merge(quick_run: Path, host: Path, *, in_place: bool = False,
          adopt: bool = False, host_idle_s: int = 900,
          force_reward_scale: bool = False, host_device: int = 0,
          dry_run: bool = False) -> Dict[str, Any]:
    """Fold a quick run's observations into a host checkpoint's search blob.

    Fork-and-replay, never a dict-level union. Every invariant in
    MonteCarloTreeSearch lives INSIDE observe()/backup() -- the merge-tolerance
    split guard (utils/mcts.py:504-511), root promotion (:518-519), the depth min
    (:535), edge dedup (:527-531), `tried` accounting (:536-540) -- and a union
    bypasses all of them. Concretely: H.nodes.update(Q.nodes) replaces the host's
    better representative with the quick run's worse one, points rep_path into
    another run's directory (which the host's _resolve will happily read), and
    imports a rep_value that is a chain anchored at a DIFFERENT seed, while best()
    ranks by rep_value across every node in the dict.

    So the merge replays gains, not values: for each record it recomputes the
    child key from the HOST's route to the parent, recomputes the value as
    host_parent.value * (1 + rel_pct/100), and recomputes the reward at the
    HOST's reward_scale.

    Returns {"ok", "reason", "report"}. Writes nothing when ok is False.
    """
    from utils import clock_lock

    quick_run, host = Path(quick_run).resolve(), Path(host).resolve()
    report: Dict[str, Any] = {"quick_run": str(quick_run), "host": str(host)}

    def _no(reason: str, detail: str) -> Dict[str, Any]:
        report["detail"] = detail
        return {"ok": False, "reason": reason, "report": report}

    _qf = find_tree_path(quick_run)
    if _qf is None:
        return _no("R0-no-quick-tree",
                   f"{quick_run} holds no {_TREE_NAME} (or {_TREE_NAME_LEGACY}); "
                   f"there is nothing to merge")
    quick_blob = json.loads(_qf.read_text(encoding="utf-8"))
    # The same two kinds `_read_tree` accepts, so a quick run cannot be resumable
    # and unmergeable at once. It could: --resume goes through _read_tree, which
    # takes the graph-era kind, and _write_tree copies `kind` forward from meta
    # unchanged -- so a graph-era run could be extended indefinitely while merge
    # refused it forever, which is the only thing the quick path exists to do.
    if quick_blob.get("kind") not in ("mcts_quick_tree", "mcgs_quick_graph") or \
            int(quick_blob.get("quick_version") or 0) != _QUICK_VERSION:
        return _no("bad-quick-tree",
                   f"{_qf} is not a v{_QUICK_VERSION} quick tree "
                   f"(kind={quick_blob.get('kind')!r}, "
                   f"quick_version={quick_blob.get('quick_version')!r})")
    records = _read_journal(quick_run / _JOURNAL_NAME)
    Q = MonteCarloTreeSearch.from_dict(quick_blob.get("mcts") or quick_blob.get("mcgs"))

    ckpt_path = host / _CHECKPOINT_NAME
    host_ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    if int(host_ckpt.get("version") or 0) != 1:
        return _no("bad-host-checkpoint",
                   f"{ckpt_path} is version {host_ckpt.get('version')}, expected 1")
    ledger_path = host / _LEDGER_NAME
    ledger = (json.loads(ledger_path.read_text(encoding="utf-8"))
              if ledger_path.exists()
              else {"version": 1, "applied": [], "merges": [], "rep_sha": {}})

    # ---- R7: the clock -----------------------------------------------------
    qclock = quick_blob.get("clock") or {}
    if not qclock.get("locked"):
        return _no("R7-clock",
                   "the quick run was measured with the GPU clock UNLOCKED, so its "
                   "numbers are not comparable with the host's -- which is the "
                   "entire premise of merging them onto one chain")
    here = clock_lock.state()
    if here.get("locked") and here.get("target_gpu_mhz") and \
            qclock.get("target_gpu_mhz") and \
            int(here["target_gpu_mhz"]) != int(qclock["target_gpu_mhz"]):
        return _no("R7-clock",
                   f"the quick run was pinned at {qclock['target_gpu_mhz']} MHz but "
                   f"this machine is currently pinned at {here['target_gpu_mhz']} MHz")
    if not here.get("locked") and records:
        # checkpoint.json records no clock state at all (see _save_checkpoint's
        # schema at main_memory_latest.py:1265-1306), so the host's frequency can
        # only be cross-checked against the machine's CURRENT pin. Reported, not
        # refused, because refusing would make every offline merge impossible --
        # and the operator can still tell from the quick run's own stamp what it
        # was measured at.
        print("[merge] NOTE: this process holds no clock lock, so the quick run's "
              f"{qclock.get('target_gpu_mhz')} MHz could not be cross-checked "
              "against the host -- checkpoint.json records no clock state.",
              flush=True)

    # ---- R8: same workload, same card --------------------------------------
    host_task = Path(str(host_ckpt.get("task") or ""))
    for rec in records:
        if Path(str(rec.get("task"))).name != host_task.name:
            return _no("R8-task",
                       f"journal record {rec.get('rollout_idx')} measured "
                       f"{rec.get('task')}, the host measured {host_task}")
        if int(rec.get("device", host_device)) != int(host_device):
            return _no("R8-task",
                       f"journal record {rec.get('rollout_idx')} measured on device "
                       f"{rec.get('device')}, --host_device says {host_device}")

    # ---- R9: the host must not be running ----------------------------------
    if not dry_run:
        live = _host_is_live(host, host_idle_s)
        if live:
            return _no("R9-host-live", live)

    host_blob = host_ckpt.get("mcts") or host_ckpt.get("mcgs")
    adopting = not (isinstance(host_blob, dict) and host_blob.get("nodes"))

    # ---- R1 / R10: is this a legitimate fork of THIS host? -----------------
    fork = quick_blob.get("fork") or {}
    if adopting and not adopt:
        return _no("R1-host-has-no-tree",
                   f"{ckpt_path} carries no search tree. Installing a foreign tree "
                   f"would make the host's next round select from states it never "
                   f"measured; pass --adopt if that is what you want.")
    if not (fork.get("forked_from_mcts") or fork.get("forked_from_mcgs")) and not adopt:
        return _no("R10-not-a-fork",
                   "the quick tree was bootstrapped from a kernel, not forked from "
                   "this host's tree, so its keys are not in the host's key space "
                   "by construction")

    # ---- R5: the two roots must agree on the seed's measured value ---------
    qroot = Q.nodes.get(Q.root) if Q.root else None
    if qroot is None:
        return _no("bad-quick-tree", "the quick tree has no root node")

    if not adopting:
        H = MonteCarloTreeSearch.from_dict(host_blob)
        hroot = H.nodes.get(H.root) if H.root else None
        if hroot is None:
            return _no("R3-root-key", "the host tree has no root node")
        if H.root != Q.root:
            return _no("R3-root-key",
                       f"host root {H.root!r} != quick root {Q.root!r}; a subtree "
                       f"hanging off a different root is unreachable by select() "
                       f"forever, yet best() and stats() still report it")
        hsha, qsha = _code_sha_of(hroot.kernel_path), _code_sha_of(qroot.kernel_path)
        if hsha is None:
            return _no("R4-root-source",
                       f"the host root's kernel {hroot.kernel_path} is gone, so the "
                       f"two roots cannot be shown to be the same code -- and under "
                       f"`mechanisms` keying the root is an x: key hashing a FILE "
                       f"STEM, so equal keys prove nothing")
        if qsha is None:
            return _no("R4-root-source",
                       f"the quick root's kernel {qroot.kernel_path} is gone")
        if hsha != qsha:
            return _no("R4-root-source",
                       f"the two roots share a key but not a source "
                       f"({hsha[:12]} vs {qsha[:12]})")
        if hroot.value > 0 and abs(qroot.value / hroot.value - 1.0) > 0.02:
            return _no("R5-root-value",
                       f"the roots' measured values disagree by "
                       f"{abs(qroot.value/hroot.value - 1)*100:.2f}% "
                       f"({hroot.value:.4f} vs {qroot.value:.4f}); "
                       f"between-process CV is 0.3-1.4%, so >2% means the two runs "
                       f"were not on one clock-lock basis")
        if abs(H.reward_scale - Q.reward_scale) > 1e-12 and not force_reward_scale:
            return _no("R6-reward-scale",
                       f"host reward_scale {H.reward_scale} != quick "
                       f"{Q.reward_scale}; pass --force_reward_scale to recompute "
                       f"every reward at the host's scale")
        if fork.get("host_checkpoint_sha") and \
                fork["host_checkpoint_sha"] != hashlib.sha1(
                    ckpt_path.read_bytes()).hexdigest():
            print("[merge] NOTE: the host has advanced since the fork (its "
                  "checkpoint.json sha differs). That is expected and is exactly "
                  "what the replay handles -- parents are re-found by BFS from the "
                  "host's CURRENT root.", flush=True)
    else:
        # --adopt: preconditions R5/R7/R8/R9 already passed above; additionally the
        # quick root must be the same code as the host's best/base kernel, or the
        # tree being installed describes a lineage the host never ran.
        H = None
        host_seed = None
        for name in ("best", "base", "current"):
            e = host_ckpt.get(name)
            if e and e.get("code_path"):
                host_seed = e
                break
        if host_seed is None:
            return _no("R4-root-source",
                       "the host checkpoint names no kernel to compare the quick "
                       "root against")
        hsha, qsha = _code_sha_of(host_seed.get("code_path")), _code_sha_of(qroot.kernel_path)
        if hsha is None or qsha is None or hsha != qsha:
            return _no("R4-root-source",
                       f"the quick root's source does not match the host's "
                       f"{host_seed.get('code_path')}")
        if str(fork.get("host_task_root") or "") not in ("", str(host)):
            return _no("R4-root-source",
                       f"this quick run was forked from "
                       f"{fork.get('host_task_root')}, not {host}")

    # ---- replay ------------------------------------------------------------
    reasons: Dict[str, int] = {}
    applied_ids = set(ledger.get("applied") or [])
    rep_sha = dict(ledger.get("rep_sha") or {})
    outcomes = (quick_blob.get("outcomes") or {})
    refused_local: set = set()
    n_applied = 0
    depth_fixes: List[str] = []
    orphan_before: List[str] = []

    if adopting:
        merged_mcts = quick_blob.get("mcts") or quick_blob["mcgs"]
        report.update(
            mode="adopt", records=len(records), applied=0, refusals={},
            nodes_before=0, nodes_after=len(merged_mcts.get("nodes") or {}),
            visits_before=0, visits_after=int(merged_mcts.get("total_visits") or 0),
            depth_fixes=[], orphans=[])
        print("[merge] --adopt: installing the quick tree wholesale. The host's "
              "base_kernel and best_kernel are NOT touched. Note the installed "
              "tree's rep_path entries point into "
              f"{quick_run}/code -- that directory must not be moved or deleted, "
              "because _resolve reads them from disk at every selection.",
              flush=True)
    else:
        pre = json.loads(json.dumps(H.to_dict()))
        prior_before = json.dumps(H.prior.to_dict() if H.prior else None, sort_keys=True)
        orphan_before = _orphans(H)
        report["nodes_before"] = len(H.nodes)
        report["visits_before"] = H.total_visits

        # LOCAL id -> HOST id. Identity for every node that existed at the fork
        # (the quick tree was cloned from the host blob, so those ids denote the
        # same node in both), and nothing else: after the fork both trees mint
        # from the same counter independently, so a quick-minted `n0007` and a
        # host-minted `n0007` are different kernels. Every id beyond the fork set
        # has to be earned by its own record landing, below.
        fork_keys = fork.get("fork_node_keys")
        if not fork_keys:
            return _no("R11-no-fork-keys",
                       "the quick run's fork metadata carries no fork_node_keys, so "
                       "its local node ids cannot be translated into host ids. "
                       "Merging by raw id would graft records onto whatever node "
                       "happens to hold the same minted id in the host.")
        id_map: Dict[str, str] = {k: k for k in fork_keys if k in H.nodes}
        report["fork_keys_mapped"] = len(id_map)

        for rec in records:
            # The LOCAL id this record produced in the quick run's own tree. It is
            # read from `outcomes` and never predicted: a node id is minted inside
            # observe(), so the graph-era `child_key_local` -- a content key the
            # rollout could compute in advance -- has no successor and is gone.
            out = outcomes.get(rec.get("obs_id") or "", {})
            local_child = out.get("landed_key_local")
            if rec.get("parent_key") in refused_local:
                reasons["refused-subtree"] = reasons.get("refused-subtree", 0) + 1
                if local_child:
                    refused_local.add(local_child)
                continue
            landed, why = _apply_record(H, rec, applied_ids, rep_sha, id_map=id_map)
            reasons[why] = reasons.get(why, 0) + 1
            if why == "applied":
                n_applied += 1
                # The child is now addressable from a later record's parent_key.
                # Without `landed_key_local` this link cannot be made, so the
                # quick run's descendants of this node will be refused rather
                # than grafted somewhere plausible -- which is the right way to
                # be wrong.
                if local_child:
                    id_map[local_child] = landed
            elif why in ("parent-not-in-target", "parent-unreachable-from-root",
                         "parent-identity-mismatch"):
                # This record produced nothing in the host, so anything the quick
                # run hung off it has no parent here either.
                if local_child:
                    refused_local.add(local_child)

        # ---- post-pass -----------------------------------------------------
        # Depth is "shortest route known when this node was last observed", not an
        # invariant: adding a shorter route lowers a node's depth (utils/mcts.py
        # :535) but never re-deepens its descendants. A stale-large depth silently
        # costs usable search depth at select()'s `node.depth >= max_depth` bail.
        if H.root and H.root in H.nodes:
            seen = {H.root: 0}
            q = deque([H.root])
            while q:
                cur = q.popleft()
                for k in H.nodes[cur].children:
                    if k in H.nodes and k not in seen:
                        seen[k] = seen[cur] + 1
                        q.append(k)
            for k, d in seen.items():
                if H.nodes[k].depth != d:
                    depth_fixes.append(f"{k}: {H.nodes[k].depth} -> {d}")
                    H.nodes[k].depth = d
        orphan_after = _orphans(H)
        new_orphans = sorted(set(orphan_after) - set(orphan_before))
        if new_orphans:
            return _no("post-orphan",
                       f"the merge created {len(new_orphans)} node(s) unreachable "
                       f"from the root: {new_orphans[:5]}")
        if orphan_before:
            print(f"[merge] WARNING: the HOST already had {len(orphan_before)} node(s) "
                  f"unreachable from its root before this merge; they are left "
                  f"alone: {orphan_before[:5]}", flush=True)
        if json.dumps(H.prior.to_dict() if H.prior else None,
                      sort_keys=True) != prior_before:
            return _no("post-prior",
                       "the merge changed the host's prior; a resumed search must "
                       "select with the policy it started with")
        # `tried` has no dedup in observe() (utils/mcts.py:536-540) and
        # siblings_context shows only the last 8 entries -- the ONLY
        # anti-repetition signal reaching the prompt. A duplicated line fills that
        # window with the same warning eight times.
        n_dedup = 0
        for node in H.nodes.values():
            seen_t, keep = set(), []
            for t in node.tried:
                sig = (t.get("mechanism"), t.get("child"), t.get("value"),
                       t.get("runnable"))
                if sig in seen_t:
                    n_dedup += 1
                    continue
                seen_t.add(sig)
                keep.append(t)
            node.tried = keep
        merged_mcts = H.to_dict()
        json.dumps(merged_mcts)  # round-trip guard: refuse to write what cannot load
        MonteCarloTreeSearch.from_dict(json.loads(json.dumps(merged_mcts)))
        report.update(mode="replay", records=len(records), applied=n_applied,
                      refusals=reasons, nodes_after=len(H.nodes),
                      visits_after=H.total_visits, depth_fixes=depth_fixes,
                      tried_dedup=n_dedup, orphans=orphan_before,
                      stats=H.stats(), pre_nodes=len(pre.get("nodes") or {}))

    if dry_run:
        return {"ok": True, "reason": None, "report": report, "dry_run": True}

    # ---- write: LEDGER FIRST, then the checkpoint --------------------------
    # A crash between the two must LOSE records, not double-count them. Ledger
    # first means a crash leaves obs_ids marked applied that are not in the
    # checkpoint -- recoverable by hand and visible in the report. Checkpoint
    # first would let the next merge re-apply records already in, and
    # _apply_record's duplicate guard reads the ledger, so it could not catch
    # them: N and W would double together, q() would not move, and the node would
    # earn a widening budget it never paid for. That is the unrecoverable
    # direction.
    ledger["applied"] = sorted(applied_ids)
    ledger["rep_sha"] = rep_sha
    ledger.setdefault("merges", []).append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "quick_run": str(quick_run),
        "quick_run_id": quick_blob.get("run_id"),
        "fork_mcts_sha": ((quick_blob.get("fork") or {}).get("fork_mcts_sha")
                          or (quick_blob.get("fork") or {}).get("fork_mcgs_sha")),
        "host_checkpoint_sha_before": hashlib.sha1(ckpt_path.read_bytes()).hexdigest(),
        "n_applied": report.get("applied", 0),
        "n_records": len(records),
        "refusals": report.get("refusals", {}),
        "mode": report.get("mode"),
        "in_place": bool(in_place),
    })
    tmp = ledger_path.parent / (ledger_path.name + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ledger_path)

    out = dict(host_ckpt)
    out["mcts"] = merged_mcts      # nothing else in the checkpoint is touched
    target = ckpt_path if in_place else ckpt_path.parent / (_CHECKPOINT_NAME + ".merged")
    tmp = target.parent / (target.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    report["written"] = str(target)
    return {"ok": True, "reason": None, "report": report}


def cmd_merge(a: argparse.Namespace) -> int:
    res = merge(a.quick_run, a.host, in_place=a.in_place, adopt=a.adopt,
                host_idle_s=a.host_idle_s,
                force_reward_scale=a.force_reward_scale,
                host_device=a.host_device, dry_run=a.dry_run)
    rep = res["report"]
    if not res["ok"]:
        print(f"\n[merge] REFUSED ({res['reason']}): {rep.get('detail')}\n"
              f"[merge] Nothing was written.", flush=True)
        return 5
    print(f"\n[merge] mode={rep.get('mode')} records={rep.get('records')} "
          f"applied={rep.get('applied')}")
    if rep.get("refusals"):
        for k, v in sorted(rep["refusals"].items()):
            print(f"[merge]   {k}: {v}")
    if rep.get("mode") == "replay":
        print(f"[merge] nodes {rep['nodes_before']} -> {rep['nodes_after']}, "
              f"total_visits {rep['visits_before']} -> {rep['visits_after']}")
        if rep.get("depth_fixes"):
            print(f"[merge] depth corrections ({len(rep['depth_fixes'])}): "
                  f"{rep['depth_fixes'][:8]}")
        if rep.get("tried_dedup"):
            print(f"[merge] deduped {rep['tried_dedup']} repeated `tried` entries")
        print(f"[merge] tree: {json.dumps(rep.get('stats'))}")
    if res.get("dry_run"):
        print("[merge] --dry_run: nothing written.")
        return 0
    print(f"[merge] wrote {rep.get('written')}")
    if not a.in_place:
        print(f"[merge] Review it, then:  mv {rep.get('written')} "
              f"{Path(a.host).resolve()/_CHECKPOINT_NAME}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _build_parser().parse_args(argv)
    if a.cmd == "run":
        if a.from_kernel and not a.task:
            print("[quick] --from_kernel requires --task.", flush=True)
            return 3
        return cmd_run(a, _explicit_dests(argv))
    return cmd_merge(a)


if __name__ == "__main__":
    raise SystemExit(main())
