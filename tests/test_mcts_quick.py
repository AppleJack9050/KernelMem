"""Mechanics test for the MCTS quick path -- no GPU, no LLM, no subprocess.

Scope, stated up front: this proves that `utils/mcts_quick.py`'s plumbing and its
merge ALGEBRA are correct. It says nothing about whether skipping ncu and the
judge costs anything in kernel quality -- that question needs a live A/B
(`mcts_quick run` vs `main_memory_latest.py --search mcts`, matched rollout budget
on one task, compared on paired gain per wall-clock hour), because the gain a
rollout produces depends on the prompt it was given and the counterfactual prompt
was never sent.

What IS checked here is everything that could silently break the loop or corrupt a
host tree:
  * importing the module drags in neither main_memory_latest nor torch -- the
    tripwire for anyone who hoists that import to module scope and makes every
    spawned bench child pay for it twice per rollout
  * the mechanism draw is never empty and never None -- it no longer keys
    anything, but it labels the edge the PUCT prior reads back off the path, and
    an unnamed edge is a hole in that signal
  * the synthesized strategy dict survives the real prompt renderer
  * a stubbed 25-rollout replay grows a real tree within max_depth
  * the three failures the graph needed guards for -- a merge-tolerance split
    rehoming a kernel under a key the caller did not have, an x-fallback key
    collision between unrelated kernels with the same file stem, and a merge
    closing a cycle that hung select() -- are unreachable rather than refused
  * replaying the journal reproduces the live tree exactly, and replaying it
    twice changes nothing (the widening-budget inflation defence)
  * every merge precondition refuses and writes nothing
  * a legitimate merge chains values on the HOST's ledger and puts exactly one
    visit on each node of the real root->parent->child lineage

Run: python -m tests.test_mcts_quick
"""
from __future__ import annotations

import contextlib
import io
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Imported FIRST and alone, so assertion 1 below is meaningful: anything the
# module drags in lands in sys.modules before the next import line runs.
import utils.mcts_quick as mq
_IMPORT_SNAPSHOT = set(sys.modules)

from utils.mcts import MechanismPrior, MonteCarloTreeSearch
# The same measured gain distribution the search replay uses: 112 parent->child
# edges on vae_block_002, bimodal, 42% regressing past -1%. Reused rather than
# re-tabulated so the two tests cannot drift apart on what "a realistic round"
# looks like.
from tests.test_mcts_replay import _sample_gain


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class _FakeInd:
    """The four attributes _one_rollout touches on a KernelIndividual.

    A local class rather than utils.individual.KernelIndividual: that module is
    cheap today, but importing it invites the torch chain into this test the first
    time someone adds an import to it, and assertion 1 would then start failing
    for a reason that has nothing to do with mcts_quick.
    """

    def __init__(self, code: str, code_path: Path):
        self.code = code
        self.code_path = code_path
        self.metrics: Dict[str, Any] = {}
        self.score: Optional[float] = None


def _stub_rollout(rng: random.Random, code_dir: Path):
    """A model call that always succeeds and writes a distinct kernel file."""
    def _fn(prompt: str, idx: int) -> _FakeInd:
        code = f"// stub kernel {idx} nonce={rng.random():.12f}\nvoid k(){{}}\n"
        path = code_dir / f"kernel_stub_{idx:04d}.py"
        path.write_text(code, encoding="utf-8")
        return _FakeInd(code, path)
    return _fn


def _stub_bench(rng: random.Random, fail_p: float = 0.12):
    """Bench outcomes at the measured compile/run failure rate."""
    def _fn(ind: _FakeInd) -> None:
        if rng.random() < fail_p:
            ind.metrics = {"runnable": False, "error_type": "CompilationError",
                           "message": "stubbed build failure"}
            ind.score = float("-inf")
        else:
            ind.metrics = {"runnable": True, "phase": "quick"}
            ind.score = 1.0 + rng.uniform(-0.1, 0.35)
    return _fn


def _stub_verdict(rng: random.Random, none_p: float = 0.15):
    """A paired verdict, or None -- which is expected, not exceptional."""
    def _fn(base: Path, cand: Path) -> Optional[Dict[str, Any]]:
        if rng.random() < none_p:
            return None            # worker exception / timeout / empty payload
        rel = _sample_gain(rng)
        se = max(0.05, abs(rel) * 0.15)
        return {"rel_pct": rel, "se_pct": se, "t": rel / se, "dof": 4, "reps": 5,
                "resolved": True, "p_one_sided": 0.01, "sigma_equiv": abs(rel) / se,
                "sigma_ok": abs(rel) / se >= 3.0, "method": "stub",
                "beats_margin": rel >= 5.0}
    return _fn


# One task string everywhere, because merge precondition R8 compares the journal's
# `task` with the host checkpoint's and a mismatch here would look like a bug in
# the merge rather than a typo in the fixture.
_TASK = "/tmp/tasks/vae_block_002.py"


def _stamp(run_id: str = "quick_test") -> Dict[str, Any]:
    return {"quick_run_id": run_id, "task": _TASK, "ref_py": _TASK,
            "device": 0, "clock_locked": True, "clock_mhz": 2407,
            "gpu_lock_enabled": True}


def _seed_tree(code_dir: Path, *, prior: Optional[MechanismPrior] = None,
                max_depth: int = 10) -> Tuple[MonteCarloTreeSearch, str]:
    g = MonteCarloTreeSearch(max_depth=max_depth, prior=prior)
    code_dir.mkdir(parents=True, exist_ok=True)
    seed = code_dir / "kernel_seed.py"
    seed.write_text("// seed\nvoid k(){}\n", encoding="utf-8")
    root = g.observe(kernel_name=seed.stem, kernel_path=str(seed), value=1.0,
                     parent_key=None, runnable=True, note="seed").key
    return g, root


def _resolver(registry: Dict[str, Any]):
    def _fn(name: Optional[str], path: Optional[str]) -> Optional[_FakeInd]:
        if not name:
            return None
        if name in registry:
            return registry[name]
        if path and Path(path).exists():
            ind = _FakeInd(Path(path).read_text(encoding="utf-8"), Path(path))
            ind.metrics = {"runnable": True}
            registry[name] = ind
            return ind
        return None
    return _fn


def _drive(tree: MonteCarloTreeSearch, work: Path, *, rounds: int, seed: int,
           count_failures: bool = False, stop_after: Optional[int] = None,
           run_id: str = "quick_test"):
    """Run *rounds* stubbed rollouts through the real loop body.

    Mirrors cmd_run's loop exactly: stop check, _one_rollout, journal append,
    _apply_record, meta update. The only substitutions are the four injected
    side-effecting helpers, so what is exercised here is the same code that runs
    against a real GPU.

    _one_rollout's own progress lines are captured rather than printed: at ~6
    lines per rollout and 45 rollouts across this file they would bury the
    ok/FAIL rows, which are the output that matters. The captured text is
    returned so a failure can still be read.
    """
    code_dir, io_dir = work / "code", work / "evaluation" / "llm_io"
    for d in (code_dir, io_dir):
        d.mkdir(parents=True, exist_ok=True)
    registry: Dict[str, Any] = {}
    rng_r, rng_b, rng_v = random.Random(seed), random.Random(seed + 1), random.Random(seed + 2)
    records: List[Dict[str, Any]] = []
    applied: set = set()
    rep_sha: Dict[str, str] = {}
    outcomes: Dict[str, Dict[str, Any]] = {}
    done = 0
    log = io.StringIO()

    with contextlib.redirect_stdout(log):
        for i in range(rounds):
            if mq._STOP["requested"]:
                break
            if stop_after is not None and i == stop_after:
                # Set from inside the loop, exactly as a signal would land: the
                # in-flight rollout still completes, the next iteration stops.
                mq._STOP["requested"] = True
            rec = mq._one_rollout(
                tree, rollout_idx=i, rng=random.Random(seed * 1000 + i),
                policy="sample", gpu_name="RTX 5090",
                registry=registry, dirs={"code": code_dir, "eval": work, "io": io_dir},
                count_failures=count_failures, stamp=_stamp(run_id),
                resolve_fn=_resolver(registry),
                prompt_fn=lambda p, g, s: f"PROMPT for {s['method_name']}",
                rollout_fn=_stub_rollout(rng_r, code_dir),
                bench_fn=_stub_bench(rng_b),
                verdict_fn=_stub_verdict(rng_v))
            if rec is None:
                continue
            records.append(rec)
            landed, why = mq._apply_record(tree, rec, applied, rep_sha)
            outcomes[rec["obs_id"]] = {"landed_key_local": landed,
                                       "apply_reason": why}
            done = i + 1
    return dict(tree=tree, records=records, applied=applied, rep_sha=rep_sha,
                outcomes=outcomes, rollouts_done=done, log=log.getvalue())


def _replay(records: List[Dict[str, Any]], code_dir: Path,
            **kw) -> MonteCarloTreeSearch:
    """Rebuild a tree from the seed by folding the journal in, in file order."""
    g, _root = _seed_tree(code_dir, **kw)
    applied: set = set()
    rep_sha: Dict[str, str] = {}
    for rec in records:
        mq._apply_record(g, rec, applied, rep_sha)
    return g


def _node_fields(g: MonteCarloTreeSearch, k: str) -> Tuple:
    n = g.nodes[k]
    return (n.N, round(n.W, 12), round(n.M, 12), n.failures, n.depth, n.kernel,
            round(n.value, 12))


# ---------------------------------------------------------------------------
# merge fixtures
# ---------------------------------------------------------------------------
def _write_host(root: Path, tree: Optional[MonteCarloTreeSearch], *,
                task: str, best_path: Optional[str],
                extra: Optional[Dict[str, Any]] = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "version": 1, "task": task, "next_round": 7, "total_rounds": 20,
        "base": None,
        "best": ({"code_path": best_path, "eval_path": None, "score": 1.0}
                 if best_path else None),
        "current": None, "repair_chain": None,
        "base_score": 1.0, "best_score": 1.0, "optimization_tree": {},
        "scores": [], "err_flags": [], "last_score_for_curve": 0.0,
        "rounds_since_improvement": 0, "structural_debt": None, "stop_reason": None,
        "mcts": tree.to_dict() if tree is not None else None,
        "opt_history_files": {}, "next_individual_id": 3,
        "timestamp": "2026-08-14T09:00:00",
    }
    ckpt.update(extra or {})
    p = root / "checkpoint.json"
    p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    return p


def _write_quick(qdir: Path, tree: MonteCarloTreeSearch,
                 records: List[Dict[str, Any]], *, task: str,
                 fork: Dict[str, Any],
                 outcomes: Optional[Dict[str, Any]] = None,
                 clock: Optional[Dict[str, Any]] = None) -> Path:
    qdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "quick_version": 1, "kind": "mcts_quick_tree", "run_id": qdir.name,
        "created": "2026-08-14T10:00:00", "task": task, "seed_mode": "from_run",
        "seed_source_path": None, "fork": fork,
        "bench": {"device": 0, "warmup": 25, "repeat": 100, "tol": 0.01},
        "verdict": {"base_margin": 0.05, "base_reps": 5, "base_max_reps": 8,
                    "base_sigma": 3.0},
        "llm": {"rollout_model": "claude-sonnet-5", "rollout_effort": "high",
                "model_name": "claude-opus-5", "temperature": 1, "top_p": 1.0},
        "policy": {"mechanism_policy": "sample", "mechanism_seed": 1,
                   "count_failures": False},
        "clock": clock or {"locked": True, "target_gpu_mhz": 2407,
                           "gpu_name": "NVIDIA GeForce RTX 5090"},
        "gpu_lock": "/tmp/kernelmem.lock", "rollouts_done": len(records),
        "applied_obs": [], "rep_sha": {}, "outcomes": outcomes or {},
    }
    mq._write_tree(qdir / "tree.json", tree, meta)
    jl = qdir / "journal.jsonl"
    jl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return qdir / "tree.json"


def _fork_meta(host_root: Path, host_ckpt: Path, blob: Dict[str, Any],
               *, forked: bool = True) -> Dict[str, Any]:
    import hashlib
    return {"host_task_root": str(host_root),
            "host_checkpoint_sha": hashlib.sha1(host_ckpt.read_bytes()).hexdigest(),
            "fork_mcts_sha": hashlib.sha1(
                json.dumps(blob, sort_keys=True).encode()).hexdigest(),
            "forked_from_mcts": forked, "fork_root": blob.get("root"),
            "fork_node_keys": sorted(blob.get("nodes") or {})}


# ---------------------------------------------------------------------------
def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mcts_quick_test_"))
    try:
        _main(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _main(tmp: Path) -> None:
    print("[quick] import hygiene")
    # The whole reason main_memory_latest is imported lazily. If this fails, every
    # spawned bench child and every spawned verdict child -- two per rollout --
    # started re-importing torch and matplotlib through __mp_main__ for nothing.
    _check("main_memory_latest" not in _IMPORT_SNAPSHOT,
           "importing utils.mcts_quick does NOT import main_memory_latest")
    _check("torch" not in _IMPORT_SNAPSHOT,
           "importing utils.mcts_quick does NOT import torch")
    src = Path(mq.os.environ.get("SOLBENCH_SRC", ""))
    _check(str(src) and src.is_dir(),
           f"SOLBENCH_SRC is set to an existing directory ({src}) -- the ref_*.py "
           f"default is /home/elek/... and dies inside the spawned bench worker")

    print("\n[quick] the mechanism draw can never be empty")
    g0, _ = _seed_tree(tmp / "m0")
    prior_empty = MechanismPrior()
    prior_real = MechanismPrior.fit(
        [("good", 5.0)] * 4 + [("bad", -4.0)] * 4 + [("meh", 0.2)] * 4,
        min_support=2)
    for label, prior in (("no prior", None), ("empty table", prior_empty),
                         ("fitted prior", prior_real)):
        g0.prior = prior
        cands = mq._mechanism_candidates(g0)
        _check(len(cands) > 0, f"{label}: candidate list is non-empty ({len(cands)})")
        sel = g0.select()
        rng = random.Random(11)
        draws = [mq._choose_mechanism(g0, sel, rng, "sample") for _ in range(500)]
        _check(all(isinstance(d, str) and d.strip() for d in draws),
               f"{label}: 500 draws, none empty or None -- the x:m-empty bucket "
               f"(37 kernels / 457% spread when it was shared) is unreachable")
    g0.prior = None

    print("\n[quick] a mechanism already tried from this node is damped")
    gd, rootd = _seed_tree(tmp / "m1")
    gd.prior = MechanismPrior.fit([("aa", 1.0)] * 3 + [("bb", 1.0)] * 3,
                                  min_support=2)
    # Both advantages are 0.0 (they equal the global mean), so prior.weights is
    # uniform and the only asymmetry left is this file's own sibling damping.
    gd.nodes[rootd].tried.append({"mechanism": "aa", "child": "k", "note": "",
                                  "value": 1.0, "runnable": True})
    seld = gd.select()
    rngd = random.Random(5)
    picks = [mq._choose_mechanism(gd, seld, rngd, "sample", candidates=["aa", "bb"])
             for _ in range(500)]
    n_aa, n_bb = picks.count("aa"), picks.count("bb")
    _check(n_aa < 0.8 * n_bb,
           f"the already-tried mechanism is drawn less often ({n_aa} vs {n_bb}; "
           f"repeat_penalty is {gd.prior.repeat_penalty}, so ~1:4 is expected)")
    _check(n_aa > 0,
           "...but is not forbidden -- a wider tile can become legal after a "
           "different change made it fit")

    print("\n[quick] the strategy dict is judge-free and renderer-safe")
    gs, _ = _seed_tree(tmp / "m2")
    sels = gs.select()
    strat = mq._build_strategy(gs, sels, "l2_cache_blocking")
    _check(bool(strat.get("method_name")), "method_name is present and non-empty")
    _check(set(strat) <= set(mq._STRATEGY_KEYS),
           "every emitted key is one _format_problem actually reads "
           f"(got {sorted(strat)})")
    _check("expected_metric_change" not in strat,
           "expected_metric_change is omitted -- it exists to be checked against a "
           "NEXT ncu profile this path never takes")
    _check("structural_rewrite" not in strat,
           "structural_rewrite is omitted -- _format_problem drops it and the "
           "ratchet branch that reads it is inert under MCTS")
    _check("l2_cache_blocking" in strat["modification_plan"]
           and strat["modification_plan"].strip().startswith("1."),
           "modification_plan is a numbered checklist naming the chosen mechanism")
    _check("No ncu profile" in strat["bottleneck"],
           "bottleneck says plainly that nothing was profiled, rather than "
           "inventing a metric the template will read as evidence")
    try:
        from prompts.optimization_memory_latest import _format_problem
        rendered = json.loads(_format_problem(strat))
        _check(rendered.get("method_name") == "l2_cache_blocking",
               "the dict survives the real prompt renderer with method_name intact")
        _check(set(rendered) == set(strat),
               "and the renderer drops none of the keys we emit")
    except ImportError as exc:
        print(f"  skipped: prompts package unavailable ({exc.__class__.__name__})")

    print("\n[quick] 25 stubbed rollouts through the real loop body")
    w1 = tmp / "run1"
    g1, _ = _seed_tree(w1 / "code")
    r1 = _drive(g1, w1, rounds=25, seed=7)
    st = g1.stats()
    print(f"  tree: {json.dumps(st)}")
    _check(st["nodes"] > 1, "the tree grew beyond the root")
    parents = [r["parent_key"] for r in r1["records"]]
    switches = sum(1 for a, b in zip(parents, parents[1:]) if a != b)
    _check(switches > 0,
           f"selection revisited non-incumbent nodes ({switches} parent switches) "
           f"-- the ratchet would score 0 here")
    _check(max(n.depth for n in g1.nodes.values()) <= 10, "max_depth was honoured")
    _check(all(n.value > 0 and n.value == n.value
               for n in g1.nodes.values()),
           "every chained value stayed finite and positive")
    _check(len(r1["records"]) == 25, "exactly one journal record per rollout")
    n_applied = sum(1 for o in r1["outcomes"].values()
                    if o["apply_reason"] == "applied")
    _check(g1.total_visits == n_applied,
           f"total_visits ({g1.total_visits}) == records that landed ({n_applied}) "
           f"-- exactly one backup per record, never one per path node")
    _check(all(r["basis"] in ("paired", "unmeasurable",
                             "blocked (drift-contaminated; no paired verdict)")
               for r in r1["records"]),
           "every edge carries the basis it was measured on, forever")

    print("\n[quick] the graph's three refusal paths are gone because they cannot occur")
    # Each of these was a real defence with a real failure behind it, and each is
    # now unreachable by construction rather than by a guard. Asserted positively:
    # the property that makes the refusal moot, not the absence of the refusal.
    gsp, rootsp = _seed_tree(_mk(tmp / "ident" / "code"))
    base = {"ts": "t", "rollout_idx": 0, "origin": "quick", "parent_key": rootsp,
            "count_failures": False, "runnable": True, "basis": "paired",
            "kernel_path": "/tmp/a.py", "mechanism": "M"}
    # (1) The MERGE-TOLERANCE SPLIT. Two children of one parent 60% apart used to
    #     pool into one node until the value guard rehomed the second under a
    #     "/s" key the caller did not know about -- and the caller then backed up
    #     the key it had passed in, so the child got no visit and the next line
    #     raised KeyError. Nothing pools now.
    a = dict(base, obs_id="A", kernel_name="ka", code_sha="sha_a", rel_pct=0.0)
    b = dict(base, obs_id="B", kernel_name="kb", code_sha="sha_b", rel_pct=60.0,
             rollout_idx=1)
    ap, rp = set(), {}
    k1, why1 = mq._apply_record(gsp, a, ap, rp)
    k2, why2 = mq._apply_record(gsp, b, ap, rp)
    _check(why1 == "applied" and why2 == "applied", "both records landed")
    _check(k1 != k2, f"two children of one parent are two nodes ({k1}, {k2})")
    _check("/s" not in k2, "no split key: there was no merge for a guard to refuse")
    _check(gsp.nodes[k1].N == 1 and gsp.nodes[k2].N == 1,
           "and backup reached BOTH -- the id came back from observe(), so the "
           "caller cannot back up a key the node does not have")
    _check(gsp.nodes[k1].parent == rootsp and gsp.nodes[k2].parent == rootsp,
           "both hang off the parent the record named")

    # (2) The X-FALLBACK COLLISION. An unnamed change hashed to a key derived from
    #     the kernel's FILE STEM, and save_kernel_code stamps at one-second
    #     resolution -- so two runs could name unrelated kernels identically and
    #     silently pool them. Ids are minted, not derived from anything.
    gx, rootx = _seed_tree(_mk(tmp / "xcoll" / "code"))
    xa = {"obs_id": "X1", "rollout_idx": 0, "origin": "q", "parent_key": rootx,
          "count_failures": False, "runnable": True, "basis": "paired",
          "kernel_name": "round003_k", "kernel_path": "/tmp/x1.py",
          "code_sha": "aaa", "rel_pct": 1.0, "mechanism": None}
    xb = dict(xa, obs_id="X2", code_sha="bbb", rel_pct=2.0, kernel_path="/tmp/x2.py")
    apx, rpx = set(), {}
    kx1, whyx1 = mq._apply_record(gx, xa, apx, rpx)
    kx2, whyx2 = mq._apply_record(gx, xb, apx, rpx)
    _check(whyx1 == "applied" and whyx2 == "applied",
           "two unnamed changes with the same file stem both land")
    _check(kx1 != kx2,
           f"under different ids ({kx1}, {kx2}) -- unrelated code cannot collide")
    _check(gx.nodes[kx1].kernel == gx.nodes[kx2].kernel == "round003_k",
           "even though they really do carry the same kernel name")

    # (3) The CYCLE-CLOSING EDGE. A merge could give an ancestor a child that was
    #     its own ancestor, and select() -- a `while True` with no visited set --
    #     hung rather than raised. observe() only ever attaches a FRESH node.
    gc, rootc = _seed_tree(_mk(tmp / "cyc" / "code"))
    chain = [rootc]
    for i, m in enumerate(("m1", "m2", "m3")):
        r = dict(base, obs_id=f"C{i}", kernel_name=f"kc{i}", code_sha=f"c{i}",
                 rel_pct=1.0, rollout_idx=i, parent_key=chain[-1], mechanism=m)
        landed, why = mq._apply_record(gc, r, set(), {})
        _check(why == "applied", f"chain step {m} landed")
        chain.append(landed)
    # Re-apply the FIRST mechanism from the deepest node: on the graph this
    # recomputed the ancestor's key and closed a loop.
    r = dict(base, obs_id="Cx", kernel_name="kcx", code_sha="cx", rel_pct=1.0,
             rollout_idx=9, parent_key=chain[-1], mechanism="m1")
    landed, why = mq._apply_record(gc, r, set(), {})
    _check(why == "applied" and landed not in chain,
           f"re-applying a path mechanism makes a NEW node ({landed}), not an edge "
           f"back into the chain")
    _check(gc.path_to(landed) == chain + [landed],
           "and its lineage is the chain plus itself -- strictly deeper, never a loop")
    # N=1 gives every node a widening budget of ceil(1*1**0.5) = 1, which its one
    # child already satisfies, so the descent walks the spine instead of bailing at
    # the root. Without this the walk never leaves the root and "it terminates" is
    # asserted on a walk that never happened -- the budget fires first, because
    # every node here has fewer children than a higher N would demand.
    for nn in gc.nodes.values():
        nn.N, nn.W, nn.M = 1, 0.5, 0.6
    walked = 0
    for _ in range(8):
        sel = gc.select()
        _check(len(sel.path) == len(set(sel.path)),
               f"select() returns a simple path ({sel.path})")
        _check(sel.path == gc.path_to(sel.node.key),
               "and that path IS the node's lineage")
        walked = max(walked, len(sel.path))
        gc.backup(list(sel.path), 0.5)
    _check(walked > 1, f"the descent really left the root (deepest walk {walked})")

    print("\n[quick] crash recovery: replaying the journal reproduces the tree")
    g_replay = _replay(r1["records"], _mk(tmp / "replay" / "code"))
    _check(set(g_replay.nodes) == set(g1.nodes),
           f"identical node key sets ({len(g_replay.nodes)} nodes)")
    _check(all(_node_fields(g_replay, k) == _node_fields(g1, k) for k in g1.nodes),
           "identical N, W, M, failures, depth, kernel and value on every node")
    _check(g_replay.total_visits == g1.total_visits,
           f"identical total_visits ({g_replay.total_visits})")
    _check(all(g_replay.nodes[k].parent == g1.nodes[k].parent for k in g1.nodes),
           "and an identical parent pointer on every node")

    print("\n[quick] replaying the SAME journal twice is a no-op")
    applied2: set = set()
    rep2: Dict[str, str] = {}
    g_twice = _replay(r1["records"], _mk(tmp / "twice" / "code"))
    # Rebuild applied_obs the way a resume would, then feed the journal in again.
    for rec in r1["records"]:
        applied2.add(rec["obs_id"])
    snapshot = {k: _node_fields(g_twice, k) for k in g_twice.nodes}
    visits = g_twice.total_visits
    for rec in r1["records"]:
        mq._apply_record(g_twice, rec, applied2, rep2)
    _check(g_twice.total_visits == visits,
           f"total_visits unchanged ({visits}) on a double apply")
    _check(all(_node_fields(g_twice, k) == snapshot[k] for k in snapshot),
           "and every N is unchanged -- a doubled N leaves q() identical while "
           "ceil(k*N**alpha) hands the node a child it never paid for")

    print("\n[quick] atomic write / read round trip")
    gp = tmp / "rt"
    gp.mkdir()
    meta = {"quick_version": 1, "kind": "mcts_quick_tree", "run_id": "rt",
            "task": "/tmp/task.py", "rollouts_done": 25,
            "applied_obs": sorted(r1["applied"]), "rep_sha": r1["rep_sha"]}
    mq._write_tree(gp / "tree.json", g1, meta)
    g_rt, meta_rt, applied_rt = mq._read_tree(gp / "tree.json")
    _check(not list(gp.glob("*.tmp")), "no .tmp file is left behind")
    _check(g_rt.root == g1.root and set(g_rt.nodes) == set(g1.nodes),
           "root and node set survive")
    _check(all(abs(g_rt.nodes[k].q() - g1.nodes[k].q()) < 1e-12 for k in g1.nodes),
           "every Q is reproduced exactly")
    _check(applied_rt == r1["applied"], "applied_obs survives")
    _check(g_rt._next_id >= g1._next_id,
           "the id counter does not rewind, so a resumed run cannot reuse an id")
    _check(g_rt.select().node.key == g1.select().node.key,
           "the restored tree selects the same node -- a resume continues, it "
           "does not restart")

    print("\n[quick] merge refuses, and writes nothing, on every precondition")
    _merge_refusals(tmp)

    print("\n[quick] merge on a legitimate fork")
    _merge_happy(tmp)

    print("\n[quick] merge after the host has moved on")
    _merge_host_advanced(tmp)

    print("\n[quick] merge translates the quick run's local node ids into host ids")
    _merge_id_translation(tmp)

    print("\n[quick] graceful stop at a rollout boundary")
    mq._STOP["requested"] = False
    try:
        w2 = tmp / "stop"
        g2, _ = _seed_tree(_mk(w2 / "code"))
        r2 = _drive(g2, w2, rounds=25, seed=3, stop_after=3, run_id="quick_stop")
        _check(len(r2["records"]) == 4,
               "the in-flight rollout finished, then the loop stopped "
               f"({len(r2['records'])} records: rollouts 0-3)")
        _check(r2["rollouts_done"] == 4, "rollouts_done matches the journal")
        gp2 = tmp / "stopgraph"
        gp2.mkdir()
        mq._write_tree(gp2 / "tree.json", g2,
                        {"quick_version": 1, "kind": "mcts_quick_tree",
                         "run_id": "s", "rollouts_done": 4,
                         "applied_obs": sorted(r2["applied"])})
        g_disk, _m, _a = mq._read_tree(gp2 / "tree.json")
        _check(all(_node_fields(g_disk, k) == _node_fields(g2, k) for k in g2.nodes),
               "the tree on disk equals the in-memory tree at exit")
    finally:
        mq._STOP["requested"] = False

    print("\n[quick] all checks passed")
    print("NOTE: this proves the quick path's mechanics and its merge algebra, not "
          "that skipping ncu and the judge costs nothing. That question needs a "
          "live A/B -- `mcts_quick run` vs `main_memory_latest.py --search mcts`, "
          "matched rollout budget on one task, compared on paired gain per "
          "wall-clock hour.")


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
def _merge_refusals(tmp: Path) -> None:
    """Each refusal built one at a time, so a passing case cannot mask a gate."""
    base = tmp / "refuse"
    code = _mk(base / "hostcode")
    seed = code / "kernel_seed.py"
    seed.write_text("// seed\nvoid k(){}\n", encoding="utf-8")

    def _host_tree() -> MonteCarloTreeSearch:
        g = MonteCarloTreeSearch()
        g.observe(key="root", kernel_name=seed.stem, kernel_path=str(seed), value=1.0)
        return g

    cases = []

    # R2 (different state_key_mode) is gone with the state key: there is no
    # abstraction left for a host and a fork to disagree about.

    # R3 -- different root key
    H = _host_tree()
    Q = MonteCarloTreeSearch.from_dict(H.to_dict())
    Q.root = "m:deadbeefdeadbeef"
    Q.nodes[Q.root] = Q.nodes.pop(H.root)
    Q.nodes[Q.root].key = Q.root
    cases.append(("R3-root-key", H, Q, {}, {}))

    # R4 -- same root key, different source
    H = _host_tree()
    other = code / "kernel_other.py"
    other.write_text("// something else entirely\nvoid z(){}\n", encoding="utf-8")
    Q = MonteCarloTreeSearch.from_dict(H.to_dict())
    Q.nodes[Q.root].kernel_path = str(other)
    cases.append(("R4-root-source", H, Q, {}, {}))

    # R5 -- the roots' measured values disagree by 5%
    H = _host_tree()
    Q = MonteCarloTreeSearch.from_dict(H.to_dict())
    Q.nodes[Q.root].value = 1.05
    cases.append(("R5-root-value", H, Q, {}, {}))

    # R6 -- reward_scale mismatch without --force_reward_scale
    H = _host_tree()
    Q = MonteCarloTreeSearch.from_dict(H.to_dict())
    Q.reward_scale = 5.0
    cases.append(("R6-reward-scale", H, Q, {}, {}))

    # R7 -- the quick run was measured unlocked
    H = _host_tree()
    Q = MonteCarloTreeSearch.from_dict(H.to_dict())
    cases.append(("R7-clock", H, Q,
                  {"clock": {"locked": False, "reason": "opted out"}}, {}))

    # R10 -- bootstrapped, not forked
    H = _host_tree()
    Q = MonteCarloTreeSearch.from_dict(H.to_dict())
    cases.append(("R10-not-a-fork", H, Q, {"forked": False}, {}))

    for i, (want, H, Q, quick_kw, merge_kw) in enumerate(cases):
        root = _mk(base / f"case{i}")
        hroot, qdir = _mk(root / "host"), _mk(root / "quick")
        ck = _write_host(hroot, H, task=_TASK,
                         best_path=str(seed))
        fork = _fork_meta(hroot, ck, H.to_dict(),
                          forked=quick_kw.pop("forked", True))
        _write_quick(qdir, Q, [], task=_TASK, fork=fork,
                     **quick_kw)
        res = mq.merge(qdir, hroot, host_idle_s=0, **merge_kw)
        _check(res["reason"] == want,
               f"{want}: refused with the right code (got {res['reason']})")
        _check(not (hroot / "checkpoint.json.merged").exists()
               and not (hroot / "mcts_quick_merged.json").exists(),
               f"{want}: nothing was written")

    # R9 -- the host is live (checkpoint.json touched just now)
    root = _mk(base / "live")
    hroot, qdir = _mk(root / "host"), _mk(root / "quick")
    H = _host_tree()
    ck = _write_host(hroot, H, task=_TASK,
                     best_path=str(seed))
    _write_quick(qdir, MonteCarloTreeSearch.from_dict(H.to_dict()), [],
                 task=_TASK,
                 fork=_fork_meta(hroot, ck, H.to_dict()))
    res = mq.merge(qdir, hroot, host_idle_s=900)
    _check(res["reason"] == "R9-host-live",
           "R9-host-live: a checkpoint.json written seconds ago blocks the merge "
           "(median normal round is 570s, so 900s clears one with margin)")
    _check(not (hroot / "checkpoint.json.merged").exists(),
           "R9-host-live: nothing was written")

    # R1 -- the host never ran a search
    root = _mk(base / "nosearch")
    hroot, qdir = _mk(root / "host"), _mk(root / "quick")
    ck = _write_host(hroot, None, task=_TASK,
                     best_path=str(seed))
    H = _host_tree()
    _write_quick(qdir, H, [], task=_TASK,
                 fork=_fork_meta(hroot, ck, H.to_dict()))
    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["reason"] == "R1-host-has-no-tree",
           "R1-host-has-no-tree: refused without --adopt (measured 2026-08-14: 0 "
           "of 8 checkpoint.json files under run/ carry a search blob)")

    # R8 -- a different workload
    root = _mk(base / "task")
    hroot, qdir = _mk(root / "host"), _mk(root / "quick")
    H = _host_tree()
    ck = _write_host(hroot, H, task=_TASK,
                     best_path=str(seed))
    rec = {"obs_id": "z", "rollout_idx": 0, "origin": "q", "task": "/tmp/tasks/other_task.py",
           "device": 0, "parent_key": H.root, "mechanism": "m", "kernel_name": "k",
           "kernel_path": "/tmp/k.py", "code_sha": "s", "rel_pct": 1.0,
           "runnable": True, "basis": "paired", "count_failures": False}
    _write_quick(qdir, MonteCarloTreeSearch.from_dict(H.to_dict()), [rec],
                 task=_TASK,
                 fork=_fork_meta(hroot, ck, H.to_dict()))
    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["reason"] == "R8-task",
           "R8-task: a journal measured on another workload is refused")


def _merge_happy(tmp: Path) -> None:
    base = _mk(tmp / "happy")
    hdir = _mk(base / "hostwork")
    H, _root = _seed_tree(_mk(hdir / "code"))
    _drive(H, hdir, rounds=10, seed=13, run_id="host_sim")
    _check(len(H.nodes) >= 4, f"the host tree has {len(H.nodes)} nodes to merge into")

    hroot = _mk(base / "host")
    seed_path = H.nodes[H.root].kernel_path
    ck = _write_host(hroot, H, task=_TASK,
                     best_path=seed_path)
    host_before = json.loads(ck.read_text(encoding="utf-8"))
    blob_before = json.loads(json.dumps(H.to_dict()))
    prior_before = json.dumps(H.prior.to_dict() if H.prior else None, sort_keys=True)

    Q = MonteCarloTreeSearch.from_dict(json.loads(json.dumps(H.to_dict())))
    qwork = _mk(base / "quickwork")
    _mk(qwork / "code")
    rq = _drive(Q, qwork, rounds=10, seed=29, run_id="quick_happy")
    qdir = _mk(base / "quick")
    _write_quick(qdir, Q, rq["records"], task=_TASK,
                 fork=_fork_meta(hroot, ck, blob_before),
                 outcomes=rq["outcomes"])

    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["ok"], f"the merge succeeded ({res.get('reason')})")
    rep = res["report"]
    print(f"  report: applied={rep['applied']}/{rep['records']} "
          f"refusals={rep['refusals']} nodes {rep['nodes_before']}->"
          f"{rep['nodes_after']} visits {rep['visits_before']}->{rep['visits_after']}")

    merged_path = hroot / "checkpoint.json.merged"
    _check(merged_path.exists() and not res["report"].get("in_place"),
           "it wrote checkpoint.json.merged, leaving checkpoint.json alone")
    out = json.loads(merged_path.read_text(encoding="utf-8"))
    _check(all(out[k] == host_before[k] for k in host_before if k != "mcts"),
           "every non-mcts key of the checkpoint is byte-identical -- not base, "
           "not best, not base_score, not next_round")
    M = MonteCarloTreeSearch.from_dict(out["mcts"])
    _check(M.total_visits == rep["visits_before"] + rep["applied"],
           f"total_visits advanced by the records that ACTUALLY landed "
           f"({rep['applied']}), not by the quick tree's total_visits "
           f"({Q.total_visits})")
    _check(not mq._orphans(M), "every node is reachable from the root by BFS")
    depths = {M.root: 0}
    from collections import deque as _dq
    dq = _dq([M.root])
    while dq:
        cur = dq.popleft()
        for k in M.nodes[cur].children:
            if k not in depths:
                depths[k] = depths[cur] + 1
                dq.append(k)
    _check(all(M.nodes[k].depth == d for k, d in depths.items()),
           "every node's depth equals its BFS depth after the post-pass")
    _check(json.dumps(M.prior.to_dict() if M.prior else None,
                      sort_keys=True) == prior_before,
           "the host's prior was not touched")
    _check((hroot / "mcts_quick_merged.json").exists(),
           "the obs_id ledger was written beside the checkpoint (its own file: "
           "to_dict rebuilds the search blob from scratch and _save_checkpoint "
           "writes a fixed schema, so neither would survive the host's next round)")

    # The load-bearing arithmetic: values are recomputed on the HOST's chain, not
    # imported. Only counted where the parent's representative did NOT move after
    # the record landed -- once a later record promotes a better rep, the parent's
    # rep_value at apply time is no longer readable from the final tree, so those
    # nodes are unverifiable from here rather than wrong. The crisp,
    # fully-determined version of this check is in _merge_host_advanced.
    checked = 0
    for rec in rq["records"]:
        if not (rec.get("runnable") and rec.get("kernel_name")):
            continue
        parent = M.nodes.get(rec["parent_key"])
        if parent is None:
            continue
        want = parent.value * (1.0 + rec["rel_pct"] / 100.0)
        for node in M.nodes.values():
            if node.kernel == rec["kernel_name"] and abs(node.value - want) < 1e-12:
                checked += 1
                break
    _check(checked > 0,
           f"{checked} merged node(s) carry host_parent.value * (1 + rel/100) "
           f"exactly -- the GAIN was replayed onto the host's chain, the value was "
           f"not imported from a chain anchored at another seed")

    # Idempotence at the merge level, which is what the ledger is for.
    res2 = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res2["ok"] and res2["report"]["applied"] == 0,
           f"re-merging the same journal applies 0 records "
           f"(all {res2['report']['refusals'].get('duplicate', 0)} are duplicates) "
           f"-- N and W cannot be doubled")


def _merge_id_translation(tmp: Path) -> None:
    """A CHAINED quick record must not be grafted onto a host node by raw id.

    Node ids are minted per tree, so after a fork both trees mint from the same
    counter independently and the same id means different kernels in each. The
    graph era had no such hazard -- a key was a content hash of the path's
    mechanism multiset, identical in both trees -- so nothing translated, and the
    conversion to minted ids reintroduced the need without the machinery.

    The shape that breaks it: the quick run chains rollout 1 off rollout 0 (so
    its parent_key is an id the quick run minted itself), and the host then runs
    a round of its own and mints THE SAME id for an unrelated kernel. Merging by
    raw id hangs the chain off that kernel -- `parent-not-in-target` cannot fire
    because the id exists, the chained value is computed off the wrong parent,
    and the reward is backed up the wrong lineage. merge() advertises exactly
    this case as supported, so it must be right rather than merely refused.
    """
    base = _mk(tmp / "idmap")
    code = _mk(base / "code")
    for nm in ("seed", "hostA", "hostB", "q0", "q1"):
        (code / f"{nm}.py").write_text(f"// {nm}\nvoid k(){{}}\n", encoding="utf-8")

    H = MonteCarloTreeSearch()
    root = H.observe(kernel_name="seed", kernel_path=str(code / "seed.py"), value=1.00).key
    hostA = H.observe(kernel_name="hostA", kernel_path=str(code / "hostA.py"),
                      value=1.05, parent_key=root, mechanism="ma").key
    H.backup([root], 0.5)
    H.backup([root, hostA], 0.6)
    hroot = _mk(base / "host")
    ck = _write_host(hroot, H, task=_TASK, best_path=str(code / "seed.py"))
    fork_blob = json.loads(json.dumps(H.to_dict()))

    # The quick run forks, then CHAINS: q1's parent is q0, which the quick run
    # minted itself and the host has never heard of.
    Q = MonteCarloTreeSearch.from_dict(json.loads(json.dumps(fork_blob)))
    q0 = Q.observe(kernel_name="q0", kernel_path=str(code / "q0.py"),
                   value=1.05 * 1.02, parent_key=hostA, mechanism="mq0").key
    q1 = Q.observe(kernel_name="q1", kernel_path=str(code / "q1.py"),
                   value=1.05 * 1.02 * 1.03, parent_key=q0, mechanism="mq1").key

    recs, outcomes = [], {}
    for i, (nm, parent, parent_rep, rel, local) in enumerate(
            [("q0", hostA, "hostA", 2.0, q0), ("q1", q0, "q0", 3.0, q1)]):
        recs.append({"obs_id": f"o{i}", "ts": "t", "rollout_idx": i, "origin": "quick",
                     "task": _TASK, "ref_py": _TASK, "device": 0,
                     "parent_key": parent, "parent_rep": parent_rep,
                     "mechanism": f"m_{nm}", "kernel_name": nm,
                     "kernel_path": str(code / f"{nm}.py"), "code_sha": nm,
                     "rel_pct": rel, "runnable": True, "basis": "paired",
                     "count_failures": False})
        outcomes[f"o{i}"] = {"landed_key_local": local, "apply_reason": "applied"}

    # The host advances AFTER the fork and mints the id the quick run gave q0.
    hostB = H.observe(kernel_name="hostB", kernel_path=str(code / "hostB.py"),
                      value=0.90, parent_key=root, mechanism="mb").key
    H.backup([root, hostB], 0.1)
    _write_host(hroot, H, task=_TASK, best_path=str(code / "seed.py"))
    _check(hostB == q0,
           f"the fixture really does collide: host minted {hostB}, quick minted {q0}")
    hostB_N_before = H.nodes[hostB].N

    qdir = _mk(base / "quick")
    _write_quick(qdir, Q, recs, task=_TASK,
                 fork=_fork_meta(hroot, ck, fork_blob), outcomes=outcomes)
    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["ok"] and res["report"]["applied"] == 2,
           f"both chained records land ({res.get('reason')})")
    M = MonteCarloTreeSearch.from_dict(
        json.loads((hroot / "checkpoint.json.merged").read_text(encoding="utf-8"))["mcts"])
    by_kernel = {n.kernel: k for k, n in M.nodes.items()}
    _check("q0" in by_kernel and "q1" in by_kernel, "both kernels are in the host")
    lineage = [M.nodes[k].kernel for k in M.path_to(by_kernel["q1"])]
    _check(lineage == ["seed", "hostA", "q0", "q1"],
           f"q1's lineage follows the quick run's chain, not the colliding id "
           f"(got {lineage})")
    _check(abs(M.nodes[by_kernel["q1"]].value - 1.05 * 1.02 * 1.03) < 1e-12,
           f"and its chained value is 1.1031, not hostB's 0.90*1.03=0.9270 "
           f"(got {M.nodes[by_kernel['q1']].value:.4f})")
    _check(M.nodes[hostB].N == hostB_N_before,
           f"the colliding host node got NO visit ({M.nodes[hostB].N}) -- the reward "
           f"was backed up the real lineage")

    print("\n[quick] a graft the map cannot justify is refused, never guessed at")
    t = MonteCarloTreeSearch()
    r0 = t.observe(kernel_name="seed", kernel_path="/s", value=1.0).key
    ka = t.observe(kernel_name="hostA", kernel_path="/a", value=1.05,
                   parent_key=r0, mechanism="ma").key
    kb = t.observe(kernel_name="hostB", kernel_path="/b", value=0.9,
                   parent_key=r0, mechanism="mb").key
    rec = {"obs_id": "z", "rollout_idx": 0, "origin": "q", "parent_key": "L9",
           "parent_rep": "hostA", "count_failures": False, "runnable": True,
           "basis": "paired", "kernel_name": "child", "kernel_path": "/c",
           "code_sha": "c", "rel_pct": 1.0, "mechanism": "mc"}
    _check(mq._apply_record(t, rec, set(), {}, id_map={})[1] == "parent-not-in-target",
           "an unmapped local id is refused")
    _check(mq._apply_record(t, rec, set(), {}, id_map={"L9": kb})[1]
           == "parent-identity-mismatch",
           "a map pointing at the wrong kernel is caught by the journal's parent_rep")
    _check(mq._apply_record(t, rec, set(), {}, id_map={"L9": ka})[1] == "applied",
           "and the right map folds normally")
    rec2 = dict(rec, obs_id="y", parent_key=ka)
    _check(mq._apply_record(t, rec2, set(), {})[1] == "applied",
           "id_map=None keeps the live loop's behaviour: the ids ARE this tree's")


def _merge_host_advanced(tmp: Path) -> None:
    """The host keeps working after the fork; the record still lands correctly.

    The graph version of this test was about route SHORTENING: the host could
    re-observe the fork's parent under a shallower parent (a legitimate
    transposition), giving it a strictly shorter route, and the merge then had to
    back up along the host's CURRENT shortest route rather than the journal's
    recorded one. On a tree a node has exactly one route and it never changes, so
    that hazard is gone.

    What can still go wrong is what is checked here. Between the fork and the
    merge the host adds children and accrues visits, so the parent's statistics
    move under the fork's feet. The record must still find its parent, chain its
    value off the host's parent value, and put exactly one visit on each node of
    the real root->parent->child lineage -- and none on the host's new sibling.
    """
    base = _mk(tmp / "advanced")
    code = _mk(base / "code")
    for name in ("r", "x", "p", "s"):
        (code / f"kernel_{name}.py").write_text(f"// {name}\nvoid k(){{}}\n",
                                                encoding="utf-8")
    H = MonteCarloTreeSearch()
    root = H.observe(kernel_name="kernel_r", kernel_path=str(code / "kernel_r.py"),
                     value=1.0).key
    xk = H.observe(kernel_name="kernel_x", kernel_path=str(code / "kernel_x.py"),
                   value=1.05, parent_key=root, mechanism="mx").key
    pk = H.observe(kernel_name="kernel_p", kernel_path=str(code / "kernel_p.py"),
                   value=1.10, parent_key=xk, mechanism="mp").key
    for path in ([root], [root, xk], [root, xk, pk]):
        H.backup(path, 0.5)

    hroot = _mk(base / "host")
    ck = _write_host(hroot, H, task=_TASK, best_path=str(code / "kernel_r.py"))
    blob = json.loads(json.dumps(H.to_dict()))
    Q = MonteCarloTreeSearch.from_dict(json.loads(json.dumps(blob)))

    rec = {"obs_id": "adv1", "ts": "t", "rollout_idx": 0, "origin": "quick",
           "task": _TASK, "ref_py": _TASK,
           "device": 0, "parent_key": pk, "mechanism": "mz", "kernel_name": "kernel_q",
           "kernel_path": str(code / "kernel_q.py"), "code_sha": "qqq",
           "rel_pct": 2.0, "runnable": True, "basis": "paired",
           "count_failures": False}
    (code / "kernel_q.py").write_text("// q\nvoid k(){}\n", encoding="utf-8")

    # The host advances AFTER the fork: a normal round gives P a sibling under X
    # and puts more visits on the root->X spine.
    sk = H.observe(kernel_name="kernel_s", kernel_path=str(code / "kernel_s.py"),
                   value=1.08, parent_key=xk, mechanism="ms").key
    H.backup([root, xk, sk], 0.6)
    _write_host(hroot, H, task=_TASK, best_path=str(code / "kernel_r.py"))
    before = {k: H.nodes[k].N for k in (root, xk, pk, sk)}
    host_nodes_before = set(H.nodes)

    qdir = _mk(base / "quick")
    _write_quick(qdir, Q, [rec], task=_TASK,
                 fork=_fork_meta(hroot, ck, blob))
    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["ok"] and res["report"]["applied"] == 1,
           f"the record still lands after the host moved on ({res.get('reason')})")
    M = MonteCarloTreeSearch.from_dict(
        json.loads((hroot / "checkpoint.json.merged").read_text(encoding="utf-8"))["mcts"])

    new_keys = set(M.nodes) - host_nodes_before
    _check(len(new_keys) == 1, f"exactly one node was added (got {len(new_keys)})")
    child = new_keys.pop()
    _check(M.nodes[child].kernel == "kernel_q", "and it is the record's kernel")
    _check(M.nodes[child].parent == pk,
           "hung off the parent the record named, not re-keyed into anything else")
    _check(M.path_to(child) == [root, xk, pk, child],
           f"its lineage is the host's real route (got {M.path_to(child)})")
    _check(M.nodes[root].N == before[root] + 1 and M.nodes[xk].N == before[xk] + 1
           and M.nodes[pk].N == before[pk] + 1 and M.nodes[child].N == 1,
           "exactly one visit on every node of that lineage, and one on the child")
    _check(M.nodes[sk].N == before[sk],
           f"the host's new sibling S got NO visit ({M.nodes[sk].N}) -- backup "
           f"walks the lineage, not the subtree")
    # The chained value is computed against the HOST's parent, on the host's
    # basis, so the merged number is comparable with every other number in the
    # host's ledger rather than with the quick run's own chain.
    _check(abs(M.nodes[child].value - 1.10 * 1.02) < 1e-12,
           f"and its value is host_P.value * (1 + 2.00/100) = "
           f"{M.nodes[child].value:.6f}, chained on the host's ledger")
    _check(M.nodes[child].key not in Q.nodes,
           "the id is minted in HOST space; the quick run's own id is not imported")


if __name__ == "__main__":
    main()
