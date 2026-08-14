"""Mechanics test for the MCGS quick path -- no GPU, no LLM, no subprocess.

Scope, stated up front: this proves that `utils/mcgs_quick.py`'s plumbing and its
merge ALGEBRA are correct. It says nothing about whether skipping ncu and the
judge costs anything in kernel quality -- that question needs a live A/B
(`mcgs_quick run` vs `main_memory_latest.py --search mcgs`, matched rollout budget
on one task, compared on paired gain per wall-clock hour), because the gain a
rollout produces depends on the prompt it was given and the counterfactual prompt
was never sent.

What IS checked here is everything that could silently break the loop or corrupt a
host graph:
  * importing the module drags in neither main_memory_latest nor torch -- the
    tripwire for anyone who hoists that import to module scope and makes every
    spawned bench child pay for it twice per rollout
  * the mechanism draw is never empty and never None, so the key space can never
    collapse into the x:m-empty bucket
  * the synthesized strategy dict survives the real prompt renderer
  * a stubbed 25-rollout replay grows a real graph within max_depth
  * observe()'s split key is used instead of the key that was passed -- the latent
    KeyError at main_memory_latest.py:3139-3142
  * x-fallback collisions and cycle-closing edges are refused, not written
  * replaying the journal reproduces the live graph exactly, and replaying it
    twice changes nothing (the widening-budget inflation defence)
  * every merge precondition refuses and writes nothing
  * a legitimate merge recomputes values on the HOST's chain and re-finds parents
    by the host's current shortest route

Run: python -m utils.test_mcgs_quick
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
import utils.mcgs_quick as mq
_IMPORT_SNAPSHOT = set(sys.modules)

from utils.mcgs import MechanismPrior, MonteCarloGraphSearch, state_key
# The same measured gain distribution the search replay uses: 112 parent->child
# edges on vae_block_002, bimodal, 42% regressing past -1%. Reused rather than
# re-tabulated so the two tests cannot drift apart on what "a realistic round"
# looks like.
from utils.test_mcgs_replay import _sample_gain


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class _FakeInd:
    """The four attributes _one_rollout touches on a KernelIndividual.

    A local class rather than scripts.individual.KernelIndividual: that module is
    cheap today, but importing it invites the torch chain into this test the first
    time someone adds an import to it, and assertion 1 would then start failing
    for a reason that has nothing to do with mcgs_quick.
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


def _seed_graph(code_dir: Path, *, prior: Optional[MechanismPrior] = None,
                max_depth: int = 10) -> Tuple[MonteCarloGraphSearch, str]:
    g = MonteCarloGraphSearch(state_key_mode="mechanisms", merge_tolerance=0.15,
                              max_depth=max_depth, prior=prior)
    code_dir.mkdir(parents=True, exist_ok=True)
    seed = code_dir / "kernel_seed.py"
    seed.write_text("// seed\nvoid k(){}\n", encoding="utf-8")
    root = state_key(mode="mechanisms", mechanisms=[], fallback=seed.stem)
    g.observe(key=root, kernel_name=seed.stem, kernel_path=str(seed), value=1.0,
              parent_key=None, runnable=True, note="seed")
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


def _drive(graph: MonteCarloGraphSearch, work: Path, *, rounds: int, seed: int,
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
                graph, rollout_idx=i, rng=random.Random(seed * 1000 + i),
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
            landed, why = mq._apply_record(graph, rec, applied, rep_sha)
            outcomes[rec["obs_id"]] = {"landed_key_local": landed,
                                       "apply_reason": why}
            done = i + 1
    return dict(graph=graph, records=records, applied=applied, rep_sha=rep_sha,
                outcomes=outcomes, rollouts_done=done, log=log.getvalue())


def _replay(records: List[Dict[str, Any]], code_dir: Path,
            **kw) -> MonteCarloGraphSearch:
    """Rebuild a graph from the seed by folding the journal in, in file order."""
    g, _root = _seed_graph(code_dir, **kw)
    applied: set = set()
    rep_sha: Dict[str, str] = {}
    for rec in records:
        mq._apply_record(g, rec, applied, rep_sha)
    return g


def _node_fields(g: MonteCarloGraphSearch, k: str) -> Tuple:
    n = g.nodes[k]
    return (n.N, round(n.W, 12), round(n.M, 12), n.failures, n.depth, n.rep,
            round(n.rep_value, 12))


# ---------------------------------------------------------------------------
# merge fixtures
# ---------------------------------------------------------------------------
def _write_host(root: Path, graph: Optional[MonteCarloGraphSearch], *,
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
        "mcgs": graph.to_dict() if graph is not None else None,
        "opt_history_files": {}, "next_individual_id": 3,
        "timestamp": "2026-08-14T09:00:00",
    }
    ckpt.update(extra or {})
    p = root / "checkpoint.json"
    p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    return p


def _write_quick(qdir: Path, graph: MonteCarloGraphSearch,
                 records: List[Dict[str, Any]], *, task: str,
                 fork: Dict[str, Any],
                 outcomes: Optional[Dict[str, Any]] = None,
                 clock: Optional[Dict[str, Any]] = None) -> Path:
    qdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "quick_version": 1, "kind": "mcgs_quick_graph", "run_id": qdir.name,
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
    mq._write_graph(qdir / "graph.json", graph, meta)
    jl = qdir / "journal.jsonl"
    jl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return qdir / "graph.json"


def _fork_meta(host_root: Path, host_ckpt: Path, blob: Dict[str, Any],
               *, forked: bool = True) -> Dict[str, Any]:
    import hashlib
    return {"host_task_root": str(host_root),
            "host_checkpoint_sha": hashlib.sha1(host_ckpt.read_bytes()).hexdigest(),
            "fork_mcgs_sha": hashlib.sha1(
                json.dumps(blob, sort_keys=True).encode()).hexdigest(),
            "forked_from_mcgs": forked, "fork_root": blob.get("root"),
            "fork_node_keys": sorted(blob.get("nodes") or {})}


# ---------------------------------------------------------------------------
def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mcgs_quick_test_"))
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
           "importing utils.mcgs_quick does NOT import main_memory_latest")
    _check("torch" not in _IMPORT_SNAPSHOT,
           "importing utils.mcgs_quick does NOT import torch")
    src = Path(mq.os.environ.get("SOLBENCH_SRC", ""))
    _check(str(src) and src.is_dir(),
           f"SOLBENCH_SRC is set to an existing directory ({src}) -- the ref_*.py "
           f"default is /home/elek/... and dies inside the spawned bench worker")

    print("\n[quick] the mechanism draw can never be empty")
    g0, _ = _seed_graph(tmp / "m0")
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

    print("\n[quick] a mechanism already tried from this state is damped")
    gd, rootd = _seed_graph(tmp / "m1")
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
    gs, _ = _seed_graph(tmp / "m2")
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
           "ratchet branch that reads it is inert under MCGS")
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
    g1, _ = _seed_graph(w1 / "code")
    r1 = _drive(g1, w1, rounds=25, seed=7)
    st = g1.stats()
    print(f"  graph: {json.dumps(st)}")
    _check(st["states"] > 1, "the graph grew beyond the root")
    parents = [r["parent_key"] for r in r1["records"]]
    switches = sum(1 for a, b in zip(parents, parents[1:]) if a != b)
    _check(switches > 0,
           f"selection revisited non-incumbent states ({switches} parent switches) "
           f"-- the ratchet would score 0 here")
    _check(max(n.depth for n in g1.nodes.values()) <= 10, "max_depth was honoured")
    _check(all(n.rep_value > 0 and n.rep_value == n.rep_value
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

    print("\n[quick] observe()'s split key is used, not the key handed in")
    gsp, rootsp = _seed_graph(_mk(tmp / "split" / "code"))
    base = {"ts": "t", "rollout_idx": 0, "origin": "quick", "parent_key": rootsp,
            "count_failures": False, "runnable": True, "basis": "paired",
            "kernel_path": "/tmp/a.py", "mechanism": "M"}
    a = dict(base, obs_id="A", kernel_name="ka", code_sha="sha_a", rel_pct=0.0)
    b = dict(base, obs_id="B", kernel_name="kb", code_sha="sha_b", rel_pct=60.0,
             rollout_idx=1)
    ap, rp = set(), {}
    k1, why1 = mq._apply_record(gsp, a, ap, rp)
    k2, why2 = mq._apply_record(gsp, b, ap, rp)
    _check(why1 == "applied" and why2 == "applied", "both records landed")
    _check(k2 is not None and k2.startswith(k1 + "/s"),
           f"the +60% child was SPLIT off ({k2}) rather than pooled with a rep "
           f"15% away")
    _check(gsp.nodes[k2].N == 1,
           "and backup reached the split node -- main_memory_latest.py:3141 backs "
           "up the pre-split key, so the child there receives no visit at all")
    _check(gsp.splits == 1, "the refusal is counted")

    print("\n[quick] an x-fallback collision is refused")
    gx, rootx = _seed_graph(_mk(tmp / "xcoll" / "code"))
    # mechanism=None from a root with via=None gives mechanisms=[], i.e. an "x:"
    # key hashing the FILE STEM. Two runs naming a kernel the same way collide by
    # construction with unrelated code -- save_kernel_code stamps at one-second
    # resolution, so this is not hypothetical.
    xa = {"obs_id": "X1", "rollout_idx": 0, "origin": "q", "parent_key": rootx,
          "count_failures": False, "runnable": True, "basis": "paired",
          "kernel_name": "round003_k", "kernel_path": "/tmp/x1.py",
          "code_sha": "aaa", "rel_pct": 1.0, "mechanism": None}
    xb = dict(xa, obs_id="X2", code_sha="bbb", rel_pct=2.0, kernel_path="/tmp/x2.py")
    apx, rpx = set(), {}
    kx1, whyx1 = mq._apply_record(gx, xa, apx, rpx)
    n_before = len(gx.nodes), gx.total_visits
    kx2, whyx2 = mq._apply_record(gx, xb, apx, rpx)
    _check(whyx1 == "applied" and kx1.startswith("x:"),
           f"the first record lands on an x: key ({kx1})")
    _check(whyx2 == "x-fallback-collision",
           "the second, with different code under the same stem, is refused")
    _check((len(gx.nodes), gx.total_visits) == n_before,
           "and the graph is untouched -- no node, no visit")

    print("\n[quick] a cycle-closing edge is refused (select() HANGS on a cycle)")
    gc = MonteCarloGraphSearch(state_key_mode="mechanisms")
    ak = state_key(mode="mechanisms", mechanisms=["m1", "m2", "m3"])
    gc.observe(key="R", kernel_name="r", kernel_path="/tmp/r.py", value=1.0)
    gc.observe(key=ak, kernel_name="a", kernel_path="/tmp/a.py", value=1.0,
               parent_key="R", mechanism="m1")
    gc.observe(key="B", kernel_name="b", kernel_path="/tmp/b.py", value=1.0,
               parent_key=ak, mechanism="m2")
    cyc = {"obs_id": "C", "rollout_idx": 0, "origin": "q", "parent_key": "B",
           "count_failures": False, "runnable": True, "basis": "paired",
           "kernel_name": "c", "kernel_path": "/tmp/c.py", "code_sha": "ccc",
           "rel_pct": 1.0, "mechanism": "m3"}
    before = (len(gc.nodes), gc.total_visits, gc.nodes[ak].N)
    landed, why = mq._apply_record(gc, cyc, set(), {})
    _check(why == "would-close-a-cycle",
           "the edge B -> {m1,m2,m3} would make A its own descendant; refused")
    _check((len(gc.nodes), gc.total_visits, gc.nodes[ak].N) == before,
           "and nothing was written -- utils/mcgs.py:582 is `while True` with no "
           "visited set, so a 2-cycle hangs the search rather than raising")

    print("\n[quick] crash recovery: replaying the journal reproduces the graph")
    g_replay = _replay(r1["records"], _mk(tmp / "replay" / "code"))
    _check(set(g_replay.nodes) == set(g1.nodes),
           f"identical node key sets ({len(g_replay.nodes)} states)")
    _check(all(_node_fields(g_replay, k) == _node_fields(g1, k) for k in g1.nodes),
           "identical N, W, M, failures, depth, rep and rep_value on every state")
    _check(g_replay.total_visits == g1.total_visits,
           f"identical total_visits ({g_replay.total_visits})")
    _check(g_replay.splits == g1.splits, "identical split count")

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
    meta = {"quick_version": 1, "kind": "mcgs_quick_graph", "run_id": "rt",
            "task": "/tmp/task.py", "rollouts_done": 25,
            "applied_obs": sorted(r1["applied"]), "rep_sha": r1["rep_sha"]}
    mq._write_graph(gp / "graph.json", g1, meta)
    g_rt, meta_rt, applied_rt = mq._read_graph(gp / "graph.json")
    _check(not list(gp.glob("*.tmp")), "no .tmp file is left behind")
    _check(g_rt.root == g1.root and set(g_rt.nodes) == set(g1.nodes),
           "root and node set survive")
    _check(all(abs(g_rt.nodes[k].q() - g1.nodes[k].q()) < 1e-12 for k in g1.nodes),
           "every Q is reproduced exactly")
    _check(g_rt.splits == g1.splits and applied_rt == r1["applied"],
           "split counter and applied_obs survive")
    _check(g_rt.select().node.key == g1.select().node.key,
           "the restored graph selects the same state -- a resume continues, it "
           "does not restart")

    print("\n[quick] merge refuses, and writes nothing, on every precondition")
    _merge_refusals(tmp)

    print("\n[quick] merge on a legitimate fork")
    _merge_happy(tmp)

    print("\n[quick] merge after the host has moved on")
    _merge_host_advanced(tmp)

    print("\n[quick] graceful stop at a rollout boundary")
    mq._STOP["requested"] = False
    try:
        w2 = tmp / "stop"
        g2, _ = _seed_graph(_mk(w2 / "code"))
        r2 = _drive(g2, w2, rounds=25, seed=3, stop_after=3, run_id="quick_stop")
        _check(len(r2["records"]) == 4,
               "the in-flight rollout finished, then the loop stopped "
               f"({len(r2['records'])} records: rollouts 0-3)")
        _check(r2["rollouts_done"] == 4, "rollouts_done matches the journal")
        gp2 = tmp / "stopgraph"
        gp2.mkdir()
        mq._write_graph(gp2 / "graph.json", g2,
                        {"quick_version": 1, "kind": "mcgs_quick_graph",
                         "run_id": "s", "rollouts_done": 4,
                         "applied_obs": sorted(r2["applied"])})
        g_disk, _m, _a = mq._read_graph(gp2 / "graph.json")
        _check(all(_node_fields(g_disk, k) == _node_fields(g2, k) for k in g2.nodes),
               "the graph on disk equals the in-memory graph at exit")
    finally:
        mq._STOP["requested"] = False

    print("\n[quick] all checks passed")
    print("NOTE: this proves the quick path's mechanics and its merge algebra, not "
          "that skipping ncu and the judge costs nothing. That question needs a "
          "live A/B -- `mcgs_quick run` vs `main_memory_latest.py --search mcgs`, "
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

    def _host_graph() -> MonteCarloGraphSearch:
        g = MonteCarloGraphSearch(state_key_mode="mechanisms")
        root = state_key(mode="mechanisms", mechanisms=[], fallback=seed.stem)
        g.observe(key=root, kernel_name=seed.stem, kernel_path=str(seed), value=1.0)
        return g

    cases = []

    # R2 -- different state_key_mode
    H = _host_graph()
    Q = MonteCarloGraphSearch.from_dict(H.to_dict())
    Q.state_key_mode = "code"
    cases.append(("R2-state-key-mode", H, Q, {}, {}))

    # R3 -- different root key
    H = _host_graph()
    Q = MonteCarloGraphSearch.from_dict(H.to_dict())
    Q.root = "m:deadbeefdeadbeef"
    Q.nodes[Q.root] = Q.nodes.pop(H.root)
    Q.nodes[Q.root].key = Q.root
    cases.append(("R3-root-key", H, Q, {}, {}))

    # R4 -- same root key, different source
    H = _host_graph()
    other = code / "kernel_other.py"
    other.write_text("// something else entirely\nvoid z(){}\n", encoding="utf-8")
    Q = MonteCarloGraphSearch.from_dict(H.to_dict())
    Q.nodes[Q.root].rep_path = str(other)
    cases.append(("R4-root-source", H, Q, {}, {}))

    # R5 -- the roots' measured values disagree by 5%
    H = _host_graph()
    Q = MonteCarloGraphSearch.from_dict(H.to_dict())
    Q.nodes[Q.root].rep_value = 1.05
    cases.append(("R5-root-value", H, Q, {}, {}))

    # R6 -- reward_scale mismatch without --force_reward_scale
    H = _host_graph()
    Q = MonteCarloGraphSearch.from_dict(H.to_dict())
    Q.reward_scale = 5.0
    cases.append(("R6-reward-scale", H, Q, {}, {}))

    # R7 -- the quick run was measured unlocked
    H = _host_graph()
    Q = MonteCarloGraphSearch.from_dict(H.to_dict())
    cases.append(("R7-clock", H, Q,
                  {"clock": {"locked": False, "reason": "opted out"}}, {}))

    # R10 -- bootstrapped, not forked
    H = _host_graph()
    Q = MonteCarloGraphSearch.from_dict(H.to_dict())
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
               and not (hroot / "mcgs_quick_merged.json").exists(),
               f"{want}: nothing was written")

    # R9 -- the host is live (checkpoint.json touched just now)
    root = _mk(base / "live")
    hroot, qdir = _mk(root / "host"), _mk(root / "quick")
    H = _host_graph()
    ck = _write_host(hroot, H, task=_TASK,
                     best_path=str(seed))
    _write_quick(qdir, MonteCarloGraphSearch.from_dict(H.to_dict()), [],
                 task=_TASK,
                 fork=_fork_meta(hroot, ck, H.to_dict()))
    res = mq.merge(qdir, hroot, host_idle_s=900)
    _check(res["reason"] == "R9-host-live",
           "R9-host-live: a checkpoint.json written seconds ago blocks the merge "
           "(median normal round is 570s, so 900s clears one with margin)")
    _check(not (hroot / "checkpoint.json.merged").exists(),
           "R9-host-live: nothing was written")

    # R1 -- the host never ran MCGS
    root = _mk(base / "nomcgs")
    hroot, qdir = _mk(root / "host"), _mk(root / "quick")
    ck = _write_host(hroot, None, task=_TASK,
                     best_path=str(seed))
    H = _host_graph()
    _write_quick(qdir, H, [], task=_TASK,
                 fork=_fork_meta(hroot, ck, H.to_dict()))
    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["reason"] == "R1-host-has-no-mcgs",
           "R1-host-has-no-mcgs: refused without --adopt (measured 2026-08-14: 0 "
           "of 8 checkpoint.json files under run/ carry an mcgs blob)")

    # R8 -- a different workload
    root = _mk(base / "task")
    hroot, qdir = _mk(root / "host"), _mk(root / "quick")
    H = _host_graph()
    ck = _write_host(hroot, H, task=_TASK,
                     best_path=str(seed))
    rec = {"obs_id": "z", "rollout_idx": 0, "origin": "q", "task": "/tmp/tasks/other_task.py",
           "device": 0, "parent_key": H.root, "mechanism": "m", "kernel_name": "k",
           "kernel_path": "/tmp/k.py", "code_sha": "s", "rel_pct": 1.0,
           "runnable": True, "basis": "paired", "count_failures": False}
    _write_quick(qdir, MonteCarloGraphSearch.from_dict(H.to_dict()), [rec],
                 task=_TASK,
                 fork=_fork_meta(hroot, ck, H.to_dict()))
    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["reason"] == "R8-task",
           "R8-task: a journal measured on another workload is refused")


def _merge_happy(tmp: Path) -> None:
    base = _mk(tmp / "happy")
    hdir = _mk(base / "hostwork")
    H, _root = _seed_graph(_mk(hdir / "code"))
    _drive(H, hdir, rounds=10, seed=13, run_id="host_sim")
    _check(len(H.nodes) >= 4, f"the host graph has {len(H.nodes)} states to merge into")

    hroot = _mk(base / "host")
    seed_path = H.nodes[H.root].rep_path
    ck = _write_host(hroot, H, task=_TASK,
                     best_path=seed_path)
    host_before = json.loads(ck.read_text(encoding="utf-8"))
    blob_before = json.loads(json.dumps(H.to_dict()))
    prior_before = json.dumps(H.prior.to_dict() if H.prior else None, sort_keys=True)

    Q = MonteCarloGraphSearch.from_dict(json.loads(json.dumps(H.to_dict())))
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
    _check(all(out[k] == host_before[k] for k in host_before if k != "mcgs"),
           "every non-mcgs key of the checkpoint is byte-identical -- not base, "
           "not best, not base_score, not next_round")
    M = MonteCarloGraphSearch.from_dict(out["mcgs"])
    _check(M.total_visits == rep["visits_before"] + rep["applied"],
           f"total_visits advanced by the records that ACTUALLY landed "
           f"({rep['applied']}), not by the quick graph's total_visits "
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
    _check((hroot / "mcgs_quick_merged.json").exists(),
           "the obs_id ledger was written beside the checkpoint (its own file: "
           "to_dict rebuilds the mcgs blob from scratch and _save_checkpoint "
           "writes a fixed schema, so neither would survive the host's next round)")

    # The load-bearing arithmetic: values are recomputed on the HOST's chain, not
    # imported. Only counted where the parent's representative did NOT move after
    # the record landed -- once a later record promotes a better rep, the parent's
    # rep_value at apply time is no longer readable from the final graph, so those
    # nodes are unverifiable from here rather than wrong. The crisp,
    # fully-determined version of this check is in _merge_host_advanced.
    checked = 0
    for rec in rq["records"]:
        if not (rec.get("runnable") and rec.get("kernel_name")):
            continue
        parent = M.nodes.get(rec["parent_key"])
        if parent is None:
            continue
        want = parent.rep_value * (1.0 + rec["rel_pct"] / 100.0)
        for node in M.nodes.values():
            if node.rep == rec["kernel_name"] and abs(node.rep_value - want) < 1e-12:
                checked += 1
                break
    _check(checked > 0,
           f"{checked} merged node(s) carry host_parent.rep_value * (1 + rel/100) "
           f"exactly -- the GAIN was replayed onto the host's chain, the value was "
           f"not imported from a chain anchored at another seed")

    # Idempotence at the merge level, which is what the ledger is for.
    res2 = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res2["ok"] and res2["report"]["applied"] == 0,
           f"re-merging the same journal applies 0 records "
           f"(all {res2['report']['refusals'].get('duplicate', 0)} are duplicates) "
           f"-- N and W cannot be doubled")


def _merge_host_advanced(tmp: Path) -> None:
    """The host acquires a SHORTER route to the parent after the fork."""
    base = _mk(tmp / "advanced")
    code = _mk(base / "code")
    for name in ("r", "x", "p"):
        (code / f"kernel_{name}.py").write_text(f"// {name}\nvoid k(){{}}\n",
                                                encoding="utf-8")
    H = MonteCarloGraphSearch(state_key_mode="mechanisms")
    root = state_key(mode="mechanisms", mechanisms=[], fallback="kernel_r")
    xk = state_key(mode="mechanisms", mechanisms=["mx"])
    pk = state_key(mode="mechanisms", mechanisms=["mx", "mp"])
    H.observe(key=root, kernel_name="kernel_r", kernel_path=str(code / "kernel_r.py"),
              value=1.0)
    H.observe(key=xk, kernel_name="kernel_x", kernel_path=str(code / "kernel_x.py"),
              value=1.05, parent_key=root, mechanism="mx")
    H.observe(key=pk, kernel_name="kernel_p", kernel_path=str(code / "kernel_p.py"),
              value=1.10, parent_key=xk, mechanism="mp")
    for path in ([root], [root, xk], [root, xk, pk]):
        H.backup(path, 0.5)

    hroot = _mk(base / "host")
    ck = _write_host(hroot, H, task=_TASK,
                     best_path=str(code / "kernel_r.py"))
    blob = json.loads(json.dumps(H.to_dict()))
    Q = MonteCarloGraphSearch.from_dict(json.loads(json.dumps(blob)))

    rec = {"obs_id": "adv1", "ts": "t", "rollout_idx": 0, "origin": "quick",
           "task": _TASK, "ref_py": _TASK,
           "device": 0, "parent_key": pk, "mechanism": "mz", "kernel_name": "kernel_q",
           "kernel_path": str(code / "kernel_q.py"), "code_sha": "qqq",
           "rel_pct": 2.0, "runnable": True, "basis": "paired",
           "count_failures": False,
           "child_key_local": state_key(mode="mechanisms",
                                        mechanisms=["mx", "mp", "mz"])}
    (code / "kernel_q.py").write_text("// q\nvoid k(){}\n", encoding="utf-8")

    # The host advances AFTER the fork: it re-observes P directly under the root,
    # which is a legitimate transposition and gives P a strictly shorter route.
    H.observe(key=pk, kernel_name="kernel_p2",
              kernel_path=str(code / "kernel_p.py"), value=1.12, parent_key=root,
              mechanism="mp")
    H.backup([root, pk], 0.6)
    _write_host(hroot, H, task=_TASK,
                best_path=str(code / "kernel_r.py"))
    x_visits_before = H.nodes[xk].N
    p_visits_before = H.nodes[pk].N

    qdir = _mk(base / "quick")
    _write_quick(qdir, Q, [rec], task=_TASK,
                 fork=_fork_meta(hroot, ck, blob))
    res = mq.merge(qdir, hroot, host_idle_s=0)
    _check(res["ok"] and res["report"]["applied"] == 1,
           f"the record still lands after the host moved ({res.get('reason')})")
    M = MonteCarloGraphSearch.from_dict(
        json.loads((hroot / "checkpoint.json.merged").read_text(encoding="utf-8"))["mcgs"])
    _check(M.nodes[xk].N == x_visits_before,
           f"the intermediate state X got NO visit ({M.nodes[xk].N}) -- the backup "
           f"walked the host's CURRENT shortest route root->P, not the journal's "
           f"recorded root->X->P")
    _check(M.nodes[pk].N == p_visits_before + 1, "the parent P did get its visit")
    child = state_key(mode="mechanisms", mechanisms=["mp", "mz"])
    _check(child in M.nodes,
           f"the child key was recomputed in HOST space from path_mechanisms of "
           f"the shorter route ({child}), not imported as "
           f"{rec['child_key_local']}")
    _check(rec["child_key_local"] not in M.nodes,
           "and the quick run's own key for that child is nowhere in the host")
    # P's representative advanced to 1.12 when the host re-observed it after the
    # fork, so the merged child must be 1.12 * 1.02 -- NOT 1.10 * 1.02 (the value
    # the quick graph would have computed) and not the quick chain's own number.
    _check(abs(M.nodes[child].rep_value - 1.12 * 1.02) < 1e-12,
           f"and its value is host_P.rep_value * (1 + 2.00/100) = "
           f"{M.nodes[child].rep_value:.6f}, computed from the host's CURRENT "
           f"representative (1.12), not the one the quick run forked (1.10)")


if __name__ == "__main__":
    main()
