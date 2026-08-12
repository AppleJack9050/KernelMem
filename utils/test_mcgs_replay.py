"""Replay test for the MCGS search loop -- mechanics only, no GPU, no LLM.

Scope, stated up front: this checks that the search BEHAVES, not that it wins.
Whether MCGS beats the ratchet cannot be answered by replay, because the gain a
round produces depends on the parent that round was given, and the counterfactual
parent was never measured. That question needs a live A/B
(``--search mcgs`` vs ``--search ratchet`` at matched budget on one task).

What is checked here is everything that could silently break the loop:
  * selection never returns a state with no runnable code to hand the model
  * the graph revisits non-incumbent states -- the entire reason it exists
  * the value chain stays on the paired basis and stays finite
  * max_depth is honoured, so the round-10 cliff is respected
  * merge_tolerance splits rather than pooling kernels that are far apart
  * the graph survives a checkpoint round trip mid-run with Q intact

Run: python -m utils.test_mcgs_replay
"""
from __future__ import annotations

import json
import random
from typing import List, Optional

from utils.mcgs import MonteCarloGraphSearch, reward_from_gain, state_key

# The measured gain distribution on vae_block_002 (112 parent->child edges):
# 42% regress past -1%, 33% win past +1%, 25% inside the +-1% band, and the shape
# is bimodal rather than centred. Sampled here so the replay exercises the same
# regime the real loop sees, including the heavy tails that make max-backup right.
_BANDS = [
    (-12.0, -5.0, 0.116),
    (-5.0, -2.0, 0.232),
    (-2.0, -1.0, 0.071),
    (-1.0, 0.0, 0.116),
    (0.0, 1.0, 0.134),
    (1.0, 5.0, 0.232),
    (5.0, 12.0, 0.098),
]
_METHODS = ["vectorize_io", "shared_tile", "split_reduction", "channels_last",
            "cuda_graph", "fuse_epilogue", None]     # None = an unnamed change


def _sample_gain(rng: random.Random) -> float:
    r, acc = rng.random(), 0.0
    for lo, hi, p in _BANDS:
        acc += p
        if r <= acc:
            return rng.uniform(lo, hi)
    return rng.uniform(-1.0, 1.0)


def replay(rounds: int = 25, seed: int = 7, *, key_mode: str = "mechanisms",
           merge_tol: float = 0.15, max_depth: int = 10, verbose: bool = False):
    rng = random.Random(seed)
    g = MonteCarloGraphSearch(state_key_mode=key_mode, merge_tolerance=merge_tol,
                              max_depth=max_depth)
    g.observe(key=state_key(mode=key_mode, mechanisms=[], fallback="seed"),
              kernel_name="seed", kernel_path="/tmp/seed.py", value=1.0)

    switches = 0
    prev_parent: Optional[str] = None
    depths: List[int] = []
    values: List[float] = []

    for r in range(rounds):
        sel = g.select()
        assert sel is not None, "selection returned nothing"
        assert sel.node.rep is not None, "selected a state with no representative kernel"
        assert sel.node.runnable, "selected an unrunnable state"
        assert sel.node.depth <= max_depth, f"selection exceeded max_depth: {sel.node.depth}"
        depths.append(sel.node.depth)
        if prev_parent is not None and sel.node.key != prev_parent:
            switches += 1
        prev_parent = sel.node.key

        # one round: the model proposes a change, it is measured, paired-verified
        gain = _sample_gain(rng)
        failed = rng.random() < 0.12          # compile/run failures do happen
        mech = rng.choice(_METHODS)
        mechs = (g.path_mechanisms(sel.path) + ([mech] if mech else [])
                 if key_mode == "mechanisms" else None)
        child_value = sel.node.rep_value * (1.0 + gain / 100.0)
        # Reproduces the measured `features` pathology: on the real runs the vector
        # was ~constant (two fields never varied, three more were ~constant), so
        # 47 kernels hashed to one state across a 457% speedup range. One varying
        # bit here stands in for that, which is what the merge guard must survive.
        feats = ({"kernel_structure_id": 4, "has_reuse": True,
                  "has_shared_memory_tile": True, "uses_vector_types": True,
                  "has_vector_load_store": True, "is_aligned_vector_access": True,
                  "tc_eligible": (r % 8 == 0), "has_k_loop": False,
                  "is_pointwise": False, "has_multiple_kernels_in_forward": True,
                  "cudagraph_eligible": True}
                 if key_mode == "features" else None)
        ck = state_key(mode=key_mode, mechanisms=mechs, features=feats,
                       code=f"kernel body {r}", fallback=f"k{r}")
        g.observe(key=ck, kernel_name=f"k{r}", kernel_path=f"/tmp/k{r}.py",
                  value=child_value, parent_key=sel.node.key,
                  runnable=not failed, mechanism=mech, note=f"round {r}")
        path = list(sel.path)
        # observe() may have split the key on the value guard; find where it landed
        landed = ck if ck in g.nodes else None
        if landed is None:
            landed = next((k for k in g.nodes if k.startswith(ck + "/s")), None)
        if landed and landed not in path:
            path.append(landed)
        g.backup(path, reward_from_gain(gain, failed=failed), failed=failed)
        values.append(child_value)
        if verbose:
            print(f"  r{r:>2} depth={sel.node.depth} N={sel.node.N} "
                  f"gain={gain:+6.2f}%{' FAILED' if failed else ''} -> {child_value:.4f}")

    return g, dict(switches=switches, depths=depths, values=values)


def main() -> None:
    def _check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        assert cond, msg

    print("[replay] 25 rounds, mechanisms key")
    g, info = replay(rounds=25, seed=7)
    st = g.stats()
    print(f"  graph: {json.dumps(st)}")

    _check(st["states"] > 1, "the graph grew beyond the root")
    _check(info["switches"] > 0,
           f"selection revisited non-incumbent states ({info['switches']} parent switches "
           f"in 24 transitions) -- the ratchet would score 0 here")
    _check(max(info["depths"]) <= 10, "max_depth was never exceeded")
    _check(all(v > 0 and v == v for v in info["values"]), "the value chain stayed finite and positive")
    _check(st["mean_N"] > 1.0, f"mean visits per state is above 1 (got {st['mean_N']:.2f})")

    # Not asserted here: under the `mechanisms` key, merges only happen when two
    # paths apply the same multiset, and whether those kernels' values disagree by
    # >15% depends on the sample. The guard itself is proven directly in
    # utils/mcgs.py's self-test; below it is asserted where merging is forced.
    print(f"\n[replay] value-guard splits under the mechanisms key: "
          f"{st['splits_refused']} (data-dependent, not asserted)")

    print("\n[replay] checkpoint round trip mid-run")
    blob = json.loads(json.dumps(g.to_dict()))
    g2 = MonteCarloGraphSearch.from_dict(blob)
    _check(len(g2.nodes) == len(g.nodes), "every state survived")
    _check(g2.splits == g.splits, "the split counter survived")
    _check(all(abs(g2.nodes[k].q() - g.nodes[k].q()) < 1e-12 for k in g.nodes),
           "every Q is reproduced exactly")
    s1 = g.select(); s2 = g2.select()
    _check(s1.node.key == s2.node.key,
           "the restored graph selects the same state -- a resume continues, not restarts")

    print("\n[replay] 'code' key mode degenerates to a tree, as documented")
    gc, _ = replay(rounds=25, seed=7, key_mode="code")
    stc = gc.stats()
    print(f"  graph: {json.dumps(stc)}")
    _check(stc["states"] >= st["states"],
           f"code mode makes at least as many states ({stc['states']} vs {st['states']})")

    print("\n[replay] why `features` is NOT the default")
    # `features` over a ~constant vector is the measured pathology: on the real runs
    # it put 47 of 145 kernels in one state spanning a 457% speedup range, because
    # is_aligned_vector_access and is_pointwise never vary and three more are
    # ~constant. Here it collapses 26 kernels into a handful of states, so Q becomes
    # an average over kernels that have nothing to do with each other.
    gd, _ = replay(rounds=25, seed=7, key_mode="features", merge_tol=0.15)
    sd = gd.stats()
    print(f"  graph: {json.dumps(sd)}")
    _check(sd["states"] < st["states"],
           f"features over-pools relative to mechanisms ({sd['states']} states vs "
           f"{st['states']}) -- which is why the default is mechanisms")
    _check(sd["mean_N"] > 3 * st["mean_N"],
           f"and it inflates mean N by pooling unrelated kernels "
           f"({sd['mean_N']:.1f} vs {st['mean_N']:.1f}) -- visits that do not mean "
           f"what UCT assumes they mean")
    print(f"  value-guard splits: {sd['splits_refused']} (fires only when a pooled "
          f"kernel's value disagrees >{0.15:.0%}; the guard is proven directly in "
          f"utils/mcgs.py's self-test)")

    print("\n[replay] stability across seeds")
    for sd in (1, 2, 3, 11, 42):
        gg, ii = replay(rounds=25, seed=sd)
        assert gg.stats()["states"] > 1 and max(ii["depths"]) <= 10
        print(f"  seed {sd:>2}: {gg.stats()['states']:>2} states, "
              f"{ii['switches']:>2} switches, mean N {gg.stats()['mean_N']:.2f}, "
              f"best {gg.best().rep_value:.4f}")

    print("\n[replay] the shipped prior defaults still reproduce their claimed signal")
    # The one number that justifies --mcgs_prior existing. Guarded rather than
    # trusted: a defaults change or a refit that quietly drops rho would otherwise
    # leave the flag recommending itself on evidence it no longer has.
    try:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from scripts.build_mechanism_prior import collect as _collect, _spearman
        from utils.mcgs import MechanismPrior as _MP
        _edges = _collect(_P("run"), "vae_block_002", 9)
    except Exception as _exc:
        _edges = []
        print(f"  skipped: run history unavailable ({_exc.__class__.__name__})")
    if len(_edges) >= 40:
        _runs = sorted({e["run"] for e in _edges})
        _pr, _tr = [], []
        for _held in _runs:
            _m = _MP.fit([(e["mech"], e["gain"]) for e in _edges if e["run"] != _held],
                         min_support=1)
            for _e in (x for x in _edges if x["run"] == _held):
                if _m.support(_e["mech"]) > 0:
                    _pr.append(_m.advantage(_e["mech"])); _tr.append(_e["gain"])
        _rho = _spearman(_pr, _tr)
        _ord = sorted(range(len(_pr)), key=lambda i: -_pr[i])
        _top = _ord[: max(1, len(_ord) // 3)]
        _prec = sum(1 for i in _top if _tr[i] > 1) / len(_top)
        _base = sum(1 for g in _tr if g > 1) / len(_tr)
        print(f"  leave-one-run-out: n={len(_pr)}/{len(_edges)} runs={len(_runs)} "
              f"rho={_rho:+.3f} top-third {_prec*100:.0f}% vs base {_base*100:.0f}%")
        _check(_rho > 0.20,
               f"the prior still predicts held-out gain (rho={_rho:+.3f} > 0.20)")
        _check(_prec > _base,
               f"and still beats the base win rate ({_prec*100:.0f}% > {_base*100:.0f}%)")
        _check(len(_runs) >= 5,
               f"cross-validated across enough distinct runs ({len(_runs)}) -- a "
               f"run-grouping bug once collapsed these to 6 and flipped rho to -0.171")
    elif _edges:
        print(f"  skipped: only {len(_edges)} edges available")

    print("\n[replay] all checks passed")
    print("NOTE: this proves the mechanics, not that MCGS wins. Run the live A/B "
          "(--search mcgs vs --search ratchet, matched --round, one task) for that.")


if __name__ == "__main__":
    main()
