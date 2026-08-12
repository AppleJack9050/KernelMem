#!/usr/bin/env python
"""Checks for the S4 gap in the ``memory_bandwidth`` decision-table bucket.

Run:  python -m utils.test_machine_check_s4

On vae_block_002 the measured bottleneck is permanently ``memory_bandwidth``
(a priority rule forces it whenever dram or l2 >= 80) and the kernel structure
is permanently ``S4`` (a fused resblock runs several kernels per forward).
That pair had no case in the table, so ``lookup_case`` returned NO_MATCH and
``allowed_methods`` came back empty for most of the run.

These checks rest on replaying the 168 machine_check results recorded under
``run/`` back through the real ``lookup_case``. The replay is only meaningful
if it is faithful, so the first test reproduces every recorded ``case_id``
before any of the others are allowed to draw conclusions from it.

``test_allowed_methods_vs_history`` is deliberately not an assertion about the
fix being good. It records that the table's method vocabulary and the methods
these runs actually used are nearly disjoint, which is the reason the S4 cases
are landed with the constraint still inert.
"""
from __future__ import annotations

import collections
import glob
import importlib.util
import json
import os
from typing import Any, Dict, List, Tuple

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YAML = os.path.join(_ROOT, "memorybank", "bottleneck_headroom_kernelstructure.yaml")
_MC_PY = os.path.join(_ROOT, "prompts", "machine_check_ver2.py")

# The three-predicate signature guarding the memory_bandwidth bucket. A case
# added to that bucket is unreachable for any round that does not match all
# three, so the expected coverage below is a subset of the forced rounds.
_MBW_SIGNATURE = ("high_dram_or_l2_throughput", "low_sm_throughput", "often_high_occupancy")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _load_lookup_case():
    spec = importlib.util.spec_from_file_location("machine_check_ver2", _MC_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.lookup_case


def _load_rules() -> dict:
    with open(_YAML) as fh:
        return yaml.safe_load(fh)


def _load_rounds() -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Recorded machine_check results under run/, deduped by identity.

    Returns the distinct results plus how many times each was seen. Round 1 of
    every run starts from the same base kernel and so produces byte-identical
    output; those repeats are real round evaluations but not distinct decision
    situations, and collapsing them keeps one situation from being asserted on
    several times. Both counts are reported so neither reading is hidden.
    """
    weight: collections.Counter = collections.Counter()
    by_key: Dict[str, Dict[str, Any]] = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "case_id" in obj and "matched_predicates" in obj:
                key = json.dumps(obj, sort_keys=True, default=str)
                by_key.setdefault(key, obj)
                weight[key] += 1
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    for path in sorted(set(glob.glob(os.path.join(_ROOT, "run", "**", "*.json"), recursive=True))):
        try:
            with open(path) as fh:
                walk(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue

    keys = list(by_key)
    return [by_key[k] for k in keys], collections.Counter({i: weight[k] for i, k in enumerate(keys)})


def _env_for(round_: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the gate environment from what the round recorded."""
    env = dict(round_.get("code_features_used") or {})
    env.update({k: v for k, v in (round_.get("key_metrics") or {}).items() if v is not None})
    env["kernel_structure"] = round_.get("kernel_structure")
    # reuse_possible is a derived field that several gate_when expressions read.
    env["reuse_possible"] = bool(
        env.get("has_reuse") or env.get("is_naive_gemm")
        or env.get("is_gemm_kloop") or env.get("is_stencil_conv")
    )
    return env


def _replay(rules: dict, rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup_case = _load_lookup_case()
    return [
        lookup_case(
            rules,
            r.get("forced_bottleneck"),
            r.get("tier"),
            r.get("kernel_structure", "S0"),
            r.get("matched_predicates", []),
            _env_for(r),
        )
        for r in rounds
    ]


# ---------------------------------------------------------------------------


def _rules_without_s4_cases() -> dict:
    """The decision table as it stood before the S4 cases were added."""
    rules = _load_rules()
    bucket = rules["machine_check"]["decision_table"]["memory_bandwidth"]
    bucket["cases"] = [
        c for c in bucket["cases"]
        if "S4" not in (c["kernel_structure"] if isinstance(c["kernel_structure"], list)
                        else [c["kernel_structure"]])
    ]
    return rules


def test_replay_is_faithful() -> None:
    """The harness must reproduce every recorded case_id before it is trusted.

    Everything downstream compares the patched table against this replay, so a
    replay that diverges from what the runs actually recorded would make the
    coverage numbers fiction. Replaying with the S4 cases stripped back out
    also pins them as the only behavioural change to the table.
    """
    rounds, weight = _load_rounds()
    _check(len(rounds) > 0, "no machine_check results found under run/ -- nothing to verify")
    replayed = _replay(_rules_without_s4_cases(), rounds)
    mismatches = [
        (i, r["case_id"], got["case_id"])
        for i, (r, got) in enumerate(zip(rounds, replayed))
        if r["case_id"] != got["case_id"]
    ]
    _check(
        not mismatches,
        f"replay diverged from recorded outcomes on {len(mismatches)} of {len(rounds)} rounds; "
        f"first three: {mismatches[:3]}",
    )
    print(f"PASS  replay reproduces all {len(rounds)} distinct recorded case_ids exactly "
          f"({sum(weight.values())} round evaluations)")


def test_memory_bandwidth_covers_s4() -> None:
    """The forced bucket must have a case for the structure it is forced onto."""
    bucket = _load_rules()["machine_check"]["decision_table"]["memory_bandwidth"]
    structures = set()
    for case in bucket.get("cases", []):
        ks = case["kernel_structure"]
        structures.update(ks if isinstance(ks, list) else [ks])
    _check(
        "S4" in structures,
        f"memory_bandwidth has no S4 case (covers {sorted(structures)}); a multi-kernel "
        "forward forced onto this bucket lands in an empty cell every round",
    )
    print(f"PASS  memory_bandwidth covers S4 (structures: {sorted(structures)})")


def test_s4_rounds_stop_returning_no_match() -> None:
    """Forced S4 rounds that match the bucket signature must resolve to a case."""
    rounds, weight = _load_rounds()
    replayed = _replay(_load_rules(), rounds)

    reachable = [
        (i, got) for i, (r, got) in enumerate(zip(rounds, replayed))
        if r.get("forced_bottleneck") == "memory_bandwidth"
        and r.get("kernel_structure") == "S4"
        and all(p in set(r.get("matched_predicates") or []) for p in _MBW_SIGNATURE)
    ]
    _check(reachable, "no forced S4 rounds match the memory_bandwidth signature -- "
                      "the fixture no longer exercises this path")
    unresolved = [i for i, got in reachable if got["case_id"] == "NO_MATCH"]
    _check(
        not unresolved,
        f"{len(unresolved)} of {len(reachable)} signature-matching forced S4 situations still "
        f"return NO_MATCH ({sum(weight[i] for i in unresolved)} round evaluations)",
    )
    resolved = collections.Counter(got["case_id"] for _, got in reachable)
    print(f"PASS  all {len(reachable)} signature-matching forced S4 situations resolve "
          f"({sum(weight[i] for i, _ in reachable)} round evaluations): {dict(resolved)}")


def test_no_regression_on_previously_matched_rounds() -> None:
    """Rounds that already matched must keep the exact case they had.

    The new cases are appended to a bucket evaluated first-match-wins, so this
    is what rules out an ordering change silently rerouting existing matches.
    """
    rounds, weight = _load_rounds()
    replayed = _replay(_load_rules(), rounds)
    changed = [
        (r["case_id"], got["case_id"])
        for r, got in zip(rounds, replayed)
        if r["case_id"] != "NO_MATCH" and r["case_id"] != got["case_id"]
    ]
    _check(not changed, f"previously-matched rounds were rerouted: {collections.Counter(changed)}")
    kept = sum(1 for r in rounds if r["case_id"] != "NO_MATCH")
    print(f"PASS  all {kept} previously-matched rounds keep their original case")


def test_tier_l_is_unreachable_for_s4() -> None:
    """An S4 case must not claim Tier-L, because S4 can never be tiered Tier-L.

    Tier-L requires ``bytes_structurally_unavoidable``, which is defined as
    ``kernel_structure_id = 0 AND ...``. Any Tier-L entry on an S4 case is
    dead weight that reads as coverage.
    """
    rules = _load_rules()
    derived = rules["machine_check"]["input_normalization"]["derived_fields"]
    unavoidable = str(derived.get("bytes_structurally_unavoidable", ""))
    _check(
        "kernel_structure_id = 0" in unavoidable,
        f"bytes_structurally_unavoidable no longer pins kernel_structure_id to 0 "
        f"({unavoidable!r}); the Tier-L reasoning below needs rechecking",
    )
    dead = []
    for name, bucket in rules["machine_check"]["decision_table"].items():
        for case in bucket.get("cases", []):
            ks = case["kernel_structure"]
            ks = ks if isinstance(ks, list) else [ks]
            headroom = case["headroom"]
            headroom = headroom if isinstance(headroom, list) else [headroom]
            if "S4" in ks and "Tier-L" in headroom:
                dead.append(f"{name}/{case.get('id')}")

    # A superfluous tier in a headroom list is inert -- it simply never
    # matches -- so the pre-existing ones are reported rather than failed.
    # The cases added here must not add to the pile.
    added = [d for d in dead if d.startswith("memory_bandwidth/MBW_S4_")]
    _check(not added, f"newly-added S4 cases claim unreachable Tier-L: {added}")
    if dead:
        print(f"NOTE  {len(dead)} pre-existing S4 cases list an unreachable Tier-L "
              f"(inert today; would activate if bytes_structurally_unavoidable ever "
              f"covers S4): {', '.join(dead)}")
    print("PASS  no newly-added S4 case claims Tier-L")


def test_allowed_methods_vs_history() -> None:
    """Record how far the table's vocabulary is from what the runs actually used.

    This does not assert the table is right. It asserts we still know the gap
    is there, so that turning the constraint on stays a deliberate decision.
    """
    rounds, weight = _load_rounds()
    replayed = _replay(_load_rules(), rounds)
    offered = set()
    for got in replayed:
        offered.update(got.get("allowed_methods") or [])

    used: collections.Counter = collections.Counter()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            name = obj.get("method_name")
            if isinstance(name, str):
                used[name] += 1
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    for path in sorted(set(glob.glob(os.path.join(_ROOT, "run", "**", "*.json"), recursive=True))):
        try:
            with open(path) as fh:
                walk(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue

    _check(used, "no method_name values recorded under run/ -- cannot compare vocabularies")
    forbidden = {m: n for m, n in used.items() if m not in offered}
    share = sum(forbidden.values()) / sum(used.values())
    top = ", ".join(f"{m}({n})" for m, n in sorted(forbidden.items(), key=lambda kv: -kv[1])[:4])
    print(
        f"NOTE  {len(forbidden)} of {len(used)} methods used in these runs are absent from the "
        f"cases that now match ({share:.0%} of selections): {top}"
    )
    _check(
        share > 0.0,
        "every method the runs used is now offered -- the constraint is no longer inert, "
        "so re-read the caveat in this module's docstring before relying on it",
    )


def main() -> None:
    test_replay_is_faithful()
    test_memory_bandwidth_covers_s4()
    test_s4_rounds_stop_returning_no_match()
    test_no_regression_on_previously_matched_rounds()
    test_tier_l_is_unreachable_for_s4()
    test_allowed_methods_vs_history()
    print("\nAll machine_check S4 checks passed.")


if __name__ == "__main__":
    main()
