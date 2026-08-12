#!/usr/bin/env python
"""Checks for the memory-bank enforcement switch.

Run:  python -m utils.test_enforcement_mode

The S4 cases added to the ``memory_bandwidth`` bucket give roughly 109 of the
recorded vae_block_002 round evaluations a non-empty ``allowed_methods`` where
they previously got NO_MATCH. Under the original prompts that list binds, and
86% of the methods those runs actually selected -- including the
``stream_pipeline_overlap`` behind the best kernel to date -- are not in it.

Advisory mode exists so the table can be populated and measured without
forbidding anything first. These checks cover the two ways that could silently
fail: a prompt that still binds while claiming to be advisory, and a rewrite
that drifts out of sync with the templates and quietly does nothing.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Phrases that mean the judge is being told it cannot leave the table.
_BINDING_MARKERS = (
    "is a HARD CONSTRAINT",
    "You CANNOT choose a method outside this list",
    "You CANNOT choose a method that is not in that list",
    "MUST be one of the allowed_methods",
    "MUST select your method from",
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _load_judge(mode: str) -> Any:
    """Import the prompt builder fresh under a given enforcement mode."""
    previous = os.environ.get("MEMORYBANK_ENFORCEMENT")
    os.environ["MEMORYBANK_ENFORCEMENT"] = mode
    try:
        sys.modules.pop("prompts.judger_optimization_memory_latest", None)
        return importlib.import_module("prompts.judger_optimization_memory_latest")
    finally:
        if previous is None:
            os.environ.pop("MEMORYBANK_ENFORCEMENT", None)
        else:
            os.environ["MEMORYBANK_ENFORCEMENT"] = previous


def _rendered(mod: Any) -> str:
    """The system prompt as the matched path renders it."""
    return mod._apply_enforcement_mode(
        mod.system_prompt_tmpl.substitute(), mod._ADVISORY_REWRITES_SYSTEM
    )


def test_default_is_advisory() -> None:
    """An unset env var must not bind, so populating the table stays safe."""
    previous = os.environ.pop("MEMORYBANK_ENFORCEMENT", None)
    try:
        sys.modules.pop("prompts.judger_optimization_memory_latest", None)
        mod = importlib.import_module("prompts.judger_optimization_memory_latest")
        _check(
            mod.ENFORCEMENT_MODE == "advisory",
            f"default enforcement is {mod.ENFORCEMENT_MODE!r}, expected 'advisory'; "
            "the S4 cases would bind method selection on the next run",
        )
    finally:
        if previous is not None:
            os.environ["MEMORYBANK_ENFORCEMENT"] = previous
    print("PASS  default enforcement mode is advisory")


def test_hard_mode_leaves_prompts_untouched() -> None:
    """hard must reproduce the original prompt byte-for-byte."""
    mod = _load_judge("hard")
    raw = mod.system_prompt_tmpl.substitute()
    _check(_rendered(mod) == raw, "hard mode altered the system prompt")
    bound = [m for m in _BINDING_MARKERS if m in raw]
    _check(bound, f"hard mode prompt contains none of the binding markers {_BINDING_MARKERS}")
    print(f"PASS  hard mode leaves the prompt unchanged ({len(bound)} binding clauses intact)")


def test_advisory_removes_every_binding_clause() -> None:
    """No binding phrasing may survive in advisory mode.

    One missed clause is the whole failure mode: the judge would keep refusing
    anything outside the table while the run is labelled advisory, and the
    comparison would silently be of the table against itself.
    """
    mod = _load_judge("advisory")
    rendered = _rendered(mod)
    survivors = [m for m in _BINDING_MARKERS if m in rendered]
    _check(not survivors, f"binding language survived advisory rewrite: {survivors}")
    _check(
        "ADVISORY" in rendered,
        "advisory prompt never tells the judge the list is advisory",
    )
    _check(
        "MAY choose a method outside the list" in rendered,
        "advisory prompt does not grant permission to leave the list",
    )
    print(f"PASS  advisory mode clears all {len(_BINDING_MARKERS)} binding markers")


def test_rewrites_fail_loudly_when_out_of_sync() -> None:
    """A rewrite that no longer matches must raise, not quietly no-op.

    If the templates are edited and a clause stops matching, silently skipping
    it would leave a binding prompt wearing an advisory label.
    """
    mod = _load_judge("advisory")
    try:
        mod._apply_enforcement_mode("a prompt with none of the clauses", mod._ADVISORY_REWRITES_SYSTEM)
    except RuntimeError as exc:
        _check("advisory rewrite failed" in str(exc), f"unexpected error text: {exc}")
        print("PASS  a stale rewrite raises instead of silently leaving the prompt binding")
        return
    raise AssertionError("_apply_enforcement_mode silently accepted a prompt missing every clause")


def test_invalid_mode_is_rejected() -> None:
    """A typo must not fall back to binding behaviour."""
    try:
        _load_judge("advisroy")
    except ValueError as exc:
        _check("MEMORYBANK_ENFORCEMENT" in str(exc), f"unexpected error text: {exc}")
        print("PASS  an unrecognised enforcement mode is rejected at import")
        return
    raise AssertionError("an invalid MEMORYBANK_ENFORCEMENT value was accepted")


def test_instruction_clause_is_softened() -> None:
    """The instruction carries its own binding clause; it must soften too."""
    mod = _load_judge("advisory")
    old, new = mod._ADVISORY_REWRITES_INSTRUCTION[0]
    softened = mod._apply_enforcement_mode(
        f"preamble\n{old}\ntrailer", mod._ADVISORY_REWRITES_INSTRUCTION
    )
    _check(old not in softened, "instruction still carries its binding clause")
    _check(new in softened, "instruction was not given the advisory clause")
    print("PASS  the instruction's binding clause is softened as well")


def main() -> None:
    test_default_is_advisory()
    test_hard_mode_leaves_prompts_untouched()
    test_advisory_removes_every_binding_clause()
    test_rewrites_fail_loudly_when_out_of_sync()
    test_invalid_mode_is_rejected()
    test_instruction_clause_is_softened()
    print("\nAll enforcement-mode checks passed.")


if __name__ == "__main__":
    main()
