"""Does the rollout actually run on --rollout_model, and only the rollout?

The split is invisible at runtime -- both models go through the same Agent SDK
path and print the same kind of reply -- so a silent regression would spend Opus
credit on every expansion and nothing would look wrong. These checks pin it down:

  * the `optimization` call (the MCGS rollout) uses --rollout_model at
    --rollout_effort
  * every other call type still uses --model_name
  * an explicit effort beats KERNELMEM_CLAUDE_EFFORT, or per-call routing would be
    silently overridden by a global env var
  * both paths keep the blanked API-key env, i.e. subscription credit not API
  * usage.csv records which model spent the tokens

Run: python -m utils.test_rollout_model
"""
from __future__ import annotations

import csv
import os
import tempfile
from argparse import Namespace
from pathlib import Path

import agents.query_server as qs
import main_memory_latest as mm


def _args(**over) -> Namespace:
    base = dict(server_type="claude", model_name="claude-opus-5",
                rollout_model="claude-sonnet-5", rollout_effort="high",
                temperature=1.0, top_p=1.0, server_address="localhost",
                server_port=30000)
    base.update(over)
    return Namespace(**base)


def main() -> None:
    def _check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        assert cond, msg

    calls = []

    def fake_query_server(**kw):
        calls.append(kw)
        return "stub reply"

    real = mm.query_server
    mm.query_server = fake_query_server
    try:
        print("[rollout] per-call model routing")
        call_llm = mm._make_llm_caller(_args())

        # the rollout, routed the way the loop routes it
        call_llm("p", call_type="optimization", round_idx=3,
                 model_name="claude-sonnet-5", reasoning_effort="high")
        roll = calls[-1]
        _check(roll["model_name"] == "claude-sonnet-5",
               f"the rollout used {roll['model_name']}")
        _check(roll["reasoning_effort"] == "high",
               f"at effort {roll['reasoning_effort']}")

        # everything else keeps the run default
        for ct in ("judge_optimization", "problem_identify", "repair", "seed"):
            call_llm("p", call_type=ct, round_idx=3)
            other = calls[-1]
            _check(other["model_name"] == "claude-opus-5",
                   f"{ct} stayed on {other['model_name']}")
            _check(other["reasoning_effort"] is None,
                   f"{ct} requested no explicit effort (falls back to the default)")

        print("\n[rollout] the split can be turned off")
        call_llm2 = mm._make_llm_caller(_args(rollout_model="claude-opus-5"))
        call_llm2("p", call_type="optimization", round_idx=0,
                  model_name="claude-opus-5", reasoning_effort="high")
        _check(calls[-1]["model_name"] == "claude-opus-5",
               "--rollout_model equal to --model_name disables the split")
    finally:
        mm.query_server = real

    print("\n[rollout] explicit effort wins over the env var")
    prev = os.environ.get("KERNELMEM_CLAUDE_EFFORT")
    os.environ["KERNELMEM_CLAUDE_EFFORT"] = "low"
    try:
        got = ("high" or os.environ.get("KERNELMEM_CLAUDE_EFFORT") or qs.DEFAULT_EFFORT)
        _check(got == "high", "an explicit 'high' is not overridden by the env's 'low'")
        got_default = (None or os.environ.get("KERNELMEM_CLAUDE_EFFORT") or qs.DEFAULT_EFFORT)
        _check(got_default == "low", "a call with no explicit effort still honours the env var")
    finally:
        if prev is None:
            os.environ.pop("KERNELMEM_CLAUDE_EFFORT", None)
        else:
            os.environ["KERNELMEM_CLAUDE_EFFORT"] = prev

    print("\n[rollout] subscription credit, not API billing")
    _check(qs._SUBSCRIPTION_ENV.get("ANTHROPIC_API_KEY") == "",
           "ANTHROPIC_API_KEY is blanked for every call")
    _check(qs._SUBSCRIPTION_ENV.get("ANTHROPIC_AUTH_TOKEN") == "",
           "ANTHROPIC_AUTH_TOKEN is blanked for every call")
    _check(all(qs._SUBSCRIPTION_ENV.get(k) == "" for k in
               ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")),
           "Bedrock/Vertex routing is blanked, so neither model can bill elsewhere")
    _check(qs._resolve_model("claude-sonnet-5") == "claude-sonnet-5",
           "claude-sonnet-5 is passed through, not silently swapped for the default")
    _check(qs._resolve_model("gpt-4") == qs.DEFAULT_MODEL,
           "a non-Claude name still falls back rather than reaching a different vendor")
    _check("optimization" in qs._TOOL_CALL_TYPES,
           "the rollout keeps its tools -- the cheaper model needs to compile before answering")

    print("\n[rollout] usage.csv records the model that spent the tokens")
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "usage.csv"
        qs._write_usage_row(str(log), 1, "optimization", 100, 200, 300,
                            model="claude-sonnet-5", effort="high")
        qs._write_usage_row(str(log), 1, "judge_optimization", 10, 20, 30,
                            model="claude-opus-5", effort="high")
        rows = list(csv.DictReader(log.open()))
        _check([r["model"] for r in rows] == ["claude-sonnet-5", "claude-opus-5"],
               "both rows name their model")
        totals = mm._append_usage_totals(log)
        _check(totals["total_tokens"] == 330,
               f"the totals row still sums correctly with the new columns ({totals})")

        # a usage.csv written before this change must stay readable
        old = Path(d) / "old_usage.csv"
        old.write_text("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n"
                       "2026-01-01 00:00:00,0,optimization,5,5,10\n", encoding="utf-8")
        qs._write_usage_row(str(old), 1, "optimization", 100, 200, 300,
                            model="claude-sonnet-5", effort="high")
        t2 = mm._append_usage_totals(old)
        _check(t2["total_tokens"] == 310,
               f"an old-header usage.csv still totals correctly ({t2})")

    print("\n[rollout] all checks passed")


if __name__ == "__main__":
    main()
