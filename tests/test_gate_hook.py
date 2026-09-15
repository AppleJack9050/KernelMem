"""Checks for the harness gate (agents/query_server Stop hook).

    python -m tests.test_gate_hook

What has to be true for the gate to be worth having:

1. **A kernel that passes the harness check is let through untouched**, and the
   bytes the gate checked are the bytes delivered.
2. **A kernel that fails is rejected with the harness error**, the agent stays
   in its session, and a later passing submission is let through.
3. **The rejection cap holds**: after `gate_refusals` rejections the kernel goes
   through as it is (the official bench then decides), never an unbounded loop.
4. **An environment fault is not charged to the agent**: if the parent that
   already passed also fails now, the stop is allowed and nothing is rejected.
5. **Doing nothing is not a pass**: a submission byte-identical to a known-good
   baseline (the rollout's parent) is nudged once, then let through for the
   round's own no-change handling. A repair handed back unchanged is gated.
6. **The hook never fails the stop**: extractor or gate exceptions allow the stop
   with `gate=error` rather than letting the CLI treat the hook as broken.
7. **Delivery prefers gated bytes**: when the call dies, hits the turn wall or
   ends with no fenced code, the last kernel that PASSED the gate is delivered
   instead of whatever ANSWER.py holds.

No GPU, no agent: the gate and the extractor are fakes, and the SDK call is
replaced by a fake `_run_query` that drives the hook the way the CLI would.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.query_server as qs  # noqa: E402
from utils.kernel_io import extract_code_block  # noqa: E402

GOOD = "import torch\nimport torch.nn as nn\n\n\nclass ModelNew(nn.Module):\n    v = 1\n"
BAD = GOOD.replace("v = 1", "v = 2")
PARENT = GOOD.replace("v = 1", "v = 0")


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


def _fenced(code: str) -> str:
    return f"Section A: done\n\n=== KERNEL CODE STARTS BELOW ===\n```python\n{code}\n```"


def _hook_input(text: str | None, transcript: str = "") -> dict:
    d = {"hook_event_name": "Stop", "stop_hook_active": False, "session_id": "s",
         "transcript_path": transcript, "cwd": "/tmp"}
    if text is not None:
        d["last_assistant_message"] = text
    return d


def _make(verdicts, *, baseline=None, known_good=False, cap=5, extract=None):
    """A hook over a scripted gate: `verdicts` maps code -> ok bool (or raises)."""
    calls = []

    def gate(code: str):
        calls.append(code)
        v = verdicts[code]
        if isinstance(v, Exception):
            raise v
        return {"ok": v, "error_type": None if v else "AccuracyError",
                "message": None if v else "Outputs are not close (max_abs_err 8.4)"}

    state = qs._new_gate_state()
    hook = qs._make_stop_gate(gate=gate, extract=extract, baseline_code=baseline,
                              baseline_known_good=known_good, cap=cap,
                              call_type="optimization", state=state)
    return hook, state, calls


def _run(hook, text, **kw):
    return asyncio.run(hook(_hook_input(text, **kw), None, {"signal": None}))


def main() -> None:
    print("[gate] 1. a passing kernel is let through")
    hook, st, calls = _make({GOOD: True})
    out = _run(hook, _fenced(GOOD))
    _check(out == {}, "no decision means the stop is allowed")
    _check(st["gate"] == "pass" and st["last_pass_code"] == GOOD,
           "the gate recorded the pass and kept the passing bytes")
    _check(st["checks"] == 1 and st["blocks"] == 0, "one check, no rejection")
    _check(calls == [GOOD], "the bytes checked are the bytes in the fenced block")

    print("\n[gate] 2. a failing kernel is rejected with the harness error, then a fix passes")
    hook, st, calls = _make({BAD: False, GOOD: True, PARENT: True},
                            baseline=PARENT, known_good=True)
    out = _run(hook, _fenced(BAD))
    _check(out.get("decision") == "block", "FAIL -> block")
    _check("rejection 1 of 5" in out["reason"] and "max_abs_err 8.4" in out["reason"],
           "the reason names the rejection count and carries the harness error")
    _check(st["blocks"] == 1 and st["last_error_type"] == "AccuracyError",
           "state records the rejection and the error kind")
    _check(st["gate"] == "blocked", "a block labels the state, so a call ended right after it is not 'gate=none'")
    _check(calls == [BAD, PARENT], "the parent was re-checked once to rule out an env fault")
    out = _run(hook, _fenced(GOOD))
    _check(out == {} and st["gate"] == "pass" and st["last_pass_code"] == GOOD,
           "the fixed kernel passes and is let through")
    _check(calls == [BAD, PARENT, GOOD], "the parent verdict is cached, not re-run")

    print("\n[gate] 3. the rejection cap holds")
    hook, st, calls = _make({BAD: False, PARENT: True}, baseline=PARENT,
                            known_good=True, cap=3)
    outs = [_run(hook, _fenced(BAD)) for _ in range(4)]
    _check([o.get("decision") for o in outs] == ["block", "block", "block", None],
           "three rejections, then the fourth submission goes through as it is")
    _check(st["gate"] == "fail_cap" and st["blocks"] == 3, "state says the cap was hit")
    _check(st["last_pass_code"] is None, "nothing passed, so nothing is offered for delivery")
    _check("rejection 3 of 3" in outs[2]["reason"], "the last rejection says it is the last")

    print("\n[gate] 4. an environment fault is not charged to the agent")
    hook, st, calls = _make({BAD: False, PARENT: False}, baseline=PARENT, known_good=True)
    out = _run(hook, _fenced(BAD))
    _check(out == {} and st["gate"] == "env_fault" and st["blocks"] == 0,
           "parent fails too -> allowed, no rejection")
    _check("parent also fails" in (st["error"] or ""), "the state explains why")
    _run(hook, _fenced(BAD))
    _check(calls == [BAD, PARENT, BAD, PARENT],
           "a FAILING parent probe is not remembered: the next FAIL probes again (it may be transient)")
    hook, st, calls = _make({BAD: False}, baseline=PARENT, known_good=False)
    out = _run(hook, _fenced(BAD))
    _check(out.get("decision") == "block" and calls == [BAD],
           "a baseline that is not known-good (repair) is never used as the env probe")

    hook, st, calls = _make({PARENT: False}, baseline=PARENT, known_good=False)
    out = _run(hook, _fenced(PARENT))
    _check(out.get("decision") == "block" and "FAIL" in out["reason"] and calls == [PARENT],
           "a repair returned unchanged is not nudged or waved through: it meets the real check")

    print("\n[gate] 5. doing nothing is not a pass")
    hook, st, calls = _make({PARENT: True}, baseline=PARENT, known_good=True)
    out = _run(hook, _fenced(PARENT))
    _check(out.get("decision") == "block" and "byte-identical" in out["reason"],
           "an unchanged submission is nudged")
    out = _run(hook, _fenced(PARENT + "\n"))
    _check(out == {} and st["gate"] == "nochange" and calls == [],
           "nudged once; the second unchanged submission is let through, gate never run")
    hook, st, calls = _make({PARENT: True}, baseline=PARENT, known_good=True, cap=1)
    _run(hook, "no kernel yet")
    out = _run(hook, _fenced(PARENT))
    _check(out == {} and st["gate"] == "nochange" and st["blocks"] == 1,
           "the nudge respects the cap: with the cap spent, an unchanged submission is not blocked again")

    print("\n[gate] 6. the hook never fails the stop")
    hook, st, calls = _make({GOOD: RuntimeError("bench exploded")})
    out = _run(hook, _fenced(GOOD))
    _check(out == {} and st["gate"] == "error" and "bench exploded" in st["error"],
           "a gate exception allows the stop and is recorded")

    def bad_extract(_text):
        raise RuntimeError("no fence")
    hook, st, calls = _make({GOOD: True}, extract=bad_extract)
    out = _run(hook, _fenced(GOOD))
    _check(out.get("decision") == "block" and "no complete kernel" in out["reason"],
           "an extractor failure reads as 'no kernel in the message' and is rejected")
    hook, st, calls = _make({GOOD: True})
    out = _run(hook, "I am done, see ANSWER.py")
    _check(out.get("decision") == "block" and st["blocks"] == 1,
           "a final message with no fenced code is rejected")
    out = _run(hook, _fenced(GOOD))
    _check(out == {} and st["gate"] == "pass", "...and the next submission is checked normally")
    hook, st, calls = _make({GOOD: True}, cap=1)
    _run(hook, "nothing here")
    out = _run(hook, "still nothing")
    _check(out == {} and st["gate"] == "fail_cap",
           "no-code rejections count toward the cap, so this cannot loop")

    from utils import kernel_io
    dump = kernel_io.ERROR_DUMP_DIR
    before = set(os.listdir(dump)) if dump and os.path.isdir(dump) else None
    hook, st, calls = _make({GOOD: True})
    _run(hook, "I am done, no fence here")
    if before is not None:
        _check(set(os.listdir(dump)) == before,
               "a fence-less message is rejected without dumping an llm_output_error file")

    print("\n[gate] 6b. cap clamp, per-attempt reset, cancellation")
    saved_env = os.environ.pop("CLAUDE_CODE_STOP_HOOK_BLOCK_CAP", None)
    try:
        _check(qs._effective_gate_cap(5) == 5, "the default cap of 5 is untouched")
        _check(qs._effective_gate_cap(9) == 7, "a cap at/above the CLI's 8 is clamped to 7")
        os.environ["CLAUDE_CODE_STOP_HOOK_BLOCK_CAP"] = "3"
        _check(qs._effective_gate_cap(5) == 2, "an inherited CLI cap of 3 clamps to 2")
        os.environ["CLAUDE_CODE_STOP_HOOK_BLOCK_CAP"] = "0"
        _check(qs._effective_gate_cap(9) == 9, "CLI cap 0 means no CLI cap, so no clamp")
    finally:
        os.environ.pop("CLAUDE_CODE_STOP_HOOK_BLOCK_CAP", None)
        if saved_env is not None:
            os.environ["CLAUDE_CODE_STOP_HOOK_BLOCK_CAP"] = saved_env

    st = qs._new_gate_state()
    st.update(checks=4, blocks=3, nudged=True, gate="fail_cap", last_pass_code=GOOD,
              parent_verdict={"ok": True})
    qs._reset_gate_attempt(st)
    _check(st["blocks"] == 0 and st["nudged"] is False and st["gate"] == "none",
           "a retried attempt starts with the full cap and a fresh nudge")
    _check(st["checks"] == 4 and st["last_pass_code"] == GOOD and st["parent_verdict"] == {"ok": True},
           "checks stay cumulative; a pass and a parent pass from attempt 1 survive")

    import threading
    import time as _time
    release = threading.Event()

    def slow_gate(code):
        release.wait(5)
        return {"ok": True}

    st = qs._new_gate_state()
    hook = qs._make_stop_gate(gate=slow_gate, extract=None, baseline_code=None,
                              baseline_known_good=False, cap=5, call_type="seed", state=st)

    async def _cancel_mid_gate():
        t = asyncio.ensure_future(hook(_hook_input(_fenced(GOOD)), None, {"signal": None}))
        await asyncio.sleep(0.1)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            return True
        return False
    was_cancelled = asyncio.run(_cancel_mid_gate())
    release.set()
    _check(was_cancelled and st["gate"] == "cancelled",
           "a hook cancelled by the CLI mid-gate re-raises and records gate=cancelled")

    print("\n[gate] 7. the final message is read from the transcript when the CLI omits it")
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d) / "t.jsonl"
        recs = [{"type": "user", "message": {"content": "hi"}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "draft"}]}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": _fenced(GOOD)}]}}]
        tp.write_text("\n".join(json.dumps(r) for r in recs))
        hook, st, calls = _make({GOOD: True})
        out = _run(hook, None, transcript=str(tp))
        _check(out == {} and calls == [GOOD], "the last assistant entry of the transcript is gated")
        recs.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}})
        tp.write_text("\n".join(json.dumps(r) for r in recs))
        hook, st, calls = _make({GOOD: True})
        out = _run(hook, None, transcript=str(tp))
        _check(out.get("decision") == "block" and calls == [],
               "a last entry with no text is not replaced by an older draft")

    print("\n[gate] 8. query_server installs the hook and delivers gated bytes")
    seen = {}

    def _result(*, is_error=False, subtype="success", num_turns=4):
        return types.SimpleNamespace(is_error=is_error, subtype=subtype, result="",
                                     usage=None, stop_reason="end_turn",
                                     num_turns=num_turns, duration_ms=1234)

    def fake_gate(code):
        ok = code in (GOOD, PARENT)     # the parent passed the harness before
        return {"ok": ok, "error_type": None if ok else "AccuracyError",
                "message": None if ok else "mismatch"}

    async def _drive(hook, *submissions):
        # Inside the fake SDK call we are already on the event loop, so the hook
        # is awaited the way the SDK awaits it -- not run in a nested loop.
        outs = []
        for s in submissions:
            outs.append(await hook(_hook_input(s), None, {"signal": None}))
        return outs

    real_run_query = qs._run_query
    try:
        # (a) reject once, then pass; the reply is delivered as-is
        async def fake_a(prompt_text, options, final_only=False):
            seen["options"] = options
            hook = options.hooks["Stop"][0].hooks[0]
            seen["outs"] = await _drive(hook, _fenced(BAD), _fenced(GOOD))
            return [_fenced(GOOD)], _result()
        qs._run_query = fake_a
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "usage.csv")
            out = qs.query_server("p", system_prompt="sys", call_type="optimization",
                                  round_idx=2, log_path=log, baseline_code=PARENT,
                                  gate=fake_gate, gate_refusals=3,
                                  baseline_known_good=True)
            opts = seen["options"]
            _check(opts.hooks and "Stop" in opts.hooks and len(opts.hooks["Stop"][0].hooks) == 1,
                   "a Stop hook is installed on the tool-mode call")
            _check(opts.hooks["Stop"][0].timeout == qs._GATE_HOOK_TIMEOUT_S,
                   "with the gate's own timeout, not the SDK default of 60 s")
            _check("HARNESS GATE" in opts.system_prompt and "3 rejections" in opts.system_prompt,
                   "the agent is told about the gate and its cap")
            _check([o.get("decision") for o in seen["outs"]] == ["block", None],
                   "the fake CLI saw one rejection then an allow")
            _check(extract_code_block(out).strip() == GOOD.strip(),
                   "the delivered reply is the passing kernel")
            rows = list(csv.DictReader(open(os.path.join(d, "calls.csv"))))
            _check(rows[-1]["outcome"] == "ok" and rows[-1]["salvaged"] == "none",
                   "calls.csv: a clean delivery")
            _check("gate=pass" in rows[-1]["detail"] and "blocks=1" in rows[-1]["detail"],
                   f"calls.csv detail carries the gate summary ({rows[-1]['detail']})")

        # (b) the turn wall after a pass: gated bytes are delivered, not ANSWER.py
        async def fake_b(prompt_text, options, final_only=False):
            hook = options.hooks["Stop"][0].hooks[0]
            await _drive(hook, _fenced(GOOD))
            raise Exception("Claude Code returned an error result: reached maximum number of turns")
        qs._run_query = fake_b
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "usage.csv")
            out = qs.query_server("p", system_prompt="sys", call_type="optimization",
                                  round_idx=3, log_path=log, baseline_code=PARENT,
                                  gate=fake_gate, gate_refusals=3, baseline_known_good=True)
            _check(extract_code_block(out).strip() == GOOD.strip(),
                   "the last PASSED kernel is delivered after the turn wall")
            rows = list(csv.DictReader(open(os.path.join(d, "calls.csv"))))
            _check(rows[-1]["outcome"] == "max_turns" and rows[-1]["salvaged"] == "gated",
                   f"calls.csv: outcome max_turns, salvaged=gated ({rows[-1]['outcome']}, {rows[-1]['salvaged']})")

        # (c) a final message with no fence after a pass: gated bytes are delivered
        async def fake_c(prompt_text, options, final_only=False):
            hook = options.hooks["Stop"][0].hooks[0]
            await _drive(hook, _fenced(GOOD))
            return ["All done; the kernel is in ANSWER.py."], _result()
        qs._run_query = fake_c
        with tempfile.TemporaryDirectory() as d:
            out = qs.query_server("p", system_prompt="sys", call_type="optimization",
                                  round_idx=4, log_path=os.path.join(d, "usage.csv"),
                                  baseline_code=PARENT, gate=fake_gate, gate_refusals=3,
                                  baseline_known_good=True)
            _check(extract_code_block(out).strip() == GOOD.strip(),
                   "a fence-less final message still delivers the gated kernel")

        # (e) attempt 1 spends the cap then the CLI connection dies; attempt 2 gets the full cap
        from claude_agent_sdk import CLIConnectionError
        attempts = {"n": 0}

        async def fake_e(prompt_text, options, final_only=False):
            attempts["n"] += 1
            hook = options.hooks["Stop"][0].hooks[0]
            if attempts["n"] == 1:
                seen["e1"] = await _drive(hook, _fenced(BAD), _fenced(BAD))
                raise CLIConnectionError("connection dropped")
            seen["e2"] = await _drive(hook, _fenced(BAD), _fenced(GOOD))
            return [_fenced(GOOD)], _result()
        qs._run_query = fake_e
        with tempfile.TemporaryDirectory() as d:
            out = qs.query_server("p", system_prompt="sys", call_type="optimization",
                                  round_idx=7, log_path=os.path.join(d, "usage.csv"),
                                  baseline_code=PARENT, gate=fake_gate, gate_refusals=2,
                                  baseline_known_good=True)
            _check([o.get("decision") for o in seen["e1"]] == ["block", "block"],
                   "attempt 1 used both rejections")
            _check([o.get("decision") for o in seen["e2"]] == ["block", None],
                   "attempt 2 rejects again instead of waving the failing kernel through at the cap")
            rows = list(csv.DictReader(open(os.path.join(d, "calls.csv"))))
            _check("gate=pass" in rows[-1]["detail"] and "checks=4" in rows[-1]["detail"],
                   f"checks are cumulative across attempts ({rows[-1]['detail']})")

        # (f) cap spent on fence-less messages, nothing passed: ANSWER.py is scored
        async def fake_f(prompt_text, options, final_only=False):
            hook = options.hooks["Stop"][0].hooks[0]
            seen["f"] = await _drive(hook, "working on it", "done, see ANSWER.py")
            return ["done, see ANSWER.py"], _result()
        qs._run_query = fake_f
        with tempfile.TemporaryDirectory() as d:
            out = qs.query_server("p", system_prompt="sys", call_type="optimization",
                                  round_idx=8, log_path=os.path.join(d, "usage.csv"),
                                  baseline_code=PARENT, gate=fake_gate, gate_refusals=1,
                                  baseline_known_good=True)
            _check([o.get("decision") for o in seen["f"]] == ["block", None],
                   "one no-code rejection, then allowed at the cap")
            _check(extract_code_block(out).strip() == PARENT.strip(),
                   "the fence-less reply is replaced by ANSWER.py instead of failing the round")
            rows = list(csv.DictReader(open(os.path.join(d, "calls.csv"))))
            _check(rows[-1]["salvaged"] == "parent" and "gate=fail_cap" in rows[-1]["detail"],
                   f"calls.csv says it was the unchanged parent at the cap ({rows[-1]['salvaged']}, {rows[-1]['detail']})")

        # (d) no gate -> no hook, no instruction
        async def fake_d(prompt_text, options, final_only=False):
            seen["options_nogate"] = options
            return [_fenced(GOOD)], _result()
        qs._run_query = fake_d
        with tempfile.TemporaryDirectory() as d:
            qs.query_server("p", system_prompt="sys", call_type="optimization", round_idx=5,
                            log_path=os.path.join(d, "usage.csv"), baseline_code=PARENT)
            o = seen["options_nogate"]
            _check(not o.hooks and "HARNESS GATE" not in o.system_prompt,
                   "without a gate the call is exactly as before")
            qs.query_server("p", system_prompt="sys", call_type="optimization", round_idx=6,
                            log_path=os.path.join(d, "usage.csv"), baseline_code=PARENT,
                            gate=fake_gate, gate_refusals=0)
            _check(not seen["options_nogate"].hooks, "gate_refusals=0 disables the hook")
    finally:
        qs._run_query = real_run_query

    print("\nall gate checks passed")


def _main_isolated() -> None:
    """Run main() with extract_code_block's llm_output_error_*.txt dumps sent to a
    temp dir. Several checks feed the extractor fence-less text on purpose, and
    left at the default the dumps land in the working directory (the repo root)."""
    from utils import kernel_io
    prev = kernel_io.ERROR_DUMP_DIR
    with tempfile.TemporaryDirectory() as dump:
        kernel_io.set_error_dump_dir(dump)
        try:
            main()
        finally:
            kernel_io.set_error_dump_dir(prev)


def test_gate_hook() -> None:
    _main_isolated()


if __name__ == "__main__":
    _main_isolated()
