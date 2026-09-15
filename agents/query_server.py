"""LLM backend for KernelMem, routed through the Claude Agent SDK.

All calls spawn the locally logged-in Claude Code CLI (`claude auth login`,
claude.ai account) so usage bills against the user's Claude subscription
credit — never a pay-per-token API key.
"""

import ast
import asyncio
import datetime
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, Optional, Tuple

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    CLIJSONDecodeError,
    HookMatcher,
    ResultMessage,
    TextBlock,
    query,
)

from utils.kernel_io import extract_code_block

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"

# ---------------------------------------------------------------------------
# Tool-enabled generation.
#
# Kernel authoring was one-shot and blind: tools=[], max_turns=1, so the model
# wrote a whole CUDA extension with no way to compile it and no feedback until
# the NEXT round's profile, a full round later. That is survivable for an
# elementwise kernel and close to hopeless for Hopper warpgroup MMA, where
# shared-memory descriptors, swizzle layouts and fence/commit/wait ordering are
# fixed by iterating, not by reasoning. Letting the writing calls compile and
# fix in place is the difference between "cannot express" and "can express".
#
# Only the calls that WRITE CUDA get tools. The judge and problem-identify calls
# analyse text and would gain nothing but latency and a wider blast radius.
# ---------------------------------------------------------------------------
_TOOL_CALL_TYPES = {"seed", "optimization", "repair"}
_AGENT_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

# ---------------------------------------------------------------------------
# External Claude skills (opt-in: KERNELMEM_SKILLS=1)
#
# Installed by utils/install_kda_skills into ~/.claude/skills, which is the only
# place the SDK looks once `setting_sources` includes "user" -- project scope
# would resolve against the per-call temp `cwd`, which does not exist yet and is
# deleted afterwards.
#
# OFF by default, for two reasons rather than caution alone:
#   1. The skills target B200/sm_100 (and Hopper sm_90). This repo benchmarks on
#      an RTX 5090 = sm_120, where tcgen05 does not exist and the B200 metric
#      names are not all valid. Loading them changes what the model proposes, so
#      it is a search-policy change and wants its own A/B, exactly as
#      --mcts_prior does.
#   2. Loading a skill costs turns from the 30-turn tool budget.
# ---------------------------------------------------------------------------
_SKILLS_ENV = "KERNELMEM_SKILLS"

# Appended only when skills are actually loaded. The correction is the point: the
# skills state their target as B200 in their own text, so without this the model
# reads authoritative-sounding sm_100 advice and applies it to sm_120.
_SKILLS_INSTRUCTION = """

SKILLS ARE AVAILABLE THIS CALL. `ncu-report-skill` (profiling method) and
`KernelWiki` (kernel technique reference) are loadable via the Skill tool.
HARDWARE CORRECTION, which overrides anything they say: those skills were written
for NVIDIA B200 / sm_100 and Hopper / sm_90. This benchmark runs on an RTX 5090
= GB202 = sm_120, consumer Blackwell. `tcgen05` MMA does not exist on sm_120 --
never propose it. B200 metric names and roofline constants in those documents
belong to another card; re-derive any number for sm_120 before relying on it.
Take their METHOD (profile, diagnose, then plan) and discard their constants.
"""


def _skills_enabled() -> bool:
    return os.environ.get(_SKILLS_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _skills_config():
    """(skills, setting_sources, tools) for the options, or (None, None, base)."""
    if not _skills_enabled():
        return None, None, _AGENT_TOOLS
    names = [n.strip() for n in os.environ.get("KERNELMEM_SKILL_NAMES", "").split(",") if n.strip()]
    # "Skill" must be allowed explicitly: allowed_tools is an allowlist, so
    # without it the skills load and then cannot be invoked.
    return (names or "all"), ["user"], _AGENT_TOOLS + ["Skill"]
_TOOL_MAX_TURNS = int(os.environ.get("KERNELMEM_AGENT_MAX_TURNS", "30"))

# The budget is STATED and given a stop rule, because withholding both was
# measurably expensive. Measured on 002 / H100, 2026-08-17 round 0: the seed call
# hit the turn cap, the retry hit it again, and 62 turns over 37.7 minutes
# delivered a kernel that had already existed at roughly turn 8 -- everything
# after it was discarded when the wall arrived. The agent was never told how many
# turns it had, and nothing in the prompt said when to stop, so it kept
# investigating: ~25 of ~31 turns per attempt went to hand-rolled micro-benchmarks
# (convbench.py, cpucost.py, ovl.py, micro.cu) re-deriving badly what the harness
# measures properly two minutes later. The old text did say "do NOT benchmark",
# but gave no reason, and an agent with no roofline feedback and no clock has no
# reason to believe it. So: say the number, say why measuring here is wasted, and
# say what ends the call.
_TOOL_MODE_INSTRUCTION = f"""

TOOLS ARE AVAILABLE THIS CALL. You have Bash, Read, Write, Edit, Glob and Grep,
and a private scratch directory as your working directory. Use them: write the
extension to a file, compile it, and fix what does not build BEFORE answering.
A kernel that fails to compile scores zero, and you can now find that out
yourself instead of spending a round on it.
- `python -c "import torch"` works; torch, nvcc and ninja are on PATH.
- torch.utils.cpp_extension.load_inline is the same mechanism the harness uses,
  so a successful load_inline in your scratch dir means it will build there too.
- Work only inside your working directory. Do not modify anything outside it.

YOU HAVE {_TOOL_MAX_TURNS} TURNS, AND A STOP RULE. Your job this call is a kernel
that COMPILES and MATCHES THE REFERENCE -- not a fast one. The moment `ANSWER.py`
compiles and matches on the shapes you can test, post it in your final message
and STOP. Do not keep exploring because turns remain.

DO NOT BENCHMARK OR PROFILE HERE, and here is why -- the old instruction said
this without a reason:
- The harness benchmarks your kernel immediately after you answer, over every
  scored shape, with interleaved repeats and a paired significance test. Nothing
  you measure in this scratch dir is read by anything.
- The NEXT round's prompt hands you full ncu counters and an nsys timeline for
  every kernel you wrote. That is real profiling data on the real shapes; your
  hand-rolled timing loop here is a worse version of it, several turns later.
- Turns spent measuring are turns not spent on correctness, which IS your job
  this call and is the only thing that can make the round score zero.
- Running out of turns is a FAILURE, not a finish: the call errors out, costs a
  retry, and the round is then scored from whatever `ANSWER.py` holds -- which is
  usually an EARLIER draft than your best work, so your later work is lost.

DELIVER EARLY AND OFTEN, TO A FILE. `ANSWER.py` in your working directory is
what gets scored if this call ends without a final message. On an OPTIMIZATION
call it ALREADY EXISTS and already holds the kernel you were asked to improve --
so the floor is already "no change", and your only job is to raise it. On a
REPAIR call it holds the BROKEN kernel you were asked to fix: that is not a floor,
so overwrite it the moment you have a fixed kernel that compiles. On a seed
call you start with nothing there, so the moment you have a complete kernel that
compiles -- however unambitious -- write it. Overwrite it whenever you have something better that still compiles.
Treat it as a ratchet: `ANSWER.py` should always hold the best COMPLETE kernel
you have so far, never a partial edit.

UPDATE `ANSWER.py` BEFORE YOU START THE NEXT EXPERIMENT, every time -- not once
at the beginning. A stale ratchet is how good work gets thrown away: on the run
that motivated this text, `ANSWER.py` still held the FIRST draft while four
better revisions sat beside it, and when the turn budget ran out the first draft
was what got scored.

This exists because your turn budget can run out mid-investigation, and if it
does, your final message never happens and everything you built is lost with the
scratch directory. `ANSWER.py` is read when that occurs. Writing it costs one
turn and makes the difference between a scored kernel and nothing at all.

OUTPUT CONTRACT (unchanged, and now strict): your FINAL message must contain
exactly ONE fenced code block holding the complete Python file, and nothing
else of substance. Intermediate messages may say whatever you like -- only the
last one is read. `ANSWER.py` is a FALLBACK, not a substitute: still post the
code in your final message.
"""


# ---------------------------------------------------------------------------
# The harness gate: a Stop hook that runs the REAL correctness check on the
# kernel the agent is about to deliver, and refuses to let the turn end while
# it fails.
#
# The agent's own test is a re-implementation in eager PyTorch on the shapes it
# picked, at a tolerance it picked. It never runs the checks the harness runs
# after the call -- the extra scored shapes, the uninitialised-memory poisoning
# pass, the device-state leak rejection -- and those are exactly what the
# candidates that DID fail the harness failed on (state leak x4, extra-shape
# mismatch x2, uninitialised memory x1 of the 11 on record). Before this, such a
# failure surfaced one full round later as a separate repair call that started
# from scratch (new session, cold builds, ~6-10 min). Here the failure comes back
# into the SAME session, with the agent's files, builds and reasoning intact, and
# a fix costs a few turns plus one gate run (~27 s cold, ~5 s once built).
#
# Mechanism (verified against SDK 0.2.110 / bundled CLI 2.1.191 with a smoke
# test): a Stop hook returning {"decision": "block", "reason": ...} keeps the
# agent in the same session and hands it the reason as the next user turn; the
# hook input carries the final message text as `last_assistant_message`. Each
# block consumes one of the call's turns, and the CLI itself allows at most
# CLAUDE_CODE_STOP_HOOK_BLOCK_CAP (default 8) consecutive blocks, so the
# refusal cap below must stay at or under that.
#
# What the gate is NOT: it is not the scorer. It runs warmup=0/repeat=1, reports
# PASS/FAIL only, never a time, and the official bench and paired verdict still
# run on the delivered kernel exactly as before. It also cannot see speed: a
# kernel slower than its parent passes, and the tree deals with that.
# ---------------------------------------------------------------------------
_GATE_INSTRUCTION = """

HARNESS GATE (STRICT). Ending your turn SUBMITS the kernel in your final message
(the fenced code block the OUTPUT CONTRACT requires). Before the submission is
accepted, the harness runs its REAL correctness check on that exact code, in the
harness's own process: it builds the file, compares ModelNew against the task
reference on the profiling shape AND every extra scored shape at the harness
tolerance, re-runs each shape on a poisoned allocator to catch reads of
uninitialised memory, and rejects kernels that corrupt device-global state.
No timing from this check is recorded or scored. If the check FAILS, your turn
does NOT end: you get the harness error verbatim, and you fix the kernel, write
it to `ANSWER.py`, and end your turn again with the fixed kernel in your final
message. You have at most {cap} rejections; after that the submission is scored
as it is, and a failing kernel costs the round. Every gate run costs up to about
30 s and one of your turns, so test correctness yourself first and submit when
you are confident. There is no way to waive the check: resubmitting the same
failing kernel is checked again and counts as another rejection, so if you
believe the harness or the reference is at fault, say so in one sentence AND
still deliver the closest-to-correct kernel you have. A final message with no
fenced kernel is rejected the same way.
"""

_GATE_NO_CHANGE_NUDGE = (
    "HARNESS GATE: your submission is byte-identical to the kernel you were given "
    "to improve, so the plan was not applied. Implement it, or state in ONE "
    "sentence why it cannot be done under the rules (for example, it would need "
    "reduced precision), then end your turn again.")

_GATE_NO_CODE = (
    "HARNESS GATE: your final message contains no complete kernel in a fenced "
    "```python block, so there is nothing to check. End your turn again with the "
    "complete kernel file in exactly one fenced code block.")

# Upper bound on one hook invocation. A gate run is bounded by the harness's own
# 20-minute bench join, and a FAIL may run the gate a second time on the parent
# to rule out an environment fault, so this must exceed two of those. The CLI
# treats a hook that overruns as non-blocking, i.e. a failing kernel would slip
# through -- so this is a ceiling, never a target.
_GATE_HOOK_TIMEOUT_S = float(os.environ.get("KERNELMEM_GATE_HOOK_TIMEOUT", "2700"))


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


def _last_assistant_text(inp: Dict[str, Any]) -> str:
    """The final message the agent is about to end its turn with.

    The bundled CLI hands it over as `last_assistant_message` (a plain string in
    the smoke test that verified this path). Older CLIs may omit it; the JSONL
    transcript at `transcript_path` then holds the same text as the last
    assistant entry.
    """
    msg = inp.get("last_assistant_message")
    if isinstance(msg, str) and msg.strip():
        return msg
    if isinstance(msg, dict):
        parts = []
        for blk in msg.get("content", []) or []:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        if "".join(parts).strip():
            return "\n".join(parts)
    path = inp.get("transcript_path")
    if not path or not os.path.isfile(path):
        return ""
    import json as _json
    last = ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = _json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                content = (rec.get("message") or {}).get("content", [])
                if isinstance(content, str):
                    last = content
                    continue
                texts = [b.get("text", "") for b in content or []
                         if isinstance(b, dict) and b.get("type") == "text"]
                # Track the last assistant entry even when it has no text (a
                # tool call): falling back to an older text entry would gate a
                # stale draft instead of what the agent is ending on.
                last = "\n".join(texts)
    except OSError:
        return ""
    return last


def _effective_gate_cap(requested: int) -> int:
    """Clamp the rejection cap below the CLI's own consecutive-block cap.

    The bundled CLI overrides a Stop hook that blocks more than
    CLAUDE_CODE_STOP_HOOK_BLOCK_CAP times in a row (default 8; 0 = no cap) and
    ends the call as a success carrying the kernel the hook just rejected. A
    cap at or above it would silently stop gating, so keep one below it.
    """
    requested = int(requested)
    try:
        cli_cap = int(os.environ.get("CLAUDE_CODE_STOP_HOOK_BLOCK_CAP", "") or 8)
    except ValueError:
        cli_cap = 8
    if cli_cap <= 0:
        return requested
    return max(1, min(requested, cli_cap - 1))


def _reset_gate_attempt(state: Dict[str, Any]) -> None:
    """Per-attempt counters back to zero before a retried SDK call.

    retry_with_backoff re-runs the same options, i.e. the same hook and state.
    The new session's system prompt promises the full cap, so the rejection
    count and the nudge must start over. `checks` stays cumulative (every gate
    run is real work), and a PASS or a parent PASS from attempt 1 stays valid.
    """
    if state:
        state["blocks"] = 0
        state["nudged"] = False
        if state.get("gate") != "pass":
            state["gate"] = "none"


def _new_gate_state() -> Dict[str, Any]:
    return {"checks": 0, "blocks": 0, "nudged": False, "gate": "none",
            "last_pass_code": None, "parent_verdict": None, "error": None,
            "last_error_type": None}


def _make_stop_gate(*, gate: Callable[[str], Dict[str, Any]],
                    extract: Optional[Callable[[str], str]],
                    baseline_code: Optional[str],
                    baseline_known_good: bool,
                    cap: int,
                    call_type: str,
                    state: Dict[str, Any]):
    """Build the Stop-hook callback for one agent call.

    *gate* takes source text and returns ``{"ok": bool, "error_type": str|None,
    "message": str|None}``; it is the harness's own check, run by the caller's
    process. *extract* pulls the kernel out of the final message the way the
    caller will pull it afterwards, so the bytes checked are the bytes delivered.
    *baseline_code* is the kernel the agent was asked to improve (or repair);
    *baseline_known_good* says whether it passed the harness before, which is
    what lets a FAIL on both be read as an environment fault rather than the
    agent's bug. *state* is mutated so the caller can log what the gate did.

    Every path returns; a hook that raised would be reported by the CLI as a
    hook failure and the stop would go through unchecked, which is worse than an
    explicit allow with `state["gate"] = "error"`.
    """
    _extract = extract or extract_code_block

    def _block(reason: str) -> Dict[str, Any]:
        state["blocks"] += 1
        # Overwritten by any later verdict. If the CLI ends the call right after
        # this block (turn wall, its own consecutive-block cap) the row must not
        # read "gate=none" as if the hook never fired.
        state["gate"] = "blocked"
        return {"decision": "block", "reason": reason}

    async def _stop_gate(inp, tool_use_id, ctx):
        try:
            text = _last_assistant_text(dict(inp) if inp else {})
            # Both extractors need a ``` fence; checking for one first also keeps
            # extract_code_block from dumping an llm_output_error_*.txt file for
            # every fence-less message the hook rejects.
            try:
                code = _extract(text) if ("```" in text and text.strip()) else ""
            except Exception:
                code = ""
            if not code or not code.strip():
                if state["blocks"] >= cap:
                    state["gate"] = "fail_cap"
                    return {}
                return _block(_GATE_NO_CODE)

            if (baseline_known_good and baseline_code is not None
                    and code.strip() == baseline_code.strip()):
                # Passing every gate by doing nothing is not a pass. One nudge,
                # then let it through: the round's own no-change handling (and
                # the log) take it from there rather than burning turns. Only for
                # a known-good baseline (the rollout's parent): a repair handed
                # back unchanged is the broken kernel, and must meet the real
                # check below like any other submission.
                if not state["nudged"] and state["blocks"] < cap:
                    state["nudged"] = True
                    return _block(_GATE_NO_CHANGE_NUDGE)
                state["gate"] = "nochange"
                return {}

            # Blocking work off the event loop: the SDK reads the CLI's stdout on
            # this loop, and a 30-second bench inside the callback would stall it.
            verdict = await asyncio.to_thread(gate, code)
            state["checks"] += 1
            if verdict.get("ok"):
                state["gate"] = "pass"
                state["last_pass_code"] = code
                return {}
            state["last_error_type"] = verdict.get("error_type")

            if baseline_known_good and baseline_code:
                # A kernel that passed the harness before failing it now is the
                # environment, not this candidate -- there is nothing for the
                # agent to fix, so do not spend its turns on it.
                pv = state.get("parent_verdict")
                if pv is None:
                    pv = await asyncio.to_thread(gate, baseline_code)
                    # Remember only a PASS. A failing probe may be transient (a
                    # timeout behind another GPU user, a crashed child); probing
                    # again on the next FAIL costs seconds, while a remembered
                    # FAIL would wave every later candidate through as env_fault.
                    if pv.get("ok"):
                        state["parent_verdict"] = pv
                if not pv.get("ok"):
                    state["gate"] = "env_fault"
                    state["error"] = (f"parent also fails the gate: "
                                      f"{pv.get('error_type')}: "
                                      f"{str(pv.get('message') or '')[:300]}")
                    return {}

            if state["blocks"] >= cap:
                state["gate"] = "fail_cap"
                return {}
            n = state["blocks"] + 1
            reason = (f"HARNESS GATE: FAIL ({verdict.get('error_type')}), rejection "
                      f"{n} of {cap}. Harness error:\n<<<\n"
                      f"{verdict.get('message') or '(no message)'}\n>>>\n"
                      f"Fix the kernel, write the complete fixed file to ANSWER.py, "
                      f"and end your turn with it in exactly one fenced code block. "
                      f"Resubmitting it unchanged is checked again and counts as "
                      f"another rejection.")
            return _block(reason)
        except asyncio.CancelledError:
            # The CLI cancels a hook that overruns its timeout (or dies mid-call).
            # CancelledError is not an Exception, so say what happened before it
            # propagates; the worker thread and its bench child cannot be
            # cancelled and finish on their own.
            state["gate"] = "cancelled"
            state["error"] = "hook cancelled while the gate was running"
            raise
        except Exception as exc:  # never let the hook itself fail the stop
            state["gate"] = "error"
            state["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[gate] hook error ({state['error']}); letting the {call_type} "
                  f"call end unchecked", flush=True)
            return {}

    return _stop_gate


def _tools_enabled(call_type: str) -> bool:
    """Whether this call gets tools. Set KERNELMEM_AGENT_TOOLS=0 to disable."""
    if os.environ.get("KERNELMEM_AGENT_TOOLS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return call_type in _TOOL_CALL_TYPES

# The Agent SDK builds the child env as {**os.environ, **options.env} at spawn
# time, so omitting a key does NOT remove it — an inherited ANTHROPIC_API_KEY
# would silently switch billing to the API. Empty-string overrides behave like
# unset vars and force claude.ai subscription auth.
_SUBSCRIPTION_ENV = {
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_AUTH_TOKEN": "",
    "CLAUDE_CODE_USE_BEDROCK": "",
    "CLAUDE_CODE_USE_VERTEX": "",
}


# ---------------------------------------------------------------------------
# Give every tool-mode agent its own torch extension build directory.
#
# The agent is told to compile with load_inline (see _TOOL_MODE_INSTRUCTION),
# and torch keys its build directory on the extension NAME alone:
# ~/.cache/torch_extensions/py313_cu130/<name>/. The name is whatever the model
# picked, and independently generated kernels reuse names constantly -- see the
# note in utils/verify_chain.py about two rounds both emitting
# `vae_resblock_fused_ms_graph_ext`, and kernel_20260806_064326.py /
# kernel_20260806_064916.py in this repo, which collide on
# `vae_resblock_fused_l2`. So an agent's throwaway compile lands in the same
# directory the harness later builds the real kernel in, and two agents running
# at once (already the case across lineages) write the same cuda.cu concurrently.
#
# Worse, that directory is guarded by torch's FileBaton, which has no timeout,
# no staleness check and no holder-liveness check, and is released only in the
# winner's `finally`. Anything that skips stack unwinding orphans it permanently:
# p.terminate()/p.kill() after the 20-minute p.join in main_memory_latest, and
# run_lineages' SIGTERM/SIGKILL. (The 600s SIGALRM in utils/compile_and_run does
# NOT, despite looking like the obvious candidate -- it raises an ordinary Python
# exception that unwinds through cpp_extension's `finally: baton.release()`.) A
# later kernel that picks a poisoned name then spins until the compile alarm and
# gets reported as an illegal memory access, so a good kernel is scored -inf and
# sent to repair for a bug it does not have.
#
# Pointing the agent at its own directory removes both. It is not free: when the
# agent's last successful compile is byte-identical to the file it returns, the
# harness's first build of that kernel used to be a ninja no-op and is now a cold
# nvcc compile -- the 482 object builds in this repo's .ninja_log files run a
# median of 9.9s and a p90 of 26.9s. Paying that once per accepted kernel to stop
# an agent from poisoning the directory the harness measures in is the right
# trade, but it is a trade. This does not contradict profiling/ncu.py's note
# about leaving TORCH_EXTENSIONS_DIR unset -- that is about the HARNESS reusing
# built .so files under ncu, a different process.
# ---------------------------------------------------------------------------
def _agent_build_env(workdir: str) -> Dict[str, str]:
    ext_dir = os.path.join(workdir, "torch_ext")
    os.makedirs(ext_dir, exist_ok=True)
    return {"TORCH_EXTENSIONS_DIR": ext_dir}


# ---------------------------------------------------------------------------
# The second delivery channel.
#
# Tool mode was retrofitted onto a design that was one-shot: `tools=[],
# max_turns=1`, where the reply simply WAS the kernel, so "your final message
# must contain the code" described a reply rather than constraining one. The
# contract never changed when the call became an agent that works over many
# turns in a filesystem. The result is a seam: the agent does its work in a
# directory and delivers through a message, and only the message is read.
#
# That seam is not hypothetical. On 2026-08-16 a seed spent 35 minutes, ran six
# successful nvcc builds and a 64-point config sweep, wrote a complete 17 KB
# kernel to its scratch dir -- and scored nothing, because the turn budget ran
# out before the turn that would have PASTED that file into a message, and the
# cleanup then deleted it. A sibling agent orphaned by a SIGKILL that same hour
# produced a kernel measured correct to 1.5e-3 against a 1e-2 tolerance and
# clearly faster; it survived only because the kill skipped the cleanup. Two
# working kernels in one hour, both discarded by the plumbing rather than by
# the benchmark.
#
# So the directory becomes a real delivery channel: the agent maintains its best
# complete kernel at ANSWER.py, and when the message never arrives the harness
# reads the file instead. Delivery stops being one all-or-nothing turn at the
# end and becomes a ratchet the agent updates as it goes.
#
# Salvage is SAFE TO ATTEMPT because it is not a shortcut past any gate. A
# salvaged kernel still has to compile, still has to match the reference within
# tol, and still has to beat the base to be promoted -- the harness validates it
# exactly as it validates a delivered one. The gate below only rejects files
# that cannot be kernels at all, so the worst case is one wasted benchmark
# rather than a bad kernel silently promoted.
# ---------------------------------------------------------------------------
_ANSWER_FILE = "ANSWER.py"


def _seed_answer(workdir: str, code: str) -> None:
    """Pre-fill ANSWER.py with the kernel this call is meant to improve.

    Never raises: a failure to seed must not fail the call, it only costs the
    floor it would have provided.
    """
    try:
        with open(os.path.join(workdir, _ANSWER_FILE), "w", encoding="utf-8") as fh:
            fh.write(code)
    except OSError:
        pass


def _salvage_answer(workdir: Optional[str]) -> Optional[str]:
    """The agent's best kernel from its scratch dir, or None if there isn't one.

    Read BEFORE the workdir is removed, and deliberately strict about what counts:

    * it must parse as a complete Python module -- a file the agent was midway
      through writing when the turn budget ran out fails here, which is the main
      thing that could otherwise be salvaged into a wasted round;
    * it must mention ``ModelNew``, the entry point the output contract names, so
      a leftover probe script (``prof.py``, ``err_test.py``) is not mistaken for
      an answer.

    Never raises: this runs on the failure path, and a salvage attempt that threw
    would replace a clear error with a confusing one.
    """
    if not workdir:
        return None
    path = os.path.join(workdir, _ANSWER_FILE)
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            src = fh.read()
    except OSError:
        return None
    if not src.strip():
        return None
    try:
        ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    if "ModelNew" not in src:
        return None
    return src


def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: Optional[int] = None,  # None means unlimited retries
    initial_delay: float = 1.0,
    max_delay: float = 300.0,  # 5 minutes
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (CLIConnectionError, CLIJSONDecodeError),
    retry_if: Optional[Callable[[BaseException], bool]] = None,
    retry_if_max_retries: int = 1,
) -> Any:
    """Call *func*, retrying the failures that are worth retrying.

    *retry_if* extends the retryable set to exceptions that cannot be named by
    class -- see ``_is_transient_cli_result_error``. It gets its own, much
    smaller budget (*retry_if_max_retries*) because those failures surface only
    after the model has already generated a full reply: one seed draw that hit
    this took 29 minutes to fail, so spending the full *max_retries* on it would
    burn hours on what may well be a deterministic failure.
    """
    delay = initial_delay
    attempt = 0
    predicate_attempts = 0

    while True:
        try:
            return func()
        except Exception as e:
            _by_predicate = bool(retry_if and retry_if(e))
            if not (isinstance(e, retryable_exceptions) or _by_predicate):
                raise
            if _by_predicate:
                predicate_attempts += 1
                if predicate_attempts > retry_if_max_retries:
                    print(f"❌ Not retrying again after {predicate_attempts - 1} retry/retries: "
                          f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                    raise
            attempt += 1
            if max_retries is not None and attempt > max_retries:
                print(f"❌ Failed after {max_retries} attempts. Last error: {type(e).__name__}: {e}", flush=True)
                raise

            error_name = type(e).__name__
            if max_retries is not None:
                print(f"⚠️  {error_name} occurred (attempt {attempt}/{max_retries}). Retrying in {delay:.1f}s...", flush=True)
            else:
                print(f"⚠️  {error_name} occurred (attempt {attempt}, unlimited retries). Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)


def _is_transient_cli_result_error(exc: BaseException) -> bool:
    """True for 'the CLI reported an error result and exited non-zero'.

    The SDK surfaces this as a BARE ``Exception`` from ``Query.receive_messages``
    -- it is the CLI's own error text, re-raised in place of the uninformative
    ProcessError -- so no CLI*Error class covers it and it used to propagate
    straight out of the run. Seen on 2026-08-04: a seed draw failed this way
    after 29 minutes with the text "...error result: success", killing a run at
    round 0 where no checkpoint existed yet.

    Matched on message text because the class carries no other signal. Kept
    narrow deliberately: a wrong match here would silently retry a real bug.

    EXHAUSTING THE TURN BUDGET IS EXPLICITLY NOT TRANSIENT. The CLI reports it
    through the same "returned an error result" channel, so the text match above
    used to catch it and retry -- and a retry cannot do better, because nothing
    about the second attempt is different: it gets a fresh full budget and walks
    into the same wall. Measured on 002 / H100, 2026-08-17 round 0: two attempts,
    30 turns each, 36.7 minutes of the round's 37.7, and the kernel that was
    finally scored came from attempt 2's ANSWER.py written at roughly turn 8.
    Salvage is the correct handling for this failure, and it is already wired --
    so return False and let the call fall straight through to it.
    """
    if not isinstance(exc, Exception):
        return False
    msg = str(exc)
    if "maximum number of turns" in msg:
        return False
    return "returned an error result" in msg


def colorize_finish_reason(reason: Optional[str]) -> str:
    colors = {
        "stop": "\033[92m",  # Green
        "end_turn": "\033[92m",
        "success": "\033[92m",
        "length": "\033[93m",  # Yellow
        "max_tokens": "\033[93m",
        "content_filter": "\033[91m",  # Red
        "stop_sequence": "\033[91m",
        "refusal": "\033[91m",
        "tool_calls": "\033[94m",  # Blue
        "function_call": "\033[94m",
        "tool_use": "\033[94m",
        "null": "\033[90m",  # Grey
    }
    reset_color = "\033[0m"
    if reason is None:
        return f"\033[90mFinish reason: unknown{reset_color}"
    color = colors.get(reason, "\033[90m")  # Default to grey
    return f"{color}Finish reason: {reason}{reset_color}"


def _resolve_model(model_name: str) -> str:
    if model_name and model_name.lower().startswith("claude"):
        return model_name
    return DEFAULT_MODEL


def _flatten_prompt(prompt: str | list[dict]) -> str:
    if isinstance(prompt, str):
        return prompt
    parts = []
    for msg in prompt:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _write_call_outcome(
    log_path: Optional[str],
    round_idx: int,
    call_type: str,
    *,
    outcome: str,
    num_turns: Optional[int],
    duration_s: Optional[float],
    salvaged: str,
    model: str = "",
    effort: str = "",
    detail: str = "",
) -> None:
    """Append one agentic call to calls.csv, next to usage.csv.

    Written for EVERY agentic call, including the ones that raise -- which is the
    point. usage.csv is written after the is_error raise, so a call that failed
    left no row at all: the run that motivated this had three rows for four calls,
    and the missing one was the call that killed it. Rather than relocate that
    block (usage is genuinely about tokens, and a failed call's token count is
    only in `result`, which may be None), failures get their own record.

    `num_turns` comes free off ResultMessage and is the number nothing else in
    this repo could answer. Diagnosing a 30-turn wall previously meant inferring
    from token ratios, because the transcript lives in the scratch dir and the
    scratch dir is deleted in the `finally` above.

    `salvaged` distinguishes the three outcomes that used to look identical
    downstream, where a failed round records only `speedup: None`:
      none    -- nothing on disk; the round produces no kernel at all
      parent  -- the ratchet floor came back untouched; the agent built nothing new
      moved   -- the agent wrote something of its own before it ran out
    """
    if not log_path:
        return
    try:
        path = os.path.join(os.path.dirname(log_path), "calls.csv")
        file_exists = os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,round_idx,call_type,outcome,num_turns,duration_s,"
                        "salvaged,model,effort,detail" + "\n")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            det = str(detail).replace(",", ";").replace("\n", " ")[:160]
            f.write(f"{timestamp},{round_idx},{call_type},{outcome},"
                    f"{'' if num_turns is None else num_turns},"
                    f"{'' if duration_s is None else f'{duration_s:.1f}'},"
                    f"{salvaged},{model},{effort},{det}\n")
    except Exception as e:
        print(f"Warning: Failed to write call outcome log: {e}")


def _write_usage_row(
    log_path: Optional[str],
    round_idx: int,
    call_type: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    model: str = "",
    effort: str = "",
) -> None:
    """Append one call to usage.csv.

    `model`/`effort` are recorded because calls no longer all use the same ones:
    the MCTS rollout runs on --rollout_model and the judge stays on --model_name,
    and a token log that does not say which model spent them cannot answer "did
    the split actually take effect". Appended at the END of the row so a usage.csv
    written before this change stays readable -- _append_usage_totals reads by
    name off the file's own header, and older rows stay positionally aligned.
    """
    if not log_path:
        return
    try:
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,"
                        "total_tokens,model,effort\n")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},"
                    f"{total_tokens},{model},{effort}\n")
    except Exception as e:
        print(f"Warning: Failed to write usage log to {log_path}: {e}")


async def _run_query(prompt_text: str, options: ClaudeAgentOptions,
                     final_only: bool = False) -> tuple[list[str], Optional[ResultMessage]]:
    """Collect assistant text. *final_only* keeps ONLY the last message's text.

    Single-turn calls must join everything, as before. But a tool-using call
    narrates while it works ("let me compile this...", a scratch test script,
    a diff), and `extract_code_block` returns the FIRST fenced block in the
    joined text -- which would be that scratch script rather than the kernel.
    Keeping only the final message makes the contract "your last message is the
    answer", which is also what the tool-mode system prompt asks for.
    """
    texts: list[str] = []
    result: Optional[ResultMessage] = None
    async for message in query(prompt=prompt_text, options=options):
        if isinstance(message, AssistantMessage):
            if final_only:
                texts = []          # keep only the most recent assistant message
            for block in message.content:
                if isinstance(block, TextBlock):
                    texts.append(block.text)
                else:
                    block_type = getattr(block, "type", type(block).__name__)
                    print(f"Skipping non-text content block of type '{block_type}'")
        elif isinstance(message, ResultMessage):
            result = message
    return texts, result


def query_server(
    prompt: str | list[dict],
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    num_completions: int = 1,
    server_port: int = 30000,
    server_address: str = "localhost",
    server_type: str = "claude",
    model_name: str = "default",
    is_reasoning_model: bool = True,
    budget_tokens: int = 0,
    reasoning_effort: Optional[str] = None,
    log_path: Optional[str] = None,
    call_type: str = "unknown",
    round_idx: int = -1,
    baseline_code: Optional[str] = None,
    gate: Optional[Callable[[str], Dict[str, Any]]] = None,
    gate_extract: Optional[Callable[[str], str]] = None,
    gate_refusals: int = 5,
    baseline_known_good: bool = False,
):
    # temperature/top_p/top_k, budget_tokens, num_completions, and the server_*
    # params are accepted for caller compatibility but have no Agent SDK
    # equivalent; the CLI controls sampling and output length itself.
    #
    # `gate` installs the harness gate (see _make_stop_gate) on tool-mode calls:
    # the agent may not end its turn while the kernel in its final message fails
    # the harness's correctness check, up to `gate_refusals` rejections.
    # `gate_extract` is how the caller will pull the kernel out of the reply, so
    # the gate checks the same bytes; `baseline_known_good` marks `baseline_code`
    # as a kernel that already passed, which turns a FAIL on both into an
    # environment fault instead of a rejection. Ignored on non-tool calls.
    #
    # `is_reasoning_model` was the last parameter in this signature that was
    # neither honoured nor listed above -- the same defect `reasoning_effort` had.
    # A caller passing False got high-effort thinking anyway and nothing said so.
    # It DOES have an Agent SDK equivalent: effort is the thinking dial, so
    # "not a reasoning model" is the bottom of it. Honoured as a DEFAULT only, so
    # the precedence rule stays the same everywhere in this function -- an
    # explicit reasoning_effort still wins.
    model = _resolve_model(model_name)
    # Per-call effort, because the calls are not alike. The MCTS rollout (the
    # `optimization` call that writes the next kernel) runs on a cheaper model at
    # high effort, while the judge and analysis calls stay where they were; before
    # this, `reasoning_effort` was accepted and then silently discarded, so every
    # call got the same env-or-default value and per-call routing was impossible.
    # An explicit request therefore WINS over KERNELMEM_CLAUDE_EFFORT, which is
    # now the default for calls that do not ask for anything -- a global env var
    # that overrode explicit choices would defeat the routing without saying so.
    effort = (reasoning_effort
              or os.environ.get("KERNELMEM_CLAUDE_EFFORT")
              or (DEFAULT_EFFORT if is_reasoning_model else "low"))

    use_tools = _tools_enabled(call_type)
    workdir: Optional[str] = None
    gate_state: Dict[str, Any] = {}   # filled only when the harness gate is installed
    if use_tools:
        # A private scratch cwd, so the agent's files land nowhere near the repo
        # or the run artifacts even though it holds a real Bash.
        workdir = tempfile.mkdtemp(prefix=f"kernelmem_agent_{call_type}_")
        # Seed the ratchet from the kernel this call was asked to IMPROVE, before
        # the agent gets a turn. An optimization or repair call is handed a
        # working kernel in its prompt, so a valid answer exists at turn 0 -- but
        # the instruction only asks the agent to write ANSWER.py "the moment you
        # have a complete kernel that compiles", and an agent that spends its
        # whole budget reading counters never reaches that condition. Measured
        # 2026-08-17: two consecutive rollouts hit the 30-turn wall having written
        # NOTHING to disk, so salvage had nothing and the run died -- while the
        # parent kernel, already scoring 1.1409, sat unused in the prompt.
        #
        # Writing it here makes the floor "no change" instead of "nothing", and
        # makes the ratchet independent of whether the agent complies. The agent
        # can only ever overwrite it with something better.
        if baseline_code:
            _seed_answer(workdir, baseline_code)
        _skills, _sources, _tools = _skills_config()
        _hooks = None
        _gate_text = ""
        if gate is not None and gate_refusals > 0:
            gate_state = _new_gate_state()
            _cap = _effective_gate_cap(gate_refusals)
            if _cap != int(gate_refusals):
                print(f"[gate] {call_type}: rejection cap {gate_refusals} clamped to {_cap} "
                      f"(the CLI overrides a Stop hook after its own consecutive-block cap)",
                      flush=True)
            _hooks = {"Stop": [HookMatcher(hooks=[_make_stop_gate(
                gate=gate, extract=gate_extract, baseline_code=baseline_code,
                baseline_known_good=baseline_known_good, cap=_cap,
                call_type=call_type, state=gate_state)],
                timeout=_GATE_HOOK_TIMEOUT_S)]}
            _gate_text = _GATE_INSTRUCTION.format(cap=_cap)
        options = ClaudeAgentOptions(
            system_prompt=((system_prompt or "") + _TOOL_MODE_INSTRUCTION
                           + _gate_text
                           + (_SKILLS_INSTRUCTION if _skills else "")),
            model=model,
            effort=effort,
            tools=_tools,
            allowed_tools=_tools,
            permission_mode="bypassPermissions",  # headless: nothing can approve a prompt
            cwd=workdir,
            max_turns=_TOOL_MAX_TURNS,
            env={**_SUBSCRIPTION_ENV, **_agent_build_env(workdir)},
            skills=_skills,
            setting_sources=_sources,
            hooks=_hooks,
        )
    else:
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            effort=effort,
            tools=[],
            max_turns=1,
            env=dict(_SUBSCRIPTION_ENV),
        )
    prompt_text = _flatten_prompt(prompt)

    salvaged: Optional[str] = None
    call_error: Optional[Exception] = None
    def _attempt():
        _reset_gate_attempt(gate_state)
        return asyncio.run(_run_query(prompt_text, options, final_only=use_tools))

    try:
        texts, result = retry_with_backoff(
            _attempt,
            max_retries=3,
            retry_if=_is_transient_cli_result_error,
        )
    except Exception as exc:
        # Held, not swallowed: re-raised below unless the scratch dir yields a
        # kernel. BaseException is deliberately NOT caught -- a KeyboardInterrupt
        # or the graceful-stop signal must keep unwinding, not be converted into
        # a salvage.
        texts, result, call_error = [], None, exc
    finally:
        # Before the tree goes, and inside the finally so it also runs when the
        # call raised. Note retry_with_backoff retries INSIDE this block, so the
        # workdir is shared across attempts and an ANSWER.py written by attempt 1
        # survives into attempt 2 -- the ratchet spans retries.
        salvaged = _salvage_answer(workdir)
        if workdir and os.environ.get("KERNELMEM_AGENT_KEEP_WORKDIR", "").strip().lower() \
                not in {"1", "true", "yes", "on"}:
            shutil.rmtree(workdir, ignore_errors=True)

    failed = (call_error is not None
              or (result is not None and result.is_error)
              or not texts)
    salvage_used = False
    orig_failed = failed          # what the CALL did, before the gate's delivery
    salvage_kind: Optional[str] = None
    gated_code = gate_state.get("last_pass_code") if gate_state else None
    if gated_code:
        # The gate holds the last kernel that PASSED the harness check. On the
        # paths where the message is not the delivery -- the call died, hit the
        # turn wall, or ended with no fenced code -- that kernel is a better
        # thing to score than whatever ANSWER.py holds: the file may be an older
        # draft, or the untouched parent (agents routinely keep their real draft
        # in a shared /tmp dir instead of the workdir), and it was never checked.
        _delivered_has_code = False
        if not failed:
            try:
                _delivered_has_code = bool(
                    ((gate_extract or extract_code_block)("\n".join(texts))).strip())
            except Exception:
                _delivered_has_code = False
        if failed or not _delivered_has_code:
            why = (f"{type(call_error).__name__}: {call_error}" if call_error is not None
                   else f"{result.subtype}: {result.result}" if (failed and result is not None)
                   else "the final message carried no fenced kernel")
            print(f"[gate] {call_type}: delivering the last kernel that passed the "
                  f"harness gate ({len(gated_code)} chars) instead of the reply ({why}).",
                  flush=True)
            texts = [_fence(gated_code)]
            salvaged = gated_code
            salvage_used = True
            salvage_kind = "gated"
            failed = False
            gate_state["gate"] = "pass"
    if (not failed and not salvage_used and salvaged and gate_state
            and gate_state.get("gate") == "fail_cap"):
        # The gate let a message through at its cap with nothing passing. If that
        # message has no fenced kernel the reply extracts to nothing and the
        # round fails outright -- while ANSWER.py, already read, holds a complete
        # kernel. Score that instead, exactly as for a call that died.
        _joined = "\n".join(texts)
        _has = False
        if "```" in _joined:
            try:
                _has = bool(((gate_extract or extract_code_block)(_joined)).strip())
            except Exception:
                _has = False
        if not _has:
            print(f"[salvage] {call_type}: the final message carried no fenced kernel "
                  f"after the harness gate's cap; scoring {_ANSWER_FILE} "
                  f"({len(salvaged)} chars) instead.", flush=True)
            texts = [_fence(salvaged)]
            salvage_used = True
    if failed and salvaged:
        why = (f"{type(call_error).__name__}: {call_error}" if call_error is not None
               else f"{result.subtype}: {result.result}" if result is not None
               else "the model returned no text")
        # Say plainly whether the agent moved the ratchet at all. "Recovered a
        # kernel" reads like a save either way, but recovering the untouched
        # baseline means the call produced NOTHING and the round is about to
        # re-score its own parent -- which is worth seeing in the log rather than
        # inferring later from two identical scores.
        moved = (baseline_code is None) or (salvaged.strip() != baseline_code.strip())
        what = (f"Recovered {len(salvaged)} chars from {_ANSWER_FILE}"
                if moved else
                f"{_ANSWER_FILE} is still the UNCHANGED parent kernel -- this call "
                f"produced no improvement")
        print(f"[salvage] {call_type}: no kernel arrived in the reply ({why}). "
              f"{what}; it still has to compile, match the reference and beat "
              f"the base like any other candidate.", flush=True)
        # Re-enter the normal path rather than returning early, so usage logging
        # and the finish-reason print below happen for a salvaged call exactly as
        # they do for a delivered one -- a scored kernel whose cost went unrecorded
        # would be worse than the failure this is fixing.
        texts = [f"```python\n{salvaged}\n```"]
        salvage_used = True

    # Before the raises, so a call that dies is recorded exactly as one that
    # lives. This is the row that was missing for every failure so far.
    gate_note = ""
    if gate_state:
        gate_note = (f"gate={gate_state.get('gate')} checks={gate_state.get('checks')} "
                     f"blocks={gate_state.get('blocks')}"
                     + (f" last_error={gate_state.get('last_error_type')}"
                        if gate_state.get("last_error_type") else "")
                     + (f" ({gate_state.get('error')})" if gate_state.get("error") else ""))
        print(f"[gate] {call_type}: {gate_note}", flush=True)
    _write_call_outcome(
        log_path, round_idx, call_type,
        outcome=("ok" if not orig_failed else
                 "max_turns" if "maximum number of turns" in str(call_error or
                                                                 getattr(result, "subtype", ""))
                 else "error"),
        num_turns=getattr(result, "num_turns", None),
        duration_s=(getattr(result, "duration_ms", None) or 0) / 1000.0 if result else None,
        salvaged=(salvage_kind or
                  ("moved" if salvage_used and (baseline_code is None
                                                or salvaged.strip() != baseline_code.strip())
                   else "parent" if salvage_used else "none")),
        model=model, effort=effort,
        detail="; ".join(x for x in (
            (f"{type(call_error).__name__}: {call_error}" if call_error is not None
             else getattr(result, "subtype", "") if orig_failed else ""),
            gate_note) if x),
    )

    if call_error is not None and not salvage_used:
        raise call_error
    if result is not None and result.is_error and not salvage_used:
        raise RuntimeError(f"Claude Agent SDK call failed ({result.subtype}): {result.result}")

    # Usage logging (usage.csv row format consumed by main_memory_latest.py)
    if result is not None and result.usage:
        usage = result.usage
        input_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        print(f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}")
        _write_usage_row(log_path, round_idx, call_type, input_tokens, output_tokens,
                         total_tokens, model=model, effort=effort)

    finish_reason = getattr(result, "stop_reason", None) or getattr(result, "subtype", None)
    print(colorize_finish_reason(finish_reason))
    if finish_reason in {"length", "max_tokens"}:
        print("Warning: Output truncated at the CLI's output limit")

    if not texts:
        raise RuntimeError("Claude Agent SDK returned no text output")
    output = "\n".join(texts)
    return output
