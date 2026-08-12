"""LLM backend for KernelMem, routed through the Claude Agent SDK.

All calls spawn the locally logged-in Claude Code CLI (`claude auth login`,
claude.ai account) so usage bills against the user's Claude subscription
credit — never a pay-per-token API key.
"""

import asyncio
import datetime
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    CLIJSONDecodeError,
    ResultMessage,
    TextBlock,
    query,
)

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
_TOOL_MAX_TURNS = int(os.environ.get("KERNELMEM_AGENT_MAX_TURNS", "30"))

_TOOL_MODE_INSTRUCTION = """

TOOLS ARE AVAILABLE THIS CALL. You have Bash, Read, Write, Edit, Glob and Grep,
and a private scratch directory as your working directory. Use them: write the
extension to a file, compile it, and fix what does not build BEFORE answering.
A kernel that fails to compile scores zero, and you can now find that out
yourself instead of spending a round on it.
- `python -c "import torch"` works; torch, nvcc and ninja are on PATH.
- torch.utils.cpp_extension.load_inline is the same mechanism the harness uses,
  so a successful load_inline in your scratch dir means it will build there too.
- Do NOT benchmark against the reference to tune here; correctness and
  compilation are what tools are for. The harness measures performance.
- Work only inside your working directory. Do not modify anything outside it.

OUTPUT CONTRACT (unchanged, and now strict): your FINAL message must contain
exactly ONE fenced code block holding the complete Python file, and nothing
else of substance. Intermediate messages may say whatever you like -- only the
last one is read.
"""


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
# trade, but it is a trade. This does not contradict run_ncu_memory.py's note
# about leaving TORCH_EXTENSIONS_DIR unset -- that is about the HARNESS reusing
# built .so files under ncu, a different process.
# ---------------------------------------------------------------------------
def _agent_build_env(workdir: str) -> Dict[str, str]:
    ext_dir = os.path.join(workdir, "torch_ext")
    os.makedirs(ext_dir, exist_ok=True)
    return {"TORCH_EXTENSIONS_DIR": ext_dir}


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
                          f"{type(e).__name__}: {str(e)[:200]}")
                    raise
            attempt += 1
            if max_retries is not None and attempt > max_retries:
                print(f"❌ Failed after {max_retries} attempts. Last error: {type(e).__name__}: {e}")
                raise

            error_name = type(e).__name__
            if max_retries is not None:
                print(f"⚠️  {error_name} occurred (attempt {attempt}/{max_retries}). Retrying in {delay:.1f}s...")
            else:
                print(f"⚠️  {error_name} occurred (attempt {attempt}, unlimited retries). Retrying in {delay:.1f}s...")
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
    """
    return isinstance(exc, Exception) and "returned an error result" in str(exc)


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
    the MCGS rollout runs on --rollout_model and the judge stays on --model_name,
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
):
    # temperature/top_p/top_k, budget_tokens, num_completions, and the server_*
    # params are accepted for caller compatibility but have no Agent SDK
    # equivalent; the CLI controls sampling and output length itself.
    #
    # `is_reasoning_model` was the last parameter in this signature that was
    # neither honoured nor listed above -- the same defect `reasoning_effort` had.
    # A caller passing False got high-effort thinking anyway and nothing said so.
    # It DOES have an Agent SDK equivalent: effort is the thinking dial, so
    # "not a reasoning model" is the bottom of it. Honoured as a DEFAULT only, so
    # the precedence rule stays the same everywhere in this function -- an
    # explicit reasoning_effort still wins.
    model = _resolve_model(model_name)
    # Per-call effort, because the calls are not alike. The MCGS rollout (the
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
    if use_tools:
        # A private scratch cwd, so the agent's files land nowhere near the repo
        # or the run artifacts even though it holds a real Bash.
        workdir = tempfile.mkdtemp(prefix=f"kernelmem_agent_{call_type}_")
        options = ClaudeAgentOptions(
            system_prompt=(system_prompt or "") + _TOOL_MODE_INSTRUCTION,
            model=model,
            effort=effort,
            tools=_AGENT_TOOLS,
            allowed_tools=_AGENT_TOOLS,
            permission_mode="bypassPermissions",  # headless: nothing can approve a prompt
            cwd=workdir,
            max_turns=_TOOL_MAX_TURNS,
            env={**_SUBSCRIPTION_ENV, **_agent_build_env(workdir)},
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

    try:
        texts, result = retry_with_backoff(
            lambda: asyncio.run(_run_query(prompt_text, options, final_only=use_tools)),
            max_retries=3,
            retry_if=_is_transient_cli_result_error,
        )
    finally:
        if workdir and os.environ.get("KERNELMEM_AGENT_KEEP_WORKDIR", "").strip().lower() \
                not in {"1", "true", "yes", "on"}:
            shutil.rmtree(workdir, ignore_errors=True)

    if result is not None and result.is_error:
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
