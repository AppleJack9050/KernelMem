"""LLM backend for KernelMem, routed through the Claude Agent SDK.

All calls spawn the locally logged-in Claude Code CLI (`claude auth login`,
claude.ai account) so usage bills against the user's Claude subscription
credit — never a pay-per-token API key.
"""

import asyncio
import datetime
import os
import time
from typing import Any, Callable, Optional

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


def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: Optional[int] = None,  # None means unlimited retries
    initial_delay: float = 1.0,
    max_delay: float = 300.0,  # 5 minutes
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (CLIConnectionError, CLIJSONDecodeError),
) -> Any:
    delay = initial_delay
    attempt = 0

    while True:
        try:
            return func()
        except retryable_exceptions as e:
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
) -> None:
    if not log_path:
        return
    try:
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
    except Exception as e:
        print(f"Warning: Failed to write usage log to {log_path}: {e}")


async def _run_query(prompt_text: str, options: ClaudeAgentOptions) -> tuple[list[str], Optional[ResultMessage]]:
    texts: list[str] = []
    result: Optional[ResultMessage] = None
    async for message in query(prompt=prompt_text, options=options):
        if isinstance(message, AssistantMessage):
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
    reasoning_effort: str = "medium",
    log_path: Optional[str] = None,
    call_type: str = "unknown",
    round_idx: int = -1,
):
    # temperature/top_p/top_k, budget_tokens, num_completions, and the server_*
    # params are accepted for caller compatibility but have no Agent SDK
    # equivalent; the CLI controls sampling and output length itself.
    model = _resolve_model(model_name)
    effort = os.environ.get("KERNELMEM_CLAUDE_EFFORT", DEFAULT_EFFORT)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model,
        effort=effort,
        tools=[],
        max_turns=1,
        env=dict(_SUBSCRIPTION_ENV),
    )
    prompt_text = _flatten_prompt(prompt)

    texts, result = retry_with_backoff(
        lambda: asyncio.run(_run_query(prompt_text, options)),
        max_retries=3,
    )

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
        _write_usage_row(log_path, round_idx, call_type, input_tokens, output_tokens, total_tokens)

    finish_reason = getattr(result, "stop_reason", None) or getattr(result, "subtype", None)
    print(colorize_finish_reason(finish_reason))
    if finish_reason in {"length", "max_tokens"}:
        print("Warning: Output truncated at the CLI's output limit")

    if not texts:
        raise RuntimeError("Claude Agent SDK returned no text output")
    output = "\n".join(texts)
    return output
