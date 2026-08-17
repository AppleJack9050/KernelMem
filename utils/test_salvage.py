"""Checks for the scratch-dir delivery channel.

    python -m utils.test_salvage

The bug this closes: an agent that does its work in a filesystem but delivers
through a message loses everything if the message never arrives. Measured on
2026-08-16 -- a seed ran six successful builds and wrote a complete 17 KB kernel,
then scored nothing because the turn budget expired before the turn that would
have pasted it into a reply, and the cleanup deleted the directory.

What has to be true for the fix to be worth having, in order of importance:

1. **A failure with a usable ANSWER.py returns a kernel instead of raising.**
2. **Salvage never fires on success.** A delivered reply must be returned byte
   for byte; a fallback that sometimes preempts the real answer would be a far
   worse bug than the one being fixed.
3. **The gate rejects what is not a kernel.** Salvage runs on the failure path,
   where the file may be a half-written edit or a leftover probe script.
4. **The read happens before the directory is deleted**, which is the whole
   reason this lives inside the `finally` rather than after it.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.query_server as qs  # noqa: E402
from utils.kernel_io import extract_code_block  # noqa: E402

KERNEL = "import torch\nimport torch.nn as nn\n\n\nclass ModelNew(nn.Module):\n    pass\n"


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    assert cond, msg


def _result(*, is_error: bool, subtype: str = "success"):
    return types.SimpleNamespace(is_error=is_error, subtype=subtype,
                                 result="…", usage=None, stop_reason="end_turn")


def _newest_workdir() -> str:
    import glob
    ds = sorted(glob.glob("/tmp/kernelmem_agent_*"), key=os.path.getmtime)
    return ds[-1]


def _run(*, outcome, answer_file: str | None, call_type: str = "seed"):
    """Drive query_server with a faked agent call.

    Intercepts at ``retry_with_backoff`` -- the module global query_server calls
    -- so everything around it (workdir creation, the finally, the salvage, the
    cleanup) is the real code path rather than a re-implementation of it.
    """
    seen = {}

    def fake_retry(func, **_kw):
        wd = _newest_workdir()
        seen["workdir"] = wd
        if answer_file is not None:                 # the agent writing its answer
            Path(wd, qs._ANSWER_FILE).write_text(answer_file, encoding="utf-8")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    real = qs.retry_with_backoff
    qs.retry_with_backoff = fake_retry
    try:
        out = qs.query_server(prompt="x", call_type=call_type, model_name="claude-opus-5")
        return out, None, seen
    except Exception as exc:
        return None, exc, seen
    finally:
        qs.retry_with_backoff = real


print("[salvage] a max-turns failure with a usable ANSWER.py yields the kernel")
out, err, seen = _run(outcome=([], _result(is_error=True, subtype="error_max_turns")),
                      answer_file=KERNEL)
_check(err is None, "the call no longer raises")
_check(out is not None and "ModelNew" in out, "a kernel came back")
_check(extract_code_block(out).strip() == KERNEL.strip(),
       "and it survives extract_code_block byte for byte")

print("[salvage] the scratch dir is still deleted after being read")
_check(not os.path.exists(seen["workdir"]),
       "workdir removed, so salvage read it BEFORE the cleanup")

print("[salvage] an exception, not just an error result, is also salvageable")
out, err, _ = _run(outcome=RuntimeError("CLI died"), answer_file=KERNEL)
_check(err is None and out is not None and "ModelNew" in out,
       "a raised exception with an answer on disk returns the kernel")

print("[salvage] with nothing usable on disk the original failure still raises")
out, err, _ = _run(outcome=([], _result(is_error=True, subtype="error_max_turns")),
                   answer_file=None)
_check(err is not None and "error_max_turns" in str(err),
       "no ANSWER.py -> the real error propagates, unchanged")
out, err, _ = _run(outcome=RuntimeError("CLI died"), answer_file=None)
_check(isinstance(err, RuntimeError) and "CLI died" in str(err),
       "the held exception is re-raised, not replaced")

print("[salvage] the gate rejects what cannot be a kernel")
for label, body in (("a half-written edit", "import torch\nclass ModelNew("),
                    ("a leftover probe script", "import torch\nprint(1)\n"),
                    ("an empty file", "   \n")):
    out, err, _ = _run(outcome=([], _result(is_error=True, subtype="error_max_turns")),
                       answer_file=body)
    _check(err is not None, f"{label} is not salvaged")

print("[salvage] NEVER fires when the model actually replied")
real_reply = "here you go\n```python\n# the genuine reply\nclass ModelNew: pass\n```"
out, err, _ = _run(outcome=([real_reply], _result(is_error=False)), answer_file=KERNEL)
_check(err is None and out == real_reply,
       "a successful reply is returned verbatim, not replaced by ANSWER.py")
_check("the genuine reply" in out and "import torch.nn" not in out,
       "the delivered code wins over the file, even when both exist")

print("[salvage] an empty reply with no answer on disk still raises")
out, err, _ = _run(outcome=([], _result(is_error=False)), answer_file=None)
_check(err is not None and "no text output" in str(err),
       "the no-text path is unchanged when there is nothing to salvage")

print("[salvage] it applies to optimization and repair too, not just seed")
for ct in ("optimization", "repair"):
    out, err, _ = _run(outcome=([], _result(is_error=True, subtype="error_max_turns")),
                       answer_file=KERNEL, call_type=ct)
    _check(err is None and out is not None, f"{ct} calls salvage as well")

print("\n[salvage] all checks passed")
