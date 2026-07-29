# utils/kernel_io.py
"""Utility helpers for Mind‑Evolution CUDA‑kernel workflow.

This tiny module centralizes two common I/O helpers that were previously
inlined in the end‑to‑end test script:

1. ``extract_code_block`` – extract the first ```python ... ``` (or generic) code
   block from LLM output. Raises if none found.
2. ``save_kernel_code`` – writes extracted code to *kernels/* with a unique
   timestamped filename and returns the *Path* object.

Keeping them here avoids duplication across evolution loops / diagnostics.
"""
from __future__ import annotations

import ast
import datetime as dt
import re
from pathlib import Path
from typing import Final
import json
from typing import Any, Dict, List
__all__: Final = [
    "extract_code_block",
    "save_kernel_code",
]

# ---------------------------------------------------------------------------
# 1. Code‑block extraction
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


# Match code‑fence opening; language tag is optional
_CODE_FENCE_OPEN_RE = re.compile(r"```(?:[A-Za-z0-9_+\-]+)?\s*\n?")

# A fence occupying a whole line — the shape a stream-continuation artifact takes.
_CODE_FENCE_LINE_RE = re.compile(r"(?m)^[ \t]*```[A-Za-z0-9_+\-]*[ \t]*$")


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def _splice(prev: list[str], nxt: list[str]) -> list[str]:
    """Join two segments that were split at a stream-continuation boundary.

    A continuation can cut mid-line and then re-emit that line in full, e.g.

        ...const at::Tensor& w2n, const at::          <- cut here
        ```python                                     <- fence re-opened
        ...const at::Tensor& w2n, const at::Tensor& n2w,   <- line restarted

    When the last line of *prev* is a strict prefix of the first line of *nxt*,
    the truncated copy is dropped so the line is not duplicated.
    """
    if prev and nxt:
        tail, head = prev[-1], nxt[0]
        if tail and tail != head and head.startswith(tail):
            return prev[:-1] + nxt
    return prev + nxt


def _is_usable_block(block: str, text: str) -> bool:
    """True when *block* can plausibly be the kernel file the caller wants.

    Parsing cleanly is not enough. A stray mid-stream ```python fence is read as
    the *closing* fence, so the extracted block can be a valid-Python fragment
    while the real file sits further down the reply. Two such fragments show up
    in practice and both parse without error:

      * a header of pure ``#`` comments — ``ast`` gives it an empty body;
      * a prefix that drops a ``class ModelNew`` the reply plainly contains.

    Either way the caller should try ``_repair_continuation_fences`` rather than
    accept the fragment, which would otherwise be written to disk and fail much
    later as "must define a ModelNew class", burning a whole round.
    """
    try:
        if not ast.parse(block).body:
            return False
    except (SyntaxError, ValueError):
        return False
    if "class ModelNew" in text and "class ModelNew" not in block:
        return False
    return True


def _repair_continuation_fences(text: str) -> str | None:
    """Rebuild code split across a spurious mid-stream fence.

    The LLM reply can arrive as several text blocks; a continuation may re-open
    its ```python fence mid-file. The non-greedy block regex then reads that
    *opening* fence as the *closing* one and silently returns a truncated file,
    which fails later as an unterminated string literal.

    Segments are spliced one at a time and the result returned as soon as it
    parses, so a reply that legitimately holds several distinct code blocks is
    left alone. Returns ``None`` when no repair yields valid Python.
    """
    fences = list(_CODE_FENCE_LINE_RE.finditer(text))
    if len(fences) < 3:
        return None

    segments = [
        text[fences[i].end():fences[i + 1].start()].strip("\n").splitlines()
        for i in range(len(fences) - 1)
    ]
    if not segments:
        return None

    # Case 1 - the continuation RESUMES the cut file. Splice segments together
    # one at a time and stop as soon as the result parses, so a reply that
    # legitimately holds several distinct code blocks is left alone.
    merged = segments[0]
    for seg in segments[1:]:
        merged = _splice(merged, seg)
        candidate = "\n".join(merged).strip() + "\n"
        if _parses(candidate):
            return candidate

    # Case 2 - the continuation RESTARTS the file from the top instead of
    # resuming (a fresh header, not the truncated line repeated). Splicing then
    # produces garbage, but one segment is a complete file on its own. Prefer
    # the longest segment that parses; the truncated prefix never will.
    for seg in sorted(segments, key=len, reverse=True):
        candidate = "\n".join(seg).strip() + "\n"
        if candidate.strip() and _parses(candidate):
            return candidate
    return None


def extract_code_block(text: str) -> str:
    """Return the **first** triple‑back‑ticked block in *text*.

    - After finding an opening fence, search for the closing fence. If none is
      found, consume until end of string.
    - If the text contains no ``` fences at all, raise and dump the raw output
      to a timestamped file for debugging.
    - If the block does not parse as Python, retry across mid-stream
      continuation fences before giving up (see ``_repair_continuation_fences``).
    """
    if text is None:
        text = ""

    m_open = _CODE_FENCE_OPEN_RE.search(text)
    if not m_open:
        # No ``` found → raise and persist raw output to disk
        dump_path = f"llm_output_error_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(dump_path, "w") as f:
            f.write(text)
        raise RuntimeError(f"No ``` code block found in LLM output – raw output saved to {dump_path}")

    start = m_open.end()
    m_close = re.search(r"```", text[start:])
    if m_close:
        end = start + m_close.start()
        block = text[start:end]
    else:
        # No closing fence: take everything to the end
        block = text[start:]

    block = block.strip() + "\n"
    if _parses(block) and _is_usable_block(block, text):
        return block

    repaired = _repair_continuation_fences(text)
    if repaired is not None:
        print(f"[extract] recovered kernel across a mid-stream continuation fence "
              f"({len(block.splitlines())} -> {len(repaired.splitlines())} lines)")
        return repaired

    # Unrepairable: return the original block so the downstream compile error
    # (and the repair round it triggers) behaves exactly as before.
    print(f"[extract] WARNING: extracted block does not parse as Python "
          f"({len(block.splitlines())} lines) and could not be repaired")
    return block



# ---------------------------------------------------------------------------
# 2. Persist kernel to file
# ---------------------------------------------------------------------------

def save_kernel_code(code: str, out_dir: Path | str = "kernels") -> Path:
    """Save *code* to *out_dir/kernel_YYYYmmdd_HHMMSS.py* and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"kernel_{stamp}.py"
    path.write_text(code, encoding="utf-8")

    return path


# utils/kernel_io.py



def _loads_lenient(candidate: str) -> Any | None:
    """json.loads, tolerating raw control characters inside string values.

    LLM replies routinely embed literal newlines/tabs in long prose fields such
    as ``modification_plan``. Python's parser rejects those in strict mode, so a
    single stray control byte would otherwise discard an entire strategy reply.
    Falls back to stripping the offending bytes if strict=False is not enough.
    """
    if not candidate:
        return None
    for attempt in range(3):
        try:
            if attempt == 0:
                return json.loads(candidate)
            if attempt == 1:
                return json.loads(candidate, strict=False)
            # last resort: drop control chars that are illegal even leniently
            cleaned = "".join(
                ch for ch in candidate if ch >= " " or ch in "\t\n\r"
            )
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            continue
    return None


def _last_decodable_json(raw: str) -> Any | None:
    """Return the last independently-decodable JSON value in *raw*, else None.

    Walks every '{'/'[' and tries ``raw_decode`` from it, so a truncated or
    otherwise broken value early in the reply cannot hide a well-formed one
    after it. ``strict=False`` mirrors _loads_lenient: literal newlines inside
    long prose fields are normal in judge replies.
    """
    decoders = (json.JSONDecoder(strict=False), json.JSONDecoder())
    found: list[Any] = []
    idx = 0
    while idx < len(raw):
        nxt = re.search(r"[{\[]", raw[idx:])
        if not nxt:
            break
        start = idx + nxt.start()
        for dec in decoders:
            try:
                obj, end = dec.raw_decode(raw, start)
            except ValueError:
                continue
            found.append(obj)
            idx = end
            break
        else:
            idx = start + 1
    return found[-1] if found else None


def extract_json(raw: str) -> Any:
    """
    Extract the first JSON object/array from a string and parse it into a Python object.
    Supports fenced code blocks like ```json ...``` or raw JSON embedded in text.

    Args:
        raw: Raw LLM output text.
    Returns:
        A Python object (``dict`` or ``list``).
    Raises:
        ValueError: If no valid JSON can be found/parsed.
    """
    if not isinstance(raw, str):
        raw = str(raw)

    # Try the ```json ...``` fenced format first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if match:
        parsed = _loads_lenient(match.group(1).strip())
        if parsed is not None:
            return parsed

    # Try matching the first { ... } or [ ... ]
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if match:
        parsed = _loads_lenient(match.group(1).strip())
        if parsed is not None:
            return parsed

    # The regex above is greedy: it spans the FIRST '{' to the LAST '}', so a
    # reply holding more than one top-level object parses as neither. That is a
    # real failure mode -- a judge sometimes abandons a half-written object and
    # restarts, leaving `{truncated...\n{complete...}`. Scan for individually
    # decodable values instead and keep the LAST one, i.e. the model's final
    # answer rather than the abandoned draft.
    parsed = _last_decodable_json(raw)
    if parsed is not None:
        return parsed

    # Fallback: attempt to parse the whole string
    parsed = _loads_lenient(raw.strip())
    if parsed is not None:
        return parsed
    raise ValueError(f"Failed to extract valid JSON from reply:\n{raw}")

def save_prompt_text(text: str, out_dir: Path, *, tag: str = "repair") -> Path:
    """
    Save *text* to ``out_dir/{tag}_YYYYMMDD-HHMMSS.txt`` and return the Path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{tag}_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path

def extract_cuda_kernel_names(py_path: Path) -> List[str]:
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    p1 = re.compile(r"""__global__\s+void\s+([A-Za-z_]\w*)\s*\(""", re.MULTILINE)
    p2 = re.compile(
        r"""__global__\s+__launch_bounds__\s*\([^)]*\)\s*void\s+([A-Za-z_]\w*)\s*\(""",
        re.MULTILINE,
    )

    names = p1.findall(src) + p2.findall(src)
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered
