# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402


COMPILE_ERROR = Template(
    """You are a senior CUDA-extension developer.
Your job is to FIX the Python script shown below so that it successfully passes the benchmark.

IMPORTANT CONTEXT
- The failure may come from build issues, CUDA execution crashes, numerical mismatch, or timeout.
- Do NOT rely on error_type names; infer the failure cause from ERROR_LOG and Main Critical Problem.

MANDATORY
- ModelNew MUST include an equivalent PyTorch module ONLY as a PARAMETER HOLDER
  (for initialization / state_dict parity with the reference).
- ModelNew MUST source weights/bias from that module.
- ModelNew MUST NOT call the holder module’s forward for replaced operators.
- Do NOT create independent parameters for replaced operators (no torch.randn for weights/bias).
- The code MUST contain at least one custom CUDA kernel (defined in the `source` CUDA string) that optimizes the computation process. Do NOT remove all CUDA kernels or replace them with pure PyTorch operations.

REPAIR STRATEGY (FOLLOW STRICTLY)
- Apply the SMALLEST change necessary to address the Main Critical Problem.
- Do NOT refactor unrelated code or redesign the kernel unless strictly required.
- You MUST choose exactly ONE repair path below based on observable evidence in ERROR_LOG.
- Follow ONLY the chosen path. Do NOT apply rules from other paths.

PATCH BUDGET (STRICT)
- You may modify up to 3 code regions if and only if they are inseparable to fix the SAME root cause.
- If WMMA is used (wmma:: appears), fixing a crash may require 2-3 coupled edits; this is allowed and still counts as "smallest change".

DECISION FLOW (evaluate in order; stop at the first matching case)

(1) Build / compilation failure detected in ERROR_LOG:
    - Fix includes, function signatures, bindings, load_inline usage, or C++/CUDA syntax.
    - Do NOT modify kernel math, indexing, or algorithmic semantics.

(2) CUDA execution crash detected (e.g., illegal memory access, launch failure):
    - Focus on bounds checks, grid/block configuration, and indexing of input/weight/output.
    - Ensure all indices are within allocated tensor ranges.
    - Do NOT change algorithmic semantics or introduce new optimizations.
    - If CUDA_CODE contains "wmma::", you MUST enforce ALL of the following (otherwise the fix is invalid):
        a) Exactly one warp participates in wmma::load_matrix_sync / wmma::mma_sync / wmma::store_matrix_sync for each tile.
        b) The full 16x16 tiles read by load_matrix_sync are initialized (32-thread blocks must load multiple elements per thread, or use 256-thread staging with only warp0 doing WMMA).
        c) store_matrix_sync is called warp-collectively with a warp-uniform base pointer (shared/global). DO NOT call store from a single thread. DO NOT store into per-thread local arrays.

(3) Numerical mismatch detected (e.g., outputs not close, max_abs_err reported):
    - Fix issues in the following strict priority order:
        a) Output coverage / indexing (full tiling of all output dimensions).
        b) Parameter parity (use parameter-holder weight/bias correctly).
        c) Output shape formula (stride / padding / output_padding / groups / dilation / bias).
        d) Unsupported reference arguments without explicit asserts.
        e) dtype / layout / contiguity assumptions.
    - Do NOT introduce new optimizations while fixing correctness.

(4) Timeout or hang detected:
    - Reduce pathological behavior (e.g., excessive kernel launches, missing parallelism).
    - Keep semantics unchanged.


OUTPUT RULES (STRICT) ────────────────────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple-quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

────────────────────────────────────────────────────────────────
ERROR LOG
────────────────────────────────────────────────────────────────
$ERROR_LOG

────────────────────────────────────────────────────────────────
OLD CODE (read-only)
────────────────────────────────────────────────────────────────
$OLD_CODE

$REFERENCE_KERNEL

────────────────────────────────────────────────────────────────
Main Critical Problem
────────────────────────────────────────────────────────────────
$Problem

```python
# <your corrected code>
```
# ==========================================================
"""
)

def _escape_template(s: str) -> str:
    return s.replace("$", "$$")

def _sanitize_text(s: str) -> str:
    return s.replace("```", "`")

def _format_problem(problem: Optional[Any]) -> str:
    if problem is None or problem == "":
        return "No prior critical problem provided."
    if isinstance(problem, Mapping):
        # Format the new JSON structure with updated field names
        ci = str(problem.get("critical_issue", "")).strip()
        rc = str(problem.get("root_cause", "")).strip()
        mf = str(problem.get("minimal_fix", "")).strip()
        tc = str(problem.get("trigger_condition", "")).strip()
        pa = str(problem.get("patch_anchor", "")).strip()
        conf = str(problem.get("confidence", "")).strip()
        evidence = problem.get("evidence", [])
        evidence_str = ", ".join([str(e).strip() for e in evidence]) if isinstance(evidence, list) else ""
        must_hold = problem.get("must_hold_invariants", [])
        must_hold_str = ", ".join([str(m).strip() for m in must_hold]) if isinstance(must_hold, list) else ""
        
        # Build formatted string with new fields
        parts = []
        if ci:
            parts.append(f"critical_issue: {ci}")
        if evidence_str:
            parts.append(f"evidence: {evidence_str}")
        if rc:
            parts.append(f"root_cause: {rc}")
        if tc:
            parts.append(f"trigger_condition: {tc}")
        if must_hold_str:
            parts.append(f"must_hold_invariants: {must_hold_str}")
        if mf:
            parts.append(f"minimal_fix: {mf}")
        if pa:
            parts.append(f"patch_anchor: {pa}")
        if conf:
            parts.append(f"confidence: {conf}")
        
        if parts:
            return "\n".join(parts)
        # Fall back to JSON if no recognized fields found
        return json.dumps(problem, ensure_ascii=False, indent=2)
    # For other types, simply convert to string
    return str(problem)

def build_error_prompt(
    *,
    old_code: str,
    error_log: str,
    problem: Optional[Any] = None,
    gpu_name: Optional[str] = None,
    reference_kernel: Optional[str] = None,
    reference_kernel_score: Optional[float] = None,
) -> str:
    """
    Build the error-repair prompt with GPU context + architecture source.

    Parameters
    ----------
    old_code : str
        The broken Python script content to show under OLD CODE.
    error_log : str
        The compiler/runtime error text to show under ERROR LOG.
    arch_path : Path
        Path to the reference architecture Python file to display.
    gpu_name : Optional[str]
        Human-readable GPU name key to lookup in gpu_specs.
        If None, attempts torch.cuda.get_device_name(0).

    Returns
    -------
    str
        The final prompt string to send to the LLM.
    """
    # Load the GPU spec dictionary
    gpu_info = _load_gpu_spec()

    # Resolve GPU name
    if gpu_name is None:
        try:
            import torch  # local import to avoid hard dependency if CPU-only
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO (file: {HW_FILE})")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")

    # Bullet list of key specs except the arch line (already printed separately)
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )
    problem_text = _format_problem(problem)

    # A kernel that already passed every shape earlier in this run. Without it the
    # repair sees only the broken lineage, so a fix found in another branch is
    # invisible here and gets regressed rather than reused.
    if reference_kernel and reference_kernel.strip():
        score_txt = (f" — speedup {reference_kernel_score:.4f}"
                     if isinstance(reference_kernel_score, (int, float)) else "")
        reference_kernel_text = (
            "────────────────────────────────────────────────────────────────\n"
            f"KNOWN-GOOD KERNEL (read-only{score_txt})\n"
            "────────────────────────────────────────────────────────────────\n"
            "This kernel PASSED every correctness check on ALL evaluated shapes\n"
            "earlier in this run. OLD CODE above did not. Before writing your fix,\n"
            "diff the two: a guard this kernel has and OLD CODE lacks (identity\n"
            "check, strong reference, defensive clone, wider cache key) is the most\n"
            "likely cause of the failure. Reuse its solution rather than inventing\n"
            "a new one. Do NOT copy it wholesale — keep OLD CODE's optimizations.\n"
            "\n"
            f"{reference_kernel.strip()}"
        )
    else:
        reference_kernel_text = ""

    # Substitute all fields
    return COMPILE_ERROR.substitute(
        ERROR_LOG=error_log.strip(),
        OLD_CODE=old_code.strip(),
        REFERENCE_KERNEL=reference_kernel_text,
        Problem=_escape_template(_sanitize_text(problem_text.strip())),
    )