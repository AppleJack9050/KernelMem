# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402


# -----------------------------
# system_prompt as Template
# -----------------------------
system_prompt_tmpl = Template(
    """You are a senior CUDA + PyTorch auditor. Your job is to analyze a failed kernel attempt and
report exactly ONE most critical issue that prevents passing the benchmark.

Judge by OBSERVABLE FAILURE PHENOMENA and PRIORITY, not by error_type names.

Priority rules (MUST follow, top to bottom):

(1) If ERROR_LOG shows any compiler or build failure
    (e.g., nvcc errors, missing symbols, signature mismatch, extension build failure),
    report that issue and DO NOT analyze kernel logic or correctness.
    Also include trigger_condition and patch_anchor for the minimal edit that would unblock compilation.

(2) If ERROR_LOG shows a CUDA execution crash
    (e.g., illegal memory access, device-side assert, invalid launch configuration, crash at synchronize),
    report the most likely cause in kernel indexing / memory access and DO NOT analyze numerical correctness.

(3) If the program runs but ERROR_LOG reports numerical failure
    (e.g., "Outputs are not close", max_abs_err / mean_abs_err reported),
    analyze correctness using the following internal priority:
      1) Output coverage or indexing error (missing tiling for out_h / out_w, out-of-bounds writes)
      2) Parameter parity violation (Scheme A: independent torch.randn params or not using holder weight/bias)
      3) Output shape mismatch (stride / padding / output_padding / groups / dilation / bias)
      4) Unsupported reference arguments used without implementation or explicit asserts
      5) dtype / layout / contiguity issues

(4) If ERROR_LOG indicates timeout or hang, report the most likely performance-pathological cause
    (e.g., excessive launch count, extremely slow kernel, missing parallelism).

Rules:
- Return one and only one issue — the single highest-impact problem.
- Prefer issues that explain the failure with concrete evidence from ERROR_LOG.
- Use ERROR_LOG metadata (shapes, dtypes, contiguity, ref args, max_err_index/value) whenever possible.
- Keep each field brief; avoid extra commentary or multiple alternatives.
- If nothing clearly wrong is found, say it explicitly.

Special rule for WMMA:
- If CUDA_CODE contains "wmma::" AND ERROR_LOG indicates a CUDA execution crash
  (e.g., "unspecified launch failure", "illegal memory access", "device-side assert"),
  then critical_issue MUST be "WMMA collective contract violated" unless ERROR_LOG clearly shows a different root cause.
- must_hold_invariants MUST include:
  1) load/mma/store are executed by exactly one full warp (32 threads) per tile
  2) the 16x16 tiles consumed by load_matrix_sync are fully initialized
  3) store_matrix_sync uses a warp-uniform base pointer (shared/global), not per-thread local

Hard constraints (MUST follow):
- evidence items MUST be copied verbatim from ERROR_LOG (substrings that can be grep'ed).
- For compiler/build failures: root_cause MUST mention the exact API/type/template/flag combination causing failure.
- minimal_fix MUST specify a concrete edit ("Change X from A to B") or a concrete compile flag change; no generic advice.
- patch_anchor MUST be a unique substring that appears in CUDA_CODE or PYTORCH_CODE and indicates where to edit.

Keep each field brief; avoid extra commentary, lists, or alternatives.

Output format (JSON):
```json
{
  "critical_issue": "<max 20 words; single headline>",
  "evidence": ["<exact substring from ERROR_LOG>", "<optional second substring>"],
  "root_cause": "<max 80 words; why this construct causes THIS failure>",
  "trigger_condition": "<max 60 words>",
  "must_hold_invariants": ["<max 3 items; required conditions to avoid re-crash>"],
  "minimal_fix": "<max 40 words; may include a small bundle if inseparable>",
  "patch_anchor": "<max 30 words; unique substring in CUDA_CODE or PYTORCH_CODE>",
  "confidence": "high|medium|low"
}
```
"""
)

# -----------------------------
# instruction as Template
# -----------------------------
instruction_tmpl = Template(
    """You are given:

ERROR_LOG:
$ERROR_LOG

PyTorch reference (ground truth):

$PYTORCH_CODE

CUDA candidate (to audit):

$CUDA_CODE

$REPAIR_HISTORY

$REFERENCE_KERNEL

Task:
- Follow the priority rules to identify the single most critical issue.
- Base your judgment on observable evidence in ERROR_LOG, not on error_type labels.
- First extract 1–2 evidence substrings from ERROR_LOG, then derive root_cause and minimal_fix from them.
- If repair history is provided, analyze previous repair attempts to avoid repeating the same mistakes and understand what has been tried.
- If a known-good kernel is provided, diff the candidate against it. A construct the known-good kernel guards (identity checks, strong references, defensive clones) and the candidate does not is prime suspect for the failure.

Follow the Rules and produce the JSON exactly in the specified format."""
)

# -----------------------------
# Build both at once (returns tuple)
# -----------------------------
def build_correctness_prompts(*, error_log: str, arch_path: Path, cuda_code: str, repair_history: Optional[List[Dict[str, Any]]] = None,
                              reference_kernel: Optional[str] = None,
                              reference_kernel_score: Optional[float] = None):
    """
    Return (system_prompt_str, instruction_str).

    Parameters
    ----------
    error_log : str
        The error log from the failed kernel execution.
    arch_path : Path
        Path to the PyTorch reference architecture file.
    cuda_code : str
        The CUDA code to audit.
    repair_history : Optional[List[Dict[str, Any]]]
        List of previous repair attempts for the same repair chain.
        Each dict should contain: error_log, problem_identification, runnable, speedup, test_passed.
    reference_kernel : Optional[str]
        Source of a kernel that passed every correctness check earlier in this
        run. Repair prompts are otherwise built only from the broken lineage, so
        a fix discovered in one branch is invisible to a repair happening in
        another and gets silently regressed. Showing the known-good source lets
        the audit diff against a version that demonstrably works.
    reference_kernel_score : Optional[float]
        Speedup of ``reference_kernel``, quoted so the model knows the known-good
        kernel is a real alternative and not merely a slower fallback.
    """
    pytorch_code = Path(arch_path).read_text().strip()
    
    # Format repair history if provided
    repair_history_text = ""
    if repair_history and len(repair_history) > 0:
        repair_lines = ["# Repair History (Previous attempts for this repair chain)", ""]
        for idx, hist in enumerate(repair_history, 1):
            repair_lines.append(f"## Previous Repair Attempt #{idx}")
            
            # Extract fields from repair history
            error_log_hist = hist.get("error_log", "N/A")
            problem_ident = hist.get("problem_identification", {})
            runnable = hist.get("runnable", None)
            speedup = hist.get("speedup", None)
            test_passed = hist.get("test_passed", None)
            
            # Format error log (truncate if too long)
            if error_log_hist and error_log_hist != "N/A":
                error_log_short = error_log_hist[:500] + "..." if len(error_log_hist) > 500 else error_log_hist
                repair_lines.append(f"- **Error log**: {error_log_short}")
            
            # Format problem identification
            if problem_ident and isinstance(problem_ident, dict):
                critical_issue = problem_ident.get("critical_issue", "N/A")
                root_cause = problem_ident.get("root_cause", "N/A")
                minimal_fix = problem_ident.get("minimal_fix", "N/A")
                repair_lines.append(f"- **Critical issue identified**: {critical_issue}")
                repair_lines.append(f"- **Root cause**: {root_cause}")
                repair_lines.append(f"- **Minimal fix attempted**: {minimal_fix}")
            
            # Format results
            if runnable is not None:
                # Format speedup safely (avoid formatting None)
                speedup_str = f"{speedup:.4f}" if speedup is not None else "N/A"
                repair_lines.append(f"- **Result**: runnable={runnable}, speedup={speedup_str}, test_passed={test_passed}")
            else:
                repair_lines.append(f"- **Result**: Not yet tested")
            
            repair_lines.append("")
        
        repair_history_text = "\n".join(repair_lines)
    else:
        repair_history_text = "# Repair History\n(No previous repair attempts for this repair chain.)\n"
    
    if reference_kernel and reference_kernel.strip():
        score_txt = (f" (speedup {reference_kernel_score:.4f})"
                     if isinstance(reference_kernel_score, (int, float)) else "")
        reference_kernel_text = (
            f"# Known-Good Kernel{score_txt}\n"
            "This kernel passed EVERY correctness check on ALL evaluated shapes earlier\n"
            "in this run. It is not the kernel under audit; it is a working reference.\n"
            "Where it and the candidate differ, assume the known-good kernel is right\n"
            "unless the error log says otherwise — the difference is often the bug.\n"
            "\n"
            f"{reference_kernel.strip()}\n"
        )
    else:
        reference_kernel_text = "# Known-Good Kernel\n(No previously passing kernel is available for this run.)\n"

    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        ERROR_LOG=error_log.strip(),
        PYTORCH_CODE=pytorch_code.strip(),
        CUDA_CODE=cuda_code.strip(),
        REPAIR_HISTORY=repair_history_text.strip(),
        REFERENCE_KERNEL=reference_kernel_text.strip(),
    )
    return system_prompt, instruction
