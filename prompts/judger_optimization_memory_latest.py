#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt builder for **single most impactful optimisation** suggestion.

Use this when you do **not** provide an error log. Instead, supply:
  - NCU metrics block (text/markdown)
  - GPU name (looked up in prompts/hardware/gpu_specs.py)
  - PyTorch reference architecture file (contains `class Model`)
  - (Optional) current CUDA candidate code to inspect

The Judge LLM must return **exactly one** optimisation target with a minimal plan.
"""

from __future__ import annotations
from pathlib import Path
from string import Template
from textwrap import dedent
import importlib.util
import os
import sys
from typing import Optional, Tuple, Dict, Any, Callable
import pandas as pd
import yaml
import tempfile
import csv

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"
YAML_RULES_PATH = ROOT / "memorybank" / "bottleneck_headroom_kernelstructure.yaml"

# ---------------------------------------------------------------------------
# Memory-bank enforcement mode
#
# The decision table gates method selection through `allowed_methods`. Binding
# that list only helps when the table's vocabulary covers the methods a problem
# actually rewards, and on vae_block_002 it does not: the best kernel to date
# used `stream_pipeline_overlap`, which is not among the catalog's 26 entries,
# and it was found in a round where the table matched nothing and the judge was
# free to invent.
#
# Advisory mode keeps the machine_check diagnosis in the prompt and records
# what the table would have permitted, without forbidding anything, so the
# in-table and invented populations can be scored against each other before the
# constraint is allowed to bind. Set MEMORYBANK_ENFORCEMENT=hard to bind it.
# ---------------------------------------------------------------------------
ENFORCEMENT_MODE = os.environ.get("MEMORYBANK_ENFORCEMENT", "advisory").strip().lower()
if ENFORCEMENT_MODE not in ("advisory", "hard"):
    raise ValueError(
        "MEMORYBANK_ENFORCEMENT must be 'advisory' or 'hard', "
        f"got {ENFORCEMENT_MODE!r}"
    )

# Each pair rewrites one binding clause into its advisory form. Both prompts
# are rewritten as a unit -- see _apply_enforcement_mode, which refuses to
# return a partially-softened prompt.
_ADVISORY_REWRITES_SYSTEM = (
    (
        "   - The `allowed_methods` list in MACHINE_CHECK RESULT is a HARD CONSTRAINT.\n"
        "   - You MUST select your method from this list. You CANNOT choose a method outside this list.",
        "   - The `allowed_methods` list in MACHINE_CHECK RESULT is ADVISORY.\n"
        "   - Prefer a method from that list when one genuinely fits the measured bottleneck.\n"
        "   - You MAY choose a method outside the list when none of them addresses what the\n"
        "     numbers show; if you do, say so explicitly in `evidence` and name the mechanism.",
    ),
    (
        "  * The allowed_methods list is NOT empty, and you MUST select your method from this list (HARD CONSTRAINT).",
        "  * The allowed_methods list is NOT empty. Treat it as a ranked suggestion, not a gate.",
    ),
    (
        "- You MUST select your method from the allowed_methods list (HARD CONSTRAINT).",
        "- Prefer the allowed_methods list; departing from it is permitted when justified by the metrics.",
    ),
    (
        '  "method_name": "<MUST be one of the allowed_methods from MACHINE_CHECK RESULT>",',
        '  "method_name": "<one of the allowed_methods, or your own short snake_case name if none fit>",',
    ),
    (
        '- The "method_name" field MUST exactly match one of the method IDs listed in the allowed_methods section of MACHINE_CHECK RESULT. You CANNOT choose a method that is not in that list.',
        '- The "method_name" field SHOULD match one of the method IDs in the allowed_methods section.\n'
        '  If no listed method fits the measured bottleneck, choose your own and give it a short\n'
        '  snake_case name; the mismatch is recorded rather than rejected.',
    ),
)

_ADVISORY_REWRITES_INSTRUCTION = (
    (
        '- Your "method_name" MUST be one of the allowed_methods listed above (this is a HARD CONSTRAINT from machine_check).',
        '- Your "method_name" SHOULD be one of the allowed_methods listed above (advisory, not binding:\n'
        '  machine_check records your choice either way).',
    ),
)


def _apply_enforcement_mode(text: str, rewrites: Tuple[Tuple[str, str], ...]) -> str:
    """Soften binding method-selection language when running advisory.

    Raises if a clause is missing rather than returning a prompt that still
    binds. A silent no-op here would look exactly like advisory mode while the
    judge kept refusing anything outside the table, which is the one failure
    that would quietly invalidate the comparison this mode exists to produce.
    """
    if ENFORCEMENT_MODE == "hard":
        return text
    for old, new in rewrites:
        if old not in text:
            raise RuntimeError(
                "advisory rewrite failed: the prompt no longer contains the binding "
                f"clause {old[:70]!r}...; update _ADVISORY_REWRITES_* to match the "
                "current template before relying on advisory mode"
            )
        text = text.replace(old, new)
    return text

# Import machine_check ver2 module
from prompts.machine_check_ver2 import run_machine_check


__all__ = ["build_single_opt_prompts"]

# -----------------------------------------------------------------------------
# GPU spec loader (shared pattern)
# -----------------------------------------------------------------------------

def _load_gpu_spec() -> dict:
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# System prompt: exactly one optimisation target
# -----------------------------------------------------------------------------

from textwrap import dedent
from string import Template


system_prompt_tmpl_no_match = Template(
    dedent(
        """
You are a senior CUDA performance engineer. You will receive:
- Target GPU specs
- PyTorch reference implementation (class Model)
- Current CUDA candidate (may be empty)
- Nsight Compute output: metrics JSON + optional advisory section text
- Optional optimization history

You must produce ONE JSON object with the following mandatory fields:

ONE-METHOD RULE (STRICT):
Exactly one core mechanism only (e.g., smem_tiling OR mem_vectorize OR warp_specialize OR reg_pressure_drop OR epilogue_fusion OR atomic_privatize OR cuda_graph_capture).

ONE Primary Optimisation Method (COUNTS as the single method):
  - Pick the hottest NON-REMOVABLE kernel/segment by GPU time/%time from metrics.
  - Propose ONE (and only one) optimisation mechanism most likely to yield the largest speedup for that hottest target.
  - The plan must be implementable as a focused diff.
PRIMARY METHOD QUALITY BAR (STRICT):
  - The chosen mechanism MUST be justified by either:
(E1) dominant stall/bound evidenced by metrics, or
(E2) a provably equivalent algebraic/semantic rewrite that reduces total work or global memory traffic (e.g., monotonic reordering, affine folding, idempotence elimination).
For cuda_graph_capture, justification may use Nsight Systems launch counts / high CPU overhead / many short kernels as evidence.

ONLY IF the hottest target is pooling/stencil/conv-like (sliding window / neighborhood access), apply the following rules:
STRUCTURED MEMORY ACCESS (MANDATORY CHECK)
  - When optimizing pooling/stencil-like access (e.g., 2×2 windows), prefer vectorization that matches layout:
     - Use float2 row loads for 2 contiguous elements, repeated across rows, rather than forcing float4 across discontinuous addresses.
  - “mem_vectorize” is only valid if the chosen vector width aligns with the access pattern and reduces transactions.
SMEM_TILING GATE (STRICT; MUST-PASS)
- Any proposal that uses shared-memory staging/tiling MUST FIRST provide a shared-memory budget check:
  bytes_per_block = tile_H * tile_W * tile_C * bytes_per_element * buffers
- If bytes_per_block exceeds the target GPU's per-block shared-memory limit (or if the limit is unknown and bytes_per_block is large), you MUST NOT propose smem_tiling.
- In that case, you MUST fall back to block/warp remapping for coalesced access (e.g., warp covers contiguous W/output elements, grid-stride traversal), without staging the full tile in shared memory.
COALESCING-FIRST REQUIREMENT (FOR POOLING/STENCIL)
- For pooling/stencil-like access, you MUST attempt coalescing via block/warp remapping before any smem staging:
  * Prefer “warp covers contiguous W dimension” (or contiguous output elements) so global loads become coalesced.
  * Prefer partial-reduction in registers + small shared memory for reductions, rather than staging the full H×W tile.

REMOVABLE/TRIVIAL KERNEL RULE (CRITICAL):
  - Trivial/removable examples: identity kernels, memcpy-only, contiguous-only, view/reshape-only, dtype-cast-only, dropout(0).
  - If the hottest kernel is removable/trivial, you MUST skip it and re-select the hottest non-removable target.
  - You are FORBIDDEN to use the primary method to optimize removable/trivial kernels.
SPECIALIZATION FOCUS (CRITICAL):
  - Optimize for the SPECIFIC fixed tensor shapes/layouts in the candidate; you MAY hardcode sizes, unroll, and use compile-time constants.

Metrics usage:
  - Use metrics JSON as primary for selecting hottest non-removable kernel and choosing the one method.
  - Advisory section text may be wrong; never follow it if metrics/code disagree.
ALLOWED ACCELERATION PATH (IMPORTANT):
  - Candidate may use load_inline extensions only.
  - Inside extensions, calling vendor libraries is ALLOWED:
    - cuBLAS / cuBLASLt (including epilogues), cuDNN (including fused variants).
  - Treat such vendor-library calls inside the extension as valid "fused/replaced" implementations.
OUTPUT FORMAT (JSON ONLY; no extra prose):
{
  "bottleneck": "<max 70 words; MUST include: [bound_type] + hottest_non_removable_kernel_or_segment + why>",
  "primary_optimisation_method": "<max 90 words; the ONE counted mechanism and why it matches metrics>",
  "method_name": "<short mechanism tag for primary method only>",
  "modification_plan": "<max 750 words; numbered checklist; then method steps; If any rewrite is applicable, modification_plan items 1–2 MUST include the exact code actions to implement it (e.g., “move tanh after pooling; remove 4× tanh calls per output”, “precompute a,b for BN per-channel and use FMA in inner loop”, “replace 4 scalar loads of a 2×2 window with two float2 row loads”).>",
  "evidence": "<>=1 numeric metrics + 1 root-cause line + brief refutation of one alternative bottleneck>",
  "expected_metric_change": "<>=2 metric directions + anti-regression constraint>",
  "headroom": "high|medium|low",
  "structural_rewrite": true|false
}

STRUCTURAL_REWRITE (default false; set it deliberately):
  true ONLY when the plan replaces the kernel's core structure -- a different MMA
  primitive or tile geometry, a different memory pipeline (adding cp.async/TMA
  multi-stage), a different algorithm -- so that the FIRST version is expected to
  be SLOWER than the current base until it is tuned. Declaring it lets the
  rewrite become the base despite being slower, for a few rounds, so it can be
  improved instead of discarded on round one. The debt is called in: if it has
  not beaten the kernel it displaced by the end of that window, the old base is
  restored and those rounds are lost. Do NOT set it for tuning, fusion or
  parameter changes -- those should win immediately and get no grace.

VALIDITY RULES:
  - primary_optimisation_method/method_name must NOT target removable/trivial kernels.
"""
    )
)

instruction_tmpl_no_match = Template(
    dedent(
        """
# Target GPU
GPU Name: $gpu_name
Architecture: $gpu_arch
Details:
$gpu_items

# PyTorch Reference
$python_code

# CUDA candidate
```python
$CUDA_CODE
```

MAINLOOP + EPILOGUE POLICY (IMPORTANT):
  - Treat cuBLAS GEMM / cuBLASLt Matmul / cuDNN Conv/ConvTranspose as default-keep mainloops.
  - Prefer mainloop-border fusion when applicable:
      (a) cuBLASLt epilogue / cuDNN fused variants inside the extension, OR
      (b) fuse all post-ops into ONE custom kernel consuming the mainloop output to minimize write-backs.
  - Only propose rewriting mainloops with custom kernels if metrics + fixed shapes justify outperforming vendor libs.
  - Exception (GEMM/Matmul/Linear): you MAY propose switching to cuBLASLt Matmul (or changing computeType/algo/layout/epilogue) when doing so is necessary to enable Tensor Cores / TF32 and is consistent with the fp32-tolerance contract.

HARD CONSTRAINTS (benchmark):
- Do NOT modify or redefine get_inputs() / get_init_inputs().
- Inputs are already on the correct device; do NOT move tensors across devices in forward.
- Outputs must be allocated on the SAME device as inputs.
- Forward-only benchmark; no backward/grad requirements. Output must match reference forward within tolerance.
- No in-place by default; only if you can prove no aliasing and no reread hazards.

Nsight Compute metrics + advisory analysis
$NCU_METRICS

$NSYS_LAUNCH_COUNTS

$OPTIMIZATION_HISTORY

MANDATORY WORKFLOW (STRICT):
  1. Hottest target selection:
    - Select the hottest NON-REMOVABLE kernel/segment by GPU time/%time from metrics.
    - If metrics only show removable/trivial kernels or are missing, choose the most expensive remaining forward segment by reasoning from op types + tensor sizes,
      and include "metrics unavailable/trivial" in evidence.
    - If the hottest time is dominated by a composite segment (e.g., GEMM + epilogue + post-op), treat it as the target.
  2. Choose ONE primary optimisation method for that hottest non-removable target:
    - Method must be mechanism-distinct from history.
    - Method MUST be tied to the dominant stall/bound evidence; avoid generic suggestions.
    - PRECISION IS NOT A LEVER. The computation must stay in the dtype the workload
      supplies — match it, do not change it in either direction.
      * Never propose downcasting to fp16/bf16/fp8, mixed-precision tensor cores, or any
        narrower storage/compute dtype than the inputs.
      * Equally, never propose upcasting an fp16/bf16 workload to fp32; use the input
        dtype natively.
      * TF32 for conv/GEMM is fine — the reference uses it.
      * Reductions may accumulate in fp32 or wider regardless of storage dtype.
      Reducing bytes moved must come from fusion, layout, caching or reuse, NOT from
      changing the dtype.

HISTORY (if present):
- The primary method must be mechanism-distinct from the latest attempt; method_name must be a short mechanism tag.

FINAL OUTPUT: exactly ONE JSON object, no markdown, no extra text.
"""
    )
)

system_prompt_tmpl = Template(
    dedent(
        """
You are a senior CUDA performance engineer. You will receive:
- Target GPU specs
- PyTorch reference implementation (contains `class Model`)
- Current CUDA candidate (may be empty)
- Nsight Compute output: metrics JSON + detailed section analysis text (OPT/INF are advisory only)
- MACHINE_CHECK RESULT: Deterministic gating output (tier, bottleneck_id, case_id, allowed_methods, key_metrics)
- LLM_ASSIST knowledge base: Method catalog and optimization knowledge to help you understand "how to do" and "why to do"
- Optimization history (if available): Previously attempted optimization strategies and their results

Your task: identify the SINGLE highest-priority performance bottleneck and propose ONE (and only one) optimization method that is most likely to yield the largest real-world speedup for the current kernel, with a minimal, actionable plan.

**SPECIALIZATION FOCUS (CRITICAL)**:
- You are NOT required to produce a generic, shape-agnostic kernel.
- Your ONLY goal is to maximize speedup for the SPECIFIC input tensor dimensions provided in the current CUDA candidate.
- You MAY specialize the kernel implementation for these exact dimensions (e.g., unroll loops for known sizes, use compile-time constants, hardcode block sizes optimized for the given shapes).
- You MAY assume tensor shapes are fixed as shown in the CUDA candidate's forward() and get_inputs().
- This specialization is ENCOURAGED if it leads to better performance for this specific case.

**CRITICAL: Two-Layer System Understanding**

1. **MACHINE_CHECK (Hard Constraint - "Can I do this?")**:
   - The machine_check layer has ALREADY determined which methods are FEASIBLE based on:
     * Metrics analysis (throughput, occupancy, stalls, etc.)
     * Kernel structure classification (S0-S4)
     * Headroom tier (Tier-H/M/L)
     * Bottleneck signatures
   - The `allowed_methods` list in MACHINE_CHECK RESULT is a HARD CONSTRAINT.
   - You MUST select your method from this list. You CANNOT choose a method outside this list.
   - Machine_check answers: "Can I do this?" - it gates what is allowed.

2. **LLM_ASSIST (Knowledge Base - "How do I do this? Why do I do this?")**:
   - The LLM_ASSIST layer provides KNOWLEDGE to help you:
     * Understand what each allowed method means (Method Catalog)
     * Understand how to implement the method (mechanism requirements, expected changes)
     * Understand why a method is appropriate (bottleneck taxonomy, optimization techniques)
     * Write better evidence and modification plans
   - LLM_ASSIST does NOT determine allowed_methods - it only helps you choose and implement from the allowed list.
   - LLM_ASSIST answers: "How do I do this?" and "Why do I do this?" - it provides knowledge and guidance.

**CRITICAL**: If optimization history is provided, you MUST carefully analyze previous attempts:

1. **Distinguish between "method mismatch" vs "poor implementation"**:
   - If a previous attempt used a method that is theoretically appropriate for the bottleneck but achieved low/negative speedup:
     * **First consider**: Was the implementation flawed? (e.g., incorrect prefetch distance, wrong tiling size, suboptimal register blocking)
     * **If implementation was likely flawed**: You may reuse the SAME method but with an IMPROVED implementation
     * **If implementation appears correct but still ineffective**: The method may be fundamentally unsuitable → try a DIFFERENT method
   
   - If a previous attempt used a method that is NOT appropriate for the current bottleneck:
     * This indicates a mismatch between method and problem → try a DIFFERENT method that better matches the bottleneck

2. **Decision criteria for method selection**:
   - **Reuse the same method with improved implementation** when:
     * The method is in the allowed_methods list and matches the bottleneck
     * Previous implementation had clear flaws (wrong parameters, incomplete application, etc.)
     * The expected_metric_change from previous attempt aligns with what's needed
   
   - **Switch to a different method** when:
     * The previous method is fundamentally incompatible with the kernel structure or bottleneck
     * Multiple attempts with the same method (with different implementations) all failed
     * A better-suited method exists in allowed_methods that addresses the bottleneck more directly

3. **Review previous attempts**:
   - Check if previous methods are still in the current allowed_methods list
   - Analyze why previous attempts were ineffective (implementation issue vs method mismatch)
   - Avoid repeating the EXACT same implementation, but consider improved versions of the same method
   - Focus on methods that have NOT been tried yet OR methods that were tried but with poor implementation

**HARD RULE: History De-duplication & No Relabeling**
If optimization history is provided, you MUST NOT propose a plan that is mechanism-identical to a previous attempt while changing only the method label.

Definitions:
- "Mechanism-identical" means the core code-change mechanisms are the same, such as:
  * grid-stride loop + per-thread multi-element processing (+ optional unroll)
  * vectorized load/store widening (float2/float4/half2) + alignment/tail handling
  * block-level reduction pattern changes / warp-shuffle reductions
  * shared-memory tiling / register blocking / pipelining

Rules:
1) If your proposed modification plan matches a prior attempt's core mechanism(s), you MUST:
   - keep the SAME method_name as that prior attempt (treat it as "refinement"), AND
   - explicitly state at least ONE concrete delta vs that attempt (e.g., different vector width, different unroll factor, different block size, different alignment/tail handling, different indexing/coalescing strategy).
2) If you cannot provide a concrete delta, you MUST choose a DIFFERENT method from allowed_methods that has not been tried, or justify why switching is necessary.
3) You MUST NOT change method_name merely to avoid repetition. Relabeling without mechanism change is invalid.

You MUST follow this decision procedure:

(0) Machine-Check Gate (ALREADY PERFORMED)
- The machine_check layer has already analyzed the metrics and determined:
  * Headroom tier (Tier-H / Tier-M / Tier-L) - see MACHINE_CHECK RESULT
  * Primary bottleneck ID - see MACHINE_CHECK RESULT
  * Kernel structure classification - see MACHINE_CHECK RESULT
  * Allowed optimization methods - see MACHINE_CHECK RESULT
- **CRITICAL**: 
  * The allowed_methods list is NOT empty, and you MUST select your method from this list (HARD CONSTRAINT).

(1) Quantitatively identify the primary bottleneck (NUMBERS REQUIRED)
- You MUST cite at least 1 concrete numeric values from the metrics JSON and explain why they indicate the primary bottleneck.
  Examples of acceptable evidence: SM throughput % of peak, DRAM throughput % of peak, compute_memory_throughput % of peak,
  achieved/theoretical occupancy indicators, registers per thread, occupancy limiters, warps active, etc.
- You MUST cite at least 1 specific conclusion from the section analysis text (OPT/INF may be used, but you may NOT rely on OPT/INF alone).
- You MUST explicitly rule out at least 1 plausible-but-secondary bottleneck using numbers (i.e., show why it is not #1 right now).
- If optimization history is provided AND you reuse a previously tried method, your evidence MUST explicitly cite the most relevant Previous Attempt number and state the concrete delta vs that attempt (1-2 sentences).
- You MUST bind each cited metric to a specific code region (exact tensor/array access or loop) causing it.

(2) Use the machine_check bottleneck identification
- The machine_check has already identified the bottleneck_id (see MACHINE_CHECK RESULT).
- You should align your bottleneck description with the bottleneck_id, but you MUST still provide quantitative evidence from the metrics.

(3) Choose ONE optimization method and do an anti-regression check
- You may output only ONE optimization method (single most impactful).
- You MUST select your method from the allowed_methods list (HARD CONSTRAINT).
- **SPECIALIZATION FOCUS**: You are optimizing for the SPECIFIC input tensor dimensions in the current CUDA candidate, NOT for a generic kernel. You MAY specialize aggressively (e.g., unroll for known sizes, use compile-time constants, hardcode block sizes) if it maximizes speedup for these exact dimensions.
- If Method Catalog is provided, use it to understand what each allowed method means, its intent, mechanism requirements, and expected metric changes.
- Use this knowledge to select the most appropriate method from the allowed list and to write better evidence/modification plans.
- You MUST state which hard metrics it should change and the expected direction of change (at least 2 metrics),
  e.g., regs/thread ↓, warps_active ↑, warps_eligible ↑, dram_throughput% ↑, SM throughput% ↑, 
  warp_issue_stalled_* metrics ↓, branch divergence ↓, stall reasons shift, etc.
  Use the Method Catalog's "expected_metric_change" field to guide your expectations.
- You MUST include an anti-regression constraint: explain what must NOT worsen (e.g., avoid large increases in registers/shared memory
  that would drop occupancy or cause spilling). If you cannot give a numeric threshold, provide a clear qualitative constraint and why.
- The chosen method MUST include a mechanism check:
  * Check the Method Catalog's "mechanism_requirements" for the selected method.
  * Check the CASE-LEVEL REQUIREMENTS (if any) for the selected method in the MACHINE_CHECK RESULT section.
  * If it relies on data reuse/caching, you MUST point to the reuse source (where the same data is used again).
  * If no reuse source exists (streaming), you MUST NOT choose that method.
  * Ensure all mechanism_requirements from the Method Catalog AND all case-level requirements are satisfied or explainable.
  * **CRITICAL**: If you cannot satisfy the case-level requirements for a method, you MUST choose a different method from the allowed_methods list.
- Do NOT replace vectorized loads/stores with scalar operations unless you can justify reduced transactions or improved coalescing from metrics.

**HARD RULE: Method-Plan Semantic Consistency (No Mismatch)**
Your "method_name" MUST be semantically consistent with the dominant mechanism in "modification plan".
Use the following binding rules (dominant mechanism decides the label):
- If the plan's dominant mechanism is "process multiple elements per thread" via grid-stride loop, per-thread unrolling, or multiple outputs per thread,
  then method_name MUST be "Increase_ILP_WorkPerThread".
- If the plan's dominant mechanism is "widen transactions / improve coalescing" via vector types (float2/float4/half2), reinterpret_cast-based wide loads/stores,
  alignment guarantees, or tail handling for vector width,
  then method_name MUST be "Improve_Coalescing_and_TransactionSize".
- If the plan's dominant mechanism is "increase concurrent outstanding memory requests" WITHOUT increasing outputs-per-thread
  (e.g., rearranging independent loads earlier, software pipelining of independent pointers, or launch shaping that increases in-flight requests per SM),
  then method_name MUST be "Increase_Memory_Level_Parallelism".
If your plan includes mechanisms that trigger multiple rules, you MUST choose the method corresponding to the PRIMARY mechanism (the one that contributes most to the expected speedup), and you MUST remove secondary mechanisms from the plan to avoid ambiguity.


(4) Practicality constraints (avoid “good idea but not implementable here”)
- The modification plan must be directly actionable on THIS kernel (no generic advice).
- Do NOT propose kernel fusion unless the CUDA candidate explicitly contains multiple distinct kernel launches that share producer-consumer tensors AND you can pinpoint the exact fusion boundary.
- If the CUDA candidate code is empty, your plan must describe the FIRST concrete kernel-level change to implement, aligned with the PyTorch reference semantics.
- You MUST NOT propose changing the benchmark workflow, adding caching outside the kernel, or altering evaluation settings.
- Focus only on kernel code and launch configuration changes that can be implemented inside the CUDA candidate.
- You MAY propose re-scheduling (reorder/fission/fuse) of multiple kernels inside ModelNew.forward ONLY if you can point to concrete producer-consumer tensors / intermediate buffers in the CUDA candidate and justify with metrics.

(5) Benchmark harness invariants (MUST RESPECT)
- You MUST NOT ask to modify or redefine `get_inputs()` or `get_init_inputs()`.
- Assume inputs are already on the correct device; do NOT propose any device transfers in `ModelNew.forward`.
- Do NOT propose changes that require `.to("cuda")`, `.cuda()`, `.cpu()`, or device-dependent branching in forward.

(6) No in-place mutation unless reference is in-place
-- Default: treat in-place writes as INVALID.
- You MAY recommend in-place ONLY if BOTH are true:
  (a) you can prove the overwritten tensor is not read again after the write in Model.forward (no future reads in the reference graph),
  (b) the overwritten tensor does not alias any other input/output (no views/aliases; must be a fresh buffer or guaranteed non-aliased).
- Even when allowed, you MUST state the exact tensor to overwrite and why it is provably safe under this benchmark.

(7) Modification plan format and completeness (APPLIES TO ALL CASES)
- **CRITICAL**: The modification plan must be COMPLETE enough that following it step-by-step fully implements the chosen optimisation method (whether selected from allowed_methods or self-generated when allowed_methods is empty).
- **Format requirement**: You MUST emit the "modification plan" as a numbered checklist (`1. ... 2. ... 3. ...`) with concrete, verifiable actions.
- Each numbered item must be actionable and directly checkable in code review. You MUST include ALL specific details relevant to implementing that particular optimization method. The details you include depend on what the optimization method requires, and may include (but are not limited to):
  * Launch shape and grid/block dimensions (if changed) or explicit statement to keep them unchanged
  * Indexing math and stride calculations (e.g., `stride = blockDim.x * gridDim.x`)
  * Vector width (e.g., float2/float4) or unroll factor (e.g., `#pragma unroll(4)`)
  * Tail/edge handling strategy (e.g., "handle remaining elements with scalar loop")
  * Resource/occupancy constraints (e.g., "keep registers per thread < 32")
  * Specific intrinsics/APIs to use (e.g., `__pipeline_memcpy_async`, `__prefetch_global_l2`)
  * Required conditionals or guards (e.g., "if idx_pref < N then prefetch")
  * Any other implementation details necessary for the specific optimization method
- Do NOT include details that are not relevant to your chosen optimization method. Include ONLY the details that are necessary to fully implement that method.
- Avoid vague advice; remove ambiguity and ensure every required mechanism is explicitly listed as its own numbered item.
- For constraints that require "keeping things unchanged" (e.g., "Keep the same launch shape"), you may state them concisely (e.g., "Keep the same launch shape") without detailed implementation explanation, since maintaining the status quo is straightforward. However, if maintaining a constraint requires a specific implementation approach (e.g., using a grid-stride loop to preserve grid size while changing per-thread work), you MUST specify HOW to achieve it (e.g., "Use grid-stride loop with stride = blockDim.x * gridDim.x to maintain the same grid size").
- Focus the plan on describing changes and modifications. Unchanged aspects can be stated concisely to save space.
- This format requirement applies regardless of whether you select from allowed_methods or create your own optimization strategy.

OUTPUT FORMAT (JSON ONLY; no extra prose):
{
  "bottleneck": "<max 50 words>",
  "optimisation method": "<max 55 words>",
  "method_name": "<MUST be one of the allowed_methods from MACHINE_CHECK RESULT>",
  "modification plan": "<max 600 words>",
  "evidence": "<must include >=1 numeric metrics + >=1 section-analysis statement + 1 ruled-out bottleneck>",
  "expected_metric_change": "<>=2 metric directions + anti-regression constraint>",
  "headroom": "high|medium|low",
  "structural_rewrite": true|false
}

STRUCTURAL_REWRITE: true ONLY when the plan replaces the kernel's core structure
(different MMA primitive or tile geometry, different memory pipeline, different
algorithm) so the first version is expected to be SLOWER until tuned. It buys a
few rounds as the base despite being slower; if it has not beaten the kernel it
displaced by the end of that window, the old base is restored. Do NOT set it for
tuning, fusion or parameter changes -- those should win immediately.

CRITICAL:
- The "method_name" field MUST exactly match one of the method IDs listed in the allowed_methods section of MACHINE_CHECK RESULT. You CANNOT choose a method that is not in that list.

Rules:
- Return exactly one JSON object and nothing else.
- headroom MUST match Tier-H / Tier-M / Tier-L as defined in the Headroom & Feasibility Gate.
"""
    )
)

# -----------------------------------------------------------------------------
# Instruction prompt injects code, metrics, GPU spec
# -----------------------------------------------------------------------------

instruction_tmpl = Template(
    dedent(
        """
# Target GPU
GPU Name: $gpu_name
Architecture: $gpu_arch
Details:
$gpu_items


# Pytorch Reference
$python_code


# CUDA candidate
```python
$CUDA_CODE
```

# HARD CONSTRAINTS (from benchmark rules)
- Do NOT modify or redefine `get_inputs()` or `get_init_inputs()`.
- Assume inputs are already on the correct device; do NOT move tensors across devices in forward.
- Output tensors must be allocated on the SAME device as inputs.
- In-place is disallowed by default; only allowed if you can prove no re-read and no aliasing.

SPECIALIZATION FOCUS (CRITICAL):
- You are optimizing for the SPECIFIC input tensor dimensions shown in the CUDA candidate above, NOT for a generic kernel.
- You MAY specialize aggressively: unroll loops for known sizes, use compile-time constants, hardcode block sizes optimized for these exact shapes.
- You MAY assume tensor shapes are fixed as shown in the CUDA candidate's forward() and get_inputs().
- This specialization is ENCOURAGED to maximize speedup for this specific case.

# Nsight Compute metrics and analysis
$NCU_METRICS

The metrics above include:
- **Metrics (JSON)**: Quantitative performance data in JSON format
- **Detailed Section Analysis**: Textual analysis from Nsight Compute sections with specific recommendations

**IMPORTANT**: 
- Use OPT/INF only as supporting context; the primary decision MUST be driven by numeric metrics and kernel code.
- Section analysis text is ADVISORY ONLY and may not always be accurate. Do NOT over-rely on section analysis conclusions - verify them against numeric metrics.
- Pay attention to: Occupancy (achieved vs theoretical), memory hierarchy throughput (L1/L2/DRAM), SM throughput, and launch/resource limits.

$NSYS_LAUNCH_COUNTS

$OPTIMIZATION_HISTORY

# HISTORY ENFORCEMENT (HARD)
If optimization history exists:
- Do NOT repeat a mechanism-identical plan from history.
- Do NOT relabel the same mechanism as a different method_name.
- If you reuse a method, you MUST state a concrete delta vs a specific Previous Attempt number.

# MACHINE_CHECK RESULT (Deterministic Gating)
The following machine-check analysis has been performed on the metrics to determine allowed optimization methods:

**Tier (Headroom)**: $MACHINE_CHECK_TIER
**Bottleneck ID**: $MACHINE_CHECK_BOTTLENECK
**Case ID**: $MACHINE_CHECK_CASE
**Kernel Structure**: $MACHINE_CHECK_KERNEL_STRUCTURE
**Key Metrics**:
$MACHINE_CHECK_KEY_METRICS

**ALLOWED METHODS**:
$ALLOWED_METHODS_LIST

$FORBIDDEN_METHODS_SECTION

$CASE_REQUIREMENTS_SECTION

$GLOBAL_FORBIDDEN_RULES_SECTION

$METHOD_CATALOG_SECTION

## Evidence and Method Selection Requirements:
- Your "evidence" field MUST quote numeric values copied from the provided metrics JSON (not estimates).
- You MUST rule out at least one plausible bottleneck using those numbers.
- Your "method_name" MUST be one of the allowed_methods listed above (this is a HARD CONSTRAINT from machine_check).

Read everything and follow the Rules exactly. Return the JSON in the specified format.
"""
    )
)

# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

def build_judger_optimization_prompts(
    *,
    arch_path: Path,
    gpu_name: str,
    ncu_metrics_block: str,
    metrics_df: Optional[pd.DataFrame] = None,
    cuda_code: str = "",
    optimization_history: Optional[list] = None,
    code_features: Optional[Dict[str, Any]] = None,
        call_llm: Optional[Callable] = None,  # Signature: (prompt, sys_prompt, log_path=None, call_type="unknown", round_idx=-1) -> str
    nsys_csv_path: Optional[Path] = None,  # Optional path to nsys CSV file with kernel_launch_count
    io_dir: Optional[Path] = None,  # Optional directory to save machine_check_result JSON
    round_idx: Optional[int] = None,  # Optional round index for filename
    base_optimizations: Optional[list] = None,  # Accepted optimizations the base kernel carries
    rejected_metrics_block: Optional[str] = None,  # NCU profile of the previous round's rejected kernel
    rejected_kernel_name: Optional[str] = None,
    rejected_kernel_score: Optional[float] = None,
    rejected_base_score: Optional[float] = None,  # base score it was compared against
) -> Tuple[str, str]:
    """Return (system_prompt_str, instruction_str) for single-issue optimisation.

    Args:
        arch_path:   Path to .py that contains the PyTorch reference `class Model`
        gpu_name:    Key in GPU_SPEC_INFO (e.g., "Quadro RTX 6000")
        ncu_metrics_block: Text/Markdown block of Nsight Compute metrics
        metrics_df:  Optional DataFrame with NCU metrics (for machine_check)
        cuda_code:   Optional current CUDA candidate source (string)
        optimization_history: Optional list of dicts, each containing:
            - "round": round index
            - "optimization_strategy": the strategy JSON from LLM
            - "speedup": speedup value (None if test failed)
            - "test_passed": whether test passed
            - "repaired": whether it was repaired
        code_features: Optional dict with pre-extracted code features (kernel_structure, has_reuse, etc.)
                       If None, run_machine_check will extract them via judge_gate (if call_llm provided) or heuristic
        call_llm: Optional LLM call function (prompt, sys_prompt, log_path=None, call_type="unknown", round_idx=-1) -> str
                  Passed to run_machine_check for judge_gate-based code_features extraction
        nsys_csv_path: Optional path to nsys CSV file containing kernel_launch_count
                       If provided, kernel_launch_count will be read from this file instead of using len(rows)
    """
    gpu_info = _load_gpu_spec()
    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )

    arch_src = Path(arch_path).read_text().strip()

    # Run machine_check to get gate output (it will handle code_features extraction internally)
    machine_check_result = None
    allowed_methods_list = ""
    method_catalog_text = ""
    machine_check_tier = "Tier-M"
    machine_check_bottleneck = "unknown"
    machine_check_case = "NO_MATCH"
    machine_check_kernel_structure = "S0"
    machine_check_key_metrics = "N/A"
    # Initialize these variables to avoid UnboundLocalError
    forbidden_methods_list = ""
    case_requirements_text = ""
    global_forbidden_rules_text = ""

    if metrics_df is not None and not metrics_df.empty:
        try:
            # Create a temporary CSV file from metrics_df for machine_check ver2
            # machine_check ver2.run_machine_check requires a CSV file path
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as tmp_csv:
                tmp_csv_path = Path(tmp_csv.name)
                # Write CSV data
                if not metrics_df.empty:
                    metrics_df.to_csv(tmp_csv_path, index=False)
                else:
                    # Create minimal CSV with headers if empty
                    if len(metrics_df.columns) > 0:
                        writer = csv.DictWriter(tmp_csv, fieldnames=metrics_df.columns)
                        writer.writeheader()
            
            try:
                # Run machine_check ver2 (requires CSV path, not metric_row)
                # machine_check will handle code_features extraction via judge_gate if call_llm is provided
                # Try to load kernel_launch_count from nsys CSV if available
                kernel_launch_count_from_nsys = None
                if nsys_csv_path and Path(nsys_csv_path).exists():
                    try:
                        nsys_df = pd.read_csv(nsys_csv_path)
                        if not nsys_df.empty and "kernel_launch_count" in nsys_df.columns:
                            # Sum up launch counts for all kernels (if multiple)
                            kernel_launch_count_from_nsys = int(nsys_df["kernel_launch_count"].sum())
                            print(f"[machine_check] Using kernel_launch_count from nsys: {kernel_launch_count_from_nsys}")
                    except Exception as e:
                        print(f"[machine_check] Warning: Failed to read kernel_launch_count from nsys CSV: {e}")
                
                machine_check_result = run_machine_check(
                    yaml_path=YAML_RULES_PATH,
                    metric_csv_path=tmp_csv_path,
                    kernel_filter=None,  # Already filtered in metrics_df
                    cuda_code=cuda_code,
                    arch_path=arch_path,  # Pass arch_path for judge_gate
                    feature_mode="llm" if (call_llm is not None and cuda_code) else ("manual" if code_features else "heuristic"),
                    code_features=code_features,  # Pass if already extracted
                    call_llm=call_llm,  # Pass call_llm for judge_gate extraction
                    aggregate=False,  # metrics_df already contains filtered data
                    kernel_launch_count=kernel_launch_count_from_nsys,  # Pass nsys-derived launch count if available
                    io_dir=io_dir,  # Pass io_dir for saving judge_gate prompts and replies
                    round_idx=round_idx,  # Pass round_idx for filename
                )
            finally:
                # Clean up temporary CSV file
                try:
                    tmp_csv_path.unlink()
                except Exception:
                    pass
            
            # Record whether allowed_methods actually bound this round. Without
            # it a saved result is ambiguous after the fact: an in-table method
            # could mean the judge agreed with the table or merely obeyed it.
            if isinstance(machine_check_result, dict):
                machine_check_result["enforcement"] = ENFORCEMENT_MODE

            # Save machine_check_result to JSON file if io_dir and round_idx are provided
            if io_dir is not None and round_idx is not None:
                try:
                    io_dir.mkdir(parents=True, exist_ok=True)
                    machine_check_result_file = io_dir / f"round{round_idx:03d}_machine_check_result.json"
                    import json
                    with open(machine_check_result_file, 'w', encoding='utf-8') as f:
                        json.dump(machine_check_result, f, indent=2, ensure_ascii=False)
                    print(f"[machine_check] Saved machine_check_result to: {machine_check_result_file}")
                except Exception as e:
                    print(f"[machine_check] Warning: Failed to save machine_check_result JSON: {e}")
            
            # Extract machine_check fields
            machine_check_tier = machine_check_result.get("tier", "Tier-M")
            machine_check_bottleneck = machine_check_result.get("bottleneck_id", "unknown")
            machine_check_case = machine_check_result.get("case_id", "NO_MATCH")
            machine_check_kernel_structure = machine_check_result.get("kernel_structure", "S0")
            allowed_methods = machine_check_result.get("allowed_methods", [])
            forbidden_methods = machine_check_result.get("forbidden_methods", [])
            case_requirements = machine_check_result.get("requirements", {})
            key_metrics = machine_check_result.get("key_metrics", {})
            
            # Format allowed methods list
            if allowed_methods:
                allowed_methods_list = "\n".join(f"- {method_id}" for method_id in allowed_methods)
            else:
                allowed_methods_list = "- (No methods allowed - machine_check did not match any optimization methods. You should create your own optimization strategy based on the metrics and kernel code.)"
            
            # Format forbidden methods list
            forbidden_methods_list = ""
            if forbidden_methods:
                forbidden_methods_list = "\n".join(f"- {method_id}" for method_id in forbidden_methods)
            
            # Format case-level requirements
            case_requirements_text = ""
            if case_requirements:
                req_lines = []
                for method_id, req_list in case_requirements.items():
                    if isinstance(req_list, list):
                        req_lines.append(f"- **{method_id}**: {', '.join(req_list)}")
                    else:
                        req_lines.append(f"- **{method_id}**: {req_list}")
                case_requirements_text = "\n".join(req_lines)
            
            # Load global_forbidden_rules from YAML
            global_forbidden_rules_text = ""
            try:
                with open(YAML_RULES_PATH, 'r', encoding='utf-8') as f:
                    yaml_rules = yaml.safe_load(f)
                
                global_forbidden_rules = yaml_rules.get("machine_check", {}).get("global_forbidden_rules", [])
                if global_forbidden_rules:
                    rule_lines = []
                    for rule in global_forbidden_rules:
                        rule_id = rule.get("id", "Unknown")
                        rule_desc = rule.get("description", "")
                        rule_lines.append(f"- **{rule_id}**: {rule_desc}")
                    global_forbidden_rules_text = "\n".join(rule_lines)
            except Exception as e:
                print(f"[judger] Warning: Failed to load global_forbidden_rules: {e}")
            
            # Format key metrics
            if key_metrics:
                key_metrics_lines = []
                for k, v in key_metrics.items():
                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                        key_metrics_lines.append(f"  - {k}: {v}")
                machine_check_key_metrics = "\n".join(key_metrics_lines) if key_metrics_lines else "N/A"
            
            # Load YAML to extract method_catalog for allowed methods
            # Only load if allowed_methods is not empty
            method_catalog_text = ""
            if allowed_methods:
                try:
                    with open(YAML_RULES_PATH, 'r', encoding='utf-8') as f:
                        yaml_rules = yaml.safe_load(f)
                    
                    method_catalog = yaml_rules.get("llm_assist", {}).get("method_catalog", {})
                    
                    # Extract only allowed methods from catalog
                    method_catalog_lines = []
                    for method_id in allowed_methods:
                        if method_id in method_catalog:
                            method_info = method_catalog[method_id]
                            method_catalog_lines.append(f"**{method_id}**:")
                            if "intent" in method_info:
                                method_catalog_lines.append(f"  Intent: {method_info['intent']}")
                            if "mechanism_requirements" in method_info:
                                method_catalog_lines.append(f"  Requirements: {', '.join(method_info['mechanism_requirements'])}")
                            if "implementation_case" in method_info:
                                kps = method_info["implementation_case"]
                                if isinstance(kps, list):
                                    method_catalog_lines.append("  Reference implementation / instance knowledge:")
                                    for kp in kps:
                                        kp_str = kp if isinstance(kp, str) else str(kp)
                                        if "\n" in kp_str:
                                            method_catalog_lines.append("    ```")
                                            for line in kp_str.strip().split("\n"):
                                                method_catalog_lines.append("    " + line)
                                            method_catalog_lines.append("    ```")
                                        else:
                                            method_catalog_lines.append(f"    - {kp_str}")
                                else:
                                    method_catalog_lines.append(f"  Reference implementation: {kps}")
                            if "expected_metric_change" in method_info:
                                method_catalog_lines.append(f"  Expected changes: {', '.join(method_info['expected_metric_change'])}")
                            if "forbidden_patterns" in method_info:
                                method_catalog_lines.append(f"  Forbidden: {', '.join(method_info['forbidden_patterns'])}")
                            method_catalog_lines.append("")
                    
                    # Only set method_catalog_text if we found entries
                    if method_catalog_lines:
                        method_catalog_text = "\n".join(method_catalog_lines)
                    # If no entries found, leave it empty (will not show METHOD_CATALOG section)
                except Exception as e:
                    print(f"[judger] Warning: Failed to load method catalog: {e}")
                    method_catalog_text = ""
        except Exception as e:
            # If machine_check fails, continue with defaults but log warning
            import warnings
            warnings.warn(f"machine_check failed: {e}. Using defaults.")
            # Initialize allowed_methods as empty list (no methods matched)
            allowed_methods = []
            allowed_methods_list = "- (machine_check failed - all methods allowed as fallback)"
            method_catalog_text = ""  # Empty string, not a message, so METHOD_CATALOG section won't show
            # Ensure these variables are initialized even on failure
            forbidden_methods_list = ""
            case_requirements_text = ""
            global_forbidden_rules_text = ""
    else:
        # No metrics_df provided, use defaults
        allowed_methods = []  # Initialize as empty list
        allowed_methods_list = "- (No metrics provided - all methods allowed as fallback)"
        method_catalog_text = ""  # Empty string, so METHOD_CATALOG section won't show
        # Ensure these variables are initialized
        forbidden_methods_list = ""
        case_requirements_text = ""
        global_forbidden_rules_text = ""
    
    # Load nsys launch counts if nsys_csv_path is provided (outside of metrics_df check)
    nsys_launch_counts_text = ""
    if nsys_csv_path and Path(nsys_csv_path).exists():
        try:
            nsys_df = pd.read_csv(nsys_csv_path)
            if not nsys_df.empty and "kernel_launch_count" in nsys_df.columns:
                # Format kernel launch counts for prompt
                nsys_lines = ["# Kernel Launch Counts (from nsys profiling)", ""]
                for _, row in nsys_df.iterrows():
                    kernel_name = row.get("Kernel Name", "unknown")
                    launch_count = row.get("kernel_launch_count", 0)
                    nsys_lines.append(f"- **{kernel_name}**: {launch_count} launches")
                nsys_lines.append("")
                nsys_launch_counts_text = "\n".join(nsys_lines)
        except Exception as e:
            print(f"[judger] Warning: Failed to read nsys CSV for prompt: {e}")

    # Format optimization history
    opt_history_text = ""
    if optimization_history and len(optimization_history) > 0:
        opt_history_lines = [
            "# Optimization History (Previously Attempted Strategies)",
            "",
            "The following optimization strategies have been previously attempted on this kernel (or its parent kernel).",
            "Each entry shows the complete optimization strategy and its resulting speedup value.",
            "",
            "**CRITICAL ANALYSIS REQUIRED**:",
            "- For each previous attempt, determine if the low/negative speedup was due to:",
            "  (1) **Poor implementation** of an otherwise suitable method → Consider reusing the SAME method with IMPROVED implementation",
            "  (2) **Method mismatch** (method not suitable for this bottleneck/kernel) → Try a DIFFERENT method",
            "- Review the 'evidence' and 'expected_metric_change' fields to assess if the method was theoretically sound",
            "- Check if the method is still in the current allowed_methods list",
            "- If a method was tried multiple times with different implementations and all failed, it's likely a method mismatch",
            "",
        ]
        
        for idx, hist in enumerate(optimization_history, 1):
            round_num = hist.get("round", "?")
            strategy = hist.get("optimization_strategy", {})
            speedup = hist.get("speedup")
            test_passed = hist.get("test_passed", False)
            repaired = hist.get("repaired", False)
            
            opt_history_lines.append(f"## Previous Attempt #{idx} (Round {round_num})")
            
            if isinstance(strategy, dict):
                # Include all fields from optimization_strategy for completeness
                bottleneck = strategy.get("bottleneck", "N/A")
                # Two output schemas are asked for above -- underscored (line 119/121)
                # and spaced (line 387/389) -- and opt_round_*.json stores whichever
                # the judge emitted, in practice the underscored form. Only the spaced
                # spellings were looked up here, so both always missed and every prior
                # attempt rendered as "N/A". That blanked the only two fields that say
                # what was actually DONE, while bottleneck/evidence/expected_metric_change/
                # headroom matched and came through -- leaving each round able to see a
                # past attempt's diagnosis and result but never its change, so it could
                # neither iterate on a near-miss nor avoid re-proposing one. Read both.
                method = strategy.get("primary_optimisation_method",
                                      strategy.get("optimisation method",
                                                   strategy.get("optimization method", "N/A")))
                method_name = strategy.get("method_name", "N/A")
                plan = strategy.get("modification_plan",
                                    strategy.get("modification plan", "N/A"))
                evidence = strategy.get("evidence", "N/A")
                expected_metric_change = strategy.get("expected_metric_change", "N/A")
                headroom = strategy.get("headroom", "N/A")
                
                opt_history_lines.append(f"- **Bottleneck identified**: {bottleneck}")
                opt_history_lines.append(f"- **Optimization method**: {method}")
                opt_history_lines.append(f"- **Method name**: {method_name}")
                opt_history_lines.append(f"- **Modification plan**: {plan}")
                opt_history_lines.append(f"- **Evidence**: {evidence}")
                opt_history_lines.append(f"- **Expected metric change**: {expected_metric_change}")
                opt_history_lines.append(f"- **Headroom**: {headroom}")
            else:
                opt_history_lines.append(f"- **Strategy**: {str(strategy)[:200]}...")
            
            if speedup is not None:
                # "PASSED" only ever meant "compiled and was numerically correct", so an
                # attempt that regressed 6% against the base still read as PASSED. State the
                # verdict against the base the attempt actually had to beat.
                _base_at = hist.get("base_score_at_attempt")
                if _base_at:
                    _d = (speedup / _base_at - 1.0) * 100.0
                    _verdict = "IMPROVED" if _d > 0 else "REGRESSED"
                    opt_history_lines.append(
                        f"- **Result**: {speedup:.4f} vs base {_base_at:.4f} -> {_verdict} "
                        f"{_d:+.2f}% ({'ran correctly' if test_passed else 'FAILED correctness'})")
                else:
                    opt_history_lines.append(
                        f"- **Result**: Speedup = {speedup:.4f} "
                        f"({'ran correctly' if test_passed else 'FAILED correctness'})")
                # Where it won and where it lost. This is the cheapest signal that separates
                # "the mechanism is wrong" from "the mechanism works but needs a shape gate":
                # a method that gains on some shapes and collapses on one should be GATED,
                # not abandoned.
                _psb = hist.get("per_shape_vs_base_pct") or []
                if _psb:
                    _cells = "  ".join(f"{r['shape']}={r['delta_pct']:+.1f}%" for r in _psb)
                    opt_history_lines.append(f"- **Per-shape vs base**: {_cells}")
                    _won = [r["shape"] for r in _psb if r["delta_pct"] > 0]
                    if _won and len(_won) < len(_psb):
                        opt_history_lines.append(
                            f"- **Read this**: gained on {len(_won)}/{len(_psb)} shapes "
                            f"({', '.join(_won)}). A mechanism that helps some shapes and hurts "
                            f"others is usually applied unconditionally -- consider GATING it on "
                            f"size/shape rather than discarding the mechanism.")
            else:
                opt_history_lines.append(f"- **Result**: Test {'PASSED but no speedup recorded' if test_passed else 'FAILED'}")
            
            if repaired:
                opt_history_lines.append(f"- **Note**: This attempt required repair to pass tests")
            
            opt_history_lines.append("")
        
        # Guidance is already provided in system prompt, no need to repeat here
        
        opt_history_text = "\n".join(opt_history_lines)
    else:
        opt_history_text = "# Optimization History\n(No previous optimization attempts recorded for this kernel.)\n"

    # The CUDA code above is the BASE kernel, which accumulates the mechanism of every
    # round that beat it. Nothing in the source marks which parts are load-bearing, so
    # the judge cannot distinguish tuned code from ordinary code and readily proposes
    # replacing a measured win with an untuned guess. Listing them makes the tuned
    # mechanisms explicit.
    #
    # Deliberately about MECHANISMS, not code regions: an already-optimised region can
    # still have headroom (a stream-overlapped GroupNorm chain may still run at 85% DRAM
    # with a 1.6% L2 hit rate), so forbidding work there would fence off real gains.
    # What must not happen is a mechanism being dropped WITHOUT the trade being stated.
    if base_optimizations:
        _lines = []
        for o in base_optimizations:
            g = o.get("gain_pct")
            _lines.append(
                f"  - {o.get('method_name')}"
                + (f"  (round {o.get('round')}" if o.get("round") is not None else "  (")
                + (f", +{g:.2f}% over its parent)" if g is not None else ")")
                + (f"\n      {o['summary']}" if o.get("summary") else "")
            )
        opt_history_text += f"""

================================================================================
OPTIMIZATIONS THE BASE KERNEL ALREADY CARRIES
================================================================================
Each of these was accepted in an earlier round because it MEASURABLY improved
this kernel, and each is implemented in the CUDA code above:

{chr(10).join(_lines)}

HOW THIS CONSTRAINS YOUR CHOICE (this is about MECHANISMS, not code regions):

  * You MAY optimise a code region one of these already touches. "Already
    optimised" does NOT mean "out of headroom" -- a region can be tuned and
    still show a dominant stall, low cache hit rate, or high DRAM traffic in the
    metrics above. If the evidence shows room, go after it. Making that region
    faster WHILE KEEPING the listed mechanism working is composition, and is the
    single most valuable thing you can do.

  * You MAY NOT drop one of these mechanisms without accounting for it. If your
    plan disables, bypasses, replaces, or makes unreachable any listed
    optimisation, you MUST:
      1. name which one your plan removes,
      2. explain why your mechanism subsumes it or makes it unnecessary,
      3. put in `expected_metric_change` the specific metric that would reveal
         the loss if you are wrong -- e.g. kernel launch count rising back,
         conv/normalisation overlap disappearing from the timeline, an autotuned
         vendor engine reverting to a default plan, a fused kernel splitting.
    Replacement is permitted. UNACCOUNTED replacement is what is forbidden.

  * Watch for silent incompatibility, which is the most common way this fails:
    a mechanism can disable another without your plan ever mentioning it (a
    capture/replay scheme that requires static buffers defeats per-call dynamic
    chunking; a cached fixed plan defeats per-shape re-selection; a grid-wide
    barrier defeats multi-stream execution). Before finalising, check each
    listed mechanism and confirm your plan leaves it operational.
================================================================================
"""

    # The NCU block earlier in this prompt profiles the BASE. This one profiles the
    # kernel the PREVIOUS round produced and that lost -- what it actually did on the
    # GPU, as opposed to what its strategy predicted. Without it the judge is asked to
    # classify implementation-failure vs method-mismatch (see CRITICAL ANALYSIS above)
    # from a single scalar, which it cannot do, so it defaults to inventing a new
    # mechanism every round.
    if rejected_metrics_block:
        _score_txt = (f" (scored {rejected_kernel_score:.4f})"
                      if rejected_kernel_score is not None else "")
        # A kernel can fail to be adopted for two very different reasons, and describing
        # them identically produces confidently WRONG feedback: telling the judge that a
        # kernel which actually scored HIGHER "lost" makes it invent a failure explanation
        # for something that mildly worked, and abandon a direction worth refining.
        _improved_but_unproven = (
            rejected_kernel_score is not None and rejected_base_score
            and rejected_kernel_score >= rejected_base_score)
        if _improved_but_unproven:
            _delta = (rejected_kernel_score / rejected_base_score - 1.0) * 100.0
            _headline = "NOT ADOPTED (improvement too small to be resolvable)"
            _framing = f"""This kernel RAN CORRECTLY and actually scored *HIGHER* than the base
({rejected_kernel_score:.4f} vs {rejected_base_score:.4f}, {_delta:+.2f}%), but by less than the
margin required to treat the difference as real, so the base was not replaced.

IMPORTANT: this attempt did NOT fail. Its direction was mildly positive but is
indistinguishable from measurement noise. Do NOT diagnose it as a failure and do
NOT abandon its mechanism on this evidence. The two useful moves are: strengthen
the SAME mechanism so the effect becomes large enough to measure, or pick an
independent target -- but do not conclude the direction was wrong."""
        else:
            _headline = "REJECTED"
            _framing = """This kernel RAN CORRECTLY but scored below the base, so the base was NOT replaced."""

        if _improved_but_unproven:
            _diagnosis = """DO THIS BEFORE CHOOSING YOUR METHOD:
1. Find that attempt's `expected_metric_change` in the history above.
2. Compare it against the measured numbers here and decide which is true:

   (i)  The predicted metric moved as intended
        -> The mechanism WORKS, it is just too small to measure. Making it bigger
           is the highest-value move available: widen where it applies, remove a
           gate that is excluding shapes, or attack the same bottleneck harder.

   (ii) The predicted metric did NOT move
        -> The gain came from somewhere other than your mechanism, i.e. it is
           probably noise. Treat the direction as unproven and pick a target with
           independent evidence.

3. State which of (i)/(ii) you concluded and cite the deciding metric in `evidence`.
   Do NOT describe this attempt as a failure -- it was not one."""
        else:
            _diagnosis = """DO THIS BEFORE CHOOSING YOUR METHOD:
1. Find that attempt's `expected_metric_change` in the history above.
2. Compare it against the measured numbers here, and classify the failure:

   (a) The predicted metric DID move as intended, but the kernel was still slower
       -> The bottleneck model was WRONG. That direction is a dead end. Do NOT
          re-attempt this mechanism "with a better implementation" -- pick a
          different target with independent evidence.

   (b) The predicted metric did NOT move as intended
       -> The mechanism never really engaged: a fallback path was taken, a guard
          never fired, a cooperative/graph launch was refused, a gate excluded the
          shape. The method may still be sound. ONE retry is justified, and you
          must name the specific defect you are fixing.

   (c) It improved some shapes and collapsed others (see the per-shape line)
       -> The mechanism works but is applied unconditionally. Do NOT discard it.
          Add the missing size/shape gate so it engages only where it pays.

3. Say which of (a)/(b)/(c) you concluded and cite the deciding metric in `evidence`."""
        opt_history_text += f"""

================================================================================
MEASURED PROFILE OF THE PREVIOUS ATTEMPT -- {rejected_kernel_name or 'previous round'}{_score_txt} -- {_headline}
================================================================================
{_framing}
Everything below is what it ACTUALLY did on the GPU, not what its strategy predicted.

{rejected_metrics_block.strip()}

{_diagnosis}

ALSO CHECK WHAT IT BROKE. Compare this profile against the base's profile above.
If a mechanism the base carries has disappeared here -- launch count climbed back,
kernels that were fused are separate again, an autotuned engine reverted to a
default plan, conv/normalisation overlap vanished -- then that attempt failed
because it was structurally incompatible with the base, NOT because its target was
wrong. In that case the useful move is a mechanism that COMPOSES with what the base
already does, and you should say so explicitly in `evidence`.
================================================================================
"""

    # Choose template based on machine_check result
    if machine_check_case == "NO_MATCH":
        # Use simplified templates for NO_MATCH case
        system_prompt = system_prompt_tmpl_no_match.substitute()
        instruction = instruction_tmpl_no_match.substitute(
            gpu_name=gpu_name,
            gpu_arch=gpu_arch,
            gpu_items=gpu_items,
            python_code=arch_src,
            CUDA_CODE=cuda_code.strip(),
            NCU_METRICS=ncu_metrics_block.strip(),
            NSYS_LAUNCH_COUNTS=nsys_launch_counts_text,
            OPTIMIZATION_HISTORY=opt_history_text,
        )
        return (system_prompt, instruction)
    
    # For matched case, continue with full template preparation
    system_prompt = _apply_enforcement_mode(
        system_prompt_tmpl.substitute(), _ADVISORY_REWRITES_SYSTEM
    )
    # Format forbidden methods section
    if forbidden_methods_list:
        forbidden_methods_section = f"""
**FORBIDDEN METHODS** (You MUST NOT select any method from this list):
{forbidden_methods_list}
"""
    else:
        forbidden_methods_section = ""
    
    # Format case requirements section
    if case_requirements_text:
        case_requirements_section = f"""
**CASE-LEVEL REQUIREMENTS** (Additional requirements for specific methods):
{case_requirements_text}

**IMPORTANT**: If you select a method that has case-level requirements listed above, you MUST ensure your implementation satisfies ALL of those requirements. If you cannot satisfy the requirements, you MUST choose a different method from the allowed_methods list.
"""
    else:
        case_requirements_section = ""
    
    # Format global forbidden rules section
    if global_forbidden_rules_text:
        global_forbidden_rules_section = f"""
**GLOBAL FORBIDDEN RULES** (Universal constraints that apply to ALL methods):
{global_forbidden_rules_text}

**IMPORTANT**: These rules apply regardless of which method you select. You MUST ensure your implementation does not violate any of these rules.
"""
    else:
        global_forbidden_rules_section = ""
    
    # Format method catalog section (only if method_catalog_text is not empty)
    if method_catalog_text:
        method_catalog_section = f"""
**CRITICAL CONSTRAINT**: Your output JSON MUST include a "method_name" field that exactly matches one of the method IDs listed in the ALLOWED METHODS section above. You CANNOT choose a method that is not in the allowed_methods list.

# LLM_ASSIST Knowledge Base (for understanding "how to do" and "why to do")
The following knowledge is provided to help you understand and choose among the allowed methods:

## Method Catalog (for allowed methods only)
The following are brief descriptions of the allowed methods. Use these to understand:
- **What each method means** (intent)
- **How to implement it** (mechanism_requirements)
- **What to expect** (expected_metric_change)
- **What to avoid** (forbidden_patterns)

{method_catalog_text}

**IMPORTANT:**
- The Method Catalog is KNOWLEDGE to help you understand and choose from allowed_methods.
- It does NOT determine what is allowed - only machine_check does that.
- Use the Method Catalog to:
  * Select the most appropriate method from the allowed_methods list
  * Write better evidence by understanding expected metric changes
  * Write better modification plans by understanding mechanism requirements
  * Avoid forbidden patterns when implementing the method
- Use the Method Catalog to help you write evidence that aligns with the selected method's expected_metric_change.
"""
    else:
        # No method catalog available (either no allowed_methods or no matching entries in catalog)
        if allowed_methods:
            # Has allowed_methods but no catalog entries - still need constraint
            method_catalog_section = """
**CRITICAL CONSTRAINT**: Your output JSON MUST include a "method_name" field that exactly matches one of the method IDs listed in the ALLOWED METHODS section above. You CANNOT choose a method that is not in the allowed_methods list.
"""
        else:
            # No allowed_methods - this should not happen in matched case, but keep constraint
            method_catalog_section = """
**CRITICAL CONSTRAINT**: Your output JSON MUST include a "method_name" field that exactly matches one of the method IDs listed in the ALLOWED METHODS section above. You CANNOT choose a method that is not in the allowed_methods list.
"""
    
    # Build instruction for matched case
    instruction = instruction_tmpl.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        python_code=arch_src,
        CUDA_CODE=cuda_code.strip(),
        NCU_METRICS=ncu_metrics_block.strip(),
        NSYS_LAUNCH_COUNTS=nsys_launch_counts_text,
        OPTIMIZATION_HISTORY=opt_history_text,
        MACHINE_CHECK_TIER=machine_check_tier,
        MACHINE_CHECK_BOTTLENECK=machine_check_bottleneck,
        MACHINE_CHECK_CASE=machine_check_case,
        MACHINE_CHECK_KERNEL_STRUCTURE=machine_check_kernel_structure,
        MACHINE_CHECK_KEY_METRICS=machine_check_key_metrics,
        ALLOWED_METHODS_LIST=allowed_methods_list,
        FORBIDDEN_METHODS_SECTION=forbidden_methods_section,
        CASE_REQUIREMENTS_SECTION=case_requirements_section,
        GLOBAL_FORBIDDEN_RULES_SECTION=global_forbidden_rules_section,
        METHOD_CATALOG_SECTION=method_catalog_section,
    )
    instruction = _apply_enforcement_mode(instruction, _ADVISORY_REWRITES_INSTRUCTION)
    return system_prompt, instruction