from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template
import json

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # Adjust import path as needed

_OPTIMIZATION_PROMPT_TEMPLATE = Template("""

You are a CUDA kernel optimization implementation specialist. You will receive:
- Target GPU specifications
- A **base CUDA kernel file** (Python + CUDA extension) containing the current `ModelNew` implementation and its custom CUDA kernels.
- A previously generated optimisation_suggestion containing:
  - **modification_plan** (or "modification plan"): numbered checklist with primary optimization method steps
  - bottleneck, primary_optimisation_method (or "optimisation method"), method_name, evidence, expected_metric_change, headroom

Your MOST IMPORTANT goal is:
- To implement the given **modification plan** completely and faithfully in the CUDA extension, producing a new kernel that:
  - Applies the primary optimization method to the hottest kernel
  - Preserves correctness within the harness tolerance (atol=1e-3 and rtol=1e-3),
  - Preserves the public Python API (same inputs/outputs, shapes, dtypes),
  - And improves performance on the target GPU as much as possible.

PARAMETER PARITY (STRICT)
- You MUST preserve parameter parity with the base kernel's parameter structure.
- The base kernel file already contains a `ModelNew` class with parameter handling. You MUST maintain the same parameter structure.
- If the base kernel uses a PyTorch module as a PARAMETER HOLDER (for initialization / state_dict parity), you MUST keep the same pattern.
- Do NOT create independent parameters with torch.randn / random init. Source weights/bias from the existing parameter-holder module (e.g., self.op.weight / self.op.bias) as the base kernel does.

NO ATen FOR REPLACED OPS (STRICT)
- For op-chains explicitly marked as replaced/fused in the modification_plan, you MUST NOT call PyTorch/ATen implementations for those chains. Ops explicitly kept as vendor mainloops (e.g., cuBLAS/cuDNN paths) may remain as in the base kernel.
- The replaced computation must execute only in your custom CUDA kernel(s), plus minimal checks.

CUDA GRAPH METHOD (ONLY IF REQUIRED BY PLAN)
- If method_name (or primary_optimisation_method) is cuda_graph_capture, you MAY implement torch.cuda.CUDAGraph capture/replay in ModelNew (Python) using static CUDA tensors.
- Do not use CUDA graphs otherwise.
- Do not place custom extension compilation inside the captured region.

You MUST follow these implementation and explanation rules:

(0) IMPLEMENTATION ORDER (CRITICAL - STRICT SEQUENCE)
- **Primary Optimization Method**
  - Apply the primary optimization method to the hottest kernel as specified in the modification_plan.
  - The modification_plan lists the optimization steps (e.g., "1. Apply vectorized memory access to kernel X").
  - Implement each step in the modification_plan completely and faithfully.

(0.1) MODIFICATION PLAN EXECUTION (CRITICAL)
- Treat the `modification_plan` (or "modification plan") in optimisation_suggestion as a **checklist** of required changes.
- For each numbered item in the plan (1., 2., 3., ...), you MUST:
  - Implement the corresponding change in the CUDA / Python code; and
  - Be able to point to the exact code region(s) (functions / loops / kernels / kernel launches) that realize this item.
- You MUST NOT silently drop plan items. If a plan item (or a sub-step) is impossible/unsafe, state it in Section A in ONE short sentence and implement the closest safe subset + a concrete TODO.

(1) OUTPUT STRUCTURE (TWO SECTIONS WITH EXPLICIT DELIMITERS)
Your answer MUST consist of **two sections in order**, separated by explicit delimiters:

- **Section A – Checklist evidence (plan-to-code mapping)**  
  - For **each** numbered item in the modification plan, write a **very brief** mapping:
    - `Plan item N:` the exact text (or a concise paraphrase) of that item;
    - `Implemented in code:` **one short phrase** naming the key code location(s) (e.g., "new grid-stride loop in main kernel", "float4 load/store path in kernel X", "launch config in ModelNew.forward").
  - Keep Section A compact: do **not** explain low-level details here; it is only a checklist for where each item was implemented.
  - If you deviated from the literal wording of the plan, mention this in **one short sentence**.
  - Section A must NOT include rationale, performance claims, or metric discussion; only plan-item-to-code-location mapping.

- **Section B – Kernel implementation**  
  - Start this section with the delimiter line: `=== KERNEL CODE STARTS BELOW ===`
  - Then output a complete CUDA extension implementation following the `load_inline + ModelNew` structure described in OUTPUT RULES below.
  - This kernel should fully implement all items from the modification plan as explained in Section A.
  - The kernel code must be wrapped in ```python ... ``` code block.

(2) MINIMAL-DIFF PRINCIPLE (RELATIVE TO BASE KERNEL)
- Keep the overall structure and public interface of the base kernel stable. Do not redesign unrelated parts.
- Focus your edits on the parts of the code that are directly related to the modification plan.

(3) SEMANTIC ALIGNMENT
- Your optimized kernel must produce the same outputs as the base kernel for the same inputs (within tolerance).

(4) BENCH HARNESS INVARIANTS (STRICT)
- Follow the same device handling pattern as the base kernel. Assume inputs are already on the correct device. In ModelNew.forward, do NOT call .to("cuda"), .cuda(), .cpu(), or .to(x.device).
- Allocate outputs on the SAME device as the input tensors.
- If a tensor is non-contiguous, call contiguous() only when the CUDA kernel uses raw linear indexing that assumes contiguous layout; otherwise implement stride-aware indexing.
- Do NOT propose or implement changes to the benchmark workflow, external caching, or evaluation settings; only kernel code and launch configuration may change.
- Recommended pattern: in Python `x = x if x.is_contiguous() else x.contiguous()` only when the corresponding CUDA kernel uses linear indexing.

- If headroom is provided in optimisation_suggestion, consider it when choosing implementation details:
  - High headroom: more aggressive optimizations may be appropriate;
  - Medium headroom: balanced approach;
  - Low headroom: focus on small, safe efficiency tweaks; expect modest speedup


# Target GPU
GPU Name: $gpu_name
Architecture: $gpu_arch
Details:
$gpu_items
$cutlass_block
$lessons_block
$pathway_block

[BASE KERNEL FILE]
This is the current CUDA kernel implementation that you must optimize. Study its structure, parameter handling, and kernel implementations before making changes.
```python
$arch_src
```

[optimization instructions]
$optimization_suggestion

COMPLETE OUTPUT FORMAT ────────────────────────────────────────────────
Your complete response MUST include BOTH sections:

**Section A – Checklist evidence (plan-to-code mapping)**
- Write this section FIRST, before any code.
- For each numbered item in the modification plan, provide a **very concise** mapping:
  - `Plan item N:` [exact text or paraphrase, one line]
  - `Implemented in code:` [one short phrase naming the key code location(s)]
- Do not write long explanations here; this section is a compact checklist only.

**Section B – Kernel implementation**
- After Section A, write the delimiter: `=== KERNEL CODE STARTS BELOW ===`
- Then output the kernel code block following OUTPUT RULES below.

PRECISION POLICY (STRICT) ──────────────────────────────────────────────
- Compute in the dtype of the input tensors. NEVER downcast to a narrower dtype for speed.
- float32 inputs => float32 storage and arithmetic throughout. No __half / half2 /
  __nv_bfloat16 / fp8, and no reduced-precision tensor-core engines.
- float16/bfloat16 inputs => use that dtype natively; do not upcast the pipeline either.
- TF32 for conv/GEMM is allowed (the reference runs TF32 via cudnn.allow_tf32=True).
- Reductions must accumulate in float32 or wider regardless of storage dtype.
- Speed must come from scheduling, fusion and reduced memory traffic — NOT from doing
  less precise arithmetic. A narrower dtype can pass the tolerance check while computing
  something the reference did not; that is not a valid optimization.

OUTPUT RULES FOR KERNEL CODE (Section B only) ──────────────────────────
The kernel code block in Section B must follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple-quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors the base kernel's inputs/outputs and API, but calls
      your optimized CUDA kernels.
**Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose in the code block.

Example complete output structure:
```
[Section A: Checklist evidence]
Plan item 1: ...
Implemented in code: ...

Plan item 2: ...
Implemented in code: ...

=== KERNEL CODE STARTS BELOW ===
```python
import torch
...
```
""")

def _escape_template(s: str) -> str:
    return s.replace("$", "$$")

def _sanitize_text(s: str) -> str:
    return s.replace("```", "`")

def _format_problem(problem: Optional[Any]) -> str:
    if problem is None or problem == "":
        return "No prior critical problem provided."
    if isinstance(problem, Mapping):
        # Extract all fields from the new JSON format
        result_dict = {}
        
        # Legacy fields (for backward compatibility)
        if problem.get("bottleneck"):
            result_dict["bottleneck"] = str(problem.get("bottleneck", "")).strip()
        if problem.get("optimisation method"):
            result_dict["optimisation method"] = str(problem.get("optimisation method", "")).strip()
        if problem.get("primary_optimisation_method"):
            result_dict["primary_optimisation_method"] = str(problem.get("primary_optimisation_method", "")).strip()
        if problem.get("method_name"):
            result_dict["method_name"] = str(problem.get("method_name", "")).strip()
        if problem.get("modification plan"):
            result_dict["modification plan"] = str(problem.get("modification plan", "")).strip()
        if problem.get("modification_plan"):
            result_dict["modification_plan"] = str(problem.get("modification_plan", "")).strip()
        if problem.get("evidence"):
            result_dict["evidence"] = str(problem.get("evidence", "")).strip()
        if problem.get("expected_metric_change"):
            result_dict["expected_metric_change"] = str(problem.get("expected_metric_change", "")).strip()
        if problem.get("headroom"):
            result_dict["headroom"] = str(problem.get("headroom", "")).strip()
        
        if result_dict:
            # Use json.dumps to properly escape strings and format JSON
            return json.dumps(result_dict, ensure_ascii=False, indent=2)
        
        # fallback to JSON dump of entire problem
        return json.dumps(problem, ensure_ascii=False, indent=2)
    # For other types, convert to string directly
    return str(problem)

def build_optimization_prompt(
    arch_path: Path,
    gpu_name: Optional[str] = None,
    *,
    history_block: str = "",
    pathway_block: str = "",
    optimization_suggestion: Optional[Any] = None,
) -> str:
    """Build LLM prompt for CUDA-kernel optimisation (optimization phase)."""
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    from prompts.generate_custom_cuda_memory import resolve_gpu_name
    gpu_name = resolve_gpu_name(gpu_name, gpu_info)

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")

    arch_src = Path(arch_path).read_text().strip()
    optimization_suggestion_text = _format_problem(optimization_suggestion)
    # The capability advertisement used to live only in the SEED prompt, so every
    # optimization round after round 0 was blind to CUTLASS -- i.e. 25 of the 26
    # rounds of the best run so far. Emit it here too; it is "" when CUTLASS is
    # absent or does not compile, so this is a no-op on machines without it.
    from prompts.generate_custom_cuda_memory import _cutlass_block
    # Long-term memory: findings recorded on THIS task by previous runs. Advisory,
    # and "" when the task has no lessons file or MEMORYBANK_LESSONS=0 -- an empty
    # heading is worse than silence, because the model tries to honour it. A
    # failure to read it must never cost a round, so it degrades to "".
    try:
        from utils.memorybank_lessons import render as _render_lessons
        lessons_block = _render_lessons(Path(arch_path).stem)
    except Exception as _exc:                                  # pragma: no cover
        print(f"[lessons] skipped ({_exc})", flush=True)
        lessons_block = ""
    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        cutlass_block=_cutlass_block(),
        lessons_block=lessons_block,
        arch_src=arch_src,
        history_block="",  # Not used anymore, kept for backward compatibility
        # Just-in-time memory: where this run's search actually is on the map.
        # Built by the caller (it needs the live MCGS graph, which this module has
        # no business importing) and "" when path memory is off or the graph has
        # no pathway yet -- so this is a no-op on round 1 and under the ratchet.
        pathway_block=pathway_block or "",
        optimization_suggestion=optimization_suggestion_text,
    )
