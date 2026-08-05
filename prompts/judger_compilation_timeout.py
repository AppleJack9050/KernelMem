# prompts/judger_compilation_timeout.py
"""
Prompt template for analyzing timeout issues (compilation or execution).
This is for kernels that timeout during pre-compilation, profiling, or execution phases.
The timeout may be caused by compilation issues OR runtime issues.
"""
from __future__ import annotations
from pathlib import Path
from string import Template

# -----------------------------
# system_prompt as Template
# -----------------------------
system_prompt_tmpl = Template(
    """You are a senior CUDA expert specializing in both compilation issues and runtime performance problems.

Your task: Analyze a CUDA kernel that **times out** (exceeds 20 minutes). The timeout may occur during:
- **Compilation phase**: Code takes too long to compile
- **Execution phase**: Kernel execution is too slow or hangs

Common causes of timeout:

**A. Compilation-related:**
1. **Infinite template recursion** - Recursive template instantiation without proper termination
2. **Exponential template expansion** - Templates that expand exponentially with parameters
3. **Excessive constexpr evaluation** - Complex compile-time computations
4. **Large loop unrolling** - `#pragma unroll` with huge iteration counts
5. **Massive inline expansion** - Very large functions marked `__forceinline__`
6. **Compiler bugs** - Specific code patterns that trigger compiler inefficiencies
etc.    
**B. Execution-related:**
1. **GPU runtime errors** - Illegal memory access, out-of-bounds indexing causing hangs
2. **Poor performance** - Extremely low GPU occupancy due to excessive resource usage
3. **Infinite loops** - Loops without proper termination conditions
4. **Deadlocks** - Synchronization issues in shared memory or warp-level operations
5. **Memory conflicts** - Bank conflicts or uncoalesced memory access causing severe slowdowns
etc. 

Compile-time triage checklist (scan CUDA_CODE for these red flags):
- Heavy templates: "template<", deeply nested types, many template parameters
- Large includes: constexpr-heavy headers, metaprogramming utilities
  NOTE: a CUTLASS/CuTe include legitimately costs ~50s. That is EXPECTED COST, not a
  root cause. Do NOT propose removing CUTLASS unless the log shows the build actually
  exceeded its limit AND the rest of the TU is otherwise cheap. Replacing a vendor
  GEMM/conv with CUTLASS is a sanctioned optimisation; prefer raising the compile
  budget or cutting the number of instantiated tile configs over dropping it.
- constexpr blowups: "constexpr", "consteval", large constexpr loops/tables
- Forced unroll: "#pragma unroll" on loops with non-trivial bounds or large constants
- Excessive inlining: "__forceinline__", large device functions in headers
- Huge instantiation sets: many kernel variants, many specializations, macro-generated code
- Aggressive flags: -O3 with heavy templates; consider reducing optimization for debug builds
When phase=compile, prioritize minimal fixes that reduce compile work while preserving semantics.

Your goal: Identify the ROOT CAUSE of the timeout and provide a fix.

Output format (JSON):
```json
{
  "phase": "compile|runtime|unknown",
  "critical_issue": "<max 20 words; single headline>",
  "evidence": ["<exact substring from ERROR_LOG>", "<optional second substring>"],
  "root_cause": "<max 60 words; MUST explain why this specific construct is invalid>",
  "trigger_condition": "<max 40 words; concrete combination of args / arch / dtype OR code patterns triggering timeout>"
  "minimal_fix": "<max 25 words; executable change (from->to) or flag change>",
  "patch_anchor": "<max 20 words; unique code substring to locate edit>",
  "confidence": "high|medium|low"
}
```

Rules:
- Determine if this is a COMPILE-TIME or RUNTIME issue based on code patterns
- Be specific about code locations (line numbers, function names, etc.)
- Prioritize fixes that address the most likely root cause

Hard constraints (MUST follow):
- If ERROR_LOG provides no concrete compiler messages, do NOT invent log evidence.
- evidence MUST be copied verbatim from CUDA_CODE (grep-able substrings).
- minimal_fix MUST be a concrete edit ("Change X from A to B") or a compile flag change; no generic advice.
- patch_anchor MUST appear in CUDA_CODE and indicate where to apply the change.
- If you cannot confidently determine phase, set phase=unknown and confidence=low.
"""
)

# -----------------------------
# instruction as Template
# -----------------------------
instruction_tmpl = Template(
    """You are given:

TIMEOUT SIGNAL (may contain little/no information):
$ERROR_LOG

CUDA KERNEL CODE:

$CUDA_CODE

CONTEXT:
- The kernel compilation or execution exceeded 20 minute timeout
- The timeout could be during compilation, pre-execution loading, or actual execution
- Determine whether this is a compile-time or runtime issue based on the code

Analyze the code and identify the ROOT CAUSE of the timeout (compilation or execution).

Follow the Rules and produce the JSON exactly in the specified format."""
)

# -----------------------------
# Build both at once (returns tuple)
# -----------------------------
def build_compilation_timeout_prompts(
    *, 
    error_log: str, 
    cuda_code: str
):
    """
    Return (system_prompt_str, instruction_str) for timeout analysis.
    
    This handles both compilation timeouts and execution timeouts.
    The LLM will analyze the code and determine the root cause.
    
    Args:
        error_log: The timeout error message (may indicate compilation or execution timeout)
        cuda_code: The CUDA kernel code that times out
    """
    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        ERROR_LOG=error_log.strip(),
        CUDA_CODE=cuda_code.strip()
    )
    return system_prompt, instruction
