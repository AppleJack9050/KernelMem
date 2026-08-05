from __future__ import annotations
"""Prompt builder for Mind‑Evolution CUDA‑kernel search (seed‑kernel version).

Generates a **single prompt** that contains:
1. Target GPU spec (from `prompts/hardware/gpu_specs.py`)
2. **Few‑shot pair** – original *and* optimised model code blocks
3. Source architecture (`class Model`) that needs to be optimised
4. Existing kernel summaries (optional, for diversity context)
5. A **diversity requirement** section ensuring the new kernel differs from all previous ones
6. Output requirements

CLI usage
---------
```bash
python -m prompts.build_prompt KernelBench/level1/19_ReLU.py \
       --gpu "Quadro RTX 6000" -o prompt.txt
```
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from string import Template
from textwrap import dedent
from typing import List

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"  # GPU spec table

# --------------------------------------------------
# Few‑shot pairs  (before / after)
# --------------------------------------------------
FEWSHOT_PAIRS = [
    {
        "base": ROOT / "prompts/few_shot/model_ex_add.py",   # original Model
        "new": ROOT / "prompts/few_shot/model_new_ex_add.py"  # optimised ModelNew
    },
]

# For backward compatibility
FEWSHOT_BASE = FEWSHOT_PAIRS[0]["base"]
FEWSHOT_NEW = FEWSHOT_PAIRS[0]["new"]

# ---------------------------------------------------------------------------
# Prompt template (with diversity requirement)
# ---------------------------------------------------------------------------
test = Template(
    dedent(
        """ 
You write custom CUDA kernels to accelerate the given architecture by replacing ANY subset of PyTorch ops with custom CUDA kernels (partial replacement is allowed), and you MAY also replace the entire forward computation if beneficial (full forward-level replacement).
You may decide to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or apply algorithmic changes.
You MAY restructure the computation graph conservatively: fuse or reorder steps ONLY when the benefit is clear and semantics are straightforward.
For the parts you replace, you are NOT required to preserve intermediate tensors or per-op boundaries; ONLY the final ModelNew.forward output must match.

Primary objective: maximize speedup under the correctness contract.
This ModelNew will be evaluated in a standalone benchmark harness (kernelbench-style): end-to-end forward correctness + latency. Intermediate boundaries may change ONLY for replaced parts, but reference semantics must hold.

SEED BENCHMARK ASSUMPTIONS
- Assume inputs may be reused across runs and must remain unmodified.
- Do NOT rely on single-use, hidden aliasing, or undefined lifetime assumptions.
- Do NOT introduce hidden global state or randomness; ModelNew.forward must be deterministic given inputs and parameters.

$reference_profile
SEED GRANULARITY SELECTION (STRICT)
- For this SEED generation, you MUST choose EXACTLY ONE granularity level:
  (A) optimize a single hotspot op,
  (B) replace several ops + schedule multiple kernels,
  (C) fuse many ops into one/few kernels,
  (D) fully rewrite forward.
- All modifications MUST be consistent with the chosen granularity.
- Do NOT mix multiple granularity strategies in this seed implementation.
- In the output code, include a short header comment in ModelNew describing:
  1) chosen granularity (A/B/C/D),
  2) which ops are replaced,
  3) which ops are fused into which kernel(s) / library calls,
  4) what remains in PyTorch and why (one line each).
  All planning notes MUST be inside Python code comments; do not output any extra prose.
  If this header comment is missing or incomplete, the output is invalid.
$granularity_mandate
CRITICAL CONSTRAINT
- You MUST preserve parameter parity with the reference PyTorch module(s).
- You MAY keep an equivalent PyTorch module ONLY as a PARAMETER HOLDER (for initialization / state_dict parity).
- You MUST NOT call the parameter-holder module’s forward for replaced operators.
- Replaced computations MUST be performed by your custom CUDA kernel(s) and/or vendor library calls invoked from within your load_inline extension.
- Do NOT create independent parameters with torch.randn / random init for replaced operators.
  Instead, source weights/bias from the parameter-holder module (e.g., self.op.weight / self.op.bias).

ALLOWED ACCELERATION TOOLBOX (IMPORTANT)
You MUST build any native code via torch.utils.cpp_extension.load_inline, but inside that extension you MAY:
1) Call NVIDIA vendor libraries directly:
  - cuBLAS / cuBLASLt (including cublasLtMatmul and epilogues like bias/activation/scale where supported)
  - cuDNN (including fused Conv/ConvTranspose variants where supported)
2) Use CUDA kernels you write yourself (custom kernels, fused post-ops, reductions, etc.).
3) Specialize for fixed shapes when the harness uses fixed shapes: hardcode sizes/strides, use compile-time constants, unroll loops, choose tuned tile sizes.

SCHEDULING / FUSION / FISSION (OPTIONAL, WHEN BENEFICIAL)
- You MAY introduce multiple custom CUDA kernels and schedule them inside ModelNew.forward.
- If fusion/reorder provides little benefit, it is acceptable to optimize only one or a few hotspot operators.
- You MAY fuse multiple ops into one kernel OR split one op into multiple kernels (fission), if it improves performance.
- Intermediate buffers may be allocated as needed, but final output must match the reference.

CUSTOM CUDA/C++ CODE REQUIREMENT
- If you do not otherwise replace any ops with a custom CUDA kernel, include a tiny semantically-neutral custom kernel via load_inline (e.g., identity/copy) and call it outside critical paths to satisfy the “custom CUDA/C++ code” requirement.
- If you already replaced/fused meaningful ops, no extra dummy kernel is needed.

PRECISION POLICY (STRICT)
- Compute in the dtype of the input tensors. NEVER downcast to a narrower dtype to gain speed.
- If the inputs are float32, ALL storage and arithmetic must stay float32. Do NOT convert
  activations, weights or intermediates to __half / half2 / __nv_bfloat16 / fp8, and do NOT
  call reduced-precision tensor-core paths (e.g. cuDNN/cuBLAS half engines).
- If the inputs arrive as float16 or bfloat16, use that dtype natively — do not upcast the
  whole pipeline to float32 either. Match what the workload gives you.
- TF32 tensor cores ARE permitted for conv/GEMM: the reference itself runs them
  (torch.backends.cudnn.allow_tf32 is True by default), so this matches baseline behaviour
  and is not a downcast.
- Reductions (e.g. GroupNorm mean/variance, dot products) must accumulate in float32 or wider
  regardless of the storage dtype.
- Rationale: a narrower dtype can pass a tolerance check while computing something the
  reference did not. Speed must come from better scheduling, fusion and memory traffic,
  not from doing less precise arithmetic.

INPUTS / BENCH HARNESS INVARIANTS (STRICT)
- You MUST NOT change or redefine get_inputs() or get_init_inputs() from the reference file.
- Assume tensors returned by get_inputs() are already on the correct device.
- In ModelNew.forward, you MUST NOT move tensors across devices:
  Do NOT call .to("cuda"), .cuda(), .cpu(), or .to(x.device).
- Allocate outputs on the SAME device as the input tensors.

CORRECTNESS CONTRACT (NON-NEGOTIABLE)
1) Output semantics MUST match the reference Model exactly for the same inputs and parameters.
2) Output shape MUST match PyTorch reference.
3) Full output coverage is REQUIRED (proper blockIdx tiling; no partial writes).
4) Dtype/layout safety:
   - You MUST either support the input dtype(s) you use, or assert and convert explicitly.

CONTIGUITY (STRICT, PERFORMANCE-SAFE)
- Do NOT call contiguous() blindly. contiguous() may allocate and copy.
- Call contiguous() ONLY when required:
  - In C++: auto x_c = x.is_contiguous() ? x : x.contiguous();
  - In Python: x = x if x.is_contiguous() else x.contiguous()
- Do not create extra contiguous copies for tensors that are already contiguous.
- Do not call view() on non-contiguous tensors; either ensure contiguous first (only if needed) or use reshape() carefully.

IN-PLACE MUTATION (SEED STAGE: DISALLOWED BY DEFAULT)
- Default: implement out-of-place outputs.
- In-place is allowed ONLY if the reference forward explicitly performs in-place.

CUDA EXTENSION REQUIREMENTS
- Use torch.utils.cpp_extension.load_inline to build extensions.
- Launch kernels on at::cuda::getDefaultCUDAStream and call C10_CUDA_KERNEL_LAUNCH_CHECK.
- If using cuBLASLt/cuDNN inside the extension, ensure proper handle/stream association and deterministic behavior.

COMPILATION OPTIONS (IMPORTANT)
Always include:
- "-O3"
- "-std=c++20" (required: nvcc 13.x miscompiles the ATen headers in C++17 mode)
- "--expt-relaxed-constexpr"
- "-lineinfo"
- "$gencode_flag" (this is the target architecture of the GPU you are compiling for; do not substitute another one)
$cutlass_block
Here are examples to show you the syntax of inline embedding custom CUDA operators in torch:
$few_shot_examples

You are given the following architecture:
$arch_src

And the kernel you need to implement is:
```python
$kernel_src
```

Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew.
Output the new model code in ONE python codeblock. Real code only (no pseudocode), must compile, no tests, no extra text.
"""
    )
)

default_system_prompt = """\
You are a senior CUDA-kernel optimisation specialist. Your job is to generate a high-quality,
compilable, and runnable Python script that builds and launches **hand-written CUDA kernels**.

OUTPUT RULES (STRICT):
output the code within:
```python
# <complete ModelNew code>
```

"""
# ---------------------------------------------------------------------------
# GPU spec loader
# ---------------------------------------------------------------------------


# CUDA 13 dropped the unversioned PFN_cuTensorMap* aliases that CUTLASS <= 3.5.x
# still references, so cutlass/cutlass.h fails to compile there without this
# prelude. _cutlass_probe() decides whether it is actually needed.
_CUTLASS_PFN_SHIM = "\n".join([
    "#include <cudaTypedefs.h>",
    "using PFN_cuTensorMapEncodeTiled  = PFN_cuTensorMapEncodeTiled_v12000;",
    "using PFN_cuTensorMapEncodeIm2col = PFN_cuTensorMapEncodeIm2col_v12000;",
])


def _cutlass_include_dir() -> str | None:
    """Path to CUTLASS/CuTe C++ headers, or None when they are not installed.

    Located rather than hardcoded so the block below disappears on a machine
    without CUTLASS instead of telling the model to add an -I flag that makes
    every kernel fail to compile. Override with CUTLASS_INCLUDE_DIR.
    """
    import importlib.util
    cands: List[Path] = []
    env = os.environ.get("CUTLASS_INCLUDE_DIR")
    if env:
        cands.append(Path(env))
    # headers-only copy vendored in this repo (see third_party/cutlass)
    cands.append(Path(__file__).resolve().parent.parent
                 / "third_party" / "cutlass" / "include")
    try:                                    # pip install nvidia-cutlass
        spec = importlib.util.find_spec("cutlass_library")
        if spec and spec.origin:
            cands.append(Path(spec.origin).parent / "source" / "include")
    except Exception:
        pass
    try:                                    # flashinfer vendors a copy
        spec = importlib.util.find_spec("flashinfer")
        if spec and spec.origin:
            cands.append(Path(spec.origin).parent / "data" / "cutlass" / "include")
    except Exception:
        pass
    cands.append(Path("/usr/local/cuda/include"))
    for c in cands:
        try:
            if (c / "cutlass" / "cutlass.h").is_file() and (c / "cute" / "tensor.hpp").is_file():
                return str(c)
        except OSError:
            continue
    return None


def _cutlass_probe(inc: str, gencode: str) -> "tuple[bool, str]":
    """nvcc-compile a small TU against `inc`; cache the verdict on disk.

    Returns (ok, shim).  `shim` is prelude source the generated kernel MUST emit
    before any cutlass include for the build to succeed (empty when the headers
    compile as-is).

    A file-existence check is not enough: it advertised CUTLASS as "verified to
    compile" on toolchains where it does not.  CUDA 13 against CUTLASS <= 3.5.x
    is exactly that case -- the headers are present and every build still fails
    on PFN_cuTensorMapEncodeTiled.  An agent told the headers were verified
    burns its attempts on that error and the compile-timeout judge then blames
    CUTLASS itself, so the branch gets abandoned for the wrong reason.
    """
    import hashlib
    import json
    import shutil
    import subprocess
    import tempfile

    nvcc = shutil.which("nvcc")
    if not nvcc:
        return (False, "")
    try:
        ver = subprocess.run([nvcc, "--version"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return (False, "")

    key = hashlib.sha1(f"{inc}|{gencode}|{ver}".encode()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "kernelmem" / "cutlass_probe.json"
    db: dict = {}
    try:
        db = json.loads(cache.read_text())
        if key in db:
            hit = db[key]
            return (bool(hit.get("ok")), str(hit.get("shim", "")))
    except Exception:
        db = {}

    # NB: plain replace, not str.format -- the C++ body contains braces.
    src = ("#include <cuda.h>\n"
           "#include <cuda_runtime.h>\n"
           "__SHIM__\n"
           "#include <cutlass/cutlass.h>\n"
           "#include <cute/tensor.hpp>\n"
           "#include <cutlass/gemm/device/gemm.h>\n"
           "int main() { return 0; }\n")

    ok, shim = False, ""
    with tempfile.TemporaryDirectory() as td:
        for cand in ("", _CUTLASS_PFN_SHIM):
            cu = Path(td) / "probe.cu"
            cu.write_text(src.replace("__SHIM__", cand))
            cmd = [nvcc, "-std=c++20", "-O0", *gencode.split(), f"-I{inc}",
                   "--expt-relaxed-constexpr", "-DNDEBUG",
                   "-c", str(cu), "-o", str(Path(td) / "probe.o")]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except Exception:
                break
            if res.returncode == 0:
                ok, shim = True, cand
                break

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        db[key] = {"ok": ok, "shim": shim}
        cache.write_text(json.dumps(db, indent=1))
    except Exception:
        pass
    return (ok, shim)


def resolve_gpu_name(gpu_name: str, gpu_info: dict) -> str:
    """Map a torch device name onto a GPU_SPEC_INFO key.

    torch.cuda.get_device_name() returns "NVIDIA GeForce RTX 5090" while the
    spec table is keyed "RTX 5090", so auto-detect could never hit and callers
    had to pass --gpu by hand. main_memory_latest.py still DEFAULTS to
    "A100-80GB", which silently feeds A100 bandwidth and tensor-core rates to a
    model reasoning about a different card -- worse than crashing, because the
    roofline arithmetic in the prompt is then quietly wrong.
    """
    if gpu_name in gpu_info:
        return gpu_name
    cleaned = (gpu_name.replace("NVIDIA", "")
                       .replace("GeForce", "")
                       .replace("Tesla", "")
                       .strip())
    if cleaned in gpu_info:
        _log_capability(f"gpu-spec:{gpu_name}", True, f"matched '{cleaned}'")
        return cleaned
    up = cleaned.upper()
    hits = [k for k in gpu_info if k.upper() == up] \
        or [k for k in gpu_info if up.startswith(k.upper()) or k.upper() in up]
    if hits:
        best = max(hits, key=len)
        _log_capability(f"gpu-spec:{gpu_name}", True, f"matched '{best}'")
        return best
    raise KeyError(
        f"{gpu_name} not present in GPU_SPEC_INFO (known: {sorted(gpu_info)}). "
        "Pass --gpu with one of those names, or add the card to "
        "prompts/hardware/gpu_specs.py.")


_CAPABILITY_LOGGED: set = set()


def _log_capability(name: str, available: bool, detail: str = "") -> None:
    """Announce a capability verdict exactly once per process.

    Capabilities used to fail SILENTLY: when CUTLASS was absent the guidance
    block simply vanished from the prompt and nothing anywhere said so, so a run
    could spend its whole budget unable to reach an optimisation without a
    single line of evidence why. Any capability gated on the environment must
    say what it decided.
    """
    if name in _CAPABILITY_LOGGED:
        return
    _CAPABILITY_LOGGED.add(name)
    mark = "available" if available else "UNAVAILABLE"
    msg = f"[capability] {name}: {mark}"
    if detail:
        msg += f" ({detail})"
    print(msg, file=sys.stderr, flush=True)


def _cutlass_block() -> str:
    """Guidance emitted only when CUTLASS headers actually COMPILE here."""
    inc = _cutlass_include_dir()
    if not inc:
        _log_capability("cutlass", False,
                        "headers not found; set CUTLASS_INCLUDE_DIR or populate "
                        "third_party/cutlass/include")
        return ""
    ok, shim = _cutlass_probe(inc, _detect_gencode_flag())
    if not ok:
        _log_capability("cutlass", False,
                        f"headers at {inc} do NOT compile with this toolchain")
        return ""
    _log_capability("cutlass", True,
                    f"{inc}{'; PFN shim required' if shim else ''}")
    lines = [
        "",
        "CUTLASS / CuTe IS AVAILABLE (compile-probed in this environment)",
        f'- Add "-I{inc}" to extra_cuda_cflags and you may #include <cutlass/...>',
        "  and <cute/tensor.hpp>. Probed with the gencode flag above and -std=c++20;",
        "  use -std=c++20, not c++17 (torch's headers do not build under c++17 here).",
    ]
    if shim:
        lines += [
            "- REQUIRED PRELUDE: this CUDA toolkit dropped the unversioned PFN_cuTensorMap*",
            "  aliases CUTLASS references. Emit these lines BEFORE any cutlass include, or",
            "  the build fails with 'identifier PFN_cuTensorMapEncodeTiled is undefined':",
        ] + [f"      {ln}" for ln in shim.splitlines()]
    lines += [
        "- PREFER IT OVER HAND-WRITTEN wgmma PTX when you own a GEMM/conv. Hopper",
        "  warpgroup MMA needs shared-memory descriptors with specific swizzle layouts and",
        "  correct wgmma.fence / commit_group / wait_group ordering; those are what",
        "  hand-written warpgroup code gets wrong, and CUTLASS already encodes them.",
        "  cute::tiled_mma / the SM90 collective builders reach the same instructions the",
        "  vendor kernels use.",
        "- IT CAN BEAT A VENDOR CONV WITHOUT BEATING ITS MATH. cuDNN's TF32 path runs a",
        "  separate fp32->TF32 convertTensor pass over the input (a full read+write of the",
        "  tensor, ~10-13% of conv time); a CUTLASS conv with ElementA=float converts",
        "  in-register in the mainloop and skips it. Owning the kernel additionally lets you",
        "  fold a following reduction (e.g. GroupNorm sum/sumsq) into the epilogue, deleting",
        "  another full read of the conv output.",
        "- ACCURACY NOTE: when the harness reference is itself cuDNN TF32, a conv whose",
        "  accumulation order differs from cuDNN's costs real error budget. CUTLASS tracks",
        "  it closely (cuDNN dispatches CUTLASS kernels); an ad-hoc TF32 conv with a",
        "  different reduction order may not. Check tolerance across several seeds, not one.",
        "- COST: a CUTLASS include adds roughly 50s to the build. Worth it when you are",
        "  replacing a vendor GEMM/conv; not worth it for an elementwise or reduction",
        "  kernel, where plain CUDA compiles in a few seconds and is just as fast.",
        "",
    ]
    return "\n".join(lines)


def _detect_gencode_flag() -> str:  # noqa: D401
    """Return the -gencode flag for the local device (falls back to sm_80)."""
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(0)
        cc = f"{major}{minor}"
        return f"-gencode=arch=compute_{cc},code=sm_{cc}"
    except Exception:  # pragma: no cover - no CUDA device visible
        return "-gencode=arch=compute_80,code=sm_80"


def _load_gpu_spec() -> dict:  # noqa: D401
    """Import `gpu_specs.py` and return the GPU_SPEC_INFO dict (robust across Python versions)."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module  # avoid re‑import
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Prompt builder core
# ---------------------------------------------------------------------------

def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
    reference_profile: str | None = None,
    force_granularity: str | None = None,
    force_algorithm: str | None = None,
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation).

    *reference_profile* is an optional measured breakdown of where the
    reference spends GPU time (see utils.reference_profile). It is injected
    directly above the granularity choice, because that choice fixes what the
    model may rewrite for the whole run and is otherwise made blind.

    *force_granularity* pins that choice instead of leaving it to the model.
    None (the default) leaves the prompt byte-identical to before. It exists
    because the choice is made once, at round 0, from a single sample, and is
    never revisited -- on vae_block_002 every seed took (A)/(B)/(C), which
    accepted a 1.35x Amdahl cap that the following 24 rounds then spent
    themselves against. Pinning it makes that a controlled variable rather
    than a coin flip.
    """
    gpu_info = _load_gpu_spec()

    # Auto‑detect GPU if not provided
    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    gpu_name = resolve_gpu_name(gpu_name, gpu_info)

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    arch_src = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    ) if gpu_arch != "Unknown" else "Not Specified"

    # The few-shot examples ship an A100 -gencode flag; retarget them at the
    # local device so the model does not copy a wrong architecture.
    gencode_flag = _detect_gencode_flag()

    # Build few-shot examples from all pairs
    few_shot_examples = []
    for i, pair in enumerate(FEWSHOT_PAIRS, 1):
        try:
            base_content = pair["base"].read_text().strip()
            new_content = pair["new"].read_text().strip().replace(
                "-gencode=arch=compute_80,code=sm_80", gencode_flag
            )
            few_shot_examples.append(
                f"Example {i}:\n"
                f"The example given architecture is:\n"
                f"'''\n{base_content}\n'''\n"
                f"The example new arch with custom CUDA kernels looks like this:\n"
                f"'''\n{new_content}\n'''\n"
            )
        except FileNotFoundError as e:
            import warnings
            warnings.warn(f"Few-shot example file not found: {e}. Skipping this example.")
    
    few_shot_examples_text = "\n".join(few_shot_examples)
    kernel_src = Path(arch_path).read_text().strip()

    # Blank (not omitted) when profiling was unavailable, so the prompt reads
    # identically to before rather than carrying an empty header.
    profile_block = f"{reference_profile.rstrip()}\n" if reference_profile else ""

    return test.substitute(
        few_shot_examples=few_shot_examples_text,
        arch_src=arch_src,
        kernel_src=kernel_src,
        gencode_flag=gencode_flag,
        cutlass_block=_cutlass_block(),
        reference_profile=profile_block,
        granularity_mandate=_granularity_mandate(force_granularity, force_algorithm),
    )


_GRANULARITY_NAMES = {
    "A": "optimize a single hotspot op",
    "B": "replace several ops + schedule multiple kernels",
    "C": "fuse many ops into one/few kernels",
    "D": "fully rewrite forward",
}


_ALGORITHM_MANDATE = {
    "implicit_gemm": [
        "- ALGORITHM: implement the convolution as a tiled implicit GEMM.",
    ],
    "winograd_f2x3": [
        "- ALGORITHM (BINDING): implement the 3x3 stride-1 convolution with the WINOGRAD",
        "  F(2x2, 3x3) minimal filtering algorithm, NOT a direct or implicit-GEMM",
        "  convolution. It needs 16 multiplies per 2x2 output tile where direct needs 36",
        "  -- 2.25x fewer. That reduction, not better scheduling, is the entire reason",
        "  this lineage exists: the vendor kernel already runs at ~97% of FLOP peak, so",
        "  it cannot be out-scheduled, only out-counted.",
        "- THE TRANSFORMS MUST BE FUSED. Keep the 4x4 transform tiles in registers or",
        "  shared memory and never write them to global memory. The transform domain is",
        "  4x the size of the tensor (16 values per 2x2 output tile), so materializing it",
        "  costs roughly 4x tensor traffic and gives back the whole 2.25x. This is",
        "  measured, not theoretical: on a 3x3 stride-1 fp32 conv block, cuDNN's",
        "  WINOGRAD_NONFUSED algorithm lost to the implicit-GEMM path it replaced,",
        "  and it lost precisely because it materializes the transforms.",
        "- Use F(2x2,3x3) ONLY. Its transform entries are 0, +/-1, +/-1/2, all exactly",
        "  representable in binary floating point, so the transforms add no representation",
        "  error. F(4x4,3x3) uses +/-1/4 and +/-1/8 and degrades accuracy materially.",
        "- Expect this to be register-pressure bound: 16 accumulators per tile is the",
        "  binding constraint. Choose the tile/warp decomposition around that.",
    ],
}


def _granularity_mandate(level: str | None, algorithm: str | None = None) -> str:
    """Binding override for the granularity menu, or "" to leave it a free choice."""
    if not level:
        return ""
    level = level.upper()
    if level not in _GRANULARITY_NAMES:
        raise ValueError(f"granularity must be one of {sorted(_GRANULARITY_NAMES)}, got {level!r}")
    lines = [
        "",
        "SEED GRANULARITY OVERRIDE (BINDING - overrides the menu above)",
        f"- The granularity for this run is FIXED to ({level}) {_GRANULARITY_NAMES[level]}.",
        f"  Do not choose any other level. Still emit the header comment, stating ({level}).",
    ]
    if level == "D":
        lines += [
            "- (D) here means you OWN the vendor GEMM/convolution named in the reference profile:",
            "  implement it yourself (your own CUDA, or CUTLASS/cuBLASLt device-side from inside",
            "  your extension). Do NOT satisfy it with cuDNN, at::conv2d, or at::cudnn_convolution.",
            "- Owning it is only worth anything if you FUSE the surrounding work into its",
            "  prologue/epilogue so intermediates never round-trip through global memory. A",
            "  standalone reimplementation that merely matches the vendor kernel wins nothing --",
            "  the whole point is the traffic you delete, not the GEMM you rewrite.",
            "- Matching the vendor kernel's raw throughput is NOT the target and is not expected.",
            "  The target is total forward time. A convolution at 80% of the vendor's throughput",
            "  that eliminates a full read AND write of the intermediate tensors is a clear win;",
            "  one at 100% that still materializes them is not.",
            "- Correctness still governs: the reference uses TF32 tensor cores with fp32 accumulate.",
            "  Reduced-precision accumulation that breaks the tolerance is a failed kernel, not a",
            "  fast one.",
            "- INSTRUCTION CHOICE MATTERS AS MUCH AS THE ALGORITHM. On sm_90 the vendor",
            "  kernels issue Hopper warpgroup MMA (wgmma, e.g. m64n128k8 = 65,536 MACs per",
            "  instruction, one warpgroup of 4 warps per tile). wmma m16n16k8 compiles to",
            "  mma.m16n8k4 = 512 MACs per instruction -- 128x less work issued per slot. A",
            "  kernel built on wmma therefore has to issue ~14x MORE tensor instructions than",
            "  the vendor even when its algorithm does 2.25x LESS arithmetic, and it becomes",
            "  issue-bound long before it becomes memory-bound. Such a kernel has been",
            "  measured idling DRAM at ~1.5% while running 11x slower than the vendor, at",
            "  HIGHER occupancy than the vendor (24.7% vs 18.2%) -- so occupancy is not the",
            "  lever. If you intend to beat a vendor GEMM/conv rather than merely fuse",
            "  around it, size the MMA tile first: raising occupancy or cutting redundant loads",
            "  cannot recover a 128x deficit in work-per-instruction.",
            "- SCOPE OF THIS SEED: a correct, sound structure beats a fast one here. ONE",
            "  straightforward tiled implicit-GEMM (shared-memory tiling; wgmma if you own a",
            "  GEMM/conv you mean to beat, otherwise mma/wmma or plain FMA)",
            "  with the surrounding work fused into its epilogue is the right size of first draft.",
            "  Do NOT write a multi-stage software-pipelined CUTLASS-class kernel in the seed:",
            "  later rounds exist to tune it, and an over-long first reply is the most common way",
            "  this seed fails outright rather than merely scoring low.",
            "- Emit ONE complete Python file and nothing else. Do not restate the reference, do not",
            "  offer alternative implementations, do not include benchmarking or test code.",
        ]
    if algorithm:
        extra = _ALGORITHM_MANDATE.get(algorithm)
        if extra is None:
            raise ValueError(f"unknown seed algorithm {algorithm!r}; "
                             f"known: {sorted(_ALGORITHM_MANDATE)}")
        lines += extra
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA‑kernel optimisation (seed generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_py", help="Path to .py containing class Model")
    parser.add_argument("--gpu", default=None, help="GPU name key in gpu_specs.py")
    parser.add_argument("-o", "--out", help="Save prompt to file")
    args = parser.parse_args()

    prompt = build_seed_prompt(Path(args.model_py), args.gpu)

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"[✓] Prompt saved to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
