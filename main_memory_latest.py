# main.py
from __future__ import annotations
import argparse
import re
import random
import time
import json
import csv
import os
import signal
import importlib.util
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from run_ncu_memory import profile_bench, load_ncu_metrics, metrics_to_prompt
from run_nsys import profile_bench as nsys_profile_bench, load_nsys_stats
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda_memory import build_seed_prompt, default_system_prompt
from utils.reference_profile import build_reference_profile_block
from prompts.judger_compilation_timeout import build_compilation_timeout_prompts
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json, extract_cuda_kernel_names
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error_memory import build_error_prompt
from prompts.optimization_memory_latest import build_optimization_prompt
from prompts.judger_repair_memory import build_correctness_prompts
from prompts.judger_optimization_memory_latest import build_judger_optimization_prompts
from utils.gpu_lock import gpu_section
from utils.torch_ext_cache import sweep_stale_batons, sweep_unheld_batons
from utils import run_timing
from utils.mcgs import (MonteCarloGraphSearch, load_code_features, reward_from_gain,
                        state_key)

# ---------------------------------------------------------------------------
# Serialize every GPU-touching entry point behind one cross-process mutex.
#
# Wrapped here, once, rather than at each call site: `compare_and_bench` alone is
# reached from the seed loop, the repair path and the optimization path, and a
# single missed site would silently let two lineages measure at the same time --
# which does not add noise, it invalidates the comparison the ratchet is built
# on. Wrapping the imported name covers every caller by construction.
#
# gpu_section() is a no-op unless KERNELMEM_GPU_LOCK is set, so single-process
# runs are byte-for-byte unaffected; only the lineage coordinator sets it.
# ---------------------------------------------------------------------------
def _serialize_on_gpu(fn, what):
    def _wrapped(*args, **kwargs):
        with gpu_section(what):
            return fn(*args, **kwargs)
    _wrapped.__name__ = getattr(fn, "__name__", what)
    _wrapped.__doc__ = getattr(fn, "__doc__", None)
    return _wrapped


compare_and_bench = _serialize_on_gpu(compare_and_bench, "bench")
profile_bench = _serialize_on_gpu(profile_bench, "ncu")
nsys_profile_bench = _serialize_on_gpu(nsys_profile_bench, "nsys")
build_reference_profile_block = _serialize_on_gpu(build_reference_profile_block, "ref_profile")

_INVOCATION_SPLITTER = "Invoked with:"

def _sanitize_error_message(exc: Exception) -> str:
    """Strip pybind's large‑tensor printouts and keep only the key error text."""
    msg = str(exc)
    if _INVOCATION_SPLITTER in msg:
        msg = msg.split(_INVOCATION_SPLITTER, 1)[0].rstrip()
    return msg

# ------------------------- CLI -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-LLM self-iterative kernel generation/optimization")
    p.add_argument(
        "arch_py",
        type=Path,
        help="Path to a single task .py file OR a directory containing many tasks (.py)",
    )
    # p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    # Default None => auto-detect from torch and normalise via resolve_gpu_name().
    # It used to default to "A100-80GB", which meant the auto-detect branch was
    # dead code and a run on any other card silently fed the model A100
    # bandwidth / tensor-core numbers -- corrupting exactly the roofline
    # reasoning the prompt asks it to do.
    p.add_argument("--gpu", default=None, help="GPU name in prompt spec (default: auto-detect)")
    p.add_argument("--server_type", default="claude", help="Label only; all calls go through the Claude Agent SDK (subscription credit)")
    p.add_argument("--server_address", default="localhost", help="Unused (kept for compatibility)")
    p.add_argument("--server_port", type=int, default=8000, help="Unused (kept for compatibility)")
    p.add_argument("--model_name", default="claude-opus-5", help="Claude model (non-Claude names fall back to claude-opus-5)")
    p.add_argument("--rollout_model", default="claude-sonnet-5",
                   help="Model for the MCGS ROLLOUT -- the `optimization` call that writes the "
                        "next kernel from the selected state. This is the call the search repeats "
                        "every round, and generation is ~95%% of the wall clock, so it is the one "
                        "worth moving off the most expensive model. The judge, problem-identify "
                        "and repair calls stay on --model_name. Both go through the Claude Agent "
                        "SDK with the API-key env blanked, so both bill subscription credit, not "
                        "the API. Set this equal to --model_name to disable the split.")
    p.add_argument("--rollout_effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Reasoning effort for the rollout call. High on purpose: the point of the "
                        "split is a cheaper MODEL, not a cheaper think -- writing a Hopper kernel "
                        "that compiles is the part the loop cannot afford to get wrong, and the "
                        "measured failure mode is candidates that do not build, not candidates "
                        "that were under-tuned. Note KERNELMEM_CLAUDE_EFFORT no longer overrides "
                        "an explicit per-call effort; it now only sets the default for calls that "
                        "do not request one.")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations per task")
    p.add_argument("--num_seeds", type=int, default=3,
                   help="Round-0 seed candidates to draw; the best-scoring one becomes the base "
                        "(1 = previous single-sample behaviour)")
    p.add_argument("--seed_granularity", default=None, choices=["A", "B", "C", "D"],
                   help="Pin the round-0 granularity instead of letting the seed choose it. "
                        "The choice fixes what every later round may rewrite and is never "
                        "revisited, yet it is made from a single sample. On vae_block_002 all "
                        "seeds took (A)/(B)/(C), keeping the vendor conv -- 74.1%% of reference "
                        "GPU time -- which caps the whole run at 1.35x by Amdahl; 24 rounds then "
                        "reached 1.20x and plateaued. 'D' forces the model to own the vendor "
                        "GEMM/conv so it can fuse the surrounding work into its epilogue. Default "
                        "(unset) leaves the prompt byte-identical to before.")
    p.add_argument("--seed_algorithm", default=None,
                   choices=["implicit_gemm", "winograd_f2x3"],
                   help="Pin the ALGORITHM the seed uses for the operator it owns "
                        "(requires --seed_granularity D to be meaningful). Only an "
                        "algorithm with a lower operation count can beat a vendor "
                        "kernel already at ~97%% of FLOP peak: winograd_f2x3 needs "
                        "16 multiplies per 2x2 tile where direct needs 36 (2.25x). "
                        "Default (unset) leaves the choice to the model.")
    p.add_argument("--work_dir", type=Path, default=Path("run"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    p.add_argument("--warmup", type=int, default=25, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=100, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-2, help="Max |err| tolerated")
    p.add_argument("--base_margin", type=float, default=0.005,
                   help="Relative margin a kernel must beat the current base by to become the new "
                        "base for later optimization rounds (0.005 = 0.5%%). Too low and the base "
                        "churns on measurement jitter; too high and optimization keeps restarting "
                        "from the seed instead of compounding. Default measured on RTX 5090 / "
                        "vae_block_002: within-session noise is 0.15-0.80%% stdev (kernel-dependent -- "
                        "multi-stream kernels are ~5x noisier), but scores are compared ACROSS "
                        "rounds, where GPU-state drift of +0.9..+1.7%% attenuates real gains. A "
                        "verified +1.26%% same-session improvement showed up as only +0.57%% "
                        "cross-round, so the margin must sit below the attenuated value. Re-derive "
                        "this for a different GPU or task. Use 0 to accept any improvement, or a "
                        "large value to pin the base to the seed (the pre-fix behaviour).")
    p.add_argument("--base_reps", type=int, default=5,
                   help="Interleaved repeats used to compare a candidate against the base. The "
                        "base is re-measured alongside the candidate instead of being read from "
                        "a score stored when it last advanced, because that stored number goes "
                        "stale: one unchanged kernel re-measured 30 min later on RTX 5090 read "
                        "+1.06%%, twice the --base_margin it feeds. Was 3; raised to 5 because "
                        "the paired test has dof = reps-1 and dof=2 cannot support --base_sigma "
                        "at all -- a true 3-sigma tail on 2 dof needs |t| >= 19.2, against 6.6 on "
                        "4 dof and 4.5 on 7. The extra reps are close to free: the paired path no "
                        "longer times the reference (time_ref=False), which was over half of each "
                        "rep and was discarded unread.")
    p.add_argument("--base_max_reps", type=int, default=8,
                   help="Cap on interleaved repeats when the decision is close. Reps are added "
                        "only while the measured difference cannot be separated from "
                        "--base_margin at --base_sigma (evaluated on the t distribution at the "
                        "reps actually taken), so clear wins and clear losses stop at "
                        "--base_reps and only genuine coin-flips pay for precision. On rounds "
                        "13-20 of the exp3 run six of seven candidates (-0.66%% to -11.06%%) "
                        "would stop early and one (+0.49%%) would escalate. Set 0 to disable the "
                        "paired re-measure and compare against the stored base score, as before.")
    p.add_argument("--base_sigma", type=float, default=3.0,
                   help="How significant a paired gain must be, in NORMAL-equivalent sigmas, "
                        "before it may advance the base. --base_margin asks 'is it big enough'; "
                        "this asks 'are we sure it is a gain at all', and a point estimate alone "
                        "cannot answer the second. Calibrated by re-measuring the exp3 chain "
                        "paired: the two real advances came back at 6.6 and 8.6 sigma while the "
                        "three that had been adopted on drift came back at -2.1, -1.7 and +0.9, "
                        "so anything in 1..6 separates them and 3 sits in the middle. It matters "
                        "most on noisy kernels -- multi-stream ones run ~5x noisier -- where a "
                        "+0.6%% reading with 0.4%% standard error clears a 0.5%% margin at only "
                        "1.5 sigma. NOTE: this is now enforced through the t distribution at "
                        "dof = --base_reps-1, not by comparing rel/se to the number directly. "
                        "Those differ enormously at these rep counts -- 3.0 sigma means |t| >= "
                        "19.2 at dof 2, 6.6 at dof 4, 4.5 at dof 7 -- and the old z-score reading "
                        "let 2 of 7 historical advances through at t~4.8 and t~13.8 on dof 2. "
                        "Set 0 to decide on the margin alone.")
    p.add_argument("--structural_grace", type=int, default=0,
                   help="Rounds a DECLARED structural rewrite may hold the base while it is "
                        "still slower than the kernel it displaced (0 = off, the ratchet-only "
                        "behaviour). The ratchet adopts a candidate only if it beats the base "
                        "immediately, which is right for tuning and fatal for restructuring: "
                        "changing MMA primitive, tile geometry or memory pipeline is slower "
                        "until it is finished, so the first version is always rejected and the "
                        "rewrite can never be reached in steps. Measured on vae_block_002: the "
                        "loop produced 10 straight wmma/FMA kernels and never once reached "
                        "wgmma, whose work-per-instruction is 128x higher. The judge declares "
                        "the intent (\"structural_rewrite\": true) and pays for it: if the "
                        "rewrite has not beaten the displaced kernel within this many rounds, "
                        "the old base is restored. best_kernel is never affected, so a run can "
                        "only report a kernel that genuinely measured best. Try 3.")
    # ---- search policy: what the next round branches from -------------------
    p.add_argument("--search", default="mcgs", choices=["ratchet", "mcgs"],
                   help="How the parent for each round is chosen. 'ratchet' is the original "
                        "hill-climber: keep one incumbent, branch from it forever, discard every "
                        "rejected candidate. Measured over the 18 saved trees in run/, that left "
                        "113 of 169 nodes (66.9%%) visited exactly once and never revisited. "
                        "'mcgs' (default on this branch) runs Monte Carlo Graph Search over kernel "
                        "STATES instead: transpositions merge, so two edit orders that reach the "
                        "same structure pool their statistics rather than splitting a budget that "
                        "only affords ~30 evaluations per run. Keep 'ratchet' available for A/B -- "
                        "cross-method claims on this codebase have been overturned before.")
    p.add_argument("--mcgs_state_key", default="mechanisms",
                   choices=["mechanisms", "features", "code"],
                   help="What makes two kernels the SAME state, i.e. what may merge. Measured by "
                        "replaying the 145 scored kernels in run/ that still have sources: "
                        "'mechanisms' (order-independent MULTISET of method_names along the path) "
                        "gives 85 states, 1.71 kernels/state -- the default. 'features' (the "
                        "code_features_used vector) gives 11 states for 145 kernels, one holding 47 "
                        "kernels across a 457%% speedup range against a 516%% total range: it pools "
                        "nearly everything, because is_aligned_vector_access and is_pointwise never "
                        "vary and three more are ~constant, so the vector carries about two bits. "
                        "'code' gives 104 states, 1.39/state, and is effectively a tree -- keep it "
                        "as the A/B control for whether merging is what helped. NOTE: genuine "
                        "commuting transpositions are ~absent from the recorded history; the "
                        "merging you actually get is re-derivation of an identical recipe.")
    p.add_argument("--mcgs_merge_tol", type=float, default=0.15,
                   help="Refuse to pool a kernel into a state whose representative differs from it "
                        "by more than this relative amount; it is split into its own state instead. "
                        "Exists because of the 'features' measurement above: without it a coarse "
                        "key makes Q an average over kernels 5x apart in speed. With it, a bad "
                        "abstraction degrades toward a tree rather than corrupting the values. "
                        "Set 0 to trust the key completely.")
    p.add_argument("--mcgs_c_puct", type=float, default=0.8,
                   help="Exploration weight in Q + c*sqrt(ln N_parent / N_child). Rewards are "
                        "mapped into [0,1] so this is comparable across tasks. Low because the "
                        "budget is ~30 evaluations: a large c spends all of it on first visits.")
    p.add_argument("--mcgs_lam", type=float, default=0.7,
                   help="Weight on the MAX term in Q = (1-lam)*mean + lam*max. High on purpose. "
                        "The measured gain distribution is bimodal -- 42%% of edges regress past "
                        "-1%%, 33%% win past +1%%, only 25%% land inside the +-1%% noise band -- and "
                        "the loop keeps the best kernel, not the average one. Mean-backup buries a "
                        "state that produced one +6%% child among four regressions, which is the "
                        "shape of every real win in the data.")
    p.add_argument("--mcgs_widen_k", type=float, default=1.0,
                   help="Progressive widening: a state may have ceil(k * N**alpha) children. The "
                        "action space is LLM-generated and unbounded, so there is no move list to "
                        "argmax over; a state earns another child only by being visited.")
    p.add_argument("--mcgs_widen_alpha", type=float, default=0.5)
    p.add_argument("--mcgs_max_depth", type=int, default=10,
                   help="Edits from the seed after which selection expands sideways instead of "
                        "deeper. Measured, not chosen: win rate over the saved runs is 41%% for "
                        "rounds 0-4 and 5-9, then 0%% for rounds 10-14, 15-19 and 20-24 -- zero "
                        "wins in 22 edges past round 10. Calibrated on vae_block_002; re-derive it "
                        "before trusting it on a task whose kernels have more structural room.")
    p.add_argument("--mcgs_reward_scale", type=float, default=3.0,
                   help="Percent gain that maps to a near-saturated reward via tanh(rel/scale). "
                        "At 3.0: 0%% -> 0.50, +1%% -> 0.58, +3%% -> 0.88. The reward is the PAIRED "
                        "relative gain, never the blocked score -- score carries +0.9..+1.7%% "
                        "cross-round drift and a corruptible T_ref denominator, and backing a max "
                        "over that up a graph compounds the bias at every level.")
    p.add_argument("--patience", type=int, default=4,
                   help="Stop after this many consecutive rounds that fail to improve best_score "
                        "by more than --base_margin (0 disables). Late rounds are where a run "
                        "spends its time without earning anything: on vae_block_002 the last 10 "
                        "of 21 rounds took 46%% of the wall clock and moved the score by +0.017%%. "
                        "Measured on that run, every value from 3 up stops after the same real "
                        "improvement and costs the same 0.017%%, so the default buys a round of "
                        "headroom over the boundary. Do NOT lower this to 2: that run had a "
                        "2-round drought before its four best rounds, and stopping there would "
                        "have cost 6.5%%.")
    p.add_argument("--temperature", type=float, default=1, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    # multi-task controls
    p.add_argument("--first_n", type=int, default=0,
                   help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="1-based index in the sorted task list to start from (only applies when using --first_n)",
    )
    p.add_argument("--num_tasks", type=int, default=1,
                   help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")
    p.add_argument("--filter_from_summary", type=Path, default=None,
                   help="Path to summary.json file. If provided, only tasks with best_runnable=false will be selected from this summary.")
    
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to an existing batch folder (run/<stamp>_<task>_<tag>). Reuses that "
                        "folder and continues each task from its checkpoint.json instead of "
                        "starting a new run. Combine with a larger --round to extend a finished run.")
    p.add_argument("--subproc_id", type=int, default=0, help="Identifier for sub-process (e.g., when running multiple in parallel)")
    
    return p


# ---------------------- naming helpers -----------------
def _slugify_tag(text: str, max_len: int = 80) -> str:
    """Collapse a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if max_len > 0:
        slug = slug[:max_len]
    return slug or "unknown"


def _build_run_tag(server_type: str, model_name: str) -> str:
    server_tag = _slugify_tag(server_type)
    model_tag = _slugify_tag(model_name)
    return f"{server_tag}_{model_tag}"


# ---------------------- small utils --------------------
def _last_n_lines(text: str, n: int = 150) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_full_cuda_source(text: str) -> str:
    """Extract CUDA source from a Python or markdown-like file.

    Order:
      1) ```cuda ... ``` fenced code
      2) source = \"\"\" ... \"\"\"
      3) fallback: raw text
    """
    m = re.search(r"```cuda\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"source\s*=\s*([\"']{3})(.*?)(?:\1)", text, flags=re.DOTALL)
    if m:
        return m.group(2).strip()
    return text.strip()


def _build_history_block(code_dir: Path, keep_last: int = 10) -> str:
    """Collect the CUDA `source` of the most recent *keep_last* kernel files from code_dir."""
    if not code_dir.exists():
        return "## Existing kernels\n(None yet)\n"

    files: List[Path] = sorted(
        list(code_dir.glob("*.py")) + list(code_dir.glob("*.cu")),
        key=lambda p: p.stat().st_mtime,
    )[-keep_last:]

    if not files:
        return "## Existing kernels\n(None yet)\n"

    snippets: List[str] = []
    for idx, p in enumerate(files, 1):
        try:
            cuda_src = _extract_full_cuda_source(_read_text(p))
        except Exception:
            cuda_src = "(failed to read/extract)"
        snippets.append(f"### Kernel {idx} · {p.name}\n```cuda\n{cuda_src}\n```")

    return "## Existing kernels\n" + "\n\n".join(snippets) + "\n"


def _ncu_profile_cached(
    *,
    bench_py: str,
    kernel_names,
    kernel_file,
    out_csv,
    device_idx,
    repeat: int,
    timeout_override,
    cache_dir: Path,
) -> Path:
    """profile_bench, skipping the ncu run when this exact source was profiled before.

    A rejected candidate leaves the base kernel unchanged, so the next round
    re-profiles source that was already measured -- 56s and 263s on the two
    occurrences in the 20260731 run. ncu over fixed source, fixed input shapes
    and a fixed device is deterministic, so the cached CSV is what the re-run
    would produce: the judge prompt is byte-identical either way and the search
    trajectory cannot change.

    Keyed on the kernel source itself, so any edit misses and profiles for real.
    Every cache failure falls through to a normal profile -- this must never be
    the reason a round dies.
    """
    out_path = Path(out_csv)
    # "rejected" profiles are advisory context for the judge; "base" profiles are the
    # measurement the round's decision rests on. Labelling them apart here is what
    # lets a reader see which of the two actually costs time.
    _site = "rejected" if "rejected" in out_path.name else "base"
    _t0 = time.perf_counter()
    key = None
    try:
        h = hashlib.sha256()
        h.update(Path(kernel_file).read_bytes())
        h.update(repr(sorted(kernel_names or [])).encode())
        h.update(f"|dev={device_idx}|rep={repeat}".encode())
        try:
            # The harness fixes the input shapes the metrics describe, so a
            # regenerated bench must not read as the same profile.
            h.update(Path(bench_py).read_bytes())
        except OSError:
            pass
        key = h.hexdigest()[:32]
        cached = cache_dir / f"{key}.csv"
        if cached.exists() and cached.stat().st_size > 0:
            shutil.copy2(cached, out_path)
            print(f"[ncu] cache hit ({key[:8]}): source unchanged since a previous "
                  f"profile, skipping ncu run", flush=True)
            run_timing.record(f"ncu:{_site}", time.perf_counter() - _t0,
                              detail="cache_hit")
            return out_path.resolve()
    except Exception as exc:
        print(f"[ncu] cache lookup skipped ({exc.__class__.__name__}: {exc})", flush=True)
        key = None

    try:
        result = profile_bench(
            bench_py=bench_py,
            kernel_names=kernel_names,
            kernel_file=kernel_file,
            out_csv=out_csv,
            device_idx=device_idx,
            repeat=repeat,
            timeout_override=timeout_override,
        )
    except BaseException as exc:
        run_timing.record(f"ncu:{_site}", time.perf_counter() - _t0,
                          detail=f"failed:{exc.__class__.__name__}")
        raise
    run_timing.record(f"ncu:{_site}", time.perf_counter() - _t0,
                      detail=f"names={len(kernel_names or []) or 1}")
    produced = Path(result) if result else out_path.resolve()
    if key:
        try:
            if produced.exists() and produced.stat().st_size > 0:
                cache_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(produced, cache_dir / f"{key}.csv")
        except Exception as exc:
            print(f"[ncu] cache store skipped ({exc.__class__.__name__}: {exc})", flush=True)
    return produced


# ------------------- LLM & eval steps ------------------
def _make_llm_caller(args):

    def call_llm(
        prompt: str,
        sys_prompt: Optional[str] = None,
        log_path: Optional[Path] = None,
        call_type: str = "unknown",
        round_idx: int = -1,
        model_name: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """One model call. *model_name*/*reasoning_effort* override the run defaults.

        The override exists for the MCGS rollout: expansion is the call the search
        makes over and over, so it runs on a cheaper model at high effort while the
        judge, problem-identify and repair calls stay on --model_name. Both go
        through the same Agent SDK path, so both bill subscription credit.
        """
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        # Timed here rather than at each call site: this is the single choke point for
        # every model call, including judge_gate, which passes log_path=None and so
        # never reaches usage.csv at all.
        _model = model_name or args.model_name
        with run_timing.phase_timer(f"llm:{call_type}", round_idx=round_idx):
            res = query_server(
                prompt=prompt,
                system_prompt=sp,
                server_type=args.server_type,
                model_name=_model,
                reasoning_effort=reasoning_effort,
                temperature=args.temperature,
                top_p=args.top_p,
                server_address=args.server_address,
                server_port=args.server_port,
                log_path=str(log_path) if log_path else None,
                call_type=call_type,
                round_idx=round_idx,
            )
        if isinstance(res, list):
            return res[0] if res else ""
        return str(res)
    return call_llm


def _extract_kernel_from_optimization_reply(raw: str) -> str:
    """Extract kernel code from optimization reply that contains mapping + kernel sections.
    
    The optimization reply format is:
    - Section A: Checklist evidence (plan-to-code mapping)
    - Delimiter: === KERNEL CODE STARTS BELOW ===
    - Section B: Kernel code block (```python ... ```)
    
    Returns the kernel code block only.
    """
    delimiter = "=== KERNEL CODE STARTS BELOW ==="
    delimiter_idx = raw.find(delimiter)
    
    if delimiter_idx != -1:
        # Extract everything after the delimiter
        kernel_section = raw[delimiter_idx + len(delimiter):].strip()
        # Extract the first code block from the kernel section
        code = extract_code_block(kernel_section)
        return code
    else:
        # Fallback: if no delimiter found, try to extract the last code block (assuming mapping doesn't have code blocks)
        # This handles cases where LLM didn't follow the format exactly
        code = extract_code_block(raw)
        return code

def _llm_to_kernel(
    prompt: str,
    code_dir: Path,
    call_llm,
    io_dir: Path,
    round_idx,
    sys_prompt: Optional[str] = None,   # New: optional system prompt
    log_path: Optional[Path] = None,
    call_type: str = "unknown",
    io_tag: Optional[str] = None,
    model_name: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> KernelIndividual:
    """LLM → code → save → KernelIndividual (no evaluation).

    *io_tag* overrides the raw-reply filename stem. Several candidates may be
    drawn within one round (best-of-N seeds), and without a distinct stem each
    would overwrite the previous one's saved reply.

    *model_name*/*reasoning_effort* override the run defaults for this one call.
    Used to put the MCGS rollout on a cheaper model at high effort while the
    judge and analysis calls stay put.
    """
    raw = call_llm(
        prompt,
        sys_prompt=sys_prompt,
        log_path=log_path,
        call_type=call_type,
        round_idx=round_idx,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
    # Ensure io_dir exists before writing
    io_dir.mkdir(parents=True, exist_ok=True)
    reply_file = io_dir / f"{io_tag if io_tag else round_idx}_raw_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")
    
    # For optimization calls, extract kernel code using delimiter-aware extraction
    if call_type == "optimization":
        code = _extract_kernel_from_optimization_reply(raw)
    else:
        # For other call types (seed, repair, etc.), use standard extraction
        code = extract_code_block(raw) or raw  # fallback
    
    path = save_kernel_code(code, code_dir)
    ind = KernelIndividual(code)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind

# ================== Top-level worker: MUST live at module top level, not inside another function ==================
def _bench_worker_entry(test_py: str,
                        ref_py: str,
                        device_idx: int,
                        warmup: int,
                        repeat: int,
                        tol: float,
                        conn) -> None:
    """
    Subprocess entry: set GPU, call compare_and_bench, and send result or error
    back to the parent via a Pipe. Note: we pass string paths here to avoid
    non-picklable objects.
    """
    import torch
    from pathlib import Path
    from utils.compile_and_run import CompilationError, CompilationTimeoutError, AccuracyError

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)

        res = compare_and_bench(
            ref_py=Path(ref_py),
            test_py=Path(test_py),
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        conn.send(("ok", res))
    except Exception as e:
        # Clean the error message if helper is available; otherwise fall back to str(e)
        try:
            cleaned = _sanitize_error_message(e)
            msg = _last_n_lines(cleaned)
        except Exception:
            msg = str(e)

        if isinstance(e, CompilationTimeoutError):
            err_type = "CompilationTimeoutError"
        elif isinstance(e, CompilationError):
            err_type = "CompilationError"
        elif isinstance(e, AccuracyError):
            err_type = "AccuracyError"
        else:
            err_type = e.__class__.__name__

        conn.send(("err", {"type": err_type, "message": msg}))
    finally:
        # Try to sync at the end so errors surface within this round
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


# ================== Top-level worker for preloading kernel (must be at module level for pickling) ==================
def _preload_worker(test_kernel_path: str, conn) -> None:
    """
    Subprocess entry: preload kernel to ensure .so is cached.
    This MUST be at module level to be picklable by multiprocessing.
    """
    try:
        import sys as _sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "preload_test_kernel_temp", 
            test_kernel_path
        )
        if spec and spec.loader:
            preload_mod = importlib.util.module_from_spec(spec)
            _sys.modules[spec.name] = preload_mod
            spec.loader.exec_module(preload_mod)
            conn.send(("ok", "loaded"))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        try:
            conn.close()
        except:
            pass


# ================== Keep original behavior: _bench_and_score (uses spawn + top-level worker) ==================
def _bench_and_score(
    ind: KernelIndividual,
    *,
    ref_py: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    phase: str = "seed",
    metrics_dir: Path | None = None,
) -> None:
    """
    Benchmark and update the individual's metrics/score; on exception, fill in
    failure info and save metrics (if a directory is provided).
    Same functionality as the original version, but runs compare_and_bench in a
    **spawned subprocess** to isolate the CUDA context.
    """
    import torch
    from multiprocessing import get_context

    _bench_t0 = time.perf_counter()
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Only pass picklable arguments (e.g., string paths)
    p = ctx.Process(
        target=_bench_worker_entry,
        args=(
            str(ind.code_path),  # type: ignore[attr-defined]
            str(ref_py),
            device_idx,
            warmup,
            repeat,
            tol,
            child_conn,
        ),
    )
    p.start()
    # Parent does not use the child end
    try:
        child_conn.close()
    except Exception:
        pass

    # ========== Add timeout protection: 20 minutes (10 minutes compile + 10 minutes test) ==========
    # Wait for child with timeout
    timeout_occurred = False
    p.join(timeout=1200)  # 20 minutes
    
    # Check if process is still alive after timeout
    if p.is_alive():
        print(f"[{phase}] WARNING: Subprocess timed out after 20 minutes, terminating...", flush=True)
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            print(f"[{phase}] WARNING: Subprocess did not terminate, killing...", flush=True)
            p.kill()
            p.join()
        timeout_occurred = True
    
    payload = None
    try:
        if timeout_occurred:
            # Don't try to receive from a terminated process
            payload = ("err", {"type": "TimeoutError", "message": "Compilation or execution exceeded 20 minute timeout. This may indicate:\n1. GPU runtime errors (e.g., illegal memory access, out-of-bounds indexing) causing the process to hang\n2. Extremely poor performance due to low GPU occupancy or resource conflicts\n3. Infinite loops or deadlocks in the kernel code\n4. Compilation taking too long due to complex template metaprogramming\n\nPlease investigate:\n- Check array indexing and boundary conditions\n- Verify memory access patterns are valid\n- Reduce per-thread memory usage to improve occupancy"})
        elif parent_conn.poll():
            payload = parent_conn.recv()
    except EOFError:
        # The child may crash before it sends the error info (e.g., a corrupted CUDA context)
        # In that case payload stays None and is handled as an error below
        pass
    except Exception as e:
        # Catch other connection errors as well so the program doesn't crash
        print(f"Warning: Failed to receive payload from child process: {e}")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass

    # —— Update metrics/score based on child payload (same logic as before) ——
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase
            # compare_and_bench reports a geometric-mean speedup across every
            # benchmarked shape ("score"); with no get_inputs_extra() hook that
            # is exactly the single-shape ratio computed here as a fallback.
            speedup = metrics.get("score")
            if speedup is None:
                speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            per_shape = metrics.get("per_shape") or []
            if len(per_shape) > 1:
                brk = "  ".join(f"{s['shape']}={s['speedup']:.4f}x" for s in per_shape)
                print(f"[{phase}] score={speedup:.4f} (geomean over {len(per_shape)} shapes: {brk})", flush=True)
            else:
                print(f"[{phase}] score={speedup:.4f}", flush=True)

            # # === Optional: on successful compile+run, copy code to root/test_kernel.py ===
            # try:
            #     from pathlib import Path as _Path
            #     import shutil as _shutil
            #     root_dir = _Path(__file__).resolve().parent
            #     dst = root_dir / "test_kernel.py"
            #     src = _Path(ind.code_path)  # type: ignore[arg-type]
            #     if src.exists():
            #         _shutil.copy2(src, dst)
            #         print(f"[{phase}] saved successful kernel to: {dst}")
            #     else:
            #         print(f"[{phase}] WARNING: source code file not found: {src}")
            # except Exception as _copy_exc:
            #     print(f"[{phase}] WARNING: failed to save test_kernel.py: {_copy_exc}")

        else:
            err_type = "RuntimeError"
            message = data
            if isinstance(data, dict):
                err_type = data.get("type", err_type) or err_type
                message = data.get("message", message)

            if not isinstance(message, str):
                message = str(message)

            print(f"\033[91mTest Error ({err_type}):\033[0m {message}", flush=True)
            ind.metrics = {
                "runnable": False,
                "phase": phase,
                "error_type": err_type,
                "message": message,
            }
            ind.score = float("-inf")
            print(f"[{phase}] failed. See metrics.message for details.", flush=True)
    else:
        # Subprocess exited unexpectedly with no payload
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": "SubprocessCrashed",
            "message": "subprocess exited unexpectedly (no payload received)",
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. Subprocess crashed.", flush=True)

    # —— As before: try to save metrics regardless of success/failure —— 
    if metrics_dir is not None:
        try:
            saved = ind.save_metrics(metrics_dir)
            print(f"[{phase}] metrics saved to: {saved}", flush=True)
        except Exception as save_exc:
            print(f"[{phase}] WARNING: failed to save metrics: {save_exc}", flush=True)

    # Light cleanup in parent
    # NOTE: Avoid CUDA operations in the parent so GPU errors from the child don't propagate here
    # The child's CUDA context is isolated; the parent neither needs to nor should clean it up
    if torch.cuda.is_available():
        try:
            # Only free memory, no synchronization (avoids triggering CUDA errors left by the child)
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Covers compile + correctness + the timed loop, i.e. everything the 1200s join
    # above bounds. A timed-out bench is recorded too -- that is 20 minutes spent.
    _bench_dt = time.perf_counter() - _bench_t0
    print(f"[{phase}] bench took {_bench_dt:.1f}s", flush=True)
    run_timing.record(f"bench:{phase}", _bench_dt,
                      detail="timeout" if timeout_occurred else "ok")



# ---------------------- task helpers -------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """If a directory, return all .py files (sorted); if a file, return [file]."""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found")


def _filter_tasks_from_summary(all_tasks: List[Path], summary_path: Path) -> List[Path]:
    """Filter tasks based on summary.json, keeping only tasks with best_runnable=false.
    
    Args:
        all_tasks: List of all available task paths
        summary_path: Path to summary.json file
        
    Returns:
        Filtered list of task paths that match best_runnable=false tasks in summary.json
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    # Load summary.json
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    
    # Extract task paths with best_runnable=false
    failed_tasks = []
    for task_info in summary_data.get("tasks", []):
        if task_info.get("best_runnable") is False:
            task_path_str = task_info.get("task", "")
            if task_path_str:
                failed_tasks.append(task_path_str)
    
    print(f"[Filter] Found {len(failed_tasks)} tasks with best_runnable=false in summary.json")
    
    # Match failed tasks to actual file paths
    # Convert task paths from summary (e.g., "KernelBench/level2/100_ConvTranspose3d_Clamp_Min_Divide.py" or "19_ReLU")
    # to actual Path objects by matching against all_tasks
    matched_tasks = []
    for failed_task in failed_tasks:
        # Extract the base name (filename without extension)
        # Handle both formats: "KernelBench/level1/19_ReLU.py" and "19_ReLU"
        task_path_obj = Path(failed_task)
        task_base_name = task_path_obj.stem  # filename without extension (e.g., "19_ReLU")
        task_filename_with_ext = task_path_obj.name  # filename with extension if present
        
        # Try multiple matching strategies
        matched = False
        for task_path in all_tasks:
            # Strategy 1: Exact match with extension (handles "KernelBench/level1/19_ReLU.py" -> "19_ReLU.py")
            if task_path.name == task_filename_with_ext:
                matched_tasks.append(task_path)
                matched = True
                break
            # Strategy 2: Match by base name (without extension)
            # This handles cases where summary has "19_ReLU" but file is "19_ReLU.py"
            if task_path.stem == task_base_name:
                matched_tasks.append(task_path)
                matched = True
                break
        
        if not matched:
            print(f"[Filter] WARNING: Could not find task file for '{failed_task}' (searched for base name: '{task_base_name}')")
    
    print(f"[Filter] Matched {len(matched_tasks)} tasks from summary to available task files")
    return sorted(matched_tasks)


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _sample_tasks(all_tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not all_tasks:
        raise RuntimeError("No .py tasks found.")
    k = max(1, min(k, len(all_tasks)))
    if seed is None or seed == 0:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(all_tasks, k)


def _plot_scores(save_path: Path, scores: List[float], err_flags: List[bool], title: str):
    """Plot per-round score curve.
    
    - Green circles (o): runnable kernels (err_flags=False)
    - Red squares (s): non-runnable kernels (err_flags=True)
    """
    xs = list(range(len(scores)))
    plt.figure()
    
    # Separate runnable and non-runnable points
    runnable_xs = []
    runnable_ys = []
    non_runnable_xs = []
    non_runnable_ys = []
    
    for x, y, is_error in zip(xs, scores, err_flags):
        if is_error:
            # Non-runnable: red square
            non_runnable_xs.append(x)
            non_runnable_ys.append(y)
        else:
            # Runnable: green circle
            runnable_xs.append(x)
            runnable_ys.append(y)
    
    # Plot runnable kernels as green circles
    if runnable_xs:
        plt.scatter(runnable_xs, runnable_ys, marker="o", color="green", 
                   s=40, alpha=0.7, label="Runnable", zorder=3)
    
    # Plot non-runnable kernels as red squares
    if non_runnable_xs:
        plt.scatter(non_runnable_xs, non_runnable_ys, marker="s", color="red", 
                   s=40, alpha=0.7, label="Non-runnable", zorder=3)
    
    # Draw connecting line for visualization
    plt.plot(xs, scores, linestyle="-", color="gray", alpha=0.3, linewidth=1, zorder=1)
    
    plt.xlabel("Round")
    plt.ylabel("Speedup (ref/test)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _append_usage_totals(log_path: Path) -> Dict[str, int]:
    """Append a totals row to usage.csv and return the summed token counts."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not log_path.exists():
        return totals

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames or not rows:
        return totals

    for row in rows:
        if row.get("call_type") == "sum" or row.get("timestamp") == "Total":
            continue
        for key in totals:
            try:
                totals[key] += int(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue

    total_row = {fn: "" for fn in fieldnames}
    for key, value in totals.items():
        if key in total_row:
            total_row[key] = str(value)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(total_row)

    return totals


# --------------------- single-task run -----------------
def _nsys_launch_table_block(full_df, top_n: int = 15) -> str:
    """Render the full per-kernel nsys launch table as a prompt block.

    NCU only profiles the kernels we wrote, so the model has no view of the
    library kernels surrounding them. Kernel launch counts expose that: e.g. a
    layout-transform kernel launching more often than the convolution it feeds
    means the block is paying repeated NCHW<->NHWC conversions on large tensors.
    """
    has_time = "time_pct" in full_df.columns and full_df["time_pct"].notna().any()
    sort_col = "time_pct" if has_time else "kernel_launch_count"
    total = int(full_df["kernel_launch_count"].sum())

    # A cuDNN exhaustive autotune probes each candidate engine a handful of
    # times, and those probes can dominate the profile: on vae_block_002 a
    # conv2d_grouped_direct probe took 76.4% of GPU time in FOUR launches, which
    # squeezed every steady-state kernel into rounding error -- the fp32->TF32
    # convertTensor pass showed as "0.5%" when it is really ~17% of the conv it
    # feeds. Round 25 read that 0.5% and correctly ignored it. So flag the
    # one-time kernels and re-base the percentages on steady-state work only.
    try:
        counts = [int(c) for c in full_df["kernel_launch_count"].tolist()]
        max_launch = max(counts) if counts else 0
        setup_cut = max(1.0, 0.10 * max_launch)
        steady_ns = sum(
            float(t) for t, c in zip(full_df["total_time_ns"].tolist(), counts)
            if c >= setup_cut and float(t) == float(t)
        )
    except Exception:
        setup_cut, steady_ns = 0.0, 0.0

    def _is_setup(row) -> bool:
        try:
            return int(row["kernel_launch_count"]) < setup_cut
        except Exception:
            return False

    df = full_df.sort_values(sort_col, ascending=False).head(top_n)

    lines = [
        "",
        "# GPU time by kernel — ALL kernels in the forward pass (Nsight Systems)",
        "",
        "Includes library kernels (cuDNN/cuBLAS/PyTorch) that NCU does not profile.",
        "NCU's per-kernel metrics cover ONLY the kernels in this candidate's source,",
        "so this table is the sole view of how the forward pass actually divides its",
        "time. Read the percentages before choosing a target: optimizing a kernel",
        "that holds 4% of the time caps the whole round at 4%, however well it goes.",
        "Look also for kernels that do no math — layout transforms, copies, casts.",
        "",
        "READ 'steady %', NOT 'raw %'. Rows marked (one-time) are cuDNN autotune",
        "probes and similar setup work: they run a handful of times regardless of",
        "the workload and can swamp the raw column. 'steady %' re-bases on the",
        "kernels that actually run every forward.",
        "",
        "ATTRIBUTE SATELLITE KERNELS TO THE OP THEY SERVE. A vendor conv/GEMM is",
        "not just its mainloop: cuDNN's TF32 path also launches a separate",
        "convertTensor (fp32->TF32) pass, and layout transforms / im2col / casts",
        "belong to the op they surround. Judging 'the conv is at 91% of TF32 peak,",
        "so it has no headroom' from the MAINLOOP ALONE is a mistake — add the",
        "satellite kernels in first. Removing a satellite needs no faster math:",
        "owning the op (e.g. via CUTLASS) can delete it outright.",
        "",
    ]
    if has_time:
        lines += ["| kernel | steady % | raw % | total (us) | launches |",
                  "|---|---|---|---|---|"]
    else:
        lines += ["| kernel | launches |", "|---|---|"]

    for _, r in df.iterrows():
        name = str(r["Kernel Name"])
        if len(name) > 90:
            name = name[:87] + "..."
        if has_time:
            pct = r["time_pct"]
            _tt = r["total_time_ns"]
            us = _tt / 1e3 if _tt == _tt else float("nan")  # NaN-safe; pandas is not imported here
            if _is_setup(r):
                steady = "(one-time)"
            elif steady_ns > 0 and _tt == _tt:
                steady = f"{100.0 * float(_tt) / steady_ns:.1f}%"
            else:
                steady = "n/a"
            lines.append(f"| `{name}` | {steady} | {pct:.1f}% | {us:,.0f} | "
                         f"{int(r['kernel_launch_count'])} |")
        else:
            lines.append(f"| `{name}` | {int(r['kernel_launch_count'])} |")

    lines.append("")
    if has_time:
        # Amdahl off STEADY-STATE work. Quoting the raw column here named the
        # cuDNN autotune probe as "hottest at 76.4%" and derived a 23.6% cap
        # from it -- a bound computed from work that does not run in the forward
        # pass at all.
        steady_rows = [r for _, r in full_df.iterrows() if not _is_setup(r)]
        top = None
        if steady_rows and steady_ns > 0:
            top = max(steady_rows,
                      key=lambda r: (float(r["total_time_ns"])
                                     if float(r["total_time_ns"]) == float(r["total_time_ns"])
                                     else -1.0))
        if top is not None:
            top_name = str(top["Kernel Name"])[:60]
            top_pct = 100.0 * float(top["total_time_ns"]) / steady_ns
            lines.append(
                f"Hottest STEADY-STATE kernel: `{top_name}` at **{top_pct:.1f}%** of "
                f"per-forward GPU time. By Amdahl, a round that leaves it untouched is "
                f"capped at {100.0 - top_pct:.1f}% even if it drives everything else to "
                f"zero. (One-time autotune probes are excluded; they inflate the raw "
                f"column but do not run per forward.)"
            )
        lines.append("")
    lines.append(f"Total kernel launches in the forward pass: **{total}**")
    lines.append("")
    return "\n".join(lines)


# ------------------- stop / resume ---------------------
# A hard kill mid-loop skips everything the post-loop writer produces --
# figures/, optimization_tree.json and summary.json are written only after the
# round loop exits, so an interrupted run loses all three even though every
# per-round artifact survived. These helpers make a stop graceful (finish the
# round, write the artifacts) and make the run resumable from where it stopped.

_STOP_REQUESTED = False
_CHECKPOINT_NAME = "checkpoint.json"


def _install_stop_handler() -> None:
    """Turn SIGINT/SIGTERM into a graceful stop at the next round boundary.

    The signal usually lands in the middle of an LLM call or a benchmark, so the
    flag is only *checked* between rounds: the in-flight round runs to
    completion, its artifacts are saved, then the loop exits normally and the
    post-loop writer runs. Signalling twice restores the default handler so an
    unresponsive run can still be killed outright.
    """
    def _handler(signum, _frame):
        global _STOP_REQUESTED
        if _STOP_REQUESTED:
            # The in-flight round dies here: the checkpoint is only written at the end
            # of a round, so everything this round has spent is about to be replayed.
            # Recorded before the re-raise so the log says the round was abandoned
            # rather than leaving a silent gap for a later reader to misread as work.
            run_timing.event("abort_signal", detail=f"signum={signum} round_abandoned")
            print("\n[stop] Second signal - aborting immediately.", flush=True)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        _STOP_REQUESTED = True
        run_timing.event("stop_signal", detail=f"signum={signum} graceful")
        print(f"\n[stop] Signal {signum} received. Finishing the current round, then writing "
              f"artifacts and a resumable checkpoint. Signal again to abort now.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not the main thread, or the platform lacks this signal


def _ckpt_entry(ind: Optional[KernelIndividual], eval_dir: Path) -> Optional[Dict[str, Any]]:
    """Serialize the pointers needed to rebuild *ind*, not the object itself.

    Code and metrics already live on disk, so the checkpoint stores paths and
    stays small; ``eval_{id:04d}.json`` is where ``save_metrics`` put them.
    """
    if ind is None or not getattr(ind, "code_path", None):
        return None
    eval_path = eval_dir / f"eval_{ind.id:04d}.json"
    return {
        "code_path": str(ind.code_path),
        "eval_path": str(eval_path) if eval_path.exists() else None,
        "score": float(ind.score) if isinstance(ind.score, (int, float)) else None,
    }


def _restore_individual(entry: Optional[Dict[str, Any]],
                        cache: Dict[str, KernelIndividual]) -> Optional[KernelIndividual]:
    """Rebuild a KernelIndividual from a checkpoint entry, reusing *cache*.

    base/best/current routinely point at the SAME kernel file. Rebuilding each
    independently would yield distinct objects for one kernel and silently break
    the identity tests the loop depends on (``best_kernel is not
    current_kernel``), so equal paths must resolve to one shared object.
    """
    if not entry or not entry.get("code_path"):
        return None
    path = Path(entry["code_path"])
    if not path.exists():
        print(f"[resume] WARNING: kernel file is gone, dropping the reference: {path}")
        return None
    key = str(path)
    if key in cache:
        return cache[key]

    ind = KernelIndividual(path.read_text(encoding="utf-8"))
    ind.code_path = path
    ind.score = entry.get("score")
    eval_path = entry.get("eval_path")
    if eval_path and Path(eval_path).exists():
        try:
            ind.metrics = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[resume] WARNING: could not read metrics {eval_path}: {exc}")
    if ind.metrics is None:
        # Score survived, so treat it as the runnable kernel it was; an unreadable
        # metrics file must not demote a kernel that previously passed.
        ind.metrics = {"runnable": ind.score is not None}
    cache[key] = ind
    return ind


def _save_checkpoint(task_root: Path, eval_dir: Path, *, task_path: Path,
                     next_round: int, total_rounds: int, state: Dict[str, Any]) -> None:
    """Write a resumable snapshot of the loop's state. Called every round.

    Written to a temp file and renamed, so a kill during the write leaves the
    previous checkpoint intact rather than a truncated one.
    """
    data = {
        "version": 1,
        "task": str(task_path),
        "next_round": int(next_round),
        "total_rounds": int(total_rounds),
        "base": _ckpt_entry(state.get("base_kernel"), eval_dir),
        "best": _ckpt_entry(state.get("best_kernel"), eval_dir),
        "current": _ckpt_entry(state.get("current_kernel"), eval_dir),
        "repair_chain": _ckpt_entry(state.get("repair_chain_kernel"), eval_dir),
        "base_score": None if state["base_score"] == float("-inf") else float(state["base_score"]),
        "best_score": None if state["best_score"] == float("-inf") else float(state["best_score"]),
        "optimization_tree": state.get("optimization_tree") or {},
        "scores": [float(s) for s in state.get("scores") or []],
        "err_flags": [bool(e) for e in state.get("err_flags") or []],
        "last_score_for_curve": float(state.get("last_score_for_curve") or 0.0),
        # Plateau streaks outlive a session: on vae_block_002 the 8-round drought
        # spanned a restart, so a counter kept only in memory would reset on
        # --resume and the stop would fire late or never.
        "rounds_since_improvement": int(state.get("rounds_since_improvement") or 0),
        # Structural-rewrite debt outlives a session too. Dropping it on resume
        # would silently forgive the debt: the rewrite would keep the base it
        # took on credit and the kernel it displaced would never come back.
        "structural_debt": (
            {
                "kernel": _ckpt_entry(state["structural_debt"].get("kernel"), eval_dir),
                "score": float(state["structural_debt"]["score"]),
                "rounds_left": int(state["structural_debt"]["rounds_left"]),
                "declared_round": int(state["structural_debt"].get("declared_round") or -1),
            }
            if state.get("structural_debt") else None
        ),
        "stop_reason": state.get("stop_reason"),
        # Visit counts are the only part of the MCGS state that cannot be
        # recomputed from artifacts on disk, so they must survive a restart or a
        # resumed run restarts its exploration with the budget already spent.
        "mcgs": state.get("mcgs"),
        "opt_history_files": {str(k): str(v) for k, v in (state.get("opt_history_files") or {}).items()},
        # Ids name the eval_XXXX.json files; rewinding this would overwrite
        # results from rounds that already finished.
        "next_individual_id": int(KernelIndividual._next_id),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        tmp = task_root / (_CHECKPOINT_NAME + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(task_root / _CHECKPOINT_NAME)
    except Exception as exc:
        print(f"[checkpoint] WARNING: failed to save checkpoint: {exc}", flush=True)


def _load_checkpoint(task_root: Path) -> Optional[Dict[str, Any]]:
    path = task_root / _CHECKPOINT_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[resume] WARNING: checkpoint unreadable ({exc}); starting from round 0.")
        return None


def _paired_verdict_worker(reference: str, base_py: str, cand_py: str,
                           device_idx: int, warmup: int, repeat: int, tol: float,
                           margin: float, min_reps: int, max_reps: int, conn,
                           sigma: float = 3.0) -> None:
    """Subprocess entry for the paired base re-measure. String paths only, so
    nothing unpicklable crosses the Pipe."""
    import torch
    from pathlib import Path as _P
    from utils.paired_bench import adaptive_paired_verdict

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)
        out = adaptive_paired_verdict(
            _P(reference), _P(base_py), _P(cand_py),
            device=device_idx, warmup=warmup, repeat=repeat, tol=tol,
            margin=margin, min_reps=min_reps, max_reps=max_reps, sigma=sigma)
        conn.send(("ok", out))
    except Exception as e:
        conn.send(("err", f"{e.__class__.__name__}: {e}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _paired_base_verdict(reference: Path, base_py: Path, cand_py: Path, *,
                         device_idx: int, warmup: int, repeat: int, tol: float,
                         margin: float, min_reps: int, max_reps: int,
                         sigma: float = 3.0,
                         timeout: int = 1800) -> Optional[Dict[str, Any]]:
    """Re-measure the base and the candidate together, and return the verdict.

    The ratchet otherwise compares a candidate measured now against `base_score`,
    a float stored when the base last advanced. In the exp3 run that number was
    taken at 15:15 and still being used as the bar at 17:36. One unchanged kernel
    re-measured half an hour apart on this machine read 1.2014 then 1.2141 ms --
    +1.06%, twice the 0.5% margin the comparison feeds. Three candidates landed
    at +0.40%, +0.47% and +0.49% against that gate, so which side of the line
    they fell on was partly a question of GPU drift.

    Runs in a spawned subprocess for the same reason the round's own benchmark
    does: a CUDA context that two kernel modules have been imported into is not
    something to hand back to the main loop. Both kernels are measured inside
    THAT one process, interleaved -- which is what makes the drift common-mode
    and cancel in the difference.

    Never raises. Any failure returns None and the caller keeps the old
    stored-score comparison, so this can add information but never cost a round.
    """
    from multiprocessing import get_context

    # Held across the whole spawned measurement, not just the launch. This is
    # the most drift-sensitive block in the run -- it interleaves two kernels
    # specifically so a shared time-varying term cancels -- and another lineage
    # benchmarking midway through would reintroduce exactly the term the
    # interleaving exists to remove, biased toward whichever kernel happened to
    # overlap it. Serialized, or the verdict is worthless.
    with gpu_section("paired_verdict"):
        ctx = get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=_paired_verdict_worker,
            args=(str(reference), str(base_py), str(cand_py), device_idx,
                  warmup, repeat, tol, margin, min_reps, max_reps, child_conn,
                  sigma),
        )
        p.start()
        try:
            child_conn.close()
        except Exception:
            pass

        p.join(timeout=timeout)
        if p.is_alive():
            print(f"[base] paired re-measure exceeded {timeout}s; terminating and "
                  f"falling back to the stored base score", flush=True)
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
            return None

        payload = None
        try:
            if parent_conn.poll():
                payload = parent_conn.recv()
        except Exception:
            payload = None

    if not payload or payload[0] != "ok" or payload[1] is None:
        if payload and payload[0] == "err":
            print(f"[base] paired re-measure failed: {payload[1]}; "
                  f"using the stored base score", flush=True)
        return None
    return payload[1]


def _base_optimization_inventory(optimization_tree: Dict[str, Any],
                                 base_name: Optional[str]) -> List[Dict[str, Any]]:
    """List the accepted optimizations the base kernel carries, oldest first.

    Once the base advances (see the ratchet in the round loop), it accumulates
    the mechanisms of every ancestor. The judge is given the base's SOURCE but
    nothing marks which parts are load-bearing, so it cannot tell that e.g. a
    `benchmark=true` conv call is the thing that bought +6.3%. Walking the tree
    recovers that, and it is all data the tree already stores.

    Each entry: method_name, round, and the gain over the immediate parent --
    the gain is what makes "this is worth preserving" concrete.
    """
    if not optimization_tree or not base_name:
        return []
    chain, cur, seen = [], base_name, set()
    while cur and cur in optimization_tree and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = optimization_tree[cur].get("parent")

    inventory: List[Dict[str, Any]] = []
    for name in reversed(chain):  # oldest ancestor first
        node = optimization_tree.get(name) or {}
        strategy = node.get("strategy")
        method = strategy.get("method_name") if isinstance(strategy, dict) else None
        if not method:
            continue  # the seed has no method
        parent = optimization_tree.get(node.get("parent") or "") or {}
        sp, psp = node.get("speedup"), parent.get("speedup")
        gain = ((sp / psp - 1.0) * 100.0) if (sp and psp) else None
        # `gain` is this kernel's score over its parent's, and the two were taken in
        # different sessions, so it carries GPU drift -- +0.9..+1.7% on this machine,
        # against a 0.5% adoption margin. Ablating the five mechanisms of the exp3
        # round-9 base showed what that is worth: three claimed +0.90%, +1.12% and
        # +1.18% but measured 0.09-0.17% when removed and re-measured side by side,
        # i.e. indistinguishable from zero. `measured_*` carries the paired,
        # same-session figure the ratchet recorded, and is None for any mechanism
        # adopted before that machinery existed -- which is the honest state, not a
        # gap to paper over with `gain_pct`.
        _pv = node.get("paired_verdict") or {}
        inventory.append({
            "method_name": method,
            "round": node.get("round"),
            "speedup": sp,
            "gain_pct": gain,
            "measured_pct": _pv.get("rel_pct"),
            "measured_se_pct": _pv.get("se_pct"),
            "measured_t": _pv.get("t"),
            "measured_resolved": _pv.get("resolved"),
            # The judge is asked for this key under two spellings depending on which
            # output schema it followed, so accept both -- reading only one silently
            # blanks the summary and the inventory entry degrades to a bare name.
            "summary": (strategy.get("primary_optimisation_method")
                        or strategy.get("optimisation method")
                        or strategy.get("optimization method")
                        or "")[:200],
        })
    return inventory


def _run_single_task(task_path: Path, args, batch_dir: Path) -> Dict[str, Any]:
    # --- per-task directories under the SAME batch_dir
    task_root = (batch_dir / task_path.stem).resolve()
    
    # Check if this is a level3 task
    is_level3 = "level3" in str(task_path) or "level3" in str(task_path.parent)
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"
    profile_dir = task_root / "profile"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"
    # Durations, beside the token log. Opened before anything else can spend time, and
    # bounded by process_start/process_exit rows so a reader can tell a gap between
    # rows apart from work -- see utils/run_timing for why that distinction matters.
    run_timing.set_timing_log(task_root / "timing.csv")
    run_timing.event("process_start",
                     detail=f"pid={os.getpid()} subproc_id={args.subproc_id} "
                            f"task={task_path.stem}")

    # === Write the contents of task_path into root/ref.py ===
    root_dir = Path(__file__).resolve().parent
    ref_py = root_dir / f"ref_{args.subproc_id}.py"
    test_kernel = root_dir / f"test_kernel_{args.subproc_id}.py"
    bench_py = root_dir / f"bench_ref_inputs_{args.subproc_id}.py"
    content = task_path.read_text(encoding="utf-8")  # read source from task_path
    with open(ref_py, "w", encoding="utf-8") as f:
        f.write(content)
    
    # === Create bench_ref_inputs_{subproc_id}.py from template ===
    if not bench_py.exists():
        template_path = root_dir / "bench_ref_inputs_0.py"
        if template_path.exists():
            template_content = template_path.read_text(encoding="utf-8")
            # Replace hardcoded ref_0.py and test_kernel_0.py with subproc_id versions
            bench_content = template_content.replace("ref_0.py", f"ref_{args.subproc_id}.py")
            bench_content = bench_content.replace("test_kernel_0.py", f"test_kernel_{args.subproc_id}.py")
            with open(bench_py, "w", encoding="utf-8") as f:
                f.write(bench_content)
        else:
            raise FileNotFoundError(f"Template file {template_path} not found. Cannot create {bench_py}")

    call_llm = _make_llm_caller(args)

    current_kernel: Optional[KernelIndividual] = None
    base_kernel: Optional[KernelIndividual] = None  # Base kernel for optimization (updated with strict conditions)
    base_score: float = float("-inf")
    best_kernel: Optional[KernelIndividual] = None  # Best kernel for statistics (updated unconditionally when score is higher)
    best_score: float = float("-inf")
    # Track optimization history: map from round_idx to opt_history_file path
    # This tracks which round's opt history should be updated after repair
    opt_history_files: Dict[int, Path] = {}
    
    # Repair chain tracking: track the first kernel in a repair chain
    # A repair chain starts when a kernel from opt phase fails, and continues until repair succeeds
    # All repair history for kernels in the same chain should be saved in the same folder
    repair_chain_kernel: Optional[KernelIndividual] = None
    
    # Optimization tree: track kernel genealogy
    # Structure: {kernel_name: {parent, speedup, ncu_passed, strategy, phase, round, ...}}
    optimization_tree: Dict[str, Dict[str, Any]] = {}

    scores: List[float] = []
    err_flags: List[bool] = []
    last_score_for_curve = 0.0  # default baseline for plotting on early failures
    rounds_since_improvement = 0  # consecutive rounds not beating best_score by --base_margin
    # Bound here, not in the loop: a resume that starts past --round never enters the
    # body, and the process_exit timing row below reads both.
    _plateau_stop = False
    _rounds_run = 0  # rounds this process actually completed (not rounds planned)
    # Outstanding structural-rewrite debt, or None. Holds the kernel the rewrite
    # displaced so it can be restored if the rewrite never pays off.
    structural_debt: Optional[Dict[str, Any]] = None

    # ---- Monte Carlo Graph Search state -------------------------------------
    # Replaces the ratchet's rule for choosing what the next round branches from.
    # `graph` owns SELECTION only: measurement (paired verdict), repair, ncu and
    # reporting are untouched, so a regression here cannot silently corrupt the
    # numbers -- it can only send the loop to a worse parent, which shows up in
    # the score curve rather than hiding in it.
    _use_mcgs = (getattr(args, "search", "ratchet") == "mcgs")
    graph = MonteCarloGraphSearch(
        c_puct=args.mcgs_c_puct, lam=args.mcgs_lam,
        widen_k=args.mcgs_widen_k, widen_alpha=args.mcgs_widen_alpha,
        max_depth=args.mcgs_max_depth, reward_scale=args.mcgs_reward_scale,
        state_key_mode=args.mcgs_state_key, merge_tolerance=args.mcgs_merge_tol)
    # Live objects for the graph's state representatives. The graph stores kernel
    # NAMES (it has to be JSON-serialisable for the checkpoint), and selection
    # needs `.code`/`.code_path`, so names are resolved through here and fall
    # back to reading the file when a resume has no live object.
    _kernel_registry: Dict[str, KernelIndividual] = {}
    # The state the current round was selected from, and its value: needed at
    # backup time, one full round after selection happened.
    _mcgs_sel = None
    _mcgs_parent_key: Optional[str] = None

    def _register(ind: Optional[KernelIndividual]) -> None:
        if ind is not None and getattr(ind, "code_path", None):
            _kernel_registry[Path(ind.code_path).stem] = ind

    def _resolve(name: Optional[str], path: Optional[str]) -> Optional[KernelIndividual]:
        """A graph state's representative as a live KernelIndividual."""
        if not name:
            return None
        got = _kernel_registry.get(name)
        if got is not None:
            return got
        if path and Path(path).exists():
            ind = KernelIndividual(Path(path).read_text(encoding="utf-8"))
            ind.code_path = Path(path)
            ind.metrics = {"runnable": True}
            _kernel_registry[name] = ind
            return ind
        return None

    def _state_key_for(ind: Optional[KernelIndividual], *, round_idx: int,
                       mechanisms: Optional[List[str]] = None) -> str:
        """State identity for a kernel, under --mcgs_state_key.

        Features come from the kernel's OWN code via the heuristic extractor, not
        from round*_machine_check_result.json. That file records the features of
        the kernel machine_check PROFILED, which is the round's PARENT (see the
        `cuda_code=parent_kernel_code` call site) -- keying a child by its
        parent's features would merge every child of a parent into one state
        regardless of what the edit did. The heuristic extractor needs no LLM
        call and is deterministic, so it can run on a kernel the same round it is
        produced; the profiled vector is only a fallback.
        """
        name = (Path(ind.code_path).stem
                if (ind is not None and getattr(ind, "code_path", None)) else f"r{round_idx}")
        feats = None
        if args.mcgs_state_key == "features":
            code = getattr(ind, "code", None)
            if code:
                try:
                    from prompts.machine_check_ver2 import extract_code_features_from_cuda
                    feats = extract_code_features_from_cuda(code)
                except Exception as _exc:
                    print(f"[mcgs] feature extraction failed ({_exc}); falling back to the "
                          f"profiled vector for this kernel.", flush=True)
            if not feats:
                feats = load_code_features(io_dir, round_idx)
        return state_key(
            mode=args.mcgs_state_key,
            features=feats,
            mechanisms=mechanisms,
            code=getattr(ind, "code", None),
            fallback=name)

    # ---- resume: rebuild the loop's state from the last completed round ----
    start_round = 0
    if getattr(args, "resume", None):
        ckpt = _load_checkpoint(task_root)
        if ckpt is None:
            print(f"[resume] No checkpoint under {task_root}; starting from round 0.")
        elif ckpt.get("task") != str(task_path):
            print(f"[resume] WARNING: checkpoint belongs to {ckpt.get('task')}, not "
                  f"{task_path}; starting from round 0.")
        else:
            _cache: Dict[str, KernelIndividual] = {}
            base_kernel = _restore_individual(ckpt.get("base"), _cache)
            best_kernel = _restore_individual(ckpt.get("best"), _cache)
            current_kernel = _restore_individual(ckpt.get("current"), _cache)
            repair_chain_kernel = _restore_individual(ckpt.get("repair_chain"), _cache)
            base_score = float(ckpt["base_score"]) if ckpt.get("base_score") is not None else float("-inf")
            best_score = float(ckpt["best_score"]) if ckpt.get("best_score") is not None else float("-inf")
            optimization_tree = ckpt.get("optimization_tree") or {}
            scores = list(ckpt.get("scores") or [])
            err_flags = list(ckpt.get("err_flags") or [])
            last_score_for_curve = float(ckpt.get("last_score_for_curve") or 0.0)
            _sd = ckpt.get("structural_debt")
            if _sd:
                structural_debt = {
                    "kernel": _restore_individual(_sd.get("kernel"), _cache),
                    "score": float(_sd.get("score") or 0.0),
                    "rounds_left": int(_sd.get("rounds_left") or 0),
                    "declared_round": int(_sd.get("declared_round") or -1),
                }
                print(f"[resume] Carrying a structural-rewrite debt forward: base owes "
                      f"{structural_debt['score']:.4f}, {structural_debt['rounds_left']} "
                      f"round(s) of grace left.", flush=True)
            rounds_since_improvement = int(ckpt.get("rounds_since_improvement") or 0)
            opt_history_files = {int(k): Path(v) for k, v in (ckpt.get("opt_history_files") or {}).items()}
            # The graph carries the visit counts, which are the only thing in this
            # method that cannot be recomputed from artifacts on disk. Dropping
            # them on resume would reset every Q to zero and restart exploration
            # from scratch with the budget already spent.
            if ckpt.get("mcgs"):
                graph = MonteCarloGraphSearch.from_dict(ckpt["mcgs"])
                _g = graph.stats()
                print(f"[mcgs] Restored the search graph: {_g['states']} states over "
                      f"{_g['kernels']} kernels, {_g['merged_states']} merged, "
                      f"mean N={_g['mean_N']:.2f}, {_g['total_visits']} visits.", flush=True)
            elif _use_mcgs:
                print("[mcgs] WARNING: this checkpoint predates --search mcgs and carries no "
                      "graph. Rebuilding from the best kernel as a fresh root; the earlier "
                      "rounds' visit statistics are not recoverable.", flush=True)
            for _ind in (base_kernel, best_kernel, current_kernel):
                _register(_ind)
            # Restoring individuals bumps the id counter, so set it afterwards;
            # rewinding would overwrite eval_XXXX.json files already on disk.
            KernelIndividual._next_id = max(int(ckpt.get("next_individual_id") or 0),
                                            KernelIndividual._next_id)
            start_round = int(ckpt.get("next_round") or 0)
            _best_txt = f"{best_score:.4f}" if best_score != float("-inf") else "none"
            # The row that makes downtime legible: paired with the previous session's
            # process_exit, it bounds a gap that is emphatically not work.
            run_timing.event("resume", round_idx=start_round,
                             detail=f"start_round={start_round} best={_best_txt}")
            print(f"[resume] Restored {task_root/_CHECKPOINT_NAME}: continuing at round "
                  f"{start_round}/{args.round}, best={_best_txt}", flush=True)
            if rounds_since_improvement:
                print(f"[resume] Carrying a {rounds_since_improvement}-round plateau streak "
                      f"forward (--patience {args.patience}).", flush=True)
            if args.patience and rounds_since_improvement >= args.patience:
                print(f"[resume] WARNING: this run already stopped on the plateau rule. It will "
                      f"run one more round and stop again unless you raise --patience or pass "
                      f"--patience 0.", flush=True)
            if start_round >= args.round:
                print(f"[resume] This run already completed {start_round} rounds. Raise --round "
                      f"above {start_round} to extend it; artifacts will be rewritten as-is.",
                      flush=True)

    # Clear orphaned torch extension build locks before anything compiles. A
    # baton left behind by a killed process never expires on its own, and the
    # kernel that later picks that extension name hangs until the compile alarm
    # and is then misreported as an illegal memory access -- scored -inf and sent
    # to repair for a bug it does not have. Swept here rather than only before
    # the seed because every round compiles, and the poisoned name is chosen by
    # the model, not by the round index. See utils/torch_ext_cache.
    sweep_stale_batons()

    for round_idx in range(start_round, args.round):
        if _STOP_REQUESTED:
            print(f"[stop] Stopping before round {round_idx}. Completed {round_idx} of "
                  f"{args.round} rounds; resume with --resume {batch_dir}", flush=True)
            break
        print(f"[{task_path.name}] Round {round_idx}")
        # Collect batons this run stranded. The startup sweep above cannot: a
        # lineage killed mid-build leaves a lock only minutes old, far inside
        # that 3600s age gate, so without this every later round of THIS run
        # stays exposed to the extension name it orphaned. Liveness is the gate
        # here instead of age -- an unheld lock means the builder is gone -- so
        # it costs nothing when the cache is healthy (no locks, no /proc walk).
        sweep_unheld_batons()
        # Reset every round: only a round whose judge JSON asks for it may take
        # structural grace, and a stale True would hand it to an unrelated round.
        _structural_declared = False
        # Same hazard, worse consequence. The selection is made in the OPT branch
        # but consumed in the accept block outside it, so a round that goes to
        # repair instead would otherwise still be holding the previous round's
        # selection -- and would back its result up that stale path, crediting a
        # state that had nothing to do with it. Cleared here so a round with no
        # selection of its own records nothing.
        _mcgs_sel = None
        _mcgs_parent_key = None
        # Assigned only on the opt path; read by the MCGS mechanism key. Reset so
        # a repair round cannot inherit the previous round's method name.
        strategy_json = None
        # Snapshot for the plateau test. best_score is updated from several
        # places (opt accept, repair accept, statistics-only bump), so comparing
        # the round's start against its end catches every path -- including the
        # repair rounds, which burn wall clock without advancing the search.
        best_score_at_round_start = best_score
        # Attribute every row written from here on -- including the ones emitted deep
        # inside run_ncu_memory, which has no notion of a round -- to this round.
        run_timing.set_round(round_idx)
        _round_t0 = time.perf_counter()

        if round_idx == 0:
            print("[Seed] Generating the initial kernel ...")
            # Profile the reference BEFORE the seed. The granularity the model
            # picks here fixes what it may rewrite for the entire run and is
            # never revisited, so it must not be chosen without knowing where
            # the time actually goes. Advisory: None on failure, prompt unchanged.
            print("[ref_profile] Profiling reference model to inform granularity choice ...", flush=True)
            ref_profile_block = build_reference_profile_block(task_path, device_idx=args.device)
            if ref_profile_block:
                (io_dir / f"round{round_idx:03d}_reference_profile.txt").write_text(
                    ref_profile_block, encoding="utf-8")
                for _l in ref_profile_block.splitlines()[:3]:
                    print(f"[ref_profile] {_l}", flush=True)
            seed_prompt = build_seed_prompt(arch_path=task_path, gpu_name=args.gpu,
                                            reference_profile=ref_profile_block,
                                            force_granularity=getattr(args, "seed_granularity", None),
                                            force_algorithm=getattr(args, "seed_algorithm", None))
            prompt_file = io_dir / f"round{round_idx:03d}_seed_prompt.txt"
            prompt_file.write_text(seed_prompt, encoding="utf-8")
            # Best-of-N seeds. Round 0 is a single temperature-1 draw, and the
            # spread between draws is wide enough to dominate what the whole
            # optimisation search adds, so one unlucky sample caps the run. Draw
            # several and keep the best; the extra cost is small next to the
            # rounds a run spends plateaued at the end.
            n_seeds = max(1, int(getattr(args, "num_seeds", 1) or 1))
            seed_cands: List[KernelIndividual] = []
            for seed_i in range(n_seeds):
                if n_seeds > 1:
                    print(f"[Seed] Drawing candidate {seed_i + 1}/{n_seeds} ...", flush=True)
                # A draw that RAISES must cost only that draw. Drawing several
                # seeds exists precisely so one bad sample cannot cap the run,
                # but that only held for kernels that came back and failed to
                # build -- an exception from the LLM call itself propagated out
                # of main() and killed the process. Round 0 has no checkpoint
                # yet, so everything already spent was lost with it: on
                # 2026-08-04 a granularity-D seed raised after 29 minutes and
                # took the whole run with it, without reaching draw 2 of 5.
                try:
                    cand = _llm_to_kernel(
                        seed_prompt, code_dir, call_llm, io_dir, round_idx,
                        log_path=log_path, call_type="seed",
                        io_tag=f"{round_idx}_seed{seed_i}" if n_seeds > 1 else None,
                    )
                except Exception as exc:
                    print(f"[Seed] candidate {seed_i + 1}/{n_seeds} failed to generate "
                          f"({type(exc).__name__}: {str(exc)[:200]}); skipping this draw",
                          flush=True)
                    continue
                _bench_and_score(
                    cand,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="seed",
                    metrics_dir=eval_dir,
                )
                seed_cands.append(cand)

            if not seed_cands:
                raise RuntimeError(
                    f"All {n_seeds} seed draws failed to generate a kernel; nothing to "
                    f"optimise. See the [Seed] lines above for the per-draw errors.")

            def _seed_score(c: KernelIndividual) -> float:
                # A candidate that did not run must never outrank one that did,
                # whatever it left in .score.
                runnable = bool(getattr(c, "metrics", {}).get("runnable", False))
                return c.score if (runnable and c.score is not None) else float("-inf")

            # max() keeps the first on ties, so with every candidate unrunnable
            # this picks candidate 1 and the existing repair path takes over
            # exactly as it did before.
            ind = max(seed_cands, key=_seed_score)
            if n_seeds > 1:
                for seed_i, cand in enumerate(seed_cands):
                    sc = _seed_score(cand)
                    shown = f"{sc:.4f}" if sc != float("-inf") else "not runnable"
                    mark = "  <-- selected" if cand is ind else ""
                    print(f"[Seed] candidate {seed_i + 1}/{n_seeds}: {shown}{mark}", flush=True)
                if _seed_score(ind) == float("-inf"):
                    print("[Seed] No candidate was runnable; continuing with candidate 1 "
                          "for repair", flush=True)


            # Record seed kernel in optimization tree
            if ind and hasattr(ind, 'code_path') and ind.code_path:
                kernel_name = ind.code_path.stem
                runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                speedup = ind.score if (ind.score is not None and runnable) else None
                optimization_tree[kernel_name] = {
                    "parent": None,  # Seed is root
                    "kernel_name": kernel_name,
                    "kernel_path": str(ind.code_path),
                    "speedup": float(speedup) if speedup is not None else None,
                    "runnable": runnable,
                    "ncu_passed": False,  # Seed doesn't go through ncu
                    "phase": "seed",
                    "round": round_idx,
                    "strategy": None,  # Seed has no strategy
                    "method_matched": False,  # Seed doesn't have optimization method matching
                    "timestamp": datetime.now().isoformat(),
                }
                # Root of the search graph. Seeded with the measured score as the
                # chain anchor: every later value is this number times a product
                # of PAIRED relative gains, so the whole ledger stays on the basis
                # the accept decisions are actually made on.
                if _use_mcgs and speedup is not None:
                    _register(ind)
                    _rk = _state_key_for(ind, round_idx=round_idx, mechanisms=[])
                    graph.observe(key=_rk, kernel_name=kernel_name,
                                  kernel_path=str(ind.code_path), value=float(speedup),
                                  parent_key=None, runnable=True, note="seed")
                    print(f"[mcgs] Root state {_rk} <- {kernel_name} at {speedup:.4f} "
                          f"(key mode: {args.mcgs_state_key})", flush=True)

        else:
            is_runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False)) if current_kernel else False

            if not is_runnable:
                print("[Repair] start repairing")
                # Check if we need to update opt history after repair
                # If current_kernel was generated in opt phase of this round, we should update opt history
                opt_history_file_to_update = opt_history_files.get(round_idx)
                
                # ========== Create repair history folder and file ==========
                # Repair chain logic: All repairs for kernels in the same chain should be saved in the same folder
                # A repair chain starts when a kernel from opt phase fails, and continues until repair succeeds
                # If there's no active repair chain, start a new one with current_kernel
                if repair_chain_kernel is None:
                    # Start a new repair chain with current_kernel (the first problematic kernel)
                    repair_chain_kernel = current_kernel
                    print(f"[repair] Starting new repair chain with kernel: {repair_chain_kernel.code_path.stem if (repair_chain_kernel and hasattr(repair_chain_kernel, 'code_path') and repair_chain_kernel.code_path) else 'unknown'}")
                
                # Use repair_chain_kernel (the first kernel in the chain) to create repair history folder
                # This ensures all repairs in the same chain are saved in the same folder
                kernel_to_repair_name = None
                kernel_to_repair_path = None
                repair_history_dir = None
                repair_history_file = None
                repair_round_num = 1
                
                if repair_chain_kernel and hasattr(repair_chain_kernel, 'code_path') and repair_chain_kernel.code_path:
                    kernel_to_repair_path = repair_chain_kernel.code_path
                    kernel_to_repair_name = kernel_to_repair_path.stem  # e.g., "kernel_20251225_185242"
                    repair_history_dir = code_dir / kernel_to_repair_name
                    repair_history_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Count existing repair history files to determine repair round number
                    if repair_history_dir.exists():
                        existing_repair_files = sorted(repair_history_dir.glob("repair_round_*.json"),
                                                       key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
                        if existing_repair_files:
                            # Get the highest round number and increment
                            last_repair_file = existing_repair_files[-1]
                            try:
                                last_round_str = last_repair_file.stem.split("_")[-1]
                                if last_round_str.isdigit():
                                    repair_round_num = int(last_round_str) + 1
                            except Exception:
                                repair_round_num = len(existing_repair_files) + 1
                    
                    repair_history_file = repair_history_dir / f"repair_round_{repair_round_num:03d}.json"
                    # Debug: Print repair chain info
                    current_kernel_name = current_kernel.code_path.stem if (current_kernel and hasattr(current_kernel, 'code_path') and current_kernel.code_path) else "None"
                    base_kernel_name = base_kernel.code_path.stem if (base_kernel and hasattr(base_kernel, 'code_path') and base_kernel.code_path) else "None"
                    best_kernel_name = best_kernel.code_path.stem if (best_kernel and hasattr(best_kernel, 'code_path') and best_kernel.code_path) else "None"
                    print(f"[repair] Creating repair history for repair chain: {kernel_to_repair_name} (repair round {repair_round_num})")
                    print(f"[repair]   - Repair chain kernel (first in chain): {kernel_to_repair_name}")
                    print(f"[repair]   - Current kernel being repaired: {current_kernel_name}")
                    print(f"[repair]   - Base kernel (for optimization): {base_kernel_name}")
                    print(f"[repair]   - Best kernel (statistics): {best_kernel_name}")
                    if kernel_to_repair_name != current_kernel_name:
                        print(f"[repair]   - ✓ Using repair chain kernel for folder (continuing existing chain)")
                    else:
                        print(f"[repair]   - ✓ Using repair chain kernel for folder (new chain)")
                
                error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get(
                    "message", "")) if current_kernel else ""

                # ========== Load repair history for the repair chain ==========
                repair_history = []
                if repair_history_dir and repair_history_dir.exists():
                    try:
                        # Read all existing repair history files (excluding the current one being created)
                        existing_repair_files = sorted(repair_history_dir.glob("repair_round_*.json"),
                                                       key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
                        for hist_file in existing_repair_files:
                            # Skip the current repair round file (not yet completed)
                            if hist_file == repair_history_file:
                                continue
                            try:
                                hist_data = json.loads(hist_file.read_text(encoding="utf-8"))
                                # Only include completed attempts (with test results)
                                if "test_timestamp" in hist_data or "runnable" in hist_data or "speedup" in hist_data:
                                    repair_history.append(hist_data)
                            except Exception as e:
                                print(f"[repair] Warning: Failed to read repair history from {hist_file}: {e}")
                        
                        if repair_history:
                            print(f"[repair] Loaded {len(repair_history)} previous repair attempts from {repair_history_dir}")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to load repair history: {e}")

                # Carry the best kernel that actually passed into the repair. Repair
                # chains branch off the broken lineage, so without this a fix found
                # in one branch is invisible to a repair in another and gets
                # regressed instead of reused. Skipped when the best kernel IS the
                # one being repaired (nothing to diff against).
                known_good_code = None
                known_good_score = None
                if (best_kernel is not None
                        and getattr(best_kernel, "code", None)
                        and best_kernel is not current_kernel
                        and best_score != float("-inf")):
                    known_good_code = best_kernel.code
                    known_good_score = best_score

                problem_system_prompt, problem_prompt = build_correctness_prompts(error_log=error_log,
                                                                                  arch_path=task_path,
                                                                                  cuda_code=current_kernel.code,
                                                                                  repair_history=repair_history if repair_history else None,
                                                                                  reference_kernel=known_good_code,
                                                                                  reference_kernel_score=known_good_score)
                prompt_file = io_dir / f"round{round_idx:03d}_problem_identify_prompt.txt"
                prompt_file.write_text(problem_prompt, encoding="utf-8")
                raw = call_llm(problem_prompt, problem_system_prompt, log_path=log_path,
                               call_type="problem_identify", round_idx=round_idx)
                io_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                reply_file = io_dir / f"{round_idx}_raw_problem_identify_reply.txt"
                reply_file.write_text(raw, encoding="utf-8")
                problem_json = extract_json(raw)

                repair_prompt = build_error_prompt(
                    old_code=current_kernel.code,
                    error_log=error_log,
                    problem=problem_json,
                    gpu_name=args.gpu,
                    reference_kernel=known_good_code,
                    reference_kernel_score=known_good_score,
                )
                if known_good_code:
                    print(f"[repair] Including known-good kernel "
                          f"{best_kernel.code_path.stem} (speedup {known_good_score:.4f}) in repair context",
                          flush=True)
                prompt_file = io_dir / f"round{round_idx:03d}_repair_prompt.txt"
                prompt_file.write_text(repair_prompt, encoding="utf-8")
                
                # ========== Save repair history before repair attempt ==========
                if repair_history_file:
                    try:
                        repair_history_data = {
                            "round": round_idx,
                            "repair_round": repair_round_num,
                            "kernel_to_repair": str(kernel_to_repair_path) if kernel_to_repair_path else None,
                            "kernel_to_repair_name": kernel_to_repair_name,
                            "error_log": error_log[:1000] if error_log else None,  # Truncate long error logs
                            "problem_identification": problem_json if problem_json else None,
                            "repair_strategy": problem_json.get("repair_strategy") if (problem_json and isinstance(problem_json, dict)) else None,
                            "timestamp": datetime.now().isoformat(),
                            "runnable": None,  # Will be updated after testing
                            "speedup": None,  # Will be updated after testing
                            "test_passed": None,  # Will be updated after testing
                            "repaired_kernel": None,  # Will be updated after testing
                            "test_timestamp": None,  # Will be updated after testing
                        }
                        repair_history_file.write_text(json.dumps(repair_history_data, indent=2, ensure_ascii=False), encoding="utf-8")
                        print(f"[repair] Saved repair history to: {repair_history_file}")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to save repair history: {e}")
                
                ind = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                     round_idx, log_path=log_path, call_type="repair")
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="repair",
                    metrics_dir=eval_dir,
                )
                
                # Record repaired kernel in optimization tree
                if ind and hasattr(ind, 'code_path') and ind.code_path:
                    kernel_name = ind.code_path.stem
                    parent_name = current_kernel.code_path.stem if (current_kernel and hasattr(current_kernel, 'code_path') and current_kernel.code_path) else None
                    runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                    speedup = ind.score if (ind.score is not None and runnable) else None
                    optimization_tree[kernel_name] = {
                        "parent": parent_name,
                        "kernel_name": kernel_name,
                        "kernel_path": str(ind.code_path),
                        "speedup": float(speedup) if speedup is not None else None,
                        "runnable": runnable,
                        "ncu_passed": False,  # Repaired kernels don't go through ncu immediately
                        "phase": "repair",
                        "round": round_idx,
                        "strategy": problem_json if problem_json else None,
                        "method_matched": False,  # Repair doesn't have optimization method matching
                        "timestamp": datetime.now().isoformat(),
                    }
                
                # ========== Update repair history after testing ==========
                if repair_history_file and repair_history_file.exists():
                    try:
                        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        speedup = ind.score if (ind.score is not None and runnable) else None
                        repair_history_data = json.loads(repair_history_file.read_text(encoding="utf-8"))
                        repair_history_data["runnable"] = runnable
                        repair_history_data["speedup"] = float(speedup) if speedup is not None else None
                        repair_history_data["test_passed"] = runnable and speedup is not None
                        repair_history_data["repaired_kernel"] = str(getattr(ind, "code_path", None)) if hasattr(ind, "code_path") else None
                        repair_history_data["test_timestamp"] = datetime.now().isoformat()
                        repair_history_file.write_text(json.dumps(repair_history_data, indent=2, ensure_ascii=False), encoding="utf-8")
                        if runnable and speedup is not None:
                            print(f"[repair] Updated repair history: speedup={speedup:.4f}")
                            # Repair chain succeeded, clear it so next problematic kernel starts a new chain
                            repair_chain_kernel = None
                            print(f"[repair] Repair chain completed successfully, cleared repair_chain_kernel")
                        else:
                            print(f"[repair] Updated repair history: repair failed (runnable={runnable}), repair chain continues")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to update repair history: {e}")
                
                # Update opt history after repair if this kernel was from opt phase
                if opt_history_file_to_update and opt_history_file_to_update.exists():
                    try:
                        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        speedup = ind.score if (ind.score is not None and runnable) else None
                        if runnable and speedup is not None:
                            opt_history = json.loads(opt_history_file_to_update.read_text(encoding="utf-8"))
                            opt_history["runnable"] = runnable
                            opt_history["speedup"] = float(speedup)
                            opt_history["test_passed"] = True
                            opt_history["test_kernel"] = str(getattr(ind, "code_path", None)) if hasattr(ind, "code_path") else None
                            opt_history["test_timestamp"] = datetime.now().isoformat()
                            opt_history["repaired"] = True  # Mark that this was repaired
                            opt_history_file_to_update.write_text(json.dumps(opt_history, indent=2, ensure_ascii=False), encoding="utf-8")
                            print(f"[repair] Updated opt history after repair: speedup={speedup:.4f}")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to update opt history after repair: {e}")
            else:
                print("Optimizing start")
                # ========== Determine the kernel to optimize: it should be base_kernel (the baseline kernel that met the update criteria) ==========
                # The optimization phase should keep iterating on base_kernel rather than current_kernel,
                # because current_kernel may have been generated in the previous round, yet is not as good as base_kernel.
                # ---- choose what this round branches from --------------------
                # Under --search mcgs the graph decides, and it decides by moving
                # base_kernel rather than by bypassing it. Everything downstream
                # (ncu naming, the `parent_kernel == base_kernel` repair path, the
                # paired verdict, prompt construction) keeps working unchanged,
                # and the paired comparison then measures the candidate against
                # exactly the state it was branched from -- which is precisely the
                # reward the backup needs. Under 'ratchet' this block is inert.
                _mcgs_sel = None
                _mcgs_parent_key = None
                if _use_mcgs and graph.root is not None:
                    _mcgs_sel = graph.select()
                    if _mcgs_sel is not None:
                        _sel_ind = _resolve(_mcgs_sel.node.rep, _mcgs_sel.node.rep_path)
                        if _sel_ind is not None:
                            _mcgs_parent_key = _mcgs_sel.node.key
                            _switched = (base_kernel is not _sel_ind)
                            base_kernel = _sel_ind
                            base_score = float(_mcgs_sel.node.rep_value)
                            _g = graph.stats()
                            print(f"[mcgs] Round {round_idx}: branching from state "
                                  f"{_mcgs_sel.node.key} (depth {_mcgs_sel.node.depth}, "
                                  f"N={_mcgs_sel.node.N}, Q={_mcgs_sel.node.q(args.mcgs_lam):.3f}, "
                                  f"value {base_score:.4f}) via {_mcgs_sel.node.rep}"
                                  f"{' [SWITCHED parent]' if _switched else ''}", flush=True)
                            print(f"[mcgs]   why: {_mcgs_sel.reason}", flush=True)
                            print(f"[mcgs]   graph: {_g['states']} states / "
                                  f"{_g['kernels']} kernels, {_g['merged_states']} merged, "
                                  f"mean N={_g['mean_N']:.2f}, depth seen "
                                  f"{_g['max_depth_seen']}", flush=True)
                        else:
                            # The representative's file is gone (hand-cleaned run
                            # dir, or a resume against moved artifacts). Fall through
                            # to the incumbent rather than dying mid-run.
                            print(f"[mcgs] WARNING: could not resolve the code for state "
                                  f"{_mcgs_sel.node.key} (rep {_mcgs_sel.node.rep}); "
                                  f"falling back to the incumbent for this round.", flush=True)
                            _mcgs_sel = None
                # parent_kernel is base_kernel_temp; it only counts as a real base_kernel once ncu profiling passes
                parent_kernel = base_kernel if base_kernel is not None else current_kernel
                
                # Make sure the test_kernel file holds the code of parent_kernel (best_kernel_temp) for ncu profiling
                if parent_kernel and hasattr(parent_kernel, 'code'):
                    with open(test_kernel, "w", encoding="utf-8") as f:
                        f.write(parent_kernel.code)
                    print(f"[opt] Updated test_kernel with {'base_kernel (temp, needs profiling)' if base_kernel else 'current_kernel'} for ncu profiling")
                
                # Extract kernel names from the test_kernel file (which should now be base_kernel_temp)
                kernel_names = extract_cuda_kernel_names(test_kernel)
                print("=============================================================")
                print(f"Detected kernel names: {kernel_names} (from {'base_kernel (temp)' if base_kernel else 'current_kernel'})")
                
                # ========== Helper function to handle compilation timeout repair ==========
                def _handle_compilation_timeout(error_stage: str, error_detail: str, kernel_to_repair: Optional[KernelIndividual]):
                    """Handle compilation timeout by calling repair LLM.
                    
                    Args:
                        error_stage: Stage where timeout occurred (e.g., "Pre-compile", "ncu")
                        error_detail: Detailed error message
                        kernel_to_repair: The kernel that needs to be repaired (parent_kernel in opt phase, current_kernel in repair phase)
                    
                    Returns:
                        KernelCode: The repaired kernel (or failed kernel if repair also fails)
                    """
                    # Determine which kernel to repair
                    # In opt phase: repair parent_kernel (best_kernel_temp)
                    # In repair phase: repair current_kernel (the failed kernel)
                    kernel_being_repaired = kernel_to_repair if kernel_to_repair is not None else current_kernel
                    
                    print(f"\n[{error_stage}] ⚠️  COMPILATION TIMEOUT DETECTED!")
                    print(f"[{error_stage}] Attempting to repair kernel: {kernel_being_repaired.code_path if hasattr(kernel_being_repaired, 'code_path') else 'unknown'}")
                    print(f"[{error_stage}] Timeout suggests code issues (e.g., infinite template expansion).")
                    print(f"[{error_stage}] Initiating compilation timeout repair in current round...\n")
                    
                    # Construct error message
                    error_log = (
                        f"[{error_stage} COMPILATION TIMEOUT]\n"
                        f"Compilation exceeded 10 minute timeout.\n"
                        f"Details: {error_detail}\n\n"
                        "IMPORTANT: This kernel previously compiled successfully, but now times out during recompilation.\n"
                        "This indicates the code has characteristics that cause exponential compile-time behavior.\n\n"
                        "Such as:\n"
                        "1. Infinite template recursion or excessive template instantiation\n"
                        "2. Exponential template expansion with nested templates\n"
                        "3. Excessive constexpr evaluation or compile-time computations\n"
                        "4. Large loop unrolling (#pragma unroll with huge iteration counts)\n"
                        "5. Massive inline expansion (very large __forceinline__ functions)\n"
                        "6. Compiler bugs triggered by specific code patterns\n\n"
                        "etc.\n\n"
                        "Required action: Fix the kernel to reduce compilation complexity."
                    )
                    
                    # Use current_kernel to track the kernel being repaired (avoid confusion with best/parent/test)
                    # Update current_kernel to the kernel being repaired before repair process
                    repair_target_code = kernel_being_repaired.code if kernel_being_repaired and hasattr(kernel_being_repaired, 'code') else ""
                    
                    # Call Judger with SPECIALIZED compilation timeout prompt
                    problem_system_prompt, problem_prompt = build_compilation_timeout_prompts(
                        error_log=error_log,
                        cuda_code=repair_target_code
                    )
                    prompt_file = io_dir / f"round{round_idx:03d}_compilation_timeout_{error_stage.lower()}_analysis.txt"
                    prompt_file.write_text(problem_prompt, encoding="utf-8")
                    
                    raw = call_llm(problem_prompt, problem_system_prompt, log_path=log_path,
                                   call_type=f"compilation_timeout_{error_stage.lower()}_analysis", round_idx=round_idx)
                    io_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                    reply_file = io_dir / f"round{round_idx:03d}_compilation_timeout_{error_stage.lower()}_analysis_reply.txt"
                    reply_file.write_text(raw, encoding="utf-8")
                    problem_json = extract_json(raw)
                    
                    # Call Repair LLM to generate fix
                    repair_prompt = build_error_prompt(
                        old_code=repair_target_code,
                        error_log=error_log,
                        problem=problem_json,
                        gpu_name=args.gpu,
                    )
                    prompt_file = io_dir / f"round{round_idx:03d}_compilation_timeout_{error_stage.lower()}_repair.txt"
                    prompt_file.write_text(repair_prompt, encoding="utf-8")
                    
                    repaired_kernel = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                                     round_idx, log_path=log_path, 
                                                     call_type=f"compilation_timeout_{error_stage.lower()}_repair")
                    
                    # Test the repaired kernel
                    print(f"[{error_stage}] Testing repaired kernel after compilation timeout fix...")
                    _bench_and_score(
                        repaired_kernel,
                        ref_py=task_path,
                        device_idx=args.device,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        tol=args.tol,
                        phase=f"compilation_timeout_{error_stage.lower()}_repair",
                        metrics_dir=eval_dir,
                    )
                    
                    # Record repaired kernel in optimization tree
                    if repaired_kernel and hasattr(repaired_kernel, 'code_path') and repaired_kernel.code_path:
                        kernel_name = repaired_kernel.code_path.stem
                        parent_name = kernel_to_repair.code_path.stem if (kernel_to_repair and hasattr(kernel_to_repair, 'code_path') and kernel_to_repair.code_path) else None
                        runnable = bool(getattr(repaired_kernel, "metrics", {}).get("runnable", False))
                        speedup = repaired_kernel.score if (repaired_kernel.score is not None and runnable) else None
                        optimization_tree[kernel_name] = {
                            "parent": parent_name,
                            "kernel_name": kernel_name,
                            "kernel_path": str(repaired_kernel.code_path),
                            "speedup": float(speedup) if speedup is not None else None,
                            "runnable": runnable,
                            "ncu_passed": False,  # Compilation timeout repairs don't go through ncu
                            "phase": f"compilation_timeout_{error_stage.lower()}_repair",
                            "round": round_idx,
                            "strategy": problem_json,
                            "method_matched": False,  # Compilation timeout repair doesn't have optimization method matching
                            "timestamp": datetime.now().isoformat(),
                        }
                    
                    # Note: repaired_kernel will be assigned to current_kernel at the end of the round
                    # Only update best_kernel if the repaired kernel's score exceeds best_score
                    return repaired_kernel
                
                # ========== Preload the kernel to ensure the compile cache exists, avoiding a recompile under ncu ==========
                precompile_timeout = False
                precompile_error_detail = ""
                
                print("[Pre-compile] Loading kernel to ensure .so is cached before ncu profiling...")
                from multiprocessing import get_context
                
                try:
                    ctx = get_context("spawn")
                    parent_conn, child_conn = ctx.Pipe(duplex=False)
                    p = ctx.Process(target=_preload_worker, args=(str(test_kernel), child_conn))
                    p.start()
                    child_conn.close()
                    
                    # 10 minute timeout
                    p.join(timeout=600)
                    
                    if p.is_alive():
                        print(f"[Pre-compile] Timeout after 10 minutes, terminating preload process...")
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()
                            p.join()
                        
                        precompile_timeout = True
                        precompile_error_detail = f"Preload process exceeded 10 minute timeout for kernel: {test_kernel}"
                    else:
                        # Check the result
                        if parent_conn.poll():
                            status, msg = parent_conn.recv()
                            if status == "ok":
                                print("[Pre-compile] Kernel loaded successfully, .so is now cached")
                            else:
                                print(f"[Pre-compile] Warning: Failed to preload kernel: {msg}")
                                precompile_timeout = True
                                precompile_error_detail = f"Preload failed with error: {msg}"
                        else:
                            print("[Pre-compile] Warning: Preload process exited without sending result")
                    
                    parent_conn.close()
                except Exception as e:
                    print(f"[Pre-compile] Warning: Failed to preload kernel: {e}")
                    precompile_timeout = True
                    precompile_error_detail = f"Preload exception: {e}"
                
                # Handle pre-compile timeout by calling repair
                if precompile_timeout:
                    # Repair parent_kernel (best_kernel_temp) that failed to pre-compile
                    ind = _handle_compilation_timeout("Pre-compile", precompile_error_detail, kernel_to_repair=parent_kernel)
                    # The repaired kernel has been tested by _bench_and_score
                    # It will be assigned to current_kernel at the end of the round
                    # IMPORTANT: If parent_kernel == base_kernel and the repaired kernel passed testing,
                    # we should update base_kernel even if score doesn't exceed base_score,
                    # because the original base_kernel cannot pass pre-compile and is "invalid"
                    if parent_kernel == base_kernel:
                        runnable_repaired = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        score_repaired = ind.score if (ind.score is not None and runnable_repaired) else None
                        if score_repaired is not None:
                            # The repaired kernel passed testing, so it should replace the unprofilable base_kernel
                            print(f"[Pre-compile] Repaired kernel (score={score_repaired:.4f}) passed testing, updating base_kernel even though score < base_score ({base_score:.4f})", flush=True)
                            base_score = score_repaired
                            base_kernel = ind
                            # Also update best_kernel unconditionally if score is higher
                            if score_repaired > best_score:
                                best_score = score_repaired
                                best_kernel = ind
                            with open(test_kernel, "w") as f:
                                f.write(base_kernel.code)
                    # Otherwise, only update base_kernel if the repaired kernel's score exceeds base_score
                    # (handled at the end of the round)
                    # Continue to score tracking and next iteration
                    
                else:
                    # ========== ncu profiling with timeout handling ==========
                    ncu_timeout = False  # Flag to indicate if ncu profiling timed out
                    ncu_error_detail = ""
                    
                    try:
                        # For level3 tasks: use repeat=5 and timeout=30 minutes (1800 seconds)
                        ncu_repeat = 3 if is_level3 else args.repeat
                        ncu_timeout_seconds = 1800 if is_level3 else None  # 30 minutes for level3, None for default
                        
                        if is_level3:
                            print(f"[ncu] Level3 task detected: using repeat={ncu_repeat}, timeout={ncu_timeout_seconds//60} minutes", flush=True)
                        
                        # Explicitly specify the kernel file to profile (parent_kernel's code)
                        kernel_file_to_profile = test_kernel  # test_kernel already contains parent_kernel.code
                        print(f"[ncu] Profiling kernel from file: {kernel_file_to_profile} (parent_kernel: {parent_kernel.code_path if parent_kernel and hasattr(parent_kernel, 'code_path') else 'N/A'})", flush=True)
                        
                        csv_path_str = f"ncu_temp_{args.subproc_id}.csv"
                        csv_path_result = _ncu_profile_cached(
                            bench_py=f"bench_ref_inputs_{args.subproc_id}.py",
                            kernel_names=kernel_names,  # pass the kernel names so only the specified kernel is monitored
                            kernel_file=kernel_file_to_profile,  # explicitly specify the kernel file to profile
                            out_csv=csv_path_str,
                            device_idx=args.device,
                            repeat=ncu_repeat,
                            timeout_override=ncu_timeout_seconds,
                            cache_dir=profile_dir / ".ncu_cache",
                        )
                        # profile_bench returns the CSV path (already resolved), ensure it's a Path object
                        csv_path = Path(csv_path_result) if csv_path_result else Path(csv_path_str).resolve()
                        # Store csv_path for error handling
                        csv_path_for_errors = csv_path.resolve()
                        
                        # Save ncu profiling results to profile folder (always save, even if parsing fails)
                        # Use parent_kernel's filename to name the ncu csv file
                        ncu_profile_path = None
                        if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                            import shutil
                            kernel_name = parent_kernel.code_path.stem  # e.g., "kernel_20251229_141824"
                            ncu_profile_path = profile_dir / f"{kernel_name}_ncu.csv"
                            if csv_path.exists() and csv_path.stat().st_size > 0:
                                try:
                                    shutil.copy2(csv_path, ncu_profile_path)
                                    print(f"[ncu] Saved profiling results to: {ncu_profile_path}")
                                except Exception as save_err:
                                    print(f"[ncu] Warning: Failed to save profiling CSV: {save_err}")
                        
                        metrics_df, sections_dict = load_ncu_metrics(csv_path, extra_keep=("Kernel Name", "Block Size", "Grid Size"),
                                                                      name_list=kernel_names, select="last")
                        metrics_block = metrics_to_prompt(metrics_df, sections_dict=sections_dict)

                        # Per-shape speedups for the kernel being profiled. The NCU/nsys
                        # data above covers ONE shape, so without this the judge cannot
                        # tell a real improvement from one that only fits that shape.
                        _ps = ((base_kernel.metrics or {}).get("per_shape")
                               if base_kernel is not None else None)
                        if _ps and len(_ps) > 1:
                            _rows = "\n".join(
                                f"  {s['shape']:<20} ref={s['ref_ms']:.3f} ms  "
                                f"test={s['test_ms']:.3f} ms  speedup={s['speedup']:.4f}x"
                                + ("   <-- profiled above" if s.get("primary") else "")
                                for s in _ps
                            )
                            metrics_block += (
                                "\n\nPER-SHAPE SPEEDUP (score = geometric mean over these shapes)\n"
                                f"{_rows}\n"
                                "NOTE: the profile above is from the shape marked 'profiled above' only.\n"
                                "A method that helps that shape but regresses another is NOT an\n"
                                "improvement — the score is the geometric mean across all of them.\n"
                            )

                        # ========== Profile the previous round's REJECTED kernel ==========
                        # The profile above is of the BASE -- the kernel we optimise FROM. When the
                        # last round produced something that ran but lost, only its scalar speedup
                        # was fed back, so the judge could not tell WHY it lost. Two failures that
                        # look identical as numbers need opposite responses:
                        #   - the predicted metric moved and it was STILL slower  -> the bottleneck
                        #     model is wrong; that direction is a dead end, change target.
                        #   - the predicted metric did NOT move                   -> the mechanism
                        #     never really engaged (fallback taken, guard never fired, cooperative
                        #     launch refused); the method may be fine, retry it fixed.
                        # Only the MOST RECENT rejection is profiled: it is the attempt the judge is
                        # reacting to, and attaching every past failure's profile would bloat an
                        # already ~125KB prompt. Older failures carry the cheap per-shape summary.
                        rejected_metrics_block = None
                        rejected_kernel_name = None
                        rejected_kernel_score = None
                        rejected_base_score = None
                        try:
                            _rej = current_kernel
                            _rej_ok = bool((getattr(_rej, "metrics", {}) or {}).get("runnable", False))
                            if (_rej is not None and _rej is not base_kernel and _rej_ok
                                    and getattr(_rej, "code", None) and _rej.score is not None):
                                rejected_kernel_name = (_rej.code_path.stem
                                                        if getattr(_rej, "code_path", None)
                                                        else "previous_attempt")
                                rejected_kernel_score = float(_rej.score)
                                rejected_base_score = (float(base_score)
                                                       if base_score != float("-inf") else None)
                                # "Not adopted" covers two DIFFERENT outcomes and they must not be
                                # described the same way: a kernel that scored BELOW the base
                                # genuinely lost, while one that scored ABOVE it but under the
                                # margin was simply not resolvable from noise. Telling the judge
                                # the second one "lost" is confidently wrong feedback -- it will
                                # invent a failure explanation for something that mildly worked.
                                if rejected_base_score and rejected_kernel_score < rejected_base_score:
                                    _why = "to explain why it lost"
                                else:
                                    _why = ("-- note it scored ABOVE the base but inside the margin "
                                            "(not resolvable from noise), so this is 'unproven', not 'failed'")
                                print(f"[ncu] Profiling last round's NOT-ADOPTED kernel "
                                      f"{rejected_kernel_name} ({rejected_kernel_score:.4f} vs base "
                                      f"{base_score:.4f}) {_why}", flush=True)
                                _rf = Path(f"rejected_kernel_{args.subproc_id}.py")
                                _rf.write_text(_rej.code, encoding="utf-8")
                                _rn = extract_cuda_kernel_names(_rf)
                                _rcsv = _ncu_profile_cached(
                                    bench_py=f"bench_ref_inputs_{args.subproc_id}.py",
                                    kernel_names=_rn, kernel_file=_rf,
                                    out_csv=f"ncu_rejected_{args.subproc_id}.csv",
                                    device_idx=args.device, repeat=ncu_repeat,
                                    timeout_override=ncu_timeout_seconds,
                                    cache_dir=profile_dir / ".ncu_cache",
                                )
                                _rdf, _rsec = load_ncu_metrics(
                                    Path(_rcsv), extra_keep=("Kernel Name", "Block Size", "Grid Size"),
                                    name_list=_rn, select="last")
                                rejected_metrics_block = metrics_to_prompt(_rdf, sections_dict=_rsec)
                                try:
                                    import shutil as _sh
                                    profile_dir.mkdir(parents=True, exist_ok=True)
                                    _sh.copy(Path(_rcsv),
                                             profile_dir / f"{rejected_kernel_name}_rejected_ncu.csv")
                                except Exception:
                                    pass
                                print(f"[ncu] Rejected-kernel profile captured", flush=True)
                        except Exception as _re:
                            # Optional diagnostic: must never break a round.
                            rejected_metrics_block = None
                            print(f"[ncu] Skipping rejected-kernel profile: "
                                  f"{_re.__class__.__name__}: {_re}", flush=True)

                        # ========== Run nsys profiling to get kernel launch counts ==========
                        nsys_rep_path = None
                        nsys_csv_path = None
                        try:
                            print(f"[nsys] Starting nsys profiling after ncu...", flush=True)
                            _nsys_t0 = time.perf_counter()
                            nsys_rep_path = nsys_profile_bench(
                                bench_py=f"bench_ref_inputs_{args.subproc_id}.py",
                                kernel_names=kernel_names,
                                kernel_file=kernel_file_to_profile,
                                out_rep=f"nsys_temp_{args.subproc_id}.nsys-rep",
                                device_idx=args.device,
                                timeout=300,  # 5 minutes timeout
                            )
                            run_timing.record("nsys", time.perf_counter() - _nsys_t0)
                            # Extract and save launch counts
                            nsys_csv_path = Path(f"nsys_temp_{args.subproc_id}.csv")
                            nsys_df = load_nsys_stats(
                                rep_path=nsys_rep_path,
                                kernel_names=kernel_names,
                                out_csv=nsys_csv_path,
                            )

                            # The filtered CSV above feeds machine_check's scalar
                            # kernel_launch_count (calibrated on our own kernels).
                            # Separately surface the FULL per-kernel launch table in
                            # the prompt: NCU only profiles our kernels, so this is
                            # the sole view of what else runs in the forward pass
                            # (cuDNN layout transforms, conv engines, elementwise).
                            try:
                                full_df = load_nsys_stats(
                                    rep_path=nsys_rep_path,
                                    kernel_names=None,
                                    out_csv=Path(f"nsys_full_{args.subproc_id}.csv"),
                                )
                                if full_df is not None and not full_df.empty:
                                    metrics_block += _nsys_launch_table_block(full_df)
                                    print(f"[nsys] Added full launch table "
                                          f"({len(full_df)} kernels) to optimization prompt", flush=True)
                            except Exception as full_err:
                                print(f"[nsys] Warning: full launch table unavailable: {full_err}")
                            print(f"[nsys] Successfully extracted kernel launch counts", flush=True)
                            
                            # Save nsys results to profile folder
                            if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                                import shutil
                                kernel_name = parent_kernel.code_path.stem
                                nsys_profile_rep_path = profile_dir / f"{kernel_name}_nsys.nsys-rep"
                                nsys_profile_csv_path = profile_dir / f"{kernel_name}_nsys.csv"
                                if nsys_rep_path.exists():
                                    shutil.copy2(nsys_rep_path, nsys_profile_rep_path)
                                    print(f"[nsys] Saved .nsys-rep to: {nsys_profile_rep_path}")
                                if nsys_csv_path.exists():
                                    shutil.copy2(nsys_csv_path, nsys_profile_csv_path)
                                    print(f"[nsys] Saved .csv to: {nsys_profile_csv_path}")
                        except Exception as nsys_error:
                            print(f"[nsys] Warning: nsys profiling failed: {nsys_error}", flush=True)
                            # Continue without nsys data - kernel_launch_count will fall back to len(rows)
                            nsys_csv_path = None
                        
                        # Update optimization tree: mark this kernel as having passed ncu
                        if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                            kernel_name = parent_kernel.code_path.stem
                            if kernel_name in optimization_tree:
                                optimization_tree[kernel_name]["ncu_passed"] = True
                                if ncu_profile_path and ncu_profile_path.exists():
                                    optimization_tree[kernel_name]["ncu_profile_path"] = str(ncu_profile_path)
                    except RuntimeError as ncu_error:
                        # Check if it's a timeout error
                        if "timed out" in str(ncu_error).lower():
                            ncu_timeout = True
                            ncu_error_detail = str(ncu_error)
                        else:
                            # Other ncu errors - still save the CSV if it exists (partial results)
                            print(f"[ncu] ERROR: Profiling failed: {ncu_error}")
                            # Try to save partial CSV results if available
                            if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                                import shutil
                                kernel_name = parent_kernel.code_path.stem
                                # Use the csv_path from profile_bench if available, otherwise try to find it
                                csv_path_temp = csv_path_for_errors if 'csv_path_for_errors' in locals() else Path(f"ncu_temp_{args.subproc_id}.csv").resolve()
                                if csv_path_temp.exists() and csv_path_temp.stat().st_size > 0:
                                    ncu_profile_path = profile_dir / f"{kernel_name}_ncu_error.csv"
                                    try:
                                        shutil.copy2(csv_path_temp, ncu_profile_path)
                                        print(f"[ncu] Saved partial profiling results (error) to: {ncu_profile_path}")
                                    except Exception as save_err:
                                        print(f"[ncu] Warning: Failed to save error CSV: {save_err}")
                                # Mark parent_kernel as not passing ncu
                                if kernel_name in optimization_tree:
                                    optimization_tree[kernel_name]["ncu_passed"] = False
                            print(f"[{task_path.name}] Using previous kernel and continuing")
                            ind = current_kernel
                            scores.append(last_score_for_curve)
                            err_flags.append(True)
                            continue
                    except Exception as ncu_error:
                        print(f"[ncu] ERROR: Unexpected profiling error: {ncu_error}")
                        # Try to save partial CSV results if available
                        if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                            import shutil
                            kernel_name = parent_kernel.code_path.stem
                            # Use the csv_path from profile_bench if available, otherwise try to find it
                            csv_path_temp = csv_path_for_errors if 'csv_path_for_errors' in locals() else Path(f"ncu_temp_{args.subproc_id}.csv").resolve()
                            if csv_path_temp.exists() and csv_path_temp.stat().st_size > 0:
                                ncu_profile_path = profile_dir / f"{kernel_name}_ncu_error.csv"
                                try:
                                    shutil.copy2(csv_path_temp, ncu_profile_path)
                                    print(f"[ncu] Saved partial profiling results (error) to: {ncu_profile_path}")
                                except Exception as save_err:
                                    print(f"[ncu] Warning: Failed to save error CSV: {save_err}")
                        print(f"[{task_path.name}] Using previous kernel and continuing")
                        ind = current_kernel
                        scores.append(last_score_for_curve)
                        err_flags.append(True)
                        continue
                    
                    # Handle ncu timeout by calling repair
                    # Mark parent_kernel as not passing ncu and save timeout CSV if available
                    if ncu_timeout and parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                        import shutil
                        kernel_name = parent_kernel.code_path.stem
                        # Use the csv_path from profile_bench if available, otherwise try to find it
                        csv_path_temp = csv_path_for_errors if 'csv_path_for_errors' in locals() else Path(f"ncu_temp_{args.subproc_id}.csv").resolve()
                        if csv_path_temp.exists() and csv_path_temp.stat().st_size > 0:
                            ncu_profile_path = profile_dir / f"{kernel_name}_ncu_timeout.csv"
                            try:
                                shutil.copy2(csv_path_temp, ncu_profile_path)
                                print(f"[ncu] Saved profiling results (timeout) to: {ncu_profile_path}")
                            except Exception as save_err:
                                print(f"[ncu] Warning: Failed to save timeout CSV: {save_err}")
                        if kernel_name in optimization_tree:
                            optimization_tree[kernel_name]["ncu_passed"] = False
                    
                    if ncu_timeout:
                        # Repair parent_kernel (best_kernel_temp) that failed ncu profiling
                        # parent_kernel has not passed profiling yet, so it's still best_kernel_temp
                        ind = _handle_compilation_timeout("ncu", ncu_error_detail, kernel_to_repair=parent_kernel)
                        # The repaired kernel has been tested by _bench_and_score
                        # It will be assigned to current_kernel at the end of the round
                        # IMPORTANT: If parent_kernel == base_kernel and the repaired kernel passed testing,
                        # we should update base_kernel even if score doesn't exceed base_score,
                        # because the original base_kernel cannot pass ncu profiling and is "invalid"
                        if parent_kernel == base_kernel:
                            runnable_repaired = bool(getattr(ind, "metrics", {}).get("runnable", False))
                            score_repaired = ind.score if (ind.score is not None and runnable_repaired) else None
                            if score_repaired is not None:
                                # The repaired kernel passed testing, so it should replace the unprofilable base_kernel
                                print(f"[ncu] Repaired kernel (score={score_repaired:.4f}) passed testing, updating base_kernel even though score < base_score ({base_score:.4f})", flush=True)
                                base_score = score_repaired
                                base_kernel = ind
                                # Also update best_kernel unconditionally if score is higher
                                if score_repaired > best_score:
                                    best_score = score_repaired
                                    best_kernel = ind
                                with open(test_kernel, "w") as f:
                                    f.write(base_kernel.code)
                        # Otherwise, only update base_kernel if the repaired kernel's score exceeds base_score
                        # (handled at the end of the round)
                        
                    else:
                        # ========== Normal optimization flow (no timeout) ==========
                        # parent_kernel (base_kernel_temp) has passed ncu profiling
                        # Now we can consider it as a valid base_kernel candidate
                        # Only update base_kernel if the optimization result exceeds base_score (with strict conditions)
                        # parent_kernel was already determined at the start of optimization phase (line ~759)
                        # Get the path for optimization history tracking
                        parent_kernel_path = getattr(parent_kernel, "code_path", None) if parent_kernel else None
                        
                        # Create optimization history directory based on parent kernel name
                        opt_history_dir = None
                        opt_history_file = None
                        optimization_history = []
                        if parent_kernel_path:
                            parent_kernel_name = parent_kernel_path.stem  # e.g., "kernel_20251225_185242"
                            opt_history_dir = code_dir / parent_kernel_name
                            opt_history_dir.mkdir(parents=True, exist_ok=True)
                            opt_history_file = opt_history_dir / f"opt_round_{round_idx:03d}.json"
                            
                            # Read all previous optimization history files from this directory
                            if opt_history_dir.exists():
                                hist_files = sorted(opt_history_dir.glob("opt_round_*.json"), 
                                                   key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
                                for hist_file in hist_files:
                                    # Skip the current round's file (not yet completed)
                                    if hist_file == opt_history_file:
                                        continue
                                    try:
                                        hist_data = json.loads(hist_file.read_text(encoding="utf-8"))
                                        # Only include completed attempts (with test results)
                                        if "test_timestamp" in hist_data or "speedup" in hist_data or "test_passed" in hist_data:
                                            optimization_history.append(hist_data)
                                    except Exception as e:
                                        print(f"[opt] Warning: Failed to read optimization history from {hist_file}: {e}")
                            
                            if optimization_history:
                                # Sort by round number to maintain chronological order
                                optimization_history.sort(key=lambda x: x.get("round", 0))
                                print(f"[opt] Loaded {len(optimization_history)} previous optimization attempts from {opt_history_dir}")
                        
                        # Use parent_kernel.code (base_kernel) for judge LLM, not current_kernel
                        # This is the kernel we want to analyze and optimize
                        parent_kernel_code = parent_kernel.code if parent_kernel and hasattr(parent_kernel, 'code') else (current_kernel.code if current_kernel else "")  # type: ignore[union-attr]

                        # What the base already carries. Every round branches from the base, so
                        # once the ratchet advances it the base accumulates earlier mechanisms --
                        # and the profile's hottest remaining targets tend to BE that tuned code.
                        # Without this list the judge cannot tell tuned code from ordinary code and
                        # proposes replacing it, which trades a measured win for an untuned guess.
                        _parent_name = (parent_kernel.code_path.stem
                                        if (parent_kernel and getattr(parent_kernel, "code_path", None))
                                        else None)
                        base_optimizations = _base_optimization_inventory(optimization_tree, _parent_name)
                        if base_optimizations:
                            print("[opt] Base carries " + ", ".join(
                                f"{o['method_name']}"
                                + (f" (+{o['gain_pct']:.2f}%)" if o.get("gain_pct") else "")
                                for o in base_optimizations), flush=True)

                        sys_judge__prompt, judge_prompt = build_judger_optimization_prompts(
                            arch_path=task_path,
                            gpu_name=args.gpu,
                            ncu_metrics_block=metrics_block,
                            metrics_df=metrics_df,  # Pass metrics_df for machine_check
                            cuda_code=parent_kernel_code,  # Use parent_kernel (best_kernel) code
                            optimization_history=optimization_history if optimization_history else None,
                            code_features=None,  # Will be extracted via judge_gate if call_llm is provided
                            call_llm=call_llm,  # Pass call_llm for code_features extraction
                            nsys_csv_path=nsys_csv_path,  # Pass nsys CSV path for kernel_launch_count
                            io_dir=io_dir,  # Pass io_dir for saving machine_check_result JSON
                            round_idx=round_idx,  # Pass round_idx for filename
                            base_optimizations=base_optimizations,
                            rejected_metrics_block=rejected_metrics_block,
                            rejected_kernel_name=rejected_kernel_name,
                            rejected_kernel_score=rejected_kernel_score,
                            rejected_base_score=rejected_base_score,
                        )
                        prompt_file = io_dir / f"round{round_idx:03d}_judge_optimization_prompt.txt"
                        prompt_file.write_text(judge_prompt, encoding="utf-8")
                        raw = call_llm(judge_prompt, sys_judge__prompt, log_path=log_path,
                                       call_type="judge_optimization", round_idx=round_idx)
                        io_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                        reply_file = io_dir / f"{round_idx}_optimization_strategy_reply.txt"
                        reply_file.write_text(raw, encoding="utf-8")
                        try:
                            strategy_json = extract_json(raw)
                        except ValueError as exc:
                            # A malformed judge reply must not kill a multi-round run.
                            # Skip this round's optimization; best_kernel is retained
                            # and the next round starts from it as usual.
                            print(f"[judge] Could not parse optimization strategy "
                                  f"({exc.__class__.__name__}); skipping round {round_idx}. "
                                  f"Raw reply kept at {reply_file}", flush=True)
                            continue  # `round_idx` advances via the enclosing for-loop

                        # Does this round's plan ask for structural grace? Only
                        # an explicit true counts; a missing field means no.
                        if isinstance(strategy_json, dict):
                            _sr = strategy_json.get("structural_rewrite")
                            _structural_declared = (
                                _sr is True
                                or (isinstance(_sr, str) and _sr.strip().lower() == "true")
                            )
                            if _structural_declared and args.structural_grace <= 0:
                                print("[base] Judge declared a structural rewrite but "
                                      "--structural_grace is 0; the ratchet applies as usual.",
                                      flush=True)

                        # Check if method was matched based on machine_check_result
                        # Read machine_check_result JSON file to determine if method was matched
                        method_matched = False
                        machine_check_result_file = io_dir / f"round{round_idx:03d}_machine_check_result.json"
                        if machine_check_result_file.exists():
                            try:
                                with open(machine_check_result_file, 'r', encoding='utf-8') as f:
                                    machine_check_result = json.load(f)
                                case_id = machine_check_result.get("case_id", "NO_MATCH")
                                # method_matched is True only if case_id is not "NO_MATCH"
                                method_matched = (case_id != "NO_MATCH")
                            except Exception as e:
                                print(f"[WARNING] Failed to read machine_check_result.json: {e}. Falling back to checking method_name.")
                                # Fallback: check if method_name exists
                                method_matched = bool(
                                    strategy_json 
                                    and isinstance(strategy_json, dict)
                                    and strategy_json.get("method_name")
                                    and str(strategy_json.get("method_name", "")).strip() != ""
                                )
                        else:
                            print(f"[WARNING] machine_check_result.json not found: {machine_check_result_file}. Falling back to checking method_name.")
                            # Fallback: check if method_name exists
                            method_matched = bool(
                                strategy_json 
                                and isinstance(strategy_json, dict)
                                and strategy_json.get("method_name")
                                and str(strategy_json.get("method_name", "")).strip() != ""
                            )

                        # Save optimization strategy to history file
                        if opt_history_file:
                            opt_history = {
                                "round": round_idx,
                                "parent_kernel": str(parent_kernel_path) if parent_kernel_path else None,
                                "parent_kernel_name": parent_kernel_name if parent_kernel_path else None,
                                "optimization_strategy": strategy_json,
                                "method_matched": method_matched,
                                "timestamp": datetime.now().isoformat(),
                                "runnable": None,  # Will be updated after testing
                                "speedup": None,   # Will be updated after testing
                                "test_passed": False,
                                "repaired": False,  # Will be set to True if repaired
                                # Previous round's kernel. `ind` holds it on a fresh run only
                                # because an earlier loop iteration bound it; after --resume the
                                # loop starts mid-run and `ind` is unbound, raising
                                # UnboundLocalError. current_kernel is assigned `ind` at the end of
                                # every round AND restored from the checkpoint, so it is the same
                                # object on a fresh run and the correct one after a resume.
                                "kernel_source": getattr(current_kernel, "code", ""),
                            }
                            opt_history_file.write_text(json.dumps(opt_history, indent=2, ensure_ascii=False), encoding="utf-8")
                            # Track this opt history file for potential repair updates
                            opt_history_files[round_idx] = opt_history_file
                            print(f"[opt] Optimization history saved to: {opt_history_file}")

                        # Build history block with previously generated kernels (keep last round_idx kernels, or at least 5)
                        # For round 0, keep_last=0 means no history; for round 1+, keep_last should be round_idx to include all previous rounds
                        history_block = _build_history_block(code_dir, keep_last=max(round_idx, 5))
                        # Use parent_kernel (best_kernel) for optimization, not current_kernel
                        opt_prompt = build_optimization_prompt(
                            arch_path=parent_kernel_path if parent_kernel_path else current_kernel.code_path,  # type: ignore[union-attr]
                            gpu_name=args.gpu,
                            history_block=history_block,  # Pass history_block to include previously generated kernels
                            optimization_suggestion=strategy_json
                        )
                        prompt_file = io_dir / f"round{round_idx:03d}_opt_prompt.txt"
                        prompt_file.write_text(opt_prompt, encoding="utf-8")
                        # THE ROLLOUT. In MCGS terms this is the expansion: one
                        # child drawn from the selected state. It is the call the
                        # search repeats every round and ~95% of the wall clock,
                        # so it runs on --rollout_model at --rollout_effort while
                        # the judge/analysis calls above stay on --model_name.
                        print(f"[rollout] Expanding with {args.rollout_model} at "
                              f"effort={args.rollout_effort} (subscription credit); "
                              f"judge/analysis remain on {args.model_name}", flush=True)
                        ind = _llm_to_kernel(opt_prompt, code_dir, call_llm, io_dir, round_idx,
                                             log_path=log_path, call_type="optimization",
                                             model_name=args.rollout_model,
                                             reasoning_effort=args.rollout_effort)
                        _bench_and_score(
                            ind,
                            ref_py=task_path,
                            device_idx=args.device,
                            warmup=args.warmup,
                            repeat=args.repeat,
                            tol=args.tol,
                            phase="opt",
                            metrics_dir=eval_dir,
                        )
                        
                        # Check if optimized kernel failed - if so, prepare for potential repair chain
                        # The repair chain will be started in the repair phase if the kernel is not runnable
                        opt_kernel_runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        opt_kernel_score = ind.score if (ind.score is not None and opt_kernel_runnable) else None
                        if not opt_kernel_runnable or opt_kernel_score is None:
                            # Optimized kernel failed, will need repair
                            # If there's no active repair chain, the repair phase will start one with this kernel
                            print(f"[opt] Optimized kernel failed (runnable={opt_kernel_runnable}), will start repair chain if needed")
                        
                        # Record optimized kernel in optimization tree
                        if ind and hasattr(ind, 'code_path') and ind.code_path:
                            kernel_name = ind.code_path.stem
                            parent_name = parent_kernel.code_path.stem if (parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path) else None
                            runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                            speedup = ind.score if (ind.score is not None and runnable) else None
                            # Check if method was matched (method_name exists and is not empty)
                            method_matched = bool(
                                strategy_json 
                                and isinstance(strategy_json, dict)
                                and strategy_json.get("method_name")
                                and str(strategy_json.get("method_name", "")).strip() != ""
                            )
                            
                            optimization_tree[kernel_name] = {
                                "parent": parent_name,
                                "kernel_name": kernel_name,
                                "kernel_path": str(ind.code_path),
                                "speedup": float(speedup) if speedup is not None else None,
                                "runnable": runnable,
                                "ncu_passed": True,  # This kernel's parent went through ncu profiling
                                "phase": "opt",
                                "round": round_idx,
                                "strategy": strategy_json if strategy_json else None,
                                "method_matched": method_matched,
                                "timestamp": datetime.now().isoformat(),
                            }
                        
                        # Update optimization history after testing
                        if opt_history_file and opt_history_file.exists():
                            try:
                                opt_history = json.loads(opt_history_file.read_text(encoding="utf-8"))
                                runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                                speedup = ind.score if (ind.score is not None and runnable) else None
                                opt_history["runnable"] = runnable
                                opt_history["speedup"] = float(speedup) if speedup is not None else None
                                opt_history["test_passed"] = runnable and speedup is not None
                                opt_history["test_kernel"] = str(getattr(ind, "code_path", None)) if hasattr(ind, "code_path") else None
                                opt_history["test_timestamp"] = datetime.now().isoformat()
                                # Record WHAT WAS MEASURED, not just the scalar. A bare speedup
                                # cannot distinguish "the mechanism is wrong" from "the mechanism
                                # works but is applied unconditionally and one shape collapsed" --
                                # opposite conclusions with opposite next moves. The per-shape
                                # split makes the second case visible without any extra profiling.
                                _ps = (getattr(ind, "metrics", {}) or {}).get("per_shape") or []
                                if _ps:
                                    opt_history["per_shape"] = [
                                        {"shape": s.get("shape"), "speedup": s.get("speedup")} for s in _ps
                                    ]
                                if base_score != float("-inf"):
                                    opt_history["base_score_at_attempt"] = float(base_score)
                                    _bps = ((base_kernel.metrics or {}).get("per_shape")
                                            if base_kernel is not None else None) or []
                                    _bmap = {s.get("shape"): s.get("speedup") for s in _bps}
                                    if _ps and _bmap:
                                        opt_history["per_shape_vs_base_pct"] = [
                                            {"shape": s.get("shape"),
                                             "delta_pct": (s["speedup"] / _bmap[s["shape"]] - 1.0) * 100.0}
                                            for s in _ps if _bmap.get(s.get("shape")) and s.get("speedup")
                                        ]
                                opt_history_file.write_text(json.dumps(opt_history, indent=2, ensure_ascii=False), encoding="utf-8")
                                if runnable and speedup is not None:
                                    print(f"[opt] Optimization history updated: speedup={speedup:.4f}")
                            except Exception as e:
                                print(f"[opt] Warning: Failed to update optimization history: {e}")

        # -------- update state + record curve --------
        # current_kernel: the kernel just generated/repaired, used for logging and any subsequent repair
        # During repair, current_kernel tracks the kernel being repaired to avoid confusion with best, test, and parent
        current_kernel = ind
        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
        this_score = ind.score if (ind.score is not None and runnable) else None

        # If a kernel from opt phase fails (not runnable or no valid score), start/continue repair chain
        # If a kernel succeeds (has valid score), clear repair chain (repair chain completed)
        if this_score is not None:
            # Kernel succeeded, clear repair chain if it exists
            if repair_chain_kernel is not None:
                print(f"[repair] Kernel succeeded (speedup={this_score:.4f}), clearing repair chain")
                repair_chain_kernel = None
        else:
            # Kernel failed, if it came from opt phase and there's no active repair chain, start one
            # Note: repair_chain_kernel is set in repair phase, so we don't set it here
            # This is just for logging/debugging
            pass

        if this_score is not None:
            last_score_for_curve = this_score
            scores.append(this_score)
            err_flags.append(False)
            
            # Advance base_kernel whenever this kernel beats the current base by more than
            # benchmark noise. base_kernel is what every later optimization round branches from
            # (see `parent_kernel = base_kernel` in the opt phase), so this is the ratchet: without
            # it, each round re-derives from the same starting kernel and improvements never
            # compound, no matter how good an intermediate result was.
            #
            # This previously required score >= base_score * 1.3 OR score - base_score >= 0.3.
            # Kernel speedups move in 1-5% steps, so at a base of ~1.12 those gates demanded ~1.46x
            # to fire -- they never did, and the base stayed pinned to the seed for entire runs
            # while best_kernel (statistics only) silently tracked the real winner.
            #
            # Note: special cases where an unprofilable base is replaced by its repair are handled
            # in the pre-compile / ncu handlers above, and intentionally bypass this margin.
            should_update_base = False
            _paired_note = None   # set only when a side-by-side measurement was taken
            # Must be cleared EVERY round, not just on the branch that measures.
            # It is read below by the best_kernel gate, and the branches that skip
            # the paired measurement (first score, unusable base) would otherwise
            # leave the PREVIOUS round's verdict visible and judge this round's
            # candidate against it.
            _verdict = None
            if base_score == float("-inf"):
                # First valid score, always update
                should_update_base = True
            elif base_score <= 0:
                # Base is unusable (negative/zero); any real speedup is an improvement
                should_update_base = (this_score > 0)
            else:
                # Decide against the base MEASURED NOW, not against base_score --
                # a float stored when the base last advanced, hours earlier. On
                # this machine one unchanged kernel drifted +1.06% in 30 minutes,
                # twice the margin, so the stored comparison mixes "is this better"
                # with "which way did the GPU go since then". _paired_base_verdict
                # interleaves the two kernels in one process so that term cancels,
                # and returns None on any failure, leaving the old path intact.
                _verdict = None
                _base_path = getattr(base_kernel, "code_path", None) if base_kernel else None
                _cand_path = getattr(ind, "code_path", None)
                if (args.base_max_reps > 0 and _base_path and _cand_path
                        and Path(_base_path).exists() and Path(_cand_path).exists()):
                    print(f"[base] Re-measuring base and candidate side by side "
                          f"({args.base_reps}-{args.base_max_reps} interleaved reps)...", flush=True)
                    _verdict = _paired_base_verdict(
                        task_path, Path(_base_path), Path(_cand_path),
                        device_idx=args.device, warmup=args.warmup, repeat=args.repeat,
                        tol=args.tol, margin=args.base_margin,
                        min_reps=args.base_reps, max_reps=args.base_max_reps,
                        sigma=args.base_sigma)

                if _verdict is not None:
                    # Two independent questions, two gates. `beats_margin` is a
                    # point estimate: is the gain big enough. It cannot tell a
                    # +0.6% with 0.05% error from a +0.6% with 0.4% error, and on
                    # a noisy kernel the second is ~1.5 sigma from zero -- i.e.
                    # quite possibly nothing. Since the ratchet is one-way and
                    # never re-verified, an advance bought with noise is permanent,
                    # so also require the gain to be clear of zero. A missing or
                    # zero standard error means there is nothing to resolve, so it
                    # passes rather than blocking on a division.
                    _se = _verdict["se_pct"]
                    # The significance decision belongs to the verdict, not here.
                    # rel_pct/se_pct is a t statistic on dof = reps-1, and this
                    # site used to compare it to --base_sigma as though it were a
                    # z score: at 3 reps (dof 2) a true 3-sigma tail needs |t| >=
                    # 19.2, so the old gate was roughly 6x more permissive than it
                    # advertised. adaptive_paired_verdict now evaluates the t
                    # distribution at the right dof and reports `sigma_ok`, with
                    # `sigma_equiv` being what the observed t is really worth.
                    # `_sig` is kept for the pre-sigma_ok fallback path only.
                    _sig = (_verdict["rel_pct"] / _se) if (_se and _se > 0) else float("inf")
                    if "sigma_ok" in _verdict:
                        _sig_ok = bool(_verdict["sigma_ok"])
                        _sig_shown = _verdict.get("sigma_equiv", _sig)
                    else:
                        _sig_ok = (args.base_sigma <= 0) or (_sig >= args.base_sigma)
                        _sig_shown = _sig
                    should_update_base = bool(_verdict["beats_margin"]) and _sig_ok
                    _paired_note = (
                        f"paired {_verdict['rel_pct']:+.2f}% +/-{_verdict['se_pct']:.2f}% "
                        f"({_sig_shown:.1f} sigma equiv, t={_verdict['t']:+.1f} on "
                        f"{_verdict.get('dof', '?')} dof, {_verdict['reps']} reps"
                        f"{', escalated' if _verdict['escalated'] else ''}, "
                        f"{'resolved' if _verdict['resolved'] else 'UNRESOLVED'}; "
                        f"{_verdict['base_ms']:.4f} -> {_verdict['cand_ms']:.4f} ms)")
                    if _verdict["beats_margin"] and not _sig_ok:
                        _paired_note += (f" -- clears the margin but is under "
                                         f"--base_sigma {args.base_sigma:.1f} at "
                                         f"{_verdict.get('dof', '?')} dof, so it is "
                                         f"not separable from noise")
                    # Persist it: an unresolved near-miss looks identical to a
                    # regression in the round record otherwise, which is exactly
                    # how round 19's +0.49% -- later the best kernel of the run --
                    # was reported to the next round as a failure.
                    try:
                        _hf = opt_history_files.get(round_idx)
                        if _hf and Path(_hf).exists():
                            _hd = json.loads(Path(_hf).read_text(encoding="utf-8"))
                            _hd["paired_verdict"] = _verdict
                            Path(_hf).write_text(json.dumps(_hd, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
                    except Exception as _exc:
                        print(f"[base] Warning: could not record paired verdict: {_exc}", flush=True)
                    # Also hang it on the tree node. The round record above is
                    # per-round; _base_optimization_inventory walks the tree's
                    # parent links to list what the BASE carries, so this is the
                    # only path by which a mechanism's measured contribution can
                    # follow it forward once later rounds inherit it. The tree is
                    # serialized to optimization_tree.json and checkpointed, so it
                    # also survives --resume.
                    try:
                        _kn = getattr(ind, "code_path", None)
                        if _kn and _kn.stem in optimization_tree:
                            optimization_tree[_kn.stem]["paired_verdict"] = {
                                k: _verdict[k] for k in
                                ("rel_pct", "se_pct", "t", "dof", "reps", "resolved",
                                 "p_one_sided", "sigma_equiv", "sigma_ok", "method")
                                if k in _verdict
                            }
                    except Exception as _exc:
                        print(f"[base] Warning: could not attach paired verdict to "
                              f"the optimization tree: {_exc}", flush=True)
                else:
                    should_update_base = this_score >= base_score * (1.0 + args.base_margin)
                    _paired_note = None

            # ---- MCGS: record the candidate and back its result up ------------
            # This replaces the ratchet's one-way base mutation. The reward is the
            # PAIRED relative gain against the state we branched from, so the value
            # chain is `seed x prod(1 + verified gain)` -- one basis end to end.
            # Falling back to the blocked delta only when no verdict exists keeps
            # --base_max_reps 0 working, and is flagged in the log because that
            # number carries the drift the paired path removes.
            if _use_mcgs and _mcgs_sel is not None and _mcgs_parent_key:
                _cand_name = (Path(ind.code_path).stem
                              if (ind and getattr(ind, "code_path", None)) else None)
                if _cand_name:
                    _register(ind)
                    if _verdict is not None:
                        _rel = float(_verdict["rel_pct"])
                        _basis = "paired"
                    elif base_score > 0 and this_score > 0:
                        _rel = (this_score / base_score - 1.0) * 100.0
                        _basis = "blocked (drift-contaminated; no paired verdict)"
                    else:
                        _rel, _basis = 0.0, "unmeasurable"
                    _parent_node = graph.nodes.get(_mcgs_parent_key)
                    _parent_value = float(_parent_node.rep_value) if _parent_node else 1.0
                    _child_value = _parent_value * (1.0 + _rel / 100.0)
                    _failed = not runnable
                    # Mechanism set: the path's applied methods plus this round's,
                    # so 'mechanisms' keying can recognise a commuting reorder.
                    _mech = (strategy_json.get("method_name")
                             if isinstance(strategy_json, dict) else None)
                    _mechs = None
                    if args.mcgs_state_key == "mechanisms":
                        # Path-accumulated and order-preserving: graph.path_mechanisms
                        # walks the `via` edge labels root-to-leaf. Repeats are kept,
                        # so applying one method twice stays distinct from once.
                        _mechs = graph.path_mechanisms(_mcgs_sel.path) + (
                            [_mech] if _mech else [])
                    _ck = _state_key_for(ind, round_idx=round_idx, mechanisms=_mechs)
                    graph.observe(key=_ck, kernel_name=_cand_name,
                                  kernel_path=str(ind.code_path) if ind.code_path else None,
                                  value=_child_value, parent_key=_mcgs_parent_key,
                                  runnable=bool(runnable), mechanism=_mech,
                                  note=f"round {round_idx}, {_rel:+.2f}% {_basis}")
                    _r = reward_from_gain(_rel, scale=args.mcgs_reward_scale, failed=_failed)
                    _path = list(_mcgs_sel.path)
                    if _ck not in _path:
                        _path.append(_ck)
                    graph.backup(_path, _r, failed=_failed)
                    _merged = len(graph.nodes[_ck].members) > 1
                    print(f"[mcgs] Child state {_ck}: {_rel:+.2f}% ({_basis}) -> reward "
                          f"{_r:.3f}, value {_child_value:.4f}, N={graph.nodes[_ck].N}"
                          f"{'  [TRANSPOSITION: merged into an existing state]' if _merged else ''}",
                          flush=True)
                    _bn = graph.best()
                    if _bn is not None:
                        print(f"[mcgs] Best state so far: {_bn.key} at {_bn.rep_value:.4f} "
                              f"via {_bn.rep} (N={_bn.N}, Q={_bn.q(args.mcgs_lam):.3f})",
                              flush=True)

            # The ratchet's base MUTATION is MCGS's job now -- base_kernel is set at
            # selection time from the chosen state, and re-deciding it here would
            # overwrite that choice one round later and collapse the graph back to a
            # hill-climber. But `should_update_base` is deliberately left intact:
            # it is the accept EVIDENCE (paired, margin-and-significance gated) and
            # the best_kernel gate below reads it. Zeroing it would leave best
            # promoted only by the blocked-score branch, i.e. exactly the
            # drift-driven path that shipped a non-reproducing number before.
            _mcgs_base_frozen = bool(_use_mcgs and _mcgs_sel is not None)

            if should_update_base and not _mcgs_base_frozen:
                if base_score == float("-inf"):
                    print(f"[base] Setting initial base_kernel: {this_score:.4f}", flush=True)
                else:
                    gain = (this_score / base_score - 1.0) * 100.0 if base_score > 0 else float("nan")
                    _how = _paired_note or f"stored-score {gain:+.2f}%"
                    print(f"[base] Ratchet: base_kernel {base_score:.4f} -> {this_score:.4f} "
                          f"[{_how}, margin {args.base_margin * 100:.1f}%]; "
                          f"later rounds now optimize from this kernel", flush=True)
                base_score = this_score
                base_kernel = ind
                with open(test_kernel, "w") as f:
                    f.write(base_kernel.code)
            elif (args.structural_grace > 0 and structural_debt is None
                  and _structural_declared and base_score > 0 and this_score > 0
                  and not _mcgs_base_frozen):
                # Inert under MCGS by design. structural_debt is a one-slot,
                # N-round-grace hand-rolled version of "keep a worse node around
                # because it may lead somewhere" -- which is what the graph does
                # natively for every node, without a grace clock or a restore.
                # Declared structural rewrite, rejected by the ratchet. Adopt it
                # anyway and remember what it displaced: a rewrite is slower
                # until it is finished, so judging it on its first version is
                # judging it on the one round where it cannot win. The debt is
                # settled below -- either the rewrite overtakes what it replaced
                # within --structural_grace rounds, or the old base comes back.
                structural_debt = {
                    "kernel": base_kernel,
                    "score": base_score,
                    "rounds_left": int(args.structural_grace),
                    "declared_round": round_idx,
                }
                delta = (this_score / base_score - 1.0) * 100.0
                print(f"[base] Structural rewrite declared: adopting {this_score:.4f} over "
                      f"{base_score:.4f} ({delta:+.2f}%, below the "
                      f"{args.base_margin * 100:.1f}% margin) for {args.structural_grace} "
                      f"rounds. It must beat {base_score:.4f} within that window or the "
                      f"displaced kernel is restored.", flush=True)
                base_score = this_score
                base_kernel = ind
                with open(test_kernel, "w") as f:
                    f.write(base_kernel.code)
            elif base_score not in (float("-inf"), 0):
                delta = (this_score / base_score - 1.0) * 100.0 if base_score > 0 else float("nan")
                _how = _paired_note or f"stored-score {delta:+.2f}%"
                if _mcgs_base_frozen:
                    # Not "kept" -- the graph will re-pick next round, possibly this
                    # very candidate. Saying "keeping base_kernel" here would read as
                    # a rejection and the state was in fact recorded and is selectable.
                    print(f"[base] MCGS holds selection ({this_score:.4f}: {_how}); the "
                          f"candidate is a graph state and may be branched from later",
                          flush=True)
                else:
                    print(f"[base] Keeping base_kernel {base_score:.4f} ({this_score:.4f}: "
                          f"{_how}, below the {args.base_margin * 100:.1f}% margin)", flush=True)

            # ---- settle any outstanding structural-rewrite debt -------------
            if structural_debt is not None:
                _owed = float(structural_debt["score"])
                if base_score >= _owed * (1.0 + args.base_margin):
                    print(f"[base] Structural rewrite paid off: {base_score:.4f} now beats the "
                          f"{_owed:.4f} it displaced; grace cleared.", flush=True)
                    structural_debt = None
                else:
                    structural_debt["rounds_left"] -= 1
                    if structural_debt["rounds_left"] > 0:
                        print(f"[base] Structural rewrite on grace: {base_score:.4f} vs the "
                              f"{_owed:.4f} it owes, {structural_debt['rounds_left']} round(s) "
                              f"left.", flush=True)
                    else:
                        print(f"[base] Structural rewrite failed to reach {_owed:.4f} in "
                              f"{args.structural_grace} rounds (best it managed: "
                              f"{base_score:.4f}); restoring the displaced kernel.", flush=True)
                        base_kernel = structural_debt["kernel"]
                        base_score = _owed
                        if base_kernel is not None:
                            with open(test_kernel, "w") as f:
                                f.write(base_kernel.code)
                        structural_debt = None

            # best_kernel decides what the run REPORTS, so it needs the same
            # evidence the base ratchet demands -- it used to take any higher
            # `this_score` with a bare `>`. `this_score` comes from the blocked
            # bench in compare_and_bench (reference timed in one block, candidate
            # in the next), which carries session drift the paired path removes:
            # on 2026-08-04 round 22 that gap was +0.47% blocked against
            # -0.06% +/-0.18% paired, for a kernel byte-equivalent in behaviour.
            # The base correctly refused it; best took it anyway and shipped
            # 1.2095 to summary.json, a number that does not reproduce. Worse,
            # best_score is a running maximum over a noisy estimator, so noise
            # can only ever ratchet it UP -- the error never cancels.
            #
            # When a verdict exists it is authoritative, including when it says
            # the candidate is NOT better. Without one (no base yet, paired
            # measurement failed, or --base_max_reps 0) fall back to the old
            # comparison, so disabling the paired path restores prior behaviour
            # exactly rather than freezing best forever.
            if _verdict is not None:
                _best_ok = should_update_base or this_score > best_score * (1.0 + args.base_margin)
                _best_why = _paired_note or "paired"
            else:
                _best_ok = this_score > best_score
                _best_why = "stored-score (no paired verdict available)"
            if _best_ok and this_score > best_score:
                print(f"[best] Updating best_kernel: {this_score:.4f} vs {best_score:.4f} "
                      f"[{_best_why}]", flush=True)
                best_score = this_score
                best_kernel = ind
            elif this_score > best_score:
                print(f"[best] Keeping best_kernel {best_score:.4f} ({this_score:.4f} scores "
                      f"higher but {_best_why}); not separable from measurement drift",
                      flush=True)
            
            # Update optimization tree: update speedup if kernel already exists
            if ind and hasattr(ind, 'code_path') and ind.code_path:
                kernel_name = ind.code_path.stem
                if kernel_name in optimization_tree:
                    optimization_tree[kernel_name]["speedup"] = float(this_score)
                    optimization_tree[kernel_name]["runnable"] = runnable
                # If kernel doesn't exist in tree (shouldn't happen, but handle it)
                elif kernel_name not in optimization_tree:
                    # This might happen for compilation timeout repairs
                    parent_name = None
                    if hasattr(ind, 'code') and current_kernel and hasattr(current_kernel, 'code_path') and current_kernel.code_path:
                        parent_name = current_kernel.code_path.stem
                    optimization_tree[kernel_name] = {
                        "parent": parent_name,
                        "kernel_name": kernel_name,
                        "kernel_path": str(ind.code_path),
                        "speedup": float(this_score),
                        "runnable": runnable,
                        "ncu_passed": False,
                        "phase": "unknown",
                        "round": round_idx,
                        "strategy": None,
                        "method_matched": False,  # Unknown phase doesn't have optimization method matching
                        "timestamp": datetime.now().isoformat(),
                    }

        else:
            # on failure: keep last score and mark error
            scores.append(last_score_for_curve)
            err_flags.append(True)

        # ---- plateau tracking ------------------------------------------------
        # Gated by the SAME margin the base uses. A gain inside it is not
        # resolvable from cross-round GPU drift, so it must not read as progress
        # and reset the counter -- on vae_block_002 the final "improvement" was
        # +0.017%, which is noise, and treating it as real would have kept the
        # loop alive for another 7 rounds.
        if best_score_at_round_start > 0:
            _improved = best_score > best_score_at_round_start * (1.0 + args.base_margin)
        else:
            _improved = best_score > best_score_at_round_start
        rounds_since_improvement = 0 if _improved else rounds_since_improvement + 1
        _plateau_stop = bool(args.patience and rounds_since_improvement >= args.patience)

        # Written before the checkpoint so that a round total exists even if the
        # checkpoint write is what fails.
        _round_dt = time.perf_counter() - _round_t0
        _rounds_run += 1
        print(f"[{task_path.name}] Round {round_idx} took {_round_dt / 60:.1f} min "
              f"({_round_dt:.0f}s)", flush=True)
        run_timing.record("round_total", _round_dt, round_idx=round_idx,
                          detail=f"best={best_score:.4f}" if best_score != float("-inf")
                                 else "best=none")

        # Round finished (whatever its outcome) -- snapshot so a stop or a crash
        # after this point resumes from the NEXT round rather than replaying this one.
        _save_checkpoint(
            task_root, eval_dir,
            task_path=task_path,
            next_round=round_idx + 1,
            total_rounds=args.round,
            state={
                "base_kernel": base_kernel,
                "best_kernel": best_kernel,
                "current_kernel": current_kernel,
                "repair_chain_kernel": repair_chain_kernel,
                "base_score": base_score,
                "best_score": best_score,
                "optimization_tree": optimization_tree,
                "scores": scores,
                "err_flags": err_flags,
                "last_score_for_curve": last_score_for_curve,
                "rounds_since_improvement": rounds_since_improvement,
                "structural_debt": structural_debt,
                "stop_reason": "plateau" if _plateau_stop else None,
                "mcgs": graph.to_dict() if _use_mcgs else None,
                "opt_history_files": opt_history_files,
            },
        )

        if _plateau_stop:
            _best_txt = f"{best_score:.4f}" if best_score != float("-inf") else "none"
            print(f"[plateau] No gain above the {args.base_margin * 100:.1f}% margin for "
                  f"{rounds_since_improvement} consecutive rounds (best={_best_txt}). Stopping "
                  f"after round {round_idx} of {args.round}; the remaining "
                  f"{args.round - round_idx - 1} would very likely have cost wall clock without "
                  f"moving the score. Resume with --resume {batch_dir} --patience 0 to override.",
                  flush=True)
            break

    # plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    # Save optimization tree to JSON
    tree_path = task_root / "optimization_tree.json"
    tree_data = {
        "task": str(task_path),
        "base_kernel": base_kernel.code_path.stem if (base_kernel and hasattr(base_kernel, 'code_path') and base_kernel.code_path) else None,
        "base_score": float(base_score) if base_score != float("-inf") else None,
        "best_kernel": best_kernel.code_path.stem if (best_kernel and hasattr(best_kernel, 'code_path') and best_kernel.code_path) else None,
        "best_score": float(best_score) if best_score != float("-inf") else None,
        "kernels": optimization_tree,
        "timestamp": datetime.now().isoformat(),
    }
    tree_path.write_text(json.dumps(tree_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{task_path.name}] Optimization tree saved to: {tree_path}")

    usage_totals = _append_usage_totals(log_path)
    
    # Calculate method_matched statistics from optimization_tree
    method_matched_stats = {"total_opt_rounds": 0, "matched_count": 0, "unmatched_count": 0}
    for kernel_info in optimization_tree.values():
        phase = kernel_info.get("phase", "")
        if phase == "opt":
            method_matched_stats["total_opt_rounds"] += 1
            if kernel_info.get("method_matched", False):
                method_matched_stats["matched_count"] += 1
            else:
                method_matched_stats["unmatched_count"] += 1
    
    # Also count from opt_round_*.json files for accuracy
    if code_dir.exists():
        for kernel_dir in code_dir.iterdir():
            if not kernel_dir.is_dir():
                continue
            opt_history_dir = kernel_dir
            if opt_history_dir.exists():
                opt_files = sorted(opt_history_dir.glob("opt_round_*.json"))
                for opt_file in opt_files:
                    try:
                        opt_data = json.loads(opt_file.read_text(encoding="utf-8"))
                        if opt_data.get("optimization_strategy"):
                            # Count only once (avoid double counting with optimization_tree)
                            # We'll use optimization_tree as primary source, but verify with opt_round files
                            pass
                    except Exception:
                        continue

    # Closes the session. A timing.csv whose last row is process_exit ended cleanly;
    # one that just stops was killed, and the gap before the next resume is downtime,
    # not work.
    run_timing.event("process_exit", round_idx=-1,
                     detail=f"stopped={'yes' if _STOP_REQUESTED else 'no'} "
                            f"plateau={'yes' if _plateau_stop else 'no'} "
                            f"rounds_run={_rounds_run}")

    return {
        "task": str(task_path),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(getattr(best_kernel, "metrics", {}).get("runnable", False)) if best_kernel else False,
        "task_dir": str(task_root),
        "figure": str(fig_path),
        "input_tokens_sum": usage_totals["input_tokens"],
        "output_tokens_sum": usage_totals["output_tokens"],
        "total_tokens_sum": usage_totals["total_tokens"],
        "method_matched_stats": method_matched_stats,
    }


# --------------------- summary saving ------------------
def _save_global_summary(batch_dir: Path, summary: List[Dict[str, Any]], avg_speedup: float, accuracy: float, total_tokens_sum: float) -> None:
    """Save summary.json and summary.csv under the batch_dir."""
    batch_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "total_tokens_sum": total_tokens_sum,
        "num_tasks": len(summary),
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "best_score", "best_runnable", "task_dir", "figure"])
        for s in summary:
            writer.writerow([s["task"], f'{s["best_score"]:.6f}', int(
                bool(s["best_runnable"])), s["task_dir"], s["figure"]])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])
        writer.writerow(["total_tokens_sum", f"{int(total_tokens_sum)}"])

    print(f"[GLOBAL] Saved: {batch_dir/'summary.json'}")
    print(f"[GLOBAL] Saved: {csv_path}")


# --------------------------- main ----------------------
def main():
    args = _build_arg_parser().parse_args()
    _install_stop_handler()

    all_tasks = _collect_tasks(args.arch_py)

    # Apply filter from summary.json if specified
    if args.filter_from_summary:
        all_tasks = _filter_tasks_from_summary(all_tasks, args.filter_from_summary)
        if not all_tasks:
            print("[ERROR] No tasks found after filtering from summary.json. Exiting.")
            return

    # ---- Resume reuses the existing batch folder; a fresh run makes a new one ----
    if args.resume:
        batch_dir = Path(args.resume).resolve()
        if not batch_dir.is_dir():
            print(f"[ERROR] --resume path is not a directory: {batch_dir}")
            return
        print(f"[BATCH] Resuming into existing folder: {batch_dir}")
    else:
        # ---- Create ONE batch folder for this run ----
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = _build_run_tag(args.server_type, args.model_name)
        # batch name hints: single file uses file stem; directory uses 'batch'
        if args.arch_py.is_file():
            batch_name = f"{stamp}_{args.arch_py.stem}_{run_tag}"
        else:
            # include sampling info for traceability
            if args.filter_from_summary:
                pick_note = "filtered_from_summary"
            elif args.first_n and args.first_n > 0:
                pick_note = f"first{args.first_n}"
            else:
                pick_note = f"num{args.num_tasks}_seed{args.shuffle_seed}"
            batch_name = f"{stamp}_batch_{pick_note}_{run_tag}"
        batch_dir = (args.work_dir / batch_name).resolve()
        batch_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BATCH] Output folder: {batch_dir}")

    # single file → run once (still inside the same batch folder)
    if args.arch_py.is_file():
        res = _run_single_task(all_tasks[0], args, batch_dir=batch_dir)
        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        total_tokens_sum = res.get("total_tokens_sum", 0)
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        return

    # directory: first_n takes precedence; else optionally sample
    # Note: If filter_from_summary is used, we already have the filtered list,
    # so we can still apply first_n or num_tasks on the filtered list if needed
    if args.first_n and args.first_n > 0:
        # support starting from an arbitrary (1-based) index in the sorted list
        start_idx = max(0, (args.start_from or 1) - 1)
        end_idx = min(len(all_tasks), start_idx + args.first_n)
        if start_idx >= len(all_tasks):
            print(f"[Task Picker] start_from={args.start_from} exceeds number of tasks ({len(all_tasks)}); nothing to run.")
            picked = []
        else:
            picked = all_tasks[start_idx:end_idx]
            print(
                f"[Task Picker] Found {len(all_tasks)} tasks, "
                f"taking {len(picked)} tasks from sorted positions [{start_idx+1}..{end_idx}]."
            )
    elif args.filter_from_summary:
        # If filter_from_summary is used, by default use all filtered tasks
        # unless num_tasks is explicitly set to a value other than default (1)
        if args.num_tasks == 1:
            # Default num_tasks=1, but filter_from_summary means use all filtered tasks
            picked = all_tasks
            print(f"[Task Picker] Using all {len(picked)} filtered tasks from summary.json.")
        else:
            # User explicitly set num_tasks, sample from filtered tasks
            picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
            print(f"[Task Picker] Found {len(all_tasks)} filtered tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")
    else:
        # Normal sampling without filter
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        res = _run_single_task(task, args, batch_dir=batch_dir)
        summary.append(res)

    # global summary using each task's best kernel
    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        total_tokens_sum = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)
        print("\n===== SUMMARY =====")
        for s in summary:
            print(f"{s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s['figure']}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        # ---- save under the SAME batch folder ----
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
    else:
        print("No tasks were run.")


if __name__ == "__main__":
	main()
