#!/usr/bin/env python
"""One-off NVIDIA CompileIQ finishing pass on a frozen KernelMem kernel.

Why this exists
---------------
The optimisation loop owns a kernel's SOURCE. The knobs inside the compiler --
register allocation, instruction scheduling, the compiler's own unrolling
heuristics -- are not something an LLM can see, so rounds spent on unroll
sweeps, ``-maxrregcount`` and launch-bounds hints were guesswork. CUDA 13.3
exposes exactly those knobs through an Advanced Controls File (ACF) that
``nvcc --apply-controls`` forwards to its components, and NVIDIA's CompileIQ
runs an evolutionary search over them for ONE kernel. So the loop no longer
touches them (see the prompts and ``utils/acf.py``), and this tool tunes them
once, on the frozen winner, after the search.

It is a finishing pass and never part of a round, for two reasons NVIDIA
states plainly: an ACF is per-kernel and per-compiler-build, so it is stale the
moment the source changes; and "expect failures, compile hangs, numeric
instability", so every candidate must be built and checked for correctness.

Cost, measured on the RTX 5090 with problem 002's CUTLASS kernel: the nvcc
front end is ~10 s (shared across candidates: ninja skips ``main.o``), ptxas
<1 s, the harness bench 30-50 s. Call it one minute per evaluation, on one
GPU, serialised -- the default 10 generations x 15 pool is 150 evaluations,
2-3 hours. ``--baseline-only`` measures one evaluation and prints the estimate.

What it does
------------
1. preflight: compileiq importable, nvcc >= 13.3 with ``--apply-controls``,
   a Blackwell-or-later GPU, a kernel that builds through ``load_inline``.
2. baseline: one fresh-process harness bench of the plain kernel -- also the
   correctness check and the ptxas register/spill report.
3. search: CompileIQ's evolutionary search. Every candidate is a fresh
   process, a private ACF file, the harness's own bench WITH its tolerance
   check; any failure (build, hang, accuracy) scores INVALID.
4. gate: the best ACF is embedded into ``<stem>_acf.py`` (self-contained,
   version-guarded, with fallback -- ``scripts/package_solution.py`` takes it
   unchanged) and measured against the plain kernel with the interleaved
   paired verdict at ``--margin`` (default 1%, the loop's accept margin).
   The search's own "best" is a single noisy sample picked from 150, so the
   paired verdict, not the search score, decides whether anything ships.
5. report: ``<out>/report.json`` and a printed summary.

Usage
-----
    python -m utils.compileiq_finish tasks/vae_block_002.py \\
        run/vae_block_002/kernel_autotune_splitk.py --out run/vae_block_002/compileiq
    python -m utils.compileiq_finish ... --baseline-only     # one bench + cost estimate
    python -m utils.compileiq_finish ... --space ptxas       # the smaller ptxas-only space
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# CompileIQ's worker pool starts a multiprocessing manager the moment Search is
# constructed; its default 'forkserver' re-imports __main__ in a fresh process
# and loses everything this script set up. 'fork' inherits it. Safe here because
# this process never initialises CUDA -- every measurement is a child process.
os.environ.setdefault("CIQ_PROCESS_MODE", "fork")

from utils import acf as acf_mod  # noqa: E402  (no torch import in this process)
from utils import ext_naming  # noqa: E402  (stdlib-only at import)

REPO = Path(__file__).resolve().parent.parent
MIN_CUDA = (13, 3)      # first toolkit with --apply-controls
MIN_COMPUTE_CAP = 10.0  # nvcc's ACF support is documented for Blackwell and later


def _log(msg: str) -> None:
    print(f"[compileiq_finish] {msg}", flush=True)


def _geomean(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def candidate_ms(res: Optional[Dict[str, Any]]) -> Optional[float]:
    """The candidate's absolute time, geomean over the benchmarked shapes.

    Absolute time, never ``score``: ``score`` carries a separately measured
    reference in its denominator (``utils/paired_bench.py`` explains the
    failure that causes). ``per_shape`` is empty for single-shape tasks, so the
    primary measurement is the fallback.
    """
    if not res:
        return None
    ms = [s.get("test_ms") for s in (res.get("per_shape") or []) if s.get("test_ms")]
    if ms:
        return _geomean([float(x) for x in ms])
    t = (res.get("test_latency_ms") or {}).get("avg")
    return float(t) if t else None


def _compute_cap(device: int) -> Optional[float]:
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(device), "--query-gpu=compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


def with_ext_suffix(src: str, suffix: str) -> str:
    """Rename the kernel's load_inline extension so two variants never share a build dir.

    The paired verdict imports the plain and the tuned kernel into ONE process,
    and torch keys its build directory -- and its rebuild decision -- on the
    extension name. Two variants under one name with different flags rebuild
    on every alternation (30 s each); ``utils/verify_chain.py`` hit the same
    thing. A distinct name costs one extra build and then both stay cached.
    """
    i = src.find("load_inline(")
    if i < 0:
        return src
    m = re.search(r'name\s*=\s*(["\'])([^"\']+)\1', src[i:i + 800])
    if not m:
        return src
    old = m.group(2)
    return src.replace(f"{m.group(1)}{old}{m.group(1)}", f"{m.group(1)}{old}{suffix}{m.group(1)}")


# --------------------------------------------------------------- evaluation
class Evaluator:
    """Fresh-process harness benches of one kernel, with or without an ACF."""

    def __init__(self, reference: Path, kernel: Path, out: Path, *, device: int,
                 warmup: int, repeat: int, tol: float, timeout: float):
        self.reference, self.kernel, self.out = reference, kernel, out
        self.device, self.warmup, self.repeat, self.tol, self.timeout = device, warmup, repeat, tol, timeout
        # One shared extension dir: each candidate changes only the cuda.o
        # command line (its ACF path is unique), so ninja rebuilds cuda.o and
        # relinks, and skips the 10 s main.o. A fresh process always starts at
        # extension version 0, so nothing accumulates.
        self.ext_dir = out / "ext"
        self.evals_dir = out / "evals"
        self.evals_csv = out / "evals.csv"
        for d in (self.ext_dir, self.evals_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.evals_csv.exists():
            self.evals_csv.write_text("tag,status,ms,seconds,acf\n")

    def _clear_stale_locks(self) -> None:
        # A SIGKILLed child leaves torch's FileBaton behind and the next build
        # would spin on it until the compile alarm; the holder is dead for sure.
        for lock in self.ext_dir.glob("*/lock"):
            try:
                lock.unlink()
            except OSError:
                pass

    def run(self, acf: Optional[Path], tag: str, *, strict: bool = True) -> Dict[str, Any]:
        stamp = f"{time.time_ns()}_{tag}"
        dump = self.evals_dir / f"{stamp}.json"
        logf = self.evals_dir / f"{stamp}.log"
        env = dict(os.environ)
        env["TORCH_EXTENSIONS_DIR"] = str(self.ext_dir)
        env[acf_mod.PTXAS_VERBOSE_ENV] = "1"
        # Keep the one-shared-folder design above: content-hashed names would
        # give each candidate ACF its own folder (a cold main.o every time, and
        # ~2 MB x 150 on a disk at 94%). Every Evaluator run is one import in a
        # fresh process under a private root, so the collisions content naming
        # exists to prevent cannot happen here. The paired verdict child below
        # alternates two kernels in ONE process and keeps content naming.
        env[ext_naming.MODE_ENV] = ext_naming.MODE_OFF
        env.pop(acf_mod.ACF_ENV, None)
        if acf is not None:
            env[acf_mod.ACF_ENV] = str(acf)
            env[acf_mod.ACF_STRICT_ENV] = "1" if strict else "0"
        cmd = [sys.executable, "-m", "utils.compile_and_run", str(self.reference), str(self.kernel),
               "--device", str(self.device), "--warmup", str(self.warmup), "--repeat", str(self.repeat),
               "--tol", str(self.tol), "--dump", str(dump), "--no-time-ref"]
        t0 = time.perf_counter()
        with open(logf, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=str(REPO), start_new_session=True)
            try:
                rc: Optional[int] = p.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)  # ninja and nvcc included
                p.wait()
                rc = None
                self._clear_stale_locks()
        secs = time.perf_counter() - t0
        res = json.loads(dump.read_text()) if (rc == 0 and dump.exists()) else None
        ms = candidate_ms(res)
        if rc is None:
            status = "timeout"
        elif res is None or ms is None:
            status = "fail"
        elif acf is not None and not (res.get("acf") or {}).get("applied"):
            status = "fallback"   # built plain after the ACF failed: not the ACF's number
            ms = None
        else:
            status = "ok"
        with open(self.evals_csv, "a") as f:
            csv.writer(f).writerow([tag, status, f"{ms:.6f}" if ms else "", f"{secs:.1f}", acf or ""])
        tail = ""
        if status != "ok":
            lines = logf.read_text(errors="replace").splitlines()
            tail = "\n".join(lines[-25:])
        _log(f"{tag}: {status}" + (f" {ms:.4f} ms" if ms else "") + f" ({secs:.0f} s)")
        return {"tag": tag, "status": status, "ms": ms, "seconds": secs, "acf": str(acf) if acf else None,
                "result": res, "log_tail": tail}


class Objective:
    """What CompileIQ calls: a hex-encoded ACF in, the candidate's ms out."""

    def __init__(self, ev: Evaluator, acf_dir: Path):
        self.ev, self.acf_dir = ev, acf_dir
        acf_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, config_hex: str) -> float:
        from compileiq.types import INVALID_SCORE
        try:
            from compileiq.types import BASELINE_CONFIG
        except ImportError:  # pragma: no cover
            BASELINE_CONFIG = ""
        if not config_hex or config_hex == BASELINE_CONFIG:
            r = self.ev.run(None, "ciq_baseline", strict=False)
        else:
            path = self.acf_dir / f"{time.time_ns()}.bin"
            path.write_bytes(bytes.fromhex(config_hex))
            r = self.ev.run(path, "cand", strict=True)
        return float(r["ms"]) if r["status"] == "ok" else INVALID_SCORE


# ------------------------------------------------------------------ search
def run_search(ev: Evaluator, out: Path, *, space: str, generations: int, pool: int):
    from compileiq.ciq import Search
    from compileiq.search_spaces.compilers import NvccSearchSpace, PtxasSearchSpace
    from compileiq.types import SearchConfiguration

    ver = acf_mod.nvcc_version() or ""
    major_minor = ".".join(ver.split(".")[:2])
    provider = (NvccSearchSpace if space == "nvcc" else PtxasSearchSpace)(version=major_minor)
    search = Search(
        objective_function=Objective(ev, out / "acf"),
        search_space=provider,
        search_config=SearchConfiguration(problem_type="min", generations=generations, pool_size=pool),
        dump_results=str(out / "search_results.csv"),
        cache_folder=str(out / "ciq_cache"),
    )
    # One worker: one GPU, and the bench must never overlap another bench.
    results = search.start(num_workers=1, task_timeout=ev.timeout + 120)
    try:
        results.save(str(out / "search_results.csv"))
    except Exception as exc:  # the CSV is a convenience, not the result
        _log(f"could not save search CSV: {exc}")
    best = results.get_best_result()
    if isinstance(best, list):  # single objective; a Pareto list would be a bug
        best = best[0]
    return results, best


# ------------------------------------------------------------------- verdict
def _verdict_child(args: argparse.Namespace) -> int:
    """Hidden subcommand: the paired verdict in its own process (it initialises CUDA)."""
    from utils.paired_bench import adaptive_paired_verdict
    v = adaptive_paired_verdict(Path(args.reference), Path(args.kernel), Path(args.verdict_cand),
                                device=args.device, margin=args.margin, tol=args.verdict_tol,
                                log=lambda m: print(m, flush=True))
    Path(args.verdict_dump).write_text(json.dumps(v))
    return 0


def paired_verdict(reference: Path, base: Path, cand: Path, out: Path, *, device: int,
                   margin: float, tol: float, timeout: float) -> Optional[Dict[str, Any]]:
    dump = out / "verdict.json"
    cmd = [sys.executable, "-m", "utils.compileiq_finish", str(reference), str(base), "--out", str(out),
           "--_verdict", str(cand), "--_verdict-dump", str(dump), "--_verdict-tol", str(tol),
           "--device", str(device), "--margin", str(margin)]
    env = dict(os.environ)
    env["TORCH_EXTENSIONS_DIR"] = str(out / "ext")
    env.pop(acf_mod.ACF_ENV, None)
    with open(out / "verdict.log", "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO),
                             start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return None
    return json.loads(dump.read_text()) if dump.exists() else None


# ---------------------------------------------------------------------- main
def preflight(kernel: Path, device: int) -> List[str]:
    problems: List[str] = []
    try:
        import compileiq  # noqa: F401
    except ImportError:
        problems.append("compileiq is not installed: pip install compileiq")
    ver = acf_mod.nvcc_version()
    if not ver:
        problems.append("nvcc not found on PATH")
    else:
        mm = tuple(int(x) for x in ver.split(".")[:2])
        if mm < MIN_CUDA:
            problems.append(f"nvcc {ver} is older than {MIN_CUDA[0]}.{MIN_CUDA[1]}, which introduced --apply-controls")
    if ver and not acf_mod.nvcc_supports_controls():
        problems.append("this nvcc does not list --apply-controls in its help")
    cc = _compute_cap(device)
    if cc is not None and cc < MIN_COMPUTE_CAP:
        problems.append(f"GPU compute capability {cc} is pre-Blackwell; nvcc applies ACFs on Blackwell and later only")
    src = kernel.read_text(encoding="utf-8")
    if acf_mod.count_load_inline_calls(src) < 1:
        problems.append("the kernel has no load_inline() call, so there is no nvcc build for an ACF to apply to")
    if acf_mod.PREAMBLE_MARKER in src:
        problems.append("the kernel already carries an ACF preamble; pass the plain kernel")
    return problems


def _ptxas_line(res: Optional[Dict[str, Any]]) -> str:
    px = (res or {}).get("ptxas") or {}
    if not px.get("n_kernels"):
        return "ptxas: no report"
    return (f"ptxas: {px['n_kernels']} kernel(s), max {px.get('max_registers')} regs/thread, "
            f"{px.get('spill_bytes', 0)} B spilled")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", type=Path, help="task / reference .py (defines Model, get_inputs)")
    p.add_argument("kernel", type=Path, help="the frozen kernel .py to tune (defines ModelNew)")
    p.add_argument("--out", type=Path, required=True, help="output directory (created)")
    p.add_argument("--space", choices=["nvcc", "ptxas"], default="nvcc",
                   help="CompileIQ search space; nvcc (default) is the broader one")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--pool", type=int, default=15, help="candidates per generation (CompileIQ needs > 5)")
    p.add_argument("--margin", type=float, default=0.01,
                   help="paired-verdict margin the tuned kernel must beat to ship (0.01 = 1%%)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-2,
                   help="harness accuracy tolerance for every candidate (the loop's default; "
                        "the SOL workloads allow ~3e-3 and TF32 needs it)")
    p.add_argument("--eval-timeout", type=float, default=900.0,
                   help="seconds per evaluation before the child (and its nvcc) is killed")
    p.add_argument("--baseline-only", action="store_true",
                   help="bench the plain kernel once, print the ptxas report and the cost estimate, stop")
    # hidden: the paired verdict runs in a child process through these
    p.add_argument("--_verdict", dest="verdict_cand", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_verdict-dump", dest="verdict_dump", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_verdict-tol", dest="verdict_tol", type=float, default=1e-2, help=argparse.SUPPRESS)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verdict_cand is not None:
        return _verdict_child(args)
    if args.pool <= 5:
        _log("--pool must be at least 6 (CompileIQ's minimum)")
        return 2

    reference, kernel, out = args.reference.resolve(), args.kernel.resolve(), args.out.resolve()
    problems = preflight(kernel, args.device)
    if problems:
        for m in problems:
            _log("preflight: " + m)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    nvcc_ver = acf_mod.nvcc_version() or ""
    _log(f"nvcc {nvcc_ver}, space={args.space}, kernel={kernel}")

    ev = Evaluator(reference, kernel, out, device=args.device, warmup=args.warmup,
                   repeat=args.repeat, tol=args.tol, timeout=args.eval_timeout)
    report: Dict[str, Any] = {
        "kernel": str(kernel), "reference": str(reference), "nvcc": nvcc_ver, "space": args.space,
        "generations": args.generations, "pool": args.pool, "margin_pct": args.margin * 100.0,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # -- 1. baseline ---------------------------------------------------------
    base = ev.run(None, "baseline", strict=False)
    if base["status"] != "ok":
        _log(f"the plain kernel does not pass the harness ({base['status']}); nothing to tune:\n{base['log_tail']}")
        report["baseline"] = {k: v for k, v in base.items() if k != "result"}
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return 1
    n_evals = args.generations * args.pool + 1
    report["baseline"] = {"ms": base["ms"], "seconds": base["seconds"],
                          "ptxas": (base["result"] or {}).get("ptxas")}
    _log(f"baseline {base['ms']:.4f} ms; {_ptxas_line(base['result'])}")
    _log(f"estimated search cost: {n_evals} evaluations x {base['seconds']:.0f} s "
         f"= {n_evals * base['seconds'] / 3600:.1f} h on this GPU")
    if args.baseline_only:
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return 0

    # -- 2. search ----------------------------------------------------------
    t0 = time.perf_counter()
    results, best = run_search(ev, out, space=args.space, generations=args.generations, pool=args.pool)
    best_ms = float(best.get("score_1", best.get("score")))
    best_hex = best["params"] if isinstance(best["params"], str) else json.dumps(best["params"])
    best_acf = out / "best.acf.bin"
    best_acf.write_bytes(bytes.fromhex(best_hex))
    gain_pct = (base["ms"] / best_ms - 1.0) * 100.0
    df = results.get_results()
    n_rows = int(len(df))
    report["search"] = {"seconds": time.perf_counter() - t0, "evaluations": n_rows,
                        "best_ms_single_sample": best_ms, "gain_pct_vs_baseline_single_sample": gain_pct,
                        "generation": best.get("generation"), "best_acf": str(best_acf)}
    _log(f"search done: {n_rows} evaluations in {(time.perf_counter() - t0) / 60:.0f} min; "
         f"best single sample {best_ms:.4f} ms ({gain_pct:+.2f}% vs baseline); ACF -> {best_acf}")

    if gain_pct < args.margin * 100.0:
        report["verdict"] = None
        report["shipped"] = False
        report["conclusion"] = (f"the best candidate's single-sample gain ({gain_pct:+.2f}%) is under the "
                                f"{args.margin * 100:.1f}% margin; nothing worth a paired verdict, nothing shipped")
        (out / "report.json").write_text(json.dumps(report, indent=2))
        _log(report["conclusion"])
        return 0

    # -- 3. embed + paired gate ---------------------------------------------
    src = kernel.read_text(encoding="utf-8")
    tuned = out / f"{kernel.stem}_acf.py"
    tuned.write_text(acf_mod.embed_acf(with_ext_suffix(src, "_acf"), best_acf.read_bytes(), nvcc=nvcc_ver),
                     encoding="utf-8")
    _log(f"tuned kernel -> {tuned}; running the interleaved paired verdict at {args.margin * 100:.1f}%")
    verdict = paired_verdict(reference, kernel, tuned, out, device=args.device, margin=args.margin,
                             tol=args.tol, timeout=args.eval_timeout * 12)
    report["verdict"] = verdict
    report["tuned_kernel"] = str(tuned)
    if verdict is None:
        report["shipped"] = False
        report["conclusion"] = "paired verdict did not complete (see verdict.log); nothing shipped"
    else:
        ok = bool(verdict.get("beats_margin")) and bool(verdict.get("resolved", True))
        report["shipped"] = ok
        report["conclusion"] = (
            f"paired: tuned kernel {verdict['rel_pct']:+.2f}% +/- {verdict['se_pct']:.2f}% "
            f"(t={verdict['t']:.1f}, {verdict['reps']} reps) vs the plain kernel; "
            + ("beats the margin -> ship " + tuned.name
               if ok else "does not beat the margin -> keep the plain kernel"))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    _log(report["conclusion"])
    if report["shipped"]:
        _log(f"package it with: python scripts/package_solution.py {tuned} <out_dir> --name-suffix acf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
