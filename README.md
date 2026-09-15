## Project Overview: KernelMem

KernelMem is an **automatic CUDA kernel generation and optimization system based on PyTorch model code, enhanced with a "long–short term memory" mechanism**.  
The core idea is: starting from PyTorch forward code, the system uses an LLM to iteratively generate candidate CUDA kernels, and combines historical optimization experience, performance/correctness feedback, and expert knowledge about kernel optimization to form a "memory loop", continuously evolving faster kernels. The long-term memory component incorporates general knowledge and best practices for kernel optimization, enabling the system to leverage proven optimization strategies across different tasks.

The main entry point of the project is the `main()` function in `main_memory_latest.py` (triggered when the script is run directly).

---

## Key Features

- **Automatic migration from PyTorch operators / models to CUDA kernels**
  - Automatically reads operator / network definitions from PyTorch task scripts.
  - Builds LLM prompts according to the task and asks the model to generate corresponding CUDA kernels.

- **Multi-round self-evolution with “memory”**
  - For each kernel across rounds, the system records:
    - Correctness results (whether it runs, whether it passes numerical checks)
    - Performance metrics (speedup, NVIDIA Nsight Compute / Nsight Systems metrics, etc.)
    - Applied optimization strategies, failure reasons, repair history
  - These are written into `code/`, `evaluation/`, `profile/`, etc., and then fed back as short term memory to guide future kernel generation and repair.

- **Automatic benchmarking and error repair**
  - Uses `utils/compile_and_run.py` to compile and benchmark generated kernels:
    - Compares numerical errors against the reference PyTorch implementation (`tol`).
    - Measures average forward latency and computes **speedup = ref_latency / test_latency**.
  - For compilation errors / runtime errors / accuracy failures:
    - Builds “memory-aware” error analysis and repair prompts via `prompts/judger_repair_memory.py` and `prompts/error_memory.py`.
    - Asks the LLM to generate more reliable kernel versions based on historical error logs and repair records.

- **NCU & NSYS profiling–driven optimization**
  - Invokes NVIDIA Nsight Compute (`ncu`) via `profiling/ncu.py` to obtain fine-grained performance metrics:
    - Memory efficiency, SM utilization, launch/occupancy, bottleneck stages, etc.
  - Invokes Nsight Systems (`nsys`) via `profiling/nsys.py` to measure kernel launch counts and runtime behavior.
  - These profiling results are converted into optimization suggestions by `prompts/judger_optimization_memory_latest.py` / `prompts/optimization_memory_latest.py`, then used to drive new kernel generations.

- **Interruptible, resumable runs**
  - `Ctrl-C` / `SIGTERM` stops at the next round boundary instead of killing the process mid-loop, so
    `figures/`, `optimization_tree.json` and `summary.json` are still written.
  - A `checkpoint.json` is saved every round, so `--resume <batch_folder>` continues from the last
    completed round after a stop, a crash, or a `kill -9`.
  - See “Stopping and resuming a run” under Quick Start.

---

## Repository layout

```
main_memory_latest.py   entry point: the generate / repair / optimize / profile loop
run_lineages.py         coordinator that runs several seed lineages as parallel main_memory_latest.py processes
agents/                 LLM backends; query_server.py is the one interface the loop talks to
prompts/                prompt builders and judges; few_shot/ examples, hardware/ GPU specs
profiling/              Nsight wrappers: ncu.py, nsys.py, the two .ncu-cfg metric sets, and
                        bench_ref_inputs.py, the driver script that ncu/nsys execute
utils/                  library code (compile_and_run, kernel_io, clock_lock, mcts*, ...) and the
                        `python -m utils.<tool>` command-line tools built on it (noise_*, ...)
scripts/                standalone scripts run by path: mechanism prior, timing report, solution
                        packager, clock-lock installer
tests/                  test suite (see "Running the tests")
docs/                   reports on problem 002
tasks/                  task files the search runs on (tasks/vae_block_002.py)
KernelBench/            the KernelBench reference tasks, level1-4
memorybank/             long-term memory: the bottleneck/headroom rule table
priors/                 fitted mechanism priors and the per-card clock presets
solbench_problems/      SOL-ExecBench problem definitions; solbench_bridge/ turns them into tasks
                        (that package exists in this checkout only as Python 3.13 bytecode)
third_party/            vendored SOL-ExecBench (+ PATCHES.md); cutlass/ and kernel-design-agents/
                        are gitignored -- .gitignore says how to fetch them
run/                    run outputs, one folder per batch (see "Outputs and Visualization")
```

- **`main_memory_latest.py`**: main entry of the project
  - Parses CLI arguments (task selection, GPU, LLM settings, number of rounds, etc.).
  - Calls the LLM to generate / repair / optimize kernels.
  - Orchestrates benchmarking, NCU/NSYS profiling, visualization, and summary.
  - Everything it needs per run is written under the run's own folder, including its
    temp files (`scratch/`); the repository root stays clean while a run is in progress.

- **`profiling/`**: Nsight integration
  - `ncu.py`: runs Nsight Compute over the bench driver (`profile_bench`), parses the CSV
    (`load_ncu_metrics`) and renders it for the prompt (`metrics_to_prompt`). Reads the two
    `.ncu-cfg` files beside it.
  - `nsys.py`: the same for Nsight Systems, giving per-kernel launch counts.
  - `bench_ref_inputs.py`: the script both profilers execute. It is copied into each run's
    `scratch/` next to `ref.py` and `test_kernel.py` and loads them from its own directory.

- **`KernelBench/`**: PyTorch reference tasks
  - `level1`, `level2`: various basic operators and small subnetworks.
  - `level3`: representative deep learning models (ResNet, VGG, LSTM, Transformer, etc.).
  - `level4`: 20 further tasks.

- **`prompts/`**: prompt design and “memory mechanism”
  - `generate_custom_cuda_memory.py`: seed prompt for the first-round kernel generation.
  - `optimization_memory_latest.py`: optimization prompts that fuse historical kernels with profiling results.
  - `judger_*_memory*.py`: judge and analysis modules for optimization strategy, compilation timeouts, runtime errors, etc., which then produce repair/optimization suggestions.
  - `few_shot/`: few-shot examples for the LLM.

- **`memorybank/`**:
  - Stores prior knowledge about hardware bottlenecks and kernel structures.
  - These act as “long-term memory” and are injected into prompts to guide better optimization choices.

- **`utils/`**:
  - `compile_and_run.py`: compile, run, compare accuracy, and measure performance.
  - `kernel_io.py`: extract code blocks from LLM replies, save them as Python/CUDA files, and read/write metrics.
  - `individual.py`: `KernelIndividual`, the record the loop keeps per generated kernel.
  - `clock_lock.py`, `gpu_lock.py`, `device_state.py`: measurement hygiene (see 1b below).
  - `mcts*.py`, `pathmemory.py`, `rank_backtest.py`: the Monte Carlo search and its tooling.
  - `noise_*.py`, `paired_bench.py`: noise-floor measurement (see 1c below).
  - `acf.py`, `compileiq_finish.py`: the compiler-knob hand-off to NVIDIA CompileIQ (see 6 below).

- **`agents/query_server.py`**:
  - Unified interface for talking to actual LLM backends (OpenAI, local vLLM/sglang, etc.).

---

## Environment Requirements

It is recommended to run the project on **Linux + NVIDIA GPU** (on Windows you need to prepare the CUDA toolchain and Nsight tools yourself).  
Typical dependencies (for reference; adjust versions to your environment):

- Python 3.9+
- PyTorch (with GPU support)
- CUDA Toolkit and matching drivers
- NVIDIA Nsight Compute (`ncu`) and Nsight Systems (`nsys`)
- Python packages:
  - `matplotlib`
  - `pandas`, `numpy` (for profiling CSV processing if needed)
  - SDK for your LLM service (e.g. `openai` or a custom HTTP client)

Using a virtualenv or Conda environment is strongly recommended.

---

## Quick Start

### 1. Install dependencies

In the project root, create a virtual environment and install required packages, for example:

```bash
conda create -n kernelmem python=3.10 -y
conda activate kernelmem

# Install dependencies as needed (example)
pip install torch matplotlib pandas numpy
# If using OpenAI models, also install: openai
```

Make sure `ncu` and `nsys` are available in your shell.

### 1b. Allow the GPU clock to be pinned (once per machine)

**Every run pins the GPU clock before it measures anything, and refuses to start
if it cannot.** An unpinned clock is chosen by the driver from temperature, power
and duty cycle — none of which the harness controls or records — so the same
kernel timed twice is timed on two effectively different machines, and no score
from such a run is comparable with any other. Pinning is root-only, so grant it
once:

```bash
sudo bash scripts/install_clock_lock_sudoers.sh   # installs a restricted wrapper + sudoers rule
python -m utils.clock_lock --status               # check: target, current clock, privilege
```

The frequency is chosen **per device**: a value measured on this machine for this
exact card, else a built-in preset for the model, else a class-based fraction of
that card's own ceiling. An unfamiliar card is measured (~45 s, once, cached in
`priors/clock_presets.json`) rather than guessed at:

```bash
python -m utils.clock_lock --calibrate            # measure what this card holds under load
```

On the RTX 5090 the preset is **2407 MHz core / 13801 MHz memory**. That is well
below the ~2763 MHz the card sustains, and deliberately so: this repo scores
against `T_SOL = max(FLOPs / 104.8e12, bytes / 1792e9)`, and the 104.8 TFLOPS
constant is `170 SM × 256 FLOP/clk × 2.41 GHz` — a clock baked into a constant.
Running faster than 2.41 GHz produces times a `T_SOL` assuming 2.41 GHz cannot
explain, which is how a kernel once scored 1.036, i.e. faster than light. The
~13% of wall-clock this costs buys numbers that mean what the report says.

Useful knobs:

| | |
|---|---|
| `--gpu_clock_mhz 2610` | pin a different frequency for this run |
| `KERNELMEM_CLOCK_KEEP=1` | leave the clock pinned between back-to-back runs |
| `KERNELMEM_CLOCK_AUTOCAL=0` | never auto-measure an unfamiliar card |
| `--no_clock_lock` | run unpinned **on purpose**; artifacts are stamped `locked: false` |

Every benchmark result carries a `clock` field recording the frequency it was
taken at, so a trace can be audited on its own.

### 1c. Verify the noise floor

Two checks answer "can this harness resolve a 1% difference?", and they answer
different halves of it. Run them on a kernel with a **fixed** config — a kernel
that autotunes per instance (`self._choice` in `kernel_autotune_splitk.py`)
re-rolls its own configuration every process, and that spread is the kernel's,
not the harness's:

```bash
# 1. Spread: re-measure an unchanged kernel across fresh processes.
python -m utils.noise_verify --kernel run/vae_block_002/kernels/nhwc_eager.py \
  --processes 12 --calls 3 --band 1.0

# 2. False positives: run the real decision rule on two identical kernels.
python -m utils.noise_null_verdict --ref tasks/vae_block_002.py \
  --kernel run/vae_block_002/kernels/nhwc_eager.py \
  --trials 15 --margin 0.01 --out run/noise/null.jsonl
```

Both tools write under `run/noise/` by default (`noise_verify` picks a stamped name unless
`--out` is given), so nothing is left in the repository root.

`noise_verify` reports the **±2σ band** — the interval a re-measurement of an
unchanged kernel lands in ~95% of the time, so any "improvement" smaller than it
is indistinguishable from measuring the same kernel twice. It watches for other
processes on the GPU *during* each measurement and discards any sample that
shared the card, which on a multi-session box is otherwise reported as harness
noise. Measured 2026-08-14 on the RTX 5090, unlocked clocks: **±2σ = 0.46% on
score**, reproduced exactly across two independent runs, with 0/15 false accepts
at a 1% margin.

### 2. Run a single task

The most basic usage is to specify a PyTorch task script as `arch_py`:

```bash
python main_memory_latest.py KernelBench/level1/001_xxx.py \
  --gpu A100-80GB \
  --server_type openai \
  --server_address localhost \
  --server_port 8000 \
  --model_name gpt-5.1-chat \
  --round 10 \
  --work_dir run \
  --device 0
```

Key arguments:

- **`arch_py`**: path to a PyTorch task script, or to a directory containing multiple tasks.
- **`--gpu`**: GPU name used in prompts (does not change the actual device, only informs the LLM of hardware specs).
- **`--server_type` / `--server_address` / `--server_port` / `--model_name`**: LLM backend configuration.
- **`--round`**: total number of rounds per task (including seed generation, repair, and optimization).
- **`--device`**: CUDA device ID.
- **`--warmup` / `--repeat` / `--tol`**: warmup iterations, benchmark repetitions, and error tolerance.
- **`--resume`**: path to an existing batch folder to continue instead of starting a new run (see “Stopping and resuming”).
- **`--rollout_model` / `--rollout_effort`**: the model that writes each child kernel (the MCTS
  rollout). Default `claude-opus-5` at `high` since 2026-09-14 (it was `claude-sonnet-5` from
  2026-08-12, a cost decision; no matched A/B between the two exists). The judge, problem-identify
  and repair calls always use `--model_name`.
- **`--gate` / `--no_gate` / `--gate_refusals`** (default on, 5): the **harness gate**. When a
  tool-mode writing call (seed, optimization, repair) tries to end its turn, the harness's own
  correctness check runs on the kernel in its final message — every scored shape, the
  uninitialised-memory and device-state-leak gates, no timing — and a FAIL is handed back into the
  *same* agent session as a rejection, with the harness error verbatim, so the agent fixes it with
  its files and builds intact. After `--gate_refusals` rejections the kernel goes through to the
  official bench as it is. Implemented as a Claude Agent SDK Stop hook in `agents/query_server.py`;
  each rejection costs one agent turn and one gate run (~30 s cold, ~3 s once the kernel is built).
  Gate activity is logged as `[gate]` lines, `bench:gate` rows in `timing.csv` (these run inside
  the enclosing `llm:<call_type>` row, so they are part of it, not extra time), and a
  `gate=... checks=N blocks=N` note in `calls.csv`. The cap is kept below the CLI's own
  consecutive-block cap (`CLAUDE_CODE_STOP_HOOK_BLOCK_CAP`, default 8), past which the CLI would
  end the call with the rejected kernel. It reports PASS/FAIL only: a kernel slower than
  its parent still passes, and a kernel that passes the gate can still fail the official bench
  (allocation-dependent uninitialised reads, OOM at full `--repeat`, timeouts).

### 3. Batch tasks and filtering

- Randomly sample tasks from a directory:

```bash
python main_memory_latest.py KernelBench/level3 \
  --num_tasks 5 --shuffle_seed 42
```

- Use `summary.json` from a previous run to only re-run tasks whose best kernel is still non-runnable:

```bash
python main_memory_latest.py KernelBench/level3 \
  --filter_from_summary path/to/previous/summary.json
```

### 4. Stopping and resuming a run

Runs are long (each round makes LLM calls, compiles, benchmarks, and profiles), so they can be
stopped and picked up again without losing work.

**Stop gracefully** — press `Ctrl-C`, or send a signal to the process:

```bash
kill <pid>          # SIGTERM; Ctrl-C / SIGINT behaves the same
```

You will see:

```
[stop] Signal 15 received. Finishing the current round, then writing
       artifacts and a resumable checkpoint. Signal again to abort now.
```

The signal usually arrives in the middle of an LLM call or a benchmark, so it is only *acted on
between rounds*: the in-flight round runs to completion, then the loop exits normally and the
post-loop writer still produces `figures/`, `optimization_tree.json` and `summary.json`.
**This matters** — those three files are written only after the round loop finishes, so a hard
`kill -9` mid-loop loses all of them even though every per-round artifact survived. Send a second
signal if you need to abort immediately instead of waiting for the round to end.

**Resume** — point `--resume` at the batch folder the run was writing to:

```bash
python main_memory_latest.py tasks/vae_block_002.py \
  --resume run/20260729_122008_vae_block_002_claude_claude-opus-5 \
  --round 12 --gpu "RTX 5090"
```

This reuses that folder instead of creating a new timestamped one, and continues each task from
its `checkpoint.json`. Restored state includes the best/base/current kernels, the optimization
tree, the per-round score curve, and the repair-chain position, so the run continues as if it had
never stopped. `usage.csv` is not part of the checkpoint — it simply keeps accumulating in the same
task folder, so a resumed run appends to the existing token log (and a fresh `TOTAL` row is added
each time the run finishes).

Notes:

- A checkpoint is written at **every** round boundary, not only on a clean stop, so a crash or a
  `kill -9` still resumes from the last completed round — only the in-flight round is lost.
- The checkpoint is written to a temp file and renamed, so an interruption during the write leaves
  the previous checkpoint intact rather than a truncated one.
- To **extend** a finished run, resume it with a larger `--round`. Resuming with `--round` at or
  below the completed count runs no new rounds and simply rewrites the artifacts.
- `--resume` expects the *batch* folder (`run/<stamp>_<task>_<tag>/`), not the per-task folder
  inside it. It works for batch runs too: each task resumes from its own checkpoint.
- The checkpoint stores *paths* to kernels and metrics rather than copies, so it stays small — but
  it is therefore tied to its run folder. Moving or renaming the folder invalidates it, and any
  kernel file that has been deleted is reported and dropped rather than failing the resume.

### 5. Running the tests

Everything under `tests/` is collected by plain `pytest` from the repository root
(`pyproject.toml` restricts collection to that directory and puts the root on the import
path). About half the files are pytest-style; the rest are self-checking scripts that print
`ok` / `FAIL` per assertion (most run their checks at import time, so `pytest` executes them
while collecting but cannot report them as tests). Run those directly:

```bash
pytest                              # the pytest-style files
python -m tests.test_mcts_quick     # a script-style one (each file's docstring names its command)
```

One needs a GPU (`test_uninit_memory_gate` compiles two CUDA extensions); `test_torch_ext_cache`
and `test_acf` need torch importable but no GPU; the rest run anywhere.

### 6. CompileIQ finishing pass (compiler knobs are not the LLM's job)

The loop owns a kernel's **source**: tile and warp shape, split-K, pipeline stages, fusion,
layout, vector width. The knobs **inside the compiler** -- register allocation, instruction
scheduling, the compiler's own unrolling heuristics -- are not something an LLM can see, so
rounds spent on `#pragma unroll` sweeps, `-maxrregcount` and launch-bounds hints were guesswork.
Since 2026-09-09 the prompts tell the LLM not to spend rounds on them, `Reduce_Unrolling` is out
of the machine-check action space, and those knobs are tuned once, on the frozen winner, by
[NVIDIA CompileIQ](https://github.com/NVIDIA/CompileIQ): an evolutionary search over the hidden
nvcc/ptxas controls that ships with CUDA 13.3, whose output is an *Advanced Controls File* (ACF)
applied with `nvcc --apply-controls`.

Why it is a finishing pass and never part of a round: NVIDIA states an ACF is per-kernel and
per-compiler-build (stale the moment the source changes) and that failures, compile hangs and
numeric instability are to be expected. Every candidate is therefore built and benchmarked by
the harness itself, tolerance check included, in a fresh process.

```bash
pip install compileiq                                   # once; needs nvcc >= 13.3 and a Blackwell+ GPU
python -m utils.compileiq_finish tasks/vae_block_002.py run/vae_block_002/kernel_autotune_splitk.py \
    --out run/vae_block_002/compileiq --baseline-only   # one bench, ptxas report, cost estimate
python -m utils.compileiq_finish tasks/vae_block_002.py run/vae_block_002/kernel_autotune_splitk.py \
    --out run/vae_block_002/compileiq                   # the search (~1 min/evaluation, 150 by default)
```

The search's own "best" is one noisy sample picked from 150, so it is not what decides. The best
ACF is embedded into `<kernel>_acf.py` -- self-contained, applied only when the local nvcc is the
exact build it was tuned with, and falling back to the plain build if the controlled build fails
-- and that file is measured against the plain kernel with the interleaved paired verdict at the
loop's 1% accept margin (`--margin`). Only a kernel that beats it is worth shipping;
`scripts/package_solution.py` takes the embedded file unchanged. `report.json`, `evals.csv` and
`search_results.csv` in `--out` record everything.

Two smaller pieces of the same hand-off run inside every round, at no extra cost:

* candidate builds get `-Xptxas -v`, and the bench result carries `ptxas` (registers and spill
  bytes per kernel). A spilling kernel is printed as a hand-off, not fed to the next round.
  `KERNELMEM_PTXAS_VERBOSE=0` turns it off. The report exists only when the build actually ran:
  a kernel served from torch's extension cache carries `n_kernels: 0`, which the tools print as
  "no report". (The first bench after upgrading rebuilds any kernel cached under the old flags once.)
* `KERNELMEM_ACF=<file>` applies an ACF to the candidate build of any bench, with fallback unless
  `KERNELMEM_ACF_STRICT=1`; `KERNELMEM_ACF_DISABLE=1` makes an embedded kernel build plain. The
  bench result records `acf.applied` / `acf.fallback`.

Expect little on a kernel that already sits at tensor-core peak (problem 002's CUTLASS conv:
0-2%); the hand-off earns its keep on hand-written SIMT kernels, where the LLM used to burn rounds
on exactly these knobs.

---

## Outputs and Visualization


Example structure for a single task:

- `code/`: all kernels generated for this task (Python/CUDA), possibly with optimization/repair history JSON.
- `evaluation/`:
  - `llm_io/`: all prompts and raw LLM replies for each round.
  - Per-round metrics JSON: whether it is runnable, error type, speedup, etc.
- `figures/`:
  - `taskname_score.png`: speedup curve across rounds, with runnable/non-runnable points distinguished.
- `profile/`:
  - `*_ncu*.csv`: Nsight Compute metrics.
  - `*_nsys*.nsys-rep` / `*_nsys*.csv`: Nsight Systems traces and stats.
- `optimization_tree.json`:
  - A “genealogy” of all kernels for the task, with parent–child relationships, speedups, NCU status, and whether an optimization method was matched.
- `scratch/`:
  - Per-run temp files: `ref.py` (a copy of the task), `test_kernel.py` (the kernel being
    profiled), `rejected_kernel.py` (the previous round's not-adopted kernel, profiled for the
    judge), `bench_ref_inputs.py` (the profiling driver) and the raw `ncu_temp.csv` /
    `ncu_rejected.csv` / `nsys_temp.*` / `nsys_full.csv` output. Gitignored; anything worth
    keeping is copied into `profile/` under the kernel's name.
- `usage.csv`:
  - Token usage for all LLM calls, with a total row appended at the end.
- `checkpoint.json`:
  - Resume state written at every round boundary: the next round to run, the best/base/current
    kernels, the optimization tree, the score curve, and the repair-chain position. Consumed by
    `--resume` (see “Stopping and resuming a run”). Safe to delete if you want a resume to start over.

For each batch directory, you will also get:

- `summary.json` / `summary.csv`: cross-task summary including average speedup, accuracy, and total tokens.

---

## Long–Short Term Memory Mechanism (Conceptual)

- **Short-term memory (local context)**:
  - Recently generated kernel snippets in the current run, recent error logs, and profiling results.
  - Constructed via helpers such as `_build_history_block` into Markdown code blocks, which are directly embedded into optimization prompts.
  - Historical artifacts such as `optimization_tree.json` and per-round `opt_round_*.json` / `repair_round_*.json`.

- **Long-term memory (cross-round / cross-task experience)**:
  - Prior knowledge stored under `memorybank/` (hardware bottlenecks, common kernel structures, feasible optimization strategies).


When generating, repairing, or optimizing kernels, the LLM consumes this memory as additional context so that it can:

- Avoid repeating the same compilation/runtime mistakes.
- Reuse optimization strategies that have worked in the past.
- Make more targeted design choices for specific hardware and operator patterns.

---

## Notes and Caveats

- The project frequently compiles and runs GPU kernels. Make sure your machine has sufficient GPU memory and proper timeout/monitoring to avoid hangs caused by buggy kernels.
- NCU / NSYS profiling can be time-consuming, especially for large-model tasks in `KernelBench/level3`. It is recommended to first debug the pipeline on small tasks with fewer rounds.
- Because a run can take hours, prefer stopping it with `Ctrl-C` / `kill` rather than `kill -9`: the
  graceful path still writes the figure, optimization tree, and summary, and leaves a checkpoint you
  can `--resume`. A `kill -9` skips the post-loop writer, though the checkpoint from the last
  completed round still allows a resume.


If you want to deploy or extend this project in your environment (e.g. connecting to your own LLM backend, adding new kernel templates / task sets), start by reading and modifying `main_memory_latest.py` and files under `prompts/`.
