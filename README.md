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
  - Invokes NVIDIA Nsight Compute (`ncu`) via `run_ncu_memory.py` to obtain fine-grained performance metrics:
    - Memory efficiency, SM utilization, launch/occupancy, bottleneck stages, etc.
  - Invokes Nsight Systems (`nsys`) via `run_nsys.py` to measure kernel launch counts and runtime behavior.
  - These profiling results are converted into optimization suggestions by `prompts/judger_optimization_memory_latest.py` / `prompts/optimization_memory_latest.py`, then used to drive new kernel generations.

- **Interruptible, resumable runs**
  - `Ctrl-C` / `SIGTERM` stops at the next round boundary instead of killing the process mid-loop, so
    `figures/`, `optimization_tree.json` and `summary.json` are still written.
  - A `checkpoint.json` is saved every round, so `--resume <batch_folder>` continues from the last
    completed round after a stop, a crash, or a `kill -9`.
  - See “Stopping and resuming a run” under Quick Start.

---

## Code Structure

- **`main_memory_latest.py`**: main entry of the project
  - Parses CLI arguments (task selection, GPU, LLM settings, number of rounds, etc.).
  - Calls the LLM to generate / repair / optimize kernels.
  - Orchestrates benchmarking, NCU/NSYS profiling, visualization, and summary.

- **`KernelBench/`**: PyTorch reference tasks
  - `level1`, `level2`: various basic operators and small subnetworks.
  - `level3`: representative deep learning models (ResNet, VGG, LSTM, Transformer, etc.).

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
