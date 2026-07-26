# SOL-ExecBench vendored .pyc forensics — commit match

Date: 2026-07-18
Vendored tree: `/home/elek/KernelMem/third_party/SOL-ExecBench/src/sol_execbench` (sourceless; only `__pycache__/*.cpython-313.pyc` survive)
Upstream clone: `/tmp/claude-1000/-home-elek-KernelMem/f79120b6-191f-442b-8076-7c775261c9db/scratchpad/upstream` (github.com/NVIDIA/SOL-ExecBench, full history)
Interpreter used for recompilation: CPython 3.13.12 (magic number verified against every pyc header).

## Method

For each of the 27 vendored pycs and each candidate commit touching `src/` before
2026-07-15 (`2d852a3`, `8d2237d`, `a8ef1ce`, `534eff0`):

1. Extracted the upstream `.py` via `git show <commit>:src/<path>`.
2. Compiled it with `compile(src, path, "exec", dont_inherit=True, optimize=0)` under 3.13.12.
3. Loaded the vendored code object (`marshal.loads(pyc[16:])`, magic checked).
4. Compared code objects recursively: `co_code`, `co_consts` (recursing into nested
   code objects, tuples, frozensets), `co_names`, `co_varnames`, `co_freevars`,
   `co_cellvars`, `co_name`, `co_qualname`, arg counts, `co_flags`.
   Ignored: `co_filename`, line tables / `co_firstlineno`.

Comparison driver: `/tmp/claude-1000/-home-elek-KernelMem/f79120b6-191f-442b-8076-7c775261c9db/scratchpad/compare_pycs.py`
(raw matrix in `compare_out.json` next to it).

## (a) Best-matching commit

```
2d852a30914d4ef7f9fac92696e7fc8eea630f52
2026-05-28 12:41:54 -0700
[Fix] Align local-eval deps with ext-prod; default skip-reference (#9)
```

Match counts per commit (out of 27 modules):

| commit  | date       | matches |
|---------|------------|---------|
| 2d852a3 | 2026-05-28 | **26/27** |
| 8d2237d | 2026-04-15 | 24/27 |
| a8ef1ce | 2026-03-23 | 22/27 |
| 534eff0 | 2026-03-18 | 21/27 |

`2d852a3` was the tip of `src/` at vendoring time (~2026-07-14); the next commit
touching `src/` is `a9fa080` (2026-07-15, "Add v1.1 timing methodology (#18)"), one
day after vendoring.

Corroboration from pyc headers (flags=0, timestamp-based): all 26 matching modules
carry source mtime **2026-07-14 13:40:33** (the vendoring checkout); `timing.py`
carries **2026-07-14 13:41:05**, i.e. it was edited ~32 s after checkout. Its
recorded source size was 19652 bytes.

## (b) Modules matching exactly at 2d852a3 (26)

All of `sol_execbench` except `core/bench/timing.py`:

- `__init__`, `sol_score`
- `cli/__init__`, `cli/main`
- `core/__init__`, `core/utils`
- `core/bench/__init__`, `clock_lock`, `correctness`, `io`, `reward_hack`, `utils`
- `core/bench/config/__init__`, `benchmark_config`, `device_config`
- `core/data/__init__`, `base_model`, `definition`, `dtypes`, `json_utils`,
  `shapes`, `solution`, `trace`, `workload`
- `driver/__init__`, `driver/problem_packager`

(Upstream `driver/templates/build_ext.py` and `driver/templates/eval_driver.py`
have no pyc — they are runtime templates, never imported; expected.)

## (c) Modules matching NOWHERE — locally patched (1)

**`sol_execbench/core/bench/timing.py`** — does not match any of the 4 candidate
commits, and additionally was tested against **every** commit in full history that
ever touched the file (`a9fa080`, `8d2237d`, `534eff0`): no match. It is a genuine
local patch on top of the 2d852a3 version (not a backport of upstream v1.1).

## (d) Dis-level summary of the timing.py local patch (vs 2d852a3)

Every function/class in the module is byte-identical to upstream 2d852a3 except
the module body and `time_runnable`; two functions exist only in the pyc.

1. **`import os` added** at module level (extra module `co_names` entry `os`).

2. **`cupti` import wrapped in try/except** setting a flag
   (module names gained `_CUPTI_IMPORT_OK`, `Exception`):
   `_CUPTI_IMPORT_OK = True` on success, `False` on failure.

3. **New module-level annotated global** `_CUPTI_USABLE_CACHE: bool | None`
   (annotation string `'bool | None'`, `__annotations__` in module names).

4. **New function `_cupti_usable()`** (0 args) — docstring:
   > Return True iff CUPTI can actually trace on this host (cached probe).
   > The soname-mismatch failure only surfaces when an activity kind is enabled,
   > so probe an enable/disable cycle once and cache the result.

   Names referenced: `_CUPTI_USABLE_CACHE`, `_CUPTI_IMPORT_OK`,
   `torch.cuda.is_available`, `cupti.activity_enable`,
   `cupti.ActivityKind.CONCURRENT_KERNEL`, `cupti.activity_disable`, `Exception`.

5. **New function `_resolve_timing_methodology(requested)`** — docstring:
   > Pick the effective timing methodology, honoring SOLBENCH_TIMING.

   Reads env var **`SOLBENCH_TIMING`** via `os.environ.get(...).strip().lower()`;
   consts include `'auto'`, the tuple `('cupti', 'cuda_events')`, `'cuda_events'`,
   `'cupti'`; falls back based on `_cupti_usable()`.

6. **`time_runnable` modified** (co_code 372 vs 350 bytes). Dis-diff shows exactly
   one semantic insertion right after the allocator is constructed:

   ```
   LOAD_GLOBAL  _resolve_timing_methodology
   LOAD_FAST    methodology
   CALL 1
   STORE_FAST   methodology
   ```

   i.e. `methodology = _resolve_timing_methodology(methodology)`. All remaining
   instruction differences are name-index shifts caused by the extra global.

Unchanged vs upstream (bytecode-identical): `get_l2_cache_size`,
`_summarize_statistics`, `_get_empty_cache_for_benchmark`, `_clear_cache`,
`clone_args`, `_demangle`, `CuptiKernelInfo`, `bench_gpu_time_with_cupti`,
`bench_time_with_cuda_events`, and all their nested code objects.

This matches the known local intent: force/auto-fallback CUPTI → cuda_events
timing (CUPTI soname mismatch on CUDA 12.8), controlled by `SOLBENCH_TIMING`
(`auto`/`cupti`/`cuda_events`).

## Bottom line

Restore all sources except `core/bench/timing.py` verbatim from upstream commit
`2d852a30914d4ef7f9fac92696e7fc8eea630f52`. `timing.py` must be reconstructed as
2d852a3 plus the local patch described in (d); its bytecode (and the surviving
pyc, which remains importable) is the authoritative reference for that patch.
