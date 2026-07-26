# Local patches to vendored SOL-ExecBench

## Provenance

- Upstream: `github.com/NVIDIA/SOL-ExecBench`
- Vendored base commit: **`2d852a30914d4ef7f9fac92696e7fc8eea630f52`**
  (2026-05-28, "[Fix] Align local-eval deps with ext-prod; default skip-reference (#9)") —
  the tip of `src/` at vendoring time (~2026-07-14).
- The vendored tree currently runs from **sourceless bytecode**
  (`__pycache__/*.cpython-313.pyc` copied next to their packages as `.pyc`
  modules) after an accidental deletion of the `.py` sources.
  **Do not drop a `.py` back into the package tree unless it is the verified
  file** — a stray `.py` would shadow the working `.pyc`.
- Evidence trail: `run/vae_block_002/forensics_commit_match.md` (commit match)
  and `run/vae_block_002/reconstructed_timing.py` (verified reconstruction of
  the one patched file, kept outside the package tree on purpose).

26 of 27 modules are bytecode-identical to sources compiled from `2d852a3`
(CPython 3.13.12, recursive code-object comparison ignoring filenames/line
tables). Exactly **one** module carries a local patch:

## Patched files

### `src/sol_execbench/core/bench/timing.py`

**Intent.** Make CUPTI optional. `cupti-python` targets CUDA 13
(`libcupti.so.13`); on this CUDA 12.8 host the library is absent, and the
failure can surface either at import time or only when an activity kind is
first enabled (soname mismatch). The patch probes CUPTI once, caches the
result, and falls back to `torch.cuda.Event` timing, with an env-var
override:

| `SOLBENCH_TIMING` | Effective methodology |
|---|---|
| `cupti` | force CUPTI |
| `cuda_events` | force `torch.cuda.Event` timing |
| `auto` / unset / other | keep an explicit `cuda_events` request; otherwise `cupti` if the probe succeeds, else `cuda_events` |

The env value is `.strip().lower()`-normalized before matching.

**Semantic diff vs upstream `2d852a3`, function by function:**

1. Module imports: adds `import os` (between `import ctypes.util` and
   `import statistics`).

2. `from cupti import cupti` is wrapped:

   ```python
   try:
       from cupti import cupti

       _CUPTI_IMPORT_OK = True
   except Exception:
       cupti = None
       _CUPTI_IMPORT_OK = False
   ```

3. New module global, declared `global` at module scope (this is what makes
   the compiler emit `STORE_GLOBAL` at module level, confirmed against the
   bytecode) and annotated with a *string* annotation:

   ```python
   global _CUPTI_USABLE_CACHE
   _CUPTI_USABLE_CACHE: 'bool | None' = None
   ```

4. New function `_cupti_usable() -> bool` — cached probe. Returns the cached
   value if set; returns `False` (caching `False`) when the import failed or
   `torch.cuda.is_available()` is false; otherwise runs one
   `cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)` /
   `cupti.activity_disable(...)` cycle in a `try/except Exception`, caching
   `True` on success and `False` on failure.

5. New function `_resolve_timing_methodology(requested: str) -> str` —
   implements the table above via
   `os.environ.get("SOLBENCH_TIMING", "auto").strip().lower()`.

6. `time_runnable(...)`: exactly one inserted statement, immediately after
   the `ShiftingMemoryPoolAllocator` is constructed and before the
   `with torch.cuda.device(device):` block:

   ```python
   methodology = _resolve_timing_methodology(methodology)
   ```

   The signature and its default (`methodology="cupti"`) are unchanged.

Everything else in `timing.py` (`get_l2_cache_size`, `_summarize_statistics`,
`_get_empty_cache_for_benchmark`, `_clear_cache`, `clone_args`, `_demangle`,
`CuptiKernelInfo`, `bench_gpu_time_with_cupti`, `bench_time_with_cuda_events`,
and all nested code objects) is bytecode-identical to upstream `2d852a3`.

## Verification status of the reconstruction

`run/vae_block_002/reconstructed_timing.py` was recompiled under CPython
3.13.12 (`compile(src, ..., dont_inherit=True, optimize=0)`) and compared
against the vendored `timing.pyc` code object recursively
(driver: scratchpad `verify_reconstruction.py`):

- **Verified identical (exact match):** `co_code`, `co_consts` (recursively,
  including all nested code objects), `co_names`, `co_varnames`,
  `co_freevars`/`co_cellvars`, `co_flags`, arg counts, stack sizes — for the
  module and every nested code object.
- **Also verified identical:** full `co_positions()` — every instruction's
  line *and column* range matches the pyc. This pins down statement layout,
  indentation, and expression forms to the character column (e.g. the
  condition is `if not _CUPTI_IMPORT_OK or not torch.cuda.is_available():`,
  not a parenthesized `and` variant), and pins every comment/blank line's
  *position* (lines 29–43, 46, 51, 81–82, 521 carry no bytecode).
- **Approximate (not recoverable from bytecode):** the *text* of comments and
  the choice of blank vs comment on no-bytecode lines; quote style of
  ordinary string literals (single vs double have equal width; double quotes
  chosen to match upstream file style). The annotation `'bool | None'` is an
  exception: future-annotations stringification preserves source text, so
  its single quotes **are** verified. The pyc header records an original
  source size of 19,652 bytes vs 19,297 reconstructed — the ~355-byte gap is
  comment text that was longer in the original.

## Restoring a source tree

1. `git checkout 2d852a3 -- src/` from upstream for everything **except**
   `core/bench/timing.py`.
2. For `timing.py`, use `run/vae_block_002/reconstructed_timing.py`
   (bytecode-verified equivalent of the lost file).
3. If sources are restored, delete the sourceless `.pyc` files sitting next
   to the packages (keep `__pycache__/` as normal), or they will mask edits.
