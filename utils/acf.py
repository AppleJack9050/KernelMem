"""Compiler-knob hand-off to NVIDIA CompileIQ (Advanced Controls Files).

Why this exists
---------------
KernelMem's LLM loop owns the SOURCE of a kernel: tile and warp shape, split-K,
pipeline stages, fusion, layout, vector width. The knobs that live INSIDE the
compiler -- register allocation, instruction scheduling, the compiler's own
unrolling heuristics -- are not something an LLM can see or reason about; it
guesses at them through ``#pragma unroll`` sweeps, ``-maxrregcount`` and
``__launch_bounds__`` minimum-block hints, and those rounds are pure noise.
CUDA 13.3 exposes exactly those knobs through an *Advanced Controls File* (ACF)
that ``nvcc --apply-controls <file>`` forwards to its components, and NVIDIA's
CompileIQ searches that space for one kernel. So: the loop stops tuning them,
and ``python -m utils.compileiq_finish`` tunes them once, on the frozen winner.

Three constraints shape everything here (all from NVIDIA's own docs):

* An ACF is **per-kernel and per-compiler-build**. Source changes every round
  in this loop, so an ACF tuned in round N is stale in round N+1 -- which is
  why it is a post-search finishing pass and never part of a round.
* "Expect failures, compile hangs, numeric instability." Every ACF build is
  therefore retried without controls by default (``strict=False``), the
  finishing pass runs the harness's tolerance check on the tuned build, and the
  shipped preamble applies the controls only when the local nvcc is the exact
  build the ACF was tuned with.
* The harness never touches nvcc flags -- the generated kernel file owns its
  ``extra_cuda_cflags``. So the only injection point that covers every kernel
  without rewriting LLM output is ``torch.utils.cpp_extension.load_inline``
  itself, patched for the duration of one kernel import.

What is here
------------
``patched_extension_builds``  the injection: ``--apply-controls`` and/or
                              ``-Xptxas -v`` appended to every load_inline /
                              load call inside the block, with fallback.
``parse_ptxas_verbose``       registers / spills per kernel out of the build
                              log, so the loop can SEE compiler-level trouble
                              (and hand it off) instead of asking the LLM to
                              fix it blind.
``embed_acf``                 a self-contained preamble that ships the ACF
                              inside the kernel .py (base64), version-guarded,
                              with fallback -- so ``scripts/package_solution.py``
                              works unchanged on the tuned kernel.

Environment variables (read by ``utils.compile_and_run``):

``KERNELMEM_ACF``            path of an ACF to apply to the candidate build.
``KERNELMEM_ACF_STRICT``     ``1``: a build that fails with the ACF is an
                             error (the finishing pass's objective wants that);
                             default: rebuild without controls and record it.
``KERNELMEM_PTXAS_VERBOSE``  ``0`` disables ``-Xptxas -v`` on candidate builds
                             (on by default; it changes no code, only the log).
``KERNELMEM_ACF_DISABLE``    ``1``: an embedded preamble builds plain. A/B switch.
"""
from __future__ import annotations

import base64
import contextlib
import functools
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ACF_ENV = "KERNELMEM_ACF"
ACF_STRICT_ENV = "KERNELMEM_ACF_STRICT"
PTXAS_VERBOSE_ENV = "KERNELMEM_PTXAS_VERBOSE"
ACF_DISABLE_ENV = "KERNELMEM_ACF_DISABLE"

CONTROLS_FLAG = "--apply-controls"
PTXAS_VERBOSE_FLAGS = ["-Xptxas", "-v"]

# Positional index of ``extra_cuda_cflags`` in each patched builder. Kernels pass
# it by keyword in practice, but a positional call must not silently lose the
# injection.
_CUDA_CFLAGS_POS = {"load_inline": 5, "load": 3}


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off")


# --------------------------------------------------------------- environment
def active_acf() -> Optional[Path]:
    """The ACF the environment asks the candidate build to apply, or None."""
    p = os.environ.get(ACF_ENV, "").strip()
    return Path(p) if p else None


def strict_enabled() -> bool:
    return _truthy(os.environ.get(ACF_STRICT_ENV))


def ptxas_verbose_enabled() -> bool:
    return not os.environ.get(PTXAS_VERBOSE_ENV, "1").strip().lower() in ("0", "false", "no", "off")


@functools.lru_cache(maxsize=None)
def nvcc_version(nvcc: str = "nvcc") -> Optional[str]:
    """The exact nvcc build, e.g. ``'13.3.33'`` -- the key an ACF is valid for.

    Exact, not major.minor: NVIDIA states an ACF is reproducible only while
    "the underlying compiler is matching", and 13.3 and 13.4 ship different
    search spaces.
    """
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"\bV(\d+\.\d+\.\d+)", out)
    return m.group(1) if m else None


@functools.lru_cache(maxsize=None)
def nvcc_supports_controls(nvcc: str = "nvcc") -> bool:
    """True when the local nvcc documents ``--apply-controls`` (CUDA >= 13.3)."""
    try:
        out = subprocess.run([nvcc, "--help"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return False
    return CONTROLS_FLAG in out


# ---------------------------------------------------------------- injection
def _inject(args: tuple, kwargs: dict, pos: int, extra: List[str],
            force_verbose: bool) -> tuple[list, dict]:
    """Copy of (args, kwargs) with *extra* appended to ``extra_cuda_cflags``."""
    a = list(args)
    kw = dict(kwargs)
    if len(a) > pos:
        a[pos] = list(a[pos] or []) + list(extra)
    else:
        kw["extra_cuda_cflags"] = list(kw.get("extra_cuda_cflags") or []) + list(extra)
    if force_verbose:
        # torch swallows ninja's output (and with it ``ptxas info``) unless the
        # build is verbose; the harness captures fd 1/2 during the import, so
        # verbose costs nothing and is the only way the stats reach the log.
        kw["verbose"] = True
    return a, kw


@contextlib.contextmanager
def patched_extension_builds(*, acf: Optional[Path] = None, strict: bool = False,
                             ptxas_verbose: bool = False) -> Iterator[Dict[str, Any]]:
    """Patch ``torch.utils.cpp_extension.load_inline``/``load`` for the block.

    Every build started inside the block gets ``-Xptxas -v`` (if
    *ptxas_verbose*) and ``--apply-controls <acf>`` (if *acf*) appended to its
    ``extra_cuda_cflags``. A build that raises WITH the ACF is retried without
    it unless *strict* -- NVIDIA says to expect ACF failures, and a kernel that
    builds plain must never be scored as a compile error because of a control
    file. The yielded record says what happened::

        {"acf": str|None, "applied": bool, "fallback": bool, "error": str|None,
         "ptxas_verbose": bool}

    ``applied`` is True only when a build with the ACF succeeded. ``fallback``
    is True when the ACF build failed and the plain rebuild was used instead.
    """
    record: Dict[str, Any] = {
        "acf": str(acf) if acf else None,
        "applied": False,
        "fallback": False,
        "error": None,
        "ptxas_verbose": bool(ptxas_verbose),
    }
    if acf is None and not ptxas_verbose:
        yield record
        return
    if acf is not None and not Path(acf).is_file():
        raise FileNotFoundError(f"{ACF_ENV} points at a missing file: {acf}")

    import torch.utils.cpp_extension as ext

    originals = {name: getattr(ext, name) for name in _CUDA_CFLAGS_POS}

    def make(name: str):
        orig = originals[name]
        pos = _CUDA_CFLAGS_POS[name]

        def call(args, kwargs, with_acf: bool):
            extra: List[str] = list(PTXAS_VERBOSE_FLAGS) if ptxas_verbose else []
            if with_acf:
                extra += [CONTROLS_FLAG, str(acf)]
            a, kw = _inject(args, kwargs, pos, extra, force_verbose=ptxas_verbose)
            return orig(*a, **kw)

        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            if acf is None:
                return call(args, kwargs, False)
            try:
                mod = call(args, kwargs, True)
            except Exception as exc:  # build, link or import failure with controls
                record["error"] = f"{type(exc).__name__}: {str(exc)[:4000]}"
                if strict:
                    raise
                record["fallback"] = True
                print(f"[acf] build with {acf} failed ({type(exc).__name__}); "
                      f"rebuilding without controls", flush=True)
                return call(args, kwargs, False)
            record["applied"] = True
            return mod

        return wrapped

    for name in originals:
        setattr(ext, name, make(name))
    try:
        yield record
    finally:
        for name, fn in originals.items():
            setattr(ext, name, fn)


# ------------------------------------------------------------- ptxas -v log
_RE_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '(sm_\d+[a-z]?)'")
_RE_FUNC = re.compile(r"ptxas info\s*:\s*Function properties for (\S+)")
_RE_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_RE_USED = re.compile(r"ptxas info\s*:\s*Used (\d+) registers"
                      r"(?:, used (\d+) barriers)?(?:, (\d+) bytes smem)?"
                      r"(?:, (\d+) bytes cmem\[0\])?")


def parse_ptxas_verbose(log: str) -> Dict[str, Any]:
    """Registers and spills per entry kernel out of a ``-Xptxas -v`` build log.

    Returns ``{"kernels": [...], "n_kernels", "spill_bytes", "max_registers",
    "spilling": [names]}``. Device functions that are not entries get
    "Function properties" lines too but no "Used N registers" line; they are
    dropped, since the LLM-facing question is "which KERNEL spills".
    """
    kernels: List[Dict[str, Any]] = []
    arch_of: Dict[str, str] = {}
    cur: Optional[Dict[str, Any]] = None
    for line in (log or "").splitlines():
        m = _RE_ENTRY.search(line)
        if m:
            arch_of[m.group(1)] = m.group(2)
            continue
        m = _RE_FUNC.search(line)
        if m:
            cur = {"name": m.group(1), "arch": arch_of.get(m.group(1)),
                   "stack_frame": 0, "spill_stores": 0, "spill_loads": 0,
                   "registers": None, "barriers": None, "smem": None}
            continue
        if cur is None:
            continue
        m = _RE_SPILL.search(line)
        if m:
            cur["stack_frame"], cur["spill_stores"], cur["spill_loads"] = (int(x) for x in m.groups())
            continue
        m = _RE_USED.search(line)
        if m:
            cur["registers"] = int(m.group(1))
            cur["barriers"] = int(m.group(2)) if m.group(2) else None
            cur["smem"] = int(m.group(3)) if m.group(3) else None
            kernels.append(cur)
            cur = None
    spill = sum(k["spill_stores"] + k["spill_loads"] for k in kernels)
    regs = [k["registers"] for k in kernels if k["registers"] is not None]
    return {
        "kernels": kernels,
        "n_kernels": len(kernels),
        "spill_bytes": spill,
        "max_registers": max(regs) if regs else None,
        "spilling": [k["name"] for k in kernels if k["spill_stores"] or k["spill_loads"]],
    }


def short_kernel_name(mangled: str, width: int = 48) -> str:
    """A readable handle for a (possibly huge) mangled kernel name."""
    m = re.match(r"_ZN?(\d+)([A-Za-z_]\w*)", mangled)
    head = m.group(2)[: int(m.group(1))] if m else mangled
    return head if len(head) <= width else head[: width - 1] + "…"


def build_summary(record: Optional[Dict[str, Any]], log: str) -> Dict[str, Any]:
    """What ``compare_and_bench`` reports about the candidate's build.

    The ptxas report is only as good as the log: a build served from torch's
    extension cache prints nothing, so it comes back with ``n_kernels: 0``.
    Inside the loop every candidate is a fresh build; tools re-benching a
    cached kernel should read 0 kernels as "no report", not "no spills".
    """
    return {
        "acf": dict(record) if record else None,
        "ptxas": parse_ptxas_verbose(log) if (record or {}).get("ptxas_verbose") else None,
    }


# ------------------------------------------------------------- embedding
_PREAMBLE = '''\
# ===== KernelMem / CompileIQ Advanced Controls File (ACF) preamble =====
# Tuned by `python -m utils.compileiq_finish` against nvcc @@NVCC@@. Compiler
# controls are per-kernel and per-compiler-build (NVIDIA's words: "expect
# failures"), so this preamble applies them ONLY when the local nvcc is exactly
# that build, and a build that fails with them is retried without them. The
# file therefore never builds worse than the plain kernel below it.
# Set KERNELMEM_ACF_DISABLE=1 to force the plain build (A/B switch).
import atexit as _km_atexit
import base64 as _km_b64
import os as _km_os
import re as _km_re
import subprocess as _km_sp
import tempfile as _km_tmp
import torch.utils.cpp_extension as _km_ext

_KM_ACF_B64 = "@@ACF_B64@@"
_KM_ACF_NVCC = "@@NVCC@@"
_KM_ACF_CALLS = @@CALLS@@   # load_inline calls in the kernel below; restore after the last


def _km_nvcc_version():
    try:
        out = _km_sp.run(["nvcc", "--version"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = _km_re.search(r"\\bV(\\d+\\.\\d+\\.\\d+)", out)
    return m.group(1) if m else None


def _km_install():
    orig = _km_ext.load_inline
    state = {"calls": 0, "path": None}
    disabled = _km_os.environ.get("KERNELMEM_ACF_DISABLE", "").strip().lower() not in ("", "0", "false", "no", "off")
    local = _km_nvcc_version()
    if disabled or local != _KM_ACF_NVCC:
        why = "disabled by KERNELMEM_ACF_DISABLE" if disabled else "nvcc %s is not %s" % (local, _KM_ACF_NVCC)
        print("[acf] compiler controls skipped (%s); building plain" % why, flush=True)
        return

    def wrapped(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] >= _KM_ACF_CALLS:
            _km_ext.load_inline = orig
        if state["path"] is None:
            fd, path = _km_tmp.mkstemp(prefix="kernelmem_", suffix=".acf.bin")
            with _km_os.fdopen(fd, "wb") as f:
                f.write(_km_b64.b64decode(_KM_ACF_B64))
            state["path"] = path
            _km_atexit.register(lambda: _km_os.path.exists(path) and _km_os.unlink(path))
        a = list(args)
        kw = dict(kwargs)
        flags = ["--apply-controls", state["path"]]
        if len(a) > 5:
            a[5] = list(a[5] or []) + flags
        else:
            kw["extra_cuda_cflags"] = list(kw.get("extra_cuda_cflags") or []) + flags
        try:
            return orig(*a, **kw)
        except Exception as exc:
            print("[acf] build with controls failed (%s); rebuilding without" % type(exc).__name__, flush=True)
            return orig(*args, **kwargs)

    _km_ext.load_inline = wrapped


_km_install()
# ===== end of ACF preamble; the tuned kernel follows unchanged =====
'''

PREAMBLE_MARKER = "KernelMem / CompileIQ Advanced Controls File (ACF) preamble"


def count_load_inline_calls(src: str) -> int:
    return len(re.findall(r"\bload_inline\s*\(", src))


def embed_acf(kernel_src: str, acf_bytes: bytes, *, nvcc: str,
              calls: Optional[int] = None) -> str:
    """The kernel source with a self-contained ACF preamble in front of it.

    *nvcc* is the exact build the ACF was tuned with (``nvcc_version()``);
    *calls* the number of ``load_inline`` calls in the kernel (counted when
    None), after which the preamble restores the original builder so nothing
    else in the process is affected.
    """
    if PREAMBLE_MARKER in kernel_src:
        raise ValueError("kernel already carries an ACF preamble; embed into the plain kernel")
    n = calls if calls is not None else count_load_inline_calls(kernel_src)
    if n < 1:
        raise ValueError("kernel has no load_inline() call; nothing for an ACF to apply to")
    if not nvcc:
        raise ValueError("nvcc version is required to guard the ACF")
    b64 = base64.b64encode(acf_bytes).decode("ascii")
    pre = (_PREAMBLE.replace("@@ACF_B64@@", b64)
           .replace("@@NVCC@@", nvcc)
           .replace("@@CALLS@@", str(n)))
    return pre + "\n" + kernel_src


def strip_acf(kernel_src: str) -> str:
    """The plain kernel back out of an embedded one (inverse of ``embed_acf``)."""
    end = "# ===== end of ACF preamble; the tuned kernel follows unchanged =====\n"
    i = kernel_src.find(end)
    if i < 0:
        return kernel_src
    return kernel_src[i + len(end):].lstrip("\n")
