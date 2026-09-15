"""Content-addressed names for torch JIT extensions (``load_inline`` / ``load``).

Why this exists
---------------
torch keys a JIT extension on its NAME alone: the build directory is
``<root>/<name>/`` (``cpp_extension._get_build_directory``) and the in-process
``ExtensionVersioner`` entry is ``entries[name]``. LLM-written kernels keep one
name across rounds while the CUDA source changes, and the harness imports
different kernels -- and the same kernel with different flags -- in one process
and across processes. Measured on run/20260911_214651_vae_block_002 (rounds 1-6,
``.ninja_log`` + ``timing.csv``): 31 of 34 nvcc builds were avoidable, about 13
of 79 minutes. Two separate causes:

1. NAME COLLISION. Round 5's paired verdict alternated base
   kernel_20260911_222630 and candidate kernel_20260911_224801, both
   ``name="vae_resblock_gnsilu"`` with different source.
   ``utils.paired_bench.adaptive_paired_verdict`` re-executes both kernels on
   every rep (``compare_and_bench -> _capture_import``), so torch saw changed
   inputs each time, bumped to ``_v1`` .. ``_v15`` and rebuilt main.o AND
   cuda.o every rep (``-DTORCH_EXTENSION_NAME=<name>_vN`` changes both
   commands): 16 builds, 449 s. Round 6's preload, rejected-kernel ncu (32.6 s)
   and nsys (29.8 s) rebuilt each other the same way across processes, because
   one directory only ever holds the last source written into it.
2. FLAG FLIP. The bench import appends ``-Xptxas -v`` (utils/acf.py) while
   preload, ncu and nsys imported plainly. Same folder, different nvcc command,
   so ninja rebuilt cuda.o on every bench<->profile switch: the ~31 s first ncu
   invocation of rounds 3-5 (``vae_resblock_gnsilu_t_pipe/.ninja_log``: ptxas
   build 22:14:06 ``b8631288``, plain cuda.o-only rebuild 22:15:48 ``f07ef309``).

The fix is two parts, and neither works alone:

* ``install()`` wraps ``load_inline``/``load`` so a build is named
  ``<name>_h<sha256(build inputs)[:16]>``. Different inputs get different
  folders (no thrash, no ``_vN`` bumps); identical inputs get the same folder
  in every process (a cache hit). The hash is hashlib over canonical JSON,
  never ``hash()``, which is salted per process.
* ``kernel_build_context()`` is the ONE way the harness imports a kernel:
  install, then ``acf.patched_extension_builds`` with the environment's flag
  policy. Used at every import site (bench, preload, ncu/nsys driver, shape
  coverage), so all of them build byte-for-byte the same command per distinct
  kernel and share one folder. A rename alone would still leave cause 2: the
  plain variant hashes differently and is simply cold.

The correctness bar
-------------------
Within a process, two calls that land on the same hashed name must present
IDENTICAL inputs to torch's ``ExtensionVersioner`` (source-file bytes, the four
flag groups, build_directory, with_cuda, with_sycl, is_python_module,
is_standalone), or torch bumps the hashed name to ``_v1`` and rebuilds -- the
very thing this exists to stop. So the payload is a superset of the versioner's
inputs, never a subset. ``tests/test_ext_naming.py::test_t7_*`` checks this by
calling the real ``load_inline`` hundreds of times with random arguments and
asserting ``get_version`` never leaves 0.

What is NOT hashed, and why that is safe:

* ``verbose``            -- in neither the versioner nor build.ninja; acf forces
                            it on the bench path only.
* ``keep_intermediates`` -- only read by ROCm/HIP hipify builds.
* ``build_directory``    -- a call that passes one is not renamed at all (a
                            rename would not move the directory).
* the build root (``TORCH_EXTENSIONS_DIR``) -- the directory already separates
  builds on disk. It IS a versioner input, so changing the env var mid-process
  and re-importing identical content gets torch's own ``_v1`` rebuild in the new
  root: slower, never a wrong module. No harness path does that (only CHILD
  environments set the variable). (A call with ``--apply-controls`` does hash
  the root, inside the absolute ACF store path it now passes to nvcc; that
  costs no reuse, since another root is another cache anyway.)
* ``MAX_JOBS``, the kernel .py path, time, pid -- not build inputs.

The other direction -- hashed but NOT a versioner input -- only ever costs a
build. One such choice is deliberate: ``load()`` hashes each source file's
ABSOLUTE PATH next to its bytes, although torch's versioner hashes the bytes
only. Headers a source includes relative to its own directory are in neither
hash, so two kernels that write identical listed sources into two directories
with different sibling headers would share a sha-only name -- and torch hands
back the first module for the second (its own bug under a literal name). The
path keeps them apart. The price: a ``load()`` kernel that writes its sources
to a fresh temp path on every exec gets a new folder and a full build on every
exec, even in one process, where a literal name would reuse the first build.
0 of 290 corpus calls use ``load()``.

Pass-through (no rename, torch sees the call as the kernel made it, list
arguments copied): ``build_directory`` given; ``is_python_module`` false (loaded
via ``torch.ops.load_library`` by path); ``is_standalone`` (an executable, found
by name); a name that is not a C identifier; or a source that hard-codes its
module init symbol -- ``PYBIND11_MODULE(<anything but TORCH_EXTENSION_NAME>`` or
``PyInit_<literal>`` -- which would fail to import under any other name (and
already fails under torch's own ``_vN`` bump today). ``KM_CAT(PyInit_,
TORCH_EXTENSION_NAME)`` style is not literal and IS renamed. 0 of 290 corpus
``load_inline`` calls hit any pass-through rule.

``--apply-controls <acf>``
--------------------------
The embedded ACF preamble (``acf._PREAMBLE``) writes its controls to a fresh
``mkstemp`` path on every exec and CompileIQ writes a ``time_ns`` path per
candidate. The versioner hashes the flag STRING, so hashing the bytes alone
would give a stable name that torch still bumps on every re-import. Each such
path is therefore copied (atomically) to ``<root>/_km_acf/<sha256>.bin`` and the
flag passed to torch is rewritten to point there; the payload carries that
store path -- exactly the string the versioner sees, and it embeds the sha.
Same bytes -> same path, same command, same folder. The store path is always
absolute: ninja runs nvcc from inside the build folder, so a relative
``TORCH_EXTENSIONS_DIR`` would otherwise hand nvcc a path it cannot open. An
unreadable path (or a store that cannot be written) is left as-is and its
string is hashed. So is every ACF path of a call where the rewrite would flip
torch's arch heuristic: ``_get_cuda_arch_flags`` drops its default
``-gencode`` set as soon as ANY cuda flag contains the substring ``arch``, so
an ACF under ``.../research/...`` (or a build root under one) must not be
swapped for a path that differs on that substring, or content naming and
``off`` would compile different binaries from the same call.

Also: list arguments are COPIED before torch sees them. torch appends to the
caller's ``extra_ldflags`` (``_prepare_ldflags``) and inserts into
``cpp_sources``/``cuda_sources`` lists in place, so acf's fallback retry used to
re-hash mutated lists and land in a folder no clean build would ever reuse.

Environment
-----------
``KERNELMEM_EXT_NAMING``  ``content`` (default) or ``off``. Read at CALL time;
                          ``off`` makes the wrapper a pure pass-through -- the
                          A/B switch, and what the CompileIQ Evaluator sets to
                          keep its deliberate one-shared-folder design.

``off`` switches off the RENAMING only. ``kernel_build_context`` still applies
one flag policy at every import site, so ``off`` is not the code before this
module existed: then preload, ncu and nsys built plain while the bench added
``-Xptxas -v``, and each switch rebuilt cuda.o. An ``off`` live run therefore
reproduces the name collisions but not the flag flips, and understates the
saving; for a faithful before/after timing compare against a worktree of the
older commit. (The paired-verdict replay and the CompileIQ Evaluator went
through the ptxas wrapper before too, so ``off`` is faithful for those.)

Build folders heal. torch writes main.cpp / cuda.cu with a truncate-then-write
OUTSIDE its FileBaton, so two processes making the same first build can leave a
library compiled from a half-written source. Under a literal name the next
different same-named kernel rewrote the folder; a content-named folder is only
ever written with the same bytes, so it would stay broken for good. When a
library that was already on disk fails to import, the wrapper marks the
folder's outputs stale and rebuilds it once (see ``_build_named``). Preventing
the race outright (an flock per hashed name) is a follow-up.

Known consequences: cache hits print no ``ptxas info`` lines, so
``compare_and_bench``'s ptxas report comes back with ``n_kernels: 0`` for a
kernel whose inputs were already built -- it is only logged and stored in the
eval JSON, never read by a prompt or a decision; folders accumulate one per
distinct build (no GC yet, ~2 MB each); compile-error logs show
``<name>_h<hex>``; a paired verdict of two same-named kernels now reuses each
kernel's loaded module from rep 2 on (pybind11 caches modules by spec name),
the regime ``utils.noise_null_verdict`` was already calibrated in, instead of
rebuilding and loading a fresh ``_vN`` copy between every rep.

This module imports only the standard library at import time; torch is imported
inside the functions that need it, so ``profiling/bench_ref_inputs.py`` and
``utils.compileiq_finish`` (which never imports torch) can use it.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Bump when the payload's shape or meaning changes: every name changes with it,
# so an old folder can never be mistaken for a build of the new recipe.
RECIPE_VERSION = 1

MODE_ENV = "KERNELMEM_EXT_NAMING"
MODE_CONTENT = "content"
MODE_OFF = "off"

CONTROLS_FLAG = "--apply-controls"   # same literal as utils.acf.CONTROLS_FLAG
ACF_STORE_NAME = "_km_acf"

_APIS = ("load_inline", "load")
_MARK = "__kernelmem_ext_naming__"    # value: the api name the wrapper wraps
_LOCK = threading.RLock()
_WARNED: set = set()

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# The original name stays as a readable prefix (and keeps today's module split
# between differently named kernels); cut so ``<prefix>_h<16>_v<N>.so`` stays far
# under NAME_MAX (255). The full name is still hashed.
_PREFIX_MAX = 180
# A source that names its own module init symbol. ``(?!TORCH_EXTENSION_NAME\b)``
# lets the macro through; ``PyInit_`` must be followed by an identifier char, so
# ``KM_CAT(PyInit_, TORCH_EXTENSION_NAME)`` and ``PyInit_##...`` are NOT literal.
_LITERAL_INIT_RE = re.compile(
    r"\bPYBIND11_MODULE\s*\(\s*(?!TORCH_EXTENSION_NAME\b)[A-Za-z_]"
    r"|\bPyInit_[A-Za-z0-9_]"
)

# Parameters deliberately left out of the payload; the module docstring says why.
_UNHASHED = frozenset({"verbose", "keep_intermediates", "build_directory"})
# Environment that reaches build.ninja (or the compiler) but not the versioner.
# Leaving one out would only cost a ninja rebuild, never a wrong module.
_ENV_KEYS = ("CXX", "CC", "PYTORCH_NVCC", "TORCH_CUDA_ARCH_LIST",
             "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES")
# How cpp_extension resolves CUDA_HOME / CUDNN_HOME when it is imported. The
# name payload reads the resolved values from torch; policy_id() must not import
# torch, so it reads these (plus the nvcc on PATH, torch's last fallback).
_TOOLKIT_ENV_KEYS = ("CUDA_HOME", "CUDA_PATH", "CUDNN_HOME", "CUDNN_PATH")
_HEALED: set = set()


def _build_env() -> Dict[str, Optional[str]]:
    """The build-affecting environment, read now. Shared by the name and policy_id()."""
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def _warn(key: str, msg: str) -> None:
    """Print once per *key*. stderr, because the bench captures fd 2 into its log."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[ext_naming] WARNING: {msg}", file=sys.stderr, flush=True)


def mode() -> str:
    """``content`` or ``off``, from ``KERNELMEM_EXT_NAMING`` (read every call)."""
    v = os.environ.get(MODE_ENV, "").strip().lower()
    if v in ("", MODE_CONTENT, "on", "1", "true", "yes"):
        return MODE_CONTENT
    if v in (MODE_OFF, "0", "false", "no", "none", "disabled"):
        return MODE_OFF
    _warn(f"mode:{v}", f"{MODE_ENV}={v!r} is not 'content' or 'off'; using 'content'")
    return MODE_CONTENT


# ------------------------------------------------------------- canonical form
def _jsonable(x: Any) -> Any:
    """A deterministic JSON value for *x* that never merges two versioner inputs.

    The versioner hashes each flag with ``hash(item)``, so a ``Path`` and the
    equal ``str`` are different inputs to it: they get different tags here.
    Being finer than torch only ever costs a build; being coarser would bump.
    """
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {"__dict__": [[_jsonable(k), _jsonable(v)] for k, v in x.items()]}
    if isinstance(x, os.PathLike):
        return {"__path__": os.fspath(x)}
    if isinstance(x, (bytes, bytearray)):
        return {"__bytes__": hashlib.sha256(bytes(x)).hexdigest()}
    return {"__obj__": f"{type(x).__module__}.{type(x).__qualname__}", "repr": repr(x)}


def _norm_cpp_sources(v: Any) -> Any:
    # cpp_extension.load_inline: ``if isinstance(cpp_sources, str): cpp_sources = [cpp_sources]``
    return _jsonable([v] if isinstance(v, str) else v)


def _norm_opt_sources(v: Any) -> Any:
    # ``cuda_sources = cuda_sources or []`` then str -> [str]: None, "" and []
    # all mean "no cuda.cu"; [""] still writes cuda.cu (and turns on with_cuda).
    v = v or []
    return _jsonable([v] if isinstance(v, str) else v)


def _norm_functions(v: Any) -> Any:
    """``functions`` exactly as load_inline turns it into ``m.def`` lines.

    None (no PYBIND11_MODULE block) stays distinct from [] (an empty block).
    str -> [str]; list -> ``{f: f}`` (dict order: first occurrence wins, never
    sorted -- order sets the m.def order and therefore main.cpp's bytes); dict
    kept in insertion order. Anything else makes torch raise, so hash it raw.
    """
    if v is None:
        return None
    if isinstance(v, str):
        v = [v]
    if isinstance(v, list):
        try:
            v = {f: f for f in v}
        except TypeError:  # unhashable entries: torch raises too
            return {"__raw__": _jsonable(v)}
    if isinstance(v, dict):
        return [[format(k), format(d)] for k, d in v.items()]
    return {"__raw__": _jsonable(v)}


def _norm_flags(v: Any) -> Any:
    # The four flag groups and extra_sycl_cflags: torch treats None, "" and []
    # identically (versioner: ``if group:``; ninja: ``x or []``), so they hash
    # the same. Items stay verbatim and ordered: [""] is NOT None -- it adds an
    # item to the versioner hash and a space to build.ninja.
    return _jsonable(v or [])


def _norm_include_paths(v: Any) -> Any:
    # build.ninja uses os.path.abspath (cwd-dependent); the versioner hashes the
    # raw strings. Hash both: "inc" and "/cwd/inc" are different versioner
    # inputs even though they compile the same.
    v = v or []
    if not isinstance(v, (list, tuple)):
        return {"raw": _jsonable(v)}
    absolute = []
    for p in v:
        try:
            absolute.append(os.path.abspath(os.fspath(p)))
        except TypeError:
            absolute.append(None)
    return {"raw": _jsonable(v), "abs": absolute}


def _source_file_digest(path: Any) -> Tuple[Dict[str, Any], str]:
    p = os.fspath(path)
    data = Path(p).read_bytes()
    return ({"path": os.path.abspath(p), "sha256": hashlib.sha256(data).hexdigest()},
            data.decode("utf-8", errors="replace"))


# ----------------------------------------------------------------- ACF paths
def _acf_store_dir(ext) -> str:
    # ABSOLUTE: ninja runs nvcc with cwd=<build folder>, so a relative
    # TORCH_EXTENSIONS_DIR ("ext") gave nvcc "ext/_km_acf/<sha>.bin", which does
    # not exist from inside ext/<name>/ -- the ACF build failed and acf silently
    # fell back to plain (or errored under strict). torch abspaths sources and
    # include paths itself but passes cuda flags verbatim.
    try:
        # <root>/_km_acf beside the build folders; torch resolves the root.
        return os.path.abspath(ext._get_build_directory(ACF_STORE_NAME, False))
    except Exception:
        root = os.environ.get("TORCH_EXTENSIONS_DIR") or ext.get_default_build_root()
        d = os.path.abspath(os.path.join(root, ACF_STORE_NAME))
        os.makedirs(d, exist_ok=True)
        return d


def _store_acf(dst: str, data: bytes) -> str:
    """*dst* (``<root>/_km_acf/<sha>.bin``) holding *data*; written atomically, never rewritten."""
    try:
        with open(dst, "rb") as f:
            if f.read() == data:
                return dst  # untouched: mtime stays put, nothing downstream sees a change
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst),
                               prefix=f".{os.path.basename(dst)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return dst


def _drops_default_arch_flags(flags: List[Any]) -> Optional[bool]:
    """torch's own test (``_get_cuda_arch_flags``): does any flag say ``arch``?

    True means torch adds NO ``-gencode`` flags of its own. None: torch would
    raise on these flags (a non-string item), so there is no decision to keep.
    """
    try:
        for flag in flags:
            if "TORCH_EXTENSION_NAME" in flag:
                continue
            if "arch" in flag:
                return True
        return False
    except TypeError:
        return None


def _canonical_cuda_flags(ext, flags: Any) -> Tuple[Any, Any]:
    """(flags to pass to torch, flags to hash) with every ACF path content-addressed."""
    if not isinstance(flags, (list, tuple)):
        return flags, _norm_flags(flags)
    given = list(flags)
    passed = list(given)
    hashed = _jsonable(given)
    stores: List[Tuple[str, bytes]] = []

    def canon(path: Any) -> Optional[str]:
        try:
            data = Path(os.fspath(path)).read_bytes()
            dst = os.path.join(_acf_store_dir(ext), f"{hashlib.sha256(data).hexdigest()}.bin")
        except (OSError, TypeError, ValueError):
            return None  # unreadable: keep the flag, hash the path string
        stores.append((dst, data))
        return dst

    i = 0
    while i < len(passed):
        item = passed[i]
        if item == CONTROLS_FLAG and i + 1 < len(passed):
            got = canon(passed[i + 1])
            if got is not None:
                passed[i + 1] = hashed[i + 1] = got
            i += 2
            continue
        if isinstance(item, str) and item.startswith(CONTROLS_FLAG + "="):
            got = canon(item[len(CONTROLS_FLAG) + 1:])
            if got is not None:
                passed[i] = hashed[i] = f"{CONTROLS_FLAG}={got}"
        i += 1

    def as_given() -> Tuple[Any, Any]:
        return (tuple(given) if isinstance(flags, tuple) else given), _jsonable(given)

    if not stores:
        return as_given()
    if _drops_default_arch_flags(given) != _drops_default_arch_flags(passed):
        # The rewrite would flip torch's arch heuristic (see the module
        # docstring): keep the kernel's own paths, and with them torch's own
        # -gencode decision. A per-exec temp ACF path then costs a build per
        # exec again, exactly as before content naming.
        _warn(f"arch:{_drops_default_arch_flags(given)}",
              f"not content-addressing --apply-controls: the store path "
              f"{stores[0][0]!r} and the given path differ on the substring 'arch', "
              f"which decides whether torch adds its default -gencode flags; "
              f"building with the given path (a fresh temp path rebuilds every exec)")
        return as_given()
    try:
        for dst, data in stores:
            _store_acf(dst, data)
    except OSError:
        return as_given()  # store not writable: the given paths still work
    return (tuple(passed) if isinstance(flags, tuple) else passed), hashed


# -------------------------------------------------------------- the planner
def _passthrough_reason(api: str, call: Dict[str, Any]) -> Optional[str]:
    name = call.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return "name is not a C identifier"
    if call.get("build_directory"):  # torch: ``build_directory or _get_build_directory(...)``
        return "explicit build_directory"
    if not call.get("is_python_module", True):
        return "is_python_module is false (torch.ops.load_library by path)"
    if call.get("is_standalone"):
        return "is_standalone (an executable, located by name)"
    return None


def _plan(ext, api: str, call: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str], str]:
    """(arguments for torch, effective name or None, why) for one bound call.

    *call* is the fully bound argument dict (defaults applied, lists already
    copied). It is not mutated. Only a renamed call gets its ACF flags
    rewritten; a pass-through goes to torch exactly as bound.
    """
    why = _passthrough_reason(api, call)
    if why:
        return call, None, why

    import torch

    rewritten: Dict[str, Any] = {}
    args: Dict[str, Any] = {}
    texts: List[str] = []
    for key, value in call.items():
        if key in _UNHASHED:
            continue
        if api == "load_inline" and key == "cpp_sources":
            args[key] = _norm_cpp_sources(value)
            texts.extend(s for s in ([value] if isinstance(value, str) else value or [])
                         if isinstance(s, str))
        elif api == "load_inline" and key in ("cuda_sources", "sycl_sources"):
            args[key] = _norm_opt_sources(value)
            texts.extend(s for s in ([value] if isinstance(value, str) else value or [])
                         if isinstance(s, str))
        elif api == "load_inline" and key == "functions":
            args[key] = _norm_functions(value)
        elif api == "load" and key == "sources":
            files = []
            for s in ([value] if isinstance(value, (str, os.PathLike)) else value or []):
                try:
                    digest, text = _source_file_digest(s)
                except (OSError, TypeError):
                    return call, None, f"source {s!r} is unreadable; torch reports it"
                files.append(digest)
                texts.append(text)
            args[key] = files
        elif key == "extra_include_paths":
            args[key] = _norm_include_paths(value)
        elif key == "extra_cuda_cflags":
            rewritten[key], args[key] = _canonical_cuda_flags(ext, value)
        elif key in ("extra_cflags", "extra_sycl_cflags", "extra_ldflags"):
            args[key] = _norm_flags(value)
        else:
            # The booleans, use_pch, no_implicit_headers, and anything a future
            # torch adds -- unknown means hash it, as given.
            args[key] = _jsonable(value)

    for text in texts:
        m = _LITERAL_INIT_RE.search(text)
        if m:
            return call, None, f"source hard-codes its module init ({m.group(0)!r}...)"

    payload = {
        "recipe": RECIPE_VERSION,
        "api": api,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}{getattr(sys, 'abiflags', '')}",
        "env": _build_env(),
        # Resolved once when cpp_extension is imported, exactly as torch uses them.
        "cuda_home": getattr(ext, "CUDA_HOME", None),
        "cudnn_home": getattr(ext, "CUDNN_HOME", None),
        "args": args,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(blob.encode("ascii")).hexdigest()[:16]
    return {**call, **rewritten}, f"{call['name'][:_PREFIX_MAX]}_h{digest}", "content"


def _signature(fn) -> Optional[inspect.Signature]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    kinds = {p.kind for p in sig.parameters.values()}
    if kinds - {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
        return None  # *args/**kwargs/positional-only: cannot re-call with keywords
    return sig


def _bind(sig: inspect.Signature, args: tuple, kwargs: dict) -> Optional[Dict[str, Any]]:
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    # Copies, never the caller's objects: torch mutates list arguments in place.
    return {k: (list(v) if isinstance(v, list) else v) for k, v in bound.arguments.items()}


def effective_name(api: str, *args: Any, **kwargs: Any) -> Optional[str]:
    """The name *api* would build under, or None for a pass-through call.

    Binds against torch's real signature. Does not build; it does materialise
    ``--apply-controls`` files into the ACF store, since the name depends on it.
    """
    import torch.utils.cpp_extension as ext

    fn = inspect.unwrap(getattr(ext, api))
    sig = _signature(fn)
    call = _bind(sig, args, kwargs) if sig is not None else None
    if call is None:
        return None
    return _plan(ext, api, call)[1]


# ---------------------------------------------------------------- the wrapper
def _make_wrapper(ext, orig, api: str):
    sig = _signature(orig)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if sig is None or mode() == MODE_OFF:
            return orig(*args, **kwargs)
        call = _bind(sig, args, kwargs)
        if call is None:
            return orig(*args, **kwargs)  # let torch raise its own TypeError
        try:
            call, name, _why = _plan(ext, api, call)
        except Exception as exc:  # naming must never be why a kernel fails to build
            _warn(f"plan:{type(exc).__name__}",
                  f"could not content-name a {api} call ({type(exc).__name__}: {exc}); "
                  f"building under the kernel's own name")
            name = None
        if name is None:
            return orig(**call)
        call["name"] = name
        return _build_named(ext, orig, call)

    setattr(wrapper, _MARK, api)
    return wrapper


def _file_identity(path: str) -> Optional[Tuple[int, int, int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _build_named(ext, orig, call: Dict[str, Any]):
    """``orig(**call)`` for a content-named python-module call, plus two repairs.

    Both exist because a hashed folder only ever holds ONE content: a literal
    name got these repairs for free the next time a different same-named kernel
    rewrote its folder, and a hashed one never will.

    1. A build that failed in THIS process left torch's versioner entry at
       (version 0, same hash) with no library on disk; torch would then skip
       ninja on the next identical call and raise "cannot open shared object
       file" with no compiler output. With no library, the entry is dropped, so
       torch re-runs ninja and shows the real error. It still lands on version
       0, never ``_v1``.
    2. A library that was already on disk and fails to import is a broken
       build, not a broken kernel -- the kernel built and imported when it was
       first built. Reproduced: two processes making the same first build race
       on torch's truncate-then-write of main.cpp (outside its FileBaton) and
       leave main.o compiled from an empty file, newer than the real source, so
       ninja never rebuilds it and every later process gets "does not define
       module export function". Its outputs are marked stale (mtime -> 1 s
       after the epoch; ninja rebuilds any output older than its inputs, nothing
       is deleted), and the build runs once more. Once per name per process.
    """
    name = call["name"]

    def fresh() -> Dict[str, Any]:
        # torch inserts into cpp_sources and appends to extra_ldflags IN PLACE,
        # so a second attempt must not see the lists the first one mutated
        # (it would write a different main.cpp than every clean build).
        return {k: (list(v) if isinstance(v, list) else v) for k, v in call.items()}

    try:
        build_dir = ext._get_build_directory(name, False)
        lib = os.path.join(build_dir, f"{name}{getattr(ext, 'LIB_EXT', '.so')}")
        entries = ext.JIT_EXTENSION_VERSIONER.entries
    except Exception:
        return orig(**call)

    before = _file_identity(lib)
    if before is None:
        entries.pop(name, None)                                          # (1)
    try:
        return orig(**fresh())
    except ImportError as exc:
        if before is None or _file_identity(lib) != before or name in _HEALED:
            raise
        _HEALED.add(name)                                                # (2)
        print(f"[ext_naming] WARNING: {name}: the existing build does not import "
              f"({str(exc)[-200:]}); marking its outputs stale and rebuilding once",
              file=sys.stderr, flush=True)
        with contextlib.suppress(OSError):
            for entry in os.scandir(build_dir):
                if entry.name.endswith(".o") or entry.name == os.path.basename(lib):
                    os.utime(entry.path, ns=(10**9, 10**9))
        entries.pop(name, None)
        try:
            return orig(**fresh())
        except ImportError:
            after = _file_identity(lib)
            if after is None or after[0] == before[0]:   # same inode: not relinked
                raise
            # Rebuilt, yet still refused: the process already mapped the broken
            # library, and dlopen of the SAME path string hands that mapping
            # back without looking at the file. The rebuilt file is a new inode
            # (the linker unlinks its output first), so a different spelling of
            # the path loads it.
            return ext._import_module_from_library(
                name, os.path.join(build_dir, "."), call.get("is_python_module", True))


def _is_ours(fn: Any) -> bool:
    return getattr(fn, _MARK, None) in _APIS


def _chain_has_wrapper(fn: Any) -> bool:
    """Is our wrapper anywhere below *fn*?

    Walks ``__wrapped__`` (acf's wrapper uses functools.wraps) AND closure cells,
    because the embedded ACF preamble's wrapper keeps ``orig`` in a closure and
    sets no ``__wrapped__``. Bounded, since a closure can hold anything.
    """
    seen: set = set()
    stack = [fn]
    while stack and len(seen) < 64:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        if _is_ours(f):
            return True
        w = getattr(f, "__wrapped__", None)
        if w is not None and callable(w):
            stack.append(w)
        for cell in (getattr(f, "__closure__", None) or ()):
            try:
                c = cell.cell_contents
            except ValueError:
                continue
            if callable(c) and not isinstance(c, type):
                stack.append(c)
    return False


def _is_pristine(ext, api: str, fn: Any) -> bool:
    return (inspect.isfunction(fn) and getattr(fn, "__name__", None) == api
            and getattr(fn, "__globals__", None) is vars(ext)
            and not hasattr(fn, "__wrapped__"))


def install() -> bool:
    """Wrap ``cpp_extension.load_inline``/``load`` once per process. Idempotent.

    Returns True when the naming wrapper is in effect beneath whatever is
    installed now. Refuses (warns, returns False, changes nothing) when a
    foreign wrapper is on top and ours is not below it: installing there would
    make the naming layer OUTERMOST, hash the flags before acf/the preamble add
    theirs, and be removed by that wrapper's own restore. Callers therefore
    install BEFORE ``acf.patched_extension_builds`` -- ``kernel_build_context``
    does exactly that.
    """
    with _LOCK:
        import torch.utils.cpp_extension as ext

        todo = []
        for api in _APIS:
            cur = getattr(ext, api, None)
            if cur is None or _chain_has_wrapper(cur):
                continue
            if not _is_pristine(ext, api, cur):
                _warn(f"install:{api}:{id(cur)}",
                      f"torch.utils.cpp_extension.{api} is already wrapped by "
                      f"{getattr(cur, '__module__', '?')}.{getattr(cur, '__qualname__', cur)!s}; "
                      f"not installing content-hashed extension names over it -- the "
                      f"naming layer must be innermost (builds keep the kernel's own name)")
                return False
            todo.append((api, cur))
        for api, cur in todo:
            setattr(ext, api, _make_wrapper(ext, cur, api))
        return True


_FROM_ENV: Any = object()


@contextlib.contextmanager
def kernel_build_context(*, force_verbose: bool, acf: Any = _FROM_ENV,
                         strict: Any = _FROM_ENV,
                         ptxas_verbose: Any = _FROM_ENV) -> Iterator[Dict[str, Any]]:
    """The single build policy for importing a kernel .py; wrap ``exec_module`` in it.

    ``install()`` first, then ``acf.patched_extension_builds`` with the flag
    policy from the environment (``KERNELMEM_ACF``, ``KERNELMEM_ACF_STRICT``,
    ``KERNELMEM_PTXAS_VERBOSE``) unless a caller that already resolved them
    passes them explicitly (``compile_and_run._capture_import``). Every site
    that imports kernels -- bench/verdict, preload, the ncu/nsys driver, shape
    coverage -- must go through this, or the sites build different commands
    and ninja rebuilds cuda.o each time the path changes (cause 2 above).

    *force_verbose* only decides whether ninja's output is streamed (the bench
    captures it for the ptxas report; profilers want it quiet). It is in
    neither the name nor the ninja command, so it never splits a build.
    Yields acf's build record.

    On exit ``load_inline``/``load`` are put back to what they were on entry
    (the naming wrapper), whatever the kernel did to them. acf restores its own
    patch, but with ``KERNELMEM_PTXAS_VERBOSE=0`` and no ACF it patches nothing
    and restores nothing, and an embedded ACF preamble restores only after the
    number of ``load_inline(`` it COUNTED in the source -- a fallback call in an
    ``except`` branch counts but never runs (kernel_20260729_145230.py: 2
    counted, 1 run). Its wrapper then stayed installed, and the next kernel
    imported in that process -- the plain base of a paired verdict -- was
    silently built with the tuned controls. (install() does not notice: the
    naming wrapper is inside the preamble's closure, so the chain looks fine.)
    """
    import torch.utils.cpp_extension as ext
    from utils import acf as acf_mod

    install()
    entry = {api: getattr(ext, api, None) for api in _APIS}
    try:
        with acf_mod.patched_extension_builds(
                acf=acf_mod.active_acf() if acf is _FROM_ENV else acf,
                strict=acf_mod.strict_enabled() if strict is _FROM_ENV else strict,
                ptxas_verbose=(acf_mod.ptxas_verbose_enabled() if ptxas_verbose is _FROM_ENV
                               else ptxas_verbose),
                force_verbose=force_verbose) as record:
            yield record
    finally:
        with _LOCK:
            for api, fn in entry.items():
                if fn is not None and getattr(ext, api, None) is not fn:
                    setattr(ext, api, fn)


def policy_id() -> str:
    """The build policy and toolchain outside a kernel's source, as one string.

    For cache keys over artifacts produced by building a kernel (the ncu CSV
    cache) and as a regime label on paired verdicts. It covers:

    * the naming recipe and mode. The mode changes no code, but it decides
      whether a profiled import collides and compiles under the profiler --
      the very ncu invocation time an ``off``/``content`` A/B looks at, which
      a cache hit would hide;
    * the ptxas flags and the ACF policy (``KERNELMEM_ACF``, ``_STRICT``);
    * ``KERNELMEM_ACF_DISABLE`` and the local nvcc build: an embedded ACF
      preamble applies its controls only when the first is off and the second
      matches, so either flips the binary of a kernel whose bytes are unchanged;
    * the build environment the name hashes (``CXX``, ``CC``, ``PYTORCH_NVCC``,
      ``TORCH_CUDA_ARCH_LIST``, ...) and where the CUDA/cuDNN toolkits come
      from, as a short digest.

    Stdlib only (no torch import); runs ``nvcc --version`` once per process.
    It cannot see ``#include``-d files or a toolkit replaced in place under the
    same version string.
    """
    from utils import acf as acf_mod

    a = acf_mod.active_acf()
    if a is None:
        acf_tag = "none"
    else:
        try:
            acf_tag = "sha256:" + hashlib.sha256(Path(a).read_bytes()).hexdigest()[:16]
        except OSError:
            acf_tag = f"unreadable:{a}"
    nvcc = acf_mod.nvcc_version()
    toolchain = {**_build_env(), **{k: os.environ.get(k) for k in _TOOLKIT_ENV_KEYS},
                 "nvcc_on_path": shutil.which("nvcc"), "nvcc_version": nvcc}
    toolchain_tag = hashlib.sha256(
        json.dumps(toolchain, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    return (f"ext_naming=r{RECIPE_VERSION}:{mode()}"
            f"|ptxas={int(acf_mod.ptxas_verbose_enabled())}"
            f"|acf={acf_tag}|acf_strict={int(acf_mod.strict_enabled())}"
            f"|acf_disable={int(acf_mod._truthy(os.environ.get(acf_mod.ACF_DISABLE_ENV)))}"
            f"|nvcc={nvcc}|toolchain={toolchain_tag}")
