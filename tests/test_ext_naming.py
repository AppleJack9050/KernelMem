"""Regression tests for content-hashed extension names (utils/ext_naming.py).

Background
----------
torch keys a JIT extension's build folder and its in-process rebuild decision
on the NAME alone. In run/20260911_214651_vae_block_002 that cost 31 of 34 nvcc
builds in rounds 1-6 (~13 of 79 min): the round-5 paired verdict alternated two
kernels both named ``vae_resblock_gnsilu`` and rebuilt both every rep
(``_v1`` .. ``_v15``, 449 s), same-named kernels profiled in one round rebuilt
each other, and the bench's ``-Xptxas -v`` versus the profilers' plain import
rebuilt cuda.o on every switch. The fix names each build after a hash of its
inputs and routes every kernel import site through one build policy.

What must hold, in order of how badly its failure would mislead a run:

* A hashed name never lets torch's versioner bump (T7): if two calls share a
  hashed name they present identical versioner inputs, or the fix rebuilds
  exactly as before -- and, worse, in a folder name that suggests a cache hit.
* Different sources never share a module (T1 markers, T10): a wrong module
  returned silently is strictly worse than a rebuild.
* The alternation, cross-process thrash and flag flip are actually gone
  (T1-T4, with ``KERNELMEM_EXT_NAMING=off`` reproducing each bug first), and
  every harness import site shares one build (T4b, T12).
* The naming layer stays innermost and survives acf's and the ACF preamble's
  restores (T5), ACF paths are content-addressed (T6), caller lists are never
  mutated (T9), and the hash is a pure function of the inputs (T8).

Run directly::

    python tests/test_ext_naming.py            # ~30 s
    python tests/test_ext_naming.py --slow     # + the pybind11 (functions=) path, ~15 s

or ``pytest tests/test_ext_naming.py`` (``KERNELMEM_SLOW_TESTS=1`` for the slow
one). No GPU (``CUDA_VISIBLE_DEVICES=""`` in every child), no CUDA sources, no
real kernel imports. Every build runs in a fresh subprocess with its own
``TORCH_EXTENSIONS_DIR=tempfile.mkdtemp()`` and a random ``PYTHONHASHSEED``,
using a C-API-only extension (no ``torch/extension.h``) that compiles in ~0.1 s.
Builds are counted two ways: calls to ``cpp_extension._run_ninja_build`` and
``main.o`` lines appended to ``.ninja_log``.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_SCRATCH_PREFIX = "km_ext_naming_test_"
_HASHED = re.compile(r"_h[0-9a-f]{16}$")

# A Python extension with no torch headers: main.o builds in ~0.1 s. The init
# symbol comes from TORCH_EXTENSION_NAME, so it imports under any name torch
# (or the naming wrapper) gives it -- exactly the property generated kernels
# get from ``functions=``. MARKER tells the parent WHICH source it got back.
_C_SRC = r'''
#include <Python.h>
#define KM_CAT_(a,b) a##b
#define KM_CAT(a,b) KM_CAT_(a,b)
static int km_exec(PyObject* m){ return PyModule_AddIntConstant(m, "marker", MARKER); }
static PyModuleDef_Slot km_slots[] = {{Py_mod_exec, (void*)km_exec}, {0, NULL}};
static struct PyModuleDef km_def = {PyModuleDef_HEAD_INIT, "km", NULL, 0, NULL, km_slots};
PyMODINIT_FUNC KM_CAT(PyInit_, TORCH_EXTENSION_NAME)(void){ return PyModuleDef_Init(&km_def); }
'''


def _src(marker: int) -> str:
    return f"#define MARKER {marker}\n" + _C_SRC


# ==========================================================================
# parent-side helpers
# ==========================================================================
class _Skip(Exception):
    pass


def _skip(msg: str):
    if os.environ.get("_KM_DIRECT_RUN"):
        raise _Skip(msg)
    import pytest
    pytest.skip(msg)


def _mkdtemp() -> Path:
    return Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX))


def _child_tmp() -> Path:
    """A child's scratch dir, inside the parent's test dir so its rmtree takes it."""
    return Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX, dir=os.environ.get("KM_TEST_TMP") or None))


def _child_env(ext_dir: Path, mode: str = "content", **extra) -> dict:
    env = dict(os.environ)
    for k in ("KERNELMEM_ACF", "KERNELMEM_ACF_STRICT", "KERNELMEM_PTXAS_VERBOSE",
              "KERNELMEM_ACF_DISABLE", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS"):
        env.pop(k, None)
    env.update({
        "TORCH_EXTENSIONS_DIR": str(ext_dir),
        "PYTHONHASHSEED": str(random.randrange(1, 2**31)),
        "CUDA_VISIBLE_DEVICES": "",            # never touch the GPU
        "PYTHONPATH": str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "KERNELMEM_EXT_NAMING": mode,
        "PYTHONDONTWRITEBYTECODE": "1",
        "KM_TEST_TMP": str(ext_dir.parent),   # every test passes <its own mkdtemp>/<sub>
    })
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _parse_result(scenario: str, rc: int, out: str, err: str) -> dict:
    for line in reversed(out.splitlines()):
        if line.startswith("RESULT "):
            if rc != 0:
                break
            return json.loads(line[len("RESULT "):])
    raise AssertionError(f"child {scenario!r} failed (rc={rc})\n--- stdout ---\n{out[-4000:]}"
                         f"\n--- stderr ---\n{err[-4000:]}")


def _child_run(scenario: str, *args: str, ext_dir: Path, mode: str = "content",
               timeout: float = 600, **env_extra) -> subprocess.CompletedProcess:
    ext_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run([sys.executable, __file__, "--child", scenario, *args],
                          env=_child_env(ext_dir, mode, **env_extra), cwd=str(ext_dir.parent),
                          capture_output=True, text=True, timeout=timeout)


def _child(scenario: str, *args: str, ext_dir: Path, mode: str = "content",
           timeout: float = 600, **env_extra) -> dict:
    p = _child_run(scenario, *args, ext_dir=ext_dir, mode=mode, timeout=timeout, **env_extra)
    return _parse_result(scenario, p.returncode, p.stdout, p.stderr)


def _compiles(ext_dir: Path, output: str = "main.o") -> dict:
    """{build folder: lines for *output* in its .ninja_log} -- real compiles only.

    *output* ``"{name}.so"`` counts links, with the folder name substituted.
    """
    out = {}
    for log in ext_dir.glob("*/.ninja_log"):
        target = output.format(name=log.parent.name)
        n = sum(1 for line in log.read_text().splitlines()
                if not line.startswith("#") and line.split("\t")[3:4] == [target])
        out[log.parent.name] = n
    return out


def _versioned_sos(ext_dir: Path) -> list:
    return sorted(p.name for p in ext_dir.glob("*/*_v[0-9]*.so"))


def _write_kernel(path: Path, marker: int) -> Path:
    """A kernel-shaped .py: module-level load_inline with the kernel corpus's keywords."""
    path.write_text(
        "import torch\n"
        "from torch.utils.cpp_extension import load_inline\n\n"
        f"_SRC = {_src(marker)!r}\n\n"
        "_ext = load_inline(name='km', cpp_sources=_SRC, no_implicit_headers=True,\n"
        "                   functions=None, extra_cuda_cflags=['-O3'], extra_ldflags=[''],\n"
        "                   verbose=False)\n"
        "MARKER = _ext.marker\n"
        "EXT_NAME = _ext.__name__\n\n"
        "class ModelNew(torch.nn.Module):\n"
        "    pass\n")
    return path


# ==========================================================================
# child-side scenarios (run as: python tests/test_ext_naming.py --child NAME ...)
# ==========================================================================
def _counting_ext():
    import torch.utils.cpp_extension as ext
    ninja = []
    real = ext._run_ninja_build

    def counting(build_directory, verbose, error_prefix):
        ninja.append(os.path.basename(build_directory))
        return real(build_directory, verbose, error_prefix)

    ext._run_ninja_build = counting
    return ext, ninja


def _versions(ext) -> dict:
    return {n: e.version for n, e in ext.JIT_EXTENSION_VERSIONER.entries.items()}


def _c_alternate(n: str = "6") -> dict:
    """T1: A,B,A,B... in one process, as paired_bench re-imports base and candidate."""
    from utils import ext_naming
    ext, ninja = _counting_ext()
    installed = ext_naming.install()
    markers, names = [], []
    for i in range(int(n)):
        m = ext.load_inline(name="km", cpp_sources=_src(1 if i % 2 == 0 else 2),
                            no_implicit_headers=True, functions=None)
        markers.append(m.marker)
        names.append(m.__name__)
    return {"installed": installed, "markers": markers, "names": names, "ninja": ninja,
            "versions": _versions(ext)}


def _c_import_one(marker: str, go_file: str = "", ready_file: str = "", ldflags: str = "") -> dict:
    """One import of source *marker* under name km (T2, T3, T10, T14).

    *ldflags*: comma-separated ``extra_ldflags``, passed as a LIST -- the
    argument torch appends its own libraries to in place.
    """
    from utils import ext_naming
    ext, ninja = _counting_ext()
    ext_naming.install()
    if ready_file:
        Path(ready_file).touch()
    if go_file:
        deadline = time.time() + 120
        while not os.path.exists(go_file) and time.time() < deadline:
            time.sleep(0.005)
    kw = {"extra_ldflags": ldflags.split(",")} if ldflags else {}
    m = ext.load_inline(name="km", cpp_sources=_src(int(marker)), no_implicit_headers=True,
                        functions=None, **kw)
    return {"marker": m.marker, "name": m.__name__, "ninja": ninja, "versions": _versions(ext)}


def _c_flag_split() -> dict:
    """T4: plain and ptxas imports of one source alternate through the REAL acf wrapper."""
    from utils import acf, ext_naming
    ext, ninja = _counting_ext()
    ext_naming.install()
    names, markers = [], []
    for i in range(4):
        kw = dict(name="km", cpp_sources=_src(1), no_implicit_headers=True, functions=None,
                  extra_cuda_cflags=["-O3"])
        if i % 2 == 0:
            m = ext.load_inline(**kw)
        else:
            with acf.patched_extension_builds(ptxas_verbose=True, force_verbose=False):
                m = ext.load_inline(**kw)
        names.append(m.__name__)
        markers.append(m.marker)
    return {"names": names, "markers": markers, "ninja": ninja, "versions": _versions(ext)}


class _Conn:
    def __init__(self):
        self.sent = []

    def send(self, x):
        self.sent.append(x)

    def close(self):
        pass


def _c_site(site: str, kernel: str) -> dict:
    """T4b: import one kernel file through a real harness import site."""
    import importlib.util
    ext, ninja = _counting_ext()
    if site == "capture_import":
        from utils import acf as acf_mod
        from utils.compile_and_run import _capture_import
        # Exactly the candidate import in compare_and_bench.
        mod, _log = _capture_import(Path(kernel), acf=acf_mod.active_acf(),
                                    acf_strict=acf_mod.strict_enabled(),
                                    ptxas_verbose=acf_mod.ptxas_verbose_enabled())
        rec = mod.__dict__.get("__kernelmem_build__")
    elif site == "preload":
        import main_memory_latest as mm
        conn = _Conn()
        mm._preload_worker(kernel, conn)
        assert conn.sent == [("ok", "loaded")], conn.sent
        mod = sys.modules["preload_test_kernel_temp"]
        rec = None
    elif site == "bench_ref_inputs":
        spec = importlib.util.spec_from_file_location(
            "km_bench_driver", REPO / "profiling" / "bench_ref_inputs.py")
        drv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drv)
        mod = drv._load_kernel_module(kernel, "bench_test_module")
        rec = None
    else:
        raise ValueError(site)
    return {"site": site, "marker": mod.MARKER, "name": mod.EXT_NAME, "ninja": ninja,
            "versions": _versions(ext), "record": rec}


def _stub_builds(ext, fail_on_controls: bool = False):
    """Replace the compile+import with a recorder; load_inline's own logic still runs."""
    seen = []

    def fake_jit(name, sources, extra_cflags, extra_cuda_cflags, *a, **k):
        seen.append({"name": name, "cuda_flags": list(extra_cuda_cflags or [])})
        if fail_on_controls and "--apply-controls" in (extra_cuda_cflags or []):
            raise RuntimeError("nvcc died")
        return "module"

    ext._jit_compile = fake_jit
    return seen


_EMBED_KERNEL = '''\
import torch
from torch.utils.cpp_extension import load_inline

_ext = load_inline(
    name="my_ext",
    cpp_sources="int x;",
    no_implicit_headers=True,
    extra_cuda_cflags=["-O3"],
)
'''


def _c_layering() -> dict:
    """T5: install order, idempotence, and survival through every restore path."""
    import subprocess as sp
    import torch.utils.cpp_extension as ext
    from utils import acf, ext_naming

    r: dict = {}
    pristine = ext.load_inline
    with acf.patched_extension_builds(ptxas_verbose=True):
        r["install_inside_acf_refused"] = ext_naming.install() is False
        r["inside_top_untouched"] = not ext_naming._is_ours(ext.load_inline)
    r["refusal_left_pristine"] = ext.load_inline is pristine

    r["first_install"] = ext_naming.install()
    w = ext.load_inline
    r["second_install"] = ext_naming.install()
    r["one_layer"] = (ext.load_inline is w and ext_naming._is_ours(w)
                      and w.__wrapped__ is pristine)
    with acf.patched_extension_builds(ptxas_verbose=True):
        r["install_inside_acf_after_install_ok"] = ext_naming.install() is True
        r["acf_on_top_during_block"] = ext.load_inline is not w

    seen = _stub_builds(ext)
    kw = dict(name="km", cpp_sources=_src(1), no_implicit_headers=True, extra_cuda_cflags=["-O3"])
    with acf.patched_extension_builds(ptxas_verbose=True):
        ext.load_inline(**kw)
    r["restored_after_normal_exit"] = ext.load_inline is w
    r["renamed_inside_acf"] = bool(_HASHED.search(seen[-1]["name"]))
    r["ptxas_flags_hashed"] = seen[-1]["cuda_flags"] == ["-O3", "-Xptxas", "-v"]
    try:
        with acf.patched_extension_builds(ptxas_verbose=True):
            raise KeyError("boom")
    except KeyError:
        pass
    r["restored_after_exception"] = ext.load_inline is w

    ctl = _child_tmp() / "c.bin"
    ctl.write_bytes(b"controls")
    seen = _stub_builds(ext, fail_on_controls=True)
    try:
        with acf.patched_extension_builds(acf=ctl, strict=True):
            ext.load_inline(**kw)
        r["strict_raised"] = False
    except RuntimeError:
        r["strict_raised"] = True
    r["restored_after_strict_failure"] = ext.load_inline is w

    # The embedded preamble: count-based restore, and its per-exec mkstemp path.
    seen = _stub_builds(ext)
    src = acf.embed_acf(_EMBED_KERNEL, b"\x01\x02tuned", nvcc="13.3.33")
    real_run = sp.run

    class _P:
        stdout = "Cuda compilation tools, release 13.3, V13.3.33\n"

    sp.run = lambda *a, **k: _P()
    try:
        for _ in range(2):
            exec(compile(src, "embedded_kernel.py", "exec"), {"__name__": "embedded_kernel"})
            r.setdefault("preamble_restored", []).append(ext.load_inline is w)
    finally:
        sp.run = real_run
    a, b = seen[-2], seen[-1]
    r["preamble_same_name_across_execs"] = a["name"] == b["name"] and bool(_HASHED.search(a["name"]))
    r["preamble_same_flags_across_execs"] = a["cuda_flags"] == b["cuda_flags"]
    r["preamble_flag_points_at_store"] = f"{os.sep}{ext_naming.ACF_STORE_NAME}{os.sep}" in a["cuda_flags"][2]
    return r


def _c_acf() -> dict:
    """T6: ACF paths are content-addressed, so fresh temp paths do not bump torch."""
    import hashlib
    from utils import ext_naming
    ext, ninja = _counting_ext()
    ext_naming.install()
    tmp = _child_tmp()
    p1, p2, p3 = tmp / "a1.bin", tmp / "a2.bin", tmp / "b.bin"
    p1.write_bytes(b"ACF-one")
    p2.write_bytes(b"ACF-one")
    p3.write_bytes(b"ACF-two")

    def name_for(flags):
        return ext_naming.effective_name("load_inline", name="km", cpp_sources=_src(1),
                                         no_implicit_headers=True, extra_cuda_cflags=flags)

    r = {
        "same_bytes_same_name": name_for(["--apply-controls", str(p1)]) == name_for(["--apply-controls", str(p2)]),
        "different_bytes_different_name": name_for(["--apply-controls", str(p1)]) != name_for(["--apply-controls", str(p3)]),
        "no_acf_differs": name_for(["--apply-controls", str(p1)]) != name_for([]),
    }
    call = ext_naming._bind(ext_naming._signature(ext.load_inline), (),
                            dict(name="km", cpp_sources=_src(1), extra_cuda_cflags=["-O3", "--apply-controls", "/no/such/acf.bin"]))
    passed, name, _ = ext_naming._plan(ext, "load_inline", call)
    r["unreadable_left_unchanged"] = passed["extra_cuda_cflags"] == ["-O3", "--apply-controls", "/no/such/acf.bin"]
    r["unreadable_still_named"] = bool(name and _HASHED.search(name))
    # A pass-through goes to torch as bound: no ACF path rewrite either.
    lit = ext_naming._bind(ext_naming._signature(ext.load_inline), (),
                           dict(name="km", cpp_sources="PYBIND11_MODULE(my_ext, m) {}",
                                extra_cuda_cflags=["--apply-controls", str(p3)]))
    passed, name, _ = ext_naming._plan(ext, "load_inline", lit)
    r["passthrough_flags_untouched"] = name is None and passed["extra_cuda_cflags"] == ["--apply-controls", str(p3)]

    recorded = []
    real_jit = ext._jit_compile

    def recording_jit(name, sources, extra_cflags, extra_cuda_cflags, *a, **k):
        recorded.append(list(extra_cuda_cflags or []))
        return real_jit(name, sources, extra_cflags, extra_cuda_cflags, *a, **k)

    ext._jit_compile = recording_jit
    mods = []
    for i in range(3):
        # A fresh mkstemp path per import, exactly what the embedded preamble does.
        fd, fresh = tempfile.mkstemp(dir=tmp, suffix=".acf.bin")
        with os.fdopen(fd, "wb") as f:
            f.write(b"ACF-one")
        mods.append(ext.load_inline(name="km", cpp_sources=_src(1), no_implicit_headers=True,
                                    functions=None, extra_cuda_cflags=["--apply-controls", fresh]))
    sha = hashlib.sha256(b"ACF-one").hexdigest()
    store = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / ext_naming.ACF_STORE_NAME / f"{sha}.bin"
    r.update({
        "names": [m.__name__ for m in mods],
        "markers": [m.marker for m in mods],
        "flags_identical": all(f == recorded[0] for f in recorded),
        "flag_is_store_path": recorded[0][1] == str(store),
        "store_bytes_ok": store.read_bytes() == b"ACF-one",
        "ninja": ninja,
        "versions": _versions(ext),
    })
    return r


_FAKE_NVCC = r'''#!@PYTHON@
# Stands in for nvcc: logs what the real one would be handed, fails the way nvcc
# does when --apply-controls names a file it cannot open (from ninja's cwd, the
# build folder), and otherwise emits an empty object so the build links.
import json, os, subprocess, sys
a = sys.argv[1:]
out = a[a.index("-o") + 1]
acf = next((a[i + 1] for i, x in enumerate(a[:-1]) if x == "--apply-controls"), None)
rec = {"cwd": os.getcwd(), "acf": acf, "acf_found": acf is None or os.path.isfile(acf),
       "gencode": any(x.startswith("-gencode") for x in a)}
with open(os.environ["KM_FAKE_NVCC_LOG"], "a") as f:
    f.write(json.dumps(rec) + "\n")
if not rec["acf_found"]:
    sys.exit("nvcc fatal: cannot open controls file %s" % acf)
subprocess.run(["@CXX@", "-x", "c++", "-c", "-fPIC", "-", "-o", out], input=b"", check=True)
with open(out + ".d", "w") as f:
    f.write(out + ":\n")
'''


def _c_acf_nvcc() -> dict:
    """H2/H3: what nvcc is really handed for an ACF build, via a logging fake nvcc.

    Real load_inline + real ninja with ``cuda_sources`` (the import is stubbed:
    the fake objects hold no module). Cases: a RELATIVE build root, an ACF path
    that contains 'arch' under a root that does not, and the reverse.
    """
    import torch.utils.cpp_extension as ext
    from utils import ext_naming
    tmp = _child_tmp()
    os.chdir(tmp)
    fake = tmp / "fake_nvcc.py"
    fake.write_text(_FAKE_NVCC.replace("@PYTHON@", sys.executable)
                    .replace("@CXX@", shutil.which("c++") or "/usr/bin/c++"))
    fake.chmod(0o755)
    log = tmp / "nvcc_log.jsonl"
    os.environ.update({"PYTORCH_NVCC": str(fake), "KM_FAKE_NVCC_LOG": str(log),
                       "TORCH_CUDA_ARCH_LIST": "12.0"})
    ext._import_module_from_library = lambda name, path, is_python_module: name
    ext_naming.install()

    def acf_in(dirname: str) -> str:
        d = tmp / dirname
        d.mkdir(exist_ok=True)
        (d / "controls.bin").write_bytes(b"ACF-BYTES")
        return str(d / "controls.bin")

    cases = {"relative_root": ("rel_ext", acf_in("neutral_acf")),
             "acf_path_says_arch": (str(tmp / "plain_root"), acf_in("search_acf")),
             "root_says_arch": (str(tmp / "research_root"), acf_in("neutral_acf")),
             "neutral": (str(tmp / "plain_root2"), acf_in("neutral_acf"))}
    out = {"tmp": str(tmp)}
    for case, (root, acf_path) in cases.items():
        os.environ["TORCH_EXTENSIONS_DIR"] = root
        before = len(log.read_text().splitlines()) if log.exists() else 0
        try:
            got = ext.load_inline(name=f"km_{case}", cpp_sources="int x;", cuda_sources="int y;",
                                  functions=None, no_implicit_headers=True,
                                  extra_cuda_cflags=["-O3", "--apply-controls", acf_path])
            err = None
        except Exception as exc:  # noqa: BLE001
            got, err = None, f"{type(exc).__name__}: {str(exc)[-300:]}"
        calls = [json.loads(x) for x in log.read_text().splitlines()[before:]] if log.exists() else []
        out[case] = {"given": acf_path, "root": root, "name": got, "error": err, "nvcc": calls}
    return out


def _c_preamble_leftover() -> dict:
    """W2: a preamble that counted more load_inline( calls than ran must not leak.

    KERNELMEM_PTXAS_VERBOSE=0 and no ACF, so acf patches (and restores) nothing.
    The tuned kernel's fallback call sits in an ``except`` branch: counted, never
    run. Then the PLAIN kernel is imported in the same process, as a paired
    verdict does.
    """
    import subprocess as sp
    import torch.utils.cpp_extension as ext
    from utils import acf, ext_naming
    assert not acf.ptxas_verbose_enabled() and acf.active_acf() is None
    ext_naming.install()
    w = ext.load_inline
    seen = _stub_builds(ext)
    plain = _EMBED_KERNEL + (
        "try:\n    pass\nexcept Exception:\n"
        "    _ext = load_inline(name='my_ext', cpp_sources='int x;', no_implicit_headers=True)\n")
    tuned = acf.embed_acf(plain, b"\x01\x02tuned", nvcc="13.3.33")
    real_run = sp.run

    class _P:
        stdout = "Cuda compilation tools, release 13.3, V13.3.33\n"

    def run(src, how):
        if how == "context":
            with ext_naming.kernel_build_context(force_verbose=False):
                exec(compile(src, "k.py", "exec"), {"__name__": "k"})
        else:   # the old call: acf's no-op patch alone
            with acf.patched_extension_builds(acf=None, strict=False, ptxas_verbose=False):
                exec(compile(src, "k.py", "exec"), {"__name__": "k"})
        return seen[-1]["cuda_flags"]

    sp.run = lambda *a, **k: _P()
    try:
        r = {"counted_calls": acf.count_load_inline_calls(plain)}
        r["bare_tuned_flags"] = run(tuned, "bare")
        r["bare_leftover"] = ext.load_inline is not w
        r["bare_next_plain_flags"] = run(plain, "bare")   # the bug, reproduced
        ext.load_inline = w
        r["ctx_tuned_flags"] = run(tuned, "context")
        r["ctx_restored"] = ext.load_inline is w
        r["ctx_next_plain_flags"] = run(plain, "context")
        r["ctx_tuned_again_flags"] = run(tuned, "context")
        r["ctx_restored_again"] = ext.load_inline is w
    finally:
        sp.run = real_run
    return r


def _c_compile_error() -> dict:
    """F2: re-importing a source that failed to compile shows the compiler error again."""
    from utils import ext_naming
    ext, ninja = _counting_ext()
    ext_naming.install()
    broken = "#define MARKER 3\nint broken( {\n"
    out = []
    for label, src in (("X", broken), ("Y", _src(2)), ("X", broken), ("X", broken)):
        try:
            m = ext.load_inline(name="km", cpp_sources=src, no_implicit_headers=True, functions=None)
            out.append([label, "ok", m.marker])
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            out.append([label, type(exc).__name__,
                        "compiler_diagnostic" if "error:" in msg else msg[-160:]])
    return {"attempts": out, "ninja": ninja, "versions": _versions(ext)}


def _c_property(n: str = "600", seed: str = "0", break_mode: str = "") -> dict:
    """T7: random real load_inline/load calls; a hashed name's version never leaves 0.

    A purely random argument stream almost never repeats a build, and a name
    seen once is trivially at version 0 -- that would test nothing. So calls
    are drawn from a POOL of semantic configurations, each realised with a
    random SURFACE form that must not change the build (str vs [str];
    None/""/[]; tuple vs list; a fresh mkstemp copy of the same ACF bytes;
    verbose; keep_intermediates; positional vs keyword). Every configuration
    also has a TWIN that differs in exactly one semantic value, so a payload
    that merged two genuinely different inputs would put them under one name
    and torch would bump it. *break_mode* installs such a merge on purpose, to
    prove the check can see one.
    """
    import inspect
    import torch.utils.cpp_extension as ext
    from utils import ext_naming

    # The stub build must leave the library behind like a real one: with no
    # library on disk the wrapper drops the versioner entry on purpose (a failed
    # build is retried, see ext_naming._build_named), which would reset every
    # name to version 0 and make this property vacuous.
    ext._write_ninja_file_and_build_library = (
        lambda **kw: Path(kw["build_directory"], kw["name"] + ext.LIB_EXT).touch())
    ext._import_module_from_library = lambda name, path, is_python_module: name
    ext._get_exec_path = lambda name, path: os.path.join(path, name)
    ext_naming.install()
    if break_mode == "abspath_only":        # the first draft's include-path hashing
        ext_naming._norm_include_paths = lambda v: [os.path.abspath(os.fspath(x)) for x in (v or [])]
    elif break_mode == "drop_functions":
        ext_naming._norm_functions = lambda v: None
    elif break_mode:
        raise ValueError(break_mode)

    rng = random.Random(int(seed))
    tmp = _child_tmp()
    os.chdir(tmp)
    inc_abs = tmp / "inc"
    inc_abs.mkdir()
    acf1, acf1b, acf2 = tmp / "acf1.bin", tmp / "acf1_copy.bin", tmp / "acf2.bin"
    acf1.write_bytes(b"one")
    acf1b.write_bytes(b"one")
    acf2.write_bytes(b"two")
    fa, fa_copy, fb = tmp / "a.cpp", tmp / "sub_a.cpp", tmp / "b.cpp"
    fa.write_text(_src(1))
    fa_copy.write_text(_src(1))
    fb.write_text(_src(2))
    own_dir = tmp / "own_build_dir"
    own_dir.mkdir()

    def fresh(data: bytes) -> str:
        fd, path = tempfile.mkstemp(dir=tmp, suffix=".acf.bin")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    A, B = _src(1), _src(2)
    # semantic key -> (weight, [surface forms]); a callable surface is realised per call
    flags_c = {"none": (3, [None, [], ""]), "O2": (1, [["-O2"], ("-O2",)]), "O2g": (1, [["-O2", "-g"]])}
    flags_cuda = {
        "none": (3, [None, [], ""]), "O3": (1, [["-O3"], ("-O3",)]),
        "ptxas": (2, [["-O3", "-Xptxas", "-v"]]),
        "acf1": (2, [lambda: ["-O3", "--apply-controls", str(acf1)],
                     lambda: ["-O3", "--apply-controls", str(acf1b)],
                     lambda: ["-O3", "--apply-controls", fresh(b"one")]]),
        "acf2eq": (1, [lambda: ["--apply-controls=" + str(acf2)],
                       lambda: ["--apply-controls=" + fresh(b"two")]]),
        "unreadable": (1, [["--apply-controls", "/no/such/file.bin"]]),
    }
    common = {
        "extra_cflags": flags_c,
        "extra_cuda_cflags": flags_cuda,
        "extra_sycl_cflags": {"none": (6, [None, []]), "fsycl": (1, [["-fsycl"]])},
        "extra_ldflags": {"none": (3, [None, [], ""]), "empty_item": (2, [[""]]),
                          "lm": (1, [["-lm"], ("-lm",)])},
        "extra_include_paths": {"none": (4, [None, []]), "rel": (1, [["inc"]]),
                                "abs": (1, [lambda: [str(inc_abs)]]), "pathobj": (1, [lambda: [inc_abs]])},
        "build_directory": {"none": (9, [None]), "own": (1, [lambda: str(own_dir)])},
        "with_cuda": {"None": (4, [None]), "T": (1, [True]), "F": (1, [False])},
        "is_python_module": {"T": (9, [True]), "F": (1, [False])},
    }
    sem = {
        "load_inline": dict(common, **{
            "name": {"km": (1, ["km"]), "km2": (1, ["km2"])},
            "cpp_sources": {"A": (2, [A, [A]]), "B": (1, [B, [B]]), "AB": (1, [[A, B]])},
            "cuda_sources": {"none": (5, [None, "", []]), "empty_item": (1, [[""]]),
                             "src": (1, ["int c;", ["int c;"]])},
            "sycl_sources": {"none": (8, [None, "", []]), "src": (1, ["int s;", ["int s;"]])},
            "functions": {"None": (2, [None]), "empty": (1, [[]]),
                          "f": (2, ["f", ["f"], {"f": "f"}, ["f", "f"]]),
                          "fg": (1, [["f", "g"], ["f", "g", "f"], {"f": "f", "g": "g"}]),
                          "gf": (1, [["g", "f"]]), "fdoc": (1, [{"f": "doc"}])},
            "with_sycl": {"None": (6, [None]), "F": (1, [False]), "T": (1, [True])},
            "with_pytorch_error_handling": {"T": (3, [True]), "F": (1, [False])},
            "use_pch": {"F": (1, [False])},    # True builds a PCH inside torch's own include dir
            "no_implicit_headers": {"F": (1, [False]), "T": (1, [True])},
        }),
        "load": dict(common, **{
            "name": {"kl": (1, ["kl"]), "kl2": (1, ["kl2"])},
            "sources": {"A": (3, [lambda: str(fa), lambda: [str(fa)], lambda: [fa]]),
                        "AB": (1, [lambda: [str(fa), str(fb)]]), "BA": (1, [lambda: [str(fb), str(fa)]]),
                        "Acopy": (1, [lambda: [str(fa_copy)]])},
            "with_sycl": {"None": (3, [None]), "F": (1, [False])},
            "is_standalone": {"F": (6, [False]), "T": (1, [True])},
        }),
    }
    surface_only = {"verbose": [False, True], "keep_intermediates": [True, False]}
    env_sem = {"TORCH_CUDA_ARCH_LIST": [None, "12.0"]}
    forced_twin = {"abspath_only": ("extra_include_paths", "rel", "abs"),
                   "drop_functions": ("functions", "f", "fg")}.get(break_mode)

    def pick(table):
        keys = list(table)
        return rng.choices(keys, weights=[table[k][0] for k in keys])[0]

    pool = []
    for _ in range(30):
        api = "load_inline" if rng.random() < 0.7 else "load"
        base = {"api": api, **{p: pick(t) for p, t in sem[api].items()},
                "env": {k: rng.choice(v) for k, v in env_sem.items()}}
        twin = {**base, "env": dict(base["env"])}
        if forced_twin and forced_twin[0] in sem[api]:
            param, base[param], twin[param] = forced_twin
        else:
            param = rng.choice(list(sem[api]))
            others = [k for k in sem[api][param] if k != base[param]]
            if others:
                twin[param] = rng.choice(others)
        pool += [base, twin]

    stats = {"calls": 0, "raised": 0, "hashed_calls": 0, "reused_hashed_calls": 0,
             "passthrough_calls": 0, "bumps": [], "per_api": {}}
    surfaces_by_name: dict = {}
    for i in range(int(n)):
        cfg = rng.choice(pool)
        api = cfg["api"]
        kw = {}
        for param, key in cfg.items():
            if param in ("api", "env"):
                continue
            v = rng.choice(sem[api][param][key][1])
            kw[param] = v() if callable(v) else v
        for param, opts in surface_only.items():
            kw[param] = rng.choice(opts)
        for k, v in cfg["env"].items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        fn = getattr(ext, api)
        params = list(inspect.signature(fn).parameters)
        args = []
        if rng.random() < 0.3:
            for p in params[: rng.randrange(1, 8)]:
                args.append(kw.pop(p))
        stats["calls"] += 1
        try:
            ret = fn(*args, **kw)
        except (AssertionError, ValueError, TypeError):
            stats["raised"] += 1              # torch's own argument errors
            continue
        stats["per_api"][api] = stats["per_api"].get(api, 0) + 1
        used = os.path.basename(str(ret))
        base_name = re.sub(r"_v\d+$", "", used)
        if _HASHED.search(base_name):
            stats["hashed_calls"] += 1
            if base_name in surfaces_by_name:
                stats["reused_hashed_calls"] += 1
            surfaces_by_name.setdefault(base_name, set()).add(repr((args, sorted(kw.items(), key=str))))
        else:
            stats["passthrough_calls"] += 1
        for name, entry in ext.JIT_EXTENSION_VERSIONER.entries.items():
            if _HASHED.search(name) and entry.version > 0:
                stats["bumps"].append({"name": name, "call": i, "api": api, "config": repr(cfg)[:600],
                                       "args": repr(args)[:300], "kwargs": repr(kw)[:900]})
        if stats["bumps"]:
            break
    stats["distinct_hashed_names"] = len(surfaces_by_name)
    stats["names_seen_with_several_surfaces"] = sum(1 for v in surfaces_by_name.values() if len(v) > 1)
    stats["max_version_any_name"] = max((e.version for e in ext.JIT_EXTENSION_VERSIONER.entries.values()),
                                        default=0)
    return stats


def _c_names() -> dict:
    """T8: effective names for a fixed table of calls (no builds)."""
    from utils import ext_naming
    tmp = _child_tmp()
    fa, fa_copy = tmp / "a.cpp", tmp / "copy_a.cpp"
    fa.write_text(_src(1))
    fa_copy.write_text(_src(1))
    S = "int f(){return 1;}\nint g(){return 2;}"
    N = ext_naming.effective_name

    def base(**over):
        kw = dict(name="km", cpp_sources=S, functions=["f"])
        kw.update(over)
        return N("load_inline", **kw)

    def with_env(k, v):
        old = os.environ.get(k)
        os.environ[k] = v
        try:
            return base()
        finally:
            if old is None:
                os.environ.pop(k)
            else:
                os.environ[k] = old

    t = {
        "base": base(),
        "cpp_list": base(cpp_sources=[S]),
        "cpp_other": base(cpp_sources=S + " "),
        "cpp_split": base(cpp_sources=S.split("\n")),
        "cuda_empty_str": base(cuda_sources=""),
        "cuda_empty_list": base(cuda_sources=[]),
        "cuda_list_of_empty": base(cuda_sources=[""]),
        "cuda_src": base(cuda_sources="void k(){}"),
        "sycl_src": base(sycl_sources="void s(){}"),
        "fn_none": base(functions=None),
        "fn_empty": base(functions=[]),
        "fn_str": base(functions="f"),
        "fn_fg": base(functions=["f", "g"]),
        "fn_gf": base(functions=["g", "f"]),
        "fn_ffg": base(functions=["f", "f", "g"]),
        "fn_dict_same": base(functions={"f": "f"}),
        "fn_dict_doc": base(functions={"f": "doc"}),
        "verbose": base(verbose=True),
        "keep_intermediates": base(keep_intermediates=False),
        "positional": N("load_inline", "km", S, None, None, ["f"]),
        "name_other": base(name="km2"),
        "extra_cflags": base(extra_cflags=["-O2"]),
        "extra_cflags_empty": base(extra_cflags=[]),
        "extra_cuda_cflags": base(extra_cuda_cflags=["-O3"]),
        "extra_cuda_cflags_ptxas": base(extra_cuda_cflags=["-O3", "-Xptxas", "-v"]),
        "cflag_moved_to_cuda": base(extra_cuda_cflags=["-O2"]),
        "extra_sycl_cflags": base(extra_sycl_cflags=["-fsycl"]),
        "ldflags_none": base(extra_ldflags=None),
        "ldflags_empty_list": base(extra_ldflags=[]),
        "ldflags_empty_item": base(extra_ldflags=[""]),
        "include_rel": base(extra_include_paths=["inc"]),
        "include_abs": base(extra_include_paths=[os.path.abspath("inc")]),
        "with_cuda_true": base(with_cuda=True),
        "with_cuda_false": base(with_cuda=False),
        "with_sycl_false": base(with_sycl=False),
        "error_handling_off": base(with_pytorch_error_handling=False),
        "no_implicit_headers": base(no_implicit_headers=True),
        "use_pch": base(use_pch=True),
        "env_cxx": with_env("CXX", "g++-14"),
        "env_cc": with_env("CC", "gcc-14"),
        "env_nvcc": with_env("PYTORCH_NVCC", "/opt/nvcc"),
        "env_arch": with_env("TORCH_CUDA_ARCH_LIST", "12.0"),
        "env_nodeps": with_env("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "1"),
        "env_max_jobs": with_env("MAX_JOBS", "3"),
        # pass-through
        "pt_build_directory": base(build_directory=str(tmp)),
        "pt_not_python_module": base(is_python_module=False),
        "pt_bad_name": base(name="bad-name"),
        "pt_pybind_literal": base(cpp_sources="PYBIND11_MODULE(my_ext, m) {}", functions=None),
        "pt_pyinit_literal": base(cpp_sources="PyMODINIT_FUNC PyInit_my_ext(void){return 0;}", functions=None),
        "pybind_macro": base(cpp_sources="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}", functions=None),
        "pyinit_km_cat": base(cpp_sources="KM_CAT(PyInit_, TORCH_EXTENSION_NAME)", functions=None),
        "pyinit_paste": base(cpp_sources="PyInit_##TORCH_EXTENSION_NAME", functions=None),
        "long_name": base(name="k" * 230),
        # load()
        "load_a": N("load", name="kl", sources=[str(fa)]),
        "load_a_str": N("load", name="kl", sources=str(fa)),
        "load_a_copy_path": N("load", name="kl", sources=[str(fa_copy)]),
        "pt_load_standalone": N("load", name="kl", sources=[str(fa)], is_standalone=True,
                                is_python_module=False),
    }
    fa.write_text(_src(2))
    t["load_a_changed"] = N("load", name="kl", sources=[str(fa)])
    return t


def _c_mutation(mode_label: str = "") -> dict:
    """T9: caller lists survive torch's in-place edits; acf's fallback retry names cleanly."""
    import torch.utils.cpp_extension as ext
    from utils import acf, ext_naming
    ext_naming.install()
    ext._run_ninja_build = lambda build_directory, verbose, error_prefix: None
    ext._import_module_from_library = lambda name, path, is_python_module: name

    cpp = ["int f(){return 1;}"]
    ld = ["-lm"]
    ext.load_inline(name="km", cpp_sources=cpp, extra_ldflags=ld, functions=["f"])
    r = {"cpp_after": cpp, "ld_after": ld}

    tmp = _child_tmp()
    ctl = tmp / "c.bin"
    ctl.write_bytes(b"controls")
    seen = _stub_builds(ext, fail_on_controls=True)
    cpp2, flags2, ld2 = ["int f(){return 1;}"], ["-O3"], ["-lm"]
    with acf.patched_extension_builds(acf=ctl, ptxas_verbose=True) as rec:
        ext.load_inline(name="km", cpp_sources=cpp2, extra_cuda_cflags=flags2, extra_ldflags=ld2,
                        functions=["f"])
    r["fallback"] = rec["fallback"]
    r["acf_attempt"], r["retry"] = seen[-2]["name"], seen[-1]["name"]
    r["retry_flags"] = seen[-1]["cuda_flags"]
    with acf.patched_extension_builds(ptxas_verbose=True):
        ext.load_inline(name="km", cpp_sources=["int f(){return 1;}"], extra_cuda_cflags=["-O3"],
                        extra_ldflags=["-lm"], functions=["f"])
    r["fresh_plain_ptxas"] = seen[-1]["name"]
    r["caller_lists_after_retry"] = [cpp2, flags2, ld2]
    return r


def _c_policy() -> dict:
    from utils import ext_naming
    out = {}
    for label, env in (("default", {}), ("ptxas_off", {"KERNELMEM_PTXAS_VERBOSE": "0"}),
                       ("naming_off", {"KERNELMEM_EXT_NAMING": "off"}),
                       ("acf", {"KERNELMEM_ACF": __file__}),
                       # an embedded preamble builds plain under this: same bytes, other binary
                       ("acf_disable", {"KERNELMEM_ACF_DISABLE": "1"}),
                       ("cxx", {"CXX": "clang++-km-test"}),
                       ("arch_list", {"TORCH_CUDA_ARCH_LIST": "9.0"}),
                       ("cuda_home", {"CUDA_HOME": "/opt/km-test-cuda"})):
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            out[label] = ext_naming.policy_id()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    out["torch_imported"] = "torch" in sys.modules
    return out


def _c_import_is_stdlib_only() -> dict:
    import utils.ext_naming  # noqa: F401
    return {"torch_imported": "torch" in sys.modules}


def _c_pybind_alternate() -> dict:
    """Slow: the functions= / torch/extension.h path generated kernels actually use."""
    from utils import ext_naming
    ext, ninja = _counting_ext()
    ext_naming.install()
    markers, names = [], []
    for i in range(4):
        mk = 1 if i % 2 == 0 else 2
        m = ext.load_inline(name="kmpb", cpp_sources=f"int km_marker(){{ return {mk}; }}",
                            functions=["km_marker"])
        markers.append(m.km_marker())
        names.append(m.__name__)
    return {"markers": markers, "names": names, "ninja": ninja, "versions": _versions(ext)}


_CHILDREN = {
    "alternate": _c_alternate, "import_one": _c_import_one, "flag_split": _c_flag_split,
    "site": _c_site, "layering": _c_layering, "acf": _c_acf, "property": _c_property,
    "names": _c_names, "mutation": _c_mutation, "policy": _c_policy,
    "stdlib_only": _c_import_is_stdlib_only, "pybind_alternate": _c_pybind_alternate,
    "acf_nvcc": _c_acf_nvcc, "preamble_leftover": _c_preamble_leftover,
    "compile_error": _c_compile_error,
}


# ==========================================================================
# tests
# ==========================================================================
def test_t1_in_process_alternation_builds_each_source_once():
    """The round-5 verdict: A,B,A,B,A,B under one name in one process."""
    tmp = _mkdtemp()
    try:
        off = tmp / "off"
        off.mkdir()
        base = _child("alternate", "6", ext_dir=off, mode="off")
        # The bug, reproduced: every alternation bumps and recompiles.
        assert base["markers"] == [1, 2, 1, 2, 1, 2], base
        assert base["names"] == ["km", "km_v1", "km_v2", "km_v3", "km_v4", "km_v5"], base["names"]
        assert len(base["ninja"]) == 6, base["ninja"]
        assert _compiles(off) == {"km": 6}, _compiles(off)
        assert _versioned_sos(off) == [f"km_v{i}.so" for i in range(1, 6)], _versioned_sos(off)

        on = tmp / "on"
        on.mkdir()
        fixed = _child("alternate", "6", ext_dir=on)
        assert fixed["installed"] is True
        assert fixed["markers"] == [1, 2, 1, 2, 1, 2], (
            f"wrong module handed back: {fixed['markers']}")
        a, b = fixed["names"][0], fixed["names"][1]
        assert a != b and all(_HASHED.search(x) for x in (a, b)), fixed["names"]
        assert fixed["names"] == [a, b] * 3, fixed["names"]
        assert len(fixed["ninja"]) == 2, fixed["ninja"]
        comp = _compiles(on)
        assert comp == {a: 1, b: 1}, comp
        assert _versioned_sos(on) == [], _versioned_sos(on)
        assert set(fixed["versions"].values()) == {0}, fixed["versions"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t2_a_second_process_reuses_the_build():
    tmp = _mkdtemp()
    try:
        ext_dir = tmp / "ext"
        ext_dir.mkdir()
        p1 = _child("import_one", "1", ext_dir=ext_dir)
        after1 = _compiles(ext_dir)
        p2 = _child("import_one", "1", ext_dir=ext_dir)
        assert p1["name"] == p2["name"] and _HASHED.search(p1["name"]), (p1["name"], p2["name"])
        assert p1["marker"] == p2["marker"] == 1
        assert _compiles(ext_dir) == after1 == {p1["name"]: 1}, (after1, _compiles(ext_dir))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t3_cross_process_thrash_is_gone():
    """preload(base) -> ncu(rejected, same name) -> nsys(base): round 6."""
    tmp = _mkdtemp()
    try:
        results = {}
        for mode in ("off", "content"):
            ext_dir = tmp / mode
            ext_dir.mkdir()
            _child("import_one", "1", ext_dir=ext_dir, mode=mode)
            _child("import_one", "2", ext_dir=ext_dir, mode=mode)
            before = sum(_compiles(ext_dir).values())
            p3 = _child("import_one", "1", ext_dir=ext_dir, mode=mode)
            assert p3["marker"] == 1, p3
            results[mode] = sum(_compiles(ext_dir).values()) - before
        assert results["off"] >= 1, f"baseline did not reproduce the thrash: {results}"
        assert results["content"] == 0, f"third process recompiled: {results}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t4_plain_and_ptxas_imports_keep_separate_builds_without_bumping():
    tmp = _mkdtemp()
    try:
        off = tmp / "off"
        off.mkdir()
        base = _child("flag_split", ext_dir=off, mode="off")
        assert max(base["versions"].values()) >= 1, f"baseline should bump on the flag flip: {base}"

        on = tmp / "on"
        on.mkdir()
        r = _child("flag_split", ext_dir=on)
        plain, ptxas = r["names"][0], r["names"][1]
        assert plain != ptxas and r["names"] == [plain, ptxas] * 2, r["names"]
        assert r["markers"] == [1, 1, 1, 1]
        assert len(r["ninja"]) == 2, r["ninja"]
        assert set(r["versions"].values()) == {0}, r["versions"]
        assert _versioned_sos(on) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t4b_bench_preload_and_profiler_sites_share_one_build():
    """_capture_import, _preload_worker and bench_ref_inputs: 1 folder, 1 compile."""
    tmp = _mkdtemp()
    try:
        ext_dir = tmp / "ext"
        ext_dir.mkdir()
        kernel = _write_kernel(tmp / "kernel_a.py", 1)
        runs = [_child("site", site, str(kernel), ext_dir=ext_dir)
                for site in ("capture_import", "preload", "bench_ref_inputs")]
        names = {r["name"] for r in runs}
        assert len(names) == 1 and _HASHED.search(next(iter(names))), [(r["site"], r["name"]) for r in runs]
        assert all(r["marker"] == 1 for r in runs)
        assert runs[0]["record"]["ptxas_verbose"] is True, runs[0]["record"]
        assert _compiles(ext_dir) == {next(iter(names)): 1}, _compiles(ext_dir)
        folders = sorted(p.name for p in ext_dir.iterdir() if p.is_dir())
        assert folders == [next(iter(names))], folders
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t5_layering_survives_every_restore_path():
    tmp = _mkdtemp()
    try:
        r = _child("layering", ext_dir=tmp / "ext")
        bad = {k: v for k, v in r.items()
               if v is not True and not (isinstance(v, list) and all(v))}
        assert not bad, f"layering checks failed: {bad}\nall: {r}"
        assert r["preamble_restored"] == [True, True]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t6_acf_paths_are_content_addressed():
    tmp = _mkdtemp()
    try:
        ext_dir = tmp / "ext"
        ext_dir.mkdir()
        r = _child("acf", ext_dir=ext_dir)
        for k in ("same_bytes_same_name", "different_bytes_different_name", "no_acf_differs",
                  "unreadable_left_unchanged", "unreadable_still_named",
                  "passthrough_flags_untouched", "flags_identical",
                  "flag_is_store_path", "store_bytes_ok"):
            assert r[k] is True, (k, r)
        assert len(set(r["names"])) == 1 and r["markers"] == [1, 1, 1], r
        assert len(r["ninja"]) == 1, r["ninja"]
        assert set(r["versions"].values()) == {0}, r["versions"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t6b_acf_store_path_is_absolute_and_keeps_torchs_arch_decision():
    """nvcc runs from inside the build folder, and torch drops its default
    -gencode set when any cuda flag contains 'arch': the ACF rewrite must hand
    nvcc a path it can open and must not change that decision versus ``off``."""
    tmp = _mkdtemp()
    try:
        runs = {mode: _child("acf_nvcc", ext_dir=tmp / mode / "ext", mode=mode)
                for mode in ("off", "content")}
        on, off = runs["content"], runs["off"]
        for case in ("relative_root", "acf_path_says_arch", "root_says_arch", "neutral"):
            a, b = on[case], off[case]
            assert a["error"] is None, (case, a)
            assert len(a["nvcc"]) == 1 and a["nvcc"][0]["acf_found"], (case, a)
            assert os.path.isabs(a["nvcc"][0]["acf"]), (case, a)
            # The same -gencode decision as the kernel's own flags get from torch.
            assert a["nvcc"][0]["gencode"] == b["nvcc"][0]["gencode"], (case, a, b)
            assert b["nvcc"][0]["acf"] == b["given"], (case, b)
        # Which path went to nvcc: depends on 'arch' in the temp path itself, so
        # only asserted where the test's own directories decide it.
        if "arch" not in on["tmp"]:
            store = f"{os.sep}{'_km_acf'}{os.sep}"
            assert store in on["relative_root"]["nvcc"][0]["acf"], on["relative_root"]
            assert store in on["neutral"]["nvcc"][0]["acf"], on["neutral"]
            assert on["neutral"]["nvcc"][0]["gencode"] is True, on["neutral"]
            for case in ("acf_path_says_arch", "root_says_arch"):
                assert on[case]["nvcc"][0]["acf"] == on[case]["given"], (case, on[case])
            assert on["acf_path_says_arch"]["nvcc"][0]["gencode"] is False
            assert on["root_says_arch"]["nvcc"][0]["gencode"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t5b_a_preamble_left_installed_never_reaches_the_next_kernel():
    tmp = _mkdtemp()
    try:
        r = _child("preamble_leftover", ext_dir=tmp / "ext", KERNELMEM_PTXAS_VERBOSE="0")
        assert r["counted_calls"] == 2, r
        # The bug, reproduced through acf's no-op patch alone:
        assert "--apply-controls" in r["bare_tuned_flags"] and r["bare_leftover"], r
        assert "--apply-controls" in r["bare_next_plain_flags"], (
            f"baseline should leak the tuned controls into the plain kernel: {r}")
        # Fixed at the harness's import helper:
        assert "--apply-controls" in r["ctx_tuned_flags"], r
        assert r["ctx_restored"] and r["ctx_restored_again"], r
        assert r["ctx_next_plain_flags"] == ["-O3"], r
        assert "--apply-controls" in r["ctx_tuned_again_flags"], r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t13_a_failed_build_shows_its_compiler_error_again():
    tmp = _mkdtemp()
    try:
        r = _child("compile_error", ext_dir=tmp / "ext")
        assert r["attempts"] == [["X", "RuntimeError", "compiler_diagnostic"], ["Y", "ok", 2],
                                 ["X", "RuntimeError", "compiler_diagnostic"],
                                 ["X", "RuntimeError", "compiler_diagnostic"]], r["attempts"]
        assert set(r["versions"].values()) == {0}, r["versions"]
        assert len(r["ninja"]) == 4, r["ninja"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _poison(build_dir: Path) -> None:
    """The end state of two processes racing on a first build (reproduced with
    16 processes and a 30 MB source): torch truncated main.cpp while the other
    process's compiler read it, so main.o holds nothing, and the rewrite landed
    with an mtime older than main.o, so ninja considers it up to date."""
    src = build_dir / "main.cpp"
    real = src.read_text()
    src.write_text("")
    subprocess.run(["ninja", "-C", str(build_dir)], check=True, capture_output=True)
    mtime = (build_dir / "main.o").stat().st_mtime_ns
    src.write_text(real)
    os.utime(src, ns=(mtime - 1000, mtime - 1000))


def test_t14_a_broken_content_named_build_heals():
    if not shutil.which("ninja"):
        _skip("ninja not on PATH")
    tmp = _mkdtemp()
    try:
        # Baseline, literal name: the broken folder stays broken until a
        # DIFFERENT same-named kernel happens to rewrite it.
        off = tmp / "off" / "ext"
        _child("import_one", "1", ext_dir=off, mode="off")
        _poison(off / "km")
        p = _child_run("import_one", "1", ext_dir=off, mode="off")
        assert p.returncode != 0 and "does not define module export function" in p.stderr, p.stderr[-800:]

        # extra_ldflags as a list: torch appends its libraries to it in place, so
        # a retry that re-sent the first attempt's lists would link differently
        # from every clean build and the NEXT process would relink again.
        args = ("1", "", "", "-lm")
        on = tmp / "on" / "ext"
        first = _child("import_one", *args, ext_dir=on)
        _poison(on / first["name"])
        p = _child_run("import_one", *args, ext_dir=on)
        healed = _parse_result("import_one", p.returncode, p.stdout, p.stderr)
        assert healed["marker"] == 1 and healed["name"] == first["name"], healed
        assert "rebuilding once" in p.stderr, p.stderr[-800:]
        assert len(healed["ninja"]) == 2, healed          # the no-op run, then the rebuild
        again = _child_run("import_one", *args, ext_dir=on)
        r = _parse_result("import_one", again.returncode, again.stdout, again.stderr)
        assert r["marker"] == 1 and "rebuilding once" not in again.stderr, again.stderr[-800:]
        assert r["ninja"] == [first["name"]], r
        # build, poison, heal -- and nothing more once healed
        assert _compiles(on) == {first["name"]: 3}, _compiles(on)
        assert _compiles(on, "{name}.so") == {first["name"]: 3}, _compiles(on, "{name}.so")
        assert _versioned_sos(on) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t7_versioner_never_bumps_a_hashed_name():
    """The correctness bar, as a property over the real load_inline/load."""
    tmp = _mkdtemp()
    try:
        seed = str(random.randrange(10**6))
        on = tmp / "on"
        on.mkdir()
        r = _child("property", "600", seed, ext_dir=on, timeout=900)
        assert not r["bumps"], f"seed={seed}: hashed name bumped: {r['bumps'][:1]}"
        assert r["calls"] == 600, r
        # Not vacuous: names really are re-used, through different surface forms.
        assert r["reused_hashed_calls"] >= 150, r
        assert r["names_seen_with_several_surfaces"] >= 15, r
        assert r["per_api"].get("load", 0) >= 30 and r["passthrough_calls"] >= 1, r

        # Teeth 1: the same stream WITHOUT naming bumps (the bug itself).
        off = tmp / "off"
        off.mkdir()
        b = _child("property", "600", seed, ext_dir=off, mode="off", timeout=900)
        assert b["max_version_any_name"] >= 1, b

        # Teeth 2: a payload that merges two real versioner inputs is caught.
        for brk in ("abspath_only", "drop_functions"):
            d = tmp / brk
            d.mkdir()
            x = _child("property", "600", seed, brk, ext_dir=d, timeout=900)
            assert x["bumps"], f"seed={seed}: break mode {brk} went undetected: {x}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t8_hash_is_a_pure_function_of_the_build_inputs():
    tmp = _mkdtemp()
    try:
        t = _child("names", ext_dir=tmp / "ext")
        t2 = _child("names", ext_dir=tmp / "ext")   # another process, another PYTHONHASHSEED
        same_keys = [k for k in t if not k.startswith("load_")]  # load_* use fresh temp paths
        diff = {k: (t[k], t2[k]) for k in same_keys if t[k] != t2[k]}
        assert not diff, f"names differ across processes: {diff}"

        b = t["base"]
        assert re.fullmatch(r"km_h[0-9a-f]{16}", b), b
        eq = ["cpp_list", "cuda_empty_str", "cuda_empty_list", "fn_str", "fn_dict_same",
              "verbose", "keep_intermediates", "positional", "extra_cflags_empty", "env_max_jobs"]
        for k in eq:
            assert t[k] == b, (k, t[k], b)
        assert t["fn_ffg"] == t["fn_fg"], "list functions dedupe in dict order, like torch"
        assert t["ldflags_none"] == t["ldflags_empty_list"] == b
        differ = ["cpp_other", "cpp_split", "cuda_list_of_empty", "cuda_src", "sycl_src", "fn_none",
                  "fn_empty", "fn_fg", "fn_dict_doc", "name_other", "extra_cflags",
                  "extra_cuda_cflags", "extra_cuda_cflags_ptxas", "extra_sycl_cflags",
                  "ldflags_empty_item", "include_rel", "include_abs", "with_cuda_true",
                  "with_cuda_false", "with_sycl_false", "error_handling_off",
                  "no_implicit_headers", "use_pch", "env_cxx", "env_cc", "env_nvcc", "env_arch",
                  "env_nodeps"]
        for k in differ:
            assert t[k] is not None and t[k] != b, (k, t[k])
        assert len({t[k] for k in differ}) == len(differ), "two different inputs share a name"
        assert t["fn_fg"] != t["fn_gf"], "functions order sets the m.def order"
        assert t["cflag_moved_to_cuda"] != t["extra_cflags"], "flag groups must not merge"
        assert t["include_rel"] != t["include_abs"], "raw include paths are a versioner input"
        for k in ("pt_build_directory", "pt_not_python_module", "pt_bad_name", "pt_pybind_literal",
                  "pt_pyinit_literal", "pt_load_standalone"):
            assert t[k] is None, (k, t[k])
        for k in ("pybind_macro", "pyinit_km_cat", "pyinit_paste"):
            assert t[k] is not None, f"{k}: macro-derived init must still be renamed"
        assert t["long_name"].startswith("k" * 180 + "_h") and len(t["long_name"]) == 180 + 18
        assert t["load_a"] == t["load_a_str"] and t["load_a"] is not None
        # Deliberately finer than torch's versioner (bytes only): sibling headers
        # included relative to the source are in neither hash, and the directory
        # stands in for them. Costs a build per exec for per-exec temp sources.
        assert t["load_a_copy_path"] != t["load_a"] and t["load_a_changed"] != t["load_a"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t9_caller_lists_are_never_mutated_and_the_fallback_retry_names_cleanly():
    tmp = _mkdtemp()
    try:
        off = tmp / "off"
        off.mkdir()
        b = _child("mutation", ext_dir=off, mode="off")
        assert b["cpp_after"] != ["int f(){return 1;}"] or b["ld_after"] != ["-lm"], (
            f"baseline should show torch mutating caller lists: {b}")

        on = tmp / "on"
        on.mkdir()
        r = _child("mutation", ext_dir=on)
        assert r["cpp_after"] == ["int f(){return 1;}"] and r["ld_after"] == ["-lm"], r
        assert r["fallback"] is True
        assert r["retry_flags"] == ["-O3", "-Xptxas", "-v"], r
        assert r["acf_attempt"] != r["retry"], r
        assert r["retry"] == r["fresh_plain_ptxas"], (
            f"the plain retry must land in the folder a clean ptxas build uses: {r}")
        assert r["caller_lists_after_retry"] == [["int f(){return 1;}"], ["-O3"], ["-lm"]], r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _concurrent(ext_dir: Path, markers: list) -> list:
    go = ext_dir.parent / f"go_{random.randrange(10**9)}"
    procs = []
    for i, mk in enumerate(markers):
        ready = ext_dir.parent / f"ready_{go.name}_{i}"
        procs.append((ready, subprocess.Popen(
            [sys.executable, __file__, "--child", "import_one", str(mk), str(go), str(ready)],
            env=_child_env(ext_dir), cwd=str(ext_dir.parent),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    deadline = time.time() + 120
    while not all(r.exists() for r, _ in procs) and time.time() < deadline:
        time.sleep(0.01)
    go.touch()
    out = []
    for _, p in procs:
        so, se = p.communicate(timeout=300)
        out.append(_parse_result("import_one", p.returncode, so, se))
    return out


def test_t10_concurrent_first_builds():
    tmp = _mkdtemp()
    try:
        ext_dir = tmp / "same"
        ext_dir.mkdir()
        res = _concurrent(ext_dir, [1, 1, 1, 1])
        assert [r["marker"] for r in res] == [1, 1, 1, 1], res
        names = {r["name"] for r in res}
        assert len(names) == 1, names
        assert _compiles(ext_dir) == {next(iter(names)): 1}, _compiles(ext_dir)

        ext_dir2 = tmp / "diff"
        ext_dir2.mkdir()
        res2 = _concurrent(ext_dir2, [1, 2])
        assert [r["marker"] for r in res2] == [1, 2], res2
        assert res2[0]["name"] != res2[1]["name"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t12_every_kernel_import_site_uses_the_build_policy():
    """Pin the wiring: one missed site brings back the flag flip for that path."""
    def body(src: str, start: str, end_marker: str = "\ndef ") -> str:
        i = src.index(start)
        j = src.find(end_marker, i + len(start))
        return src[i: j if j > 0 else len(src)]

    car = (REPO / "utils" / "compile_and_run.py").read_text()
    cap = body(car, "def _capture_import(")
    assert "ext_naming.kernel_build_context(" in cap
    assert cap.index("kernel_build_context(") < cap.index("spec.loader.exec_module(module)")
    assert "patched_extension_builds(" not in cap, "acf must be reached through the helper"

    mm = (REPO / "main_memory_latest.py").read_text()
    pre = body(mm, "def _preload_worker(")
    assert "kernel_build_context(force_verbose=False)" in pre
    assert pre.index("kernel_build_context(") < pre.index("spec.loader.exec_module(preload_mod)")
    ncu = body(mm, "def _ncu_profile_cached(")
    assert "ext_naming.policy_id()" in ncu
    assert ncu.index("ext_naming.policy_id()") < ncu.index("key = h.hexdigest()")

    bri = (REPO / "profiling" / "bench_ref_inputs.py").read_text()
    loader = body(bri, "def _load_kernel_module(")
    assert "from utils.ext_naming import kernel_build_context" in loader
    assert "except ImportError" in loader and "WARNING" in loader
    assert "with kernel_build_context(force_verbose=False):" in loader
    assert '_load_kernel_module(args.test, "bench_test_module")' in bri
    assert '_load_module(args.test' not in bri

    sct = (REPO / "utils" / "shape_coverage_test.py").read_text()
    assert sct.index("kernel_build_context(force_verbose=False)") < sct.index('_load(Path(a.kernel), "_cand")')

    ciq = (REPO / "utils" / "compileiq_finish.py").read_text()
    ev = body(ciq, "    def run(self, acf", "\n    def ")
    assert "env[ext_naming.MODE_ENV] = ext_naming.MODE_OFF" in ev
    verdict = body(ciq, "def paired_verdict(")
    assert "MODE_ENV" not in verdict and "EXT_NAMING" not in verdict, (
        "the verdict child alternates two kernels in one process; it needs content naming")

    # Every paired verdict records the build regime it was measured in.
    pb = (REPO / "utils" / "paired_bench.py").read_text()
    verdict_fn = body(pb, "def adaptive_paired_verdict(")
    assert '"build_policy": _build_policy(),' in verdict_fn
    assert "ext_naming.policy_id()" in body(pb, "def _build_policy(")

    # ncu / nsys give the profiled python the repo root, which the driver needs.
    for prof in ("ncu.py", "nsys.py"):
        s = (REPO / "profiling" / prof).read_text()
        assert 'env["PYTHONPATH"] = _repo_root' in s, prof


def test_t12_policy_id_tracks_the_environment_and_imports_no_torch():
    tmp = _mkdtemp()
    try:
        assert _child("stdlib_only", ext_dir=tmp / "ext")["torch_imported"] is False
        p = _child("policy", ext_dir=tmp / "ext")
        assert p["torch_imported"] is False
        labels = ("default", "ptxas_off", "naming_off", "acf", "acf_disable", "cxx",
                  "arch_list", "cuda_home")
        ids = [p[k] for k in labels]
        assert len(set(ids)) == len(labels), dict(zip(labels, ids))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_slow_pybind11_functions_path_alternation():
    if not (os.environ.get("KERNELMEM_SLOW_TESTS") or "--slow" in sys.argv):
        _skip("slow (compiles torch/extension.h twice); --slow or KERNELMEM_SLOW_TESTS=1")
    tmp = _mkdtemp()
    try:
        ext_dir = tmp / "ext"
        r = _child("pybind_alternate", ext_dir=ext_dir, timeout=1200)
        assert r["markers"] == [1, 2, 1, 2], r
        assert len(set(r["names"])) == 2 and len(r["ninja"]) == 2, r
        assert set(r["versions"].values()) == {0}, r
        assert sorted(_compiles(ext_dir).values()) == [1, 1], _compiles(ext_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    os.environ["_KM_DIRECT_RUN"] = "1"
    failures, skipped = [], 0
    for fn in ALL_TESTS:
        t0 = time.time()
        try:
            fn()
            print(f"ok    {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except _Skip as exc:
            skipped += 1
            print(f"skip  {fn.__name__}: {exc}", flush=True)
        except AssertionError as exc:
            failures.append(f"FAIL  {fn.__name__}: {exc}")
            print(failures[-1], flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            print(failures[-1], flush=True)

    print()
    ran = len(ALL_TESTS) - skipped
    print(f"{'FAILED' if failures else 'OK'} - {ran - len(failures)}/{ran} checks passed"
          + (f", {skipped} skipped" if skipped else ""))
    return 1 if failures else 0


def _child_main(argv: list) -> int:
    scenario, args = argv[0], argv[1:]
    result = _CHILDREN[scenario](*args)
    sys.stdout.flush()
    print("RESULT " + json.dumps(result, default=str), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(_child_main(sys.argv[2:]))
    sys.exit(main())
