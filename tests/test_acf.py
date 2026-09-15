"""Checks for the compiler-knob hand-off (utils/acf.py, utils/compileiq_finish.py).

Run:  pytest tests/test_acf.py        (torch importable, no GPU, no nvcc needed)

What must hold, in order of how badly its failure would mislead a run:

* A build that fails WITH an ACF is retried without it unless strict, and the
  record says which happened. Without that, a bad control file would turn a
  good kernel into a "compile error" and send it to repair for a bug it does
  not have -- the same misreport torch_ext_cache exists to prevent.
* The injection reaches ``extra_cuda_cflags`` whether it is passed by keyword
  or positionally, and the original builder is restored on every exit path.
* ``parse_ptxas_verbose`` reads the real ``ptxas -v`` layout (entry line,
  properties line, spill line, registers line) and counts only entry kernels.
* The embedded preamble applies the controls only under the exact nvcc it was
  tuned with, falls back on failure, restores the builder after the kernel's
  last ``load_inline``, and ``strip_acf`` is its inverse.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

from utils import acf
from utils.compileiq_finish import candidate_ms, with_ext_suffix

ext = pytest.importorskip("torch.utils.cpp_extension")


# ------------------------------------------------------------- injection
class _Build:
    """A stand-in for load_inline that records its flags and can be told to fail.

    Its positional order is torch 2.11's (``sycl_sources`` at index 3, so
    ``extra_cuda_cflags`` is index 6). The old stub omitted sycl_sources, which
    is how a hard-coded index 5 in acf.py passed its positional test while
    pointing at ``extra_cflags`` in the real torch.
    """

    def __init__(self, fail_when=None):
        self.calls = []
        self.fail_when = fail_when  # predicate on extra_cuda_cflags

    def __call__(self, name, cpp_sources, cuda_sources=None, sycl_sources=None, functions=None,
                 extra_cflags=None, extra_cuda_cflags=None, **kw):
        flags = list(extra_cuda_cflags or [])
        self.calls.append({"flags": flags, "kw": kw})
        if self.fail_when and self.fail_when(flags):
            raise RuntimeError("nvcc died")
        return "module"


@pytest.fixture
def stub(monkeypatch):
    b = _Build()
    monkeypatch.setattr(ext, "load_inline", b)
    monkeypatch.setattr(ext, "load", b)
    return b


def test_no_acf_and_no_verbose_is_a_no_op(stub):
    before = ext.load_inline
    with acf.patched_extension_builds() as rec:
        assert ext.load_inline is before
    assert rec == {"acf": None, "applied": False, "fallback": False, "error": None,
                   "ptxas_verbose": False}


def test_acf_flag_is_appended_by_keyword_and_positionally(stub, tmp_path):
    f = tmp_path / "c.bin"
    f.write_bytes(b"\x00")
    with acf.patched_extension_builds(acf=f) as rec:
        ext.load_inline("k", "", "", extra_cuda_cflags=["-O3"])
        ext.load_inline("k", "", "", None, None, None, ["-O3"])  # positional index 6
    assert rec["applied"] and not rec["fallback"]
    for call in stub.calls:
        assert call["flags"] == ["-O3", "--apply-controls", str(f)]
    assert ext.load_inline is stub  # restored


def test_ptxas_verbose_adds_flags_and_forces_verbose(stub):
    with acf.patched_extension_builds(ptxas_verbose=True) as rec:
        ext.load_inline("k", "", "", extra_cuda_cflags=["-O3"], verbose=False)
    assert stub.calls[0]["flags"] == ["-O3", "-Xptxas", "-v"]
    assert stub.calls[0]["kw"]["verbose"] is True
    assert rec["ptxas_verbose"] and rec["acf"] is None


def test_force_verbose_is_independent_of_the_flags(stub):
    """Flags decide WHICH binary; verbose only whether the log streams.

    Profilers and the preload need the bench's flags (or ninja rebuilds cuda.o
    on every bench<->profile switch) but not its log.
    """
    with acf.patched_extension_builds(ptxas_verbose=True, force_verbose=False):
        ext.load_inline("k", "", "", extra_cuda_cflags=["-O3"], verbose=False)
    assert stub.calls[-1]["flags"] == ["-O3", "-Xptxas", "-v"]
    assert stub.calls[-1]["kw"]["verbose"] is False
    with acf.patched_extension_builds(force_verbose=True) as rec:
        assert ext.load_inline is not stub          # verbose alone still patches
        ext.load_inline("k", "", "", extra_cuda_cflags=["-O3"], verbose=False)
    assert stub.calls[-1]["flags"] == ["-O3"] and stub.calls[-1]["kw"]["verbose"] is True
    assert rec["ptxas_verbose"] is False
    assert ext.load_inline is stub


def test_positional_injection_matches_the_real_torch_signature(monkeypatch, tmp_path):
    """Bind against torch's own load_inline, not a stub someone has to keep in sync."""
    import functools
    import inspect

    real = ext.load_inline
    while hasattr(real, "__wrapped__"):
        real = real.__wrapped__
    seen = []

    @functools.wraps(real)
    def recorder(*args, **kwargs):
        seen.append(inspect.signature(real).bind(*args, **kwargs).arguments)
        return "module"

    monkeypatch.setattr(ext, "load_inline", recorder)
    f = tmp_path / "c.bin"
    f.write_bytes(b"\x00")
    with acf.patched_extension_builds(acf=f, ptxas_verbose=True):
        # name, cpp, cuda, sycl, functions, extra_cflags, extra_cuda_cflags, ..., verbose (11)
        ext.load_inline("k", "", "", None, None, ["-Wall"], ["-O3"], None, None, None, None, False)
    got = seen[0]
    assert got["extra_cflags"] == ["-Wall"], "nvcc flags leaked into the g++ flags"
    assert got["extra_cuda_cflags"] == ["-O3", "-Xptxas", "-v", "--apply-controls", str(f)]
    assert got["verbose"] is True


def test_failed_acf_build_falls_back_to_plain_and_says_so(monkeypatch, tmp_path):
    b = _Build(fail_when=lambda flags: "--apply-controls" in flags)
    monkeypatch.setattr(ext, "load_inline", b)
    monkeypatch.setattr(ext, "load", b)
    f = tmp_path / "c.bin"
    f.write_bytes(b"\x00")
    with acf.patched_extension_builds(acf=f, ptxas_verbose=True) as rec:
        assert ext.load_inline("k", "", "", extra_cuda_cflags=["-O3"]) == "module"
    assert rec["fallback"] and not rec["applied"]
    assert rec["error"].startswith("RuntimeError")
    assert len(b.calls) == 2
    assert "--apply-controls" in b.calls[0]["flags"]
    assert b.calls[1]["flags"] == ["-O3", "-Xptxas", "-v"]  # plain rebuild keeps the stats


def test_strict_raises_and_still_restores(monkeypatch, tmp_path):
    b = _Build(fail_when=lambda flags: "--apply-controls" in flags)
    monkeypatch.setattr(ext, "load_inline", b)
    monkeypatch.setattr(ext, "load", b)
    f = tmp_path / "c.bin"
    f.write_bytes(b"\x00")
    with pytest.raises(RuntimeError):
        with acf.patched_extension_builds(acf=f, strict=True):
            ext.load_inline("k", "", "")
    assert ext.load_inline is b
    assert len(b.calls) == 1


def test_a_missing_acf_is_an_error_not_a_silent_plain_build(stub, tmp_path):
    with pytest.raises(FileNotFoundError):
        with acf.patched_extension_builds(acf=tmp_path / "nope.bin"):
            pass


def test_environment_readers(monkeypatch):
    monkeypatch.delenv(acf.ACF_ENV, raising=False)
    monkeypatch.delenv(acf.PTXAS_VERBOSE_ENV, raising=False)
    monkeypatch.delenv(acf.ACF_STRICT_ENV, raising=False)
    assert acf.active_acf() is None and acf.ptxas_verbose_enabled() and not acf.strict_enabled()
    monkeypatch.setenv(acf.ACF_ENV, "/x/y.bin")
    monkeypatch.setenv(acf.PTXAS_VERBOSE_ENV, "0")
    monkeypatch.setenv(acf.ACF_STRICT_ENV, "1")
    assert acf.active_acf() == Path("/x/y.bin")
    assert not acf.ptxas_verbose_enabled() and acf.strict_enabled()


# ------------------------------------------------------------ ptxas -v
_LOG = """\
[1/2] nvcc ... -Xptxas -v ...
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z9big_kernelPfS_i' for 'sm_120'
ptxas info    : Function properties for _Z9big_kernelPfS_i
    0 bytes stack frame, 128 bytes spill stores, 96 bytes spill loads
ptxas info    : Used 255 registers, used 1 barriers, 17024 bytes smem
ptxas info    : Function properties for _Z6helperPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Compiling entry function '_Z10tiny_kernelPf' for 'sm_120'
ptxas info    : Function properties for _Z10tiny_kernelPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 0 barriers
"""


def test_parse_ptxas_verbose_reads_the_real_layout():
    px = acf.parse_ptxas_verbose(_LOG)
    assert px["n_kernels"] == 2                      # the helper has no registers line
    assert px["spill_bytes"] == 224 and px["max_registers"] == 255
    assert px["spilling"] == ["_Z9big_kernelPfS_i"]
    big, tiny = px["kernels"]
    assert big["arch"] == "sm_120" and big["smem"] == 17024 and big["barriers"] == 1
    assert tiny["registers"] == 22 and tiny["smem"] is None


def test_parse_ptxas_verbose_on_a_plain_log_is_empty():
    px = acf.parse_ptxas_verbose("ninja: no work to do.\n")
    assert px == {"kernels": [], "n_kernels": 0, "spill_bytes": 0, "max_registers": None, "spilling": []}


def test_build_summary_only_parses_when_verbose_was_on():
    assert acf.build_summary({"ptxas_verbose": False}, _LOG)["ptxas"] is None
    assert acf.build_summary({"ptxas_verbose": True}, _LOG)["ptxas"]["n_kernels"] == 2
    assert acf.build_summary(None, _LOG) == {"acf": None, "ptxas": None}


def test_short_kernel_name_demangles_the_head():
    assert acf.short_kernel_name("_Z10big_kernelPfS_i") == "big_kernel"
    assert acf.short_kernel_name("_ZN7cutlass6KernelI" + "x" * 500).startswith("cutlass")
    assert acf.short_kernel_name("plain") == "plain"


# ------------------------------------------------------------- embedding
_KERNEL = '''\
import torch
from torch.utils.cpp_extension import load_inline

_ext = load_inline(
    name="my_ext",
    cpp_sources="",
    cuda_sources="",
    extra_cuda_cflags=["-O3"],
)


class ModelNew(torch.nn.Module):
    pass
'''


def _exec_embedded(src: str, monkeypatch, builder, nvcc: str):
    """Run an embedded kernel file with the builder and nvcc version stubbed."""
    monkeypatch.setattr(ext, "load_inline", builder)
    import subprocess

    class _P:
        stdout = f"Cuda compilation tools, release 13.3, V{nvcc}\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _P())
    ns: dict = {"__name__": "embedded_kernel"}
    exec(compile(src, "embedded_kernel.py", "exec"), ns)
    return ns


def test_embed_applies_only_under_the_exact_nvcc(monkeypatch):
    payload = b"\x01\x02\x03controls"
    src = acf.embed_acf(_KERNEL, payload, nvcc="13.3.33")
    assert base64.b64encode(payload).decode() in src
    assert src.endswith(_KERNEL)

    b = _Build()
    _exec_embedded(src, monkeypatch, b, nvcc="13.3.33")
    flags = b.calls[0]["flags"]
    assert flags[:2] == ["-O3", "--apply-controls"] and flags[2].endswith(".acf.bin")
    assert Path(flags[2]).read_bytes() == payload
    assert ext.load_inline is b  # restored after the kernel's single call

    # A positional call (index 6 in torch 2.11) gets the controls in the nvcc flags.
    positional = src.replace('cpp_sources="",\n    cuda_sources="",\n    extra_cuda_cflags=["-O3"],',
                             '"", "", None, None, ["-Wall"], ["-O3"],')
    positional = positional.replace('name="my_ext",', '"my_ext",')
    assert '"", "", None, None, ["-Wall"], ["-O3"],' in positional
    bp = _Build()
    _exec_embedded(positional, monkeypatch, bp, nvcc="13.3.33")
    assert bp.calls[0]["flags"][:2] == ["-O3", "--apply-controls"]

    b2 = _Build()
    _exec_embedded(src, monkeypatch, b2, nvcc="13.4.0")
    assert b2.calls[0]["flags"] == ["-O3"]  # different build: plain, no controls


def test_embed_falls_back_when_the_controlled_build_fails(monkeypatch):
    src = acf.embed_acf(_KERNEL, b"\x00", nvcc="13.3.33")
    b = _Build(fail_when=lambda flags: "--apply-controls" in flags)
    _exec_embedded(src, monkeypatch, b, nvcc="13.3.33")
    assert len(b.calls) == 2 and b.calls[1]["flags"] == ["-O3"]


def test_embed_honours_the_disable_switch(monkeypatch):
    src = acf.embed_acf(_KERNEL, b"\x00", nvcc="13.3.33")
    monkeypatch.setenv(acf.ACF_DISABLE_ENV, "1")
    b = _Build()
    _exec_embedded(src, monkeypatch, b, nvcc="13.3.33")
    assert b.calls[0]["flags"] == ["-O3"]


def test_embed_refuses_bad_inputs():
    with pytest.raises(ValueError):
        acf.embed_acf("import torch\n", b"\x00", nvcc="13.3.33")     # no load_inline
    with pytest.raises(ValueError):
        acf.embed_acf(_KERNEL, b"\x00", nvcc="")                       # no version to guard on
    once = acf.embed_acf(_KERNEL, b"\x00", nvcc="13.3.33")
    with pytest.raises(ValueError):
        acf.embed_acf(once, b"\x00", nvcc="13.3.33")                   # already embedded


def test_strip_acf_is_the_inverse_of_embed():
    src = acf.embed_acf(_KERNEL, b"\x00\xff", nvcc="13.3.33")
    assert acf.strip_acf(src) == _KERNEL
    assert acf.strip_acf(_KERNEL) == _KERNEL


def test_embedded_file_is_still_packageable():
    src = acf.embed_acf(_KERNEL, b"\x00", nvcc="13.3.33")
    assert "ModelNew" in src and "def run(" not in src   # scripts/package_solution.py's two checks


# ------------------------------------------------------ finishing-pass helpers
def test_with_ext_suffix_renames_only_the_extension():
    out = with_ext_suffix(_KERNEL, "_acf")
    assert 'name="my_ext_acf"' in out and out.count("my_ext") == 1
    assert with_ext_suffix("no extension here", "_acf") == "no extension here"
    # the renamed file embeds and still counts one load_inline call
    assert acf.count_load_inline_calls(out) == 1


def test_candidate_ms_ranks_on_absolute_time_geomean():
    res = {"test_latency_ms": {"avg": 9.0},
           "per_shape": [{"test_ms": 1.0}, {"test_ms": 4.0}, {"test_ms": None}]}
    assert candidate_ms(res) == pytest.approx(2.0)          # geomean of 1 and 4
    assert candidate_ms({"test_latency_ms": {"avg": 9.0}}) == 9.0
    assert candidate_ms(None) is None and candidate_ms({}) is None
