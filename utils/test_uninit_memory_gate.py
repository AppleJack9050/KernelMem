"""Regression test: the correctness gate must reject kernels that read
uninitialized GPU memory.

Background
----------
Run D produced a kernel that scored 1.1627 in the search and then failed 2 of
the 20 SOL-ExecBench workloads. Its GroupNorm launched ``nchunk`` blocks but
gave each ``ppc = ceil(HW / nchunk)`` pixels; the two quantities are derived
from each other with opposite rounding, so the trailing blocks can end up with
no pixels left. Those blocks return without storing their partial-sum slot, and
``gn_finalize_kernel`` sums every slot -- including the ones nobody wrote, out
of an ``at::empty()`` buffer -- into each group's mean and variance.

There are two independent ways for the two to disagree, which is why the shapes
below are both tested:

* ``nchunk = HW / 16`` floors, so 16 not dividing HW loses pixels  -> 1x131x131.
* ``nchunk`` is then clamped to ``2048 / bn``, and the clamp re-creates the same
  disagreement at a different granularity -> 1x293x293, where ppc becomes 42 and
  3 blocks are stranded. Repairing only the ``HW / 16`` rounding does NOT fix
  this one; nchunk has to be derived back from ppc.

The search never saw any of it for two compounding reasons, and this file pins
the fix for both:

1. Every shape scored during the search had ``HW % 16 == 0``, so the arithmetic
   happened to line up. Guarded by ``test_extras_include_an_awkward_shape``.
2. Even on a bad shape a single allclose can pass, because the buffer it reads
   happens to hold harmless values. Guarded by the poisoning tests, which use
   the run D kernel itself as a fixture.

Run directly::

    python utils/test_uninit_memory_gate.py

Requires a CUDA device and compiles two CUDA extensions, so it is slow on a
cold cache (a few minutes) and near-instant afterwards.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.compile_and_run import (  # noqa: E402
    _allocator_dependence_note,
    _fresh_allocator,
    _poison_allocator,
    _shape_tag,
)

_TESTDATA = Path(__file__).resolve().parent / "testdata"
_BUGGY = _TESTDATA / "uninit_partials_kernel.py"
_FIXED = _TESTDATA / "uninit_partials_kernel_fixed.py"

_TOL = 2.8e-3            # SOL-ExecBench max_atol for this problem
# HW = 17161, odd. nchunk = HW/16 = 1072 stays under the 2048/bn clamp, so this
# shape exercises the rounding bug on its own.
_AWKWARD = (1, 131, 131)
# HW = 85849. nchunk would be 5365, so the clamp binds and pins it at 2048,
# giving ppc = 42 and 3 stranded blocks. Deriving nchunk from ppc is the only
# fix that covers this; repairing the HW/16 rounding alone does not.
_CLAMPED = (1, 293, 293)
_ALIGNED = (2, 64, 64)   # HW = 4096, tiles exactly -> must never be flagged


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inputs(task, b: int, h: int, w: int):
    inp = [x.cuda() if torch.is_tensor(x) else x for x in task.get_inputs()]
    inp[0] = torch.randn(b, 256, h, w, device="cuda")
    return inp


def _note_for(model, inp, dev, tol=_TOL):
    """Run the model on a zero-filled pool, then ask the gate if it reproduces."""
    _fresh_allocator(dev)
    with torch.no_grad():
        clean = model(*inp).contiguous()
    return _allocator_dependence_note(model, inp, dev, clean, tol, _shape_tag(inp)), clean


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_poison_allocator_reports_success(dev):
    assert _poison_allocator(dev), "poisoning did nothing - the gate would be inert"
    print("PASS  _poison_allocator fills the pool")


def test_buggy_kernel_is_caught_on_awkward_shape(task, dev):
    """The whole point: run D's kernel matched the reference on a clean
    allocator for 1x131x131, so a plain allclose let it through."""
    mod = _load(_BUGGY, "uninit_buggy")
    model = mod.ModelNew().cuda().eval()
    ref = task.Model().cuda().eval()
    for shape in (_AWKWARD, _CLAMPED):
        inp = _inputs(task, *shape)
        with torch.no_grad():
            expected = ref(*inp)
        note, clean = _note_for(model, inp, dev)
        passed_clean = torch.allclose(expected, clean, atol=_TOL, rtol=1e-5)
        assert note is not None, (
            f"gate FAILED to flag the known-bad kernel on {_shape_tag(inp)} "
            f"(plain allclose verdict was {'pass' if passed_clean else 'fail'})"
        )
        assert "UNINITIALIZED GPU MEMORY" in note
        print(f"PASS  buggy kernel flagged on {_shape_tag(inp)} "
              f"(plain allclose alone said: "
              f"{'PASS - would have shipped' if passed_clean else 'fail'})")


def test_buggy_kernel_is_clean_on_aligned_shape(task, dev):
    """HW % 16 == 0 leaves no stranded block, so the same kernel is genuinely
    deterministic here. Flagging it would be a false positive."""
    mod = _load(_BUGGY, "uninit_buggy")
    model = mod.ModelNew().cuda().eval()
    inp = _inputs(task, *_ALIGNED)
    note, _ = _note_for(model, inp, dev)
    assert note is None, f"false positive on well-tiled shape {_shape_tag(inp)}:\n{note}"
    print(f"PASS  no false positive on {_shape_tag(inp)} (same kernel, HW % 16 == 0)")


def test_fixed_kernel_passes_everywhere(task, dev):
    """No false positives once the producer/consumer agree on the block count."""
    mod = _load(_FIXED, "uninit_fixed")
    model = mod.ModelNew().cuda().eval()
    ref = task.Model().cuda().eval()
    for shape in (_AWKWARD, _CLAMPED, _ALIGNED):
        inp = _inputs(task, *shape)
        with torch.no_grad():
            expected = ref(*inp)
        note, clean = _note_for(model, inp, dev)
        assert note is None, f"false positive on fixed kernel at {_shape_tag(inp)}:\n{note}"
        assert torch.allclose(expected, clean, atol=_TOL, rtol=1e-5), \
            f"fixed kernel is numerically wrong at {_shape_tag(inp)}"
        print(f"PASS  fixed kernel clean and correct on {_shape_tag(inp)}")


def test_extras_include_an_awkward_shape(task):
    """The task must score at least one shape that no tile size divides."""
    extras = task.get_inputs_extra()
    hws = []
    for e in extras:
        t = next(x for x in e if torch.is_tensor(x) and x.dim() == 4)
        hws.append(t.shape[2] * t.shape[3])
    awkward = [hw for hw in hws if any(hw % tile for tile in (8, 16, 32, 64))]
    assert awkward, (
        f"every extra shape tiles exactly (HW = {hws}); the score cannot see a "
        f"grid/workload rounding disagreement. Add a shape with odd H or W."
    )
    print(f"PASS  extras include an awkward shape (HW = {awkward}, all HW = {hws})")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP  no CUDA device")
        return 0
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    task = _load(_ROOT / "tasks" / "vae_block_002.py", "uninit_task")

    failures = []
    for fn, args in (
        (test_extras_include_an_awkward_shape, (task,)),
        (test_poison_allocator_reports_success, (dev,)),
        (test_buggy_kernel_is_caught_on_awkward_shape, (task, dev)),
        (test_buggy_kernel_is_clean_on_aligned_shape, (task, dev)),
        (test_fixed_kernel_passes_everywhere, (task, dev)),
    ):
        try:
            fn(*args)
        except AssertionError as exc:
            failures.append(f"FAIL  {fn.__name__}: {exc}")
            print(failures[-1])

    print()
    print(f"{'FAILED' if failures else 'OK'} - "
          f"{5 - len(failures)}/5 checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
