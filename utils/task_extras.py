"""Attach extra scoring shapes to a generated SOL-ExecBench task.

Why this exists
---------------
``solbench_bridge.problem_to_task`` emits a task that exposes exactly one
workload through ``get_inputs()``. Scoring a candidate on one shape is not
enough, for two different reasons that both bit us on problem #2:

1. A kernel can specialise on the single scored shape and regress everywhere
   else. Run C did this -- a CUDA-graph design with a one-entry cache that won
   the scored shape and lost the suite.
2. A kernel's index arithmetic can be wrong only on shapes that do not tile
   exactly. Run D launched ``nchunk`` blocks but gave each ``ceil(HW/nchunk)``
   pixels; the two disagree unless the division is exact, and the stranded
   blocks left their partial-sum slots unwritten. Every shape scored during
   that search had ``H*W`` divisible by 16, so the search never saw it and
   shipped a kernel that failed 2 of the 20 benchmark workloads.

So the shape set has to cover SIZE (small/large, catching #1) and ALIGNMENT
(at least one extent that no tile size divides, catching #2). Picking only by
size is what left the hole.

Usage::

    python utils/task_extras.py tasks/vae_block_003.py \\
        solbench_problems/L1/003_.../ --primary-uuid <uuid>

Idempotent: re-running on a task that already has ``get_inputs_extra`` reports
what is there and changes nothing unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["choose_extra_workloads", "render_extras_block", "attach_extras"]

# Tile sizes CUDA kernels reach for. An extent divisible by none of these forces
# the guard/remainder paths that exact-tiling shapes never reach.
_TILES = (8, 16, 32, 64)
_SPATIAL_KEYS = ("height", "width", "seq_len", "length", "n", "m", "k", "size")


def _spatial_extent(axes: dict[str, Any]) -> int:
    """Product of the axes a kernel is most likely to tile over."""
    prod = 1
    for k, v in axes.items():
        if k in _SPATIAL_KEYS and isinstance(v, int) and v > 0:
            prod *= v
    return prod


def _work(axes: dict[str, Any]) -> int:
    """Rough total work, used only to order shapes small -> large."""
    prod = 1
    for v in axes.values():
        if isinstance(v, int) and v > 0:
            prod *= v
    return prod


def _is_awkward(axes: dict[str, Any]) -> bool:
    """True when the tiled extent is divisible by none of the usual tile sizes."""
    hw = _spatial_extent(axes)
    return hw > 1 and all(hw % t for t in _TILES)


def choose_extra_workloads(workloads: list[dict], primary_uuid: str | None = None,
                           n_size: int = 2, max_work_ratio: float = 6.0) -> list[dict]:
    """Pick the extra workloads to score alongside the primary one.

    Returns up to ``n_size`` size-spread shapes (smallest, and the largest that
    is still affordable) plus, if the problem has one, an AWKWARD shape whose
    tiled extent no common tile size divides. The awkward pick is not optional
    padding -- it is the only member of the set that can expose a grid/work
    rounding disagreement.

    ``max_work_ratio`` caps the big shape at N times the primary's work. Every
    extra shape is re-benchmarked for each candidate in every round, so an
    unbounded "largest" pick taxes the whole search: on problem #2 the true
    largest workload is 16x the primary and would add seconds per candidate for
    coverage that a 4-6x shape already provides. Pass ``float("inf")`` to
    genuinely take the largest.
    """
    pool = [w for w in workloads if w.get("uuid") != primary_uuid]
    if not pool:
        return []
    pool = sorted(pool, key=lambda w: _work(w["axes"]))

    primary = next((w for w in workloads if w.get("uuid") == primary_uuid), None)
    budget = _work(primary["axes"]) * max_work_ratio if primary else float("inf")

    chosen: list[dict] = []
    seen: set[str] = set()

    def take(w):
        if w is not None and w["uuid"] not in seen:
            seen.add(w["uuid"])
            chosen.append(w)

    if n_size >= 1:
        take(pool[0])                      # smallest: launch-overhead dominated
    if n_size >= 2 and len(pool) > 1:
        affordable = [w for w in pool if _work(w["axes"]) <= budget]
        # if nothing fits the budget, fall back to the cheapest above the primary
        take(affordable[-1] if affordable else pool[-1])

    # The alignment probe. Prefer the cheapest awkward shape -- it is bought for
    # coverage, not for timing signal, so cost should be minimal.
    awkward = [w for w in pool if _is_awkward(w["axes"])]
    if awkward:
        take(awkward[0])
    return chosen


def render_extras_block(chosen: list[dict], workloads: list[dict]) -> str:
    """Render the ``_WORKLOAD_EXTRA`` / ``get_inputs_extra`` source block."""
    if not chosen:
        return ""
    lines = [
        "",
        "",
        "# Additional shapes scored alongside get_inputs(). Chosen to cover two",
        "# distinct failure modes, not just a size range:",
        "#   * SIZE   - smallest and largest, so a kernel that specialises on one",
        "#              shape (e.g. CUDA-graph capture over static buffers) cannot",
        "#              win the score by fitting the primary shape alone.",
        "#   * TILING - at least one shape whose spatial extent is divisible by",
        "#              none of 8/16/32/64. Kernels routinely derive a grid from one",
        "#              rounding of a division and the per-block work from another;",
        "#              the two agree only when the tile divides the extent exactly,",
        "#              so a set of power-of-two shapes cannot see the disagreement.",
        "#              Generated by utils/task_extras.py -- keep the awkward entry.",
        "_WORKLOAD_EXTRA = [",
    ]
    for w in chosen:
        hw = _spatial_extent(w["axes"])
        tag = "awkward: tiled extent %d is divisible by none of 8/16/32/64" % hw \
            if _is_awkward(w["axes"]) else \
            "size probe: %s" % json.dumps(w["axes"])
        lines.append(f"    # {tag}")
        lines.append(f"    json.loads({json.dumps(json.dumps(w))}),")
    lines += [
        "]",
        "_WKLS_EXTRA = [Workload(**w) for w in _WORKLOAD_EXTRA]",
        "",
        "",
        "def get_inputs_extra():",
        '    """Optional KernelMem hook: extra shapes to include in the score.',
        "",
        "    Returns a list of input tuples, NOT including get_inputs(). Absent this",
        "    function KernelMem scores on get_inputs() alone.",
        '    """',
        "    return [",
        '        gen_inputs(_DEFN, w, device="cpu", custom_inputs_fn=_CUSTOM_FN)',
        "        for w in _WKLS_EXTRA",
        "    ]",
        "",
    ]
    return "\n".join(lines)


def attach_extras(task_py: Path, problem_dir: Path, *,
                  primary_uuid: str | None = None, force: bool = False) -> bool:
    """Append the extras block to *task_py*. Returns True if the file changed."""
    text = task_py.read_text(encoding="utf-8")
    if "def get_inputs_extra" in text and not force:
        print(f"[task_extras] {task_py.name} already defines get_inputs_extra; "
              f"nothing to do (use --force to regenerate)")
        return False

    wl_path = problem_dir / "workload.jsonl"
    workloads = [json.loads(l) for l in wl_path.read_text().splitlines() if l.strip()]

    if primary_uuid is None:
        # the generated task records which workload it chose
        for line in text.splitlines():
            if "uuid=" in line and "Chosen workload" in line:
                primary_uuid = line.split("uuid=")[1].split()[0]
                break

    chosen = choose_extra_workloads(workloads, primary_uuid)
    if not chosen:
        print(f"[task_extras] {wl_path} has no usable extra workloads")
        return False

    if not any(_is_awkward(w["axes"]) for w in chosen):
        print("[task_extras] WARNING: this problem has no workload whose spatial "
              "extent escapes 8/16/32/64. The score cannot detect a grid/work "
              "rounding disagreement; the allocator-poisoning check in "
              "utils/compile_and_run.py is the only remaining guard.")

    task_py.write_text(text.rstrip("\n") + "\n" + render_extras_block(chosen, workloads),
                       encoding="utf-8")
    for w in chosen:
        mark = "  <-- awkward" if _is_awkward(w["axes"]) else ""
        print(f"[task_extras] + {json.dumps(w['axes'])}{mark}")
    return True


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("task_py", type=Path, help="generated KernelMem task .py")
    p.add_argument("problem_dir", type=Path, help="SOL-ExecBench problem directory")
    p.add_argument("--primary-uuid", default=None,
                   help="uuid of the workload get_inputs() already returns")
    p.add_argument("--force", action="store_true",
                   help="regenerate even if get_inputs_extra already exists")
    a = p.parse_args()
    changed = attach_extras(a.task_py, a.problem_dir,
                            primary_uuid=a.primary_uuid, force=a.force)
    return 0 if changed or True else 1


if __name__ == "__main__":
    sys.exit(_cli())
