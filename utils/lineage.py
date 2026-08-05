"""Lineage planning: what to fund, and when to stop funding it.

Why this exists
---------------
The round loop is a hill-climber over a landscape the SEED chooses, and the
choice is made once, at round 0, and never revisited. Measured on vae_block_002:
the seed scored 1.1253 and 21 rounds of optimization took it to 1.2039, so the
seed decided 93.5% of the result and the entire optimization budget decided 7%.
Meanwhile the spread BETWEEN seed draws was 10-41% -- larger than everything the
loop delivers. Budget spent on rounds is therefore spent on the small term.

A lineage is one such hill: a granularity plus an algorithm, with its own base
kernel and its own ratchet. Running several means the seed decision stops being
a one-shot bet. Two rules keep that from becoming an excuse to burn tokens:

* fund by CEILING -- do not start a lineage whose best conceivable outcome
  cannot beat what you already have (`ceilings`);
* stop by TRAJECTORY -- kill a funded lineage once its own score history says
  it will not arrive in the rounds remaining (`trajectory_verdict`).

Both are computed from data the run already collects, so neither needs a model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Operation-count ratio versus a direct/implicit-GEMM convolution. Winograd
# F(m x m, r x r) needs (m+r-1)^2 multiplies where direct needs (m*r)^2, so
# F(2x2,3x3) is 36/16 = 2.25. The op-count part is an EXACT arithmetic fact.
#
# But an op-count ratio is only a usable SPEED ceiling if the cheaper algorithm
# runs on the SAME execution units. On fp32 conv it does not, and the gap is
# large enough to invert the conclusion. Measured on RTX 5090 (sm_120), 2026-08-05:
#
#   * TF32 Winograd is INADMISSIBLE when the harness reference is cuDNN TF32.
#     End-to-end, F(2x2,3x3)+TF32 gives 7.4e-3 .. 1.02e-2 against the reference
#     and FAILS the 0.01 budget on 4x256x256x256; F(4x4,3x3)+TF32 gives ~9e-2,
#     ~10x over. The output transform amplifies the low-precision rounding
#     injected in the Winograd domain (A^T entries reach 8).
#   * fp32 Winograd IS accurate (1.1e-4, same as direct fp32) but cannot use the
#     tensor cores at all -- no fp32 MMA instruction exists. It runs on the CUDA
#     cores at 62.7 TFLOP/s achievable versus 105.9 for TF32.
#
#   => the realisable ratio for fp32 Winograd is 2.25 * (62.7/105.9) ~= 1.33,
#      not 2.25. Quoting 2.25 funded a lineage on a ceiling of 2.9x that no
#      admissible implementation can reach, while own_gemm (which DOES work) was
#      refused. Ratios below are effective, i.e. op-count x execution-unit rate.
# Revisit per GPU: on a card where TF32 is much faster than fp32 the derate is
# harsher (H100 is ~3.7x), and on a benchmark whose reference is NOT TF32 the
# admissibility objection disappears and the raw op-count ratio applies.
ALGORITHM_WORK_RATIO: Dict[str, float] = {
    "implicit_gemm": 1.0,      # op-count parity with the vendor; see ceilings()
                               # -- parity does NOT mean parity in time, because
                               # owning the op deletes the vendor's satellite
                               # convertTensor/transform passes.
    "winograd_f2x3": 1.33,     # 2.25 op-count x (62.7/105.9) fp32-vs-TF32 derate
    "winograd_f4x3": 2.37,     # 4.0 op-count x same derate; TF32 form is
                               # inadmissible on accuracy, see memorybank
}


@dataclass
class LineageSpec:
    """One hill to climb."""
    id: str
    granularity: str                    # A | B | C | D
    algorithm: str = "implicit_gemm"    # key into ALGORITHM_WORK_RATIO
    owns_vendor_op: bool = False        # True => the vendor GEMM/conv is replaced
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.id}({self.granularity}/{self.algorithm})"


@dataclass
class LineageState:
    """Live state of a funded lineage. Scores are read from its checkpoint."""
    spec: LineageSpec
    subproc_id: int
    batch_dir: Path
    ceiling: float
    scores: List[float] = field(default_factory=list)
    rounds_done: int = 0
    status: str = "pending"             # pending|running|done|killed|unfunded
    reason: str = ""
    best: Optional[float] = None


def ceilings(total_us: float, vendor_us: float, spec: LineageSpec,
             residual_us: float = 0.0, vendor_overhead_us: float = 0.0) -> float:
    """Best speedup this lineage could ever reach. An UPPER bound, deliberately.

    Upper bounds are the right tool for a funding gate: if even the bound loses
    to the incumbent, no amount of tuning inside that lineage can win, and the
    decision is safe without predicting how well anything will be implemented.

    Keeping the vendor operator caps you at Amdahl -- ``total/vendor`` -- reached
    only if every other kernel becomes free.

    Replacing it with the SAME algorithm used to cap at that same number, on the
    reasoning that "matching the vendor is the best case". That is false, and it
    cost a real opportunity: `vendor_us` is not all mainloop. cuDNN's TF32 conv
    also launches a separate fp32->TF32 convertTensor pass -- a full read+write
    of the input, measured at ~17% of the conv's own time on vae_block_002 --
    plus layout transforms and casts. A same-algorithm CUTLASS conv converts
    in-register in the mainloop and deletes that pass outright, measuring
    1.04-1.13x faster than cuDNN across the four scored shapes WHILE ITS GEMM IS
    ~3% SLOWER. It wins on memory traffic, not arithmetic.

    So owning the operator can beat the vendor at algorithmic parity, and
    ``ALGORITHM_WORK_RATIO`` is no longer the sole term that raises the bound.
    Pass *vendor_overhead_us* -- the satellite kernels attributable to the vendor
    op (convertTensor / transform / im2col / cast) -- and it is removed from the
    floor when the lineage owns the op. Leave it 0 and behaviour is unchanged.

    *residual_us* is work that survives even perfect fusion (e.g. a final
    elementwise pass that must read and write the tensor). Pass it when known --
    it tightens the bound and prevents funding a lineage on an Amdahl figure
    that ignores unavoidable traffic.
    """
    if vendor_us <= 0 or total_us <= 0:
        return float("inf")
    ratio = ALGORITHM_WORK_RATIO.get(spec.algorithm, 1.0)
    # NOTE: vendor_overhead_us is measured SEPARATELY from vendor_us (the marker
    # lists in utils.reference_profile are disjoint), so it is added to the
    # keep-the-vendor floor rather than subtracted from the owning one.
    overhead = max(0.0, vendor_overhead_us)
    if not spec.owns_vendor_op:
        # Vendor kernel retained: mainloop AND its satellites are both fixed.
        floor = vendor_us + overhead + residual_us
    else:
        # Owning the op deletes the satellites even at algorithmic parity; only
        # the mainloop is subject to ALGORITHM_WORK_RATIO.
        floor = (vendor_us / max(ratio, 1e-9)) + residual_us
    return total_us / max(floor, 1e-9)


def fund(spec_ceiling: float, incumbent: Optional[float], min_gain: float = 0.10
         ) -> tuple[bool, str]:
    """Should this lineage be started at all?

    ``min_gain`` exists because a ceiling that only just clears the incumbent is
    not worth a lineage: the bound is optimistic by construction, so a lineage
    must promise real headroom, not a rounding error. On vae_block_002 the
    keep-the-vendor-conv lineage bounds at 1.35x against an incumbent of 1.2039
    (+12%) yet had already stalled at 1.2039 in practice, because the bound
    ignores the ~250us of GroupNorm traffic that cannot be fused away while the
    vendor kernel must read its input from memory. Pass that as ``residual_us``
    to ``ceilings`` and the same lineage correctly fails this gate.
    """
    if incumbent is None or incumbent <= 0:
        return True, "no incumbent to beat"
    need = incumbent * (1.0 + min_gain)
    if spec_ceiling >= need:
        return True, f"ceiling {spec_ceiling:.2f}x >= {need:.2f}x required"
    return False, (f"ceiling {spec_ceiling:.2f}x cannot reach {need:.2f}x "
                   f"(incumbent {incumbent:.4f} + {min_gain*100:.0f}%)")


def trajectory_verdict(scores: List[float], target: float, rounds_left: int,
                       grace: int = 5) -> tuple[bool, str]:
    """Keep climbing, or cut this lineage loose?

    Returns (keep, reason). A lineage is judged on ITS OWN history -- never
    against the global best -- because that is the entire point of separating
    lineages: a structural change is worse before it is tuned, and comparing it
    to the incumbent would kill it on evaluation one. The current granularity-D
    seeds sit at 0.46-0.64 against an incumbent of 1.2039 and would all die
    instantly under a global comparison, yet their ceiling is higher.

    Inside the grace window nothing is judged. After it, the lineage's recent
    per-round improvement is extrapolated over the rounds remaining; if even
    that optimistic line cannot reach *target*, further rounds are spending
    budget on an outcome the lineage's own evidence has already ruled out.
    """
    # scores[0] is the SEED, not an optimization round: a lineage that has run
    # r rounds has r+1 entries here. Gating on len(scores) therefore spent one
    # of the grace rounds on the seed draw, and --grace 5 bought only four
    # rounds of tuning -- 20% less room than asked for, on the one mechanism
    # whose entire purpose is to give a structural change room before judging
    # it. The printed "(n/grace rounds)" was off by the same one.
    n = len(scores)
    opt_rounds = max(0, n - 1)
    if opt_rounds < max(1, grace):
        return True, f"within grace ({opt_rounds}/{grace} rounds)"
    if rounds_left <= 0:
        return False, "no rounds left"

    best = max(scores)
    if best >= target:
        return True, f"already at {best:.4f} >= target {target:.4f}"

    # Rate over the grace window, using the best-so-far curve rather than raw
    # per-round scores: raw scores swing on rejected candidates, and this asks
    # "is the lineage still finding things", not "was the last round lucky".
    running = []
    cur = -float("inf")
    for s in scores:
        cur = max(cur, s)
        running.append(cur)
    window = running[-grace:]
    gain_per_round = (window[-1] - window[0]) / max(len(window) - 1, 1)

    if gain_per_round <= 0:
        return False, (f"no improvement over the last {len(window)} rounds "
                       f"(best {best:.4f}, target {target:.4f})")

    projected = best + gain_per_round * rounds_left
    if projected < target:
        return False, (f"projected {projected:.4f} after {rounds_left} more rounds "
                       f"(+{gain_per_round*100:.2f}%/round from {best:.4f}) "
                       f"falls short of {target:.4f}")
    return True, (f"projected {projected:.4f} >= target {target:.4f} "
                  f"(+{gain_per_round*100:.2f}%/round)")


def read_lineage_progress(batch_dir: Path, task_stem: str) -> tuple[List[float], int]:
    """Scores and completed-round count for a running lineage, from its checkpoint.

    Reads rather than communicates: the child writes checkpoint.json at every
    round boundary already, so the coordinator needs no IPC and a child that
    dies mid-round still leaves a readable history.
    """
    ck = batch_dir / task_stem / "checkpoint.json"
    if not ck.exists():
        return [], 0
    try:
        d = json.loads(ck.read_text(encoding="utf-8"))
    except Exception:
        return [], 0
    scores = [float(s) for s in (d.get("scores") or []) if isinstance(s, (int, float))]
    return scores, int(d.get("next_round") or 0)


def read_stop_reason(batch_dir: Path, task_stem: str) -> Optional[str]:
    """Why the child stopped, as the child itself recorded it.

    A child that stops on its own plateau rule exits 0, exactly like one that
    ran every round -- so the coordinator cannot tell "gave up at round 5" from
    "completed 10" by exit status, and the summary reports both as
    ``exited rc=0``. That distinction is the whole point of comparing lineages
    afterward, so read the reason the child already wrote down.
    """
    ck = batch_dir / task_stem / "checkpoint.json"
    if not ck.exists():
        return None
    try:
        return json.loads(ck.read_text(encoding="utf-8")).get("stop_reason")
    except Exception:
        return None


def vendor_split(profile: Dict[str, Any]) -> tuple[float, float]:
    """(total_us, vendor_gemm_us) from a utils.reference_profile payload."""
    from utils.reference_profile import _is_vendor_gemm
    total = float(profile.get("total_us") or 0.0)
    vendor = sum(float(k.get("us") or 0.0) for k in (profile.get("kernels") or [])
                 if _is_vendor_gemm(str(k.get("name") or "")))
    return total, vendor


def vendor_overhead(profile: Dict[str, Any]) -> float:
    """Time in vendor-internal satellite kernels (convertTensor, transforms).

    Disjoint from `vendor_split`'s GEMM total. Feed it to `ceilings` as
    `vendor_overhead_us`: it is fixed cost for a keep-the-vendor lineage and
    free to delete for a lineage that owns the operator.
    """
    from utils.reference_profile import _is_vendor_satellite
    return sum(float(k.get("us") or 0.0) for k in (profile.get("kernels") or [])
               if _is_vendor_satellite(str(k.get("name") or "")))
