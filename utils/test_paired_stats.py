#!/usr/bin/env python
"""Checks for the paired-verdict statistics.

Run:  python -m utils.test_paired_stats

These cover the two things the ratchet's correctness rests on:

* ``_t_sf`` is a real Student-t tail, not an approximation that happens to look
  right near t=3. It is stdlib-only so the benchmark subprocess never needs
  scipy, which means nothing else validates it -- so this does, against scipy
  when scipy is importable, and against closed forms when it is not.
* ``_paired_stats`` recovers a known effect from synthetic data with an injected
  shared drift, which is the exact situation the interleaving creates and the
  reason an unpaired Welch is the wrong estimator for it.
"""
from __future__ import annotations

import math
import statistics as st

from utils.paired_bench import (_p_to_sigma, _paired_stats, _sigma_to_p, _t_sf,
                                _welch)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_t_sf_against_scipy() -> None:
    """The stdlib tail must match scipy everywhere it is asked about."""
    try:
        from scipy import stats as sps
    except ImportError:
        print("SKIP  scipy not installed; t-tail checked against closed forms only")
        return
    worst = 0.0
    for dof in range(1, 31):
        for t in (-40.0, -19.2, -6.6, -3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 4.5, 6.6, 19.2, 40.0):
            mine, theirs = _t_sf(t, dof), float(sps.t.sf(t, dof))
            worst = max(worst, abs(mine - theirs))
    _check(worst < 1e-12, f"_t_sf disagrees with scipy by {worst:.2e}")
    print(f"PASS  _t_sf matches scipy.stats.t.sf to {worst:.1e} over dof 1-30")


def test_t_sf_closed_forms() -> None:
    """Cauchy (dof=1) and dof=2 have exact tails; symmetry and limits must hold."""
    for dof in (1, 2, 5, 30):
        _check(abs(_t_sf(0.0, dof) - 0.5) < 1e-12, f"tail at t=0 must be 0.5 (dof={dof})")
        for t in (0.7, 2.0, 5.0):
            _check(abs(_t_sf(t, dof) + _t_sf(-t, dof) - 1.0) < 1e-12,
                   f"tail must be symmetric (dof={dof}, t={t})")
    for t in (0.5, 1.0, 3.0):
        _check(abs(_t_sf(t, 1) - (0.5 - math.atan(t) / math.pi)) < 1e-12,
               "dof=1 must equal the Cauchy tail")
        _check(abs(_t_sf(t, 2) - 0.5 * (1.0 - t / math.sqrt(2.0 + t * t))) < 1e-12,
               "dof=2 must equal its closed form")
    # Large dof approaches the normal.
    _check(abs(_t_sf(3.0, 100000) - _sigma_to_p(3.0)) < 1e-6,
           "t must approach the normal as dof grows")
    _check(math.isinf(_t_sf(float("inf"), 5)) is False and _t_sf(float("inf"), 5) == 0.0,
           "t=+inf must give a zero tail")
    print("PASS  _t_sf matches closed forms at dof 1 and 2, is symmetric, and "
          "approaches the normal")


def test_sigma_roundtrip() -> None:
    for s in (1.0, 2.0, 3.0, 5.0):
        _check(abs(_p_to_sigma(_sigma_to_p(s)) - s) < 1e-9, f"sigma roundtrip failed at {s}")
    _check(abs(_sigma_to_p(3.0) - 0.001349898) < 1e-9, "3 sigma must be p=0.00135")
    print("PASS  sigma<->p roundtrip, and 3 sigma is p=0.00135")


def test_paired_recovers_effect_under_shared_drift() -> None:
    """The case the interleaving is built for: a real effect plus common drift.

    Base and candidate differ by a known 2%, and every rep is multiplied by a
    shared factor that ramps 1.00 -> 1.06 across the session. Pairing must
    recover the 2% with a small error bar; Welch must recover the same 2% (it is
    unbiased too) but with a far larger one, because it charges the shared ramp
    to the two kernels separately.
    """
    drift = [1.00, 1.015, 1.03, 1.045, 1.06]
    base = [1.0000 * d for d in drift]
    cand = [1.0000 / 1.02 * d for d in drift]      # candidate is exactly 2% faster

    s = _paired_stats(base, cand)
    _check(abs(s["rel_pct"] - 2.0) < 1e-9, f"paired effect {s['rel_pct']} != 2.0")
    _check(s["se_pct"] < 1e-9, f"shared drift must cancel, got se={s['se_pct']}")
    _check(s["dof"] == len(base) - 1, "dof must be n-1")

    w = _welch([1.0 / t for t in cand], [1.0 / t for t in base])
    se_welch = w["se"] / st.mean([1.0 / t for t in base]) * 100.0
    _check(se_welch > 1.0, f"Welch should carry the drift, got se={se_welch}")
    print(f"PASS  paired recovers +2.00% with se={s['se_pct']:.2e}% where Welch "
          f"reports se={se_welch:.2f}%")


def test_paired_sign_convention() -> None:
    """rel_pct > 0 must mean the CANDIDATE is faster (took less time)."""
    s = _paired_stats([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    _check(s["rel_pct"] > 0, "a candidate at half the time must read positive")
    _check(abs(s["rel_pct"] - 100.0) < 1e-9, f"2x faster must read +100%, got {s['rel_pct']}")
    s = _paired_stats([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    _check(abs(s["rel_pct"] - (-50.0)) < 1e-9, f"2x slower must read -50%, got {s['rel_pct']}")
    print("PASS  sign convention: positive rel_pct means the candidate is faster")


def test_zero_spread_is_infinite_t() -> None:
    s = _paired_stats([1.0, 1.0, 1.0], [0.9, 0.9, 0.9])
    _check(s["se_log"] == 0.0, "identical ratios must give zero spread")
    _check(math.isinf(s["t"]) and s["t"] > 0, "zero spread with a gain must give +inf t")
    s = _paired_stats([1.0, 1.0], [1.0, 1.0])
    _check(s["t"] == 0.0, "no difference and no spread must give t=0, not inf")
    print("PASS  zero-spread cases give infinite / zero t rather than dividing by zero")


def test_dof_is_what_gates_small_samples() -> None:
    """The finding that motivated raising --base_reps: dof 2 is unusable.

    A 3-sigma claim needs |t| >= 19.2 at dof 2 but only 4.5 at dof 7, so the same
    measurement can be significant at 8 reps and not at 3. Assert the ordering so
    a future change to the tail cannot silently restore the z-score behaviour.
    """
    p3 = _sigma_to_p(3.0)
    crit = {}
    for dof in (2, 4, 7):
        lo, hi = 0.0, 1e4                      # bisection: no inverse-t needed
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if _t_sf(mid, dof) > p3:
                lo = mid
            else:
                hi = mid
        crit[dof] = (lo + hi) / 2.0
    _check(crit[2] > 15.0, f"dof=2 critical value should be ~19.2, got {crit[2]:.2f}")
    _check(6.0 < crit[4] < 7.5, f"dof=4 critical value should be ~6.6, got {crit[4]:.2f}")
    _check(4.0 < crit[7] < 5.0, f"dof=7 critical value should be ~4.5, got {crit[7]:.2f}")
    _check(crit[2] > crit[4] > crit[7] > 3.0,
           "critical values must fall with dof and stay above the naive 3.0")
    print(f"PASS  3-sigma critical |t|: dof2={crit[2]:.1f}  dof4={crit[4]:.1f}  "
          f"dof7={crit[7]:.1f}  (naive z-score reading would use 3.0)")


def main() -> None:
    test_t_sf_against_scipy()
    test_t_sf_closed_forms()
    test_sigma_roundtrip()
    test_paired_recovers_effect_under_shared_drift()
    test_paired_sign_convention()
    test_zero_spread_is_infinite_t()
    test_dof_is_what_gates_small_samples()
    print("\nAll paired-statistics checks passed.")


if __name__ == "__main__":
    main()
