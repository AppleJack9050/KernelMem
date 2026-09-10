"""The CompileIQ seam in utils/compileiq_finish.py, without a GPU or nvcc.

Run:  pytest tests/test_compileiq_search_seam.py   (skips unless compileiq is
installed and its nvcc 13.3 search space is available -- downloaded once,
then cached under ~/.cache/compileiq).

``run_search`` drives a real ``compileiq.ciq.Search`` over the real search
space with a fake Evaluator whose latency is a deterministic function of the
control bytes. What must hold: the objective is called once per candidate,
every candidate arrives as a hex blob, the best row carries ``score_1`` and a
hex ``params`` that decodes to a non-empty ACF, and the ACF lands in --out.
That is the exact contract the finishing pass builds on, and the one thing the
unit tests in test_acf.py cannot cover.
"""
from __future__ import annotations

import hashlib
import os

import pytest

pytest.importorskip("compileiq")
os.environ.setdefault("CIQ_PROCESS_MODE", "fork")

from utils import compileiq_finish as cf  # noqa: E402


class _FakeEvaluator:
    timeout = 30.0

    def __init__(self):
        self.calls = []

    def run(self, acf, tag, *, strict=True):
        blob = acf.read_bytes() if acf else b""
        h = int(hashlib.sha256(blob).hexdigest()[:8], 16) / 0xFFFFFFFF
        ms = 1.0 if acf is None else 0.9 + 0.2 * h
        self.calls.append((tag, acf, ms))
        return {"tag": tag, "status": "ok", "ms": ms, "seconds": 0.0,
                "acf": str(acf) if acf else None, "result": {}, "log_tail": ""}


def test_search_returns_a_decodable_best_acf(tmp_path, monkeypatch):
    monkeypatch.setattr(cf.acf_mod, "nvcc_version", lambda *a, **k: "13.3.33")
    try:
        from compileiq.search_spaces.compilers import NvccSearchSpace
        NvccSearchSpace(version="13.3").retrieve()
    except Exception as exc:  # offline, or no space published for this version
        pytest.skip(f"nvcc 13.3 search space unavailable: {exc}")

    ev = _FakeEvaluator()
    results, best = cf.run_search(ev, tmp_path, space="nvcc", generations=1, pool=6)

    cands = [c for c in ev.calls if c[0] == "cand"]
    assert len(cands) == 6 and all(c[1].is_file() and c[1].stat().st_size > 0 for c in cands)
    assert len(results.get_results()) == 6
    assert set(best) >= {"score_1", "params", "generation"}
    blob = bytes.fromhex(best["params"])
    assert blob and min(c[2] for c in cands) == pytest.approx(float(best["score_1"]))
    assert (tmp_path / "search_results.csv").exists()
    assert sorted(tmp_path.glob("acf/*.bin"))  # every candidate's ACF was materialised
