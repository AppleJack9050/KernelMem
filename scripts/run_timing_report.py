#!/usr/bin/env python3
"""Summarise a run's ``timing.csv``: where each round's wall clock went.

Run it on a task directory or straight at the file::

    python scripts/run_timing_report.py run/<batch>/<task>/
    python scripts/run_timing_report.py run/<batch>/<task>/timing.csv

The reason this exists rather than a one-off pandas snippet: the obvious way to cost
a phase is to subtract two artifact mtimes, and that number is wrong whenever the
harness was not running in between. A run stopped after round 7 and resumed 51 minutes
later once made an ncu pass look like a 52-minute profiler stall; re-running that same
profile later took 1.05 minutes. So this report does two things a mtime scan cannot:

* it adds up *measured* phase durations, and
* it prints DOWNTIME separately, derived from the process_exit / resume rows, and flags
  the leftover as UNACCOUNTED rather than silently attributing it to the last phase.

A large UNACCOUNTED figure means something real is untimed -- that is a prompt to
instrument it, not a phase you may attribute to whatever ran nearby.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_LIFECYCLE = {"process_start", "process_exit", "resume", "stop_signal", "abort_signal"}
_FMT = "%Y-%m-%d %H:%M:%S"


def _load(path: Path) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["_t"] = datetime.strptime(r["timestamp"].strip(), _FMT)
                r["_round"] = int(r["round_idx"])
                r["_sec"] = float(r["seconds"])
            except Exception:
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["_t"])
    return rows


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:6.1f}s"
    return f"{seconds / 60:6.1f}m"


def _sessions(rows: List[dict]) -> List[Tuple[datetime, datetime, str]]:
    """Bracket each process lifetime, so gaps between them can be named as downtime."""
    out, start = [], None
    for r in rows:
        if r["phase"] in ("process_start",):
            if start is not None:            # previous session never wrote an exit
                out.append((start, None, "killed"))
            start = r["_t"]
        elif r["phase"] == "process_exit" and start is not None:
            out.append((start, r["_t"], "clean"))
            start = None
    if start is not None:
        out.append((start, None, "running-or-killed"))
    return out


def report(path: Path) -> int:
    rows = _load(path)
    if not rows:
        print(f"{path}: no usable rows")
        return 1

    print(f"\n=== {path} ===")
    print(f"{len(rows)} rows, {rows[0]['_t']} -> {rows[-1]['_t']}")

    # ---- sessions and downtime -------------------------------------------------
    sess = _sessions(rows)
    if sess:
        print("\n-- sessions --")
        prev_end = None
        downtime = 0.0
        for s, e, how in sess:
            if prev_end is not None:
                gap = (s - prev_end).total_seconds()
                if gap > 0:
                    downtime += gap
                    print(f"   DOWNTIME  {_fmt(gap)}   {prev_end} -> {s}   "
                          f"(process not running)")
            span = (e - s).total_seconds() if e else float("nan")
            print(f"   session   {_fmt(span) if e else '     ?':>7}   {s} -> "
                  f"{e if e else '(no process_exit: killed or still running)'}  [{how}]")
            prev_end = e or prev_end
        if downtime:
            print(f"\n   total downtime between sessions: {_fmt(downtime)}  "
                  f"<- never attribute this to a phase")

    # ---- per-round breakdown ---------------------------------------------------
    per_round: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    round_total: Dict[int, float] = {}
    for r in rows:
        if r["phase"] in _LIFECYCLE:
            continue
        if r["phase"] == "round_total":
            round_total[r["_round"]] = r["_sec"]
        elif r["_round"] >= 0:
            per_round[r["_round"]][r["phase"]] += r["_sec"]

    if per_round or round_total:
        print("\n-- per round --")
        for rnd in sorted(set(per_round) | set(round_total)):
            phases = per_round.get(rnd, {})
            # ncu_invocation rows are the components of ncu:*; counting both double-counts.
            summed = sum(v for k, v in phases.items()
                         if not k.startswith("ncu_invocation")
                         and k != "ncu_profile_total")
            total = round_total.get(rnd)
            head = f"  round {rnd:>3}  total {_fmt(total) if total else '      ?'}"
            if total:
                unacct = total - summed
                head += f"   measured {_fmt(summed)}   UNACCOUNTED {_fmt(unacct)}"
                if unacct > 0.25 * total and unacct > 60:
                    head += "  <-- large; something in this round is untimed"
            print(head)
            for name, secs in sorted(phases.items(), key=lambda kv: -kv[1]):
                if name.startswith("ncu_invocation") or name == "ncu_profile_total":
                    continue
                print(f"        {name:34s} {_fmt(secs)}")

    # ---- aggregate by phase ----------------------------------------------------
    agg: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["phase"] in _LIFECYCLE or r["phase"] == "round_total":
            continue
        agg[r["phase"]].append(r["_sec"])
    if agg:
        print("\n-- phase totals across the run --")
        print(f"   {'phase':36s} {'n':>4} {'total':>8} {'median':>8} {'max':>8}")
        for name, vals in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
            v = sorted(vals)
            print(f"   {name:36s} {len(v):4d} {_fmt(sum(v))} {_fmt(v[len(v) // 2])} "
                  f"{_fmt(v[-1])}")

    # ---- anything that failed or timed out -------------------------------------
    bad = [r for r in rows if r["detail"].startswith(("failed", "timeout", "error"))]
    if bad:
        print("\n-- failed / timed-out phases --")
        for r in bad:
            print(f"   round {r['_round']:>3}  {r['phase']:34s} {_fmt(r['_sec'])}  "
                  f"{r['detail']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="+",
                    help="a timing.csv, or a task directory containing one")
    args = ap.parse_args()

    rc = 0
    for t in args.target:
        p = Path(t)
        if p.is_dir():
            found = sorted(p.rglob("timing.csv"))
            if not found:
                print(f"{p}: no timing.csv found")
                rc = 1
                continue
            for f in found:
                rc |= report(f)
        elif p.exists():
            rc |= report(p)
        else:
            print(f"{p}: not found")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
