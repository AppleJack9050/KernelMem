"""Link the kernel-design-agents skills so KernelMem's agent calls can load them.

    python -m utils.install_kda_skills            # link
    python -m utils.install_kda_skills --status
    python -m utils.install_kda_skills --uninstall

What this does
--------------
Symlinks ``third_party/kernel-design-agents/skills/*`` into ``~/.claude/skills/``,
which is upstream's own install step and the only place the Agent SDK looks when
``setting_sources`` includes ``"user"``. Symlinks, never copies: the upstream
repositories ship no LICENSE, so their contents stay in a checkout that carries
their git history rather than being absorbed into this tree.

Why user scope and not project scope
------------------------------------
``query_server`` gives each tool-mode call a fresh temp ``cwd`` and deletes it
afterwards, so a project-scoped ``.claude/skills`` would be looked for inside a
directory that does not exist yet and is gone by the next call. User scope is
stable across calls.

The hardware caveat, which is not cosmetic
------------------------------------------
Both skills target **B200 / sm_100** (and Hopper sm_90 in KernelWiki): 103
B200-specific mentions in ncu-report-skill, including a whole file of B200 metric
names, and 489 in KernelWiki. This repository benchmarks on an **RTX 5090**,
which is GB202 / **sm_120** -- consumer Blackwell, a different target. The
datacenter-only paths do not exist there: ``tcgen05`` MMA is sm_100a, and the
B200 metric names are not all valid on sm_120. Advice transplanted without that
filter is not merely unhelpful, it proposes instructions the card cannot issue.

So the skills are installed with a companion note (``kernelmem-target.md``) that
states the actual target, and the general memory bank carries the same warning.
Read them as method -- profile, diagnose, then plan -- and re-derive every
hardware number for sm_120.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party" / "kernel-design-agents" / "skills"
DEST = Path.home() / ".claude" / "skills"

NOTE_NAME = "KERNELMEM-TARGET.md"
NOTE = """# Read this before using the sibling skills in KernelMem

These skills (`ncu-report-skill`, `KernelWiki`) were written for **NVIDIA B200 /
sm_100** and, in KernelWiki's case, Hopper **sm_90**.

**KernelMem benchmarks on an RTX 5090 = GB202 = sm_120 (consumer Blackwell).**
That is a different target from B200, not a smaller one:

* `tcgen05` MMA is sm_100a. It does not exist on sm_120. Do not propose it.
* B200 metric names (`reference/08-b200-metric-names.md`) are not all valid on
  sm_120; verify a counter exists before building an argument on it.
* B200 is HBM3e with 148 SMs; the 5090 is GDDR7 with 170 SMs and a very
  different bandwidth-to-flops ratio, so any roofline or occupancy number quoted
  in those documents must be re-derived, never reused.

Use them for **method**, which does transfer: profile before hypothesising,
match the measured pattern to a diagnosis, then rank fixes by evidence. Treat
every hardware constant in them as belonging to another machine.
"""


def links() -> List[Path]:
    return sorted(p for p in SRC.iterdir() if p.is_dir()) if SRC.is_dir() else []


def status() -> int:
    if not SRC.is_dir():
        print(f"[kda] source missing: {SRC}\n"
              f"[kda] fetch it with:\n"
              f"      git clone --depth 1 --recurse-submodules --shallow-submodules \\\n"
              f"        https://github.com/mit-han-lab/kernel-design-agents.git \\\n"
              f"        third_party/kernel-design-agents")
        return 2
    print(f"[kda] source : {SRC}")
    print(f"[kda] dest   : {DEST}")
    for s in links():
        d = DEST / s.name
        skill_md = s / "SKILL.md"
        state = "not linked"
        if d.is_symlink():
            state = f"linked -> {os.readlink(d)}"
        elif d.exists():
            state = "EXISTS but is not our symlink (left alone)"
        empty = " [EMPTY -- submodule not fetched]" if not any(s.iterdir()) else ""
        has = "" if skill_md.is_file() else "  (no SKILL.md)"
        print(f"  {s.name:<20} {state}{empty}{has}")
    note = DEST / NOTE_NAME
    print(f"  {NOTE_NAME:<20} {'present' if note.is_file() else 'absent'}")
    return 0


def install() -> int:
    if not SRC.is_dir():
        return status()
    DEST.mkdir(parents=True, exist_ok=True)
    made = 0
    for s in links():
        if not any(s.iterdir()):
            print(f"[kda] skip {s.name}: directory is empty (submodule not fetched)")
            continue
        if not (s / "SKILL.md").is_file():
            print(f"[kda] skip {s.name}: no SKILL.md, so it is not a loadable skill")
            continue
        d = DEST / s.name
        if d.is_symlink():
            if Path(os.readlink(d)).resolve() == s.resolve():
                print(f"[kda] ok   {s.name} already linked")
                continue
            d.unlink()
        elif d.exists():
            # Never clobber a real directory the user may have authored.
            print(f"[kda] REFUSE {s.name}: {d} exists and is not a symlink")
            continue
        d.symlink_to(s, target_is_directory=True)
        print(f"[kda] link {s.name} -> {s}")
        made += 1
    (DEST / NOTE_NAME).write_text(NOTE, encoding="utf-8")
    print(f"[kda] wrote {DEST / NOTE_NAME}")
    print(f"[kda] {made} new link(s). Enable in runs with KERNELMEM_SKILLS=1.")
    return 0


def uninstall() -> int:
    for s in links():
        d = DEST / s.name
        if d.is_symlink():
            d.unlink()
            print(f"[kda] unlinked {s.name}")
    note = DEST / NOTE_NAME
    if note.is_file():
        note.unlink()
        print(f"[kda] removed {NOTE_NAME}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        return status()
    if a.uninstall:
        return uninstall()
    return install()


if __name__ == "__main__":
    raise SystemExit(main())
