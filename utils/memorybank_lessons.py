"""Editable long-term memory: durable engineering findings, per task.

    python -m utils.memorybank_lessons list  vae_block_002
    python -m utils.memorybank_lessons add   vae_block_002 --id my-finding \
        --claim "..." --evidence "..." --action "..." --confidence measured
    python -m utils.memorybank_lessons edit  vae_block_002 --id my-finding --action "..."
    python -m utils.memorybank_lessons rm    vae_block_002 --id my-finding
    python -m utils.memorybank_lessons render vae_block_002      # what the prompt sees
    python -m utils.memorybank_lessons validate

Why a separate file rather than editing the rule YAML
-----------------------------------------------------
`bottleneck_headroom_kernelstructure.yaml` is 1470 lines and its `machine_check`
layer *gates every optimization round*: a malformed edit does not produce a worse
suggestion, it stops the run. Findings are knowledge, not gating, so they live in
their own additive file under ``memorybank/lessons/<task>.yaml``. Nothing here can
change what `machine_check` permits.

What belongs here
-----------------
A lesson is something a future run would otherwise have to re-learn by spending
GPU hours: a measured ceiling, a method that repeatedly fails on this shape, a
trap in the harness. Each carries its evidence and a confidence, because an
unsourced claim injected into every prompt is how a guess becomes folklore:

* ``measured``  -- a number from a run in this repo; cite it
* ``observed``  -- seen repeatedly, not separately measured
* ``inferred``  -- reasoned from measurements, not directly tested

Deliberately advisory
---------------------
Rendered into the optimization prompt as context, never as a constraint. The
`allowed_methods` experience is the precedent: the catalog's 26 entries did not
cover the method that produced the best kernel to date, so binding the model to
recorded knowledge would have forbidden the win. Set MEMORYBANK_LESSONS=0 to
drop the block entirely (e.g. for an A/B against a run without it).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
LESSONS_DIR = ROOT / "memorybank" / "lessons"
RULES_YAML = ROOT / "memorybank" / "bottleneck_headroom_kernelstructure.yaml"

CONFIDENCE = ("measured", "observed", "inferred")
_ENV = "MEMORYBANK_LESSONS"

# The cross-task scope. `... add general --id X` writes here, and it is injected
# for EVERY task, so the bar for an entry is that it would still be true on a
# problem with different shapes and a different operator.
GENERAL = "general"

# A lesson the model cannot act on is prompt budget spent for nothing, so
# `action` is required alongside the claim.
REQUIRED = ("id", "claim", "evidence", "action", "confidence")


def enabled() -> bool:
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


def path_for(task: str) -> Path:
    return LESSONS_DIR / f"{Path(task).stem}.yaml"


def load(task: str) -> Dict[str, Any]:
    p = path_for(task)
    if not p.exists():
        return {"task": Path(task).stem, "lessons": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("task", Path(task).stem)
    data.setdefault("lessons", [])
    return data


def save(task: str, data: Dict[str, Any]) -> Path:
    """Write atomically, then re-read. A half-written lessons file would be
    injected into the next prompt as truncated advice."""
    p = path_for(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Capture the pristine state BEFORE the first mutation, never after. This is
    # the only moment the original still exists, and it is the moment a caller is
    # least likely to be thinking about backups -- utils.pathmemory --write and
    # any future automated writer both land here. A baseline taken on the second
    # write would preserve the first write's damage and look like a backup.
    ensure_baseline()
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88)
    tmp = p.with_suffix(".yaml.tmp")
    tmp.write_text(body, encoding="utf-8")
    reread = yaml.safe_load(tmp.read_text(encoding="utf-8"))
    if not isinstance(reread, dict) or "lessons" not in reread:
        tmp.unlink(missing_ok=True)
        raise ValueError("refusing to install a lessons file that does not re-parse")
    tmp.replace(p)
    return p


# --------------------------------------------------------------- the baseline
#
# Why this exists at all, when the repo is a git repository: `memorybank/lessons/`
# is NOT tracked. `bottleneck_headroom_kernelstructure.yaml` and
# `gate_value_from_kernel_struct` are, so `git checkout` recovers those -- but the
# lessons files are exactly the ones a writer mutates, and for them git offers
# nothing. An automated `--write` against an untracked, hand-curated file is a
# one-way door without this.
#
# The baseline is deliberately a plain directory of plain YAML, not an archive or
# a pickle: the thing you want when memory is corrupted is to be able to read the
# backup with `cat` and copy it back by hand if the tooling is what broke.
BASELINE_DIR = LESSONS_DIR / ".baseline"
_MANIFEST = "manifest.json"

# The gating YAML is tracked by git, so it is included for completeness rather
# than as the primary safety net: `validate_all` already warns that a hand edit
# there shows up as a dead run rather than a bad suggestion, and having the known
# state sitting next to the lessons makes a restore one command instead of two.
_ALSO = (RULES_YAML,)


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _baseline_members() -> List[Path]:
    out = sorted(LESSONS_DIR.glob("*.yaml")) if LESSONS_DIR.exists() else []
    return out + [p for p in _ALSO if p.exists()]


def snapshot(*, force: bool = False) -> Optional[Path]:
    """Copy the current long-term memory into ``.baseline/``.

    Refuses to overwrite an existing baseline unless *force*. That refusal is the
    feature: "the original" is a specific state, and a snapshot command that
    silently re-baselines makes it whatever was there most recently -- which,
    after the write you are trying to undo, is the damage.

    Returns the baseline directory, or None if there was nothing to copy.
    """
    import json
    import shutil
    members = _baseline_members()
    if not members:
        return None
    if BASELINE_DIR.exists() and not force:
        return BASELINE_DIR
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    manifest: Dict[str, Any] = {
        "taken": datetime.now().isoformat(timespec="seconds"),
        "note": ("Pristine copy of the long-term memory, taken before the first "
                 "automated write. Restore with: python -m utils.memorybank_lessons "
                 "restore [--task NAME]"),
        "files": {},
    }
    for src in members:
        # Flattened by name: the lessons live in one directory and the rules file
        # has a unique name, so a nested layout would buy nothing and would make
        # a hand-restore harder to eyeball.
        dst = BASELINE_DIR / src.name
        shutil.copy2(src, dst)
        manifest["files"][src.name] = {
            "from": str(src.relative_to(ROOT)),
            "sha256": _sha256(src),
            "bytes": src.stat().st_size,
        }
    (BASELINE_DIR / _MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return BASELINE_DIR


def ensure_baseline() -> Optional[Path]:
    """Take the baseline if there is not one yet. Never raises.

    Called from ``save()``, i.e. on the write path, so a failure here must not be
    able to block a legitimate edit -- losing the backup is bad, losing the edit
    AND surfacing a confusing traceback from a backup helper is worse.
    """
    try:
        if BASELINE_DIR.exists():
            return BASELINE_DIR
        return snapshot()
    except Exception as exc:                                   # pragma: no cover
        print(f"[lessons] baseline snapshot skipped ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return None


def baseline_status() -> List[Dict[str, Any]]:
    """One row per baselined file: name, whether the live copy still matches."""
    import json
    if not BASELINE_DIR.exists():
        return []
    try:
        manifest = json.loads((BASELINE_DIR / _MANIFEST).read_text(encoding="utf-8"))
    except Exception:
        manifest = {"files": {}}
    rows = []
    for name, meta in sorted((manifest.get("files") or {}).items()):
        live = ROOT / meta.get("from", f"memorybank/lessons/{name}")
        if not live.exists():
            state = "MISSING (live file deleted)"
        elif _sha256(live) == meta.get("sha256"):
            state = "unchanged"
        else:
            state = "CHANGED since the baseline"
        rows.append({"name": name, "state": state, "live": live,
                     "taken": manifest.get("taken", "?")})
    return rows


def restore(task: Optional[str] = None) -> List[Path]:
    """Put the baseline back. *task* restores one file; None restores all.

    The live file is copied to ``<name>.pre-restore`` first, so an accidental
    restore is itself undoable -- the failure mode of a restore command that
    discards the current state is that it turns one mistake into two.
    """
    import shutil
    import json
    if not BASELINE_DIR.exists():
        raise FileNotFoundError(
            f"no baseline at {BASELINE_DIR} -- take one with "
            f"`python -m utils.memorybank_lessons snapshot`")
    manifest = json.loads((BASELINE_DIR / _MANIFEST).read_text(encoding="utf-8"))
    want = f"{Path(task).stem}.yaml" if task else None
    done: List[Path] = []
    for name, meta in sorted((manifest.get("files") or {}).items()):
        if want and name != want:
            continue
        src = BASELINE_DIR / name
        if not src.exists():
            continue
        live = ROOT / meta.get("from", f"memorybank/lessons/{name}")
        live.parent.mkdir(parents=True, exist_ok=True)
        if live.exists() and _sha256(live) != meta.get("sha256"):
            shutil.copy2(live, live.with_suffix(live.suffix + ".pre-restore"))
        shutil.copy2(src, live)
        done.append(live)
    if want and not done:
        raise FileNotFoundError(f"{want} is not in the baseline")
    return done


def validate_lesson(l: Dict[str, Any]) -> List[str]:
    errs = [f"missing '{k}'" for k in REQUIRED if not str(l.get(k, "")).strip()]
    if l.get("confidence") and l["confidence"] not in CONFIDENCE:
        errs.append(f"confidence {l['confidence']!r} not in {CONFIDENCE}")
    return errs


def validate_all() -> List[str]:
    """Check every lessons file AND that the gating YAML still parses.

    The second half matters more than the first: this module never writes that
    file, but a human editing memory by hand very reasonably might, and the
    failure shows up as a dead run rather than a bad suggestion.
    """
    errs: List[str] = []
    try:
        rules = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
        for layer in ("machine_check", "llm_assist"):
            if layer not in (rules or {}):
                errs.append(f"{RULES_YAML.name}: '{layer}' layer is missing")
    except Exception as exc:
        errs.append(f"{RULES_YAML.name}: does not parse -- {exc}")
    for p in sorted(LESSONS_DIR.glob("*.yaml")) if LESSONS_DIR.exists() else []:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            errs.append(f"{p.name}: does not parse -- {exc}")
            continue
        seen = set()
        for i, l in enumerate(data.get("lessons") or []):
            for e in validate_lesson(l):
                errs.append(f"{p.name}[{i}]: {e}")
            if l.get("id") in seen:
                errs.append(f"{p.name}: duplicate id {l.get('id')!r}")
            seen.add(l.get("id"))
    return errs


def _active(task: str) -> List[Dict[str, Any]]:
    order = {c: i for i, c in enumerate(CONFIDENCE)}
    ls = [l for l in (load(task).get("lessons") or []) if not l.get("retired")]
    ls.sort(key=lambda l: (order.get(l.get("confidence"), 9), l.get("id", "")))
    return ls


def _bullets(lessons: List[Dict[str, Any]], limit: int) -> List[str]:
    out = []
    for l in lessons[:limit]:
        out.append(f"- **{l['id']}** [{l['confidence']}] {l['claim']}")
        out.append(f"  - evidence: {l['evidence']}")
        out.append(f"  - so: {l['action']}")
    if len(lessons) > limit:
        out.append(f"- ... and {len(lessons) - limit} further findings not shown.")
    return out


def render(task: str, *, limit: int = 40) -> str:
    """The block injected into the optimization prompt.

    Two scopes, kept apart on purpose. GENERAL is craft that transfers between
    tasks -- hardware facts, measurement discipline, harness traps -- and is
    injected for every task. The per-task file holds findings true of one
    problem's shapes and structure. Merging them would let a number measured on
    one block be read as a law, which is the specific way a memory bank turns
    into folklore.

    Returns "" when there is nothing to say or the feature is off: an empty
    heading is worse than silence, because the model will try to honour it.
    """
    if not enabled():
        return ""
    general = _active(GENERAL)
    specific = _active(task) if Path(task).stem != GENERAL else []
    if not general and not specific:
        return ""
    out = [
        "### LONG-TERM MEMORY -- what this repository has already measured",
        "",
        "ADVISORY context, not constraints. These exist so you do not spend a round",
        "rediscovering something already paid for. A finding that contradicts what you",
        "measure THIS round is out of date -- trust your measurement and say so.",
    ]
    if general:
        out += ["", "#### General engineering experience (applies to every task)", ""]
        out += _bullets(general, limit)
    if specific:
        out += ["", f"#### Specific to `{Path(task).stem}` -- do not generalise these numbers", ""]
        out += _bullets(specific, limit)
    return "\n".join(out)


# ------------------------------------------------------------------------ CLI
def _print_lessons(data: Dict[str, Any]) -> None:
    ls = data.get("lessons") or []
    if not ls:
        print("  (none)")
        return
    for l in ls:
        flag = " RETIRED" if l.get("retired") else ""
        print(f"  {l.get('id')} [{l.get('confidence')}]{flag}")
        print(f"      claim   : {l.get('claim')}")
        print(f"      evidence: {l.get('evidence')}")
        print(f"      so      : {l.get('action')}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Edit the per-task long-term memory.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _task(p):
        p.add_argument("task", help="task name or path, e.g. vae_block_002")

    _task(sub.add_parser("list"))
    _task(sub.add_parser("render"))
    p_add = sub.add_parser("add"); _task(p_add)
    p_edit = sub.add_parser("edit"); _task(p_edit)
    p_rm = sub.add_parser("rm"); _task(p_rm)
    sub.add_parser("validate")
    p_snap = sub.add_parser("snapshot", help="copy the long-term memory to .baseline/")
    p_snap.add_argument("--force", action="store_true",
                        help="re-baseline over an existing one (this DISCARDS the "
                             "original state you would restore to)")
    sub.add_parser("baseline", help="show the baseline and what has changed since")
    p_rest = sub.add_parser("restore", help="put the baseline back")
    p_rest.add_argument("--task", default=None,
                        help="restore only this task's file (default: all)")
    for p in (p_add, p_edit):
        p.add_argument("--id", required=True)
        p.add_argument("--claim"); p.add_argument("--evidence"); p.add_argument("--action")
        p.add_argument("--confidence", choices=CONFIDENCE)
        p.add_argument("--source", default="")
        p.add_argument("--retired", action="store_true")
    p_rm.add_argument("--id", required=True)

    a = ap.parse_args(argv)

    if a.cmd == "snapshot":
        existed = BASELINE_DIR.exists()
        if existed and not a.force:
            print(f"a baseline already exists at {BASELINE_DIR}")
            print("leaving it alone -- it is the ORIGINAL, and re-taking it now would "
                  "replace that with the current state.\nUse --force only if you mean "
                  "to make the current state the new original.")
            return 0
        d = snapshot(force=a.force)
        if d is None:
            print("nothing to snapshot -- no lessons files found", file=sys.stderr)
            return 2
        print(f"{'re-' if existed else ''}baselined {len(_baseline_members())} file(s) -> {d}")
        for r in baseline_status():
            print(f"  {r['name']}")
        return 0

    if a.cmd == "baseline":
        rows = baseline_status()
        if not rows:
            print(f"no baseline yet. Take one with:\n"
                  f"  python -m utils.memorybank_lessons snapshot")
            return 0
        print(f"baseline: {BASELINE_DIR}   taken {rows[0]['taken']}")
        for r in rows:
            print(f"  {r['state']:28} {r['name']}")
        if any(r["state"] != "unchanged" for r in rows):
            print("\nrestore with: python -m utils.memorybank_lessons restore")
        return 0

    if a.cmd == "restore":
        try:
            done = restore(a.task)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        for p in done:
            print(f"restored {p.relative_to(ROOT)}")
        errs = validate_all()
        if errs:
            print("WARNING after restore:\n" + "\n".join("  " + e for e in errs),
                  file=sys.stderr)
            return 1
        print("  ok -- restored memory parses and is well-formed")
        return 0

    if a.cmd == "validate":
        errs = validate_all()
        print("\n".join(f"  FAIL {e}" for e in errs) if errs else "  ok -- memorybank parses, lessons well-formed")
        return 1 if errs else 0

    if a.cmd == "list":
        data = load(a.task)
        print(f"{path_for(a.task)}")
        _print_lessons(data)
        return 0

    if a.cmd == "render":
        txt = render(a.task)
        print(txt or "(nothing to inject)")
        return 0

    data = load(a.task)
    lessons = data.setdefault("lessons", [])
    idx = next((i for i, l in enumerate(lessons) if l.get("id") == a.id), None)

    if a.cmd == "rm":
        if idx is None:
            print(f"no lesson with id {a.id!r}", file=sys.stderr)
            return 2
        lessons.pop(idx)
        print(f"removed {a.id}")
    else:
        if a.cmd == "add" and idx is not None:
            print(f"id {a.id!r} already exists -- use `edit`", file=sys.stderr)
            return 2
        if a.cmd == "edit" and idx is None:
            print(f"no lesson with id {a.id!r} -- use `add`", file=sys.stderr)
            return 2
        rec = lessons[idx] if idx is not None else {"id": a.id}
        for f in ("claim", "evidence", "action", "confidence", "source"):
            v = getattr(a, f, None)
            if v:
                rec[f] = v
        if a.retired:
            rec["retired"] = True
        rec.setdefault("recorded", str(date.today()))
        errs = validate_lesson(rec)
        if errs:
            print("refusing to write:\n" + "\n".join("  " + e for e in errs), file=sys.stderr)
            return 2
        if idx is None:
            lessons.append(rec)
        print(f"{'updated' if idx is not None else 'added'} {a.id}")

    save(a.task, data)
    errs = validate_all()
    if errs:
        print("WARNING after write:\n" + "\n".join("  " + e for e in errs), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
