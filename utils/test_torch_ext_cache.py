"""Regression test for the two torch-extension-cache fixes.

Background
----------
torch keys every JIT extension build directory on the extension NAME alone
(``~/.cache/torch_extensions/py313_cu130/<name>/``), and guards it with a
``FileBaton`` that has no timeout, no staleness check and no holder-liveness
check -- ``release()`` runs only in the winner's ``finally``. Two consequences
bit this project:

1. Tool-mode agents are told to compile with ``load_inline``, so an agent's
   throwaway build landed in the same shared directory the harness later built
   the real kernel in. Names collide constantly between independently generated
   kernels, and concurrent lineages made two processes write one ``cuda.cu``.
   Fixed by giving each agent call a private ``TORCH_EXTENSIONS_DIR``.
2. Every kill path here -- the 20-minute ``p.join`` in main_memory_latest, the
   600s SIGALRM in utils/compile_and_run, the SIGTERM/SIGKILL in run_lineages --
   orphans a baton forever. Found in the wild on 2026-08-06:
   ``cudnn_conv3x3_nhwc_v7_a/`` held a 0-byte ``lock`` from 2026-07-31 with no
   ``.so`` ever built, while its ``cuda.cu`` carried a later mtime -- a second
   process had written its sources in, spun on the dead baton and built nothing.
   A kernel picking a poisoned name hangs to the compile alarm and is then
   misreported as an illegal memory access, so a good kernel is scored -inf and
   sent to repair for a bug it does not have. Fixed by ``sweep_stale_batons``.

The single most important check in this file is
``test_a_live_lock_survives_the_sweep``. The sweep's only liveness signal is
age, and deleting a lock a live build still holds would let two processes
compile into one directory -- trading a hang for silent corruption, which is a
strictly worse failure. Everything else here is cheap insurance; that one is
the load-bearing guarantee.

Run directly::

    python utils/test_torch_ext_cache.py

Filesystem and env manipulation under a temp dir: no CUDA, no compiler, no
network, and nothing outside tempfile.mkdtemp is touched. One check
(``test_a_real_filebaton_held_by_a_real_process_is_protected_then_collected``)
spawns a subprocess that imports torch to exercise a genuine ``FileBaton``, so
the suite takes a few seconds rather than well under one.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.torch_ext_cache import (  # noqa: E402
    _DEFAULT_MAX_AGE_S,
    extensions_root,
    sweep_stale_batons,
)

_HOUR = 3600.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _make_cache(tmp: Path, entries) -> Path:
    """Build a fake extension cache. *entries* is (name, lock_age_s|None, has_so)."""
    root = tmp / "torch_extensions"
    for name, lock_age, has_so in entries:
        d = root / "py313_cu130" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "cuda.cu").write_text("// sources")
        if has_so:
            (d / f"{name}.so").write_text("ELF")
        if lock_age is not None:
            lock = d / "lock"
            lock.touch()
            stamp = time.time() - lock_age
            os.utime(lock, (stamp, stamp))
    return root


def _locks(root: Path):
    return {p.parent.name for p in root.glob("**/lock")}


class _captured:
    """Collect everything the sweep prints, so we can pin what it does NOT say.

    Several of the sweep's guards (``is_file()``, the ``FileNotFoundError``
    branch) are invisible in the return value, because the broad ``except
    OSError`` below them keeps the behaviour correct either way. What they
    actually buy is silence on the expected paths -- a run that logged
    ``[baton] WARNING`` on every concurrent sweep would be training the reader
    to ignore the one warning that matters.
    """

    def __enter__(self):
        self._buf = io.StringIO()
        self._old = sys.stdout
        sys.stdout = self._buf
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old
        return False

    @property
    def text(self) -> str:
        return self._buf.getvalue()


class _EnvGuard:
    """Set/unset env vars for the duration of a block, then restore exactly."""

    def __init__(self, **kv):
        self._kv = kv
        self._old = {}

    def __enter__(self):
        for k, v in self._kv.items():
            self._old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


# --------------------------------------------------------------------------
# Fix #1 -- per-agent TORCH_EXTENSIONS_DIR
# --------------------------------------------------------------------------
def test_agent_env_points_into_the_workdir_and_creates_it():
    from agents.query_server import _agent_build_env

    workdir = tempfile.mkdtemp(prefix="kernelmem_agent_test_")
    try:
        env = _agent_build_env(workdir)
        ext = env["TORCH_EXTENSIONS_DIR"]
        assert os.path.isabs(ext), f"must be absolute, got {ext!r}"
        assert Path(ext).is_relative_to(workdir), f"{ext} escapes the workdir {workdir}"
        assert os.path.isdir(ext), "torch will not create it for us reliably; we must"
        # Idempotent: query_server builds options once per call, but a retry or a
        # second call against the same workdir must not explode on an existing dir.
        assert _agent_build_env(workdir) == env
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_agent_env_overrides_an_inherited_extensions_dir():
    """The whole point of the fix: the shared cache must not win.

    The SDK builds the child env as {**os.environ, **options.env}, so an
    inherited TORCH_EXTENSIONS_DIR is only neutralised if ours is in options.env.
    """
    from agents.query_server import _SUBSCRIPTION_ENV, _agent_build_env

    workdir = tempfile.mkdtemp(prefix="kernelmem_agent_test_")
    try:
        with _EnvGuard(TORCH_EXTENSIONS_DIR="/home/someone/.cache/torch_extensions"):
            options_env = {**_SUBSCRIPTION_ENV, **_agent_build_env(workdir)}
            child_env = {**os.environ, **options_env}          # what the SDK does
            assert Path(child_env["TORCH_EXTENSIONS_DIR"]).is_relative_to(workdir), (
                "an inherited TORCH_EXTENSIONS_DIR survived into the child; the agent "
                "would still build in the shared cache")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_agent_env_keeps_every_subscription_override():
    """Merging must not drop the empty-string auth overrides.

    Those force claude.ai subscription auth; losing one would silently switch
    billing to the API, which is why they are set as empty strings rather than
    omitted.
    """
    from agents.query_server import _SUBSCRIPTION_ENV, _agent_build_env

    workdir = tempfile.mkdtemp(prefix="kernelmem_agent_test_")
    try:
        merged = {**_SUBSCRIPTION_ENV, **_agent_build_env(workdir)}
        for key, value in _SUBSCRIPTION_ENV.items():
            assert merged[key] == value, f"{key} was clobbered by the merge"
        assert "TORCH_EXTENSIONS_DIR" not in _SUBSCRIPTION_ENV, (
            "the helper must not mutate the shared _SUBSCRIPTION_ENV dict")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _capture_options(call_type: str):
    """Run query_server with the network stubbed out; return (options, workdir)."""
    import agents.query_server as qs

    seen = {}

    class _Recorder:
        def __init__(self, **kw):
            seen.update(kw)

    real_options, real_retry = qs.ClaudeAgentOptions, qs.retry_with_backoff
    qs.ClaudeAgentOptions = _Recorder
    qs.retry_with_backoff = lambda fn, **kw: (["stub reply"], None)
    try:
        out = qs.query_server(prompt="hi", system_prompt="sys", call_type=call_type)
        assert out == "stub reply"
    finally:
        qs.ClaudeAgentOptions, qs.retry_with_backoff = real_options, real_retry
    return seen


def test_a_tool_mode_call_gets_a_private_build_dir_and_cleans_it_up():
    opts = _capture_options("seed")          # 'seed' is in _TOOL_CALL_TYPES
    ext = opts["env"].get("TORCH_EXTENSIONS_DIR")
    assert ext, "a tool-mode call must get its own TORCH_EXTENSIONS_DIR"
    assert Path(ext).is_relative_to(opts["cwd"]), (
        f"build dir {ext} is not inside the agent workdir {opts['cwd']}")
    assert not Path(opts["cwd"]).exists(), (
        "query_server's finally must rmtree the workdir, taking the build dir with it")


def test_a_non_tool_call_gets_no_build_dir():
    """The judge/problem-identify calls have tools=[] and compile nothing."""
    opts = _capture_options("judge_optimization")
    assert "TORCH_EXTENSIONS_DIR" not in opts["env"]
    assert opts.get("cwd") is None


# --------------------------------------------------------------------------
# Fix #2 -- the stale-baton sweep
# --------------------------------------------------------------------------
def test_a_live_lock_survives_the_sweep():
    """THE load-bearing check.

    A baton's mtime is set once at O_CREAT and never refreshed, so age is a
    proxy for "how long has this build been running". Deleting a lock a live
    build still holds lets a second process compile into the same directory --
    silent corruption, strictly worse than the hang the sweep exists to prevent.
    Under run_lineages several children sweep the shared cache concurrently
    while others are mid-build, so this is the everyday case, not a corner.
    """
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [
            ("building_now", 0.0, False),                 # just started
            ("building_a_while", 20 * 60, False),         # 20 min: past the SIGALRM, still < threshold
            ("long_build", _DEFAULT_MAX_AGE_S - 60, False),  # 59 min, just inside
        ])
        removed = sweep_stale_batons(root=root, verbose=False)
        assert removed == [], f"a live build's lock was deleted: {removed}"
        assert _locks(root) == {"building_now", "building_a_while", "long_build"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stale_locks_are_removed_at_any_depth_with_or_without_a_so():
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [
            ("orphaned_no_so", 138 * _HOUR, False),   # the real-world signature
            ("orphaned_with_so", 3 * _HOUR, True),
            ("fresh", 60.0, False),
        ])
        # A lock directly under the root, not under py313_cu130/, must also be seen.
        shallow = root / "shallow"
        shallow.mkdir()
        (shallow / "lock").touch()
        old = time.time() - 10 * _HOUR
        os.utime(shallow / "lock", (old, old))

        removed = sweep_stale_batons(root=root, verbose=False)
        assert {p.parent.name for p in removed} == {
            "orphaned_no_so", "orphaned_with_so", "shallow"}, removed
        assert _locks(root) == {"fresh"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_sweep_touches_nothing_but_locks():
    """Sources and built objects must survive: only the lock is ever removed."""
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("orphaned", 5 * _HOUR, True)])
        d = root / "py313_cu130" / "orphaned"
        (d / "build.ninja").write_text("rules")
        (d / "lock_dir").mkdir()          # a DIRECTORY named like a lock, near enough
        before = {p.name for p in d.iterdir()}

        sweep_stale_batons(root=root, verbose=False)

        after = {p.name for p in d.iterdir()}
        assert before - after == {"lock"}, f"the sweep removed more than the lock: {before - after}"
        assert (d / "cuda.cu").read_text() == "// sources"
        assert (d / "orphaned.so").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_directory_named_lock_is_never_unlinked():
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = tmp / "torch_extensions"
        d = root / "py313_cu130" / "weird"
        (d / "lock").mkdir(parents=True)          # a dir, not a baton
        old = time.time() - 99 * _HOUR
        os.utime(d / "lock", (old, old))

        with _captured() as out:
            removed = sweep_stale_batons(root=root, verbose=True)
        assert removed == []
        assert (d / "lock").is_dir()
        # Skipped by the is_file() guard, not by failing to unlink it: a
        # directory must not produce a warning the reader has to triage.
        assert "WARNING" not in out.text, out.text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_future_mtime_is_treated_as_live():
    """Clock skew (NFS, a VM resume) must not make the sweep aggressive."""
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("from_the_future", -10 * _HOUR, False)])
        assert sweep_stale_batons(root=root, verbose=False) == []
        assert _locks(root) == {"from_the_future"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_threshold_is_tunable_and_zero_disables_the_sweep():
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("old", 2 * _HOUR, False)])

        with _EnvGuard(KERNELMEM_BATON_MAX_AGE="0"):
            assert sweep_stale_batons(root=root, verbose=False) == [], "0 must disable"
        assert sweep_stale_batons(root=root, max_age_s=0, verbose=False) == []
        assert _locks(root) == {"old"}

        # Tuning down still works, subject to the floor: the lock is 2h old, so
        # it clears the clamped 1800s threshold either way.
        with _EnvGuard(KERNELMEM_BATON_MAX_AGE="2000"):
            assert len(sweep_stale_batons(root=root, verbose=False)) == 1
        assert _locks(root) == set()

        # Garbage and negatives must fall back / clamp rather than raise or
        # delete everything in sight.
        for junk in ("", "   ", "abc", "1e999", "-5"):
            with _EnvGuard(KERNELMEM_BATON_MAX_AGE=junk):
                sweep_stale_batons(root=root / "missing", verbose=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_sweep_never_raises():
    """A cache it cannot read must not take a run down: the run works without it."""
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        assert sweep_stale_batons(root=tmp / "does_not_exist", verbose=False) == []

        as_file = tmp / "not_a_dir"
        as_file.write_text("x")
        assert sweep_stale_batons(root=as_file, verbose=False) == []

        # An unreadable subdirectory in the middle of the tree.
        root = _make_cache(tmp, [("ok_and_old", 9 * _HOUR, False)])
        walled = root / "py313_cu130" / "walled"
        walled.mkdir()
        (walled / "lock").touch()
        os.chmod(walled, 0o000)
        try:
            if os.access(walled, os.R_OK):        # running as root: skip
                return
            removed = sweep_stale_batons(root=root, verbose=False)
            assert {p.parent.name for p in removed} == {"ok_and_old"}, (
                "an unreadable sibling must not stop the sweep reaching the rest")
        finally:
            os.chmod(walled, 0o755)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_concurrent_sweeps_agree_and_never_raise():
    """run_lineages children sweep the same shared cache at the same moment."""
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [(f"orphan_{i}", 4 * _HOUR, False) for i in range(40)])
        start = threading.Barrier(6)
        removed, errors = [], []

        def worker():
            try:
                start.wait()
                removed.append(sweep_stale_batons(root=root, verbose=True))
            except BaseException as exc:            # noqa: BLE001 - the point is nothing escapes
                errors.append(exc)

        with _captured() as out:
            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"a racing sweep raised: {errors}"
        # Losing a race is the expected outcome for 5 of 6 sweepers on every
        # lock, and must be handled by the FileNotFoundError branch rather than
        # falling through to the generic OSError warning.
        assert "WARNING" not in out.text, out.text
        assert _locks(root) == set(), "every orphan should be gone"
        # Each lock is claimed by exactly one sweeper; the losers see the
        # FileNotFoundError and skip rather than double-counting.
        total = sum(len(r) for r in removed)
        assert total == 40, f"expected each of the 40 locks removed once, got {total}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_lock_a_live_process_holds_open_is_never_removed():
    """The liveness gate, and the reason it exists.

    ``~/.cache/torch_extensions`` is user-global: a notebook, another checkout
    or a training script can be mid-build in it, bounded by none of this repo's
    compile budgets. If the sweep unlinked such a lock, torch's
    ``FileBaton.release()`` -- which has no ``try`` -- would raise
    FileNotFoundError out of ``load_inline``'s ``finally`` and kill an otherwise
    successful compile, while a second process took the same directory. A baton
    holds its fd open for the whole build (waiters hold nothing), so an open
    descriptor is the exact signal.
    """
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("held_open", 99 * _HOUR, False),
                                 ("truly_orphaned", 99 * _HOUR, False)])
        held = root / "py313_cu130" / "held_open" / "lock"

        fd = os.open(held, os.O_RDONLY)     # stand in for the baton's own fd
        try:
            removed = sweep_stale_batons(root=root, verbose=False)
        finally:
            os.close(fd)

        assert {p.parent.name for p in removed} == {"truly_orphaned"}, removed
        assert held.exists(), "a lock a live process holds open was deleted"
        # And once the holder lets go, the same lock is collectable.
        assert {p.parent.name for p in sweep_stale_batons(root=root, verbose=False)} \
            == {"held_open"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_HOLDER = '''
import sys, time
from torch.utils.file_baton import FileBaton
b = FileBaton(sys.argv[1])
assert b.try_acquire(), "could not acquire"
print("ACQUIRED", flush=True)
time.sleep(300)
'''


def test_a_real_filebaton_held_by_a_real_process_is_protected_then_collected():
    """End-to-end against torch's actual FileBaton, not a stand-in.

    ``test_a_lock_a_live_process_holds_open_is_never_removed`` uses a plain
    ``os.open`` to simulate the holder. That proves the /proc scan works on
    *some* open descriptor; it does not prove it works on the one torch actually
    keeps, which is the only fd that matters. A test that mocks the thing under
    test is exactly the kind that stays green while the real path is broken.

    So: acquire a genuine FileBaton in a genuine subprocess, age the lock past
    the threshold, and check both directions -- the sweep must refuse it while
    the process lives, and collect it once SIGKILL (the real orphaning path,
    which runs no ``finally``) has taken the holder away.
    """
    import shutil as _sh
    import signal
    import subprocess

    import utils.torch_ext_cache as tec
    from utils.torch_ext_cache import _open_inodes

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    proc = None
    try:
        holder = tmp / "holder.py"
        holder.write_text(_HOLDER)
        root = tmp / "torch_extensions"
        d = root / "py313_cu130" / "real_ext"
        d.mkdir(parents=True)
        lock = d / "lock"

        proc = subprocess.Popen([sys.executable, str(holder), str(lock)],
                                stdout=subprocess.PIPE, text=True)
        # torch import dominates; generous, and the finally always reaps.
        deadline = time.time() + 120
        line = ""
        while time.time() < deadline and not line:
            line = (proc.stdout.readline() or "").strip()
            if proc.poll() is not None and not line:
                raise AssertionError("holder exited before acquiring the baton")
        assert line == "ACQUIRED", f"holder did not acquire the baton (got {line!r})"

        st = lock.stat()
        ancient = time.time() - 99 * _HOUR
        os.utime(lock, (ancient, ancient))     # age gate alone would delete it

        held = _open_inodes()
        assert held is not None and (st.st_dev, st.st_ino) in held, (
            "torch's own baton fd was not visible in /proc -- the liveness gate "
            "is scanning for the wrong thing")
        assert sweep_stale_batons(root=root, verbose=False) == [], (
            "the startup sweep deleted a lock a live torch build was holding")
        # The per-round sweep matters more here: liveness is its ONLY gate, so
        # if the /proc scan ever stops seeing torch's fd this is what breaks.
        assert tec.sweep_unheld_batons(root=root, verbose=False) == [], (
            "the per-round sweep deleted a lock a live torch build was holding")
        assert lock.exists()

        proc.send_signal(signal.SIGKILL)       # the real orphaning path
        proc.wait(timeout=30)
        for _ in range(50):                    # let the kernel reap the fds
            if (st.st_dev, st.st_ino) not in (_open_inodes() or set()):
                break
            time.sleep(0.1)

        assert lock.exists(), "SIGKILL must leave the lock orphaned (no finally runs)"
        # Collected by the per-round sweep on liveness alone, WITHOUT the 99h
        # ageing above doing any work -- which is the whole point of it.
        stranded = time.time() - 120           # "stranded moments ago"
        os.utime(lock, (stranded, stranded))
        assert sweep_stale_batons(root=root, verbose=False) == [], (
            "a 2-minute-old lock is far inside the startup gate")
        removed = tec.sweep_unheld_batons(root=root, verbose=False)
        assert [p.parent.name for p in removed] == ["real_ext"], (
            f"the orphan was not collected once its holder died: {removed}")
        assert not lock.exists()
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        _sh.rmtree(tmp, ignore_errors=True)


def test_the_per_round_sweep_collects_a_baton_this_run_just_stranded():
    """The case the startup sweep structurally cannot reach.

    A lineage killed mid-build leaves a lock whose mtime is minutes old --
    nowhere near the 3600s startup gate -- so every later round of THAT run
    would keep hanging on the name it stranded. Liveness, not age, is what
    makes this collectable.
    """
    from utils.torch_ext_cache import sweep_unheld_batons

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("stranded_2min_ago", 120.0, False)])
        # The startup sweep must NOT touch it: too young, and rightly so.
        assert sweep_stale_batons(root=root, verbose=False) == []
        # The per-round sweep must, because nobody holds it.
        removed = sweep_unheld_batons(root=root, verbose=False)
        assert [p.parent.name for p in removed] == ["stranded_2min_ago"], removed
        assert _locks(root) == set()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_per_round_sweep_still_refuses_a_lock_someone_holds():
    """Liveness is the only gate, so it had better be load-bearing."""
    from utils.torch_ext_cache import sweep_unheld_batons

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("building", 600.0, False), ("dead", 600.0, False)])
        live = root / "py313_cu130" / "building" / "lock"
        fd = os.open(live, os.O_RDONLY)
        try:
            removed = sweep_unheld_batons(root=root, verbose=False)
        finally:
            os.close(fd)
        assert [p.parent.name for p in removed] == ["dead"], removed
        assert live.exists(), "a held lock was collected by the per-round sweep"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_ORPHAN_HOLDER = '''
import os, subprocess, sys, time
from torch.utils.file_baton import FileBaton
lock, builddir = sys.argv[1], sys.argv[2]
b = FileBaton(lock)
assert b.try_acquire(), "could not acquire"
# Stands in for cpp_extension._run_ninja_build: subprocess.run(['ninja'],
# cwd=build_directory). No process group, no PDEATHSIG -- it outlives us.
child = subprocess.Popen(["sleep", "120"], cwd=builddir)
print("READY", child.pid, flush=True)
time.sleep(120)
'''


def test_a_killed_builder_with_a_live_toolchain_is_not_collected():
    """The defect that made "unheld implies dead" false.

    FileBaton's fd is O_CLOEXEC, so ninja and nvcc never hold it -- only the
    Python process does. And ninja runs with cwd=build_directory, in no separate
    process group and with no PDEATHSIG. So when this harness kills a compile
    that blew its join budget, the fd dies while the toolchain keeps writing.
    An fd-only liveness check reads that as "dead" and frees the directory for a
    second builder: clobbered objects, a half-linked .so, or an undefined symbol
    blamed on the kernel under test.

    The orphans remain visible by cwd, which is what the sweep must key on.
    """
    import shutil as _sh
    import signal
    import subprocess

    import utils.torch_ext_cache as tec

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    holder = child_pid = None
    try:
        script = tmp / "orphan_holder.py"
        script.write_text(_ORPHAN_HOLDER)
        root = tmp / "torch_extensions"
        d = root / "py313_cu130" / "mid_build"
        d.mkdir(parents=True)
        lock = d / "lock"

        holder = subprocess.Popen([sys.executable, str(script), str(lock), str(d)],
                                  stdout=subprocess.PIPE, text=True)
        deadline, line = time.time() + 120, ""
        while time.time() < deadline and not line:
            line = (holder.stdout.readline() or "").strip()
            if holder.poll() is not None and not line:
                raise AssertionError("holder exited before acquiring")
        assert line.startswith("READY"), line
        child_pid = int(line.split()[1])

        # Kill the Python holder exactly as _bench_and_score / _preload_worker do.
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=30)
        stranded = time.time() - 300                   # well past the 60s floor
        os.utime(lock, (stranded, stranded))

        assert lock.exists(), "SIGKILL must strand the lock"
        st = lock.stat()
        fds = tec._open_inodes()
        assert (st.st_dev, st.st_ino) not in fds, (
            "premise check: with the Python holder dead, NOTHING holds the lock fd -- "
            "which is exactly why an fd-only check was wrong")

        with _captured() as out:
            removed = tec.sweep_unheld_batons(root=root, verbose=True)
        assert removed == [], (
            "freed a build directory an orphaned toolchain is still writing into")
        assert lock.exists()
        assert "still building in that directory" in out.text, out.text

        # Once the toolchain is gone too, the lock is genuinely collectable.
        os.kill(child_pid, signal.SIGKILL)
        for _ in range(50):
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        removed = tec.sweep_unheld_batons(root=root, verbose=False)
        assert [p.parent.name for p in removed] == ["mid_build"], removed
    finally:
        for pid in (child_pid,):
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if holder is not None and holder.poll() is None:
            holder.kill()
            holder.wait(timeout=30)
        _sh.rmtree(tmp, ignore_errors=True)


def test_a_lock_whose_build_dir_is_a_live_cwd_is_never_removed():
    """The same guard, cheaply: no torch, no subprocess kill dance."""
    import utils.torch_ext_cache as tec

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    cwd_before = os.getcwd()
    try:
        root = _make_cache(tmp, [("being_built", 600.0, False), ("dead", 600.0, False)])
        os.chdir(root / "py313_cu130" / "being_built")   # this process is the "ninja"
        removed = tec.sweep_unheld_batons(root=root, verbose=False)
        assert [p.parent.name for p in removed] == ["dead"], removed
        assert (root / "py313_cu130" / "being_built" / "lock").exists()
    finally:
        os.chdir(cwd_before)
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_documented_off_switch_disables_both_sweeps():
    """KERNELMEM_BATON_MAX_AGE=0 must stop ALL lock removal, not half of it."""
    from utils.torch_ext_cache import sweep_unheld_batons

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("old", 9 * _HOUR, False)])
        with _EnvGuard(KERNELMEM_BATON_MAX_AGE="0"):
            assert sweep_stale_batons(root=root, verbose=False) == []
            assert sweep_unheld_batons(root=root, verbose=False) == [], (
                "the per-round sweep ignored the documented off switch")
        assert _locks(root) == {"old"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_per_round_sweep_confirms_before_unlinking():
    """A holder finishing release() between the two looks must not be clobbered.

    Age cannot cover this window (it runs from O_CREAT, so any real build has
    cleared the floor long before release()). The second observation can: a
    holder completes os.remove microseconds after os.close, so by the confirm
    pass the path is gone -- and if a NEW builder has taken the directory, the
    inode has changed and the candidate is dropped.
    """
    import utils.torch_ext_cache as tec

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        real_probe = tec._open_inodes

        # (a) The holder completes release() during the confirm delay: the lock
        #     simply vanishes and the candidate must be dropped, not chased.
        root = _make_cache(tmp, [("releasing", 600.0, False)])
        lock = root / "py313_cu130" / "releasing" / "lock"
        state = {"n": 0}

        def probe_then_release():
            state["n"] += 1
            result = real_probe()
            if state["n"] == 1:
                lock.unlink()                # os.remove inside release()
            return result

        tec._open_inodes = probe_then_release
        try:
            assert tec.sweep_unheld_batons(root=root, verbose=False) == []
        finally:
            tec._open_inodes = real_probe

        # (b) A builder shows up DURING the confirm delay -- the case that only
        #     a genuine second observation can catch. The first probe sees an
        #     unheld lock; a fd is opened partway through the sleep; the second
        #     probe must see it. Nothing is monkeypatched here, so removing the
        #     sleep-and-reprobe makes this fail rather than silently pass on the
        #     stale first snapshot.
        #
        #     Note also that ext4 hands back the identical inode on
        #     unlink+create, so inode identity could never have caught this;
        #     liveness is doing the work.
        root2 = _make_cache(tmp / "b", [("reacquired", 600.0, False)])
        lock2 = root2 / "py313_cu130" / "reacquired" / "lock"
        before_ino = lock2.stat().st_ino
        fds, errors = [], []

        def acquire_midway():
            try:
                time.sleep(tec._CONFIRM_DELAY_S * 0.3)
                fds.append(os.open(lock2, os.O_RDONLY))
            except BaseException as exc:            # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=acquire_midway)
        t.start()
        try:
            removed = tec.sweep_unheld_batons(root=root2, verbose=False)
        finally:
            t.join(timeout=30)
            for fd in fds:
                os.close(fd)

        assert not errors, errors
        assert fds, "the stand-in builder never acquired; test proves nothing"
        assert removed == [], (
            "deleted a lock a builder acquired during the confirm delay -- the "
            "second observation is not happening")
        assert lock2.exists() and lock2.stat().st_ino == before_ino
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_per_round_sweep_keeps_an_acquisition_floor():
    """The floor covers the ACQUISITION side, not release().

    It is tempting to say the floor covers the close->remove window inside
    release(). It does not, and believing so was a real bug: age runs from
    O_CREAT, so a build longer than the floor has already cleared it by the time
    it reaches release(). The confirm pass covers that window; the floor covers
    a lock created after the one-shot /proc snapshot, which has age ~0.
    """
    from utils.torch_ext_cache import _RACE_FLOOR_S, sweep_unheld_batons

    assert _RACE_FLOOR_S >= 30.0
    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        root = _make_cache(tmp, [("just_created", 0.0, False),
                                 ("a_moment_ago", _RACE_FLOOR_S - 5, False)])
        assert sweep_unheld_batons(root=root, verbose=False) == []
        assert _locks(root) == {"just_created", "a_moment_ago"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_without_proc_the_per_round_sweep_removes_nothing():
    """It must NOT fall back to age the way the startup sweep does.

    The startup sweep can fall back safely because its gate is 3600s. This one
    runs at 60s, so falling back would mean deleting minute-old locks on no
    evidence at all -- precisely the corruption the module exists to prevent.
    """
    import utils.torch_ext_cache as tec

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    real = tec._open_inodes
    try:
        root = _make_cache(tmp, [("old_and_unheld", 9 * _HOUR, False)])
        tec._open_inodes = lambda: None                  # stand in for "no /proc"
        with _captured() as out:
            removed = tec.sweep_unheld_batons(root=root, verbose=True)
        assert removed == [], "removed a lock without any evidence its holder died"
        assert _locks(root) == {"old_and_unheld"}
        assert "none will be removed" in out.text, out.text
        # The startup sweep, with its 3600s gate, may still take it.
        tec._open_inodes = lambda: None
        assert len(tec.sweep_stale_batons(root=root, verbose=False)) == 1
    finally:
        tec._open_inodes = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_per_round_sweep_runs_every_round_before_any_compile():
    src = (Path(__file__).resolve().parent.parent / "main_memory_latest.py").read_text()
    assert "sweep_unheld_batons" in src
    loop = src.index("for round_idx in range(start_round, args.round):")
    call = src.index("sweep_unheld_batons()")
    assert call > loop, "the per-round sweep must be INSIDE the round loop"
    # Before the seed branch, so round 0 is covered too.
    assert call < src.index("if round_idx == 0:")


def test_without_proc_the_sweep_falls_back_to_age_and_says_so():
    """The liveness gate degrades to the age gate rather than silently no-opping.

    Refusing to sweep when liveness is unknowable would be the conservative
    choice for a destructive operation, but it would also turn the sweep into a
    no-op wherever /proc is absent -- silently reinstating the hang this module
    exists to prevent. So we fall back to age alone and print a NOTE, and the
    NOTE is the only thing distinguishing "cannot tell" from "nothing is held".
    """
    import utils.torch_ext_cache as tec

    # The probe must report "cannot tell", not "nothing is held", when there is
    # no /proc -- reachable here because _open_inodes takes the root as an arg.
    assert tec._open_inodes(Path("/definitely/not/proc")) is None
    assert tec._open_inodes() is not None, "on this Linux host /proc must be readable"

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    real = tec._open_inodes
    try:
        root = _make_cache(tmp, [("orphan", 9 * _HOUR, False), ("fresh", 30.0, False)])
        tec._open_inodes = lambda: None            # stand in for "no /proc"
        with _captured() as out:
            removed = tec.sweep_stale_batons(root=root, verbose=True)
        assert {p.parent.name for p in removed} == {"orphan"}, removed
        assert _locks(root) == {"fresh"}, "the age gate must still apply"
        assert "/proc is unavailable" in out.text, (
            "falling back to age alone must be stated, not silent")
    finally:
        tec._open_inodes = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_liveness_probe_is_walked_at_most_once_per_sweep():
    """/proc is walked lazily and once: never per-lock, never when nothing is old."""
    import utils.torch_ext_cache as tec

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    real = tec._open_inodes
    calls = []
    try:
        tec._open_inodes = lambda: (calls.append(1), real())[1]

        root = _make_cache(tmp, [(f"fresh_{i}", 60.0, False) for i in range(5)])
        tec.sweep_stale_batons(root=root, verbose=False)
        assert calls == [], "nothing cleared the age gate, so /proc must not be walked"

        root2 = _make_cache(tmp / "b", [(f"old_{i}", 9 * _HOUR, False) for i in range(20)])
        tec.sweep_stale_batons(root=root2, verbose=False)
        assert len(calls) == 1, f"/proc walked {len(calls)} times for 20 locks"
    finally:
        tec._open_inodes = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_tuned_threshold_cannot_drop_below_a_legitimate_build():
    """A floor on the knob, so tuning it cannot cause the bug it prevents.

    _bench_and_score allows a build 1200s before it kills it. Without a floor,
    KERNELMEM_BATON_MAX_AGE=300 would delete the lock of a perfectly normal
    10-minute nvcc compile.
    """
    from utils.torch_ext_cache import _MIN_MAX_AGE_S, _clamp_max_age

    assert _MIN_MAX_AGE_S >= 1200.0, "the floor must clear the p.join budget"
    for tiny in (1.0, 60.0, 300.0, _MIN_MAX_AGE_S - 1):
        assert _clamp_max_age(tiny) == _MIN_MAX_AGE_S
    for big in (_MIN_MAX_AGE_S, _DEFAULT_MAX_AGE_S, 99999.0):
        assert _clamp_max_age(big) == big
    for off in (0.0, -1.0):
        assert _clamp_max_age(off) == 0.0, "0 must still mean disabled, not floored"

    tmp = Path(tempfile.mkdtemp(prefix="ext_cache_test_"))
    try:
        # A 10-minute-old lock is a plausible live build and must survive both
        # the env knob and the explicit argument.
        root = _make_cache(tmp, [("compiling", 10 * 60, False)])
        with _EnvGuard(KERNELMEM_BATON_MAX_AGE="300"):
            assert sweep_stale_batons(root=root, verbose=False) == []
        assert sweep_stale_batons(root=root, max_age_s=300, verbose=False) == []
        assert _locks(root) == {"compiling"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_extensions_root_follows_the_env_and_defaults_to_the_torch_location():
    with _EnvGuard(TORCH_EXTENSIONS_DIR="/custom/place"):
        assert extensions_root() == Path("/custom/place")
    for blank in ("", "   "):
        with _EnvGuard(TORCH_EXTENSIONS_DIR=blank):
            assert extensions_root() == Path.home() / ".cache" / "torch_extensions"
    with _EnvGuard(TORCH_EXTENSIONS_DIR=None):
        assert extensions_root() == Path.home() / ".cache" / "torch_extensions"


def test_the_harness_still_builds_in_the_shared_cache():
    """The fix is agent-side only.

    If the harness process itself ever inherited a per-agent build dir, its
    kernels would be rebuilt from scratch every round and, worse, land in a
    directory that is rmtree'd out from under it.
    """
    import agents.query_server as qs

    before = os.environ.get("TORCH_EXTENSIONS_DIR")
    workdir = tempfile.mkdtemp(prefix="kernelmem_agent_test_")
    try:
        qs._agent_build_env(workdir)
        assert os.environ.get("TORCH_EXTENSIONS_DIR") == before, (
            "_agent_build_env must not mutate the harness's own environment")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_the_sweep_runs_before_any_round_compiles():
    """Pin the wiring: a resumed run must sweep too, not just a fresh one."""
    src = (Path(__file__).resolve().parent.parent / "main_memory_latest.py").read_text()
    assert "from utils.torch_ext_cache import sweep_stale_batons" in src
    call = src.index("sweep_stale_batons()")
    loop = src.index("for round_idx in range(start_round, args.round):")
    assert call < loop, "the sweep must run before the round loop, not inside it"
    # Between the resume block and the loop => a --resume run sweeps as well.
    resume = src.index("[resume] Restored")
    assert resume < call < loop


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failures = []
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except AssertionError as exc:
            failures.append(f"FAIL  {fn.__name__}: {exc}")
            print(failures[-1])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            print(failures[-1])

    print()
    print(f"{'FAILED' if failures else 'OK'} - "
          f"{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
