"""Pin the GPU core clock for the whole of a run, and refuse to measure without it.

Why this exists
---------------
Every ``config.json`` this repo has ever written records ``"lock_clocks": false``,
and ``REPORT_002_vae_resblock.md:547`` calls that out as the reason a trace cannot
be audited on its own: two numbers produced by the same kernel on the same card
are only comparable if the card was running at the same frequency both times, and
nothing in the artifacts said what that frequency was. A boost clock is set by
the driver from temperature, power and duty cycle -- none of which are recorded,
none of which the harness controls, and all of which differ between a kernel
measured first in a round and the same kernel re-measured an hour later.

Locking removes that free variable. After ``ensure_locked()`` the SM clock is a
constant of the run, it is written into the run's artifacts, and a later
re-measurement can be held to the same frequency instead of merely hoped to have
landed near it.

What it costs, honestly
-----------------------
The lock target is the frequency the card SUSTAINS under load, which is below the
short-burst boost clock. Absolute latencies therefore come out slower than an
unlocked short measurement, and a run takes correspondingly longer in wall clock.
It costs much less than that in *score*: the harness reports ``T_ref / T_k``, and
both sides are measured at the same locked frequency, so the ratio largely
divides the clock out. What is bought is that the two sides are also at the same
frequency as each other on a different day -- which is what "reproducible" means.

Design
------
* **Detect the device, then pick a frequency for THAT device.** There is no
  single "good" lock frequency: what a card holds is set by its own power limit
  and cooling. This 5090 holds 89% of its boost ceiling; an A100 holds ~76% of
  its. So the target comes from, in order, an operator override, a value
  measured on this machine for this exact card, a built-in preset for the model,
  and only then a class-based fraction of that card's own ceiling. An unknown
  card is measured (~45 s, once, cached in ``priors/clock_presets.json``) rather
  than guessed at -- see ``calibrate()``.

* **Core clock only.** ``nvidia-smi -lgc <mhz>,<mhz>`` pins min and max to the
  same value, so idle boost cannot move it between benchmarks either. Memory
  clock is pinned too, but only when the device exposes more than one supported
  memory clock -- a GeForce RTX 5090 exposes exactly one (14001 MHz), so there is
  nothing to pin and the module says so rather than reporting a phantom lock.

* **One locker per process tree.** The top-level run locks, exports
  ``KERNELMEM_CLOCKS_LOCKED_BY=<pid>``, and unlocks at exit. Children -- the
  spawned per-candidate benchmark processes, the lineage subprocesses -- inherit
  that variable, VERIFY the clock is still where the parent put it, and never
  lock or unlock themselves. A child that unlocked on exit would silently
  un-pin the rest of the parent's run.

* **Fail closed.** ``ensure_locked()`` raises unless the clock is actually
  pinned. Locking needs ``sudo``; a run that quietly continued at boost clock
  after the lock failed would produce exactly the unauditable trace this module
  exists to prevent. ``KERNELMEM_CLOCK_LOCK=0`` opts out deliberately, prints a
  banner, and stamps ``locked: false`` into the artifacts so the opt-out is
  visible in the record rather than inferred from its absence.

Setup (once per machine)
------------------------
``nvidia-smi -lgc`` is root-only::

    sudo bash scripts/install_clock_lock_sudoers.sh

which drops a ``/etc/sudoers.d`` rule permitting exactly the clock subcommands.

CLI
---
    python -m utils.clock_lock --status      # what card, what target, is it pinned
    python -m utils.clock_lock --calibrate   # measure this card, cache its target
    python -m utils.clock_lock --lock [--mhz 2407]
    python -m utils.clock_lock --unlock
"""
from __future__ import annotations

import atexit
import datetime
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Environment contract
# --------------------------------------------------------------------------
ENV_POLICY = "KERNELMEM_CLOCK_LOCK"        # "0"/"off"/"false"/"no" = deliberate opt-out
ENV_OWNER = "KERNELMEM_CLOCKS_LOCKED_BY"   # pid of the process that took the lock
ENV_STATE = "KERNELMEM_CLOCK_STATE"        # JSON snapshot, inherited by children
ENV_GPU_MHZ = "KERNELMEM_GPU_CLK_MHZ"      # override the target core clock
ENV_DRAM_MHZ = "KERNELMEM_DRAM_CLK_MHZ"    # override the target memory clock
ENV_KEEP = "KERNELMEM_CLOCK_KEEP"          # "1" = leave pinned at exit
ENV_AUTOCAL = "KERNELMEM_CLOCK_AUTOCAL"    # "0" = never auto-measure a new card

# Seconds to let the clock settle before verifying it took.
VERIFY_DELAY_S = 2.0
# Reported clocks are quantised (~8 MHz steps on Blackwell) and sampled while the
# GPU may be between kernels, so an exact match is the wrong test.
TOLERANCE_MHZ = 45


class ClockLockError(RuntimeError):
    """The GPU clock is not pinned and the run requires that it be."""


# --------------------------------------------------------------------------
# Lock targets
#
# The number is the clock the card HOLDS under sustained load, measured on the
# card itself -- not the boost ceiling, which no card holds, and not a round
# number picked for looking tidy. Locking above the sustained clock does not
# raise the clock; the driver clamps to what the power/thermal budget allows and
# the lock becomes a fiction that verification then fails on.
#
# To add a card: run a continuous load for ~2 minutes, watch
# `nvidia-smi --query-gpu=clocks.sm --format=csv`, and take the settled value.
# --------------------------------------------------------------------------
CLOCK_PRESETS: Dict[str, Dict[str, Optional[int]]] = {
    # NOTE: one frequency per DEVICE, never one frequency for all devices. What
    # a card can hold is a property of that part's power limit and cooling: this
    # 5090 holds 89% of its boost ceiling, an A100 holds ~76% of its. Applying
    # either number to the other card would pin one below what it can do and pin
    # the other above what it can hold -- and a pin above the sustainable clock
    # is not a pin at all, it just fails verification.
    # 2407 MHz is chosen for COMPARABILITY, not for throughput, and the
    # distinction is the whole reason this number is not simply "as fast as the
    # card can hold". This repo scores against
    #     T_SOL = max(FLOPs / 104.8e12, bytes / 1792e9)      (REPORT_002:523)
    # and that 104.8 TFLOPS is 170 SM x 256 FLOP/clk x 2.41 GHz -- a constant
    # with a clock baked into it. Run the card faster than 2.41 GHz and measured
    # times come in under a T_SOL that assumed 2.41, which is how a kernel came
    # to score 1.036, i.e. faster than the speed of light. Pinning at the bin
    # nearest 2.41 GHz (2407 is a supported bin; the driver does not round) makes
    # measured times directly comparable with t_sol.json and REPORT_002.
    #
    # It is comfortably holdable: measured 2026-08-14 on this box over 100 s of
    # continuous TF32 conv+GEMM, the card settles at 2763 MHz (floor 2752,
    # ceiling 3105) at 78 C, power-capped rather than heat-capped, so 2407 is
    # ~13% below what the card would sustain. That 13% is the wall-clock price of
    # the pin, paid deliberately so the numbers mean what the report says.
    # Memory: 190 of 191 loaded samples sat at 13801 MHz, never at the 14001 MHz
    # top bin, so 13801 is the clock to pin rather than the advertised maximum.
    "NVIDIA GeForce RTX 5090": {"gpu_mhz": 2407, "dram_mhz": 13801},
    "NVIDIA B200": {"gpu_mhz": 1500, "dram_mhz": 3996},
    "NVIDIA H100": {"gpu_mhz": 1410, "dram_mhz": 1593},
    "NVIDIA A100": {"gpu_mhz": 1065, "dram_mhz": 1215},
}

# Cards measured on THIS machine land here, keyed by their exact name, so a card
# absent from the table above is measured once and never guessed at again.
MEASURED_PRESETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "priors", "clock_presets.json")

# Last-resort fractions of the card's OWN boost ceiling, used only when a device
# is unknown and calibration was declined. Split by class because the classes
# behave differently: consumer parts advertise a burst ceiling they hold ~85-90%
# of, datacenter parts advertise a boost bin they hold ~75% of under a real load.
# Still a guess -- ensure_locked() prefers measuring the actual card.
FALLBACK_FRACTIONS = {"consumer": 0.85, "datacenter": 0.75}

# Substrings that mark a datacenter part; everything else is treated as consumer.
_DATACENTER_MARKERS = ("A100", "H100", "H200", "B100", "B200", "GB200", "A800",
                       "H800", "V100", "L40", "L4", "A40", "A30", "A10",
                       "TESLA", "QUADRO", "RTX PRO", "GH200")


def device_class(name: str) -> str:
    up = (name or "").upper()
    return "datacenter" if any(m in up for m in _DATACENTER_MARKERS) else "consumer"


# --------------------------------------------------------------------------
# nvidia-smi plumbing
# --------------------------------------------------------------------------
def _run(cmd: List[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _query(fields: str, smi_index: Optional[int] = None) -> List[List[str]]:
    """``nvidia-smi --query-gpu`` as rows of stripped strings. [] on any failure."""
    cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    if smi_index is not None:
        cmd += ["-i", str(smi_index)]
    try:
        res = _run(cmd)
    except (OSError, subprocess.SubprocessError):
        return []
    if res.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")]
            for line in res.stdout.strip().splitlines() if line.strip()]


def smi_index(device_idx: int = 0) -> int:
    """The nvidia-smi index for a torch device index.

    They are not the same number whenever ``CUDA_VISIBLE_DEVICES`` is set or
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` reorders the enumeration -- and both are set
    in parts of this harness. Locking the wrong GPU would silently leave the one
    being benchmarked unpinned, so match on UUID and only fall back to identity.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return device_idx
        uuid = str(getattr(torch.cuda.get_device_properties(device_idx), "uuid", "") or "")
    except Exception:
        return device_idx
    if not uuid:
        return device_idx
    for row in _query("index,uuid"):
        if len(row) >= 2 and uuid.replace("GPU-", "") in row[1].replace("GPU-", ""):
            try:
                return int(row[0])
            except ValueError:
                return device_idx
    return device_idx


def device_name(smi_idx: int = 0) -> str:
    rows = _query("name", smi_idx)
    return rows[0][0] if rows else ""


def current_clocks(smi_idx: int = 0) -> Dict[str, Any]:
    """Core / memory clock plus the context needed to explain an off-target read."""
    rows = _query("clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,"
                  "utilization.gpu,clocks_throttle_reasons.active", smi_idx)
    if not rows:
        return {}
    f = rows[0]

    def _int(x: str) -> Optional[int]:
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None

    return {"sm_mhz": _int(f[0]), "mem_mhz": _int(f[1]), "max_sm_mhz": _int(f[2]),
            "temp_c": _int(f[3]), "power_w": f[4] if len(f) > 4 else None,
            "util_pct": _int(f[5]) if len(f) > 5 else None,
            "throttle": f[6] if len(f) > 6 else ""}


def supported_clocks(smi_idx: int = 0) -> Tuple[List[int], List[int]]:
    """(memory clocks, core clocks) the driver will accept, descending."""
    cmd = ["nvidia-smi", "--query-supported-clocks=mem,gr",
           "--format=csv,noheader,nounits", "-i", str(smi_idx)]
    try:
        res = _run(cmd)
    except (OSError, subprocess.SubprocessError):
        return [], []
    if res.returncode != 0:
        return [], []
    mem, gr = set(), set()
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mem.add(int(parts[0]))
            gr.add(int(parts[1]))
        except ValueError:
            continue
    return sorted(mem, reverse=True), sorted(gr, reverse=True)


# --------------------------------------------------------------------------
# Privileged path
#
# Pinning a clock is root-only. Three ways in, in order of preference:
#
#   1. the root-owned wrapper installed by scripts/install_clock_lock_sudoers.sh,
#      granted NOPASSWD. Preferred because the sudoers rule then names one
#      binary with a validated argument grammar, instead of nvidia-smi with
#      wildcards -- and `nvidia-smi -f <path>` writes to <path> as root, so a
#      wildcard rule covering `-lgc` covers that too;
#   2. running as root already (container / cloud box), where sudo is pointless;
#   3. blanket passwordless sudo, which some machines already have.
# --------------------------------------------------------------------------
WRAPPER = "/usr/local/sbin/kernelmem-gpu-clock"


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _have_wrapper() -> bool:
    return os.path.isfile(WRAPPER) and os.access(WRAPPER, os.X_OK)


def privilege_mode() -> str:
    """Which of the three routes this process will take: wrapper/root/sudo/none."""
    if _have_wrapper():
        return "wrapper"
    if _is_root():
        return "root"
    return "sudo"


def _privileged(op: str, smi_idx: int, mhz: Optional[int] = None) -> List[str]:
    """argv for a privileged clock operation, for the route available here."""
    mode = privilege_mode()
    if mode == "wrapper":
        argv = ["sudo", "-n", WRAPPER, op, str(smi_idx)]
        if op == "lock":
            argv.append(str(mhz))
        return argv

    prefix = [] if mode == "root" else ["sudo", "-n"]
    if op == "lock":
        return prefix + ["nvidia-smi", "-i", str(smi_idx), "-lgc", f"{mhz},{mhz}"]
    if op == "unlock":
        return prefix + ["nvidia-smi", "-i", str(smi_idx), "-rgc"]
    return prefix + ["nvidia-smi", "-i", str(smi_idx), "--query-gpu=clocks.sm",
                     "--format=csv,noheader,nounits"]


def _privileged_memory(smi_idx: int, mhz: Optional[int], reset: bool = False) -> Optional[List[str]]:
    """argv for pinning/releasing the memory clock.

    Worth pinning wherever the card offers a choice: this 5090 idles its memory
    at 405 MHz and runs it at 13801 under load, and half this repo's workloads
    are memory-bandwidth bound, so an unpinned memory clock is the same free
    variable as an unpinned core clock. ``unlock`` on the wrapper route already
    resets it, hence None for the reset case there.
    """
    mode = privilege_mode()
    if mode == "wrapper":
        return None if reset else ["sudo", "-n", WRAPPER, "lockmem",
                                   str(smi_idx), str(mhz)]
    prefix = [] if mode == "root" else ["sudo", "-n"]
    if reset:
        return prefix + ["nvidia-smi", "-i", str(smi_idx), "-rmc"]
    return prefix + ["nvidia-smi", "-i", str(smi_idx), "-lmc", f"{mhz},{mhz}"]


def can_lock(smi_idx: int = 0) -> bool:
    """Whether a clock can actually be pinned here, without a password prompt."""
    try:
        return _run(_privileged("status", smi_idx), timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def required() -> bool:
    """True unless the operator deliberately opted out for this process tree."""
    return os.environ.get(ENV_POLICY, "1").strip().lower() not in ("0", "off", "false", "no")


def _autocal() -> bool:
    """Measure an unknown card automatically? On unless switched off."""
    return os.environ.get(ENV_AUTOCAL, "1").strip().lower() not in ("0", "off", "false", "no")


def owner_pid() -> Optional[int]:
    raw = os.environ.get(ENV_OWNER, "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Ownership that survives the owner.
#
# Releasing the pin is already handled for every exit the process can observe:
# atexit for a normal return, SIGINT/SIGTERM handlers for a stop. What none of
# that can cover is SIGKILL, an OOM kill, or the power going out -- no code of
# ours runs, and the card stays pinned with nothing left to say who pinned it.
#
# That is not a hypothetical either. ENV_OWNER lives in the environment, so it
# dies with the process, and the NEXT run therefore sees a pinned card with no
# owner and takes the "pre-existing external lock" branch -- adopting a dead
# run's frequency as though an operator had chosen it. Measured on 2026-08-16: a
# SIGKILLed run left the card at 2482 MHz, the next run adopted it instead of
# applying the 2407 MHz target, and its reference profile read 2224 us against
# the ~2526 us the same reference takes at target. Every score from it would
# have been inflated against a T_SOL that assumes 2.41 GHz.
#
# So ownership is written to a FILE next to the machine, not into the
# environment. A pin whose recorded owner is gone is stale: ours to reclaim at
# our own target, rather than a stranger's choice to respect. In /tmp because
# the GPU is machine-wide and so is the claim on it, and because a reboot -- the
# one event that also resets the clocks -- clears it for free.
# --------------------------------------------------------------------------
def _lockfile(smi_idx: int) -> Path:
    return Path(tempfile.gettempdir()) / f"kernelmem_clock_gpu{smi_idx}.json"


def _proc_start_ticks(pid: int) -> Optional[int]:
    """Field 22 of /proc/<pid>/stat: when this pid started, in clock ticks.

    Recorded alongside the pid because pids are reused. Without it, a stale
    record whose pid has been recycled by an unrelated process reads as "the
    owner is alive", and the pin is left in place forever.
    """
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
            data = fh.read()
        # comm can contain spaces and parens; everything after the last ')' is
        # positional, and starttime is the 20th field of that remainder.
        return int(data[data.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _write_owner(smi_idx: int, gpu_mhz: Optional[int]) -> None:
    """Claim the pin on disk. Never raises: failing to record must not fail a lock."""
    pid = os.getpid()
    try:
        _lockfile(smi_idx).write_text(json.dumps({
            "pid": pid,
            "start_ticks": _proc_start_ticks(pid),
            "target_gpu_mhz": gpu_mhz,
            "gpu_name": device_name(smi_idx),
            "taken": datetime.datetime.now().isoformat(timespec="seconds"),
            "argv0": os.path.basename(sys.argv[0] if sys.argv else "?"),
        }, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _read_owner(smi_idx: int) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_lockfile(smi_idx).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _clear_owner(smi_idx: int) -> None:
    try:
        _lockfile(smi_idx).unlink()
    except OSError:
        pass


def _owner_alive(rec: Optional[Dict[str, Any]]) -> bool:
    """Is the process that took this pin still running?

    Both halves must agree. `kill(pid, 0)` alone answers "some process has this
    pid", which after pid reuse is a different question from the one being asked.
    A record written before start_ticks was captured (or on a system without
    /proc) falls back to the liveness check alone, which is the safe direction:
    it can only make us treat a stale pin as live, never the reverse.
    """
    if not rec:
        return False
    pid = rec.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True                      # someone else's process, but it exists
    want = rec.get("start_ticks")
    if want is None:
        return True
    now = _proc_start_ticks(pid)
    return now is None or now == want


def reclaim_stale(smi_idx: int = 0, *, verbose: bool = True) -> bool:
    """Release a pin whose owner is dead. True if one was reclaimed.

    Called before the external-lock branch, so a run that was killed cannot
    hand its frequency to the next run as if it were policy.
    """
    rec = _read_owner(smi_idx)
    if rec is None:
        return False
    if _owner_alive(rec):
        return False
    if verbose:
        print(f"[clock] a previous run (pid {rec.get('pid')}, started "
              f"{rec.get('taken')}) left this GPU pinned at "
              f"{rec.get('target_gpu_mhz')} MHz and is no longer running. "
              f"Releasing that stale pin rather than adopting it.", flush=True)
    unlock(smi_idx, keep_claim=True)
    _clear_owner(smi_idx)
    return True


def _snap(target: int, supported: List[int]) -> int:
    """The nearest clock the driver actually offers, so the record is exact."""
    if not supported:
        return target
    return min(supported, key=lambda c: (abs(c - target), c))


def _load_measured() -> Dict[str, Dict[str, Any]]:
    """Presets measured on this machine, keyed by exact device name."""
    try:
        with open(MEASURED_PRESETS) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_measured(name: str, entry: Dict[str, Any]) -> None:
    data = _load_measured()
    data[name] = entry
    try:
        os.makedirs(os.path.dirname(MEASURED_PRESETS), exist_ok=True)
        with open(MEASURED_PRESETS, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        print(f"[clock] could not cache the calibration ({exc}); it will be "
              f"re-measured next run", flush=True)


def measure_sustained_mhz(device_idx: int = 0, seconds: float = 45.0,
                          verbose: bool = True) -> Dict[str, Any]:
    """Run a load on THIS card and report the core clock it actually holds.

    The only honest way to pick a lock target for an unfamiliar device. The
    boost ceiling is a burst number and differs from the sustainable clock by
    ~11% on a 5090 and ~25% on an A100 -- a single fraction applied to both is
    wrong for at least one of them, so measure instead of extrapolating.
    """
    import threading

    import torch

    smi_idx = smi_index(device_idx)
    samples: List[Dict[str, Any]] = []
    stop = threading.Event()

    def _sample() -> None:
        while not stop.is_set():
            obs = current_clocks(smi_idx)
            if obs.get("sm_mhz") is not None:
                samples.append(obs)
            time.sleep(0.5)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(f"cuda:{device_idx}")
    # Big enough to keep every SM busy and hold the card at its power limit,
    # small enough to fit a modest card. TF32 GEMM is the densest sustained load
    # available without writing a kernel, and it is the same numeric class the
    # harness benchmarks in.
    n = 8192
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)

    th = threading.Thread(target=_sample, daemon=True)
    th.start()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            for _ in range(4):
                torch.mm(a, b)
            torch.cuda.synchronize()
    finally:
        stop.set()
        th.join(timeout=5)
        del a, b
        torch.cuda.empty_cache()

    loaded = [s for s in samples if (s.get("util_pct") or 0) > 50] or samples
    if not loaded:
        return {"ok": False, "error": "no clock samples were collected"}
    # The last third is the settled state: the first samples still carry the
    # boost the card gives a cold, briefly-loaded GPU.
    settled = loaded[max(1, 2 * len(loaded) // 3):]
    tail = sorted(s["sm_mhz"] for s in settled)
    settled_floor = tail[0]
    settled_p50 = tail[len(tail) // 2]
    # The memory clock is measured the same way and for the same reason: it has
    # its own bins (this card idles at 405 MHz and loads at 13801), and a
    # bandwidth-bound kernel is timed against whichever bin it happened to get.
    mem_tail = sorted(s["mem_mhz"] for s in settled if s.get("mem_mhz") is not None)
    temps = [s.get("temp_c") for s in loaded if s.get("temp_c") is not None]
    out = {
        "ok": True,
        "gpu_name": device_name(smi_idx),
        "n_samples": len(loaded),
        "seconds": round(time.monotonic() - t0, 1),
        "settled_floor_mhz": settled_floor,
        "settled_median_mhz": settled_p50,
        "settled_mem_median_mhz": mem_tail[len(mem_tail) // 2] if mem_tail else None,
        "max_sm_mhz": loaded[0].get("max_sm_mhz"),
        "temp_max_c": max(temps) if temps else None,
        "throttle": sorted({s.get("throttle", "") for s in loaded}),
    }
    if verbose:
        print(f"[clock] {out['gpu_name']}: holds {settled_p50} MHz core / "
              f"{out['settled_mem_median_mhz']} MHz mem under load "
              f"(core floor {settled_floor}, ceiling {out['max_sm_mhz']}, "
              f"{out['temp_max_c']} C) over {out['seconds']} s", flush=True)
    return out


def calibrate(device_idx: int = 0, seconds: float = 45.0,
              verbose: bool = True) -> Optional[int]:
    """Measure this card, cache a lock target for it, and return that target.

    The target is just under the measured floor: a pin at or above the floor is
    the first thing a hot, power-capped card drops below, and a pin the card
    cannot hold fails verification and stops the run.

    This optimises for holdability, which is the right default for a card nobody
    has characterised. It is NOT the right answer where a roofline constant has a
    clock baked into it -- the 5090 preset is pinned at 2407 MHz for exactly that
    reason, well under what it sustains, so that measured times stay comparable
    with a T_SOL derived at 2.41 GHz. If this card is scored against such a
    model, set that model's clock with KERNELMEM_GPU_CLK_MHZ instead.
    """
    from utils import gpu_lock

    with gpu_lock.gpu_section("clock calibration"):
        m = measure_sustained_mhz(device_idx, seconds=seconds, verbose=verbose)
    if not m.get("ok"):
        return None
    smi_idx = smi_index(device_idx)
    mem_clocks, gr_clocks = supported_clocks(smi_idx)
    target = _snap(int(m["settled_floor_mhz"] * 0.98), gr_clocks)
    # Memory clocks are coarse bins, not a continuum, so take the bin the card
    # actually ran in rather than shaving a margin off it and landing lower.
    mem_target = (_snap(int(m["settled_mem_median_mhz"]), mem_clocks)
                  if m.get("settled_mem_median_mhz") and len(mem_clocks) > 1 else None)
    entry = dict(m, gpu_mhz=target, dram_mhz=mem_target,
                 measured_on=time.strftime("%Y-%m-%d"))
    _save_measured(m["gpu_name"], entry)
    if verbose:
        print(f"[clock] calibrated {m['gpu_name']} -> lock target {target} MHz "
              f"(cached in {MEASURED_PRESETS})", flush=True)
    return target


def resolve_target(smi_idx: int = 0) -> Tuple[Optional[int], Optional[int], str]:
    """(core MHz, memory MHz or None, how it was chosen) for THIS device.

    Precedence, most specific first:
      1. an explicit operator override for this run;
      2. a value measured on this machine for this exact card;
      3. a built-in preset for this model;
      4. a fraction of this card's own ceiling, by device class -- a guess, and
         labelled as one so ensure_locked() can offer to measure instead.
    """
    name = device_name(smi_idx)
    mem_clocks, gr_clocks = supported_clocks(smi_idx)

    env_gpu = os.environ.get(ENV_GPU_MHZ, "").strip()
    env_dram = os.environ.get(ENV_DRAM_MHZ, "").strip()
    if env_gpu:
        try:
            gpu = _snap(int(env_gpu), gr_clocks)
            dram = int(env_dram) if env_dram else None
            return gpu, dram, f"{ENV_GPU_MHZ}={env_gpu}"
        except ValueError:
            pass

    measured = _load_measured().get(name)
    if measured and measured.get("gpu_mhz"):
        return (_snap(int(measured["gpu_mhz"]), gr_clocks), measured.get("dram_mhz"),
                f"measured on this machine {measured.get('measured_on', '')} "
                f"(holds {measured.get('settled_median_mhz')} MHz)".strip())

    for key, preset in CLOCK_PRESETS.items():
        if key in name:
            gpu = preset["gpu_mhz"]
            dram = preset["dram_mhz"]
            # A preset for a card with one memory clock means "do not pin memory",
            # and pinning is skipped below regardless of what the table says.
            return (_snap(gpu, gr_clocks) if gpu else None, dram,
                    f"preset for {key}")

    if gr_clocks:
        cls = device_class(name)
        frac = FALLBACK_FRACTIONS[cls]
        target = _snap(int(round(frac * max(gr_clocks))), gr_clocks)
        return target, None, (f"UNCALIBRATED {name!r}: guessed {frac:.0%} of this "
                              f"card's own {max(gr_clocks)} MHz ceiling ({cls} "
                              f"class) -- run `python -m utils.clock_lock "
                              f"--calibrate` to measure it instead")
    return None, None, f"no supported-clock list available for {name!r}"


# --------------------------------------------------------------------------
# Lock / verify / unlock
# --------------------------------------------------------------------------
def verify(gpu_mhz: int, smi_idx: int = 0,
           tolerance_mhz: int = TOLERANCE_MHZ) -> Tuple[bool, Dict[str, Any]]:
    """Is the core clock actually sitting on the target right now?"""
    obs = current_clocks(smi_idx)
    sm = obs.get("sm_mhz")
    ok = sm is not None and abs(sm - gpu_mhz) <= tolerance_mhz
    return ok, obs


def detect_external_lock(smi_idx: int = 0) -> Dict[str, Any]:
    """Is the clock already pinned by something outside this process tree?

    Machines that manage clocks centrally -- a systemd unit, a cluster prolog, or
    an operator who ran ``nvidia-smi -lgc`` by hand -- satisfy the requirement
    without the harness doing anything, and a run must not fail its preflight
    just because it was not the one that took the lock.

    Two independent signals, because neither is conclusive alone:

    * the driver reports the applications-clocks event reason active, which is
      what NVML raises for a ``-lgc`` style lock;
    * the clock is idling at a frequency an unlocked card would not idle at.
      This card drops to ~675 MHz when idle and unlocked, so a high, steady idle
      clock is real evidence of a pin rather than a coincidence.
    """
    rows = _query("clocks_event_reasons.applications_clocks_setting", smi_idx)
    reason = rows[0][0] if rows and rows[0] else ""
    reason_active = reason.strip().lower() == "active"

    obs = current_clocks(smi_idx)
    sm, mx = obs.get("sm_mhz"), obs.get("max_sm_mhz")

    # Steady across two samples a moment apart: a boosting card wanders, a pinned
    # one does not. Only meaningful while the GPU is not under our own load.
    steady = False
    if sm is not None:
        time.sleep(0.4)
        again = current_clocks(smi_idx).get("sm_mhz")
        steady = again is not None and abs(again - sm) <= 15

    idle_but_high = bool(
        sm is not None and mx and (obs.get("util_pct") or 0) < 20 and sm >= 0.6 * mx
    )
    locked = bool(reason_active or (steady and idle_but_high))
    evidence = []
    if reason_active:
        evidence.append("driver reports applications-clocks setting active")
    if steady and idle_but_high:
        evidence.append(f"idle clock steady at {sm} MHz "
                        f"({sm / mx:.0%} of the {mx} MHz ceiling)")
    return {"locked": locked, "mhz": sm, "evidence": "; ".join(evidence),
            "observed": obs}


def lock(smi_idx: int = 0, gpu_mhz: Optional[int] = None,
         dram_mhz: Optional[int] = None) -> Dict[str, Any]:
    """Pin the clock on one GPU. Returns a state dict; never raises."""
    mem_clocks, gr_clocks = supported_clocks(smi_idx)
    why = "explicit argument"
    if gpu_mhz is None:
        gpu_mhz, dram_mhz, why = resolve_target(smi_idx)

    state: Dict[str, Any] = {
        "locked": False,
        "gpu_name": device_name(smi_idx),
        "smi_index": smi_idx,
        "target_gpu_mhz": gpu_mhz,
        "target_dram_mhz": None,
        "max_gpu_mhz": max(gr_clocks) if gr_clocks else None,
        "chosen_by": why,
        "dram_note": "",
        "error": "",
    }
    if gpu_mhz is None:
        state["error"] = "could not resolve a target clock (is nvidia-smi present?)"
        return state

    state["privilege"] = privilege_mode()
    try:
        res = _run(_privileged("lock", smi_idx, gpu_mhz), timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        state["error"] = f"{exc.__class__.__name__}: {exc}"
        return state
    if res.returncode != 0:
        state["error"] = (res.stderr or res.stdout or "").strip() or \
                         f"clock lock command exited {res.returncode}"
        return state

    # Memory clock: only meaningful where the card offers a choice. On a GeForce
    # part there is exactly one supported memory clock, so -lmc pins nothing and
    # tends to fail outright; recording why beats reporting a lock that is not one.
    if len(mem_clocks) > 1:
        want = _snap(dram_mhz, mem_clocks) if dram_mhz else max(mem_clocks)
        argv = _privileged_memory(smi_idx, want)
        if argv is None:
            state["dram_note"] = "memory clock not pinned on this privilege route"
        else:
            try:
                mres = _run(argv, timeout=60)
                if mres.returncode == 0:
                    state["target_dram_mhz"] = want
                else:
                    state["dram_note"] = ((mres.stderr or mres.stdout or "").strip()
                                          or f"-lmc exited {mres.returncode}")
            except (OSError, subprocess.SubprocessError) as exc:
                state["dram_note"] = f"{exc.__class__.__name__}: {exc}"
    else:
        only = mem_clocks[0] if mem_clocks else None
        state["dram_note"] = (f"single supported memory clock ({only} MHz); "
                              f"nothing to pin" if only else
                              "no supported memory clocks reported")

    time.sleep(VERIFY_DELAY_S)
    ok, obs = verify(gpu_mhz, smi_idx)
    state["locked"] = bool(ok)
    state["observed"] = obs
    if ok:
        # Claim it on disk the moment the pin is real, so that even a SIGKILL one
        # instruction later leaves a record saying who to blame and what to reclaim.
        _write_owner(smi_idx, gpu_mhz)
    if not ok:
        state["error"] = (f"clock did not settle on {gpu_mhz} MHz "
                          f"(observed {obs.get('sm_mhz')} MHz, "
                          f"throttle={obs.get('throttle') or 'none'})")
        unlock(smi_idx)
    return state


def unlock(smi_idx: int = 0, *, keep_claim: bool = False) -> None:
    """Release the pin. Best-effort: a failed unlock must not fail a finished run.

    Drops the on-disk claim too, so a released pin cannot later be mistaken for a
    stale one. *keep_claim* exists for reclaim_stale, which clears the record
    itself after deciding the record was the thing being acted on.
    """
    if not keep_claim:
        _clear_owner(smi_idx)
    try:
        _run(_privileged("unlock", smi_idx), timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass
    argv = _privileged_memory(smi_idx, None, reset=True)
    if argv is not None:
        try:
            _run(argv, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass


_ORIGINAL_HANDLERS: Dict[int, Any] = {}
_WARNED_OPT_OUT = False


def _release_on_exit(smi_idx: int) -> None:
    """Unlock at exit, including on SIGINT/SIGTERM, only in the locking process.

    Releasing is the default because a pinned card does not drop to idle clocks:
    this 5090 idles at ~675 MHz unlocked and would sit at 2407 MHz around the
    clock otherwise, burning power and heat between runs. ``KERNELMEM_CLOCK_KEEP=1``
    holds the pin for back-to-back runs, where the churn is the bigger cost.
    """
    def _cleanup() -> None:
        if owner_pid() != os.getpid():
            return
        if os.environ.get(ENV_KEEP, "").strip().lower() in ("1", "yes", "true", "on"):
            # Deliberately kept, so drop the claim on the way out. Leaving it would
            # make the next run read this as a dead owner and reclaim the very pin
            # KEEP exists to preserve -- the two features would cancel out.
            _clear_owner(smi_idx)
            print(f"[clock] leaving the clock pinned ({ENV_KEEP}=1); release it with "
                  f"`python -m utils.clock_lock --unlock`", flush=True)
            return
        unlock(smi_idx)                       # also drops the on-disk claim
        os.environ.pop(ENV_OWNER, None)

    atexit.register(_cleanup)

    def _handler(signum, frame):
        _cleanup()
        prev = _ORIGINAL_HANDLERS.get(signum)
        if callable(prev):
            prev(signum, frame)
            return
        # Re-raise through the default disposition so the exit status still says
        # "killed by signal" -- Ctrl-C on a run must not look like a clean exit.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            _ORIGINAL_HANDLERS[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not the main thread; atexit still covers the normal path


# --------------------------------------------------------------------------
# The call every entry point makes
# --------------------------------------------------------------------------
def state() -> Dict[str, Any]:
    """The lock state of this process tree, for stamping into artifacts."""
    raw = os.environ.get(ENV_STATE, "")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {"locked": False, "reason": "clock lock was never established"}


def describe(st: Optional[Dict[str, Any]] = None) -> str:
    st = state() if st is None else st
    if not st.get("locked"):
        return f"[clock] NOT LOCKED -- {st.get('error') or st.get('reason') or 'unknown'}"
    obs = st.get("observed") or {}
    dram = f", mem {st['target_dram_mhz']} MHz" if st.get("target_dram_mhz") else ""
    who = "" if st.get("owned_by_this_run", True) else " [pre-existing]"
    return (f"[clock] locked{who} {st['gpu_name']} to {st['target_gpu_mhz']} MHz"
            f"{dram} (ceiling {st.get('max_gpu_mhz')} MHz, observed "
            f"{obs.get('sm_mhz')} MHz) -- {st.get('chosen_by')}")


def _setup_help(err: str) -> str:
    return (
        "GPU clock is not locked, and every run in this repo requires it.\n"
        f"Reason: {err}\n\n"
        "An unlocked run is not reproducible: the driver picks the clock from "
        "temperature, power and duty cycle, none of which the harness controls or "
        "records, so the same kernel measured twice is measured on two different "
        "machines as far as the numbers are concerned.\n\n"
        "Fix (once per machine -- pinning a clock is root-only):\n"
        "    sudo bash scripts/install_clock_lock_sudoers.sh\n\n"
        "Then re-run. To check: python -m utils.clock_lock --status\n\n"
        "To measure without a lock ON PURPOSE (the result is not comparable with "
        "locked runs and is stamped 'locked: false' in the artifacts):\n"
        f"    {ENV_POLICY}=0 <your command>\n"
        f"To pin a different frequency: {ENV_GPU_MHZ}=<mhz> <your command>\n"
        f"To leave the clock pinned between back-to-back runs: {ENV_KEEP}=1"
    )


def ensure_locked(device_idx: int = 0, what: str = "run",
                  verbose: bool = True) -> Dict[str, Any]:
    """Guarantee the GPU core clock is pinned for this process tree.

    Idempotent and inheritance-aware:

    * the process that locks records itself in ``KERNELMEM_CLOCKS_LOCKED_BY`` and
      unlocks at exit;
    * a child that inherits it re-verifies the clock and returns -- it neither
      locks nor unlocks, because unlocking would un-pin its parent's run;
    * with ``KERNELMEM_CLOCK_LOCK=0`` it prints a banner and returns an
      explicitly-unlocked state instead of raising.

    Raises ``ClockLockError`` when the lock is required and could not be had.
    """
    idx = smi_index(device_idx)

    if not required():
        st = {"locked": False, "policy": "opted out via " + ENV_POLICY,
              "gpu_name": device_name(idx), "smi_index": idx,
              "observed": current_clocks(idx),
              "reason": "clock lock explicitly disabled for this run"}
        os.environ[ENV_STATE] = json.dumps(st)
        # Once per process, not once per candidate: this is reached from every
        # benchmark, and a warning repeated a thousand times is a warning nobody
        # reads. It is still stamped into every artifact.
        global _WARNED_OPT_OUT
        if not _WARNED_OPT_OUT:
            _WARNED_OPT_OUT = True
            print(f"[clock] WARNING: {ENV_POLICY}=0 -- running at unlocked boost "
                  f"clock ({st['observed'].get('sm_mhz')} MHz now). Timings from "
                  f"this {what} are NOT comparable with locked runs.", flush=True)
        return st

    # 1. Already locked by this process tree: verify, never re-lock. A child that
    #    re-locked would also unlock on exit and un-pin its parent's run.
    if owner_pid() is not None:
        st = state()
        target = st.get("target_gpu_mhz")
        if target:
            ok, obs = verify(int(target), idx)
            st = dict(st, observed=obs, locked=bool(ok))
            if not ok:
                raise ClockLockError(
                    f"the GPU clock was pinned to {target} MHz by pid "
                    f"{owner_pid()} but now reads {obs.get('sm_mhz')} MHz "
                    f"(throttle={obs.get('throttle') or 'none'}). Something "
                    f"released or overrode the lock mid-run; timings on either "
                    f"side of that are not comparable.")
            return st

    # 1b. A pin left by a run that died without releasing it. Checked BEFORE the
    #     external-lock branch, because that branch cannot tell the two apart:
    #     both look like "pinned, not by me". Adopting a dead run's frequency is
    #     the failure this exists to prevent -- it silently measures at a clock
    #     the repo's T_SOL constant does not assume.
    reclaim_stale(idx, verbose=verbose)

    # 2. Already locked by something outside this run -- an operator, a systemd
    #    unit, a cluster prolog. The requirement is that the clock be pinned, not
    #    that this process be the one that pinned it, so adopt it and leave it
    #    alone: whoever set it owns releasing it.
    ext = detect_external_lock(idx)
    if ext.get("locked"):
        target, _dram, why = resolve_target(idx)
        st = {"locked": True, "gpu_name": device_name(idx), "smi_index": idx,
              "target_gpu_mhz": ext.get("mhz"), "target_dram_mhz": None,
              "max_gpu_mhz": (ext.get("observed") or {}).get("max_sm_mhz"),
              "chosen_by": f"pre-existing external lock ({ext['evidence']})",
              "owned_by_this_run": False, "observed": ext.get("observed"),
              "dram_note": "", "error": ""}
        os.environ[ENV_STATE] = json.dumps(st)
        if verbose:
            print(describe(st), flush=True)
            if target and ext.get("mhz") and abs(ext["mhz"] - target) > TOLERANCE_MHZ:
                print(f"[clock] note: the external lock sits at {ext['mhz']} MHz, "
                      f"not this repo's {target} MHz target ({why}). The run is "
                      f"reproducible at {ext['mhz']} MHz, but its absolute times "
                      f"are not comparable with runs taken at {target} MHz.",
                      flush=True)
        return st

    # 3. Not locked: lock it.
    if not can_lock(idx):
        mode = privilege_mode()
        raise ClockLockError(_setup_help(
            f"the clock is not pinned and this process cannot pin it -- the "
            f"privileged path ({mode}) is not usable without a password"))

    # An unfamiliar card gets measured, not guessed at. This costs ~45 s once per
    # card per machine and the answer is cached; the alternative is pinning a
    # frequency derived from some other card's behaviour.
    target, _dram, why = resolve_target(idx)
    if why.startswith("UNCALIBRATED") and _autocal():
        print(f"[clock] {device_name(idx)!r} has no measured lock target on this "
              f"machine. Measuring what it holds under load (~45 s, once) ...",
              flush=True)
        try:
            if calibrate(device_idx, verbose=verbose):
                target, _dram, why = resolve_target(idx)
        except Exception as exc:  # calibration is a convenience, not a gate
            print(f"[clock] calibration failed ({exc.__class__.__name__}: {exc}); "
                  f"falling back to {target} MHz", flush=True)

    st = lock(idx)
    if not st.get("locked"):
        raise ClockLockError(_setup_help(st.get("error") or "unknown failure"))
    st["owned_by_this_run"] = True

    os.environ[ENV_OWNER] = str(os.getpid())
    os.environ[ENV_STATE] = json.dumps(st)
    _release_on_exit(idx)
    if verbose:
        print(describe(st), flush=True)
        if st.get("dram_note"):
            print(f"[clock] memory clock: {st['dram_note']}", flush=True)
    return st


def assert_still_locked(device_idx: int = 0, what: str = "benchmark",
                        fatal: bool = False) -> bool:
    """Re-check the pin right before a measurement.

    Cheap (one nvidia-smi query) next to a benchmark, and it catches the case the
    run-start check cannot: another user, another tool, or a driver reset moving
    the clock halfway through a long run. Warns by default rather than raising,
    since aborting a nearly-finished run does not un-corrupt the reps already
    taken -- but the warning lands in the log next to the affected numbers.
    """
    st = state()
    if not st.get("locked"):
        return not required()
    target = st.get("target_gpu_mhz")
    if not target:
        return True
    ok, obs = verify(int(target), smi_index(device_idx))
    if not ok:
        msg = (f"[clock] WARNING: clock drifted off the {target} MHz lock during "
               f"{what}: now {obs.get('sm_mhz')} MHz "
               f"(throttle={obs.get('throttle') or 'none'}, "
               f"temp={obs.get('temp_c')} C). These timings are not comparable "
               f"with the rest of the run.")
        print(msg, flush=True)
        if fatal:
            raise ClockLockError(msg)
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect or set the GPU core clock lock.")
    ap.add_argument("--device", type=int, default=0, help="torch device index")
    ap.add_argument("--status", action="store_true", help="report clock and lock target")
    ap.add_argument("--lock", action="store_true", help="pin the clock and exit (stays pinned)")
    ap.add_argument("--unlock", action="store_true", help="release the pin")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure what this card holds under load and cache a "
                         "lock target for it")
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="calibration load duration")
    ap.add_argument("--mhz", type=int, default=None, help="override the target core clock")
    args = ap.parse_args()

    idx = smi_index(args.device)
    if args.calibrate:
        return 0 if calibrate(args.device, seconds=args.seconds) else 1

    if args.unlock:
        unlock(idx)
        print(f"[clock] released; now {current_clocks(idx).get('sm_mhz')} MHz")
        return 0

    if args.lock:
        st = lock(idx, gpu_mhz=args.mhz)
        # `--lock` means "pin it and walk away", so this process is not the owner
        # in the sense the claim records -- it is about to exit, and a claim whose
        # pid is dead is exactly what reclaim_stale releases. Dropping it makes the
        # pin a deliberate external lock, which is what the operator asked for.
        _clear_owner(idx)
        print(describe(st))
        if st.get("dram_note"):
            print(f"[clock] memory clock: {st['dram_note']}")
        return 0 if st.get("locked") else 1

    name = device_name(idx)
    gpu, _dram, why = resolve_target(idx)
    obs = current_clocks(idx)
    mem_clocks, gr_clocks = supported_clocks(idx)
    ext = detect_external_lock(idx)
    priv = "usable" if can_lock(idx) else "NOT usable -- run sudo bash scripts/install_clock_lock_sudoers.sh"
    owner = owner_pid()

    print(f"GPU {idx}: {name}")
    print(f"  now:        {obs.get('sm_mhz')} MHz core / {obs.get('mem_mhz')} MHz mem"
          f"  temp={obs.get('temp_c')} C  util={obs.get('util_pct')}%"
          f"  throttle={obs.get('throttle') or 'none'}")
    print(f"  ceiling:    {max(gr_clocks) if gr_clocks else '?'} MHz core, "
          f"{len(mem_clocks)} supported memory clock(s)")
    print(f"  target:     {gpu} MHz  ({why})")
    print(f"  pinned now: {'yes -- ' + ext['evidence'] if ext['locked'] else 'no'}")
    print(f"  privilege:  {privilege_mode()} route, {priv}")
    print(f"  policy:     {'required' if required() else 'opted out via ' + ENV_POLICY}")
    print(f"  this tree:  {'locked by pid ' + str(owner) if owner else 'not locked'}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
