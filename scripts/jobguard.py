#!/usr/bin/env python3
"""Job guard for model-loading and LLM-batch make targets (8 GB M1, and the
WSL2 eval PC; added after the 2026-10-17 out-of-memory reboot). Stdlib only.

    jobguard.py --name rag-eval --cwd ml-service -- .venv/bin/python -m eval.rag_eval
    jobguard.py --name dev --check-only -- honcho start     # servers: checks, no lock

Before starting:
- memory: refuses when memory pressure is warn/critical or free memory is below
  JOBGUARD_MIN_FREE_PCT (default 20%).
  macOS: kern.memorystatus_vm_pressure_level >= 2; free from `memory_pressure -Q`.
  Linux/WSL2: PSI /proc/pressure/memory when the kernel has it (some avg10 >= 10%
  is warn, full avg10 >= 10% is critical; no pressure check without PSI); free is
  MemAvailable / MemTotal from /proc/meminfo (under WSL2 that is the VM's memory,
  capped by .wslconfig). GPU memory (nvidia-smi) is recorded, not checked;
- services: refuses while the ml-service (uvicorn app.main) or honcho (make dev)
  runs, unless --allow-services; stop them when the step doesn't need them;
- one job at a time: an exclusive flock on datasets/raw/.jobguard.lock; a second
  guarded job is refused and told who holds it. --check-only only tests it.
JOBGUARD_FORCE=1 skips the memory and service checks (never the lock).

While running: the command is its own process group; Ctrl-C/TERM are forwarded.
After it exits: leftover processes in the group (tokenizer/multiprocessing
workers) get SIGTERM then SIGKILL, and peak RSS + swap (+ GPU memory when
nvidia-smi exists) are printed and appended to datasets/raw/jobguard.jsonl.
Exit status is the command's.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "datasets" / "raw" / ".jobguard.lock"
LOG = ROOT / "datasets" / "raw" / "jobguard.jsonl"
SERVICE_PATTERNS = {
    "ml-service": "uvicorn app.main",
    "make dev (honcho)": "honcho start",
}
PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}
MEMINFO = Path("/proc/meminfo")
PSI_MEMORY = Path("/proc/pressure/memory")
# PSI avg10 (% of the last 10 s that tasks stalled on memory) mapped onto the macOS levels.
PSI_WARN_SOME, PSI_CRITICAL_FULL = 10.0, 10.0


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def macos_memory_state() -> dict:
    level = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()
    free = re.search(r"free percentage:\s*(\d+)%", _run(["memory_pressure", "-Q"]))
    swap = re.search(r"used = ([\d.]+)M", _run(["sysctl", "-n", "vm.swapusage"]))
    return {
        "pressure_level": int(level) if level.isdigit() else None,
        "free_pct": int(free.group(1)) if free else None,
        "swap_used_mb": float(swap.group(1)) if swap else None,
    }


def parse_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo as {field: kB}."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def parse_psi(text: str) -> dict[str, float]:
    """/proc/pressure/memory as {"some": avg10, "full": avg10}."""
    return {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"^(some|full)\s+avg10=([\d.]+)", text, re.MULTILINE)
    }


def psi_level(psi: dict[str, float]) -> int | None:
    if not psi:
        return None
    if psi.get("full", 0.0) >= PSI_CRITICAL_FULL:
        return 4
    return 2 if psi.get("some", 0.0) >= PSI_WARN_SOME else 1


def linux_memory_state(meminfo: str | None = None, psi: str | None = None) -> dict:
    mem = parse_meminfo(_read(MEMINFO) if meminfo is None else meminfo)
    stalls = parse_psi(_read(PSI_MEMORY) if psi is None else psi)
    total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
    swap = None
    if "SwapTotal" in mem and "SwapFree" in mem:
        swap = round((mem["SwapTotal"] - mem["SwapFree"]) / 1024, 2)
    return {
        "pressure_level": psi_level(stalls),
        "free_pct": int(100 * avail / total) if total and avail is not None else None,
        "swap_used_mb": swap,
        "psi_avg10": stalls or None,
    }


def gpu_memory_used_mb() -> float | None:
    """Memory used on the first NVIDIA GPU (None without nvidia-smi)."""
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    first = out.strip().splitlines()[:1]
    try:
        return float(first[0]) if first else None
    except ValueError:
        return None


def memory_state() -> dict:
    m = linux_memory_state() if sys.platform.startswith("linux") else macos_memory_state()
    gpu = gpu_memory_used_mb()
    if gpu is not None:
        m["gpu_used_mb"] = gpu
    return m


def peak_children_rss_mb() -> float:
    """Peak RSS of the largest finished child (ru_maxrss: bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return rss / (2**20 if sys.platform == "darwin" else 1024)


def memory_refusal(m: dict, min_free: int) -> str | None:
    if m["pressure_level"] is not None and m["pressure_level"] >= 2:
        name = PRESSURE_NAMES.get(m["pressure_level"], m["pressure_level"])
        return f"memory pressure is {name} (level {m['pressure_level']})"
    if m["free_pct"] is not None and m["free_pct"] < min_free:
        return f"only {m['free_pct']}% memory free (< {min_free}%)"
    return None


def running_services() -> dict[str, list[int]]:
    found = {}
    for name, pat in SERVICE_PATTERNS.items():
        pids = [int(p) for p in _run(["pgrep", "-f", pat]).split() if int(p) != os.getpid()]
        if pids:
            found[name] = pids
    return found


def acquire(name: str, check_only: bool):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.pread(fd, 4096, 0).decode(errors="replace").strip() or "unknown"
        os.close(fd)
        return None, holder
    if check_only:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return -1, None
    os.ftruncate(fd, 0)
    info = {"name": name, "pid": os.getpid(), "since": datetime.now(UTC).isoformat()}
    os.pwrite(fd, json.dumps(info).encode(), 0)
    return fd, None


def kill_group(pgid: int) -> int:
    """SIGTERM then SIGKILL whatever is left in the job's process group."""
    for sig, wait in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except (
            ProcessLookupError,
            PermissionError,
        ):  # EPERM: only zombies left (macOS)
            return 0 if sig == signal.SIGTERM else 1
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return 1
            time.sleep(0.1)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one guarded model/LLM job.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--check-only", action="store_true", help="servers: check, don't hold the lock")
    ap.add_argument("--allow-services", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given")
    tag = f"jobguard[{args.name}]"
    force = os.environ.get("JOBGUARD_FORCE") == "1"

    fd, holder = acquire(args.name, args.check_only)
    if fd is None:
        print(
            f"{tag}: refused: another job is running ({holder}). One job at a time.",
            file=sys.stderr,
        )
        return 75
    before = memory_state()
    if not force:
        why = memory_refusal(before, int(os.environ.get("JOBGUARD_MIN_FREE_PCT", "20")))
        if why:
            print(f"{tag}: refused: {why}; swap used {before['swap_used_mb']} MB. Close apps "
                  "or wait, then retry (JOBGUARD_FORCE=1 overrides).", file=sys.stderr)  # fmt: skip
            return 75
        svc = {} if args.allow_services else running_services()
        if svc:
            print(f"{tag}: refused: {', '.join(f'{k} pids {v}' for k, v in svc.items())} running; "
                  "stop it first (this step doesn't need it).", file=sys.stderr)  # fmt: skip
            return 75
    print(f"{tag}: start; memory {before}", file=sys.stderr)

    if args.check_only:
        os.chdir(args.cwd or ".")
        os.execvp(cmd[0], cmd)

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=args.cwd, start_new_session=True)

    def forward(sig, _frame):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    rc = proc.wait()
    leftovers = kill_group(proc.pid)
    peak_mb = peak_children_rss_mb()
    after = memory_state()
    rec = {
        "at": datetime.now(UTC).isoformat(),
        "name": args.name,
        "cmd": cmd,
        "exit": rc,
        "seconds": round(time.monotonic() - t0, 1),
        "peak_rss_mb": round(peak_mb, 1),
        "killed_leftovers": bool(leftovers),
        "memory_before": before,
        "memory_after": after,
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{tag}: exit {rc} after {rec['seconds']} s; peak RSS {rec['peak_rss_mb']} MB; "
          f"swap used {after['swap_used_mb']} MB"
          + (f"; GPU used {after['gpu_used_mb']} MB" if "gpu_used_mb" in after else "")
          + ("; killed leftover processes" if leftovers else ""), file=sys.stderr)  # fmt: skip
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
