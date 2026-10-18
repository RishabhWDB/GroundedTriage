#!/usr/bin/env python3
"""Job guard for model-loading and LLM-batch make targets (8 GB M1; added after
the 2026-10-17 out-of-memory reboot). Stdlib only.

    jobguard.py --name rag-eval --cwd ml-service -- .venv/bin/python -m eval.rag_eval
    jobguard.py --name dev --check-only -- honcho start     # servers: checks, no lock

Before starting:
- memory: refuses when macOS reports memory pressure warn/critical
  (kern.memorystatus_vm_pressure_level >= 2) or free memory < JOBGUARD_MIN_FREE_PCT
  (default 20%, from `memory_pressure -Q`);
- services: refuses while the ml-service (uvicorn app.main) or honcho (make dev)
  runs, unless --allow-services; stop them when the step doesn't need them;
- one job at a time: an exclusive flock on datasets/raw/.jobguard.lock; a second
  guarded job is refused and told who holds it. --check-only only tests it.
JOBGUARD_FORCE=1 skips the memory and service checks (never the lock).

While running: the command is its own process group; Ctrl-C/TERM are forwarded.
After it exits: leftover processes in the group (tokenizer/multiprocessing
workers) get SIGTERM then SIGKILL, and peak RSS + swap are printed and appended
to datasets/raw/jobguard.jsonl. Exit status is the command's.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import resource
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


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def memory_state() -> dict:
    level = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()
    free = re.search(r"free percentage:\s*(\d+)%", _run(["memory_pressure", "-Q"]))
    swap = re.search(r"used = ([\d.]+)M", _run(["sysctl", "-n", "vm.swapusage"]))
    return {
        "pressure_level": int(level) if level.isdigit() else None,
        "free_pct": int(free.group(1)) if free else None,
        "swap_used_mb": float(swap.group(1)) if swap else None,
    }


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
    peak_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20  # bytes on macOS
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
          + ("; killed leftover processes" if leftovers else ""), file=sys.stderr)  # fmt: skip
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
