"""Run metadata recorded with every training/eval run: git commit, peak RSS and
swap (the Mac has 8 GB RAM, the eval PC 4 GB VRAM, so timings are only
comparable when memory pressure is known)."""

from __future__ import annotations

import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

from app.settings import REPO_ROOT


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip()
    except OSError:
        return ""


def swap_usage() -> str:
    """macOS `vm.swapusage`; on Linux the same shape from /proc/meminfo (empty elsewhere)."""
    if sys.platform.startswith("linux"):
        return _linux_swap_usage()
    if sys.platform != "darwin":
        return ""
    try:
        return subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def _linux_swap_usage(meminfo: str | None = None) -> str:
    try:
        text = Path("/proc/meminfo").read_text() if meminfo is None else meminfo
    except OSError:
        return ""
    kb = {k: int(v) for k, v in re.findall(r"^(SwapTotal|SwapFree):\s+(\d+)", text, re.M)}
    if len(kb) < 2:
        return ""
    total, free = kb["SwapTotal"] / 1024, kb["SwapFree"] / 1024
    return f"total = {total:.2f}M  used = {total - free:.2f}M  free = {free:.2f}M"


def peak_rss_mb() -> float:
    """Peak resident set size of this process (ru_maxrss: bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def run_info(**extra) -> dict:
    return {
        "git_commit": git_commit(),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "peak_rss_mb": peak_rss_mb(),
        "swap_usage_at_end": swap_usage(),
        "date": time.strftime("%Y-%m-%d"),
        **extra,
    }
