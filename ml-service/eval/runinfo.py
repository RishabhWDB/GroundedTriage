"""Run metadata recorded with every training/eval run: git commit, peak RSS and
swap (the dev machine has 8 GB RAM, so timings are only comparable when memory
pressure is known)."""

from __future__ import annotations

import platform
import resource
import subprocess
import sys
import time

from app.settings import REPO_ROOT


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip()
    except OSError:
        return ""


def swap_usage() -> str:
    """macOS `vm.swapusage` (empty elsewhere)."""
    if sys.platform != "darwin":
        return ""
    try:
        return subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


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
