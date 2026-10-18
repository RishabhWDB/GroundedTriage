"""Crash-safe file writes (after the 2026-10-17 out-of-memory reboot).

- Whole files (caches, splits, configs, hand-reviewed CSVs) are written to a
  unique temp file in the same directory, fsynced, then `os.replace`d over the
  target, and the directory is fsynced. A crash leaves the old file or the new
  one, never a half-written one; a stray `.<name>.*.tmp` is harmless.
- Append-only JSONL progress files can't be renamed per row, so each line is
  flushed and fsynced as it is written, and `read_jsonl` / `open_append` drop
  a truncated last line (the only damage a crash mid-append can do).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temp path next to `path` (same suffix, so libraries that infer
    the format from it still work); on a clean exit it replaces `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}"
    )
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def _complete_prefix(data: bytes) -> int:
    """Length of the prefix holding only complete, parseable lines. Only the
    last line may be dropped; a bad line in the middle is real corruption."""
    if not data:
        return 0
    cut = (data[:-1] if data.endswith(b"\n") else data).rfind(b"\n") + 1
    try:
        json.loads(data[cut:])
        return len(data)
    except ValueError:
        return cut


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file, ignoring a truncated last line (logged)."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    keep = _complete_prefix(data)
    if keep < len(data):
        logger.warning("%s: ignoring a truncated last line (%d bytes)", path, len(data) - keep)
    rows = []
    for i, line in enumerate(data[:keep].splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{path}: line {i} is corrupt: {e}") from e
    return rows


def repair_jsonl(path: Path) -> int:
    """Truncate a partial last line so appends start on a fresh line. Returns
    the number of bytes removed."""
    path = Path(path)
    if not path.exists():
        return 0
    data = path.read_bytes()
    keep = _complete_prefix(data)
    if keep < len(data):
        with open(path, "rb+") as f:
            f.truncate(keep)
            os.fsync(f.fileno())
        logger.warning("%s: removed a truncated last line (%d bytes)", path, len(data) - keep)
    elif data and not data.endswith(b"\n"):
        with open(path, "ab") as f:
            f.write(b"\n")
            os.fsync(f.fileno())
    return len(data) - keep


@contextmanager
def open_append(path: Path) -> Iterator[IO[str]]:
    """Open a JSONL file for appending after repairing its tail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl(path)
    with open(path, "a", encoding="utf-8") as f:
        yield f


def append_line(f: IO[str], record: dict, **dumps_kw) -> None:
    """Write one JSON line and make it durable before the next LLM call."""
    f.write(json.dumps(record, **dumps_kw) + "\n")
    f.flush()
    os.fsync(f.fileno())
