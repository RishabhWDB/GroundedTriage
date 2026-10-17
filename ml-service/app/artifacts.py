"""Model artifacts from a path or a URL (P8.2).

    python -m app.artifacts pack fast_tier spam_filter [--out models.tar.gz]
    python -m app.artifacts fetch            # == make models-fetch

The ml-service always loads from MODELS_DIR (default ./models). On a host
without the weights (Render), set MODELS_URL to a .tar.gz made by `pack`
(e.g. a GitHub release asset or any HTTPS URL) and MODELS_SHA256 to the
sha256 that `pack` printed; `fetch` downloads it, checks the hash, and
extracts it into MODELS_DIR. It is idempotent: a marker file records the
installed hash, so a restart with the same bundle downloads nothing.

The archive is untrusted input until its hash matches, and extraction is
strict even after: regular files and directories only, no absolute paths,
no "..", no links, and every member must land inside MODELS_DIR.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

from app.settings import get_settings

logger = logging.getLogger(__name__)

MARKER = ".bundle_sha256"
MAX_BYTES = 2 << 30  # 2 GiB: a fast tier + spam filter bundle is a few MB; a deep tier ~0.7 GB


class ArtifactError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack(models_dir: Path, names: list[str], out: Path) -> str:
    """Bundle models_dir/<name> for each name into a deterministic .tar.gz."""
    for n in names:
        if not (models_dir / n).is_dir():
            raise ArtifactError(f"{models_dir / n} is not a directory")

    def clean(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = 0
        return ti

    # fixed gzip header (no name, mtime 0) so the same files give the same sha256
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for n in sorted(names):
                for p in sorted((models_dir / n).rglob("*")):
                    if p.is_symlink():
                        raise ArtifactError(f"refusing to pack a symlink: {p}")
                    tar.add(
                        p, arcname=str(p.relative_to(models_dir)), recursive=False, filter=clean
                    )
    return sha256_file(out)


def safe_extract(archive: Path, dest: Path) -> list[str]:
    dest = dest.resolve()
    names = []
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if (
                m.name.startswith("/")
                or ".." in Path(m.name).parts
                or not target.is_relative_to(dest)
            ):
                raise ArtifactError(f"unsafe path in bundle: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ArtifactError(f"unsupported member type in bundle: {m.name!r}")
        tar.extractall(dest, members=members, filter="data")
        names = sorted({Path(m.name).parts[0] for m in members})
    return names


def download(url: str, to: Path, client: httpx.Client | None = None) -> None:
    if not url.startswith("https://"):
        raise ArtifactError("MODELS_URL must be https://")
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True)
    try:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            size = 0
            with open(to, "wb") as f:
                for chunk in r.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ArtifactError(f"bundle larger than {MAX_BYTES} bytes")
                    f.write(chunk)
    finally:
        if own:
            client.close()


def fetch(url: str, sha256: str, models_dir: Path, client: httpx.Client | None = None) -> str:
    """Install the bundle at url into models_dir unless it is already there.
    Returns "present" or "installed"."""
    if not sha256 or len(sha256) != 64:
        raise ArtifactError("MODELS_SHA256 (64 hex chars) is required with MODELS_URL")
    marker = models_dir / MARKER
    if marker.exists() and marker.read_text().strip() == sha256:
        return "present"
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        download(url, archive, client)
        got = sha256_file(archive)
        if got != sha256:
            raise ArtifactError(f"sha256 mismatch: got {got}, want {sha256}; nothing installed")
        staging = Path(tmp) / "x"
        staging.mkdir()
        names = safe_extract(archive, staging)
        for n in names:  # replace whole artifact dirs, never merge old and new files
            target = models_dir / n
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(staging / n), target)
    marker.write_text(sha256 + "\n")
    logger.info("installed model bundle %s… into %s: %s", sha256[:12], models_dir, ", ".join(names))
    return "installed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="bundle artifact dirs from MODELS_DIR")
    p.add_argument("names", nargs="+")
    p.add_argument("--out", default="models-bundle.tar.gz")
    sub.add_parser(
        "fetch", help="download MODELS_URL into MODELS_DIR (no-op when unset or present)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = get_settings()
    try:
        if args.cmd == "pack":
            digest = pack(s.models_dir, args.names, Path(args.out))
            print(f"{args.out}\nMODELS_SHA256={digest}")
            return 0
        if not s.models_url:
            print(f"MODELS_URL unset; using {s.models_dir} as is")
            return 0
        print(fetch(s.models_url, s.models_sha256, s.models_dir))
        return 0
    except (ArtifactError, httpx.HTTPError, tarfile.TarError) as exc:
        print(f"artifacts: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
