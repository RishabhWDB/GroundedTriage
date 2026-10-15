"""Bundle the processed splits for fine-tuning on the GPU PC (P2.1).

    python -m data.export        # == make export-splits

Creates datasets/export/groundedtriage-splits-<test content hash[:12]>.tar.gz
containing processed/{train,val,test}.parquet, processed/manifest.json,
config/taxonomy.yaml (categories + urgency scale), datasets/README.md (card)
and docs/model_artifacts.md. Files are verified against the manifest first,
and the archive is deterministic (sorted members, fixed mtime/owner).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from app.settings import REPO_ROOT

PROCESSED = REPO_ROOT / "datasets" / "processed"
EXPORT_DIR = REPO_ROOT / "datasets" / "export"
EXTRA_FILES = ["config/taxonomy.yaml", "datasets/README.md", "docs/model_artifacts.md"]


class ExportError(RuntimeError):
    pass


def verify(processed: Path) -> dict:
    manifest_path = processed / "manifest.json"
    if not manifest_path.exists():
        raise ExportError("datasets/processed/manifest.json missing: run `make data` first")
    manifest = json.loads(manifest_path.read_text())
    for name, meta in manifest["splits"].items():
        path = processed / f"{name}.parquet"
        if not path.exists():
            raise ExportError(f"{path.name} missing: run `make data`")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["file_sha256"]:
            raise ExportError(f"{path.name} does not match manifest.json: re-run `make data`")
    return manifest


def _add(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def export(root: Path = REPO_ROOT, processed: Path = PROCESSED, out_dir: Path = EXPORT_DIR) -> Path:
    manifest = verify(processed)
    tag = manifest["splits"]["test"]["content_sha256"][:12]
    files = {f"processed/{p.name}": p for p in sorted(processed.glob("*.parquet"))}
    files["processed/manifest.json"] = processed / "manifest.json"
    for rel in EXTRA_FILES:
        path = root / rel
        if not path.exists():
            raise ExportError(f"{rel} missing")
        files[rel] = path
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"groundedtriage-splits-{tag}.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for arcname in sorted(files):
            _add(tar, arcname, files[arcname].read_bytes())
    with open(out, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue())
    return out


def main() -> int:
    try:
        out = export()
    except ExportError as exc:
        print(f"export failed: {exc}")
        return 1
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out.relative_to(REPO_ROOT)}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
