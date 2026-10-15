import hashlib
import json
import tarfile

import pytest

from data.export import ExportError, export


def setup_tree(tmp_path):
    proc = tmp_path / "datasets" / "processed"
    proc.mkdir(parents=True)
    splits = {}
    for name in ["train", "val", "test"]:
        data = f"{name}-bytes".encode()
        (proc / f"{name}.parquet").write_bytes(data)
        splits[name] = {
            "rows": 1,
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "content_sha256": "ab" * 32,
        }
    (proc / "manifest.json").write_text(json.dumps({"splits": splits}))
    for rel in ["config/taxonomy.yaml", "datasets/README.md", "docs/model_artifacts.md"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel)
    return proc


def test_export_is_complete_and_deterministic(tmp_path):
    proc = setup_tree(tmp_path)
    out = export(tmp_path, proc, tmp_path / "out")
    assert out.name == "groundedtriage-splits-" + "ab" * 6 + ".tar.gz"
    first = out.read_bytes()
    with tarfile.open(out) as tar:
        assert sorted(tar.getnames()) == [
            "config/taxonomy.yaml",
            "datasets/README.md",
            "docs/model_artifacts.md",
            "processed/manifest.json",
            "processed/test.parquet",
            "processed/train.parquet",
            "processed/val.parquet",
        ]
    assert export(tmp_path, proc, tmp_path / "out").read_bytes() == first  # byte-identical


def test_export_refuses_stale_or_missing(tmp_path):
    proc = setup_tree(tmp_path)
    (proc / "test.parquet").write_bytes(b"tampered")
    with pytest.raises(ExportError, match="does not match"):
        export(tmp_path, proc, tmp_path / "out")
    (proc / "manifest.json").unlink()
    with pytest.raises(ExportError, match="make data"):
        export(tmp_path, proc, tmp_path / "out")
