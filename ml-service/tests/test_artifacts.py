import io
import tarfile

import httpx
import pytest

from app.artifacts import MARKER, ArtifactError, fetch, pack, safe_extract, sha256_file


def make_models(root):
    (root / "fast_tier").mkdir(parents=True)
    (root / "fast_tier" / "model.joblib").write_bytes(b"weights")
    (root / "fast_tier" / "model_card.json").write_text("{}")
    (root / "spam_filter").mkdir()
    (root / "spam_filter" / "model.joblib").write_bytes(b"spam")


def client_for(data: bytes, calls: list):
    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(200, content=data)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_pack_is_deterministic_and_fetch_installs_once(tmp_path):
    src = tmp_path / "src"
    make_models(src)
    a, b = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    da = pack(src, ["spam_filter", "fast_tier"], a)
    assert da == pack(src, ["fast_tier", "spam_filter"], b)
    dest = tmp_path / "models"
    (dest / "fast_tier").mkdir(parents=True)
    (dest / "fast_tier" / "stale.bin").write_bytes(b"old")  # replaced, not merged
    calls = []
    client = client_for(a.read_bytes(), calls)
    assert fetch("https://x/m.tar.gz", da, dest, client) == "installed"
    assert (dest / "fast_tier" / "model.joblib").read_bytes() == b"weights"
    assert not (dest / "fast_tier" / "stale.bin").exists()
    assert (dest / MARKER).read_text().strip() == da
    assert fetch("https://x/m.tar.gz", da, dest, client) == "present" and len(calls) == 1


def test_fetch_rejects_hash_mismatch_and_http(tmp_path):
    src = tmp_path / "src"
    make_models(src)
    a = tmp_path / "a.tar.gz"
    pack(src, ["fast_tier"], a)
    with pytest.raises(ArtifactError, match="mismatch"):
        fetch("https://x/m", "0" * 64, tmp_path / "m", client_for(a.read_bytes(), []))
    assert not (tmp_path / "m" / "fast_tier").exists()
    with pytest.raises(ArtifactError, match="https"):
        fetch("http://x/m", "0" * 64, tmp_path / "m", client_for(b"", []))
    with pytest.raises(ArtifactError, match="MODELS_SHA256"):
        fetch("https://x/m", "", tmp_path / "m", client_for(b"", []))


def tar_with(tmp_path, add):
    p = tmp_path / "evil.tar.gz"
    with tarfile.open(p, "w:gz") as tar:
        add(tar)
    return p


@pytest.mark.parametrize("name", ["../escape.txt", "/abs.txt", "fast_tier/../../x"])
def test_safe_extract_rejects_traversal(tmp_path, name):
    def add(tar):
        ti = tarfile.TarInfo(name)
        ti.size = 1
        tar.addfile(ti, io.BytesIO(b"x"))

    with pytest.raises(ArtifactError, match="unsafe"):
        safe_extract(tar_with(tmp_path, add), tmp_path / "out")


def test_safe_extract_rejects_links(tmp_path):
    def add(tar):
        ti = tarfile.TarInfo("fast_tier/link")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tar.addfile(ti)

    with pytest.raises(ArtifactError, match="unsupported"):
        safe_extract(tar_with(tmp_path, add), tmp_path / "out")


def test_sha256_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
