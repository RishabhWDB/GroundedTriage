import hashlib
import json

import numpy as np

from app.taxonomy import topic_categories
from tests.test_fast_tier import toy_frame
from triage.emb_fast_tier import EmbFastTier
from triage.embedder import DIM, EmbeddingCache
from triage.fast_tier import CARD_FILE, MODEL_FILE


class FakeEmbedder:
    """Bag of hashed words -> normalised 384-d vector; counts encoded texts."""

    model_name, max_length = "fake/emb", 32

    def __init__(self):
        self.encoded = 0
        self.load_s, self.device = 0.0, "cpu"

    def encode(self, texts):
        self.encoded += len(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_embedding_cache_reuses_vectors(tmp_path):
    emb = FakeEmbedder()
    cache = EmbeddingCache(emb, tmp_path)
    a = cache.encode(["hello world", "refund please", "hello world"])
    assert a.shape == (3, DIM) and emb.encoded == 2 and np.allclose(a[0], a[2])
    again = EmbeddingCache(emb, tmp_path)  # reloaded from disk
    b = again.encode(["refund please"])
    assert emb.encoded == 2 and np.allclose(b[0], a[1])
    assert cache.encode([]).shape == (0, DIM)


def test_train_and_load(tmp_path, monkeypatch):
    from triage import train_emb_fast_tier as t

    d = tmp_path / "processed"
    d.mkdir()
    toy_frame(14, seed=1).to_parquet(d / "train.parquet")
    toy_frame(5, seed=2).to_parquet(d / "val.parquet")
    split = {"content_sha256": "cd" * 32}
    (d / "manifest.json").write_text(json.dumps({"splits": {"train": split}}))
    monkeypatch.setattr(t, "RESULTS", tmp_path / "results")
    monkeypatch.setattr("triage.embedder.CACHE_DIR", tmp_path / "cache")
    emb = FakeEmbedder()
    m = t.train(False, d, tmp_path / "model", tmp_path / "out", embedder=emb, grid=(1.0, 4.0),
                latency_n=5)  # fmt: skip
    assert m["val"]["overall"]["macro_f1"] > 0.7
    assert m["card"]["embedding_model"] == "fake/emb" and m["tfidf_reference"] is None
    assert (tmp_path / "out" / "report.md").exists()
    tier = EmbFastTier.load(tmp_path / "model", emb)
    p = tier.predict(["refund invoice payment card"])[0]
    assert p.category == "Billing" and set(p.probs) == set(topic_categories())
    assert tier.predict([]) == []


def test_load_rejects_bad_artifacts(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression

    assert EmbFastTier.load(tmp_path / "none") is None
    lr = LogisticRegression().fit(np.eye(2), ["Billing", "Spam"])
    joblib.dump(lr, tmp_path / MODEL_FILE)
    (tmp_path / CARD_FILE).write_text(json.dumps({"embedding_model": "x", "max_length": 8}))
    assert EmbFastTier.load(tmp_path) is None
    (tmp_path / CARD_FILE).write_text("{}")
    assert EmbFastTier.load(tmp_path) is None
