import json

import pytest

from app.taxonomy import categories, topic_categories
from triage.deep_tier import DeepTier
from triage.hf_model import ArtifactError, read_label_map

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.hf_fixtures import save_tiny_classifier  # noqa: E402


def test_missing_directory(tmp_path, caplog):
    d = DeepTier(tmp_path / "deep_tier", device="cpu")
    assert not d.available and not d.loaded
    assert "does not exist" in d.reason
    assert "category model unavailable" in caplog.text
    with pytest.raises(RuntimeError):
        d.predict(["hello"])


def test_missing_files(tmp_path):
    (tmp_path / "deep_tier").mkdir()
    (tmp_path / "deep_tier" / "config.json").write_text("{}")
    d = DeepTier(tmp_path / "deep_tier", device="cpu")
    assert not d.available
    assert "model.safetensors" in d.reason and "label_map.json" in d.reason


@pytest.mark.parametrize(
    "label_map,problem",
    [
        ({"0": "Technical Issue", "1": "Billing"}, "exactly the 5 topic categories"),
        ({str(i): c for i, c in enumerate(categories())}, "exactly the 5 topic categories"),
        ({str(i + 1): c for i, c in enumerate(topic_categories())}, "ids must be 0..4"),
        (
            {"0": "Billing", **{str(i + 1): c for i, c in enumerate(topic_categories()[:4])}},
            "duplicate",
        ),
        ({"a": "Billing"}, "must be integers"),
    ],
)
def test_invalid_label_map(tmp_path, label_map, problem):
    d = save_tiny_classifier(tmp_path / "deep_tier", topic_categories(), "category")
    (d / "label_map.json").write_text(json.dumps(label_map))
    dt = DeepTier(d, device="cpu")
    assert not dt.available and problem in dt.reason


@pytest.mark.parametrize(
    "override,problem",
    [
        ({"task": "urgency"}, "expected 'category'"),
        ({"max_length": None}, "max_length"),
        ({"val_metrics": {"accuracy": 0.9}}, "macro_f1"),
    ],
)
def test_invalid_model_card(tmp_path, override, problem):
    d = save_tiny_classifier(
        tmp_path / "deep_tier", topic_categories(), "category", card_overrides=override
    )
    dt = DeepTier(d, device="cpu")
    assert not dt.available and problem in dt.reason


def test_label_count_mismatch_with_config(tmp_path):
    d = save_tiny_classifier(tmp_path / "deep_tier", topic_categories()[:4] + ["Other"], "category")
    (d / "label_map.json").write_text(json.dumps(dict(enumerate(topic_categories()))))
    cfg = json.loads((d / "config.json").read_text())
    cfg["num_labels"] = 4
    cfg["id2label"] = {str(i): str(i) for i in range(4)}
    (d / "config.json").write_text(json.dumps(cfg))
    dt = DeepTier(d, device="cpu")
    assert not dt.available and "config.json has 4 labels" in dt.reason


def test_happy_path_lazy_load_batching_truncation(tmp_path):
    # Label ids in a non-taxonomy order: predictions must follow label_map.
    labels = list(reversed(topic_categories()))
    d = save_tiny_classifier(tmp_path / "deep_tier", labels, "category", max_length=8)
    dt = DeepTier(d, device="cpu", batch_size=2)
    assert dt.available and not dt.loaded  # lazy
    assert dt.labels == labels and dt.max_length == 8
    long_text = " ".join(["login error"] * 200)  # far beyond max_length and position embeddings
    texts = ["refund invoice please", "meeting call today", long_text, "win prize", "help"]
    preds = dt.predict(texts)
    assert dt.loaded and dt.device == "cpu"
    assert len(preds) == 5
    for p in preds:
        assert p.category in topic_categories()
        assert set(p.probs) == set(topic_categories())
        assert sum(p.probs.values()) == pytest.approx(1, abs=1e-5)
        assert p.probs[p.category] == pytest.approx(p.confidence)
    # Batching does not change results.
    single = [dt.predict([t])[0] for t in texts]
    for a, b in zip(preds, single, strict=True):
        assert a.category == b.category and a.confidence == pytest.approx(b.confidence, abs=1e-5)
    assert dt.predict([]) == []
    assert dt.version.startswith("deep_tier-")


def test_read_label_map_rejects_empty(tmp_path):
    p = tmp_path / "label_map.json"
    p.write_text("{}")
    with pytest.raises(ArtifactError):
        read_label_map(p)
