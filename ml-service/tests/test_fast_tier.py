import json

import pandas as pd
import pytest

from app.taxonomy import topic_categories
from eval.classification import calibration, coverage_curve
from triage.fast_tier import CARD_FILE, MODEL_FILE, FastTier, build_calibrated

WORDS = {
    "Technical Issue": "error crash bug login fails server down timeout",
    "Billing": "invoice charged refund payment card subscription",
    "Feature Request": "please add feature option export dark mode",
    "General Inquiry": "question information hours policy pricing info",
    "Meeting Scheduling": "schedule meeting call reschedule calendar slot",
}


def toy_frame(n_per_class: int = 12, seed: int = 0) -> pd.DataFrame:
    import random

    rng = random.Random(seed)
    rows = []
    for cat, words in WORDS.items():
        vocab = words.split()
        for i in range(n_per_class):
            text = " ".join(rng.choice(vocab) for _ in range(8))
            rows.append(
                {
                    "id": f"{cat[:3]}-{seed}-{i}",
                    "origin": "real",
                    "category": cat,
                    "subject": "",
                    "body": text,
                    "text": text,
                    "urgency": None,
                }
            )
    return pd.DataFrame(rows)


def save_toy_model(directory, classes_ok: bool = True):
    import joblib

    df = toy_frame()
    if not classes_ok:  # an old 6-class model or a missing class
        df = df[df["category"] != "Feature Request"]
    clf = build_calibrated(4.0, folds=3).fit(df["text"], df["category"])
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, directory / MODEL_FILE)
    (directory / CARD_FILE).write_text(json.dumps({"version": "fast-test"}))


def test_load_and_predict(tmp_path):
    save_toy_model(tmp_path / "ft")
    model = FastTier.load(tmp_path / "ft")
    assert model is not None and model.version == "fast-test"
    preds = model.predict(["refund my invoice payment", "the server crash error"])
    assert [p.category for p in preds] == ["Billing", "Technical Issue"]
    for p in preds:
        assert 0 < p.confidence <= 1
        assert set(p.probs) == set(topic_categories())
        assert sum(p.probs.values()) == pytest.approx(1, abs=1e-6)
    assert model.predict([]) == []


def test_load_missing_or_bad(tmp_path, caplog):
    assert FastTier.load(tmp_path / "absent") is None
    d = tmp_path / "corrupt"
    d.mkdir()
    (d / MODEL_FILE).write_bytes(b"not a pickle")
    (d / CARD_FILE).write_text("{}")
    assert FastTier.load(d) is None
    save_toy_model(tmp_path / "five", classes_ok=False)
    assert FastTier.load(tmp_path / "five") is None
    assert "do not match the topic categories" in caplog.text


def test_train_script_on_toy_splits(tmp_path):
    from triage.train_fast_tier import train

    data = tmp_path / "processed"
    data.mkdir()
    toy_frame(14, seed=1).to_parquet(data / "train.parquet")
    val = toy_frame(5, seed=2)
    val.loc[val["category"] == "Meeting Scheduling", "origin"] = "synthetic"
    val.to_parquet(data / "val.parquet")
    split = {"content_sha256": "ab" * 32}
    (data / "manifest.json").write_text(
        json.dumps({"splits": {"train": split, "test": split, "val": split}})
    )
    m = train(tmp_path / "model", tmp_path / "results", data, grid=(1.0, 4.0), folds=3, latency_n=5)
    assert m["val"]["overall"]["macro_f1"] > 0.7  # toy data, tiny calibration folds
    assert {r["C"] for r in m["c_selection"]} == {1.0, 4.0}
    assert {r["method"] for r in m["calibration_selection"]} == {"none", "sigmoid", "isotonic"}
    card = json.loads((tmp_path / "model" / CARD_FILE).read_text())
    assert card["version"].startswith("fast-") and card["val_metrics"]["macro_f1"] > 0.7
    for f in ("metrics.json", "report.md", "confusion_matrix.png", "reliability.png"):
        assert (tmp_path / "results" / f).exists()
    assert "peak_rss_mb" in m["run"]


def test_calibration_and_coverage():
    cal = calibration([0.9, 0.9, 0.6, 0.6], [True, False, True, True])
    # bins: 0.9 -> acc 0.5 (gap 0.4), 0.6 -> acc 1.0 (gap 0.4)
    assert cal["ece"] == pytest.approx(0.4)
    cov = coverage_curve([0.9, 0.5, 0.95], [True, False, True], [0.8, 0.99])
    assert cov[0] == {"tau": 0.8, "accepted_share": 0.6667, "accepted_accuracy": 1.0}
    assert cov[1]["accepted_share"] == 0.0 and cov[1]["accepted_accuracy"] is None
