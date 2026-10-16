import json

import numpy as np
import pytest

from eval.classification import urgency_metrics
from triage.urgency import (
    SCALE,
    UrgencyModel,
    UrgencyPredictor,
    heuristic_urgency,
)

# ---- heuristic ----


@pytest.mark.parametrize(
    "text,level,signals",
    [
        ("How do I change my avatar?", "medium", []),
        ("Just curious, no rush: is there a dark mode?", "low", ["low"]),
        ("URGENT: login broken, please fix ASAP", "high", ["urgency"]),
        (
            "I need the invoice today, this is the third time I ask",
            "medium",
            ["deadline", "escalation"],
        ),
        (
            "Need the invoice by Friday. Third time asking, I am frustrated!!",
            "high",
            ["deadline", "escalation", "negative"],
        ),
        ("Production is down for all users, urgent", "critical", ["urgency", "impact"]),
        ("Outage again. Escalate this to your manager.", "critical", ["escalation", "impact"]),
        ("Our website is down", "medium", ["impact"]),  # impact alone: +2 -> medium, not critical
        (
            "Emergency: server is down, not urgent for you maybe but whenever possible",
            "critical",
            ["urgency", "impact", "low"],
        ),
    ],
)
def test_heuristic_levels(text, level, signals):
    r = heuristic_urgency(text)
    assert r.urgency == level and r.source == "heuristic"
    assert sorted(r.signals) == sorted(signals)
    assert 0.4 <= r.confidence <= 0.8


def test_heuristic_group_counts_once():
    a = heuristic_urgency("urgent")
    b = heuristic_urgency("urgent urgent ASAP immediately emergency")
    assert a.urgency == b.urgency == "high"  # urgency group is +3 once
    c = heuristic_urgency("urgent, but no rush")  # 3 - 2 = 1
    assert c.urgency == "medium"
    assert a.signals == b.signals == ["urgency"]


def test_heuristic_word_boundaries():
    # "today" inside "todays" / "critical" inside "uncritical" should not fire.
    assert heuristic_urgency("uncritical todays note").signals == []
    assert heuristic_urgency("").urgency == "medium"


# ---- ordinal evaluation ----


def test_urgency_metrics_toy_arrays():
    m = urgency_metrics(["low", "medium", "high", "high"], ["low", "high", "low", "high"])
    assert m["accuracy"] == 0.5
    assert m["ordinal_mae"] == pytest.approx((0 + 1 + 2 + 0) / 4)
    assert m["off_by_two_or_more"] == 0.25
    assert m["per_class"]["low"]["precision"] == 0.5


def test_urgency_metrics_critical_on_full_scale():
    m = urgency_metrics(["high", "low"], ["critical", "low"])
    assert m["ordinal_mae"] == 0.5  # high -> critical is one step
    assert m["accuracy"] == 0.5
    with pytest.raises(ValueError):
        urgency_metrics(["high"], ["urgent"])


# ---- model loader (same three cases as the deep tier) ----

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tests.hf_fixtures import save_tiny_classifier  # noqa: E402


def test_urgency_model_missing_dir_falls_back(tmp_path):
    m = UrgencyModel(tmp_path / "urgency", device="cpu")
    assert not m.available and "does not exist" in m.reason
    p = UrgencyPredictor(m)
    assert not p.model_available
    r = p.predict(["URGENT please help ASAP today"])[0]
    assert r.source == "heuristic" and r.urgency == "high"


@pytest.mark.parametrize(
    "labels",
    [["high", "medium", "low"], ["low", "medium"], ["low", "medium", "high", "urgent"]],
)
def test_urgency_model_invalid_label_map(tmp_path, labels):
    d = save_tiny_classifier(tmp_path / "urgency", labels, "urgency")
    m = UrgencyModel(d, device="cpu")
    assert not m.available and "label_map.json" in m.reason


def test_urgency_model_happy_path_and_critical_override(tmp_path):
    d = save_tiny_classifier(tmp_path / "urgency", ["low", "medium", "high"], "urgency")
    m = UrgencyModel(d, device="cpu")
    assert m.available and not m.loaded
    p = UrgencyPredictor(m)
    texts = ["help question", "production is down for all users, urgent"]
    out = p.predict(texts)
    assert m.loaded
    assert out[0].source == "model" and out[0].urgency in ("low", "medium", "high")
    assert out[1].source == "heuristic" and out[1].urgency == "critical"
    probs = m.predict_proba(texts)
    assert probs.shape == (2, 3) and np.allclose(probs.sum(axis=1), 1, atol=1e-5)


def test_urgency_model_with_critical_label(tmp_path):
    d = save_tiny_classifier(tmp_path / "urgency", SCALE, "urgency")
    assert UrgencyModel(d, device="cpu").available


def test_urgency_model_load_failure_falls_back(tmp_path):
    d = save_tiny_classifier(tmp_path / "urgency", ["low", "medium", "high"], "urgency")
    (d / "model.safetensors").write_bytes(b"garbage")
    m = UrgencyModel(d, device="cpu")
    assert m.available  # validation is file-level; the failure shows at load
    r = UrgencyPredictor(m).predict(["no rush"])[0]
    assert r.source == "heuristic" and r.urgency == "low"
    assert not m.available and "failed to load" in m.reason
    json.loads((d / "model_card.json").read_text())  # artifact untouched


def test_eval_urgency_on_toy_val(tmp_path):
    import pandas as pd

    from triage.eval_urgency import evaluate

    d = tmp_path / "processed"
    d.mkdir()
    pd.DataFrame(
        {
            "text": ["URGENT asap", "no rush at all", "hello", "spam"],
            "urgency": ["high", "low", "medium", None],
        }
    ).to_parquet(d / "val.parquet")
    m = evaluate(tmp_path / "out", d, model=UrgencyModel(tmp_path / "none", device="cpu"))
    assert m["rows"] == 3 and m["heuristic"]["accuracy"] == 1.0
    assert m["model"]["status"].startswith("unavailable")
    assert (tmp_path / "out" / "report.md").exists()
