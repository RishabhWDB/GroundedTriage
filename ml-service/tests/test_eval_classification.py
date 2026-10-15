import pandas as pd
import pytest

from eval.classification import (
    classification_metrics,
    confusion,
    latency_summary,
    plot_confusion,
    sliced_metrics,
    urgency_metrics,
)


def frame():
    return pd.DataFrame(
        {
            "gold": ["Billing", "Billing", "Spam", "Meeting Scheduling", "Technical Issue"],
            "pred": ["Billing", "Spam", "Spam", "Meeting Scheduling", "Meeting Scheduling"],
            "origin": ["real", "real", "real", "synthetic", "real"],
            "text": ["refund", "invoice", "win", "book a call", "zoom call keeps dropping"],
        }
    )


def test_classification_metrics_macro_over_given_labels():
    m = classification_metrics(["A", "A", "B"], ["A", "B", "B"], ["A", "B"])
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"]["A"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert m["macro_f1"] == pytest.approx((0.6667 + 0.6667) / 2, abs=1e-3)
    assert classification_metrics([], []) == {"n": 0}


def test_sliced_metrics_reporting_rule():
    m = sliced_metrics(frame())
    assert set(m) == {
        "overall",
        "origin_real",
        "origin_synthetic",
        "without_meeting_scheduling",
        "real_meeting_mentions",
    }
    assert m["overall"]["n"] == 5 and len(m["overall"]["labels"]) == 6
    assert m["origin_synthetic"]["accuracy"] == 1.0
    wm = m["without_meeting_scheduling"]
    assert wm["n"] == 4 and "Meeting Scheduling" not in wm["labels"]
    assert wm["accuracy"] == 0.5  # Technical predicted as Meeting counts as an error
    mm = m["real_meeting_mentions"]
    assert mm["n"] == 1 and mm["predicted_meeting"] == 1 and mm["accuracy"] == 0.0


def test_urgency_metrics_ordinal_mae():
    m = urgency_metrics(["low", "high", "medium"], ["high", "high", "medium"])
    assert m["ordinal_mae"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-4)


def test_confusion_latency_and_plot(tmp_path):
    cm = confusion(["A", "B", "B"], ["A", "A", "B"], ["A", "B"])
    assert cm == [[1, 0], [1, 1]]
    lat = latency_summary([10, 20, 30, 40])
    assert lat["p50_ms"] == 25 and lat["n"] == 4 and latency_summary([]) == {"n": 0}
    pytest.importorskip("matplotlib")
    out = tmp_path / "cm.png"
    plot_confusion(cm, ["A", "B"], out, "t")
    assert out.stat().st_size > 1000
