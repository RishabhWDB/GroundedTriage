import pandas as pd

from eval.sample_check import load_sample, report_markdown, run, top_features
from tests.test_fast_tier import save_toy_model
from triage.fast_tier import FastTier


def test_sample_check_on_toy_model(tmp_path):
    save_toy_model(tmp_path / "ft")
    fast = FastTier.load(tmp_path / "ft")
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "subject": ["", ""],
            "body": ["refund my invoice payment", "server crash error login"],
            "label": ["Billing", "Technical Issue"],
        }
    )
    df["text"] = df["body"]
    m = run(fast, None, df)
    assert m["correct"] == {"fast": 2, "cascade": 2}
    assert m["predicted_spam"]["fast"] == 0 and m["gold_spam"] == 0
    assert "| fast | 2/2 | 0 |" in report_markdown(m)
    assert len(top_features(fast, "Billing", 5)) == 5 and top_features(fast, "Spam") == []


def test_load_sample_has_labels():
    df = load_sample()
    assert len(df) == 22 and df["label"].ne("").all() and df["text"].str.len().gt(0).all()
