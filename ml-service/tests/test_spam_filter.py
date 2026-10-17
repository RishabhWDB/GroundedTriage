import json
import random

import numpy as np
import pandas as pd
import pytest

from triage.fast_tier import CARD_FILE, MODEL_FILE
from triage.spam_filter import HAM, REVIEW, SPAM, SpamFilter, decide
from triage.train_spam_filter import operating_point, threshold_for_fpr, train


def test_threshold_for_fpr():
    neg = [0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
    t = threshold_for_fpr(neg, 0.1)  # 1 false positive allowed of 10
    assert 0.5 < t <= 0.5000001 and np.mean(np.array(neg) >= t) == 0.1
    t0 = threshold_for_fpr(neg, 0.0)
    assert t0 > 0.9 and np.mean(np.array(neg) >= t0) == 0.0
    assert threshold_for_fpr(neg, 1.0) == 0.0


def test_decide_bands():
    assert decide(0.8, 0.5, 0.2) == SPAM
    assert decide(0.5, 0.5, 0.2) == SPAM
    assert decide(0.3, 0.5, 0.2) == REVIEW
    assert decide(0.1, 0.5, 0.2) == HAM


def test_operating_point_by_group():
    df = pd.DataFrame(
        {
            "is_spam": [True, True, False, False, False, False],
            "source_dataset": ["sa", "sa", "tobi", "tobi", "spamassassin_ham", "spamassassin_ham"],
        }
    )
    o = operating_point(df, [0.9, 0.3, 0.1, 0.6, 0.3, 0.0], spam_t=0.5, review_t=0.2)
    assert o["recall_at_mute"] == 0.5 and o["recall_at_review"] == 1.0
    assert o["fpr_support_at_mute"] == 0.5 and o["fpr_ham_at_mute"] == 0.0
    assert o["ham"]["review_band"] == 0.5


_rng = random.Random(1)
SPAM_WORDS = "winner prize click free casino viagra offer cash now".split()
HAM_WORDS = "list thread patch kernel meeting notes lunch photos weekend".split()
SUPPORT_WORDS = "login error invoice refund dashboard export feature api account".split()


def toy_split(n, seed):
    rng = random.Random(seed)
    rows = []
    for kind, words, n_k in (
        ("spam", SPAM_WORDS, n),
        ("ham", HAM_WORDS, n),
        ("tb", SUPPORT_WORDS, 3 * n),
    ):
        for i in range(n_k):
            text = " ".join(rng.choice(words) for _ in range(10))
            rows.append(
                {
                    "id": f"{kind}-{seed}-{i}",
                    "text": text,
                    "is_spam": kind == "spam",
                    "source_dataset": {
                        "spam": "spamassassin",
                        "ham": "spamassassin_ham",
                        "tb": "tobi_bueck",
                    }[kind],
                }
            )
    return pd.DataFrame(rows)


def test_train_load_and_rules(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    for name, seed in (("spam_train", 1), ("spam_val", 2), ("spam_test", 3)):
        toy_split(30, seed).to_parquet(d / f"{name}.parquet")
    (d / "manifest.json").write_text(
        json.dumps({"splits": {"spam_train": {"content_sha256": "ab" * 32}}})
    )
    m = train(tmp_path / "model", tmp_path / "out", d, grid=(1.0,))
    card = json.loads((tmp_path / "model" / CARD_FILE).read_text())
    assert card["default_rule"] == "support" and set(card["rules"]) == {
        "support",
        "support_and_ham",
    }
    assert (
        card["rules"]["support_and_ham"]["spam_threshold"]
        >= card["rules"]["support"]["spam_threshold"]
    )
    assert m["test"]["operating_point"]["recall_at_mute"] == 1.0
    assert m["val"]["operating_point"]["fpr_support_at_mute"] <= 0.01
    assert (tmp_path / "out" / "report.md").exists()

    f = SpamFilter.load(tmp_path / "model")
    assert f is not None and f.rule == "support" and f.version.endswith(":support")
    alt = SpamFilter.load(tmp_path / "model", rule="support_and_ham")
    assert alt.spam_threshold == card["rules"]["support_and_ham"]["spam_threshold"]
    assert SpamFilter.load(tmp_path / "model", rule="nope") is None
    d1, d2 = f.predict(
        ["winner prize click free cash now offer", "login error on the dashboard api"]
    )
    assert d1.decision == SPAM and d2.decision != SPAM and 0 <= d2.score < d1.score <= 1
    assert f.predict([]) == []


def test_load_rejects_bad_artifacts(tmp_path):
    assert SpamFilter.load(tmp_path / "absent") is None
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / MODEL_FILE).write_bytes(b"nope")
    (bad / CARD_FILE).write_text("{}")
    assert SpamFilter.load(bad) is None


@pytest.mark.parametrize("spam_t,review_t", [(0.2, 0.5), (1.5, 0.1)])
def test_load_rejects_bad_thresholds(tmp_path, spam_t, review_t):
    import joblib

    from triage.spam_filter import build_filter

    df = toy_split(10, 0)
    model = build_filter(1.0).fit(df["text"], df["is_spam"])
    joblib.dump(model, tmp_path / MODEL_FILE)
    (tmp_path / CARD_FILE).write_text(
        json.dumps({"spam_threshold": spam_t, "review_threshold": review_t})
    )
    assert SpamFilter.load(tmp_path) is None
