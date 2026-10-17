import json

import pandas as pd
import pytest

from data import ood
from eval.ood_eval import by_source, report_markdown, score


def test_clean_tweet():
    body, brand = ood.clean_tweet("@AppleSupport  my phone   won't charge @115712 help")
    assert brand == "AppleSupport" and body == "my phone won't charge help"


def test_parse_labels():
    text = json.dumps({"labels": [{"i": 2, "category": "Billing"}, {"i": 1, "category": "None"}]})
    assert ood.parse_labels(text, 2) == ["None", "Billing"]
    with pytest.raises(ValueError):
        ood.parse_labels(json.dumps({"labels": [{"i": 1, "category": "Complaint"}]}), 1)
    with pytest.raises(ValueError):
        ood.parse_labels(json.dumps({"labels": []}), 1)


def test_pick_by_quota_keeps_candidate_order():
    df = pd.DataFrame(
        {"id": [f"tw-{i}" for i in range(6)],
         "llm_category": ["Billing", "None", "Billing", "Spam", "Billing", "Spam"]}
    )  # fmt: skip
    got = ood.pick_by_quota(df, {"Billing": 2, "Spam": 5})
    assert got["id"].tolist() == ["tw-0", "tw-2", "tw-3", "tw-5"]


def test_load_ood_scores_only_reviewed(tmp_path, monkeypatch):
    tw = tmp_path / "twitter.csv"
    ood.write_csv(
        tw,
        [
            {"id": "tw-1", "source": "twitter", "brand": "x", "subject": "", "body": "refund pls",
             "llm_category": "Billing", "category": "Billing", "needs_review": "false",
             "reviewer_notes": ""},
            {"id": "tw-2", "source": "twitter", "brand": "x", "subject": "", "body": "app broken",
             "llm_category": "Technical Issue", "category": "Technical Issue",
             "needs_review": "true", "reviewer_notes": ""},
        ],
        ood.REVIEW_FIELDS,
    )  # fmt: skip
    team = tmp_path / "team.csv"
    row = {"id": "t1", "author": "a", "subject": "hi", "body": "can we call tue?",
           "category": "Meeting Scheduling", "notes": ""}  # fmt: skip
    ood.write_csv(team, [row], ood.TEAM_FIELDS)
    monkeypatch.setattr(ood, "TWITTER_CSV", tw)
    monkeypatch.setattr(ood, "TEAM_CSV", team)
    monkeypatch.setattr(ood, "ENRON_CSV", tmp_path / "none.csv")
    df, counts = ood.load_ood()
    assert counts["twitter_pending_review"] == 1 and counts["twitter"] == 1
    assert counts["team"] == 1 and counts["sample_emails"] == 22
    assert "tw-2" not in set(df["id"])
    df_all, _ = ood.load_ood(with_pending=True)
    assert "tw-2" in set(df_all["id"])


def test_score_counts_muted_customers():
    df = pd.DataFrame(
        {"category": ["Billing", "Spam", "Technical Issue"], "source": ["a", "a", "b"]}
    )
    s = score(df, ["Spam", "Spam", "Technical Issue"], [False, False, True])
    assert (s["non_spam_muted"], s["non_spam_n"], s["spam_caught"], s["spam_n"]) == (1, 2, 1, 1)
    assert s["non_spam_in_review"] == 1 and s["correct"] == 2
    bs = by_source(df, ["Spam", "Spam", "Technical Issue"], None)
    assert bs["b"]["correct"] == 1 and bs["all"]["n"] == 3
    m = {"rows": 3, "counts": {}, "gold": {}, "systems": {"fast_only": bs}, "versions": {}}
    assert "| fast_only | all | 3 | 2 |" in report_markdown(m)
