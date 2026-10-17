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


def test_load_ood_marks_reviewed_and_splits(tmp_path, monkeypatch):
    tw = tmp_path / "twitter.csv"
    base = {"source": "twitter", "brand": "x", "subject": "", "reviewer_notes": ""}
    ood.write_csv(
        tw,
        [
            {**base, "id": "tw-1", "body": "refund pls", "llm_category": "Billing",
             "category": "Billing", "needs_review": "false", "ood_split": "ood-dev"},
            {**base, "id": "tw-2", "body": "app broken", "llm_category": "Technical Issue",
             "category": "Technical Issue", "needs_review": "true", "ood_split": "ood-test"},
            {**base, "id": "tw-3", "body": "lol", "llm_category": "None",
             "category": "None", "needs_review": "false", "ood_split": "ood-test"},
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
    assert counts["twitter_pending_review"] == 1
    assert "tw-3" not in set(df["id"])  # category None is dropped
    r = df.set_index("id")
    assert bool(r.loc["tw-1", "reviewed"]) and not bool(r.loc["tw-2", "reviewed"])
    assert r.loc["tw-1", "ood_split"] == "ood-dev" and r.loc["t1", "ood_split"] == ood.hash_split(
        "t1"
    )
    samples = df[df["source"] == "sample_emails"]
    assert len(samples) == 22 and samples["reviewed"].all()
    assert set(df["ood_split"]) <= {"ood-dev", "ood-test"}


def test_assign_splits_is_stratified_stable_and_order_free():
    df = pd.DataFrame(
        {"id": [f"r{i}" for i in range(50)], "source": "s",
         "category": ["A"] * 30 + ["B"] * 20}
    )  # fmt: skip
    a = ood.assign_splits(df, ["source", "category"])
    b = ood.assign_splits(df.sample(frac=1, random_state=3), ["source", "category"])
    assert (a == b.loc[a.index]).all()
    dev = df[a == "ood-dev"]
    assert (dev["category"] == "A").sum() == 12 and (dev["category"] == "B").sum() == 8
    relabelled = df.assign(category="A")
    # Persisted splits don't move on relabel; recomputing would, which is why they are stored.
    assert ood.assign_splits(relabelled, ["source", "category"]).eq("ood-dev").sum() == 20


def test_persist_splits_keeps_reviews(tmp_path):
    p = tmp_path / "tw.csv"
    rows = [{"id": f"tw-{i}", "source": "twitter", "brand": "", "subject": "", "body": "b",
             "llm_category": "Billing", "category": "Billing", "needs_review": "false",
             "reviewer_notes": "checked"} for i in range(10)]  # fmt: skip
    ood.write_csv(p, rows, [f for f in ood.REVIEW_FIELDS if f != "ood_split"])
    assert ood.persist_splits(p, ood.REVIEW_FIELDS) == 10
    df = ood.read_csv(p)
    assert (df["ood_split"] == "ood-dev").sum() == 4 and (df["reviewer_notes"] == "checked").all()
    assert ood.persist_splits(p, ood.REVIEW_FIELDS) == 0  # idempotent


def test_score_counts_muted_customers():
    df = pd.DataFrame(
        {"category": ["Billing", "Spam", "Technical Issue"], "source": ["a", "a", "b"]}
    )
    s = score(df, ["Spam", "Spam", "Technical Issue"], [False, False, True])
    assert (s["non_spam_muted"], s["non_spam_n"], s["spam_caught"], s["spam_n"]) == (1, 2, 1, 1)
    assert s["non_spam_in_review"] == 1 and s["correct"] == 2
    bs = by_source(df, ["Spam", "Spam", "Technical Issue"], None)
    assert bs["b"]["correct"] == 1 and bs["all"]["n"] == 3
    m = {"split": "ood-dev", "rows": 3, "reviewed_rows": 3, "pending_rows": 0,
         "pending_by_source": {}, "gold_reviewed": {}, "reviewed": {"fast_only": bs},
         "preliminary_incl_unreviewed": {"fast_only": bs}, "versions": {}}  # fmt: skip
    md = report_markdown(m)
    assert "| fast_only | all | 3 | 2 |" in md and "PRELIMINARY" in md and "ood-dev" in md
