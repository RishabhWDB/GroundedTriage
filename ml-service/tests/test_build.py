import pandas as pd
import pytest

from app.textinput import model_input
from data.build import assemble, cap_train, content_sha256, stratified_split


def rows(category, origin, n, prefix, urgency="medium"):
    return pd.DataFrame(
        {
            "id": [f"{prefix}-{i:04d}" for i in range(n)],
            "source_dataset": prefix,
            "origin": origin,
            "subject": [f"subj {prefix} {i}" for i in range(n)],
            "body": [f"body text {prefix} number {i}" for i in range(n)],
            "category": category,
            "urgency": urgency if origin == "real" and category != "Spam" else None,
        }
    )


def frames():
    return [
        rows("Technical Issue", "real", 1000, "tech"),
        rows("Billing", "real", 200, "bill"),
        rows("Spam", "real", 100, "spam"),
        rows("Meeting Scheduling", "synthetic", 40, "meet"),
    ]


def test_stratified_split_proportions_disjoint_and_deterministic():
    df = pd.concat(frames(), ignore_index=True)
    a = stratified_split(df)
    b = stratified_split(df.sample(frac=1, random_state=3))  # input order must not matter
    for k in a:
        assert a[k]["id"].tolist() == b[k]["id"].tolist()
    ids = [set(a[k]["id"]) for k in a]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(v) for v in a.values()) == len(df)
    for cat, n in [("Technical Issue", 1000), ("Billing", 200), ("Meeting Scheduling", 40)]:
        got = {k: int((v["category"] == cat).sum()) for k, v in a.items()}
        assert got["val"] == round(n * 0.15) and got["test"] == round(n * 0.15)


def test_cap_train_only_limits_large_classes():
    train = pd.concat(frames(), ignore_index=True)
    capped, cap, removed = cap_train(train, ratio=5)
    assert cap == 200  # 5 x 40 meeting rows
    counts = capped["category"].value_counts()
    assert counts["Technical Issue"] == 200 and counts["Billing"] == 200 and counts["Spam"] == 100
    assert removed == {"Technical Issue": 800}
    assert cap_train(train, ratio=5)[0]["id"].tolist() == capped["id"].tolist()  # seeded


def test_assemble_dedupes_caps_and_formats():
    fr = frames()
    dup = fr[0].iloc[:3].copy()
    dup["id"] = ["x-1", "x-2", "x-3"]
    dup["body"] = dup["body"].str.upper()  # case-insensitive duplicate across sources
    splits, info = assemble(fr + [dup])
    assert info["cross_source_duplicates"] == 3
    total_val_test = len(splits["val"]) + len(splits["test"])
    assert (
        total_val_test
        == round(1000 * 0.15) * 2 + round(200 * 0.15) * 2 + round(100 * 0.15) * 2 + 6 * 2
    )
    tr = splits["train"]
    assert tr["category"].value_counts().max() <= info["train_cap"]
    r = tr.iloc[0]
    assert r["text"] == model_input(r["subject"], r["body"])
    assert splits["test"].query("category == 'Spam'")["urgency"].isna().all()


def test_assemble_rejects_unknown_category():
    with pytest.raises(ValueError, match="not in taxonomy"):
        assemble([rows("Complaint", "real", 5, "c")])


def test_content_hash_is_stable():
    splits, _ = assemble(frames())
    assert content_sha256(splits["test"]) == content_sha256(splits["test"].copy())
    changed = splits["test"].copy()
    changed.loc[0, "body"] = "different"
    assert content_sha256(changed) != content_sha256(splits["test"])


def test_model_input():
    assert model_input("Hi", "Body") == "Hi\n\nBody"
    assert model_input("", " Body ") == "Body" and model_input(None, None) == ""
