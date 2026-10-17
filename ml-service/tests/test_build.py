import random

import pandas as pd
import pytest

from app.textinput import model_input
from data.build import assemble, cap_train, content_sha256, stratified_split

_rng = random.Random(0)
VOCAB = [f"w{k}" for k in range(5000)]


def words(n=12):
    """Random text, so fixture rows are not near-duplicates of each other."""
    return " ".join(_rng.sample(VOCAB, n))


def rows(category, origin, n, prefix, urgency="medium"):
    return pd.DataFrame(
        {
            "id": [f"{prefix}-{i:04d}" for i in range(n)],
            "source_dataset": prefix,
            "origin": origin,
            "subject": [words(3) for _ in range(n)],
            "body": [words() for _ in range(n)],
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


def test_cap_train_uses_smallest_real_class():
    train = pd.concat(frames(), ignore_index=True)
    capped, cap, removed = cap_train(train, ratio=5)
    assert cap == 500  # 5 x 100 spam rows; the 40 synthetic meeting rows don't set the cap
    counts = capped["category"].value_counts()
    assert counts["Technical Issue"] == 500 and counts["Billing"] == 200
    assert counts["Spam"] == 100 and counts["Meeting Scheduling"] == 40
    assert removed == {"Technical Issue": 500}
    assert cap_train(train, ratio=5)[0]["id"].tolist() == capped["id"].tolist()  # seeded


def test_assemble_dedupes_caps_and_formats():
    fr = frames()
    dup = fr[0].iloc[:3].copy()
    dup["id"] = ["x-1", "x-2", "x-3"]
    dup["body"] = dup["body"].str.upper()  # case-insensitive duplicate across sources
    splits, info = assemble(fr + [dup])
    assert info["cross_source_duplicates"] == 3
    total_val_test = len(splits["val"]) + len(splits["test"])
    # Topic splits hold no Spam (the spam filter decides it).
    assert total_val_test == round(1000 * 0.15) * 2 + round(200 * 0.15) * 2 + 6 * 2
    assert "Spam" not in set(splits["train"]["category"]) | set(splits["test"]["category"])
    tr = splits["train"]
    assert tr["category"].value_counts().max() <= info["train_cap"]
    r = tr.iloc[0]
    assert r["text"] == model_input(r["subject"], r["body"])
    assert splits["spam_test"].query("category == 'Spam'")["urgency"].isna().all()


def test_train_urgency_is_uncapped_labelled_train():
    splits, info = assemble(frames())
    tu = splits["train_urgency"]
    assert tu["urgency"].notna().all()
    assert set(tu["category"]) == {"Technical Issue", "Billing"}  # spam/synthetic unlabelled
    assert (tu["category"] == "Technical Issue").sum() == 700  # uncapped
    assert info["train_cap"] == 700  # 5 x 140 Billing: Spam no longer sets the topic cap
    assert not set(tu["id"]) & (set(splits["val"]["id"]) | set(splits["test"]["id"]))


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


def test_split_keeps_groups_together_and_hits_targets():
    df = pd.concat(frames(), ignore_index=True)
    df["group"] = df["id"]
    tech = df.index[df["category"] == "Technical Issue"]
    for g in range(30):  # 30 groups of 5 technical rows
        df.loc[tech[g * 5 : g * 5 + 5], "group"] = f"g{g}"
    a = stratified_split(df)
    b = stratified_split(df.sample(frac=1, random_state=1))
    assert all(a[k]["id"].tolist() == b[k]["id"].tolist() for k in a)
    where = {i: k for k, v in a.items() for i in v["id"]}
    for g in range(30):
        ids = df.loc[df["group"] == f"g{g}", "id"]
        assert len({where[i] for i in ids}) == 1
    tech_val = int((a["val"]["category"] == "Technical Issue").sum())
    assert abs(tech_val - 150) <= 5  # within one group of the target
    assert "group" not in a["train"].columns


def test_assemble_puts_near_duplicates_in_one_split():
    fr = frames()
    base = fr[0].iloc[0]
    para = pd.DataFrame(
        [
            {**base.to_dict(), "id": f"para-{k}", "body": base["body"] + f" extra{k}"}
            for k in range(6)
        ]
    )
    splits, info = assemble(fr + [para])
    ids = {base["id"], *para["id"]}
    homes = {
        k.removeprefix("spam_")
        for k, v in splits.items()
        if k != "train_urgency" and ids & set(v["id"])
    }
    assert len(homes) == 1  # e.g. train and spam_train: the same split for both tasks
    assert info["near_dup_groups"] >= 1 and info["near_dup_rows_in_groups"] >= 7


def ham(n, prefix="ham"):
    df = rows("Spam", "real", n, prefix)
    df["category"] = None
    df["source_dataset"] = "spamassassin_ham"
    return df


def test_spam_splits_share_the_topic_assignment():
    splits, info = assemble(frames() + [ham(300)])
    for split in ("train", "val", "test"):
        s = splits[f"spam_{split}"]
        assert set(s.columns) >= {"is_spam", "category"}
        assert s.loc[s["is_spam"], "category"].eq("Spam").all()
        assert s.loc[~s["is_spam"], "category"].ne("Spam").all()
        # Every topic row of a split is in the same spam split (uncapped for train).
        assert set(splits[split]["id"]) <= set(s["id"])
        assert (s["source_dataset"] == "spamassassin_ham").sum() > 0
    names = ("spam_train", "spam_val", "spam_test")
    ids = [set(splits[n]["id"]) for n in names]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, ids)) == 1000 + 200 + 100 + 40 + 300
    assert (splits["spam_val"]["source_dataset"] == "spamassassin_ham").sum() == 45
    assert info["topic_categories"] == [
        "Technical Issue", "Billing", "Feature Request", "General Inquiry", "Meeting Scheduling"
    ]  # fmt: skip
    a = content_sha256(splits["spam_test"])
    assert a == content_sha256(splits["spam_test"].copy())


def test_style_aug_split_keeps_train_sources_only():
    from data.build import style_aug_split

    splits, _ = assemble(frames())
    train = splits["train"]
    src = train.iloc[:3]
    aug = pd.DataFrame(
        {
            "id": [f"aug-{i}" for i in src["id"]] + ["aug-gone"],
            "aug_of": [*src["id"], "not-in-train"],
            "source_dataset": "style_aug",
            "origin": "augmented",
            "category": [*src["category"], "Billing"],
            "urgency": [*src["urgency"], None],
            "subject": "hey",
            "body": "informal rewrite",
        }
    )
    out = style_aug_split(train, aug)
    assert list(out["aug_of"]) == sorted(src["id"]) or set(out["aug_of"]) == set(src["id"])
    assert (out["origin"] == "augmented").all() and (out["text"] == "hey\n\ninformal rewrite").all()
    assert content_sha256(out) != content_sha256(out.assign(aug_of="x"))
    bad = aug.iloc[:1].assign(category="Spam")
    with pytest.raises(ValueError, match="disagree"):
        style_aug_split(train, bad)
    assert style_aug_split(train, aug.iloc[:0]).empty


def test_aug_negatives_only_enter_spam_train():
    from data.build import add_aug_negatives, style_aug_split

    splits, _ = assemble(frames() + [ham(50)])
    src = splits["train"].iloc[:4]
    aug = pd.DataFrame(
        {
            "id": [f"aug-{i}" for i in src["id"]],
            "aug_of": list(src["id"]),
            "source_dataset": "style_aug",
            "origin": "augmented",
            "category": list(src["category"]),
            "urgency": list(src["urgency"]),
            "subject": "hey",
            "body": "so the app keeps crashing lol",
        }
    )
    tsa = style_aug_split(splits["train"], aug)
    out = add_aug_negatives(splits["spam_train"], tsa)
    assert len(out) == len(splits["spam_train"]) + 4
    added = out[out["source_dataset"] == "style_aug"]
    assert (~added["is_spam"]).all() and out["id"].is_unique
    assert add_aug_negatives(splits["spam_train"], tsa.iloc[:0]) is splits["spam_train"]
