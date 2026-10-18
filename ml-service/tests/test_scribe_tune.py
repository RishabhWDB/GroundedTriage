import numpy as np
import pandas as pd

from app.taxonomy import categories
from eval.scribe_tune import load_candidates, predict_combo, search, stratified_sample


def test_candidates_file_starts_with_untuned_wording():
    from app.taxonomy import category_descriptions

    templates, desc = load_candidates()
    assert templates[0] == "This email is {}."
    untuned = category_descriptions()
    assert all(desc[c][0] == " ".join(untuned[c].split()) for c in categories())
    assert all(3 <= len(v) <= 5 for v in desc.values())


def test_stratified_sample_caps_each_class():
    df = pd.DataFrame({"id": [f"r{i}" for i in range(30)], "category": ["A"] * 20 + ["B"] * 10})
    s = stratified_sample(df, per_class=8, seed=1)
    assert s["category"].value_counts().to_dict() == {"A": 8, "B": 8}
    assert list(s["id"]) == list(stratified_sample(df, per_class=8, seed=1)["id"])


def build(n_cands):
    templates = ["T0 {}", "T1 {}"]
    hyp_index, k = {}, 0
    for t in templates:
        for lab, n in n_cands.items():
            for c in range(n):
                hyp_index[(t, lab, c)] = k
                k += 1
    return templates, hyp_index, k


def test_search_finds_the_better_wording_and_rules_win():
    n_cands = {"A": 2, "B": 2}
    templates, idx, n_h = build(n_cands)
    gold = ["A", "A", "B", "B"]
    logits = np.zeros((4, n_h), dtype=np.float32)
    # untuned (candidate 0): A's description fires on everything -> B rows wrong
    for i in range(4):
        logits[i, idx[("T0 {}", "A", 0)]] = 2.0
        logits[i, idx[("T0 {}", "B", 0)]] = 1.0
    # B's candidate 1 fires strongly only on B rows
    logits[2, idx[("T0 {}", "B", 1)]] = 5.0
    logits[3, idx[("T0 {}", "B", 1)]] = 5.0
    rule_pred = [None, None, None, None]
    res = search(logits, idx, templates, n_cands, gold, rule_pred)
    assert res["untuned_macro_f1"] < 1.0
    assert (
        res["template"] == "T0 {}" and res["choice"] == {"A": 0, "B": 1} and res["macro_f1"] == 1.0
    )
    # a rule decision overrides zero-shot
    assert predict_combo(logits, idx, "T0 {}", {"A": 0, "B": 0}, ["B", None, None, None])[0] == "B"


def test_search_keeps_untuned_on_ties():
    n_cands = {"A": 3, "B": 2}
    templates, idx, n_h = build(n_cands)
    logits = np.zeros((2, n_h), dtype=np.float32)
    for t in templates:
        logits[0, idx[(t, "A", 0)]] = 1.0
        logits[1, idx[(t, "B", 0)]] = 1.0
    res = search(logits, idx, templates, n_cands, ["A", "B"], [None, None])
    assert res["choice"] == {"A": 0, "B": 0} and res["template"] == "T0 {}"
