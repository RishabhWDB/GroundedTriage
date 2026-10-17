"""Data-quality audit of the processed splits (P2.1g).

    python -m data.audit          # == make data-audit

Four checks, written to results/data_audit/{report.md,audit.json}:

1. A seeded sample of synthetic Meeting Scheduling emails, printed in full.
2. Synthetic diversity: exact and near-duplicate pairs within the synthetic
   rows, and the most common subject openings.
3. Leakage: near-duplicate pairs across every pair of splits (train pool,
   val, test), with the test side broken down per category x origin. The
   group-aware split (data.build) should make every count 0.
4. Shortcut check: a subject-only TF-IDF+LR classifier trained on train and
   scored on val, next to body-only and full-text models, plus a probe of real
   val tickets that mention a meeting or call.

Near-duplicate = cosine > NEAR_DUP over word-unigram TF-IDF (data.neardup).
For the leakage check the vectorizer is fitted on all rows, as in the build,
so the similarities are the ones the split used.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from app.settings import REPO_ROOT
from app.taxonomy import topic_categories
from data.build import OUT_DIR
from data.neardup import NEAR_DUP, similar_pairs, vectorizer
from data.textclean import dedupe_key
from eval.classification import MEETING_WORDS

logger = logging.getLogger(__name__)

SEED = 42
N_SAMPLES = 8
RESULTS_DIR = REPO_ROOT / "results" / "data_audit"
MEETING = "Meeting Scheduling"


def exact_duplicate_pairs(texts: list[str]) -> int:
    counts = Counter(dedupe_key(t) for t in texts)
    return sum(n * (n - 1) // 2 for n in counts.values())


def near_duplicate_pairs(texts: list[str], threshold: float = NEAR_DUP) -> list[tuple]:
    """(i, j, cosine) for every pair i < j above threshold."""
    x = vectorizer().fit_transform(texts)
    sim = (x @ x.T).toarray()
    i, j = np.where(np.triu(sim, k=1) > threshold)
    return [(int(a), int(b), float(sim[a, b])) for a, b in zip(i, j, strict=True)]


def subject_openings(subjects: list[str], words: int = 2, top: int = 10) -> list[tuple]:
    def opening(s: str) -> str:
        toks = re.findall(r"[a-z0-9']+", s.lower())
        return " ".join(toks[:words])

    return Counter(opening(s) for s in subjects).most_common(top)


def cross_max_similarity(
    pool: list[str],
    queries: list[str],
    threshold: float = NEAR_DUP,
    chunk: int = 500,
    corpus: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query: the max cosine against the pool, the index of that pool
    row, and how many pool rows exceed threshold. The vectorizer is fitted on
    ``corpus`` (default: both sets)."""
    vec = vectorizer().fit(corpus if corpus is not None else pool + queries)
    p, q = vec.transform(pool), vec.transform(queries)
    best = np.zeros(len(queries), dtype=np.float32)
    arg = np.zeros(len(queries), dtype=np.int64)
    over = np.zeros(len(queries), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        sim = (q[start : start + chunk] @ p.T).toarray()
        best[start : start + chunk] = sim.max(axis=1)
        arg[start : start + chunk] = sim.argmax(axis=1)
        over[start : start + chunk] = (sim > threshold).sum(axis=1)
    return best, arg, over


def leakage_table(
    pool: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = NEAR_DUP,
    corpus: list[str] | None = None,
) -> dict:
    best, arg, over = cross_max_similarity(
        pool["text"].tolist(), test["text"].tolist(), threshold, corpus=corpus
    )
    t = test[["category", "origin"]].copy()
    t["leaked"] = best > threshold
    t["pairs"] = over
    t["neighbour_same_category"] = (
        pool["category"].to_numpy()[arg] == test["category"].to_numpy()
    ) & t["leaked"]
    rows = []
    for (cat, origin), g in t.groupby(["category", "origin"], sort=True):
        rows.append(
            {
                "category": cat,
                "origin": origin,
                "test_rows": len(g),
                "test_rows_with_near_dup": int(g["leaked"].sum()),
                "share": round(float(g["leaked"].mean()), 4),
                "near_dup_pairs": int(g["pairs"].sum()),
                "nearest_same_category": int(g["neighbour_same_category"].sum()),
            }
        )
    return {
        "per_stratum": rows,
        "total_test_rows_with_near_dup": int(t["leaked"].sum()),
        "total_pairs": int(t["pairs"].sum()),
        "max_similarity_quantiles": {
            q: round(float(np.quantile(best, q)), 4) for q in (0.5, 0.9, 0.99)
        },
    }


def cross_split_pairs(
    parts: dict[str, pd.DataFrame], corpus: list[str], threshold: float = NEAR_DUP
) -> dict:
    """Near-duplicate pairs between every pair of splits, with a vectorizer
    fitted on ``corpus``."""
    vec = vectorizer().fit(corpus)
    mats = {k: vec.transform(v["text"].tolist()) for k, v in parts.items()}
    names = list(parts)
    out = {}
    for a_i, a in enumerate(names):
        for b in names[a_i + 1 :]:
            i, j = similar_pairs(mats[a], mats[b], threshold)
            out[f"{a}~{b}"] = {
                "pairs": len(i),
                f"{a}_rows": len(set(i.tolist())),
                f"{b}_rows": len(set(j.tolist())),
            }
    return out


def fit_eval(train: pd.DataFrame, val: pd.DataFrame, col: str) -> tuple[dict, np.ndarray]:
    vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    xtr = vec.fit_transform(train[col].fillna(""))
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(xtr, train["category"])
    pred = clf.predict(vec.transform(val[col].fillna("")))
    labels = topic_categories()  # the topic task (Spam is the spam filter's)
    p, r, f, n = precision_recall_fscore_support(
        val["category"], pred, labels=labels, zero_division=0
    )
    per_class = {
        lab: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4)}
        for i, lab in enumerate(labels)
        if n[i]
    }
    others = [lab for lab in per_class if lab != MEETING]
    return {
        "accuracy": round(accuracy_score(val["category"], pred), 4),
        "macro_f1": round(f1_score(val["category"], pred, labels=labels, average="macro"), 4),
        "per_class": per_class,
        "meeting_f1": per_class.get(MEETING, {}).get("f1"),
        "other_classes_mean_f1": round(float(np.mean([per_class[o]["f1"] for o in others])), 4),
    }, pred


def shortcut_check(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    out, preds = {}, {}
    for col in ("subject", "body", "text"):
        out[col], preds[col] = fit_eval(train, val, col)
    real = val["origin"].to_numpy() == "real"
    mentions = real & val["text"].str.contains(MEETING_WORDS).to_numpy()
    out["hard_negative_probe"] = {
        "real_val_rows": int(real.sum()),
        "real_val_rows_mentioning_meeting_words": int(mentions.sum()),
        "predicted_meeting_by_subject_model": int((preds["subject"][mentions] == MEETING).sum()),
        "predicted_meeting_by_text_model": int((preds["text"][mentions] == MEETING).sum()),
        "pattern": MEETING_WORDS.pattern,
    }
    out["empty_subject_share_val"] = round(float((val["subject"].fillna("") == "").mean()), 4)
    return out


def synthetic_section(syn: pd.DataFrame) -> dict:
    texts = syn["text"].tolist()
    pairs = near_duplicate_pairs(texts)
    ids = syn["id"].tolist()
    return {
        "rows": len(syn),
        "exact_duplicate_body_pairs": exact_duplicate_pairs(syn["body"].tolist()),
        "exact_duplicate_subject_pairs": exact_duplicate_pairs(syn["subject"].tolist()),
        "unique_subjects": int(syn["subject"].str.lower().nunique()),
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_examples": [
            {"a": ids[i], "b": ids[j], "cosine": round(c, 4)}
            for i, j, c in sorted(pairs, key=lambda x: -x[2])[:10]
        ],
        "subject_openings_top10": subject_openings(syn["subject"].tolist()),
        "median_body_chars": int(syn["body"].str.len().median()),
    }


def load() -> dict[str, pd.DataFrame]:
    return {
        n: pd.read_parquet(OUT_DIR / f"{n}.parquet")
        for n in ("train", "val", "test", "train_urgency", "spam_train", "spam_val", "spam_test")
    }


def train_pool(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every row any model trains on: capped topic train, the urgency file and
    the spam filter's train split."""
    pool = pd.concat([splits["train"], splits["train_urgency"], splits["spam_train"]])
    return pool.drop_duplicates("id").sort_values("id").reset_index(drop=True)


def render(audit: dict, samples: pd.DataFrame) -> str:
    s, lk, sc = audit["synthetic"], audit["leakage"], audit["shortcut"]
    lines = [
        "# Data audit (P2.1g)",
        "",
        "Generated by `make data-audit` (`ml-service/data/audit.py`); do not edit by hand.",
        f"Near-duplicate = cosine > {NEAR_DUP} on word-unigram TF-IDF (sublinear tf). Unigrams",
        "ignore word order, so these counts are an upper bound.",
        "",
        f"## 1. {N_SAMPLES} random synthetic Meeting Scheduling emails (seed {SEED})",
        "",
    ]
    for _, r in samples.iterrows():
        lines += [f"### {r['id']}", "", f"**Subject:** {r['subject']}", "", "```text"]
        lines += [r["body"], "```", ""]
    lines += [
        "## 2. Synthetic diversity",
        "",
        f"- Rows: {s['rows']}; median body {s['median_body_chars']} chars.",
        f"- Exact duplicate body pairs: {s['exact_duplicate_body_pairs']}; exact duplicate "
        f"subject pairs: {s['exact_duplicate_subject_pairs']} ({s['unique_subjects']} unique subjects).",
        f"- Near-duplicate pairs (subject + body): {s['near_duplicate_pairs']}.",
        "",
    ]
    if s["near_duplicate_examples"]:
        lines += ["| a | b | cosine |", "|---|---|---|"]
        lines += [f"| {e['a']} | {e['b']} | {e['cosine']} |" for e in s["near_duplicate_examples"]]
        lines.append("")
    lines += ["Top 10 subject openings (first two words):", "", "| opening | rows |", "|---|---|"]
    lines += [f"| {o} | {n} |" for o, n in s["subject_openings_top10"]]
    lines += [
        "",
        "## 3. Leakage across splits",
        "",
        f"Vectorizer fitted on all {audit['corpus_rows']} rows (same corpus as the build: "
        f"{audit['corpus_matches_build']}). Near-duplicate pairs between splits:",
        "",
        "| splits | pairs | rows on each side |",
        "|---|---|---|",
        *[
            f"| {k} | {v['pairs']} | "
            + ", ".join(f"{n}: {c}" for n, c in v.items() if n != "pairs")
            + " |"
            for k, v in audit["cross_split_pairs"].items()
        ],
        "",
        "### Train → test, per category × origin",
        "",
        f"Pool: every training row (capped `train` ∪ `train_urgency` ∪ `spam_train`, {audit['train_pool_rows']} rows); val/test are the union of both tasks' splits.",
        f"Test rows with a near-duplicate in the pool: {lk['total_test_rows_with_near_dup']} of "
        f"{audit['test_rows']}; pairs: {lk['total_pairs']}. Max-similarity quantiles: "
        f"{lk['max_similarity_quantiles']}.",
        "",
        "| category | origin | test rows | with near-dup | share | pairs | nearest has same category |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['category']} | {r['origin']} | {r['test_rows']} | {r['test_rows_with_near_dup']} "
        f"| {r['share']:.2%} | {r['near_dup_pairs']} | {r['nearest_same_category']} |"
        for r in lk["per_stratum"]
    ]
    lines += [
        "",
        "## 4. Shortcut check (TF-IDF 1–2-gram + LR, class_weight=balanced; train → val)",
        "",
        "| input | accuracy | macro-F1 | Meeting F1 | mean F1 of other classes |",
        "|---|---|---|---|---|",
    ]
    for col in ("subject", "body", "text"):
        m = sc[col]
        lines.append(
            f"| {col} | {m['accuracy']} | {m['macro_f1']} | {m['meeting_f1']} | "
            f"{m['other_classes_mean_f1']} |"
        )
    lines += ["", "Per-class F1:", "", "| class | subject | body | text |", "|---|---|---|---|"]
    for lab in sc["text"]["per_class"]:
        lines.append(
            f"| {lab} | "
            + " | ".join(str(sc[c]["per_class"][lab]["f1"]) for c in ("subject", "body", "text"))
            + " |"
        )
    hp = sc["hard_negative_probe"]
    lines += [
        "",
        f"Empty subjects in val: {sc['empty_subject_share_val']:.1%}.",
        "",
        f"Hard-negative probe: {hp['real_val_rows_mentioning_meeting_words']} of "
        f"{hp['real_val_rows']} real val tickets mention a meeting word "
        f"(`{hp['pattern']}`); predicted Meeting Scheduling by the subject-only model: "
        f"{hp['predicted_meeting_by_subject_model']}, by the full-text model: "
        f"{hp['predicted_meeting_by_text_model']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    splits = load()
    syn = splits["train"].pipe(lambda d: pd.concat([d, splits["val"], splits["test"]]))
    syn = syn[syn["origin"] == "synthetic"].sort_values("id").reset_index(drop=True)
    pool = train_pool(splits)
    # val/test of both tasks (the topic splits are subsets of the spam splits).
    val = pd.concat([splits["val"], splits["spam_val"]]).drop_duplicates("id")
    test = pd.concat([splits["test"], splits["spam_test"]]).drop_duplicates("id")
    corpus_df = pd.concat([pool, val, test]).sort_values("id")
    corpus = corpus_df["text"].tolist()
    manifest = json.loads((OUT_DIR / "manifest.json").read_text())
    build_rows = sum(manifest["splits"][k]["rows"] for k in ("spam_train", "spam_val", "spam_test"))
    if build_rows != len(corpus):
        logger.warning("audit corpus %d rows != build corpus %d", len(corpus), build_rows)
    # Ham has no category; label it so per-category tables keep those rows.
    pool, val, test = (d.assign(category=d["category"].fillna("Ham")) for d in (pool, val, test))
    parts = {"train_pool": pool, "val": val, "test": test}
    audit = {
        "synthetic": synthetic_section(syn),
        "leakage": leakage_table(pool, test, corpus=corpus),
        "cross_split_pairs": cross_split_pairs(parts, corpus),
        "corpus_rows": len(corpus),
        "corpus_matches_build": build_rows == len(corpus),
        "shortcut": shortcut_check(splits["train"], splits["val"]),
        "train_pool_rows": len(pool),
        "test_rows": len(test),
    }
    samples = syn.sample(n=N_SAMPLES, random_state=SEED).sort_values("id")
    audit["synthetic"]["sample_ids"] = samples["id"].tolist()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "audit.json").write_text(json.dumps(audit, indent=2, default=float))
    (RESULTS_DIR / "report.md").write_text(render(audit, samples))
    logger.info("wrote %s", RESULTS_DIR.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
