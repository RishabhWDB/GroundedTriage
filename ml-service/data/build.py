"""Assemble the train/val/test splits (P2.1, reworked in P3.7).

    python -m data.build          # == make data

Sources: Tobi-Bueck (real support tickets), SpamAssassin spam and ham, and
synthetic meeting requests (datasets/synthetic/meeting_requests.jsonl). Rows
are deduplicated across sources (exact body match), grouped into
near-duplicate groups (data.neardup: TF-IDF cosine > 0.9, connected
components), and split 70/15/15 so every group lands in exactly one split,
stratified by label x origin (label = category, or "Ham"), seed 42.

Two tasks are cut from that one assignment, so no row is in train for one
task and in val/test for the other:

- topic (fast + deep tiers): the five topic categories (no Spam, no ham).
  train/val/test; only train is capped at CAP_RATIO x the smallest *real*
  topic class. train_urgency = the uncapped topic train rows with an urgency
  label.
- spam filter: spam_train/spam_val/spam_test = every row of the split (topic
  tickets uncapped, spam, ham) with is_spam.

Writes datasets/processed/*.parquet, manifest.json (file + content SHA-256s)
and the dataset card datasets/README.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.settings import REPO_ROOT
from app.taxonomy import categories, filter_categories, topic_categories, trained_urgency_levels
from app.textinput import MODEL_INPUT_FORMAT, model_input
from data import spamassassin, synthetic_meetings, tobi_bueck
from data.neardup import NEAR_DUP, near_duplicate_groups
from data.textclean import dedupe_key

logger = logging.getLogger(__name__)

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
CAP_RATIO = 5
RAW_DIR = REPO_ROOT / "datasets" / "raw"
OUT_DIR = REPO_ROOT / "datasets" / "processed"
CARD_PATH = REPO_ROOT / "datasets" / "README.md"
COLUMNS = ["id", "source_dataset", "origin", "category", "urgency", "subject", "body", "text"]
SPAM_COLUMNS = [*COLUMNS, "is_spam"]
AUG_COLUMNS = [*COLUMNS, "aug_of"]
STYLE_AUG_PATH = REPO_ROOT / "datasets" / "synthetic" / "style_aug.jsonl"
HAM = "Ham"  # stratum label for ham rows (no topic category)
TOPIC_SPLITS = ("train", "val", "test")
SPAM_SPLITS = {"train": "spam_train", "val": "spam_val", "test": "spam_test"}


def dedupe_across_sources(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    dup = df["body"].map(dedupe_key).duplicated(keep="first")
    return df[~dup].reset_index(drop=True), int(dup.sum())


def stratified_split(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Group-aware split, stratified by label x origin (label = category, or
    "Ham" for rows without one).

    Rows sharing a ``group`` value always land in the same split (rows without
    the column are their own group). Each group is labelled with its majority
    label (ties: taxonomy order, then Ham) and origin; within each stratum the
    groups are shuffled (seeded per stratum), ordered largest first, and each is assigned to the
    split furthest below its target row count. With singleton groups this gives
    exactly round(n * 0.15) val and test rows per stratum. Rows are sorted by id
    first, and groups are keyed by their smallest id, so the result doesn't
    depend on input order."""
    df = df.sort_values("id", kind="stable").reset_index(drop=True)
    group = df["group"] if "group" in df else df["id"]
    key = group.map(df.groupby(group)["id"].min())
    labels = [*categories(), HAM]
    label = df["category"].fillna(HAM)
    order = {c: i for i, c in enumerate(labels)}
    groups = (
        df.assign(_key=key, _rank=label.map(order))
        .groupby("_key", sort=True)
        .agg(
            size=("id", "size"),
            origin=("origin", "first"),
            label=("_rank", lambda r: labels[int(r.value_counts().sort_index().idxmax())]),
        )
    )
    assignment: dict[str, str] = {}
    for (lab, origin), stratum in groups.groupby(["label", "origin"], sort=True):
        # One RNG per stratum, so adding a stratum never reshuffles the others.
        stratum_seed = int.from_bytes(hashlib.sha256(f"{lab}|{origin}".encode()).digest()[:4])
        rng = np.random.default_rng([seed, stratum_seed])
        n = int(stratum["size"].sum())
        target = {
            "val": int(round(n * FRACTIONS["val"])),
            "test": int(round(n * FRACTIONS["test"])),
        }
        target["train"] = n - target["val"] - target["test"]
        filled = dict.fromkeys(FRACTIONS, 0)
        shuffled = stratum.iloc[rng.permutation(len(stratum))]
        shuffled = shuffled.sort_values("size", ascending=False, kind="stable")
        for gkey, size in shuffled["size"].items():
            split = max(FRACTIONS, key=lambda k: target[k] - filled[k])  # ties: train, val, test
            filled[split] += int(size)
            assignment[gkey] = split
    where = key.map(assignment)
    cols = [c for c in df.columns if c != "group"]
    return {
        name: df.loc[where == name, cols].sort_values("id").reset_index(drop=True)
        for name in FRACTIONS
    }


def cap_train(
    train: pd.DataFrame, ratio: int = CAP_RATIO, seed: int = SEED
) -> tuple[pd.DataFrame, int, dict]:
    """Downsample any class above ratio x the smallest real class (train only).
    Counting only origin == "real" rows keeps synthetic data from dictating how
    much real data is used."""
    counts = train.loc[train["origin"] == "real", "category"].value_counts()
    cap = int(counts.min() * ratio)
    kept, removed = [], {}
    for cat, group in train.groupby("category", sort=True):
        if len(group) > cap:
            removed[cat] = len(group) - cap
            group = group.sample(n=cap, random_state=seed)
        kept.append(group)
    return pd.concat(kept).sort_values("id").reset_index(drop=True), cap, removed


def content_sha256(df: pd.DataFrame) -> str:
    """Hash of the split's content, independent of parquet/pyarrow versions."""
    cols = (
        SPAM_COLUMNS
        if "is_spam" in df.columns
        else AUG_COLUMNS
        if "aug_of" in df.columns
        else COLUMNS
    )
    h = hashlib.sha256()
    for row in df[cols].itertuples(index=False):
        vals = [None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v for v in row]
        h.update(json.dumps(vals, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_synthetic(path: Path = synthetic_meetings.OUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m data.synthetic_meetings` first")
    df = pd.read_json(path, lines=True)
    return df.drop(columns=["spec"])


def assemble(frames: list[pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Pure part of the build (unit-tested). Returns the topic splits (train,
    val, test, train_urgency) and the spam-filter splits (spam_train,
    spam_val, spam_test). Rows with no category are ham (spam-filter
    negatives only)."""
    df = pd.concat(frames, ignore_index=True)
    unknown = set(df["category"].dropna()) - set(categories())
    if unknown:
        raise ValueError(f"categories not in taxonomy: {unknown}")
    df, cross_dups = dedupe_across_sources(df)
    df["text"] = [model_input(s, b) for s, b in zip(df["subject"], df["body"], strict=True)]
    df["urgency"] = df["urgency"].where(df["urgency"].notna(), None)
    df["category"] = df["category"].where(df["category"].notna(), None)
    df = df.sort_values("id", kind="stable").reset_index(drop=True)
    df["group"] = near_duplicate_groups(df["text"].tolist())
    sizes = df["group"].value_counts()
    multi = sizes[sizes > 1]
    mixed = df[df["group"].isin(multi.index)].groupby("group")["category"].nunique(dropna=False)
    parts = stratified_split(df[COLUMNS + ["group"]])

    spam_cats = set(filter_categories())
    splits: dict[str, pd.DataFrame] = {}
    for name in TOPIC_SPLITS:
        topic = parts[name][parts[name]["category"].isin(topic_categories())]
        splits[name] = topic.reset_index(drop=True)
    uncapped = splits["train"]
    splits["train"], cap, removed = cap_train(uncapped)
    splits["train_urgency"] = uncapped[uncapped["urgency"].notna()].reset_index(drop=True)
    for name, spam_name in SPAM_SPLITS.items():
        s = parts[name].copy()
        s["is_spam"] = s["category"].isin(spam_cats)
        splits[spam_name] = s.reset_index(drop=True)
    return splits, {
        "cross_source_duplicates": cross_dups,
        "near_dup_threshold": NEAR_DUP,
        "near_dup_groups": int(len(multi)),
        "near_dup_rows_in_groups": int(multi.sum()),
        "near_dup_largest_group": int(sizes.max()) if len(sizes) else 0,
        "near_dup_mixed_category_groups": int((mixed > 1).sum()),
        "train_cap": cap,
        "train_cap_removed": removed,
        "topic_categories": topic_categories(),
    }


def style_aug_split(train: pd.DataFrame, aug: pd.DataFrame) -> pd.DataFrame:
    """train_style_aug: augmented rows whose source row is in the (capped)
    train split, with labels re-checked against that source."""
    if aug.empty:
        return pd.DataFrame(columns=AUG_COLUMNS)
    src = train.set_index("id")
    aug = aug[aug["aug_of"].isin(src.index)].copy()
    if (aug["category"].to_numpy() != src.loc[aug["aug_of"], "category"].to_numpy()).any():
        raise ValueError("style_aug rows disagree with their source category")
    aug["urgency"] = aug["urgency"].where(aug["urgency"].notna(), None)
    aug["text"] = [model_input(s, b) for s, b in zip(aug["subject"], aug["body"], strict=True)]
    return aug[AUG_COLUMNS].sort_values("id").reset_index(drop=True)


def _counts(splits: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    return (
        pd.DataFrame({k: v[col].value_counts(dropna=False) for k, v in splits.items()})
        .fillna(0)
        .astype(int)
    )


def write_card(splits, info, stats, manifest) -> str:
    topic = {k: splits[k] for k in TOPIC_SPLITS}
    cat = (
        pd.DataFrame({k: v.groupby(["category", "origin"]).size() for k, v in topic.items()})
        .fillna(0)
        .astype(int)
    )
    spam = (
        pd.DataFrame(
            {
                k: splits[k].groupby(["is_spam", "source_dataset"]).size()
                for k in SPAM_SPLITS.values()
            }
        )
        .fillna(0)
        .astype(int)
    )
    urg = (
        pd.DataFrame(
            {
                k: splits[k]["urgency"].fillna("(none)").value_counts()
                for k in (*TOPIC_SPLITS, "train_urgency", "train_style_aug")
                if k in splits
            }
        )
        .fillna(0)
        .astype(int)
    )
    tb, sa, ham = stats["tobi_bueck"], stats["spamassassin"], stats["ham"]
    lines = [
        "# Datasets",
        "",
        "Generated by `make data` (`ml-service/data/build.py`) — do not edit by hand.",
        "Raw/processed data are never committed; `datasets/synthetic/` (our own generated",
        "rows) and this card are.",
        "",
        "## Sources",
        "",
        "| source_dataset | origin | category | licence / terms | rows used |",
        "|---|---|---|---|---|",
        f"| tobi_bueck — Tobi-Bueck/customer-support-tickets (Hugging Face), English rows | real | 4 support classes (queue/type rules in config/taxonomy.yaml) | CC-BY-NC-4.0 | {tb['rows_out']} |",
        f"| spamassassin — Apache SpamAssassin public corpus (20030228_spam, 20050311_spam_2) | real | Spam (spam filter positives) | no formal licence; offered for spam-filter development; not redistributed | {sa['rows_out']} |",
        f"| spamassassin_ham — Apache SpamAssassin public corpus (20030228_easy_ham, _easy_ham_2, _hard_ham) | real | none: spam-filter negatives only | as above | {ham['rows_out']} |",
        f"| synthetic_meetings — gpt-oss-120b, seed {synthetic_meetings.SEED}, prompt in data/synthetic_meetings.py | synthetic | Meeting Scheduling | generated for this project | {info['synthetic_rows']} |",
        "",
        "Dropped: albertobircoci/support-ticket-priority-dataset-50k (no text column).",
        "",
        "## Build steps",
        "",
        f"- Tobi-Bueck: {tb['rows_total']} rows → {tb['rows_english']} English → {tb['dropped_by_rule']} dropped by rule (Human Resources), {tb['empty_body']} empty, {tb['duplicates_removed']} duplicate bodies → {tb['rows_out']}.",
        f"- SpamAssassin ham: {ham['messages']} messages → {ham['too_short']} too short, {ham['non_english']} non-English, {ham['duplicates_removed']} duplicates → {ham['rows_out']} ({ham['truncated']} truncated).",
        f"- SpamAssassin: {sa['messages']} messages → {sa['too_short']} too short, {sa['non_english']} non-English, {sa['duplicates_removed']} duplicates → {sa['rows_out']} ({sa['truncated']} truncated to {spamassassin.MAX_BODY_CHARS} chars).",
        f"- Cross-source duplicate bodies removed: {info['cross_source_duplicates']}.",
        f"- Near-duplicates: word-unigram TF-IDF cosine > {info['near_dup_threshold']} (`data/neardup.py`), linked into groups (connected components): {info['near_dup_groups']} groups of 2+ rows covering {info['near_dup_rows_in_groups']} rows (largest {info['near_dup_largest_group']}; {info['near_dup_mixed_category_groups']} groups mix categories and are stratified by their majority category). Each group lands in exactly one split.",
        f"- Split: group-aware 70/15/15 stratified by label × origin (label = category, or Ham), seed {SEED}, one RNG per stratum. Both tasks below are cut from this one assignment, so no row is in train for one task and in val/test for the other.",
        f"- **Topic task** (`train`/`val`/`test`, fast + deep tiers): the 5 topic categories {', '.join(info['topic_categories'])}. Spam is decided by the binary spam filter, not the topic classifier.",
        f"- Topic train only: classes capped at {CAP_RATIO}× the smallest real topic class = {info['train_cap']} rows; removed {info['train_cap_removed']}. val/test are not resampled. Remaining imbalance is handled with class weights at training time (docs/model_artifacts.md).",
        "- **Spam-filter task** (`spam_train`/`spam_val`/`spam_test`, `is_spam`): every row of the split: spam (positives), ham and uncapped support tickets (negatives), so neither writing style nor source predicts the label on its own.",
        *(
            [
                f"- `train_style_aug` (optional, train only): {info['style_aug_rows']} of a planned ~{info['style_aug_target']} informal-style paraphrases of `train` rows by gpt-oss-120b (`data/style_aug.py`, seed and prompt in the module; `origin=augmented`, `aug_of` = source id). For the with/without ablation; never in val/test."
            ]
            if "style_aug_rows" in info
            else []
        ),
        f"- `train_urgency`: the uncapped train rows with an urgency label ({len(splits['train_urgency'])} rows). Same row pool as `train` before the cap, so it shares no rows with val/test.",
        f"- Model input (`text` column): {MODEL_INPUT_FORMAT} (`app/textinput.py`).",
        f"- Urgency: only Tobi-Bueck rows are labelled ({', '.join(trained_urgency_levels())}); spam and synthetic rows have none. No English 'critical' rows exist.",
        "",
        "## Topic rows per split by category × origin",
        "",
        cat.to_markdown(),
        "",
        "## Spam-filter splits by is_spam × source",
        "",
        spam.to_markdown(),
        "",
        "## Urgency labels per split (`train_urgency` is the urgency model's train file)",
        "",
        urg.to_markdown(),
        "",
        "## Split files",
        "",
        "| split | rows | file sha256 | content sha256 |",
        "|---|---|---|---|",
    ]
    for name, m in manifest["splits"].items():
        lines.append(f"| {name} | {m['rows']} | `{m['file_sha256']}` | `{m['content_sha256']}` |")
    lines += [
        "",
        "The content hash is independent of the parquet library version; compare it on the PC",
        "to confirm the test split is identical.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tb_df, tb_st = tobi_bueck.build(RAW_DIR)
    sa_df, sa_st = spamassassin.build(RAW_DIR)
    ham_df, ham_st = spamassassin.build(RAW_DIR, "ham")
    syn_df = load_synthetic()
    splits, info = assemble([tb_df, sa_df, ham_df, syn_df])
    info["synthetic_rows"] = len(syn_df)
    if STYLE_AUG_PATH.exists():
        aug = pd.read_json(STYLE_AUG_PATH, lines=True, dtype={"urgency": object})
        splits["train_style_aug"] = style_aug_split(splits["train"], aug)
        info["style_aug_rows"] = len(splits["train_style_aug"])
        info["style_aug_target"] = round(len(splits["train"]) * 0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "fractions": FRACTIONS, "cap_ratio": CAP_RATIO, "splits": {}, **info}
    for name, df in splits.items():
        path = OUT_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["splits"][name] = {
            "rows": len(df),
            "file_sha256": file_sha256(path),
            "content_sha256": content_sha256(df),
        }
    manifest["sources"] = {
        "tobi_bueck": {"raw_sha256": tb_st.raw_sha256},
        "spamassassin": {"archive_sha256": sa_st.archive_sha256},
        "spamassassin_ham": {"archive_sha256": ham_st.archive_sha256},
        "synthetic_meetings": {"file_sha256": file_sha256(synthetic_meetings.OUT_PATH)},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=int))
    stats = {"tobi_bueck": asdict(tb_st), "spamassassin": asdict(sa_st), "ham": asdict(ham_st)}
    CARD_PATH.write_text(write_card(splits, info, stats, manifest))
    for name, m in manifest["splits"].items():
        logger.info("%s: %d rows, content sha256 %s", name, m["rows"], m["content_sha256"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
