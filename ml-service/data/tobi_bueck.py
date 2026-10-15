"""Builder for Tobi-Bueck/customer-support-tickets (Hugging Face, CC-BY-NC-4.0).

Downloads the auto-converted parquet over plain HTTPS (no credentials), keeps
English rows, maps queue/type to the 6-category taxonomy and priority to the
urgency scale (rules in config/taxonomy.yaml), cleans text and removes
duplicate bodies.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

from app.taxonomy import mapping
from data.mapping import DROP, apply_category_rules, map_urgency
from data.textclean import clean_body, clean_subject, dedupe_key

logger = logging.getLogger(__name__)

DATASET = "Tobi-Bueck/customer-support-tickets"
PARQUET_URL = (
    "https://huggingface.co/api/datasets/Tobi-Bueck/customer-support-tickets"
    "/parquet/default/train/0.parquet"
)
COLUMNS = ["id", "source_dataset", "origin", "subject", "body", "category", "urgency"]


@dataclass
class BuildStats:
    rows_total: int = 0
    rows_english: int = 0
    dropped_by_rule: int = 0
    unmapped: int = 0
    empty_body: int = 0
    duplicates_removed: int = 0
    rows_out: int = 0
    raw_sha256: str = ""
    category_counts: dict[str, int] = field(default_factory=dict)
    urgency_counts: dict[str, int] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path, *, force: bool = False) -> Path:
    path = raw_dir / "tobi_bueck" / "train.parquet"
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    logger.info("downloading %s", DATASET)
    with httpx.stream("GET", PARQUET_URL, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.rename(path)
    return path


def build_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, BuildStats]:
    """Pure transformation (unit-tested): raw dataset rows -> normalised rows."""
    m = mapping("tobi_bueck")
    st = BuildStats(rows_total=len(raw))
    df = raw
    for column, value in m["filter"].items():
        df = df[df[column] == value]
    st.rows_english = len(df)

    category = apply_category_rules(df, m["category_rules"])
    st.dropped_by_rule = int((category == DROP).sum())
    st.unmapped = int(category.isna().sum())
    keep = category.notna() & (category != DROP)
    df, category = df[keep], category[keep]

    out = pd.DataFrame(
        {
            "id": "tb-" + df.index.astype(str),
            "source_dataset": "tobi_bueck",
            "origin": "real",
            "subject": df["subject"].map(clean_subject),
            "body": df["body"].map(clean_body),
            "category": category.astype(str),
            "urgency": map_urgency(df["priority"], m["urgency_map"]),
        }
    )
    empty = out["body"] == ""
    st.empty_body = int(empty.sum())
    out = out[~empty]
    dup = out["body"].map(dedupe_key).duplicated(keep="first")
    st.duplicates_removed = int(dup.sum())
    out = out[~dup].reset_index(drop=True)

    st.rows_out = len(out)
    st.category_counts = out["category"].value_counts().to_dict()
    st.urgency_counts = out["urgency"].value_counts().to_dict()
    return out[COLUMNS], st


def build(raw_dir: Path) -> tuple[pd.DataFrame, BuildStats]:
    path = download(raw_dir)
    df, st = build_frame(pd.read_parquet(path))
    st.raw_sha256 = sha256_file(path)
    return df, st
