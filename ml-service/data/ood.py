"""Human-written out-of-domain (OOD) evaluation set (P3.11). Never trained on.

    python -m data.ood          # == make ood-set (refuses to overwrite a reviewed file)

Sources (datasets/ood/, see datasets/ood/README.md):

- twitter.csv: ~150 inbound first messages from "Customer Support on
  Twitter" (Kaggle thoughtvector/customer-support-on-twitter, CC BY-NC-SA 4.0,
  attribution in the README). A seeded candidate pool is pre-labelled by the
  judge LLM (qwen, no reasoning) into the 5 topic classes + Spam or "None";
  None rows are dropped and the rest are picked by per-class quota. Every row
  is written with needs_review=true; Arya edits `category` and sets
  needs_review=false. Only reviewed rows are ever scored.
- enron_spam.csv: ~20 spam emails from Enron-Spam (Metsis et al. 2006,
  enron1), a different corpus from SpamAssassin, to test whether the spam
  filter generalises. Taken from the CC0 Kaggle copy
  venky73/spam-mails-dataset (the original AUEB server's TLS chain fails
  verification). Text is the corpus's preprocessed form (lower-cased,
  tokenised). Corpus labels, so needs_review=false.
- team_written.csv: a template for ~40 support-style emails written by Arya's
  team (informal and meeting-request ones included); rows count as reviewed.
- datasets/sample_emails.csv (the 22 hand-written samples) is added at
  evaluation time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from app.llm import LLM, LLMError, build_llm
from app.settings import REPO_ROOT, get_settings
from app.taxonomy import categories
from data.synthetic_meetings import Cache
from data.textclean import clean_body, dedupe_key

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

SEED = 4242
OOD_DIR = REPO_ROOT / "datasets" / "ood"
RAW = REPO_ROOT / "datasets" / "raw"
TWCS_CSV = RAW / "twcs" / "twcs" / "twcs.csv"
TWITTER_CSV = OOD_DIR / "twitter.csv"
ENRON_CSV = OOD_DIR / "enron_spam.csv"
TEAM_CSV = OOD_DIR / "team_written.csv"
SAMPLE_CSV = REPO_ROOT / "datasets" / "sample_emails.csv"
CACHE_DIR = RAW / "llm_cache" / "ood_prelabel"
ENRON_CSV_RAW = RAW / "enron_spam" / "spam_ham_dataset.csv"  # Kaggle venky73/spam-mails-dataset

N_CANDIDATES = 600
PRELABEL_BATCH = 20
# Per-class targets for the ~150 Twitter rows (rare classes take what exists).
QUOTAS = {
    "Technical Issue": 45,
    "Billing": 40,
    "General Inquiry": 40,
    "Feature Request": 20,
    "Meeting Scheduling": 10,
    "Spam": 10,
}
N_ENRON = 20
REVIEW_FIELDS = ["id", "source", "brand", "subject", "body", "llm_category", "category",
                 "needs_review", "reviewer_notes"]  # fmt: skip
TEAM_FIELDS = ["id", "author", "subject", "body", "category", "notes"]

_MENTION = re.compile(r"@\w+")

PRELABEL_SYSTEM = (
    "You label inbound customer messages sent to companies' support accounts. Categories:\n"
    + "\n".join(f"- {c}" for c in categories())
    + "\n- None: fits none of these (e.g. pure praise, a reply fragment, unclear)\n"
    "Technical Issue = something is broken or not working (including account/login access). "
    "Billing = charges, refunds, payments, subscriptions, orders/returns money. Feature "
    "Request = asks for a new or changed feature. General Inquiry = a question or request for "
    "information, or general feedback/complaint not about a fault or billing. Meeting "
    "Scheduling = asks to set up, move or confirm a call/meeting/appointment. Spam = "
    "unsolicited promotion or scam. Output JSON only."
)
PRELABEL_USER = (
    "Label each message.\n\n{items}\n\n"
    'Return {{"labels": [{{"i": 1, "category": "..."}}, ...]}} with exactly {n} entries.'
)


def clean_tweet(text: str) -> tuple[str, str]:
    """(body without @mentions, first mentioned handle as the brand)."""
    mentions = _MENTION.findall(text)
    brand = mentions[0][1:] if mentions else ""
    body = re.sub(r"\s+", " ", _MENTION.sub("", text)).strip()
    return body, brand


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def twitter_candidates(
    path: Path = TWCS_CSV, n: int = N_CANDIDATES, seed: int = SEED
) -> pd.DataFrame:
    """Seeded sample of inbound thread-opening tweets: English, >= 60 chars
    without mentions, deduplicated."""
    keep = []
    cols = ["tweet_id", "inbound", "text", "in_response_to_tweet_id"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=250_000, dtype={"text": str}):
        c = chunk[(chunk["inbound"]) & (chunk["in_response_to_tweet_id"].isna())]
        keep.append(c[["tweet_id", "text"]])
    pool = pd.concat(keep, ignore_index=True)
    pool = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    rows, seen = [], set()
    for r in pool.itertuples(index=False):
        body, brand = clean_tweet(str(r.text))
        if len(body) < 60 or not brand:
            continue
        key = dedupe_key(body)
        if key in seen or not is_english(body):
            continue
        seen.add(key)
        rows.append({"id": f"tw-{r.tweet_id}", "source": "twitter", "brand": brand,
                     "subject": "", "body": body})  # fmt: skip
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def parse_labels(text: str, n: int) -> list[str]:
    data = json.loads(text)
    labels = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, list) or len(labels) != n:
        raise ValueError(f"expected {n} labels")
    out = [""] * n
    for e in labels:
        i, cat = int(e["i"]) - 1, str(e["category"]).strip()
        if not 0 <= i < n or (cat not in categories() and cat != "None"):
            raise ValueError(f"bad label entry {e}")
        out[i] = cat
    if "" in out:
        raise ValueError("missing label")
    return out


def prelabel(df: pd.DataFrame, llm: LLM, model: str, cache: Cache, attempts: int = 3) -> list[str]:
    labels: list[str] = []
    for b in range(0, len(df), PRELABEL_BATCH):
        batch = df.iloc[b : b + PRELABEL_BATCH]
        items = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch["body"]))
        messages = [
            {"role": "system", "content": PRELABEL_SYSTEM},
            {"role": "user", "content": PRELABEL_USER.format(items=items, n=len(batch))},
        ]
        got = None
        for attempt in range(attempts):
            msgs = [dict(m) for m in messages]
            if attempt:
                msgs[-1]["content"] += f"\n(Attempt {attempt + 1}.)"
            key = cache.key(model, msgs)
            text = cache.get(key)
            if text is None:
                text = llm.complete(msgs, temperature=0.0, json_mode=True)
                cache.put(key, text)
            try:
                got = parse_labels(text, len(batch))
                break
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("prelabel batch %d attempt %d rejected: %s", b, attempt + 1, exc)
        labels += got if got is not None else ["None"] * len(batch)
    return labels


def pick_by_quota(df: pd.DataFrame, quotas: dict[str, int] = QUOTAS) -> pd.DataFrame:
    """First rows (in the seeded candidate order) of each labelled class, up to its quota."""
    parts = [df[df["llm_category"] == c].head(q) for c, q in quotas.items()]
    return pd.concat(parts).sort_values("id").reset_index(drop=True)


def enron_spam_rows(path: Path = ENRON_CSV_RAW, n: int = N_ENRON, seed: int = SEED) -> list[dict]:
    """A seeded sample of English Enron spam (>= 40 body chars)."""
    df = pd.read_csv(path)
    rows = []
    for r in df[df["label"] == "spam"].itertuples(index=False):
        text = str(r.text)
        first, _, rest = text.partition("\n")
        subject = first[len("Subject:") :].strip() if first.startswith("Subject:") else ""
        body = clean_body(rest if subject else text)
        if len(body) >= 40 and is_english(f"{subject}\n{body}"):
            rows.append({"id": f"enron-{r[0]}", "source": "enron_spam", "subject": subject,
                         "body": body, "category": "Spam", "needs_review": "false"})  # fmt: skip
    rows.sort(key=lambda r: r["id"])
    return sorted(random.Random(seed).sample(rows, min(n, len(rows))), key=lambda r: r["id"])


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_ood(with_pending: bool = False) -> tuple[pd.DataFrame, dict]:
    """All OOD rows that may be scored: reviewed Twitter rows, Enron spam (text
    from the archive), team-written rows and the 22 sample emails. Returns
    (frame with id/source/subject/body/category, counts incl. pending)."""
    frames, counts = [], {}
    tw = read_csv(TWITTER_CSV)
    if len(tw):
        reviewed = tw["needs_review"].str.strip().str.lower() == "false"
        counts["twitter_pending_review"] = int((~reviewed).sum())
        keep = tw if with_pending else tw[reviewed]
        keep = keep[keep["category"].isin(categories())]
        frames.append(keep[["id", "source", "subject", "body", "category"]])
    en = read_csv(ENRON_CSV)
    if len(en):
        frames.append(en[["id", "source", "subject", "body", "category"]])
    team = read_csv(TEAM_CSV)
    if len(team):
        team = team[team["category"].isin(categories()) & team["body"].str.strip().ne("")]
        frames.append(team.assign(source="team")[["id", "source", "subject", "body", "category"]])
    sample = pd.read_csv(SAMPLE_CSV, dtype=str).fillna("").rename(columns=str.lower)
    frames.append(
        pd.DataFrame(
            {
                "id": [f"sample-{i:02d}" for i in range(len(sample))],
                "source": "sample_emails",
                "subject": sample["subject"],
                "body": sample["body"],
                "category": sample["label"],
            }
        )
    )
    df = pd.concat(frames, ignore_index=True)
    counts.update(df["source"].value_counts().to_dict())
    return df, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite a (possibly reviewed) CSV")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if TWITTER_CSV.exists() and not args.force:
        logger.error("%s exists (may hold your review); use --force to overwrite", TWITTER_CSV)
        return 1
    if not TWCS_CSV.exists():
        logger.error("%s missing: download thoughtvector/customer-support-on-twitter", TWCS_CSV)
        return 1
    settings = get_settings()
    # Pre-labels only need to be a good first guess for the human review: no reasoning.
    llm = build_llm(settings.model_copy(update={"judge_reasoning_effort": ""}), "judge")
    cands = twitter_candidates()
    try:
        cands["llm_category"] = prelabel(cands, llm, settings.judge_model, Cache(CACHE_DIR))
    except LLMError as exc:
        logger.error("prelabel stopped: %s (cached batches are kept; re-run)", exc)
        return 1
    logger.info("candidate pre-labels: %s", cands["llm_category"].value_counts().to_dict())
    picked = pick_by_quota(cands)
    rows = [
        {**r, "category": r["llm_category"], "needs_review": "true", "reviewer_notes": ""}
        for r in picked.to_dict("records")
    ]
    write_csv(TWITTER_CSV, rows, REVIEW_FIELDS)
    enron = enron_spam_rows()
    write_csv(ENRON_CSV, enron, ["id", "source", "subject", "body", "category", "needs_review"])
    if not TEAM_CSV.exists():
        write_csv(TEAM_CSV, [], TEAM_FIELDS)
    sha = hashlib.sha256(ENRON_CSV_RAW.read_bytes()).hexdigest()
    logger.info("twitter %d rows, enron %d (source csv sha256 %s); usage %s",
                len(rows), len(enron), sha[:12], llm.usage)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
