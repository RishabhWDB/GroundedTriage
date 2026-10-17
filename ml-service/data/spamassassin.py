"""Builder for the SpamAssassin public corpus: spam, and (since P3.6) ham.

Spam rows are the Spam class. Ham rows (legitimate personal and mailing-list
mail, also human-written) are negatives for the binary spam filter only
(P3.9): they have no topic category, so the filter cannot use "human-written
vs LLM-written" as a spam cue.

Source: https://spamassassin.apache.org/old/publiccorpus/ (Apache SpamAssassin
project). No formal licence: offered "as a service to spam filter developers";
copyright stays with the original senders. Archives are downloaded by script,
never committed or redistributed; cite the project in the report.

Archives are untrusted input: messages are read straight from the tar stream in
memory (never extracted to disk) and parsed with the stdlib email package.
"""

from __future__ import annotations

import email
import email.policy
import hashlib
import logging
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from app.taxonomy import mapping
from data.textclean import clean_body, clean_subject, dedupe_key

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0  # deterministic language detection

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
# Latest assembled version of each set (older versions overlap them).
ARCHIVES = ["20030228_spam.tar.bz2", "20050311_spam_2.tar.bz2"]
HAM_ARCHIVES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
]
MAX_BODY_CHARS = 4000
MIN_BODY_CHARS = 20
COLUMNS = ["id", "source_dataset", "origin", "subject", "body", "category", "urgency"]


@dataclass
class BuildStats:
    messages: int = 0
    parse_errors: int = 0
    too_short: int = 0
    non_english: int = 0
    duplicates_removed: int = 0
    truncated: int = 0
    rows_out: int = 0
    archive_sha256: dict[str, str] = field(default_factory=dict)


def download(raw_dir: Path, archives: list[str] = ARCHIVES) -> list[Path]:
    out = []
    for name in archives:
        path = raw_dir / "spamassassin" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("downloading %s", name)
            with httpx.stream("GET", BASE_URL + name, follow_redirects=True, timeout=300) as r:
                r.raise_for_status()
                tmp = path.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                tmp.rename(path)
        out.append(path)
    return out


def iter_messages(archive: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (member name, raw bytes) for each regular file, without extracting."""
    with tarfile.open(archive, mode="r:bz2") as tar:
        for m in tar:
            if not m.isfile() or m.name.endswith("cmds"):
                continue
            f = tar.extractfile(m)
            if f is not None:
                yield m.name, f.read()


def parse_message(raw: bytes) -> tuple[str, str]:
    """Return (subject, body text); prefers text/plain, falls back to HTML."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    subject = str(msg.get("Subject", "") or "")
    part = msg.get_body(preferencelist=("plain", "html"))
    body = ""
    if part is not None:
        try:
            body = part.get_content()
        except (LookupError, UnicodeError, AssertionError):
            payload = part.get_payload(decode=True) or b""
            body = payload.decode("latin-1", errors="replace")
    return subject, body


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def build_frame(
    messages: list[tuple[str, bytes]], kind: str = "spam"
) -> tuple[pd.DataFrame, BuildStats]:
    """kind="spam": category Spam, source spamassassin, ids sa-*.
    kind="ham": no category, source spamassassin_ham, ids sah-*."""
    if kind not in ("spam", "ham"):
        raise ValueError(f"kind must be spam or ham, got {kind!r}")
    st = BuildStats(messages=len(messages))
    category = mapping("spam")["category"] if kind == "spam" else None
    source, prefix = ("spamassassin", "sa-") if kind == "spam" else ("spamassassin_ham", "sah-")
    rows, seen = [], set()
    for name, raw in messages:
        try:
            subject, body = parse_message(raw)
        except Exception:  # malformed spam is common; skip it
            st.parse_errors += 1
            continue
        subject, body = clean_subject(subject), clean_body(body)
        if len(body) < MIN_BODY_CHARS:
            st.too_short += 1
            continue
        if not is_english(f"{subject}\n{body[:1000]}"):
            st.non_english += 1
            continue
        key = dedupe_key(body)
        if key in seen:
            st.duplicates_removed += 1
            continue
        seen.add(key)
        if len(body) > MAX_BODY_CHARS:
            body, st.truncated = body[:MAX_BODY_CHARS], st.truncated + 1
        rid = prefix + hashlib.sha1(name.encode()).hexdigest()[:12]
        rows.append((rid, source, "real", subject, body, category, None))
    df = pd.DataFrame(rows, columns=COLUMNS)
    st.rows_out = len(df)
    return df, st


def build(raw_dir: Path, kind: str = "spam") -> tuple[pd.DataFrame, BuildStats]:
    archives = download(raw_dir, ARCHIVES if kind == "spam" else HAM_ARCHIVES)
    messages = [m for a in archives for m in iter_messages(a)]
    df, st = build_frame(messages, kind)
    for a in archives:
        st.archive_sha256[a.name] = hashlib.sha256(a.read_bytes()).hexdigest()
    return df, st
