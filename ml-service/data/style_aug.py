"""Optional train-only style augmentation (P3.8). Never used for val/test.

    python -m data.style_aug [--limit N]       # == make style-aug (resumable)

Why: in the topic data every real ticket is a formal LLM-written Tobi-Bueck
ticket, so a model can tie "formal LLM prose" to the support classes (P3.5a).
This rewrites ~25% of `train` into informal, human customer-email style with
the same request and label, for Arya's with/without ablation on the deep
tier (and the fast tier here).

- Selection: AUG_FRACTION of each category in the capped `train`, seeded
  (SEED), then shuffled, so any prefix of the work is a stratified sample.
  This matters because Groq's free tier allows ~1,000 requests/day and
  8,000 tokens/minute: a full run takes several days and is resumed.
- Prompt: fixed (below); a seeded style spec per row (register, length,
  greeting, typos, subject style). gpt-oss-120b, temperature 0.9, JSON mode,
  BATCH_SIZE rows per call. Responses are cached by sha256 of the request
  under datasets/raw/llm_cache/style_aug/ (gitignored).
- Output: datasets/synthetic/style_aug.jsonl (committed): id = "aug-" + source
  id, aug_of, category and urgency copied, origin "augmented". The build
  (data.build) writes train_style_aug.parquet from the rows whose source is
  in the current train split.
- Checks per row: non-empty, no bracket/angle placeholders, not a
  near-copy of the source (word-unigram cosine <= MAX_COSINE).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from app.fileio import append_line, atomic_write_text, open_append, read_jsonl
from app.llm import LLM, JSONModeError, LLMError, build_llm
from app.settings import REPO_ROOT, get_settings
from data.build import OUT_DIR
from data.neardup import vectorizer
from data.synthetic_meetings import Cache, parse_response

logger = logging.getLogger(__name__)

SEED = 2026  # the meeting generator uses 42, the stress set 1337
AUG_FRACTION = 0.25
BATCH_SIZE = 8
MAX_COSINE = 0.9
OUT_PATH = REPO_ROOT / "datasets" / "synthetic" / "style_aug.jsonl"
CACHE_DIR = REPO_ROOT / "datasets" / "raw" / "llm_cache" / "style_aug"

REGISTERS = [
    "casual and friendly",
    "terse, like a quick message typed on a phone",
    "annoyed but polite",
    "rambling, with a bit of irrelevant context",
    "plain English from a non-native speaker (simple grammar, small mistakes)",
    "matter-of-fact, no pleasantries",
]
LENGTHS = ["much shorter than the original", "about as long as the original"]
GREETINGS = ["no greeting", "'hi'", "'hey'", "'hello team'", "the support person's first name"]
TYPOS = ["no typos", "one or two small typos", "lowercase sentence starts"]
SUBJECTS = ["a short casual subject", "a lowercase subject", "a subject that is a question"]

SYSTEM_PROMPT = (
    "You rewrite customer-support tickets as real customers would write them in an email: "
    "informal, human, unpolished. Keep the SAME request, problem, product and key facts, so "
    "the ticket's topic does not change. Do not add new problems or requests. Replace any "
    "anonymised tokens such as <name>, <acc_num> or <tel_num> with realistic invented values. "
    "Never use bracketed placeholders, 'Dear Customer Support', 'I hope this message finds "
    "you well' or other stock phrases. Output JSON only."
)
USER_TEMPLATE = (
    "Rewrite each ticket below following its style spec.\n\n{items}\n\n"
    'Return {{"emails": [{{"subject": "...", "body": "..."}}, ...]}} with exactly {n} emails '
    "in the same order."
)


@dataclass(frozen=True)
class Style:
    register: str
    length: str
    greeting: str
    typos: str
    subject: str

    def describe(self) -> str:
        return (
            f"Style: {self.register}; {self.length}; greeting: {self.greeting}; "
            f"{self.typos}; {self.subject}."
        )


def select(train: pd.DataFrame, fraction: float = AUG_FRACTION, seed: int = SEED) -> pd.DataFrame:
    """A stratified sample of train (fraction of each category), in a seeded
    shuffled order so any prefix stays roughly stratified."""
    parts = [
        g.sample(n=max(1, round(len(g) * fraction)), random_state=seed)
        for _, g in train.groupby("category", sort=True)
    ]
    picked = pd.concat(parts)
    return picked.sample(frac=1, random_state=seed).reset_index(drop=True)


def styles(n: int, seed: int = SEED) -> list[Style]:
    rng = random.Random(seed)
    return [
        Style(
            rng.choice(REGISTERS),
            rng.choice(LENGTHS),
            rng.choice(GREETINGS),
            rng.choice(TYPOS),
            rng.choice(SUBJECTS),
        )
        for _ in range(n)
    ]


def build_prompt(rows: list[dict], specs: list[Style]) -> list[dict[str, str]]:
    items = "\n\n".join(
        f"### Ticket {i + 1}\n{s.describe()}\nSubject: {r['subject'] or '(none)'}\n{r['body']}"
        for i, (r, s) in enumerate(zip(rows, specs, strict=True))
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(items=items, n=len(rows))},
    ]


def too_similar(originals: list[str], rewrites: list[str]) -> list[bool]:
    vec = vectorizer().fit(originals + rewrites)
    a, b = vec.transform(originals), vec.transform(rewrites)
    sims = (a.multiply(b)).sum(axis=1).A1
    return [bool(s > MAX_COSINE) for s in sims]


def load_done(path: Path = OUT_PATH) -> dict[str, dict]:
    return {r["aug_of"]: r for r in read_jsonl(path)}


def generate(
    llm: LLM,
    model: str,
    cache: Cache,
    train: pd.DataFrame,
    out: Path = OUT_PATH,
    limit: int | None = None,
    attempts: int = 3,
) -> dict:
    picked = select(train)
    specs = styles(len(picked))
    done = load_done(out)
    todo = [i for i, r in enumerate(picked.itertuples()) if r.id not in done]
    if limit is not None:
        todo = todo[:limit]
    stats = {"selected": len(picked), "already_done": len(done), "written": 0, "rejected": 0}
    with open_append(out) as f:
        for b in range(0, len(todo), BATCH_SIZE):
            idx = todo[b : b + BATCH_SIZE]
            rows = [picked.iloc[i].to_dict() for i in idx]
            batch_specs = [specs[i] for i in idx]
            emails = None
            for attempt in range(attempts):
                messages = build_prompt(rows, batch_specs)
                if attempt:
                    messages[-1]["content"] += f"\n(Attempt {attempt + 1}.)"
                key = cache.key(model, messages)
                text = cache.get(key)
                if text is None:
                    try:
                        text = llm.complete(messages, temperature=0.9, json_mode=True)
                    except JSONModeError:
                        # Groq JSON mode sometimes fails server-side: a bad attempt.
                        logger.warning("batch at %d attempt %d: JSON mode failed", b, attempt + 1)
                        continue
                    cache.put(key, text)
                try:
                    emails = parse_response(text, len(rows))
                    break
                except ValueError as exc:
                    logger.warning("batch at %d attempt %d rejected: %s", b, attempt + 1, exc)
            if emails is None:
                stats["rejected"] += len(rows)
                continue
            similar = too_similar([r["body"] for r in rows], [e["body"] for e in emails])
            for r, s, e, sim in zip(rows, batch_specs, emails, similar, strict=True):
                if sim:
                    stats["rejected"] += 1
                    continue
                rec = {
                    "id": f"aug-{r['id']}",
                    "aug_of": r["id"],
                    "source_dataset": "style_aug",
                    "origin": "augmented",
                    "category": r["category"],
                    "urgency": r["urgency"],
                    "subject": e["subject"],
                    "body": e["body"],
                    "style": asdict(s),
                }
                append_line(f, rec, ensure_ascii=False)
                stats["written"] += 1
            if (b // BATCH_SIZE) % 10 == 0:
                logger.info("%d/%d rows this run", b + len(idx), len(todo))
    stats["total_done"] = len(load_done(out))
    return stats


def recheck(train: pd.DataFrame, path: Path = OUT_PATH) -> int:
    """Drop written rows that are near-copies of their source body (rows from
    before the body-vs-body check). Returns the number removed."""
    done = list(load_done(path).values())
    if not done:
        return 0
    src = train.set_index("id")["body"]
    keep = [r for r in done if r["aug_of"] in src.index]
    similar = too_similar([src[r["aug_of"]] for r in keep], [r["body"] for r in keep])
    kept = [r for r, s in zip(keep, similar, strict=True) if not s]
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    return len(done) - len(kept)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="rows to process this run")
    ap.add_argument("--recheck", action="store_true", help="only drop near-copies already written")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    llm = build_llm(settings, "draft")
    train = pd.read_parquet(OUT_DIR / "train.parquet")
    if args.recheck:
        logger.info("recheck removed %d near-copies", recheck(train))
        return 0
    try:
        stats = generate(llm, settings.groq_model, Cache(CACHE_DIR), train, limit=args.limit)
    except LLMError as exc:
        logger.error("stopped: %s (finished rows are kept; re-run to resume)", exc)
        logger.info("done so far: %d rows; token usage: %s", len(load_done()), llm.usage)
        return 1
    logger.info("stats %s; token usage: %s", stats, llm.usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
