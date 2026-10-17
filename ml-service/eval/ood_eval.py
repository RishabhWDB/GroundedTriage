"""OOD evaluation gate (P3.11; replaces the P3.5a sample check).

    python -m eval.ood_eval [--with-baseline]     # == make ood-eval

Scores every system on one half of the human-written OOD set
(data.ood.load_ood: Twitter, Enron spam, team-written rows, the 22 sample
emails): **ood-dev** (default; choose variants, thresholds and rules on val +
ood-dev only) or **ood-test** (--test; report once per final system, logged
to results/ood/test_log.jsonl). Each report has a table on reviewed labels
and a separate PRELIMINARY table that includes needs_review=true rows.

Systems:
- scribe (with --with-baseline; loads BART-MNLI in this process, ~1.6 GB);
- fast_only: the 5-class fast tier alone (it cannot predict Spam);
- cascade[<rule>]: spam filter (each stored threshold rule) + fast tier
  (+ deep tier when models/deep_tier/ is valid).

**Required gate:** run it, and report it, before accepting any new model
(fast tier, deep tier, spam filter, threshold rule). OOD is a headline next
to test in P7. Writes results/ood/{dev,test}/{report.md, ood_eval.json, predictions.csv}.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from app.interfaces import TicketText
from app.settings import REPO_ROOT, get_settings
from app.taxonomy import categories
from app.textinput import model_input
from data.ood import load_ood
from eval.classification import classification_metrics
from eval.runinfo import run_info
from triage.cascade import Cascade, load_tau
from triage.deep_tier import DeepTier
from triage.fast_tier import FastTier
from triage.spam_filter import SpamFilter
from triage.urgency import UrgencyPredictor

logger = logging.getLogger(__name__)

RESULTS = REPO_ROOT / "results" / "ood"
SPAM = "Spam"


def score(df: pd.DataFrame, pred: list[str], possible_spam: list[bool] | None = None) -> dict:
    gold = df["category"].tolist()
    is_spam = df["category"].eq(SPAM).to_numpy()
    p = pd.Series(pred)
    m = classification_metrics(gold, pred)
    out = {
        "n": len(df),
        "correct": int(sum(g == q for g, q in zip(gold, pred, strict=True))),
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "non_spam_muted": int((p[~is_spam] == SPAM).sum()),
        "non_spam_n": int((~is_spam).sum()),
        "spam_caught": int((p[is_spam] == SPAM).sum()),
        "spam_n": int(is_spam.sum()),
    }
    if possible_spam is not None:
        ps = pd.Series(possible_spam)
        out["non_spam_in_review"] = int(ps[~is_spam].sum())
        out["spam_in_review"] = int(ps[is_spam].sum())
    return out


def by_source(df: pd.DataFrame, pred: list[str], possible: list[bool] | None) -> dict:
    out = {"all": score(df, pred, possible)}
    for src, idx in df.groupby("source").groups.items():
        rows = df.loc[idx]
        pos = [df.index.get_loc(i) for i in idx]
        out[src] = score(
            rows, [pred[i] for i in pos], None if possible is None else [possible[i] for i in pos]
        )
    return out


def run_systems(
    df: pd.DataFrame, fasts: dict[str, FastTier], deep, spam_dir: Path, with_baseline: bool
):
    """fasts: {"": base fast tier, "aug": the style-aug ablation, ...}."""
    texts = [model_input(s, b) for s, b in zip(df["subject"], df["body"], strict=True)]
    tau = load_tau()
    preds: dict[str, list[str]] = {}
    possible: dict[str, list[bool] | None] = {}
    versions: dict[str, dict] = {}
    card_path = spam_dir / "model_card.json"
    rules = list(json.loads(card_path.read_text()).get("rules", {})) if card_path.exists() else []

    for tag, fast in fasts.items():
        sfx = f"[{tag}]" if tag else ""
        fp = fast.predict(texts)
        preds[f"fast_only{sfx}"] = [p.category for p in fp]
        possible[f"fast_only{sfx}"] = None
        versions[f"fast_only{sfx}"] = {"fast_tier": fast.version}
        for rule in rules or [None]:
            spam = SpamFilter.load(spam_dir, rule) if rule else None
            c = Cascade(fast, deep, UrgencyPredictor(None), tau, spam=spam)
            res = c.run(texts)
            name = f"cascade[{rule or 'no_spam_filter'}{'|' + tag if tag else ''}]"
            preds[name] = [r.category for r in res]
            possible[name] = [r.possible_spam for r in res]
            versions[name] = c.model_versions()

    if with_baseline:
        from baseline.scribe import ScribeClassifier

        tickets = [
            TicketText(id=r.id, subject=r.subject, body=r.body) for r in df.itertuples(index=False)
        ]
        preds["scribe"] = [p.category for p in ScribeClassifier().classify(tickets)]
        possible["scribe"] = None
        versions["scribe"] = {"scribe": "bart-large-mnli + rules"}
    return preds, possible, versions


def evaluate(split: str = "ood-dev", with_baseline: bool = False, out: Path = RESULTS):
    """Score every system on one OOD split. ood-dev: for choosing variants and
    thresholds. ood-test: report once per final system; every run is appended
    to results/ood/test_log.jsonl so repeated looks are visible."""
    if split not in ("ood-dev", "ood-test"):
        raise ValueError("split must be ood-dev or ood-test")
    s = get_settings()
    fast = FastTier.load(s.models_dir / "fast_tier")
    if fast is None:
        print("ood-eval needs a 5-class models/fast_tier/: run `make fast-tier`")
        return None
    df, counts = load_ood()
    df = df[df["ood_split"] == split].reset_index(drop=True)
    deep = DeepTier(s.models_dir / "deep_tier")
    fasts = {"": fast}
    if (s.models_dir / "fast_tier_aug").exists():
        aug = FastTier.load(s.models_dir / "fast_tier_aug")
        if aug is not None:
            fasts["aug"] = aug
    preds, possible, versions = run_systems(
        df, fasts, deep, s.models_dir / "spam_filter", with_baseline
    )
    rev = df["reviewed"].to_numpy()
    reviewed = df[rev].reset_index(drop=True)
    m = {
        "split": split,
        "rows": len(df),
        "reviewed_rows": int(rev.sum()),
        "pending_rows": int((~rev).sum()),
        "pending_by_source": df[~rev]["source"].value_counts().to_dict(),
        "counts": counts,
        "gold_reviewed": reviewed["category"].value_counts().to_dict(),
        "reviewed": {
            k: by_source(
                reviewed,
                [p for p, r in zip(v, rev, strict=True) if r],
                None
                if possible[k] is None
                else [p for p, r in zip(possible[k], rev, strict=True) if r],
            )
            for k, v in preds.items()
        },  # fmt: skip
        "preliminary_incl_unreviewed": {k: by_source(df, v, possible[k]) for k, v in preds.items()},
        "versions": versions,
        "run": run_info(),
    }
    sub = out / ("test" if split == "ood-test" else "dev")
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "ood_eval.json").write_text(json.dumps(m, indent=2, default=str))
    pd.DataFrame(
        {"id": df["id"], "source": df["source"], "reviewed": df["reviewed"],
         "gold": df["category"], **preds}
    ).to_csv(sub / "predictions.csv", index=False)  # fmt: skip
    (sub / "report.md").write_text(report_markdown(m))
    if split == "ood-test":
        with open(out / "test_log.jsonl", "a") as f:
            f.write(json.dumps({"date": m["run"]["date"], "git_commit": m["run"]["git_commit"],
                                "systems": list(preds), "versions": versions}) + "\n")  # fmt: skip
    return m


def _table(results: dict) -> list[str]:
    lines = [
        "| system | slice | n | correct | accuracy | macro-F1 | non-spam muted | non-spam in "
        "review | spam caught |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    sources = ["all"] + sorted({s for v in results.values() for s in v} - {"all"})
    for sys_, by_src in results.items():
        for src in sources:
            r = by_src.get(src)
            if not r:
                continue
            review = r.get("non_spam_in_review", "–")
            lines.append(
                f"| {sys_} | {src} | {r['n']} | {r['correct']} | {r['accuracy']} | "
                f"{r['macro_f1']} | {r['non_spam_muted']}/{r['non_spam_n']} | {review} | "
                f"{r['spam_caught']}/{r['spam_n']} |"
            )
    return lines


def report_markdown(m: dict) -> str:
    use = (
        "**ood-dev**: for choosing variants, thresholds and rules (with val)."
        if m["split"] == "ood-dev"
        else "**ood-test**: reported once per final system; never used for choices."
    )
    lines = [
        f"# OOD evaluation, {m['split']} (human-written; never trained on)",
        "",
        f"Generated by `make ood-eval{'-test' if m['split'] == 'ood-test' else ''}`. {use}",
        f"Categories: {', '.join(categories())}. Macro-F1 is over the classes present in each "
        "slice's gold.",
        "",
        f"## Reviewed labels ({m['reviewed_rows']} rows; gold {m['gold_reviewed']})",
        "",
        *_table(m["reviewed"]),
        "",
        f"## PRELIMINARY: all {m['rows']} rows, including {m['pending_rows']} with "
        f"needs_review=true (LLM pre-labels: {m['pending_by_source']})",
        "",
        "Numbers in this table rest partly on unreviewed labels and are preliminary.",
        "",
        *_table(m["preliminary_incl_unreviewed"]),
        "",
        f"Model versions: {m['versions']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-baseline", action="store_true", help="also run SCRIBE (BART-MNLI)")
    ap.add_argument("--test", action="store_true", help="score ood-test (final systems only)")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    m = evaluate("ood-test" if args.test else "ood-dev", args.with_baseline, Path(args.out))
    if m is None:
        return 1
    print(report_markdown(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
