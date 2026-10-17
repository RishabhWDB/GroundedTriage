"""OOD evaluation gate (P3.11; replaces the P3.5a sample check).

    python -m eval.ood_eval [--with-baseline]     # == make ood-eval

Scores every system on the human-written OOD set (data.ood.load_ood: reviewed
Twitter rows, Enron spam, team-written rows, the 22 sample emails). Unreviewed
rows are never scored; their count is reported as pending.

Systems:
- scribe (with --with-baseline; loads BART-MNLI in this process, ~1.6 GB);
- fast_only: the 5-class fast tier alone (it cannot predict Spam);
- cascade[<rule>]: spam filter (each stored threshold rule) + fast tier
  (+ deep tier when models/deep_tier/ is valid).

**Required gate:** run it, and report it, before accepting any new model
(fast tier, deep tier, spam filter, threshold rule). OOD is a headline next
to test in P7. Writes results/ood/{report.md, ood_eval.json, predictions.csv}.
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


def run_systems(df: pd.DataFrame, fast: FastTier, deep, spam_dir: Path, with_baseline: bool):
    texts = [model_input(s, b) for s, b in zip(df["subject"], df["body"], strict=True)]
    tau = load_tau()
    preds: dict[str, list[str]] = {}
    possible: dict[str, list[bool] | None] = {}
    versions: dict[str, dict] = {}

    fp = fast.predict(texts)
    preds["fast_only"] = [p.category for p in fp]
    possible["fast_only"] = None
    versions["fast_only"] = {"fast_tier": fast.version}

    card_path = spam_dir / "model_card.json"
    rules = list(json.loads(card_path.read_text()).get("rules", {})) if card_path.exists() else []
    for rule in rules or [None]:
        spam = SpamFilter.load(spam_dir, rule) if rule else None
        c = Cascade(fast, deep, UrgencyPredictor(None), tau, spam=spam)
        res = c.run(texts)
        name = f"cascade[{rule or 'no_spam_filter'}]"
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


def evaluate(with_baseline: bool = False, out: Path = RESULTS) -> dict | None:
    s = get_settings()
    fast = FastTier.load(s.models_dir / "fast_tier")
    if fast is None:
        print("ood-eval needs a 5-class models/fast_tier/: run `make fast-tier`")
        return None
    df, counts = load_ood()
    df = df.reset_index(drop=True)
    deep = DeepTier(s.models_dir / "deep_tier")
    preds, possible, versions = run_systems(
        df, fast, deep, s.models_dir / "spam_filter", with_baseline
    )
    m = {
        "rows": len(df),
        "counts": counts,
        "gold": df["category"].value_counts().to_dict(),
        "systems": {k: by_source(df, v, possible[k]) for k, v in preds.items()},
        "versions": versions,
        "run": run_info(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "ood_eval.json").write_text(json.dumps(m, indent=2, default=str))
    pd.DataFrame({"id": df["id"], "source": df["source"], "gold": df["category"], **preds}).to_csv(
        out / "predictions.csv", index=False
    )
    (out / "report.md").write_text(report_markdown(m))
    return m


def report_markdown(m: dict) -> str:
    systems = list(m["systems"])
    sources = ["all"] + sorted({s for v in m["systems"].values() for s in v} - {"all"})
    lines = [
        "# OOD evaluation (human-written; never trained on)",
        "",
        "Generated by `make ood-eval`. Required gate before accepting any model. Only reviewed "
        f"rows are scored: {m['rows']} rows; counts {m['counts']}; gold {m['gold']}.",
        f"Categories: {', '.join(categories())}. Macro-F1 is over the classes present in each "
        "slice's gold.",
        "",
        "| system | slice | n | correct | accuracy | macro-F1 | non-spam muted | non-spam in "
        "review | spam caught |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sys_ in systems:
        for src in sources:
            r = m["systems"][sys_].get(src)
            if not r:
                continue
            review = r.get("non_spam_in_review", "–")
            lines.append(
                f"| {sys_} | {src} | {r['n']} | {r['correct']} | {r['accuracy']} | "
                f"{r['macro_f1']} | {r['non_spam_muted']}/{r['non_spam_n']} | {review} | "
                f"{r['spam_caught']}/{r['spam_n']} |"
            )
    lines += ["", f"Model versions: {m['versions']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-baseline", action="store_true", help="also run SCRIBE (BART-MNLI)")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    m = evaluate(args.with_baseline, Path(args.out))
    if m is None:
        return 1
    print(report_markdown(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
