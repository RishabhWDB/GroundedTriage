"""SCRIBE (val-tuned descriptions): the strengthened baseline of the Phase 2 review.

    python -m eval.scribe_tune          # == make scribe-tune (BART on MPS, ~30 min first time)

Only the zero-shot wording is tuned: the label descriptions (4 candidates per
class) and the hypothesis template (3 candidates), all fixed beforehand in
config/scribe_candidates.yaml. Rules, model and sentiment stay SCRIBE's.

- Data: a seeded stratified sample of val (the categorised rows of
  spam_val.parquet, i.e. the 6-class val incl. spam), at most PER_CLASS rows
  per category. Test is never touched here.
- Rows the regex rules accept keep the rule prediction (as in SCRIBE); the
  rest are decided by zero-shot. The entailment logit of every (row,
  template x description) pair is computed once and cached
  (datasets/raw/scribe_tune/, checkpointed every NLI_BLOCK rows, so a crash
  resumes where it stopped); the pipeline's single-label prediction is the
  argmax over the chosen descriptions, so every combination is scored exactly
  without re-running NLI.
- Search, per template: coordinate ascent from the untuned descriptions; for
  each class try its candidates and keep a change only if val macro-F1 rises
  (ties keep the current wording); repeat until nothing changes (<= 5 passes).
  The template with the best macro-F1 wins (ties: the untuned template).
- Output: config/scribe_tuned.yaml (frozen choice) and
  results/scribe_tune/search.json. The tuned baseline is then evaluated once
  on test by eval.compare, labelled a strengthened baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date

import numpy as np
import pandas as pd
import yaml

from app.fileio import atomic_path, atomic_write_text
from app.settings import REPO_ROOT
from app.taxonomy import categories
from data.build import OUT_DIR
from eval.classification import classification_metrics
from eval.runinfo import run_info

logger = logging.getLogger(__name__)

CANDIDATES = REPO_ROOT / "config" / "scribe_candidates.yaml"
TUNED = REPO_ROOT / "config" / "scribe_tuned.yaml"
CACHE = REPO_ROOT / "datasets" / "raw" / "scribe_tune"
RESULTS = REPO_ROOT / "results" / "scribe_tune"
PER_CLASS, SEED, MAX_PASSES = 120, 2027, 5
NLI_BLOCK = 40  # rows per checkpoint of the NLI pass


def load_candidates(path=CANDIDATES) -> tuple[list[str], dict[str, list[str]]]:
    c = yaml.safe_load(path.read_text())
    desc = c["descriptions"]
    if list(desc) != categories():
        raise ValueError(f"{path}: classes must be the taxonomy categories in order")
    return c["templates"], desc


def nli_logits(zs, texts, hyps, rule_pred, cache) -> np.ndarray:
    """Entailment logits for the zero-shot rows (-inf for rule-decided rows),
    checkpointed atomically to `<cache>.partial.npz` every NLI_BLOCK rows."""
    partial = cache.with_suffix(".partial.npz")
    zs_rows = [i for i, r in enumerate(rule_pred) if r is None]
    logits = np.full((len(texts), len(hyps)), -np.inf, dtype=np.float32)
    done = np.zeros(len(texts), dtype=bool)
    if partial.exists():
        z = np.load(partial)
        if z["logits"].shape == logits.shape:
            logits, done = z["logits"], z["done"]
    todo = [i for i in zs_rows if not done[i]]
    logger.info(
        "NLI: %d of %d zero-shot rows x %d hypotheses to go", len(todo), len(zs_rows), len(hyps)
    )
    for b in range(0, len(todo), NLI_BLOCK):
        rows = todo[b : b + NLI_BLOCK]
        logits[rows] = zs.entailment_logits([texts[i] for i in rows], hyps)
        done[rows] = True
        with atomic_path(partial) as tmp:
            np.savez(tmp, logits=logits, done=done)
        logger.info("NLI: %d/%d rows", b + len(rows), len(todo))
    with atomic_path(cache) as tmp:
        np.save(tmp, logits)
    partial.unlink(missing_ok=True)
    return logits


def six_class_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT_DIR / f"spam_{split}.parquet")
    return df[df["category"].notna()].reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame, per_class: int = PER_CLASS, seed: int = SEED
) -> pd.DataFrame:
    parts = [
        g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby("category")
    ]
    return pd.concat(parts).sort_values("id").reset_index(drop=True)


def predict_combo(
    logits: np.ndarray,
    hyp_index: dict,
    template: str,
    choice: dict[str, int],
    rule_pred: list[str | None],
) -> list[str]:
    """Argmax over the chosen description of each class; rule-accepted rows keep
    their rule label."""
    labels = list(choice)
    cols = [hyp_index[(template, lab, choice[lab])] for lab in labels]
    zs = np.asarray(labels)[logits[:, cols].argmax(axis=1)]
    return [r if r is not None else z for r, z in zip(rule_pred, zs, strict=True)]


def search(logits, hyp_index, templates, n_cands: dict[str, int], gold, rule_pred) -> dict:
    labels = list(n_cands)

    def score(t, ch):
        return classification_metrics(
            gold, predict_combo(logits, hyp_index, t, ch, rule_pred), labels
        )["macro_f1"]

    untuned = {lab: 0 for lab in labels}
    per_template = {}
    for t in templates:
        cur, best, trace = dict(untuned), score(t, untuned), []
        for _ in range(MAX_PASSES):
            changed = False
            for lab in labels:
                for k in range(n_cands[lab]):
                    if k == cur[lab]:
                        continue
                    trial = {**cur, lab: k}
                    s = score(t, trial)
                    if s > best:
                        cur, best, changed = trial, s, True
                        trace.append({"class": lab, "candidate": k, "macro_f1": s})
            if not changed:
                break
        per_template[t] = {"choice": cur, "macro_f1": best, "trace": trace}
    winner = max(templates, key=lambda t: (per_template[t]["macro_f1"], t == templates[0]))
    return {
        "untuned_macro_f1": score(templates[0], untuned),
        "per_template": per_template,
        "template": winner,
        "choice": per_template[winner]["choice"],
        "macro_f1": per_template[winner]["macro_f1"],
    }


def main(argv: list[str] | None = None) -> int:
    from app.interfaces import TicketText
    from baseline.models import ZeroShot
    from baseline.rules import apply_rules

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    templates, desc = load_candidates()
    sample = stratified_sample(six_class_split("val"), args.per_class)
    texts = [TicketText(id=r.id, subject=r.subject, body=r.body).text for r in sample.itertuples()]
    rules = [apply_rules(r.subject, r.body) for r in sample.itertuples()]
    rule_pred = [m.category if m is not None and m.accepted else None for m in rules]

    hyps, hyp_index = [], {}
    for t in templates:
        for lab, cands in desc.items():
            for k, d in enumerate(cands):
                hyp_index[(t, lab, k)] = len(hyps)
                hyps.append(t.format(d))
    key = hashlib.sha256(json.dumps([list(sample["id"]), hyps]).encode()).hexdigest()[:16]
    cache = CACHE / f"logits-{key}.npy"
    if cache.exists():
        logits = np.load(cache)
        logger.info("loaded cached logits %s", cache.name)
    else:
        logits = nli_logits(ZeroShot(args.device), texts, hyps, rule_pred, cache)

    gold = list(sample["category"])
    res = search(
        logits, hyp_index, templates, {lab: len(c) for lab, c in desc.items()}, gold, rule_pred
    )
    chosen = {lab: desc[lab][k] for lab, k in res["choice"].items()}
    final_pred = predict_combo(logits, hyp_index, res["template"], res["choice"], rule_pred)
    detail = classification_metrics(gold, final_pred, categories())
    untuned_pred = predict_combo(
        logits, hyp_index, templates[0], {lab: 0 for lab in desc}, rule_pred
    )
    summary = {
        "date": date.today().isoformat(),
        "sample": {
            "split": "val (6-class incl. spam)",
            "n": len(sample),
            "per_class": args.per_class,
            "seed": SEED,
            "by_class": sample["category"].value_counts().to_dict(),
            "rule_decided": sum(r is not None for r in rule_pred),
        },
        "untuned_macro_f1": res["untuned_macro_f1"],
        "tuned_macro_f1": res["macro_f1"],
        "template": res["template"],
        "descriptions": chosen,
        "per_template": {
            t: {"macro_f1": v["macro_f1"], "choice": v["choice"]}
            for t, v in res["per_template"].items()
        },
        "trace": res["per_template"][res["template"]]["trace"],
        "tuned_detail": detail,
        "untuned_detail": classification_metrics(gold, untuned_pred, categories()),
        "run": run_info(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RESULTS / "search.json", json.dumps(summary, indent=2, default=str))
    atomic_write_text(
        TUNED,
        "# SCRIBE (val-tuned descriptions), frozen by `make scribe-tune` (eval/scribe_tune.py).\n"
        "# Strengthened baseline: only the zero-shot wording differs from SCRIBE.\n"
        "# Do not edit by hand; evaluate once on test via eval.compare.\n"
        + yaml.safe_dump(
            {
                "version": 1,
                "date": summary["date"],
                "template": res["template"],
                "descriptions": chosen,
                "val_sample_n": len(sample),
                "val_sample_macro_f1": {
                    "untuned": res["untuned_macro_f1"],
                    "tuned": res["macro_f1"],
                },
            },
            sort_keys=False,
            width=200,
        ),
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("sample", "untuned_macro_f1", "tuned_macro_f1", "template")},
            indent=2,
        )
    )
    print(yaml.safe_dump(chosen, width=200))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
