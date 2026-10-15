"""P2.3: run the SCRIBE baseline on a split and write results/baseline/.

    python -m baseline evaluate --split test        # == make baseline-eval (~45 min on M1)

Outputs (committed):
- metrics.json: the reporting-rule slices (overall / per origin / without
  Meeting Scheduling / real tickets mentioning meetings), rule-handled share
  and per-stage accuracy, sentiment-urgency metrics, latency, run metadata;
- confusion_matrix.png (overall, 6x6);
- latency.md: model load, throughput (batched) and single-ticket latency;
- predictions.jsonl: one row per ticket, for P7.1 comparisons without re-running.

The run is resumable: predictions are appended to predictions.partial.jsonl as
batches finish, and finished ids are skipped on restart.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd

from app.interfaces import TicketText
from app.settings import REPO_ROOT
from app.taxonomy import categories
from baseline import models
from baseline.rules import CUTOFF
from baseline.scribe import ScribeClassifier
from data.build import OUT_DIR
from data.stress_meetings import load_reviewed
from eval.classification import (
    MEETING,
    classification_metrics,
    confusion,
    latency_summary,
    plot_confusion,
    sliced_metrics,
    urgency_metrics,
)

logger = logging.getLogger(__name__)

RESULTS = REPO_ROOT / "results" / "baseline"
STAGES = ("rules", "zero_shot", "sentiment")


def _tickets(df: pd.DataFrame) -> list[TicketText]:
    return [
        TicketText(id=r.id, subject=r.subject or "", body=r.body or "")
        for r in df.itertuples(index=False)
    ]


def _timed(clf: ScribeClassifier, tickets: list[TicketText]):
    before = dict(clf.stats.seconds)
    t0 = time.perf_counter()
    preds = clf.classify(tickets)
    total = time.perf_counter() - t0
    stage = {k: clf.stats.seconds[k] - before[k] for k in STAGES}
    return preds, total, stage


def run_predictions(
    clf: ScribeClassifier, df: pd.DataFrame, partial: Path, batch_size: int
) -> pd.DataFrame:
    done: dict[str, dict] = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
        logger.info("resuming: %d predictions already in %s", len(done), partial.name)
    todo = df[~df["id"].isin(done)]
    with open(partial, "a") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo.iloc[start : start + batch_size]
            preds, total, stage = _timed(clf, _tickets(batch))
            n = len(batch)
            for row, p in zip(batch.itertuples(index=False), preds, strict=True):
                rec = {
                    "id": row.id,
                    "pred": p.category,
                    "confidence": round(p.confidence, 6),
                    "stage": p.stage,
                    "pred_urgency": p.urgency,
                    "sentiment": int(p.signals["sentiment"]),
                    "rules": [k[5:] for k in p.signals if k.startswith("rule:")],
                    "batch_ms_per_ticket": round(total * 1000 / n, 3),
                    **{
                        f"batch_{k}_ms_per_ticket": round(v * 1000 / n, 3) for k, v in stage.items()
                    },
                }
                f.write(json.dumps(rec) + "\n")
                done[row.id] = rec
            f.flush()
            if (start // batch_size) % 10 == 0:
                logger.info("%d/%d", len(done), len(df))
    return pd.DataFrame([done[i] for i in df["id"]])


def single_ticket_latency(clf: ScribeClassifier, df: pd.DataFrame, n: int, seed: int) -> dict:
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    rows = []
    for t in _tickets(sample):
        preds, total, stage = _timed(clf, [t])
        rows.append(
            {
                "total": total * 1000,
                "stage": preds[0].stage,
                **{k: v * 1000 for k, v in stage.items()},
            }
        )
    lat = pd.DataFrame(rows)
    return {
        "end_to_end": latency_summary(lat["total"]),
        "end_to_end_rule_decided": latency_summary(lat.loc[lat["stage"] == "rule", "total"]),
        "end_to_end_zero_shot": latency_summary(lat.loc[lat["stage"] == "zero_shot", "total"]),
        **{k: latency_summary(lat[k]) for k in STAGES},
    }


def _swap_usage() -> str:
    """macOS swap in use when the run ends (latency is sensitive to memory pressure)."""
    try:
        return subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip()
    except OSError:
        return ""


def latency_markdown(m: dict) -> str:
    lat = m["latency"]
    lines = [
        "# SCRIBE baseline latency",
        "",
        f"Device `{m['run']['device']}`, {m['run']['machine']} (M1, 8 GB RAM). Generated by "
        "`make baseline-eval`.",
        f"Swap in use at the end of the run: {m['run']['swap_usage_at_end']}. Latency on this "
        "machine is sensitive to memory pressure; outliers in the single-ticket table are "
        "likely swapping.",
        "",
        f"- Model load (BART-MNLI + twitter-roberta, from local cache): {lat['model_load_s']} s",
        f"- Batched run (batch {m['run']['batch_size']}): {lat['batched']['tickets']} tickets in "
        f"{lat['batched']['wall_s']} s = {lat['batched']['tickets_per_s']} tickets/s",
        "",
        "## Batched, amortised per ticket (ms)",
        "",
        "| stage | mean | p50 | p95 | max |",
        "|---|---|---|---|---|",
    ]
    for k, v in lat["batched"]["per_ticket_ms"].items():
        lines.append(f"| {k} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} | {v['max_ms']} |")
    s = lat["single_ticket"]
    lines += [
        "",
        f"## Single ticket at a time ({s['end_to_end']['n']} random test tickets, ms)",
        "",
        "| measure | n | mean | p50 | p95 | max |",
        "|---|---|---|---|---|---|",
    ]
    for k in ("end_to_end", "end_to_end_rule_decided", "end_to_end_zero_shot", *STAGES):
        v = s[k]
        if v["n"]:
            cells = [v["n"], v["mean_ms"], v["p50_ms"], v["p95_ms"], v["max_ms"]]
            lines.append(f"| {k} | " + " | ".join(map(str, cells)) + " |")
    lines.append("")
    return "\n".join(lines)


def evaluate(
    split: str = "test",
    limit: int | None = None,
    batch_size: int = 4,
    latency_n: int = 100,
    out: Path = RESULTS,
    device: str | None = None,
    seed: int = 42,
) -> dict:
    df = pd.read_parquet(OUT_DIR / f"{split}.parquet")
    if limit:  # seeded random sample, for smoke runs
        df = df.sample(n=min(limit, len(df)), random_state=seed).sort_values("id")
    manifest = json.loads((OUT_DIR / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)

    clf = ScribeClassifier(device=device)
    t0 = time.perf_counter()  # load both models outside the timed run
    clf.classify([TicketText(id="warmup", subject="Hello", body="A short warm-up message.")])
    load_s = time.perf_counter() - t0
    clf.stats = type(clf.stats)()

    t0 = time.perf_counter()
    preds = run_predictions(clf, df, out / "predictions.partial.jsonl", batch_size)
    wall = time.perf_counter() - t0
    res = df[["id", "origin", "category", "urgency", "text"]].merge(preds, on="id")
    res = res.rename(columns={"category": "gold"})

    m: dict = {"classification": sliced_metrics(res)}
    labels = categories()
    cm = confusion(res["gold"], res["pred"], labels)
    m["confusion_matrix"] = {"labels": labels, "matrix": cm}
    plot_confusion(
        cm,
        labels,
        out / "confusion_matrix.png",
        f"SCRIBE baseline, {split} split (n={len(res)}): macro-F1 "
        f"{m['classification']['overall']['macro_f1']:.3f}",
    )
    stage = {}
    for name, g in res.groupby("stage"):
        stage[name] = {
            "n": len(g),
            "share": round(len(g) / len(res), 4),
            "accuracy": round(float((g["gold"] == g["pred"]).mean()), 4),
            "predicted": g["pred"].value_counts().to_dict(),
        }
    rule_rows = res[res["stage"] == "rule"]
    m["stages"] = {
        "rule_cutoff": CUTOFF,
        "rule_handled_share": round(len(rule_rows) / len(res), 4),
        "by_stage": stage,
        "rule_precision_by_category": {
            c: round(float((g["gold"] == c).mean()), 4) for c, g in rule_rows.groupby("pred")
        },
    }
    labelled = res[res["urgency"].notna()]
    m["urgency_from_sentiment"] = {
        "note": "SCRIBE has no urgency model; sentiment -1/0/+1 maps to high/medium/low",
        **urgency_metrics(labelled["urgency"], labelled["pred_urgency"]),
    }
    per_ticket = {
        "total": latency_summary(preds["batch_ms_per_ticket"]),
        **{k: latency_summary(preds[f"batch_{k}_ms_per_ticket"]) for k in STAGES},
    }
    m["latency"] = {
        "model_load_s": round(load_s, 2),
        "batched": {
            "tickets": len(df),
            "wall_s": round(wall, 1),
            "tickets_per_s": round(len(df) / wall, 3) if wall else None,
            "per_ticket_ms": per_ticket,
            "note": "per-ticket values are batch time / batch size; on a resumed run, "
            "wall_s covers only this session",
        },
        "single_ticket": single_ticket_latency(clf, df, latency_n, seed),
    }
    stress = load_reviewed()
    if stress is None:
        m["stress_set"] = {
            "status": "pending: datasets/stress/meeting_stress.csv not fully reviewed"
        }
    else:
        sdf = pd.DataFrame(stress)
        sp = clf.classify(_tickets(sdf))
        sdf["pred"] = [p.category for p in sp]
        m["stress_set"] = {
            "status": "reviewed",
            "overall": classification_metrics(sdf["category"], sdf["pred"]),
            "by_stress_type": {
                t: {
                    "n": len(g),
                    "accuracy": round(float((g["category"] == g["pred"]).mean()), 4),
                    "predicted_meeting": int((g["pred"] == MEETING).sum()),
                }
                for t, g in sdf.groupby("stress_type")
            },
        }
    import torch
    import transformers

    m["run"] = {
        "system": "scribe-baseline",
        "split": split,
        "rows": len(res),
        "split_content_sha256": manifest["splits"][split]["content_sha256"],
        "limit": limit,
        "batch_size": batch_size,
        "device": clf.zero_shot._pipe.device.type,
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "models": {"zero_shot": models.ZERO_SHOT_MODEL, "sentiment": models.SENTIMENT_MODEL},
        "hypothesis_template": models.HYPOTHESIS_TEMPLATE,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_commit": _git_commit(),
        "swap_usage_at_end": _swap_usage(),
        "date": time.strftime("%Y-%m-%d"),
    }
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    (out / "latency.md").write_text(latency_markdown(m))
    keep = ["id", "origin", "gold", "pred", "confidence", "stage", "rules", "urgency",
            "pred_urgency", "sentiment"]  # fmt: skip
    with open(out / "predictions.jsonl", "w") as f:
        for r in res[keep].to_dict(orient="records"):
            r["urgency"] = r["urgency"] if isinstance(r["urgency"], str) else None
            f.write(json.dumps(r) + "\n")
    (out / "predictions.partial.jsonl").unlink(missing_ok=True)
    return m
