"""Classification metrics shared by the baseline snapshot (P2.3) and the
baseline-vs-system comparison (P7.1).

Reporting rule (CLAUDE.md): macro-F1 is the headline, accuracy is secondary.
Every result is reported overall, real-only (without synthetic rows), per
origin, and without the Meeting Scheduling class.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from app.taxonomy import categories

MEETING = "Meeting Scheduling"
# Real tickets that mention a meeting/call but are labelled something else.
MEETING_WORDS = re.compile(
    r"\b(?:meeting|call|zoom|teams|webex|schedul\w*|reschedul\w*|calendar|appointment)\b", re.I
)
URGENCY_LEVELS = ["low", "medium", "high"]


def _r(x: float) -> float:
    return round(float(x), 4)


def classification_metrics(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None
) -> dict:
    """Accuracy, macro/weighted P/R/F1 and per-class P/R/F1/support over ``labels``
    (default: the taxonomy categories present in gold). Predictions outside
    ``labels`` count as errors."""
    gold, pred = list(gold), list(pred)
    if not gold:
        return {"n": 0}
    if labels is None:
        present = set(gold)
        labels = [c for c in categories() if c in present]
    labels = list(labels)
    p, r, f, n = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        gold, pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "n": len(gold),
        "accuracy": _r(accuracy_score(gold, pred)) if gold else None,
        "macro_f1": _r(mf),
        "macro_precision": _r(mp),
        "macro_recall": _r(mr),
        "weighted_f1": _r(f1_score(gold, pred, labels=labels, average="weighted", zero_division=0)),
        "labels": labels,
        "per_class": {
            lab: {
                "precision": _r(p[i]),
                "recall": _r(r[i]),
                "f1": _r(f[i]),
                "support": int(n[i]),
            }
            for i, lab in enumerate(labels)
        },
    }


def sliced_metrics(df: pd.DataFrame, gold: str = "gold", pred: str = "pred") -> dict:
    """The reporting-rule slices over a frame with gold/pred/origin columns."""
    out = {"overall": classification_metrics(df[gold], df[pred], categories())}
    for origin, g in df.groupby("origin", sort=True):
        out[f"origin_{origin}"] = classification_metrics(g[gold], g[pred])
    no_meet = df[df[gold] != MEETING]
    out["without_meeting_scheduling"] = classification_metrics(
        no_meet[gold], no_meet[pred], [c for c in categories() if c != MEETING]
    )
    mention = df[
        (df["origin"] == "real") & (df[gold] != MEETING) & df["text"].str.contains(MEETING_WORDS)
    ]
    out["real_meeting_mentions"] = {
        "n": len(mention),
        "accuracy": _r((mention[gold] == mention[pred]).mean()) if len(mention) else None,
        "predicted_meeting": int((mention[pred] == MEETING).sum()),
        "gold_counts": mention[gold].value_counts().to_dict(),
        "pattern": MEETING_WORDS.pattern,
    }
    return out


def confusion(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    return confusion_matrix(list(gold), list(pred), labels=list(labels)).tolist()


def urgency_metrics(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """Low/medium/high only (rows with no gold urgency must be dropped first)."""
    gold, pred = list(gold), list(pred)
    idx = {lvl: i for i, lvl in enumerate(URGENCY_LEVELS)}
    m = classification_metrics(gold, pred, URGENCY_LEVELS)
    if not gold:
        return m
    m["ordinal_mae"] = _r(
        mean_absolute_error([idx[g] for g in gold], [idx.get(p, 1) for p in pred])
    )
    return m


def latency_summary(ms: Sequence[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean_ms": _r(a.mean()),
        "p50_ms": _r(np.percentile(a, 50)),
        "p95_ms": _r(np.percentile(a, 95)),
        "max_ms": _r(a.max()),
    }


def plot_confusion(cm: list[list[int]], labels: Sequence[str], path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray(cm)
    rows = a.sum(axis=1, keepdims=True)
    norm = np.divide(a, rows, out=np.zeros_like(a, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(
                j,
                i,
                f"{a[i, j]}\n{norm[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold (row-normalised colour)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
