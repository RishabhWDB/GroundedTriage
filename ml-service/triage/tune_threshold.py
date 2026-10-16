"""Pick the cascade threshold tau on val (P3.4).

    python -m triage.tune_threshold      # == make tune-tau

Needs both tiers: the fast tier (models/fast_tier/) and the deep tier
(models/deep_tier/, fine-tuned on the GPU PC). Without the deep tier it exits
cleanly with a message: there is nothing to trade off, and the curve is never
faked.

For each tau in 0.00..1.00 (step 0.01): tickets with fast confidence < tau go
to the deep tier. Records the deep-tier share, macro-F1 and accuracy.
Selection rule: **the smallest deep-tier share whose val macro-F1 is within
0.5 pt (0.005) of the best macro-F1 over the sweep**. Writes
results/cascade/{tau_sweep.json, cost_curve.png} and tau to
config/cascade.yaml. Test is never read.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.settings import REPO_ROOT
from data.build import OUT_DIR
from eval.classification import classification_metrics
from eval.runinfo import run_info
from triage.cascade import CONFIG_PATH

logger = logging.getLogger(__name__)

RESULTS = REPO_ROOT / "results" / "cascade"
TOLERANCE = 0.005
TAUS = np.round(np.arange(0, 1.0001, 0.01), 2)


def sweep(gold, fast_pred, fast_conf, deep_pred, taus=TAUS) -> list[dict]:
    gold, fast_pred, deep_pred = (np.asarray(x, dtype=object) for x in (gold, fast_pred, deep_pred))
    fast_conf = np.asarray(fast_conf, dtype=float)
    rows = []
    for tau in taus:
        to_deep = fast_conf < tau
        pred = np.where(to_deep, deep_pred, fast_pred)
        m = classification_metrics(gold, pred)  # classes present in gold
        rows.append(
            {
                "tau": float(tau),
                "deep_share": round(float(to_deep.mean()), 4),
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
            }
        )
    return rows


def pick_tau(rows: list[dict], tolerance: float = TOLERANCE) -> dict:
    """Smallest deep share within `tolerance` of the best macro-F1 (ties: smaller tau)."""
    best = max(r["macro_f1"] for r in rows)
    ok = [r for r in rows if r["macro_f1"] >= best - tolerance - 1e-12]
    choice = min(ok, key=lambda r: (r["deep_share"], r["tau"]))
    return {**choice, "best_macro_f1": best, "tolerance": tolerance}


def plot_cost_curve(rows: list[dict], choice: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    share = [r["deep_share"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.plot(share, [r["macro_f1"] for r in rows], label="macro-F1")
    ax.plot(share, [r["accuracy"] for r in rows], label="accuracy", alpha=0.7)
    ax.axvline(choice["deep_share"], color="grey", ls="--", lw=1)
    ax.annotate(
        f"tau={choice['tau']:.2f}",
        (choice["deep_share"], choice["macro_f1"]),
        textcoords="offset points",
        xytext=(6, -14),
        fontsize=8,
    )
    ax.set_xlabel("share of val tickets sent to the deep tier")
    ax.set_ylabel("val score")
    ax.set_title("Cascade cost curve (val)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_tau(choice: dict, versions: dict, path: Path = CONFIG_PATH) -> None:
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg.update(
        {
            "tau": choice["tau"],
            "tuned": True,
            "tuned_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rule": f"smallest deep-tier share within {choice['tolerance']} of best val macro-F1",
            "val": {k: choice[k] for k in ("deep_share", "macro_f1", "accuracy", "best_macro_f1")},
            "models": versions,
        }
    )
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def tune(fast, deep, out: Path = RESULTS, data_dir: Path = OUT_DIR, config: Path = CONFIG_PATH):
    """Returns the chosen row, or None when the deep tier is unavailable."""
    if deep is None or not deep.available:
        reason = getattr(deep, "reason", "no deep tier")
        print(
            "tau tuning needs the deep tier in models/deep_tier/ (fine-tuned on the GPU PC; "
            f"see docs/model_artifacts.md). Not available: {reason}. Nothing was changed."
        )
        return None
    val = pd.read_parquet(data_dir / "val.parquet")
    texts = list(val["text"])
    f = fast.predict(texts)
    d = deep.predict(texts)
    rows = sweep(
        val["category"], [p.category for p in f], [p.confidence for p in f], [p.category for p in d]
    )
    choice = pick_tau(rows)
    versions = {"fast_tier": fast.version, "deep_tier": deep.version}
    out.mkdir(parents=True, exist_ok=True)
    (out / "tau_sweep.json").write_text(
        json.dumps(
            {
                "choice": choice,
                "models": versions,
                "rows": rows,
                "run": run_info(split="val", rows=len(val)),
            },
            indent=2,
        )
    )
    plot_cost_curve(rows, choice, out / "cost_curve.png")
    write_tau(choice, versions, config)
    return choice


def main(argv: list[str] | None = None) -> int:
    from triage.deep_tier import DeepTier
    from triage.fast_tier import FastTier

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from app.settings import get_settings

    fast = FastTier.load(get_settings().models_dir / "fast_tier")
    if fast is None:
        print("tau tuning needs the fast tier: run `make fast-tier` first.")
        return 1
    choice = tune(fast, DeepTier())
    if choice:
        print(
            f"tau={choice['tau']:.2f}: deep share {choice['deep_share']}, val macro-F1 "
            f"{choice['macro_f1']} (best {choice['best_macro_f1']}); written to {CONFIG_PATH}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
