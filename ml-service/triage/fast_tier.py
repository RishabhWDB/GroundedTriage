"""Fast tier (P3.1): TF-IDF (word 1-2 grams + char 3-5 grams) + logistic
regression with class_weight="balanced", probability-calibrated with
CalibratedClassifierCV (5-fold, one final model; sigmoid or isotonic, picked
on val ECE by the training script).

Trained in this repo by triage.train_fast_tier; this module builds the
pipeline and loads/serves the saved artifact:

    models/fast_tier/
    ├── model.joblib       the fitted CalibratedClassifierCV
    └── model_card.json    version stamp, training data hash, C, val metrics

The model input is app.textinput.model_input(subject, body), the same `text`
column the splits carry. joblib artifacts are pickles: only load ones this
repo trained.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.taxonomy import categories

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"
CALIBRATION_FOLDS = 5


def build_pipeline(c: float = 4.0, max_iter: int = 2000):
    """The uncalibrated feature + LR pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=300_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter)
    return Pipeline([("features", features), ("lr", lr)])


def build_calibrated(c: float = 4.0, folds: int = CALIBRATION_FOLDS, method: str = "isotonic"):
    """The pipeline wrapped in probability calibration (sigmoid or isotonic).
    ensemble=False keeps one final model (fitted on all of train) plus calibrators fitted on
    cross-validated predictions, so the artifact holds one vocabulary."""
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(build_pipeline(c), method=method, cv=folds, ensemble=False)


@dataclass
class TierPrediction:
    category: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)


class FastTier:
    """A loaded fast-tier model. Use FastTier.load(); it never raises on a bad
    artifact, it returns None and logs why."""

    def __init__(self, model, card: dict):
        self.model = model
        self.card = card
        self.classes: list[str] = [str(c) for c in model.classes_]

    @property
    def version(self) -> str:
        return self.card.get("version", "unknown")

    @classmethod
    def load(cls, directory: Path) -> FastTier | None:
        directory = Path(directory)
        missing = [f for f in (MODEL_FILE, CARD_FILE) if not (directory / f).exists()]
        if missing:
            logger.warning("fast tier unavailable: %s missing in %s", missing, directory)
            return None
        try:
            import joblib
            import sklearn

            card = json.loads((directory / CARD_FILE).read_text())
            t0 = time.perf_counter()
            model = joblib.load(directory / MODEL_FILE)
        except Exception as exc:  # corrupt pickle, version skew, bad JSON
            logger.warning("fast tier unavailable: cannot load %s: %s", directory, exc)
            return None
        if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
            logger.warning(
                "fast tier unavailable: %s is not a probabilistic classifier", MODEL_FILE
            )
            return None
        classes = {str(c) for c in model.classes_}
        if classes != set(categories()):
            logger.warning(
                "fast tier unavailable: classes %s do not match the taxonomy %s",
                sorted(classes),
                categories(),
            )
            return None
        trained_with = card.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "fast tier trained with scikit-learn %s, running %s; retrain if results look off",
                trained_with,
                sklearn.__version__,
            )
        logger.info(
            "fast tier %s loaded from %s in %.2fs",
            card.get("version"),
            directory,
            time.perf_counter() - t0,
        )
        return cls(model, card)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """(n, len(self.classes)) calibrated probabilities."""
        return self.model.predict_proba(texts)

    def predict(self, texts: list[str]) -> list[TierPrediction]:
        if not texts:
            return []
        probs = self.predict_proba(texts)
        out = []
        for row in probs:
            i = int(np.argmax(row))
            out.append(
                TierPrediction(
                    category=self.classes[i],
                    confidence=float(row[i]),
                    probs={c: float(p) for c, p in zip(self.classes, row, strict=True)},
                )
            )
        return out
