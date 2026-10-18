"""Loader for the fine-tuned Hugging Face classifiers copied in from the GPU PC
(deep tier P3.2, urgency P3.3). Contract: docs/model_artifacts.md.

Validation is cheap and happens at construction (file checks, label_map,
model_card). Weights load lazily on the first predict() and stay resident
(one model per role; 8 GB RAM). On MPS or CUDA the model runs in fp16 unless
MODEL_FP16=false. Any problem, at validation or at load, makes the
artifact unavailable with a logged reason; nothing here raises to the
service.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.device import select_device, use_fp16

logger = logging.getLogger(__name__)

REQUIRED_FILES = ("config.json", "label_map.json", "model_card.json")
WEIGHT_FILES = ("model.safetensors",)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

# Checks the ordered label list; returns an error message or None.
LabelCheck = Callable[[list[str]], str | None]


class ArtifactError(ValueError):
    pass


def read_label_map(path: Path) -> list[str]:
    """label_map.json -> labels ordered by id. Ids must be 0..n-1."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ArtifactError("label_map.json must be a non-empty object {id: label}")
    try:
        ids = sorted(int(k) for k in raw)
    except ValueError as exc:
        raise ArtifactError(f"label_map.json ids must be integers: {exc}") from None
    if ids != list(range(len(raw))):
        raise ArtifactError(f"label_map.json ids must be 0..{len(raw) - 1}, got {ids}")
    labels = [raw[str(i)] for i in ids]
    if len(set(labels)) != len(labels):
        raise ArtifactError("label_map.json has duplicate labels")
    return labels


def validate_artifact(
    directory: Path, task: str, check_labels: LabelCheck
) -> tuple[list[str], dict]:
    """Labels (by id) and model card, or ArtifactError naming what is wrong."""
    if not directory.is_dir():
        raise ArtifactError(f"{directory} does not exist")
    missing = [f for f in REQUIRED_FILES if not (directory / f).exists()]
    if not any((directory / f).exists() for f in WEIGHT_FILES):
        missing.append(" or ".join(WEIGHT_FILES))
    if not any((directory / f).exists() for f in TOKENIZER_FILES):
        missing.append(" or ".join(TOKENIZER_FILES))
    if missing:
        raise ArtifactError(f"missing files in {directory}: {', '.join(missing)}")
    labels = read_label_map(directory / "label_map.json")
    problem = check_labels(labels)
    if problem:
        raise ArtifactError(f"label_map.json: {problem}")
    try:
        card = json.loads((directory / "model_card.json").read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"model_card.json is not valid JSON: {exc}") from None
    if card.get("task") != task:
        raise ArtifactError(f"model_card.json task is {card.get('task')!r}, expected {task!r}")
    max_length = card.get("max_length")
    if not isinstance(max_length, int) or max_length <= 0:
        raise ArtifactError("model_card.json needs a positive integer max_length")
    if "macro_f1" not in (card.get("val_metrics") or {}):
        raise ArtifactError("model_card.json val_metrics.macro_f1 is required")
    config = json.loads((directory / "config.json").read_text())
    n = config.get("num_labels") or len(config.get("id2label") or {})
    if n and n != len(labels):
        raise ArtifactError(f"config.json has {n} labels but label_map.json has {len(labels)}")
    return labels, card


def artifact_version(card: dict, directory: Path) -> str:
    return card.get("version") or f"{directory.name}-{card.get('trained_at', 'unknown')}"


class HFClassifier:
    """One fine-tuned AutoModelForSequenceClassification directory."""

    def __init__(
        self,
        directory: Path,
        task: str,
        check_labels: LabelCheck,
        device: str | None = None,
        fp16: bool = True,
        batch_size: int = 16,
    ):
        self.directory = Path(directory)
        self.task = task
        self.batch_size = batch_size
        self._device_override = device
        self._fp16 = fp16
        self._model = None
        self._tokenizer = None
        self.device: str | None = None
        self.labels: list[str] = []
        self.card: dict = {}
        self.reason = ""
        try:
            self.labels, self.card = validate_artifact(self.directory, task, check_labels)
        except (ArtifactError, OSError, json.JSONDecodeError) as exc:
            self._unavailable(str(exc))
            return
        if importlib.util.find_spec("torch") is None or (
            importlib.util.find_spec("transformers") is None
        ):
            self._unavailable("torch/transformers not installed (pip install -e '.[models]')")

    def _unavailable(self, reason: str) -> None:
        self.reason = reason
        logger.warning("%s model unavailable: %s", self.task, reason)

    @property
    def available(self) -> bool:
        return not self.reason

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def max_length(self) -> int:
        return int(self.card.get("max_length", 0))

    @property
    def version(self) -> str:
        return artifact_version(self.card, self.directory) if self.card else ""

    def load(self) -> bool:
        """Load weights now (idempotent). False, with a reason, on failure."""
        if self._model is not None:
            return True
        if not self.available:
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            t0 = time.perf_counter()
            self.device = select_device(self._device_override)
            tokenizer = AutoTokenizer.from_pretrained(self.directory)
            model = AutoModelForSequenceClassification.from_pretrained(self.directory)
            if model.config.num_labels != len(self.labels):
                raise ArtifactError(
                    f"model has {model.config.num_labels} outputs, label_map has {len(self.labels)}"
                )
            if use_fp16(self.device, self._fp16):
                model = model.half()
            model.to(self.device).eval()
        except Exception as exc:  # bad weights, tokenizer, device
            self._unavailable(f"failed to load {self.directory}: {exc}")
            return False
        self._model, self._tokenizer = model, tokenizer
        logger.info(
            "%s model %s loaded on %s (fp16=%s) in %.1fs",
            self.task,
            self.version,
            self.device,
            use_fp16(self.device, self._fp16),
            time.perf_counter() - t0,
        )
        return True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """(n, len(self.labels)) softmax probabilities in label_map id order.
        Raises RuntimeError if the model is unavailable (callers check
        `available` first and fall back)."""
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        if not self.load():
            raise RuntimeError(f"{self.task} model unavailable: {self.reason}")
        import torch

        out = []
        for start in range(0, len(texts), self.batch_size):
            enc = self._tokenizer(
                texts[start : start + self.batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out)
