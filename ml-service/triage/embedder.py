"""Sentence embeddings with BAAI/bge-small-en-v1.5 (384-d), shared by the
embedding fast tier (P3.13) and, later, KB retrieval (P5).

bge recipe: CLS-token pooling, L2-normalised. Inputs are truncated to
max_length tokens. Weights load lazily on the first encode() (one model per
process; 8 GB RAM) and run in fp16 on MPS (MODEL_FP16), fp32 on CPU.

EmbeddingCache stores vectors keyed by sha256(model | max_length | text) in
one .npz per model, so the training/eval texts are embedded once.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from app.device import select_device
from app.fileio import atomic_path
from app.settings import REPO_ROOT, get_settings

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
MAX_LENGTH = 256
CACHE_DIR = REPO_ROOT / "datasets" / "raw" / "emb_cache"


class Embedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
        device: str | None = None,
        fp16: bool | None = None,
        batch_size: int = 32,
    ):
        self.model_name, self.max_length, self.batch_size = model_name, max_length, batch_size
        self._device_override = device
        self._fp16 = get_settings().model_fp16 if fp16 is None else fp16
        self._model = None
        self._tokenizer = None
        self.device: str | None = None
        self.load_s: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        t0 = time.perf_counter()
        self.device = select_device(self._device_override)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        if self.device == "mps" and self._fp16:
            model = model.half()
        self._model = model.to(self.device).eval()
        self.load_s = time.perf_counter() - t0
        logger.info("embedder %s on %s loaded in %.1fs", self.model_name, self.device, self.load_s)

    def encode(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised."""
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        self.load()
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
                cls = self._model(**enc).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class EmbeddingCache:
    def __init__(self, embedder: Embedder, root: Path | None = None):
        self.embedder = embedder
        root = root or CACHE_DIR  # resolved at call time (tests point it elsewhere)
        safe = embedder.model_name.replace("/", "__")
        self.path = root / f"{safe}__{embedder.max_length}.npz"
        self._vecs: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path)
            self._vecs = dict(zip(data["keys"].tolist(), data["vecs"], strict=True))

    def key(self, text: str) -> str:
        e = self.embedder
        return hashlib.sha256(f"{e.model_name}|{e.max_length}|{text}".encode()).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [self.key(t) for t in texts]
        missing = sorted(
            {k: t for k, t in zip(keys, texts, strict=True) if k not in self._vecs}.items()
        )
        if missing:
            logger.info("embedding %d new texts (%d cached)", len(missing), len(self._vecs))
            vecs = self.embedder.encode([t for _, t in missing])
            for (k, _), v in zip(missing, vecs, strict=True):
                self._vecs[k] = v
            self.save()
        if not keys:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([self._vecs[k] for k in keys])

    def save(self) -> None:
        keys = list(self._vecs)
        with atomic_path(self.path) as tmp:
            np.savez(tmp, keys=np.array(keys), vecs=np.stack([self._vecs[k] for k in keys]))
