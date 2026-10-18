"""Cross-encoder reranker (P5.6, optional): rerank the dense top-N chunks with
cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters) and keep the top-k.

The reranker only decides the order. Each returned chunk keeps its cosine
score, so the faithfulness gate's retrieval floor means the same thing with
or without reranking. Weights load lazily on first use (fp16 on MPS/CUDA, fp32 on
CPU); inputs are (query, chunk text) pairs truncated to 512 tokens.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from app.device import select_device, use_fp16
from app.settings import get_settings

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = 512, batch_size: int = 32,
                 device: str | None = None, fp16: bool | None = None):  # fmt: skip
        self.model_name, self.max_length, self.batch_size = model_name, max_length, batch_size
        self._device_override = device
        self._fp16 = get_settings().model_fp16 if fp16 is None else fp16
        self._model = None
        self._tokenizer = None
        self.device: str | None = None
        self.load_s: float | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.perf_counter()
        self.device = select_device(self._device_override)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if use_fp16(self.device, self._fp16):
            model = model.half()
        self._model = model.to(self.device).eval()
        self.load_s = time.perf_counter() - t0
        logger.info("reranker %s on %s loaded in %.1fs", self.model_name, self.device, self.load_s)

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance logits, higher = more relevant."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        self.load()
        import torch

        out = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            enc = self._tokenizer([query] * len(batch), batch, truncation=True,
                                  max_length=self.max_length, padding=True,
                                  return_tensors="pt").to(self.device)  # fmt: skip
            with torch.inference_mode():
                logits = self._model(**enc).logits.float().reshape(-1)
            out.append(logits.cpu().numpy())
        return np.concatenate(out)

    def rerank(self, query: str, passages: list[str]) -> list[int]:
        """Indices of passages, best first (stable for ties)."""
        s = self.scores(query, passages)
        return sorted(range(len(passages)), key=lambda i: (-s[i], i))
