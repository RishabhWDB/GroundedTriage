"""The ml-service's model registry: built once, lazily.

- Deep tier and urgency artifacts are validated (file checks only) the first
  time anything asks; their weights load on the first prediction that needs
  them, then stay resident (one transformer per role; 8 GB RAM).
- The fast tier loads on the first /classify.
- The SCRIBE baseline models are never loaded in this process.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

from app.settings import Settings, get_settings
from triage.cascade import Cascade, load_tau
from triage.deep_tier import DeepTier
from triage.fast_tier import FastTier
from triage.urgency import UrgencyModel, UrgencyPredictor

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._deep: DeepTier | None = None
        self._urgency: UrgencyModel | None = None
        self._fast: FastTier | None = None
        self._fast_tried = False
        self.fast_reason = ""

    @property
    def deep(self) -> DeepTier:
        with self._lock:
            if self._deep is None:
                self._deep = DeepTier(self.settings.models_dir / "deep_tier")
            return self._deep

    @property
    def urgency(self) -> UrgencyModel:
        with self._lock:
            if self._urgency is None:
                self._urgency = UrgencyModel(self.settings.models_dir / "urgency")
            return self._urgency

    @property
    def fast_loaded(self) -> bool:
        return self._fast is not None

    def fast(self) -> FastTier | None:
        with self._lock:
            if not self._fast_tried:
                self._fast_tried = True
                self._fast = FastTier.load(self.settings.models_dir / "fast_tier")
                if self._fast is None:
                    self.fast_reason = "models/fast_tier/ missing or invalid: run `make fast-tier`"
            return self._fast

    def cascade(self) -> Cascade | None:
        fast = self.fast()
        if fast is None:
            return None
        return Cascade(fast, self.deep, UrgencyPredictor(self.urgency), load_tau())


@lru_cache
def get_registry() -> Registry:
    return Registry(get_settings())
