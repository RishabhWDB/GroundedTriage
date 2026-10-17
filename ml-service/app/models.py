"""The ml-service's model registry: built once, lazily.

- Deep tier and urgency artifacts are validated (file checks only) the first
  time anything asks; their weights load on the first prediction that needs
  them, then stay resident (one transformer per role; 8 GB RAM).
- The fast tier and the spam filter load on the first /classify (the
  filter with the SPAM_RULE operating point; absent filter = no Spam output).
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
from triage.spam_filter import SpamFilter
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
        self._spam: SpamFilter | None = None
        self._spam_tried = False
        self._llm = None
        self._judge = None
        self._embedder = None
        self._reranker = None
        self._db = None
        # One psycopg connection, used under this lock: FastAPI runs sync handlers in threads.
        self.db_lock = threading.Lock()

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

    def spam(self) -> SpamFilter | None:
        with self._lock:
            if not self._spam_tried:
                self._spam_tried = True
                self._spam = SpamFilter.load(
                    self.settings.models_dir / "spam_filter", self.settings.spam_rule
                )
            return self._spam

    def embedder(self):
        """bge-small (max_length 512), shared by retrieval; loads weights on first encode."""
        with self._lock:
            if self._embedder is None:
                from rag.kb import KB_MAX_LENGTH
                from triage.embedder import Embedder

                self._embedder = Embedder(max_length=KB_MAX_LENGTH)
            return self._embedder

    def retriever(self):
        """A Retriever over DATABASE_URL, or None when it is unset. Use under db_lock."""
        with self._lock:
            if self._db is None or self._db.closed:
                if not self.settings.database_url:
                    return None
                from rag.kb import connect

                self._db = connect(self.settings.database_url)
        from rag.retriever import Retriever

        return Retriever(
            self._db, self.embedder().encode, reranker=self.reranker(), reconnect=self._reconnect
        )

    def _reconnect(self):
        """A fresh DB connection after the old one died (Retriever retries once)."""
        from rag.kb import connect

        with self._lock:
            self._db = connect(self.settings.database_url)
            return self._db

    def reranker(self):
        """The cross-encoder when RETRIEVAL_RERANK is on (lazy, one instance), else None."""
        if not self.settings.retrieval_rerank:
            return None
        with self._lock:
            if self._reranker is None:
                from rag.reranker import CrossEncoderReranker

                self._reranker = CrossEncoderReranker()
            return self._reranker

    def llm(self):
        """The drafter LLM (Groq), built on first use; None without GROQ_API_KEY."""
        with self._lock:
            if self._llm is None and self.settings.groq_api_key:
                from app.llm import build_llm

                self._llm = build_llm(self.settings, "draft")
            return self._llm

    def judge(self):
        """The faithfulness judge LLM (JUDGE_MODEL), built on first use; None without a key."""
        with self._lock:
            key = self.settings.judge_api_key or self.settings.groq_api_key
            if self._judge is None and key:
                from app.llm import build_llm

                self._judge = build_llm(self.settings, "judge")
            return self._judge

    def cascade(self) -> Cascade | None:
        fast = self.fast()
        if fast is None:
            return None
        return Cascade(
            fast, self.deep, UrgencyPredictor(self.urgency), load_tau(), spam=self.spam()
        )


@lru_cache
def get_registry() -> Registry:
    return Registry(get_settings())
