"""Retriever (P5.2): query = subject + cleaned body (app.textinput.model_input).

- dense (default): bge-small query embedding (with bge's query instruction)
  -> pgvector cosine top-k over the HNSW index on kb_chunks.embedding.
- hybrid (flag): reciprocal-rank fusion (RRF, k=60) of the dense top-N and a
  keyword top-N from Postgres full-text search (`kb_chunks.tsv`, ts_rank_cd).
  ts_rank_cd is a BM25-like ranking, not exact BM25.

Scores: dense = cosine similarity (1 - cosine distance), which the
faithfulness gate (P5.4) compares to its retrieval floor. Hybrid results keep
each chunk's cosine score (computed for keyword-only hits too) so the floor
means the same thing in both modes; RRF only decides the order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.schemas import RetrievedChunk
from rag.kb import vector_literal

logger = logging.getLogger(__name__)

# bge-*-en-v1.5: prefix short retrieval queries with this instruction.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RRF_K = 60
CANDIDATES = 50  # per ranking, before fusion

DENSE_SQL = """
select id, doc_id, title, heading, text, 1 - (embedding <=> %(q)s::vector) as score
from kb_chunks
order by embedding <=> %(q)s::vector
limit %(n)s"""

KEYWORD_SQL = """
select id, ts_rank_cd(tsv, query) as rank
from kb_chunks, websearch_to_tsquery('english', %(text)s) query
where tsv @@ query
order by rank desc
limit %(n)s"""

BY_IDS_SQL = """
select id, doc_id, title, heading, text, 1 - (embedding <=> %(q)s::vector) as score
from kb_chunks where id = any(%(ids)s)"""


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    mode: str  # dense | hybrid

    @property
    def max_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion of several ranked id lists (ties: first seen)."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda c: -score[c])


class Retriever:
    def __init__(self, conn, encode_query, ef_search: int = 100):
        """conn: a psycopg connection (search_path incl. extensions);
        encode_query: list[str] -> (n, 384) normalised vectors."""
        self.conn = conn
        self.encode_query = encode_query
        self.ef_search = ef_search

    def _embed(self, query: str) -> str:
        return vector_literal(self.encode_query([QUERY_INSTRUCTION + query])[0])

    def search(self, query: str, k: int = 5, hybrid: bool = False) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult([], "hybrid" if hybrid else "dense")
        q = self._embed(query)
        with self.conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")
            n = CANDIDATES if hybrid else k
            cur.execute(DENSE_SQL, {"q": q, "n": n})
            dense = {r[0]: r for r in cur.fetchall()}
            if not hybrid:
                rows = list(dense.values())[:k]
                self.conn.commit()
                return RetrievalResult([self._chunk(r) for r in rows], "dense")
            cur.execute(KEYWORD_SQL, {"text": query, "n": CANDIDATES})
            keyword = [r[0] for r in cur.fetchall()]
            fused = rrf([list(dense), keyword])[:k]
            missing = [cid for cid in fused if cid not in dense]
            if missing:
                cur.execute(BY_IDS_SQL, {"q": q, "ids": missing})
                dense.update({r[0]: r for r in cur.fetchall()})
        self.conn.commit()
        return RetrievalResult([self._chunk(dense[cid]) for cid in fused if cid in dense], "hybrid")

    @staticmethod
    def _chunk(row) -> RetrievedChunk:
        cid, doc_id, title, heading, text, score = row
        return RetrievedChunk(chunk_id=cid, doc_id=doc_id, title=title or "", text=text,
                              score=round(float(score), 6))  # fmt: skip
