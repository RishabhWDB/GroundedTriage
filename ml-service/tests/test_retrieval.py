import os
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Registry, get_registry
from app.settings import Settings
from eval.retrieval import aggregate, doc_ranking, score_question
from rag import kb
from rag.chunking import Section
from rag.retriever import QUERY_INSTRUCTION, Retriever, rrf


def test_rrf_and_scoring_helpers():
    assert rrf([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]
    assert doc_ranking(["d1", "d1", "d2", "d1", "d3"]) == ["d1", "d2", "d3"]
    s = score_question(["x", "gold", "y"], {"gold"})
    assert (s["recall@1"], s["recall@3"], s["recall@5"], s["rr"]) == (0.0, 1.0, 1.0, 0.5)
    miss = score_question(["x"], {"gold"})
    assert miss["rr"] == 0.0 and miss["recall@5"] == 0.0
    agg = aggregate([s, miss])
    assert agg == {"n": 2, "recall@1": 0.0, "recall@3": 0.5, "recall@5": 0.5, "mrr": 0.25}


def test_retrieve_endpoint_without_db(tmp_path):
    reg = Registry(Settings(models_dir=tmp_path, database_url=""))
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        r = TestClient(app).post("/retrieve", json={"subject": "refund", "body": "how?"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503


# Deterministic "embeddings": one dimension per topic word, so cosine is predictable.
TOPICS = ["refund", "password", "domain", "invoice", "calendar"]


def topic_encode(texts):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.removeprefix(QUERY_INSTRUCTION).lower()
        for j, w in enumerate(TOPICS):
            out[i, j] = t.count(w)
        out[i, 383] = 0.1  # never all-zero
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def kb_schema():
    conn = kb.connect(os.environ["TEST_DATABASE_URL"])
    schema = f"test_ret_{uuid.uuid4().hex[:8]}"
    root = kb.REPO_ROOT / "migrations"
    with conn.cursor() as cur:
        cur.execute(f'create schema "{schema}"')
        cur.execute(f'SET search_path TO "{schema}", public, extensions')
        for name in ("00002_core_tables.sql", "00008_kb_chunks_fts.sql"):
            up = (root / name).read_text().split("-- +goose Down")[0].replace("-- +goose Up", "")
            cur.execute(up)
    conn.commit()
    spec = [
        ("d-refund", "Refunds", "Refund policy", "How to get a refund for an invoice."),
        ("d-pass", "Passwords", "Reset", "Reset your password from the login page."),
        ("d-domain", "Domains", "Connect", "Connect a domain to your site. Zebracorn setup."),
    ]
    docs = [kb.KBDocument(i, "t", title, "u", [Section(h, [p])]) for i, title, h, p in spec]
    kb.ingest(docs, conn, topic_encode)
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f'drop schema "{schema}" cascade')
    conn.commit()
    conn.close()


@pytest.mark.integration
def test_dense_and_hybrid_search(kb_schema):
    r = Retriever(kb_schema, topic_encode)
    dense = r.search("I need a refund please", k=2)
    assert dense.mode == "dense" and [c.doc_id for c in dense.chunks][0] == "d-refund"
    assert (
        dense.chunks[0].score > dense.chunks[1].score and dense.max_score == dense.chunks[0].score
    )
    assert dense.chunks[0].chunk_id == "d-refund:0" and "Refund policy" in dense.chunks[0].text
    # "Zebracorn" has no embedding signal; only the keyword side finds it.
    kw = r.search("zebracorn", k=3, hybrid=True)
    assert kw.mode == "hybrid" and kw.chunks[0].doc_id == "d-domain"
    assert all(0 <= c.score <= 1.0001 for c in kw.chunks)  # cosine kept even for keyword-only hits
    assert r.search("   ", k=3).chunks == []
