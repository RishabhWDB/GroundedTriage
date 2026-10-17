import json
import os
import uuid

import numpy as np
import pytest

from rag import kb
from rag.chunking import Section

ARTICLE = {
    "id": "doc1",
    "url": "https://support.example/a",
    "title": "Refund policy",
    "html_content": (
        "<div>Intro text.</div><h2>How to request</h2><p>Open Billing and click Refund.</p>"
    ),
    "contents": "Refund policy\nIntro text.",
    "article_type": "article",
}


def write_wixqa(root, articles):
    (root / "wix_kb_corpus").mkdir(parents=True)
    (root / "wix_kb_corpus" / "wix_kb_corpus.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in articles)
    )
    (root / "wixqa_expertwritten").mkdir()
    (root / "wixqa_expertwritten" / "test.jsonl").write_text(
        json.dumps(
            {
                "question": "How do I get a refund?",
                "answer": "Open Billing.",
                "article_ids": "['doc1', 'doc2']",
            }
        )
        + "\n"  # fmt: skip
    )


def test_load_wixqa_and_qa(tmp_path):
    no_html = {**ARTICLE, "id": "doc2", "html_content": "", "contents": "# T\nPlain body."}
    write_wixqa(tmp_path, [ARTICLE, no_html])
    docs = list(kb.load_wixqa(tmp_path))
    assert [d.id for d in docs] == ["doc1", "doc2"]
    assert [s.heading for s in docs[0].sections] == ["", "How to request"]
    assert docs[1].sections[0].paragraphs == ["Plain body."]  # falls back to contents
    assert list(kb.load_wixqa(tmp_path, limit=1))[0].id == "doc1"
    qa = kb.load_wixqa_qa("expertwritten", tmp_path)
    assert qa[0].article_ids == ["doc1", "doc2"] and qa[0].kind == "expertwritten"


def test_load_folder(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "refunds.md").write_text("# Refunds\nWithin 30 days.\n")
    (tmp_path / "b.html").write_text("<h1>Login help</h1><p>Reset your password.</p>")
    (tmp_path / "notes.txt").write_text("ignored")
    docs = list(kb.load_folder(tmp_path))
    assert [(d.id, d.title) for d in docs] == [
        ("file:a/refunds.md", "Refunds"),
        ("file:b.html", "Login help"),
    ]


def test_document_hash_changes_with_content():
    a = kb.KBDocument("d", "s", "T", "u", [Section("H", ["one"])])
    b = kb.KBDocument("d", "s", "T", "u", [Section("H", ["two"])])
    assert (
        a.sha256() != b.sha256()
        and a.sha256() == kb.KBDocument("d", "s", "T", "u", [Section("H", ["one"])]).sha256()
    )
    assert kb.vector_literal([0.5, -1]) == "[0.500000,-1.000000]"


def fake_encode(texts):
    rng = np.random.default_rng(len(texts))
    v = rng.normal(size=(len(texts), 384)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def kb_conn():
    """A throwaway schema with migration 00002's tables on TEST_DATABASE_URL."""
    conn = kb.connect(os.environ["TEST_DATABASE_URL"])
    schema = f"test_kb_{uuid.uuid4().hex[:8]}"
    up = (
        (kb.REPO_ROOT / "migrations" / "00002_core_tables.sql")
        .read_text()
        .split("-- +goose Down")[0]
    )
    with conn.cursor() as cur:
        cur.execute(f'create schema "{schema}"')
        cur.execute(f'SET search_path TO "{schema}", public, extensions')
        cur.execute(up.replace("-- +goose Up", ""))
    conn.commit()
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f'drop schema "{schema}" cascade')
    conn.commit()
    conn.close()


@pytest.mark.integration
def test_ingest_is_idempotent_and_replaces_stale_chunks(kb_conn):
    long = kb.KBDocument("long", "test", "Long", "u", [Section("", [" ".join(["w"] * 700)])])
    short = kb.KBDocument("short", "test", "Short", "u", [Section("H", ["tiny"])])
    s1 = kb.ingest([long, short], kb_conn, fake_encode)
    assert s1["documents_written"] == 2 and s1["chunks_written"] == 4  # 700 words -> 3 chunks
    s2 = kb.ingest([long, short], kb_conn, fake_encode)
    assert s2["documents_unchanged"] == 2 and s2["chunks_written"] == 0
    shrunk = kb.KBDocument("long", "test", "Long", "u", [Section("", ["now short"])])
    s3 = kb.ingest([shrunk], kb_conn, fake_encode)
    assert s3["documents_written"] == 1
    with kb_conn.cursor() as cur:
        cur.execute("select count(*), max(ordinal) from kb_chunks where doc_id = 'long'")
        assert cur.fetchone() == (1, 0)  # stale chunks 1 and 2 deleted
        cur.execute("select vector_dims(embedding) from kb_chunks limit 1")
        assert cur.fetchone()[0] == 384
    st = kb.storage(kb_conn)
    assert st["documents"] == 2 and st["chunks"] == 2
