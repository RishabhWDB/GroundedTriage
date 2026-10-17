"""KB ingestion (P5.1): documents -> heading-aware chunks -> bge-small
embeddings -> kb_documents / kb_chunks (pgvector) on DATABASE_URL.

    python -m rag.kb --wixqa                 # == make kb (WixQA help-centre KB)
    python -m rag.kb --folder docs/kb        # any folder of .md / .html files

Sources:
- WixQA (Hugging Face Wix/WixQA, MIT): 6,221 Wix help-centre articles
  (wix_kb_corpus) plus the expert-written and simulated QA pairs kept for the
  retrieval evaluation (P5.2). Downloaded to datasets/raw/wixqa/.
- a folder of .md / .html files (doc id = "file:<relative path>").

Idempotent: each document's sha256 is stored in kb_documents.metadata; an
unchanged document is skipped, a changed one is re-chunked (stale chunks
deleted). Embeddings are cached on disk (triage.embedder.EmbeddingCache,
max_length 512), so a re-run only embeds new text. Every connection sets
`search_path TO public, extensions` (pgvector lives in `extensions`).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.settings import REPO_ROOT
from rag.chunking import (
    MAX_WORDS,
    OVERLAP_WORDS,
    Section,
    chunk_sections,
    html_sections,
    markdown_sections,
)

logger = logging.getLogger(__name__)

WIXQA_DIR = REPO_ROOT / "datasets" / "raw" / "wixqa"
WIXQA_URL = "https://huggingface.co/datasets/Wix/WixQA/resolve/main/"
WIXQA_FILES = (
    "wix_kb_corpus/wix_kb_corpus.jsonl",
    "wixqa_expertwritten/test.jsonl",
    "wixqa_simulated/test.jsonl",
)
# Part of every stored document hash: changing the chunker re-chunks everything.
CHUNKER = f"packed-v2:{MAX_WORDS}:{OVERLAP_WORDS}"
KB_MAX_LENGTH = 512  # bge-small's limit; chunks are ~400 tokens
DOC_BATCH = 100

UPSERT_DOCUMENT = """
insert into kb_documents (id, source, title, url, metadata)
values (%s, %s, %s, %s, %s)
on conflict (id) do update set source = excluded.source, title = excluded.title,
  url = excluded.url, metadata = excluded.metadata"""
UPSERT_CHUNK = """
insert into kb_chunks (id, doc_id, ordinal, title, heading, text, token_count, embedding)
values (%s, %s, %s, %s, %s, %s, %s, %s::vector)
on conflict (id) do update set title = excluded.title, heading = excluded.heading,
  text = excluded.text, token_count = excluded.token_count, embedding = excluded.embedding"""


@dataclass
class KBDocument:
    id: str
    source: str
    title: str
    url: str
    sections: list[Section]
    metadata: dict = field(default_factory=dict)

    def sha256(self) -> str:
        h = hashlib.sha256(f"{CHUNKER}\n{self.title}\n{self.url}".encode())
        for s in self.sections:
            h.update(f"\n#{s.heading}\n".encode())
            for p in s.paragraphs:
                h.update(p.encode() + b"\n")
        return h.hexdigest()


@dataclass
class QAPair:
    question: str
    answer: str
    article_ids: list[str]
    kind: str  # expertwritten | simulated


def download_wixqa(root: Path = WIXQA_DIR) -> Path:
    for rel in WIXQA_FILES:
        path = root / rel
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading WixQA %s", rel)
        with httpx.stream("GET", WIXQA_URL + rel, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
            tmp.rename(path)
    return root


def load_wixqa(root: Path = WIXQA_DIR, limit: int | None = None) -> Iterator[KBDocument]:
    with open(root / WIXQA_FILES[0], encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            r = json.loads(line)
            sections = html_sections(r.get("html_content") or "")
            if not any(s.paragraphs for s in sections):
                sections = markdown_sections(r.get("contents") or "")
            yield KBDocument(
                id=r["id"], source="wixqa", title=r.get("title") or "", url=r.get("url") or "",
                sections=sections, metadata={"article_type": r.get("article_type", "")},
            )  # fmt: skip


def load_folder(folder: Path) -> Iterator[KBDocument]:
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in (".md", ".markdown", ".html", ".htm") or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        is_md = path.suffix.lower() in (".md", ".markdown")
        sections = markdown_sections(text) if is_md else html_sections(text)
        title = next((s.heading for s in sections if s.heading), path.stem)
        rel = path.relative_to(folder).as_posix()
        yield KBDocument(id=f"file:{rel}", source=f"folder:{folder.name}", title=title, url=rel,
                         sections=sections)  # fmt: skip


def load_wixqa_qa(kind: str = "expertwritten", root: Path = WIXQA_DIR) -> list[QAPair]:
    """QA pairs for the retrieval evaluation; article_ids are WixQA doc ids."""
    out = []
    with open(root / f"wixqa_{kind}" / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids = r.get("article_ids") or []
            if isinstance(ids, str):
                ids = ast.literal_eval(ids)  # stored as a Python-literal string
            out.append(QAPair(r["question"], r.get("answer", ""), [str(i) for i in ids], kind))
    return out


# TCP keepalives make a silently dropped connection (pooler/NAT idle timeout)
# fail within ~1 min instead of blocking forever in recv; seen in P7.2, where
# minutes of judge calls passed between queries.
KEEPALIVES = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}


def connect(url: str | None = None, statement_timeout_ms: int = 120_000):
    import psycopg

    url = url or os.environ.get("DATABASE_URL", "")
    if not url:
        raise SystemExit("DATABASE_URL is unset")
    conn = psycopg.connect(url, autocommit=False, connect_timeout=15, **KEEPALIVES)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public, extensions")
        cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
    conn.commit()
    return conn


def vector_literal(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def _existing_hashes(conn, ids: list[str]) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("select id, metadata->>'sha256' from kb_documents where id = any(%s)", (ids,))
        return {i: h for i, h in cur.fetchall()}


def ingest(docs: Iterable[KBDocument], conn, encode, doc_batch: int = DOC_BATCH) -> dict:
    """encode(list[str]) -> (n, 384) array. Returns counts."""
    stats = {"documents_seen": 0, "documents_written": 0, "documents_unchanged": 0,
             "chunks_written": 0, "empty_documents": 0}  # fmt: skip
    batch: list[KBDocument] = []

    def flush():
        if not batch:
            return
        known = _existing_hashes(conn, [d.id for d in batch])
        todo = []
        for d in batch:
            sha = d.sha256()
            if known.get(d.id) == sha:
                stats["documents_unchanged"] += 1
                continue
            chunks = chunk_sections(d.sections)
            if not chunks:
                stats["empty_documents"] += 1
                continue
            todo.append((d, sha, chunks))
        texts = [f"{d.title}\n{c.text}" if d.title and c.heading != d.title else c.text
                 for d, _, chunks in todo for c in chunks]  # fmt: skip
        vecs = encode(texts) if texts else []
        k = 0
        with conn.cursor() as cur:
            for d, sha, chunks in todo:
                meta = {**d.metadata, "sha256": sha, "chunks": len(chunks)}
                cur.execute(
                    UPSERT_DOCUMENT,
                    (d.id, d.source, d.title, d.url, json.dumps(meta)),
                )
                rows = []
                for c in chunks:
                    rows.append((f"{d.id}:{c.ordinal}", d.id, c.ordinal, d.title, c.heading, c.text,
                                 c.token_count, vector_literal(vecs[k])))  # fmt: skip
                    k += 1
                cur.executemany(
                    UPSERT_CHUNK,
                    rows,
                )
                cur.execute(
                    "delete from kb_chunks where doc_id = %s and ordinal >= %s", (d.id, len(chunks))
                )
                stats["documents_written"] += 1
                stats["chunks_written"] += len(chunks)
        conn.commit()
        batch.clear()

    for d in docs:
        stats["documents_seen"] += 1
        batch.append(d)
        if len(batch) >= doc_batch:
            flush()
            logger.info("kb: %s", stats)
    flush()
    return stats


def storage(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """select (select count(*) from kb_documents), (select count(*) from kb_chunks),
                      pg_total_relation_size('kb_documents'), pg_total_relation_size('kb_chunks'),
                      pg_relation_size('kb_chunks_embedding_hnsw_idx'),
                      pg_database_size(current_database()),
                      (select coalesce(sum(token_count), 0) from kb_chunks)"""
        )
        docs, chunks, doc_bytes, chunk_bytes, hnsw_bytes, db_bytes, tokens = cur.fetchone()
    mb = 1024 * 1024
    return {
        "documents": docs,
        "chunks": chunks,
        "approx_tokens": int(tokens),
        "kb_documents_mb": round(doc_bytes / mb, 1),
        "kb_chunks_mb_incl_indexes": round(chunk_bytes / mb, 1),
        "hnsw_index_mb": round(hnsw_bytes / mb, 1),
        "database_mb": round(db_bytes / mb, 1),
    }


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from eval.runinfo import run_info
    from triage.embedder import Embedder, EmbeddingCache

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wixqa", action="store_true", help="ingest the WixQA help-centre KB")
    ap.add_argument("--folder", type=Path, help="ingest a folder of .md/.html documents")
    ap.add_argument("--limit", type=int, default=None, help="first N WixQA articles (smoke runs)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(REPO_ROOT / ".env")
    if not (args.wixqa or args.folder):
        ap.error("give --wixqa and/or --folder")
    cache = EmbeddingCache(Embedder(max_length=KB_MAX_LENGTH))
    conn = connect()
    t0 = time.perf_counter()
    stats = {}
    if args.wixqa:
        stats["wixqa"] = ingest(load_wixqa(download_wixqa(), args.limit), conn, cache.encode)
    if args.folder:
        stats["folder"] = ingest(load_folder(args.folder), conn, cache.encode)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    stats["storage"] = storage(conn)
    stats["run"] = run_info()
    conn.close()
    out = REPO_ROOT / "results" / "kb"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ingest.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "run"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
