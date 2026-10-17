-- +goose Up
-- P5.2: keyword side of the optional hybrid retriever (Postgres full-text
-- search, ts_rank_cd; BM25-like). Generated from title + text, so ingestion
-- needs no change.
alter table kb_chunks add column if not exists tsv tsvector
    generated always as (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))) stored;
create index if not exists kb_chunks_tsv_idx on kb_chunks using gin (tsv);

-- +goose Down
drop index if exists kb_chunks_tsv_idx;
alter table kb_chunks drop column if exists tsv;
