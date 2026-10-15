-- +goose Up
create table if not exists ingest_checkpoints (
    source     text primary key,
    since      timestamptz not null,
    updated_at timestamptz not null default now()
);

-- +goose Down
drop table if exists ingest_checkpoints;
