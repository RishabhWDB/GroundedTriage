-- +goose Up
-- P7.3 monitoring: per-call stage timings (worker -> ml-service), one row per
-- (day, metric) computed by the worker's monitoring job, and the moment a
-- route was resolved (for the SLA breach rate).
create table if not exists stage_timings (
    id         bigint generated always as identity primary key,
    stage      text not null,
    ms         double precision not null,
    ok         boolean not null,
    created_at timestamptz not null default now()
);
create index if not exists stage_timings_created_idx on stage_timings (created_at);

create table if not exists metrics_daily (
    day         date not null,
    metric      text not null,
    value       double precision,
    detail      jsonb not null default '{}'::jsonb,
    computed_at timestamptz not null default now(),
    primary key (day, metric)
);

alter table routes add column if not exists resolved_at timestamptz;
update routes set resolved_at = updated_at where status = 'resolved' and resolved_at is null;

-- +goose Down
alter table routes drop column if exists resolved_at;
drop table if exists metrics_daily;
drop table if exists stage_timings;
