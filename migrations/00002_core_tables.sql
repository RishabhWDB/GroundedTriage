-- +goose Up
create table if not exists tickets (
    id           uuid primary key default gen_random_uuid(),
    source       text not null check (source in ('email', 'zendesk', 'csv')),
    external_id  text not null,
    sender       text not null default '',
    subject      text not null default '',
    body         text not null default '',
    received_at  timestamptz not null,
    thread_id    text,
    raw_ref      text not null default '',
    content_hash text not null,
    created_at   timestamptz not null default now(),
    unique (source, external_id)
);
create index if not exists tickets_content_hash_idx on tickets (content_hash);
create index if not exists tickets_received_at_idx on tickets (received_at desc);

create table if not exists classifications (
    id                  bigint generated always as identity primary key,
    ticket_id           uuid not null references tickets (id) on delete cascade,
    category            text not null,
    category_confidence real not null,
    tier                text not null check (tier in ('fast', 'deep')),
    low_confidence      boolean not null default false,
    urgency             text not null check (urgency in ('low', 'medium', 'high', 'critical')),
    urgency_confidence  real not null,
    urgency_source      text not null check (urgency_source in ('model', 'heuristic')),
    models_loaded       jsonb not null default '{}',
    model_versions      jsonb not null default '{}',
    created_at          timestamptz not null default now()
);
create index if not exists classifications_ticket_idx on classifications (ticket_id);

create table if not exists routes (
    id          bigint generated always as identity primary key,
    ticket_id   uuid not null references tickets (id) on delete cascade,
    action      text not null check (action in ('mute', 'queue', 'reply', 'schedule')),
    queue       text not null default '',
    priority    integer not null default 0,
    sla_due_at  timestamptz,
    reason      text not null default '',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index if not exists routes_ticket_idx on routes (ticket_id);
create index if not exists routes_queue_sla_idx on routes (queue, sla_due_at);

create table if not exists kb_documents (
    id         text primary key,
    source     text not null default '',
    title      text not null default '',
    url        text not null default '',
    metadata   jsonb not null default '{}',
    created_at timestamptz not null default now()
);

create table if not exists kb_chunks (
    id          text primary key,
    doc_id      text not null references kb_documents (id) on delete cascade,
    ordinal     integer not null,
    title       text not null default '',
    heading     text not null default '',
    text        text not null,
    token_count integer not null default 0,
    embedding   vector(384),
    metadata    jsonb not null default '{}',
    created_at  timestamptz not null default now(),
    unique (doc_id, ordinal)
);
create index if not exists kb_chunks_embedding_hnsw_idx
    on kb_chunks using hnsw (embedding vector_cosine_ops);

create table if not exists reply_drafts (
    id                 uuid primary key default gen_random_uuid(),
    ticket_id          uuid not null references tickets (id) on delete cascade,
    subject            text not null default '',
    body               text not null default '',
    citations          text[] not null default '{}',
    faithfulness_score real,
    unsupported_claims jsonb not null default '[]',
    status             text not null
        check (status in ('grounded', 'abstained', 'approved', 'sent', 'rejected')),
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now()
);
create index if not exists reply_drafts_ticket_idx on reply_drafts (ticket_id);
create index if not exists reply_drafts_status_idx on reply_drafts (status);

create table if not exists audit_log (
    id          bigint generated always as identity primary key,
    at          timestamptz not null default now(),
    actor       text not null default 'system',
    entity_type text not null,
    entity_id   text not null default '',
    action      text not null,
    details     jsonb not null default '{}'
);
create index if not exists audit_log_entity_idx on audit_log (entity_type, entity_id);
create index if not exists audit_log_at_idx on audit_log (at desc);

-- +goose Down
-- Down migrations are never run against DATABASE_URL without Arya's approval.
drop table if exists audit_log;
drop table if exists reply_drafts;
drop table if exists kb_chunks;
drop table if exists kb_documents;
drop table if exists routes;
drop table if exists classifications;
drop table if exists tickets;
