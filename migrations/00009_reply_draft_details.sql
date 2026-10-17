-- +goose Up
-- P5.5: what the agent review UI (P6.2) shows besides the draft itself.
alter table reply_drafts add column if not exists abstain_reason text not null default '';
alter table reply_drafts add column if not exists retrieval_max_score real;
alter table reply_drafts add column if not exists retrieval_mode text not null default '';
-- Per-claim judgements: [{claim, cited, supported, reason}].
alter table reply_drafts add column if not exists claims jsonb not null default '[]';
-- The retrieved chunk ids shown to the drafter, in rank order (citations are a subset).
alter table reply_drafts add column if not exists retrieved_chunk_ids text[] not null default '{}';
create index if not exists reply_drafts_ticket_idx on reply_drafts (ticket_id);

-- +goose Down
drop index if exists reply_drafts_ticket_idx;
alter table reply_drafts drop column if exists retrieved_chunk_ids;
alter table reply_drafts drop column if exists claims;
alter table reply_drafts drop column if exists retrieval_mode;
alter table reply_drafts drop column if exists retrieval_max_score;
alter table reply_drafts drop column if exists abstain_reason;
