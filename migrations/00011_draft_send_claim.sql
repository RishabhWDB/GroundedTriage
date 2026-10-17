-- +goose Up
-- P6.3: a sender claims an approved draft before calling the adapter, so a
-- draft is never sent twice. A claim left behind by a crash is not retried
-- automatically (the send may have happened); it is flagged for a human.
alter table reply_drafts add column if not exists send_started_at timestamptz;
alter table reply_drafts add column if not exists send_attempts integer not null default 0;
create index if not exists reply_drafts_to_send_idx on reply_drafts (reviewed_at)
    where status = 'approved' and sent_at is null;

-- +goose Down
drop index if exists reply_drafts_to_send_idx;
alter table reply_drafts drop column if exists send_attempts;
alter table reply_drafts drop column if exists send_started_at;
