-- +goose Up
-- P6: agent review and outbound sending of reply drafts.
alter table reply_drafts add column if not exists edited boolean not null default false;
alter table reply_drafts add column if not exists reviewer text not null default '';
alter table reply_drafts add column if not exists reviewed_at timestamptz;
alter table reply_drafts add column if not exists review_note text not null default '';
-- P6.3: one send per draft. The key is fixed when the draft is approved;
-- sent_at is set once the adapter accepted the reply.
alter table reply_drafts add column if not exists idempotency_key text;
alter table reply_drafts add column if not exists sent_at timestamptz;
alter table reply_drafts add column if not exists send_error text not null default '';
create unique index if not exists reply_drafts_idempotency_key_idx on reply_drafts (idempotency_key)
    where idempotency_key is not null;
create index if not exists reply_drafts_status_idx on reply_drafts (status);

-- +goose Down
drop index if exists reply_drafts_status_idx;
drop index if exists reply_drafts_idempotency_key_idx;
alter table reply_drafts drop column if exists send_error;
alter table reply_drafts drop column if exists sent_at;
alter table reply_drafts drop column if exists idempotency_key;
alter table reply_drafts drop column if exists review_note;
alter table reply_drafts drop column if exists reviewed_at;
alter table reply_drafts drop column if exists reviewer;
alter table reply_drafts drop column if exists edited;
