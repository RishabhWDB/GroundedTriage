-- +goose Up
-- Tickets that are stored but must not be classified (e.g. non-English) are
-- kept with ingest_status = 'skipped' and a reason, so they stay visible and
-- can be routed to a human queue (P4).
alter table tickets add column if not exists ingest_status text not null default 'accepted';
alter table tickets add column if not exists skip_reason text not null default '';
-- +goose StatementBegin
do $$
begin
    if not exists (select 1 from pg_constraint
                   where conname = 'tickets_ingest_status_check'
                     and conrelid = 'tickets'::regclass) then
        alter table tickets add constraint tickets_ingest_status_check
            check (ingest_status in ('accepted', 'skipped'));
    end if;
end $$;
-- +goose StatementEnd
create index if not exists tickets_ingest_status_idx on tickets (ingest_status);

-- +goose Down
drop index if exists tickets_ingest_status_idx;
alter table tickets drop constraint if exists tickets_ingest_status_check;
alter table tickets drop column if exists skip_reason;
alter table tickets drop column if exists ingest_status;
