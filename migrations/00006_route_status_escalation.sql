-- +goose Up
-- P4.3: routes carry a status (open until an agent resolves it, P6) and an
-- escalation count; the SLA escalation job only looks at open routes.
alter table routes add column if not exists status text not null default 'open';
alter table routes add column if not exists escalations integer not null default 0;
alter table routes add column if not exists escalated_at timestamptz;
-- +goose StatementBegin
do $$
begin
    if not exists (select 1 from pg_constraint
                   where conname = 'routes_status_check' and conrelid = 'routes'::regclass) then
        alter table routes add constraint routes_status_check check (status in ('open', 'resolved'));
    end if;
end $$;
-- +goose StatementEnd
create index if not exists routes_open_sla_idx on routes (sla_due_at) where status = 'open';

-- +goose Down
drop index if exists routes_open_sla_idx;
alter table routes drop constraint if exists routes_status_check;
alter table routes drop column if exists escalated_at;
alter table routes drop column if exists escalations;
alter table routes drop column if exists status;
