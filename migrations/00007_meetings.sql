-- +goose Up
-- P4.4: one scheduling outcome per Meeting Scheduling ticket routed to the
-- `schedule` action. booked = event created; counter_offer = a slot to
-- propose in the reply; no_slot = nothing free in the horizon;
-- not_a_meeting = the extraction found no meeting request.
create table if not exists meetings (
    id              bigint generated always as identity primary key,
    ticket_id       uuid not null unique references tickets (id) on delete cascade,
    status          text not null check (status in ('booked', 'counter_offer', 'no_slot', 'not_a_meeting')),
    requested_start timestamptz,
    slot_start      timestamptz,
    slot_end        timestamptz,
    event_id        text not null default '',
    timezone        text not null default '',
    venue           text not null default '',
    reason          text not null default '',
    extraction      jsonb not null default '{}',
    created_at      timestamptz not null default now()
);

-- +goose Down
drop table if exists meetings;
