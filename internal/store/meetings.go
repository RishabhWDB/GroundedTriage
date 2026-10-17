package store

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Meeting is one scheduling outcome (meetings table).
type Meeting struct {
	TicketID       string
	Status         string // booked | counter_offer | no_slot | not_a_meeting
	RequestedStart *time.Time
	SlotStart      *time.Time
	SlotEnd        *time.Time
	EventID        string
	Timezone       string
	Venue          string
	Reason         string
	Extraction     any // stored as jsonb
}

// Meetings is the repository for the meetings table.
type Meetings struct{ Pool *pgxpool.Pool }

// PendingSchedule returns up to limit tickets whose current route is the
// `schedule` action and that have no meeting outcome yet, oldest first.
func (r Meetings) PendingSchedule(ctx context.Context, limit int) ([]model.Ticket, error) {
	rows, err := r.Pool.Query(ctx, `
		select t.id::text, t.source, t.external_id, t.sender, t.subject, t.body, t.received_at,
		       t.thread_id, t.raw_ref, t.content_hash, t.ingest_status, t.skip_reason
		from tickets t
		join lateral (select action from routes r where r.ticket_id = t.id
		              order by r.created_at desc, r.id desc limit 1) r on true
		where r.action = 'schedule'
		  and not exists (select 1 from meetings m where m.ticket_id = t.id)
		order by t.received_at, t.id
		limit $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("pending meetings: %w", err)
	}
	defer rows.Close()
	var out []model.Ticket
	for rows.Next() {
		var t model.Ticket
		if err := rows.Scan(&t.ID, &t.Source, &t.ExternalID, &t.Sender, &t.Subject, &t.Body,
			&t.ReceivedAt, &t.ThreadID, &t.RawRef, &t.ContentHash, &t.IngestStatus, &t.SkipReason); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// Insert stores an outcome (one per ticket).
func (r Meetings) Insert(ctx context.Context, m Meeting) error {
	ext, err := json.Marshal(m.Extraction)
	if err != nil {
		return err
	}
	if m.Extraction == nil {
		ext = []byte("{}")
	}
	_, err = r.Pool.Exec(ctx, `
		insert into meetings (ticket_id, status, requested_start, slot_start, slot_end, event_id,
		                      timezone, venue, reason, extraction)
		values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
		m.TicketID, m.Status, m.RequestedStart, m.SlotStart, m.SlotEnd, m.EventID,
		m.Timezone, m.Venue, m.Reason, ext)
	if err != nil {
		return fmt.Errorf("insert meeting for %s: %w", m.TicketID, err)
	}
	return nil
}

// Get returns a ticket's meeting outcome; ok is false when there is none.
func (r Meetings) Get(ctx context.Context, ticketID string) (m Meeting, ok bool, err error) {
	rows, err := r.Pool.Query(ctx, `
		select ticket_id::text, status, requested_start, slot_start, slot_end, event_id, timezone, venue, reason
		from meetings where ticket_id = $1`, ticketID)
	if err != nil {
		return m, false, err
	}
	defer rows.Close()
	if !rows.Next() {
		return m, false, rows.Err()
	}
	err = rows.Scan(&m.TicketID, &m.Status, &m.RequestedStart, &m.SlotStart, &m.SlotEnd, &m.EventID,
		&m.Timezone, &m.Venue, &m.Reason)
	return m, err == nil, err
}
