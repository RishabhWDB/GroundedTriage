package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// ErrDuplicate is returned when a ticket with the same (source, external_id)
// already exists.
var ErrDuplicate = errors.New("duplicate ticket")

// Tickets is the repository for the tickets table.
type Tickets struct{ Pool *pgxpool.Pool }

// Insert stores t and returns its generated id.
func (r Tickets) Insert(ctx context.Context, t model.Ticket) (string, error) {
	status := t.IngestStatus
	if status == "" {
		status = model.IngestAccepted
	}
	var id string
	err := r.Pool.QueryRow(ctx, `
		insert into tickets (source, external_id, sender, subject, body, received_at,
		                     thread_id, raw_ref, content_hash, ingest_status, skip_reason)
		values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		returning id::text`,
		t.Source, t.ExternalID, t.Sender, t.Subject, t.Body, t.ReceivedAt,
		t.ThreadID, t.RawRef, t.ContentHash, status, t.SkipReason,
	).Scan(&id)
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" {
		return "", ErrDuplicate
	}
	if err != nil {
		return "", fmt.Errorf("insert ticket: %w", err)
	}
	return id, nil
}

// Get returns the ticket with the given id, or pgx.ErrNoRows.
func (r Tickets) Get(ctx context.Context, id string) (model.Ticket, error) {
	var t model.Ticket
	err := r.Pool.QueryRow(ctx, `
		select id::text, source, external_id, sender, subject, body, received_at,
		       thread_id, raw_ref, content_hash, ingest_status, skip_reason
		from tickets where id = $1`, id,
	).Scan(&t.ID, &t.Source, &t.ExternalID, &t.Sender, &t.Subject, &t.Body,
		&t.ReceivedAt, &t.ThreadID, &t.RawRef, &t.ContentHash, &t.IngestStatus, &t.SkipReason)
	if errors.Is(err, pgx.ErrNoRows) {
		return t, err
	}
	if err != nil {
		return t, fmt.Errorf("get ticket: %w", err)
	}
	return t, nil
}

// ContentHashExists reports whether any stored ticket has this content hash.
func (r Tickets) ContentHashExists(ctx context.Context, hash string) (bool, error) {
	var exists bool
	err := r.Pool.QueryRow(ctx, `select exists (select 1 from tickets where content_hash = $1)`, hash).Scan(&exists)
	if err != nil {
		return false, fmt.Errorf("content hash lookup: %w", err)
	}
	return exists, nil
}
