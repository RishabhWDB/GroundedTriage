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
	var id string
	err := r.Pool.QueryRow(ctx, `
		insert into tickets (source, external_id, sender, subject, body, received_at,
		                     thread_id, raw_ref, content_hash)
		values ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		returning id::text`,
		t.Source, t.ExternalID, t.Sender, t.Subject, t.Body, t.ReceivedAt,
		t.ThreadID, t.RawRef, t.ContentHash,
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
		       thread_id, raw_ref, content_hash
		from tickets where id = $1`, id,
	).Scan(&t.ID, &t.Source, &t.ExternalID, &t.Sender, &t.Subject, &t.Body,
		&t.ReceivedAt, &t.ThreadID, &t.RawRef, &t.ContentHash)
	if errors.Is(err, pgx.ErrNoRows) {
		return t, err
	}
	if err != nil {
		return t, fmt.Errorf("get ticket: %w", err)
	}
	return t, nil
}
