package store

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// UnknownOutcome prefixes send_error for a claim abandoned mid-send.
const UnknownOutcome = "send outcome unknown"

// Outbound is the send queue over reply_drafts (P6.3).
type Outbound struct{ Pool *pgxpool.Pool }

// Claimed is an approved draft claimed for sending, with its ticket.
type Claimed struct {
	DraftID string
	Ticket  model.Ticket
	Draft   model.ReplyDraft
	Attempt int
}

// Claim marks up to limit approved, unsent, unclaimed drafts as being sent
// (send_started_at = now) and returns them. Rows are locked with SKIP
// LOCKED, so concurrent senders never claim the same draft. Drafts that
// failed maxAttempts times or whose outcome is unknown are not claimed.
func (o Outbound) Claim(ctx context.Context, now time.Time, limit, maxAttempts int) ([]Claimed, error) {
	rows, err := o.Pool.Query(ctx, `
		with c as (
		    select id from reply_drafts
		    where status = 'approved' and sent_at is null and send_started_at is null
		      and send_attempts < $3 and send_error not like $4
		    order by reviewed_at, id
		    limit $2
		    for update skip locked
		)
		update reply_drafts d set send_started_at = $1, send_attempts = d.send_attempts + 1
		from c where d.id = c.id
		returning d.id::text, d.ticket_id::text, d.subject, d.body, d.citations, d.status,
		          coalesce(d.idempotency_key, ''), d.send_attempts`,
		now, limit, maxAttempts, UnknownOutcome+"%")
	if err != nil {
		return nil, fmt.Errorf("claim drafts: %w", err)
	}
	var out []Claimed
	for rows.Next() {
		var c Claimed
		if err := rows.Scan(&c.DraftID, &c.Draft.TicketID, &c.Draft.Subject, &c.Draft.Body, &c.Draft.Citations,
			&c.Draft.Status, &c.Draft.IdempotencyKey, &c.Attempt); err != nil {
			rows.Close()
			return nil, err
		}
		out = append(out, c)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for i := range out {
		t, err := Tickets{Pool: o.Pool}.Get(ctx, out[i].Draft.TicketID)
		if err != nil {
			return nil, fmt.Errorf("ticket of draft %s: %w", out[i].DraftID, err)
		}
		out[i].Ticket = t
	}
	return out, nil
}

// MarkSent records a successful send and resolves the ticket's route.
func (o Outbound) MarkSent(ctx context.Context, draftID string, now time.Time) error {
	tx, err := o.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx) //nolint:errcheck // no-op after commit
	var ticketID string
	err = tx.QueryRow(ctx, `
		update reply_drafts set status = 'sent', sent_at = $2, send_error = '', send_started_at = null, updated_at = $2
		where id = $1 and status = 'approved' returning ticket_id::text`, draftID, now).Scan(&ticketID)
	if err != nil {
		return fmt.Errorf("mark draft %s sent: %w", draftID, err)
	}
	if _, err := tx.Exec(ctx, `
		update routes set status = 'resolved', resolved_at = $2, updated_at = $2
		where id = (select id from routes where ticket_id = $1 order by created_at desc, id desc limit 1)`,
		ticketID, now); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// MarkFailed records a known send failure and releases the claim for a retry.
func (o Outbound) MarkFailed(ctx context.Context, draftID, msg string) error {
	_, err := o.Pool.Exec(ctx, `
		update reply_drafts set send_error = $2, send_started_at = null, updated_at = now() where id = $1`,
		draftID, msg)
	return err
}

// FlagStale marks claims older than before as outcome-unknown: the sender
// stopped mid-send, so the reply may or may not have gone out. They keep
// their claim and are never retried automatically. Returns the draft ids.
func (o Outbound) FlagStale(ctx context.Context, before time.Time) ([]string, error) {
	rows, err := o.Pool.Query(ctx, `
		update reply_drafts set send_error = $2, updated_at = now()
		where status = 'approved' and sent_at is null and send_started_at < $1 and send_error not like $3
		returning id::text`,
		before, UnknownOutcome+": the sender stopped mid-send; check the channel before re-sending", UnknownOutcome+"%")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}
