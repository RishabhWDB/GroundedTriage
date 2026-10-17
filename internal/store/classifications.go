package store

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Classifications is the repository for the classifications table. Rows keep
// history: a ticket may be classified again (e.g. after a model upgrade) and
// the latest row wins.
type Classifications struct{ Pool *pgxpool.Pool }

// Insert stores c (c.TicketID must be a tickets.id).
func (r Classifications) Insert(ctx context.Context, c model.Classification) error {
	loaded, err := json.Marshal(c.ModelsLoaded)
	if err != nil {
		return err
	}
	versions := c.ModelVersions
	if versions == nil {
		versions = map[string]string{}
	}
	vjson, err := json.Marshal(versions)
	if err != nil {
		return err
	}
	_, err = r.Pool.Exec(ctx, `
		insert into classifications (ticket_id, category, category_confidence, tier, low_confidence,
		                             urgency, urgency_confidence, urgency_source, models_loaded, model_versions,
		                             spam_score, possible_spam)
		values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`,
		c.TicketID, c.Category, c.CategoryConfidence, c.Tier, c.LowConfidence,
		c.Urgency, c.UrgencyConfidence, c.UrgencySource, loaded, vjson, c.SpamScore, c.PossibleSpam)
	if err != nil {
		return fmt.Errorf("insert classification for %s: %w", c.TicketID, err)
	}
	return nil
}

// Latest returns the most recent classification of a ticket; ok is false when
// it has none.
func (r Classifications) Latest(ctx context.Context, ticketID string) (c model.Classification, ok bool, err error) {
	var loaded, versions []byte
	rows, err := r.Pool.Query(ctx, `
		select ticket_id::text, category, category_confidence, tier, low_confidence, urgency,
		       urgency_confidence, urgency_source, models_loaded, model_versions,
		       spam_score, possible_spam
		from classifications where ticket_id = $1
		order by created_at desc, id desc limit 1`, ticketID)
	if err != nil {
		return c, false, fmt.Errorf("latest classification: %w", err)
	}
	defer rows.Close()
	if !rows.Next() {
		return c, false, rows.Err()
	}
	if err := rows.Scan(&c.TicketID, &c.Category, &c.CategoryConfidence, &c.Tier, &c.LowConfidence,
		&c.Urgency, &c.UrgencyConfidence, &c.UrgencySource, &loaded, &versions,
		&c.SpamScore, &c.PossibleSpam); err != nil {
		return c, false, fmt.Errorf("scan classification: %w", err)
	}
	if err := json.Unmarshal(loaded, &c.ModelsLoaded); err != nil {
		return c, false, err
	}
	if err := json.Unmarshal(versions, &c.ModelVersions); err != nil {
		return c, false, err
	}
	return c, true, nil
}

// Unclassified returns up to limit accepted tickets that have no
// classification yet, oldest first. Skipped tickets (e.g. non-English) are
// never returned: they go to a human queue unclassified (P4).
func (r Classifications) Unclassified(ctx context.Context, limit int) ([]model.Ticket, error) {
	rows, err := r.Pool.Query(ctx, `
		select t.id::text, t.source, t.external_id, t.sender, t.subject, t.body, t.received_at,
		       t.thread_id, t.raw_ref, t.content_hash, t.ingest_status, t.skip_reason
		from tickets t
		where t.ingest_status = 'accepted'
		  and not exists (select 1 from classifications c where c.ticket_id = t.id)
		order by t.received_at, t.id
		limit $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("unclassified tickets: %w", err)
	}
	defer rows.Close()
	var out []model.Ticket
	for rows.Next() {
		var t model.Ticket
		if err := rows.Scan(&t.ID, &t.Source, &t.ExternalID, &t.Sender, &t.Subject, &t.Body,
			&t.ReceivedAt, &t.ThreadID, &t.RawRef, &t.ContentHash, &t.IngestStatus, &t.SkipReason); err != nil {
			return nil, fmt.Errorf("scan ticket: %w", err)
		}
		out = append(out, t)
	}
	return out, rows.Err()
}
