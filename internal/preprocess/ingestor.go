package preprocess

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// TicketStore is the persistence the Ingestor needs (store.Tickets satisfies it).
type TicketStore interface {
	Insert(ctx context.Context, t model.Ticket) (string, error)
	ContentHashExists(ctx context.Context, hash string) (bool, error)
}

// Outcome of processing one ticket.
const (
	OutcomeInserted         = "inserted"         // stored, accepted for classification
	OutcomeInsertedSkipped  = "inserted_skipped" // stored with ingest_status=skipped (non-English)
	OutcomeDuplicateID      = "duplicate_external_id"
	OutcomeDuplicateContent = "duplicate_content"
	OutcomeDropped          = "dropped" // not stored (empty)
)

// Result describes what happened to one ticket.
type Result struct {
	Ticket  model.Ticket // normalised; ID set when inserted
	Outcome string       // one of the Outcome* constants
	Reason  string       // skip/drop reason or duplicate detail
}

// Stats summarises a batch.
type Stats struct {
	Inserted, InsertedSkipped, DuplicateID, DuplicateContent, Dropped int
}

// Ingestor normalises tickets and persists the new ones.
type Ingestor struct {
	Store TicketStore
	Log   *slog.Logger
}

func (in *Ingestor) log() *slog.Logger {
	if in.Log != nil {
		return in.Log
	}
	return slog.Default()
}

// Process normalises and stores each ticket. Duplicates are rejected by
// (source, external_id) and by content_hash. Non-English tickets are stored
// with ingest_status "skipped" (never classified); empty tickets are dropped.
// Skips and drops are logged with a reason. The batch stops only on a storage
// error.
//
// The content-hash check-then-insert is not atomic; it assumes one ingestion
// worker. Concurrent workers could store the same content twice.
func (in *Ingestor) Process(ctx context.Context, tickets []model.Ticket) ([]Result, Stats, error) {
	var stats Stats
	results := make([]Result, 0, len(tickets))
	for _, raw := range tickets {
		t, skip, detail := Normalize(raw)
		res := Result{Ticket: t}
		if skip != "" {
			res.Reason = skip + ": " + detail
			in.log().Info("ingest: ticket not accepted for classification", "source", t.Source,
				"external_id", t.ExternalID, "reason", res.Reason, "stored", skip != SkipEmpty)
		}
		switch {
		case skip == SkipEmpty:
			res.Outcome = OutcomeDropped
			stats.Dropped++
		default:
			exists, err := in.Store.ContentHashExists(ctx, t.ContentHash)
			if err != nil {
				return results, stats, err
			}
			if exists {
				res.Outcome, res.Reason = OutcomeDuplicateContent, "same sender/subject/body already stored"
				stats.DuplicateContent++
				break
			}
			id, err := in.Store.Insert(ctx, t)
			switch {
			case errors.Is(err, store.ErrDuplicate):
				res.Outcome = OutcomeDuplicateID
				stats.DuplicateID++
			case err != nil:
				return results, stats, fmt.Errorf("store ticket %s/%s: %w", t.Source, t.ExternalID, err)
			case t.IngestStatus == model.IngestSkipped:
				res.Ticket.ID, res.Outcome = id, OutcomeInsertedSkipped
				stats.InsertedSkipped++
			default:
				res.Ticket.ID, res.Outcome = id, OutcomeInserted
				stats.Inserted++
			}
		}
		results = append(results, res)
	}
	return results, stats, nil
}
