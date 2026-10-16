package pipeline

import (
	"context"
	"log/slog"
	"sort"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// CheckpointStore persists per-source high-water marks (store.Checkpoints).
type CheckpointStore interface {
	Get(ctx context.Context, source string) (time.Time, bool, error)
	Set(ctx context.Context, source string, since time.Time) error
}

// AuditWriter appends audit_log entries (store.Audit).
type AuditWriter interface {
	Write(ctx context.Context, e store.AuditEntry) error
}

// TicketProcessor normalises and stores a batch (preprocess.Ingestor).
type TicketProcessor interface {
	Process(ctx context.Context, tickets []model.Ticket) ([]preprocess.Result, preprocess.Stats, error)
}

// Poller fetches from every enabled source and ingests new tickets.
type Poller struct {
	Sources     ingest.Sources
	Ingestor    TicketProcessor
	Checkpoints CheckpointStore
	Audit       AuditWriter
	// Overlap re-reads this window before the checkpoint on every poll, so
	// late-arriving items are not missed; dedupe drops the repeats.
	Overlap time.Duration
	// PollTimeout bounds one poll. A poll in progress is allowed to finish
	// (up to this timeout) when Run's context is cancelled.
	PollTimeout time.Duration
	// AfterPoll, when set, runs after every poll in Run (e.g. classification),
	// under the same timeout-bounded context.
	AfterPoll func(ctx context.Context)
	Log       *slog.Logger
}

// SourceResult is the outcome of polling one source.
type SourceResult struct {
	Fetched int
	Stats   preprocess.Stats
	Err     error
}

const maxAuditIDs = 100

func (p *Poller) log() *slog.Logger {
	if p.Log != nil {
		return p.Log
	}
	return slog.Default()
}

// PollOnce polls every source once. A failing source is logged and audited
// and does not stop the others; its checkpoint is not advanced.
func (p *Poller) PollOnce(ctx context.Context) map[string]SourceResult {
	names := make([]string, 0, len(p.Sources))
	for n := range p.Sources {
		names = append(names, n)
	}
	sort.Strings(names)
	out := make(map[string]SourceResult, len(names))
	for _, name := range names {
		res := p.pollSource(ctx, name, p.Sources[name])
		out[name] = res
		if res.Err != nil {
			p.log().Error("ingest: poll failed", "source", name, "err", res.Err)
			p.audit(ctx, store.AuditEntry{EntityType: "source", EntityID: name, Action: "ingest.error",
				Details: map[string]any{"error": res.Err.Error()}})
		}
	}
	return out
}

func (p *Poller) pollSource(ctx context.Context, name string, src ingest.Source) SourceResult {
	checkpoint, _, err := p.Checkpoints.Get(ctx, name)
	if err != nil {
		return SourceResult{Err: err}
	}
	since := checkpoint
	if !since.IsZero() {
		since = since.Add(-p.Overlap)
	}
	tickets, err := src.Fetch(ctx, since)
	if err != nil {
		return SourceResult{Err: err}
	}
	results, stats, err := p.Ingestor.Process(ctx, tickets)
	if err != nil {
		return SourceResult{Fetched: len(tickets), Stats: stats, Err: err}
	}

	next := checkpoint
	for _, t := range tickets {
		if t.ReceivedAt.After(next) {
			next = t.ReceivedAt
		}
	}
	if next.After(checkpoint) {
		if err := p.Checkpoints.Set(ctx, name, next); err != nil {
			return SourceResult{Fetched: len(tickets), Stats: stats, Err: err}
		}
	}

	var inserted []string
	for _, r := range results {
		stored := r.Outcome == preprocess.OutcomeInserted || r.Outcome == preprocess.OutcomeInsertedSkipped
		if stored && len(inserted) < maxAuditIDs {
			inserted = append(inserted, r.Ticket.ID)
		}
	}
	p.log().Info("ingest: polled", "source", name, "since", since, "fetched", len(tickets),
		"inserted", stats.Inserted, "inserted_skipped", stats.InsertedSkipped, "dup_id", stats.DuplicateID,
		"dup_content", stats.DuplicateContent, "dropped", stats.Dropped, "checkpoint", next)
	p.audit(ctx, store.AuditEntry{EntityType: "source", EntityID: name, Action: "ingest.poll", Details: map[string]any{
		"since": since, "checkpoint": next, "fetched": len(tickets),
		"inserted": stats.Inserted, "duplicate_external_id": stats.DuplicateID,
		"duplicate_content": stats.DuplicateContent, "inserted_skipped": stats.InsertedSkipped,
		"dropped":             stats.Dropped,
		"inserted_ticket_ids": inserted,
	}})
	return SourceResult{Fetched: len(tickets), Stats: stats}
}

func (p *Poller) audit(ctx context.Context, e store.AuditEntry) {
	if p.Audit == nil {
		return
	}
	if err := p.Audit.Write(ctx, e); err != nil {
		p.log().Warn("ingest: audit write failed", "action", e.Action, "err", err)
	}
}

// Run polls immediately and then every interval until ctx is cancelled.
// A poll in progress when ctx is cancelled runs to completion (bounded by
// PollTimeout), so shutdown never leaves a half-written batch behind.
func (p *Poller) Run(ctx context.Context, interval time.Duration) {
	timeout := p.PollTimeout
	if timeout == 0 {
		timeout = 5 * time.Minute
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		pollCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
		p.PollOnce(pollCtx)
		if p.AfterPoll != nil {
			p.AfterPoll(pollCtx)
		}
		cancel()
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}
