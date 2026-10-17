package outbound

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// Queue is the send queue (store.Outbound).
type Queue interface {
	Claim(ctx context.Context, now time.Time, limit, maxAttempts int) ([]store.Claimed, error)
	MarkSent(ctx context.Context, draftID string, now time.Time) error
	MarkFailed(ctx context.Context, draftID, msg string) error
	FlagStale(ctx context.Context, before time.Time) ([]string, error)
}

// AuditWriter appends audit_log rows (store.Audit).
type AuditWriter interface {
	Write(ctx context.Context, e store.AuditEntry) error
}

// Sender claims approved drafts and sends each through the adapter named by
// its ticket's source (ingest.Source.Reply), passing the draft's
// idempotency key.
//
//   - success -> draft `sent`, route resolved, audit `reply.sent`;
//   - adapter error -> send_error recorded, claim released, retried next run
//     (at most MaxAttempts sends attempted);
//   - a claim older than Lease (the process died mid-send) -> flagged
//     "send outcome unknown" and never resent automatically.
type Sender struct {
	Queue       Queue
	Sources     ingest.Sources // by Ticket.Source (csv | email | zendesk)
	Audit       AuditWriter
	BatchSize   int           // default 20
	MaxAttempts int           // default 5
	Lease       time.Duration // default 10m
	Now         func() time.Time
	Log         *slog.Logger
}

// SendStats summarises one run.
type SendStats struct {
	Claimed, Sent, Failed, Stale int
}

func (s *Sender) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

func (s *Sender) log() *slog.Logger {
	if s.Log != nil {
		return s.Log
	}
	return slog.Default()
}

// SendApproved runs one send pass.
func (s *Sender) SendApproved(ctx context.Context) (SendStats, error) {
	var st SendStats
	batch, maxAttempts, lease := s.BatchSize, s.MaxAttempts, s.Lease
	if batch <= 0 {
		batch = 20
	}
	if maxAttempts <= 0 {
		maxAttempts = 5
	}
	if lease <= 0 {
		lease = 10 * time.Minute
	}
	stale, err := s.Queue.FlagStale(ctx, s.now().Add(-lease))
	if err != nil {
		return st, err
	}
	st.Stale = len(stale)
	for _, id := range stale {
		s.log().Error("outbound: send outcome unknown; not resending", "draft", id)
		s.audit(ctx, store.AuditEntry{EntityType: "draft", EntityID: id, Action: "reply.send_unknown"})
	}
	claimed, err := s.Queue.Claim(ctx, s.now(), batch, maxAttempts)
	if err != nil {
		return st, err
	}
	st.Claimed = len(claimed)
	for _, c := range claimed {
		src, ok := s.Sources[c.Ticket.Source]
		var sendErr error
		if !ok {
			sendErr = fmt.Errorf("no enabled %q source to reply through", c.Ticket.Source)
		} else {
			sendErr = src.Reply(ctx, c.Ticket, c.Draft)
		}
		details := map[string]any{"ticket_id": c.Ticket.ID, "source": c.Ticket.Source,
			"idempotency_key": c.Draft.IdempotencyKey, "attempt": c.Attempt}
		if sendErr != nil {
			st.Failed++
			details["error"] = sendErr.Error()
			if err := s.Queue.MarkFailed(ctx, c.DraftID, sendErr.Error()); err != nil {
				return st, err
			}
			s.log().Warn("outbound: send failed", "draft", c.DraftID, "attempt", c.Attempt, "err", sendErr)
			s.audit(ctx, store.AuditEntry{EntityType: "draft", EntityID: c.DraftID, Action: "reply.send_failed", Details: details})
			continue
		}
		if err := s.Queue.MarkSent(ctx, c.DraftID, s.now()); err != nil {
			// The adapter accepted it: leave the claim so it isn't resent; FlagStale reports it.
			return st, fmt.Errorf("sent draft %s but could not record it: %w", c.DraftID, err)
		}
		st.Sent++
		s.audit(ctx, store.AuditEntry{EntityType: "draft", EntityID: c.DraftID, Action: "reply.sent", Details: details})
	}
	if st.Claimed+st.Stale > 0 {
		s.log().Info("outbound: run", "claimed", st.Claimed, "sent", st.Sent, "failed", st.Failed, "stale", st.Stale)
	}
	return st, nil
}

func (s *Sender) audit(ctx context.Context, e store.AuditEntry) {
	if s.Audit == nil {
		return
	}
	if err := s.Audit.Write(ctx, e); err != nil {
		s.log().Warn("outbound: audit write failed", "err", err)
	}
}
