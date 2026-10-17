package pipeline

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// DraftStore persists drafts (store.ReplyDrafts).
type DraftStore interface {
	PendingReplies(ctx context.Context, limit int) ([]store.PendingReply, error)
	Insert(ctx context.Context, d store.StoredDraft) (string, error)
}

// RouteReassigner moves a ticket to another queue (store.Routes).
type RouteReassigner interface {
	Reassign(ctx context.Context, ticketID, queue, reason string) (bool, error)
}

// GroundedReplyML is the ml-service side of a reply (mlclient.Client).
type GroundedReplyML interface {
	Retrieve(ctx context.Context, req mlclient.RetrieveRequest) (mlclient.RetrieveResponse, error)
	Draft(ctx context.Context, req mlclient.DraftRequest) (mlclient.GeneratedDraft, error)
	Verify(ctx context.Context, req mlclient.VerifyRequest) (mlclient.Verification, error)
}

// Replier handles tickets routed to the `reply` action (P5.5): retrieve ->
// draft -> verify -> store the draft. A grounded draft waits for agent
// review (P6); an abstention is stored with its reason and the ticket is
// moved to the human queue. An ml-service failure leaves the ticket pending
// for the next run.
type Replier struct {
	Drafts       DraftStore
	Routes       RouteReassigner
	ML           GroundedReplyML
	Audit        AuditWriter
	K            int    // chunks to retrieve (default 5)
	Hybrid       bool   // hybrid retrieval (default dense)
	AbstainQueue string // default human_review
	BatchSize    int    // default 10 (each ticket costs three LLM calls)
	Log          *slog.Logger
}

// ReplyStats summarises one run.
type ReplyStats struct {
	Pending, Grounded, Abstained, Failed int
}

func (r *Replier) log() *slog.Logger {
	if r.Log != nil {
		return r.Log
	}
	return slog.Default()
}

// ReplyPending drafts replies for up to BatchSize tickets.
func (r *Replier) ReplyPending(ctx context.Context) (ReplyStats, error) {
	var st ReplyStats
	limit := r.BatchSize
	if limit <= 0 {
		limit = 10
	}
	pending, err := r.Drafts.PendingReplies(ctx, limit)
	if err != nil {
		return st, err
	}
	st.Pending = len(pending)
	for _, p := range pending {
		d, err := r.replyOne(ctx, p)
		if err != nil {
			st.Failed++
			r.log().Warn("reply: ticket failed; retrying next run", "ticket", p.Ticket.ID, "err", err)
			continue
		}
		if _, err := r.Drafts.Insert(ctx, d); err != nil {
			return st, err
		}
		details := map[string]any{"status": d.Status, "faithfulness": d.FaithfulnessScore,
			"citations": d.Citations, "retrieval_max_score": d.RetrievalMaxScore}
		if d.Status == "abstained" {
			st.Abstained++
			details["reason"] = d.AbstainReason
			queue := r.AbstainQueue
			if queue == "" {
				queue = "human_review"
			}
			if _, err := r.Routes.Reassign(ctx, p.Ticket.ID, queue, "draft abstained: "+d.AbstainReason); err != nil {
				return st, err
			}
		} else {
			st.Grounded++
		}
		r.audit(ctx, store.AuditEntry{EntityType: "ticket", EntityID: p.Ticket.ID, Action: "reply." + d.Status,
			Details: details})
	}
	if st.Pending > 0 {
		r.log().Info("reply: run", "pending", st.Pending, "grounded", st.Grounded, "abstained", st.Abstained,
			"failed", st.Failed)
	}
	return st, nil
}

func (r *Replier) replyOne(ctx context.Context, p store.PendingReply) (store.StoredDraft, error) {
	k := r.K
	if k <= 0 {
		k = 5
	}
	t := p.Ticket
	ret, err := r.ML.Retrieve(ctx, mlclient.RetrieveRequest{TicketID: t.ID, Subject: t.Subject, Body: t.Body,
		K: k, Hybrid: r.Hybrid})
	if err != nil {
		return store.StoredDraft{}, fmt.Errorf("retrieve: %w", err)
	}
	ids := make([]string, len(ret.Chunks))
	for i, c := range ret.Chunks {
		ids[i] = c.ChunkID
	}
	maxScore := ret.MaxScore
	base := store.StoredDraft{RetrievalMaxScore: &maxScore, RetrievalMode: ret.Mode, RetrievedChunkIDs: ids}
	if len(ret.Chunks) == 0 {
		base.ReplyDraft = model.ReplyDraft{TicketID: t.ID, Status: "abstained"}
		base.AbstainReason = "no retrieved sources"
		return base, nil
	}
	// Below the gate's retrieval floor the draft would abstain anyway: skip
	// the drafting and judging calls.
	if ret.BelowFloor {
		base.ReplyDraft = model.ReplyDraft{TicketID: t.ID, Status: "abstained"}
		base.AbstainReason = fmt.Sprintf("best retrieval score %.3f < floor %g", ret.MaxScore, ret.RetrievalFloor)
		return base, nil
	}
	draft, err := r.ML.Draft(ctx, mlclient.DraftRequest{TicketID: t.ID, Subject: t.Subject, Body: t.Body,
		Category: p.Category, Urgency: p.Urgency, Chunks: ret.Chunks})
	if err != nil {
		return store.StoredDraft{}, fmt.Errorf("draft: %w", err)
	}
	v, err := r.ML.Verify(ctx, mlclient.VerifyRequest{TicketID: t.ID, Subject: draft.Subject, Body: draft.Body,
		Citations: draft.Citations, Chunks: ret.Chunks})
	if err != nil {
		return store.StoredDraft{}, fmt.Errorf("verify: %w", err)
	}
	base.ReplyDraft = v.Draft
	base.ReplyDraft.TicketID = t.ID
	base.AbstainReason = v.AbstainReason
	base.Claims = v.Claims
	return base, nil
}

func (r *Replier) audit(ctx context.Context, e store.AuditEntry) {
	if r.Audit == nil {
		return
	}
	if err := r.Audit.Write(ctx, e); err != nil {
		r.log().Warn("reply: audit write failed", "err", err)
	}
}
