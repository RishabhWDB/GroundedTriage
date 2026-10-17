package store

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// StoredDraft is a ReplyDraft plus the review details kept with it.
type StoredDraft struct {
	model.ReplyDraft
	AbstainReason     string
	RetrievalMaxScore *float64
	RetrievalMode     string
	Claims            any // jsonb: [{claim, cited, supported, reason}]
	RetrievedChunkIDs []string
}

// ReplyDrafts is the repository for reply_drafts.
type ReplyDrafts struct{ Pool *pgxpool.Pool }

// PendingReply is a ticket routed to `reply` without a draft, with its latest classification.
type PendingReply struct {
	Ticket   model.Ticket
	Category string
	Urgency  string
}

// PendingReplies returns up to limit tickets whose current route action is
// `reply` and that have no draft yet, oldest first.
func (r ReplyDrafts) PendingReplies(ctx context.Context, limit int) ([]PendingReply, error) {
	rows, err := r.Pool.Query(ctx, `
		select t.id::text, t.source, t.external_id, t.sender, t.subject, t.body, t.received_at,
		       t.thread_id, t.raw_ref, t.content_hash, t.ingest_status, t.skip_reason,
		       coalesce(c.category, ''), coalesce(c.urgency, 'medium')
		from tickets t
		join lateral (select action from routes r where r.ticket_id = t.id
		              order by r.created_at desc, r.id desc limit 1) r on true
		left join lateral (select category, urgency from classifications c where c.ticket_id = t.id
		                   order by c.created_at desc, c.id desc limit 1) c on true
		where r.action = 'reply'
		  and not exists (select 1 from reply_drafts d where d.ticket_id = t.id)
		order by t.received_at, t.id
		limit $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("pending replies: %w", err)
	}
	defer rows.Close()
	var out []PendingReply
	for rows.Next() {
		var p PendingReply
		t := &p.Ticket
		if err := rows.Scan(&t.ID, &t.Source, &t.ExternalID, &t.Sender, &t.Subject, &t.Body, &t.ReceivedAt,
			&t.ThreadID, &t.RawRef, &t.ContentHash, &t.IngestStatus, &t.SkipReason, &p.Category, &p.Urgency); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// Insert stores a draft and returns its id.
func (r ReplyDrafts) Insert(ctx context.Context, d StoredDraft) (string, error) {
	unsupported, err := json.Marshal(nonNil(d.UnsupportedClaims))
	if err != nil {
		return "", err
	}
	claims, err := json.Marshal(d.Claims)
	if err != nil {
		return "", err
	}
	if d.Claims == nil {
		claims = []byte("[]")
	}
	var id string
	err = r.Pool.QueryRow(ctx, `
		insert into reply_drafts (ticket_id, subject, body, citations, faithfulness_score,
		                          unsupported_claims, status, abstain_reason, retrieval_max_score,
		                          retrieval_mode, claims, retrieved_chunk_ids)
		values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) returning id::text`,
		d.TicketID, d.Subject, d.Body, nonNil(d.Citations), d.FaithfulnessScore, unsupported, d.Status,
		d.AbstainReason, d.RetrievalMaxScore, d.RetrievalMode, claims, nonNil(d.RetrievedChunkIDs)).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("insert draft for %s: %w", d.TicketID, err)
	}
	return id, nil
}

// Latest returns the newest draft of a ticket.
func (r ReplyDrafts) Latest(ctx context.Context, ticketID string) (d StoredDraft, ok bool, err error) {
	var unsupported []byte
	rows, err := r.Pool.Query(ctx, `
		select ticket_id::text, subject, body, citations, faithfulness_score, unsupported_claims, status,
		       abstain_reason, retrieval_max_score, retrieval_mode, retrieved_chunk_ids
		from reply_drafts where ticket_id = $1 order by created_at desc limit 1`, ticketID)
	if err != nil {
		return d, false, err
	}
	defer rows.Close()
	if !rows.Next() {
		return d, false, rows.Err()
	}
	if err := rows.Scan(&d.TicketID, &d.Subject, &d.Body, &d.Citations, &d.FaithfulnessScore, &unsupported,
		&d.Status, &d.AbstainReason, &d.RetrievalMaxScore, &d.RetrievalMode, &d.RetrievedChunkIDs); err != nil {
		return d, false, err
	}
	err = json.Unmarshal(unsupported, &d.UnsupportedClaims)
	return d, err == nil, err
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
