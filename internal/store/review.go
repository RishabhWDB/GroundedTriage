package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// ErrNotFound and ErrConflict are returned by the review queries.
var (
	ErrNotFound = errors.New("not found")
	ErrConflict = errors.New("conflict")
)

// Review holds the read and review queries behind the REST API (P6.1).
type Review struct{ Pool *pgxpool.Pool }

// TicketFilter selects tickets for GET /tickets.
type TicketFilter struct {
	Category string // a taxonomy category
	Urgency  string
	Status   string // open | resolved | unrouted
	Queue    string
	Limit    int
	Offset   int
}

// TicketSummary is one inbox row.
type TicketSummary struct {
	ID            string     `json:"id"`
	Source        string     `json:"source"`
	Sender        string     `json:"sender"`
	Subject       string     `json:"subject"`
	ReceivedAt    time.Time  `json:"received_at"`
	IngestStatus  string     `json:"ingest_status"`
	Category      *string    `json:"category"`
	Urgency       *string    `json:"urgency"`
	Tier          *string    `json:"tier"`
	LowConfidence bool       `json:"low_confidence"`
	PossibleSpam  bool       `json:"possible_spam"`
	Action        *string    `json:"action"`
	Queue         *string    `json:"queue"`
	Priority      *int       `json:"priority"`
	SLADueAt      *time.Time `json:"sla_due_at"`
	RouteStatus   *string    `json:"route_status"`
	DraftStatus   *string    `json:"draft_status"`
	Faithfulness  *float64   `json:"faithfulness_score"`
}

// RouteView is a ticket's current route.
type RouteView struct {
	model.Route
	Status      string     `json:"status"`
	Escalations int        `json:"escalations"`
	EscalatedAt *time.Time `json:"escalated_at"`
}

// MeetingView is a ticket's scheduling outcome.
type MeetingView struct {
	Status         string     `json:"status"`
	RequestedStart *time.Time `json:"requested_start"`
	SlotStart      *time.Time `json:"slot_start"`
	SlotEnd        *time.Time `json:"slot_end"`
	EventID        string     `json:"event_id"`
	Timezone       string     `json:"timezone"`
	Venue          string     `json:"venue"`
	Reason         string     `json:"reason"`
}

// DraftView is a reply draft with its review state.
type DraftView struct {
	ID                string          `json:"id"`
	TicketID          string          `json:"ticket_id"`
	Subject           string          `json:"subject"`
	Body              string          `json:"body"`
	Citations         []string        `json:"citations"`
	FaithfulnessScore *float64        `json:"faithfulness_score"`
	UnsupportedClaims json.RawMessage `json:"unsupported_claims"`
	Status            string          `json:"status"`
	AbstainReason     string          `json:"abstain_reason"`
	RetrievalMaxScore *float64        `json:"retrieval_max_score"`
	RetrievalMode     string          `json:"retrieval_mode"`
	Claims            json.RawMessage `json:"claims"`
	RetrievedChunkIDs []string        `json:"retrieved_chunk_ids"`
	Edited            bool            `json:"edited"`
	Reviewer          string          `json:"reviewer"`
	ReviewedAt        *time.Time      `json:"reviewed_at"`
	ReviewNote        string          `json:"review_note"`
	IdempotencyKey    *string         `json:"idempotency_key"`
	SentAt            *time.Time      `json:"sent_at"`
	SendError         string          `json:"send_error"`
	CreatedAt         time.Time       `json:"created_at"`
}

// ChunkView is a KB chunk shown when an agent clicks a citation.
type ChunkView struct {
	ID      string `json:"id"`
	DocID   string `json:"doc_id"`
	Title   string `json:"title"`
	Heading string `json:"heading"`
	Text    string `json:"text"`
	URL     string `json:"url"`
}

// TicketDetail is everything the ticket view shows.
type TicketDetail struct {
	Ticket         model.Ticket          `json:"ticket"`
	Classification *model.Classification `json:"classification"`
	Route          *RouteView            `json:"route"`
	Meeting        *MeetingView          `json:"meeting"`
	Draft          *DraftView            `json:"draft"`
}

const latestJoins = `
	left join lateral (select * from classifications c where c.ticket_id = t.id
	                   order by c.created_at desc, c.id desc limit 1) c on true
	left join lateral (select * from routes r where r.ticket_id = t.id
	                   order by r.created_at desc, r.id desc limit 1) r on true
	left join lateral (select * from reply_drafts d where d.ticket_id = t.id
	                   order by d.created_at desc limit 1) d on true`

// ListTickets returns one page of tickets, most urgent first (priority, then
// SLA deadline, then age), and the total matching count.
func (r Review) ListTickets(ctx context.Context, f TicketFilter) ([]TicketSummary, int, error) {
	var where []string
	var args []any
	add := func(cond string, v any) {
		args = append(args, v)
		where = append(where, fmt.Sprintf(cond, len(args)))
	}
	if f.Category != "" {
		add("c.category = $%d", f.Category)
	}
	if f.Urgency != "" {
		add("c.urgency = $%d", f.Urgency)
	}
	if f.Queue != "" {
		add("r.queue = $%d", f.Queue)
	}
	switch f.Status {
	case "":
	case "unrouted":
		where = append(where, "r.id is null")
	default:
		add("r.status = $%d", f.Status)
	}
	cond := ""
	if len(where) > 0 {
		cond = "where " + strings.Join(where, " and ")
	}
	var total int
	if err := r.Pool.QueryRow(ctx, "select count(*) from tickets t"+latestJoins+" "+cond, args...).Scan(&total); err != nil {
		return nil, 0, fmt.Errorf("count tickets: %w", err)
	}
	limit, offset := f.Limit, f.Offset
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	args = append(args, limit, offset)
	rows, err := r.Pool.Query(ctx, `
		select t.id::text, t.source, t.sender, t.subject, t.received_at, t.ingest_status,
		       c.category, c.urgency, c.tier, coalesce(c.low_confidence, false), coalesce(c.possible_spam, false),
		       r.action, r.queue, r.priority, r.sla_due_at, r.status, d.status, d.faithfulness_score
		from tickets t`+latestJoins+" "+cond+fmt.Sprintf(`
		order by r.priority asc nulls last, r.sla_due_at asc nulls last, t.received_at desc, t.id
		limit $%d offset $%d`, len(args)-1, len(args)), args...)
	if err != nil {
		return nil, 0, fmt.Errorf("list tickets: %w", err)
	}
	defer rows.Close()
	out := []TicketSummary{}
	for rows.Next() {
		var s TicketSummary
		var faith *float32
		if err := rows.Scan(&s.ID, &s.Source, &s.Sender, &s.Subject, &s.ReceivedAt, &s.IngestStatus,
			&s.Category, &s.Urgency, &s.Tier, &s.LowConfidence, &s.PossibleSpam, &s.Action, &s.Queue,
			&s.Priority, &s.SLADueAt, &s.RouteStatus, &s.DraftStatus, &faith); err != nil {
			return nil, 0, err
		}
		if faith != nil {
			v := float64(*faith)
			s.Faithfulness = &v
		}
		out = append(out, s)
	}
	return out, total, rows.Err()
}

// GetTicket returns a ticket with its latest classification, route, meeting and draft.
func (r Review) GetTicket(ctx context.Context, id string) (TicketDetail, error) {
	var d TicketDetail
	t, err := Tickets{Pool: r.Pool}.Get(ctx, id)
	if errors.Is(err, pgx.ErrNoRows) {
		return d, ErrNotFound
	}
	if err != nil {
		return d, err
	}
	d.Ticket = t
	if c, ok, err := (Classifications{Pool: r.Pool}).Latest(ctx, id); err != nil {
		return d, err
	} else if ok {
		d.Classification = &c
	}
	var rv RouteView
	err = r.Pool.QueryRow(ctx, `
		select ticket_id::text, action, queue, priority, sla_due_at, reason, status, escalations, escalated_at
		from routes where ticket_id = $1 order by created_at desc, id desc limit 1`, id).
		Scan(&rv.TicketID, &rv.Action, &rv.Queue, &rv.Priority, &rv.SLADueAt, &rv.Reason, &rv.Status,
			&rv.Escalations, &rv.EscalatedAt)
	switch {
	case err == nil:
		d.Route = &rv
	case !errors.Is(err, pgx.ErrNoRows):
		return d, err
	}
	var mv MeetingView
	err = r.Pool.QueryRow(ctx, `
		select status, requested_start, slot_start, slot_end, event_id, timezone, venue, reason
		from meetings where ticket_id = $1`, id).
		Scan(&mv.Status, &mv.RequestedStart, &mv.SlotStart, &mv.SlotEnd, &mv.EventID, &mv.Timezone, &mv.Venue, &mv.Reason)
	switch {
	case err == nil:
		d.Meeting = &mv
	case !errors.Is(err, pgx.ErrNoRows):
		return d, err
	}
	if dv, err := r.LatestDraft(ctx, id); err == nil {
		d.Draft = &dv
	} else if !errors.Is(err, ErrNotFound) {
		return d, err
	}
	return d, nil
}

const draftCols = `id::text, ticket_id::text, subject, body, citations, faithfulness_score, unsupported_claims,
	status, abstain_reason, retrieval_max_score, retrieval_mode, claims, retrieved_chunk_ids, edited,
	reviewer, reviewed_at, review_note, idempotency_key, sent_at, send_error, created_at`

func scanDraft(row pgx.Row) (DraftView, error) {
	var d DraftView
	var faith, maxScore *float32
	var unsupported, claims []byte
	err := row.Scan(&d.ID, &d.TicketID, &d.Subject, &d.Body, &d.Citations, &faith, &unsupported, &d.Status,
		&d.AbstainReason, &maxScore, &d.RetrievalMode, &claims, &d.RetrievedChunkIDs, &d.Edited, &d.Reviewer,
		&d.ReviewedAt, &d.ReviewNote, &d.IdempotencyKey, &d.SentAt, &d.SendError, &d.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return d, ErrNotFound
	}
	if err != nil {
		return d, err
	}
	if faith != nil {
		v := float64(*faith)
		d.FaithfulnessScore = &v
	}
	if maxScore != nil {
		v := float64(*maxScore)
		d.RetrievalMaxScore = &v
	}
	d.UnsupportedClaims, d.Claims = unsupported, claims
	return d, nil
}

// LatestDraft returns a ticket's newest draft.
func (r Review) LatestDraft(ctx context.Context, ticketID string) (DraftView, error) {
	return scanDraft(r.Pool.QueryRow(ctx, "select "+draftCols+
		" from reply_drafts where ticket_id = $1 order by created_at desc limit 1", ticketID))
}

// GetDraft returns a draft by id.
func (r Review) GetDraft(ctx context.Context, id string) (DraftView, error) {
	return scanDraft(r.Pool.QueryRow(ctx, "select "+draftCols+" from reply_drafts where id = $1", id))
}

// Chunks returns KB chunks by id, in the given order (missing ids skipped).
func (r Review) Chunks(ctx context.Context, ids []string) ([]ChunkView, error) {
	out := []ChunkView{}
	if len(ids) == 0 {
		return out, nil
	}
	rows, err := r.Pool.Query(ctx, `
		select c.id, c.doc_id, c.title, c.heading, c.text, coalesce(d.url, '')
		from kb_chunks c left join kb_documents d on d.id = c.doc_id
		where c.id = any($1)`, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	byID := map[string]ChunkView{}
	for rows.Next() {
		var c ChunkView
		if err := rows.Scan(&c.ID, &c.DocID, &c.Title, &c.Heading, &c.Text, &c.URL); err != nil {
			return nil, err
		}
		byID[c.ID] = c
	}
	for _, id := range ids {
		if c, ok := byID[id]; ok {
			out = append(out, c)
		}
	}
	return out, rows.Err()
}

// transition updates a draft only when its current state allows it.
func (r Review) transition(ctx context.Context, id, set, allowed string, args ...any) (DraftView, error) {
	all := append([]any{id}, args...)
	d, err := scanDraft(r.Pool.QueryRow(ctx, "update reply_drafts set "+set+", updated_at = now() where id = $1 and "+
		allowed+" returning "+draftCols, all...))
	if errors.Is(err, ErrNotFound) {
		if _, gerr := r.GetDraft(ctx, id); errors.Is(gerr, ErrNotFound) {
			return d, ErrNotFound
		}
		return d, ErrConflict
	}
	return d, err
}

// ApproveDraft approves a grounded draft, or any draft an agent has edited
// to have a body (including an abstained one). It fixes the idempotency key
// that outbound sending (P6.3) uses.
func (r Review) ApproveDraft(ctx context.Context, id, reviewer, note string) (DraftView, error) {
	return r.transition(ctx, id,
		"status = 'approved', reviewer = $2, review_note = $3, reviewed_at = now(), idempotency_key = 'draft-' || id::text",
		"(status = 'grounded' or (edited and status = 'abstained')) and btrim(body) <> ''", reviewer, note)
}

// RejectDraft rejects a draft that hasn't been sent.
func (r Review) RejectDraft(ctx context.Context, id, reviewer, note string) (DraftView, error) {
	return r.transition(ctx, id, "status = 'rejected', reviewer = $2, review_note = $3, reviewed_at = now()",
		"status in ('grounded', 'abstained', 'approved') and sent_at is null", reviewer, note)
}

// EditDraft replaces the subject and body of an unsent, unapproved draft and
// marks it edited (its faithfulness score described the original text).
func (r Review) EditDraft(ctx context.Context, id, reviewer, subject, body string) (DraftView, error) {
	return r.transition(ctx, id, "subject = $2, body = $3, edited = true, reviewer = $4, reviewed_at = now()",
		"status in ('grounded', 'abstained')", subject, body, reviewer)
}
