package store

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Routes is the repository for the routes table (one current route per
// ticket; escalation updates it in place and is recorded in audit_log).
type Routes struct{ Pool *pgxpool.Pool }

// RouteCandidate is a ticket ready to be routed: skipped tickets (no
// classification) and accepted tickets with a classification.
type RouteCandidate struct {
	Ticket         model.Ticket
	Classification *model.Classification
}

// OpenRoute is an open route with an SLA deadline.
type OpenRoute struct {
	ID          int64
	TicketID    string
	Queue       string
	Priority    int
	SLADueAt    time.Time
	Escalations int
}

// Insert stores r and returns its id.
func (r Routes) Insert(ctx context.Context, rt model.Route) (int64, error) {
	var id int64
	err := r.Pool.QueryRow(ctx, `
		insert into routes (ticket_id, action, queue, priority, sla_due_at, reason)
		values ($1, $2, $3, $4, $5, $6) returning id`,
		rt.TicketID, rt.Action, rt.Queue, rt.Priority, rt.SLADueAt, rt.Reason).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("insert route for %s: %w", rt.TicketID, err)
	}
	return id, nil
}

// Latest returns the newest route of a ticket; ok is false when it has none.
func (r Routes) Latest(ctx context.Context, ticketID string) (rt model.Route, ok bool, err error) {
	rows, err := r.Pool.Query(ctx, `
		select ticket_id::text, action, queue, priority, sla_due_at, reason
		from routes where ticket_id = $1 order by created_at desc, id desc limit 1`, ticketID)
	if err != nil {
		return rt, false, err
	}
	defer rows.Close()
	if !rows.Next() {
		return rt, false, rows.Err()
	}
	err = rows.Scan(&rt.TicketID, &rt.Action, &rt.Queue, &rt.Priority, &rt.SLADueAt, &rt.Reason)
	return rt, err == nil, err
}

// Unrouted returns up to limit tickets without a route, oldest first:
// skipped tickets, and accepted tickets that have a classification (their
// latest one). Accepted tickets still waiting for classification are left.
func (r Routes) Unrouted(ctx context.Context, limit int) ([]RouteCandidate, error) {
	rows, err := r.Pool.Query(ctx, `
		select t.id::text, t.source, t.external_id, t.sender, t.subject, t.body, t.received_at,
		       t.thread_id, t.raw_ref, t.content_hash, t.ingest_status, t.skip_reason,
		       c.category, c.category_confidence, c.tier, c.low_confidence, c.urgency,
		       c.urgency_confidence, c.urgency_source, c.models_loaded, c.model_versions,
		       c.spam_score, c.possible_spam
		from tickets t
		left join lateral (
		    select * from classifications c where c.ticket_id = t.id
		    order by c.created_at desc, c.id desc limit 1
		) c on true
		where not exists (select 1 from routes r where r.ticket_id = t.id)
		  and (t.ingest_status = 'skipped' or c.id is not null)
		order by t.received_at, t.id
		limit $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("unrouted tickets: %w", err)
	}
	defer rows.Close()
	var out []RouteCandidate
	for rows.Next() {
		var t model.Ticket
		var (
			category, tier, urgency, urgencySource *string
			catConf, urgConf, spamScore            *float64
			lowConf, possibleSpam                  *bool
			loaded, versions                       []byte
		)
		if err := rows.Scan(&t.ID, &t.Source, &t.ExternalID, &t.Sender, &t.Subject, &t.Body,
			&t.ReceivedAt, &t.ThreadID, &t.RawRef, &t.ContentHash, &t.IngestStatus, &t.SkipReason,
			&category, &catConf, &tier, &lowConf, &urgency, &urgConf, &urgencySource,
			&loaded, &versions, &spamScore, &possibleSpam); err != nil {
			return nil, fmt.Errorf("scan unrouted: %w", err)
		}
		cand := RouteCandidate{Ticket: t}
		if category != nil {
			c := &model.Classification{TicketID: t.ID, Category: *category, CategoryConfidence: *catConf,
				Tier: *tier, LowConfidence: *lowConf, Urgency: *urgency, UrgencyConfidence: *urgConf,
				UrgencySource: *urgencySource, SpamScore: spamScore, PossibleSpam: *possibleSpam}
			_ = json.Unmarshal(loaded, &c.ModelsLoaded)
			_ = json.Unmarshal(versions, &c.ModelVersions)
			cand.Classification = c
		}
		out = append(out, cand)
	}
	return out, rows.Err()
}

// Breached returns open routes whose deadline is before now, most overdue first.
func (r Routes) Breached(ctx context.Context, now time.Time, limit int) ([]OpenRoute, error) {
	rows, err := r.Pool.Query(ctx, `
		select id, ticket_id::text, queue, priority, sla_due_at, escalations
		from routes where status = 'open' and sla_due_at is not null and sla_due_at < $1
		order by sla_due_at limit $2`, now, limit)
	if err != nil {
		return nil, fmt.Errorf("breached routes: %w", err)
	}
	defer rows.Close()
	var out []OpenRoute
	for rows.Next() {
		var o OpenRoute
		if err := rows.Scan(&o.ID, &o.TicketID, &o.Queue, &o.Priority, &o.SLADueAt, &o.Escalations); err != nil {
			return nil, err
		}
		out = append(out, o)
	}
	return out, rows.Err()
}

// Escalate sets a breached route's new priority and deadline. It only
// updates when the route is still open with the deadline the caller saw, so
// two escalation runs can't escalate the same breach twice. ok reports
// whether the row was updated.
func (r Routes) Escalate(ctx context.Context, o OpenRoute, newPriority int, newDue, now time.Time) (bool, error) {
	tag, err := r.Pool.Exec(ctx, `
		update routes set priority = $2, sla_due_at = $3, escalations = escalations + 1,
		                  escalated_at = $4, updated_at = $4
		where id = $1 and status = 'open' and sla_due_at = $5`,
		o.ID, newPriority, newDue, now, o.SLADueAt)
	if err != nil {
		return false, fmt.Errorf("escalate route %d: %w", o.ID, err)
	}
	return tag.RowsAffected() == 1, nil
}

// Reassign moves a ticket's current route to another queue (e.g. an abstained
// draft goes to human review), keeping its priority and SLA and appending the
// reason. It returns false when the ticket has no route.
func (r Routes) Reassign(ctx context.Context, ticketID, queue, reason string) (bool, error) {
	tag, err := r.Pool.Exec(ctx, `
		update routes set queue = $2, action = 'queue', reason = reason || '; ' || $3, updated_at = now()
		where id = (select id from routes where ticket_id = $1 order by created_at desc, id desc limit 1)`,
		ticketID, queue, reason)
	if err != nil {
		return false, fmt.Errorf("reassign route of %s: %w", ticketID, err)
	}
	return tag.RowsAffected() == 1, nil
}
