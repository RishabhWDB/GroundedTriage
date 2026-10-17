package pipeline

import (
	"context"
	"log/slog"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/routing"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// RouteStore persists routes (store.Routes).
type RouteStore interface {
	Unrouted(ctx context.Context, limit int) ([]store.RouteCandidate, error)
	Insert(ctx context.Context, r model.Route) (int64, error)
	Breached(ctx context.Context, now time.Time, limit int) ([]store.OpenRoute, error)
	Escalate(ctx context.Context, o store.OpenRoute, newPriority int, newDue, now time.Time) (bool, error)
}

// RuleEvaluator picks a route (routing.Engine or a *routing.RuleSet).
type RuleEvaluator interface {
	Route(in routing.Input) model.Route
}

// Router routes classified and skipped tickets (layer 3) and escalates
// open routes that breach their SLA.
type Router struct {
	Store     RouteStore
	Rules     RuleEvaluator
	SLA       routing.SLAPolicy
	Clock     routing.Clock
	Audit     AuditWriter
	BatchSize int // default 100
	Log       *slog.Logger
}

// RouteStats summarises one RoutePending run.
type RouteStats struct {
	Routed   int
	ByAction map[string]int
	ByQueue  map[string]int
}

// EscalateStats summarises one EscalateBreached run.
type EscalateStats struct {
	Breached, Escalated, AtTop int
}

func (r *Router) log() *slog.Logger {
	if r.Log != nil {
		return r.Log
	}
	return slog.Default()
}

func (r *Router) now() time.Time {
	if r.Clock != nil {
		return r.Clock.Now()
	}
	return time.Now()
}

func (r *Router) limit() int {
	if r.BatchSize > 0 {
		return r.BatchSize
	}
	return 100
}

// Decide builds the route for one candidate (rules, no-reply detection, SLA).
func (r *Router) Decide(c store.RouteCandidate, now time.Time) model.Route {
	in := routing.Input{Ticket: c.Ticket, Classification: c.Classification}
	in.NoReply, in.NoReplyReason = routing.DetectNoReply(c.Ticket)
	rt := r.Rules.Route(in)
	r.SLA.Apply(&rt, now)
	return rt
}

// RoutePending routes up to BatchSize unrouted tickets.
func (r *Router) RoutePending(ctx context.Context) (RouteStats, error) {
	st := RouteStats{ByAction: map[string]int{}, ByQueue: map[string]int{}}
	cands, err := r.Store.Unrouted(ctx, r.limit())
	if err != nil || len(cands) == 0 {
		return st, err
	}
	muted := map[string]string{}
	for _, c := range cands {
		rt := r.Decide(c, r.now())
		if _, err := r.Store.Insert(ctx, rt); err != nil {
			return st, err
		}
		st.Routed++
		st.ByAction[rt.Action]++
		st.ByQueue[rt.Queue]++
		if rt.Action == "mute" && len(muted) < maxAuditIDs {
			muted[rt.TicketID] = rt.Reason
		}
	}
	r.log().Info("route: run", "routed", st.Routed, "by_action", st.ByAction)
	r.audit(ctx, store.AuditEntry{EntityType: "pipeline", EntityID: "route", Action: "route.run",
		Details: map[string]any{"routed": st.Routed, "by_action": st.ByAction, "by_queue": st.ByQueue,
			"muted_with_reason": muted}})
	return st, nil
}

// EscalateBreached raises every open route past its deadline one priority
// level and re-arms its SLA; each escalation is audited.
func (r *Router) EscalateBreached(ctx context.Context) (EscalateStats, error) {
	var st EscalateStats
	now := r.now()
	open, err := r.Store.Breached(ctx, now, r.limit())
	if err != nil {
		return st, err
	}
	st.Breached = len(open)
	for _, o := range open {
		e := r.SLA.Escalate(o.Priority, now)
		ok, err := r.Store.Escalate(ctx, o, e.NewPriority, e.NewDue, now)
		if err != nil {
			return st, err
		}
		if !ok { // resolved or escalated meanwhile
			continue
		}
		st.Escalated++
		action := "sla.escalated"
		if e.AtTop {
			st.AtTop++
			action = "sla.breached_p1"
		}
		r.audit(ctx, store.AuditEntry{EntityType: "ticket", EntityID: o.TicketID, Action: action,
			Details: map[string]any{"route_id": o.ID, "queue": o.Queue, "old_priority": e.OldPriority,
				"new_priority": e.NewPriority, "was_due": o.SLADueAt, "new_due": e.NewDue,
				"escalations": o.Escalations + 1}})
	}
	if st.Breached > 0 {
		r.log().Info("sla: escalation run", "breached", st.Breached, "escalated", st.Escalated, "at_p1", st.AtTop)
	}
	return st, nil
}

func (r *Router) audit(ctx context.Context, e store.AuditEntry) {
	if r.Audit == nil {
		return
	}
	if err := r.Audit.Write(ctx, e); err != nil {
		r.log().Warn("route: audit write failed", "action", e.Action, "err", err)
	}
}
