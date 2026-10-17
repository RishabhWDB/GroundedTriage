package pipeline

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/routing"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) Now() time.Time          { return c.t }
func (c *fakeClock) Advance(d time.Duration) { c.t = c.t.Add(d) }

// fakeRouteStore keeps routes in memory with the store's escalation semantics.
type fakeRouteStore struct {
	pending []store.RouteCandidate
	routes  []store.OpenRoute
	stored  []model.Route
}

func (f *fakeRouteStore) Unrouted(context.Context, int) ([]store.RouteCandidate, error) {
	out := f.pending
	f.pending = nil
	return out, nil
}

func (f *fakeRouteStore) Insert(_ context.Context, r model.Route) (int64, error) {
	f.stored = append(f.stored, r)
	if r.SLADueAt != nil {
		f.routes = append(f.routes, store.OpenRoute{ID: int64(len(f.stored)), TicketID: r.TicketID,
			Queue: r.Queue, Priority: r.Priority, SLADueAt: *r.SLADueAt})
	}
	return int64(len(f.stored)), nil
}

func (f *fakeRouteStore) Breached(_ context.Context, now time.Time, _ int) ([]store.OpenRoute, error) {
	var out []store.OpenRoute
	for _, o := range f.routes {
		if o.SLADueAt.Before(now) {
			out = append(out, o)
		}
	}
	return out, nil
}

func (f *fakeRouteStore) Escalate(_ context.Context, o store.OpenRoute, prio int, due, _ time.Time) (bool, error) {
	for i := range f.routes {
		if f.routes[i].ID == o.ID && f.routes[i].SLADueAt.Equal(o.SLADueAt) {
			f.routes[i].Priority, f.routes[i].SLADueAt = prio, due
			f.routes[i].Escalations++
			return true, nil
		}
	}
	return false, nil
}

func newRouter(t *testing.T, st *fakeRouteStore, clk *fakeClock, au *memAudit) *Router {
	t.Helper()
	rules, err := routing.LoadRules("../../config/routing_rules.yaml")
	if err != nil {
		t.Fatal(err)
	}
	sla, err := routing.LoadSLA("../../config/sla.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return &Router{Store: st, Rules: ruleSetEval{rules}, SLA: sla, Clock: clk, Audit: au}
}

type ruleSetEval struct{ rs *routing.RuleSet }

func (r ruleSetEval) Route(in routing.Input) model.Route { return r.rs.Evaluate(in) }

func classified(id, sender, subject, body, category, urgency string) store.RouteCandidate {
	return store.RouteCandidate{
		Ticket: model.Ticket{ID: id, Sender: sender, Subject: subject, Body: body, IngestStatus: model.IngestAccepted},
		Classification: &model.Classification{TicketID: id, Category: category, CategoryConfidence: 0.9,
			Tier: "fast", Urgency: urgency, UrgencySource: "heuristic"},
	}
}

func TestRoutePendingAppliesRulesNoReplyAndSLA(t *testing.T) {
	clk := &fakeClock{t: time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC)}
	st := &fakeRouteStore{pending: []store.RouteCandidate{
		classified("bill", "priya@acme.com", "Charged twice", "Refund please", "Billing", "high"),
		classified("news", "noreply@newsletter.io", "Weekly", "Top stories. Unsubscribe anytime.", "General Inquiry", "low"),
		{Ticket: model.Ticket{ID: "pt", IngestStatus: model.IngestSkipped, SkipReason: "non_english"}},
	}}
	au := &memAudit{}
	stats, err := newRouter(t, st, clk, au).RoutePending(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Routed != 3 || stats.ByAction["reply"] != 1 || stats.ByAction["mute"] != 1 || stats.ByAction["queue"] != 1 {
		t.Fatalf("stats %+v", stats)
	}
	byID := map[string]model.Route{}
	for _, r := range st.stored {
		byID[r.TicketID] = r
	}
	if b := byID["bill"]; b.Queue != "billing" || b.Priority != 2 || !b.SLADueAt.Equal(clk.t.Add(4*time.Hour)) {
		t.Fatalf("billing route %+v", b)
	}
	if n := byID["news"]; n.Action != "mute" || n.SLADueAt != nil || !strings.Contains(n.Reason, "no-reply sender") {
		t.Fatalf("newsletter route %+v", n)
	}
	if p := byID["pt"]; p.Queue != "human_review" || !strings.Contains(p.Reason, "non_english") || !p.SLADueAt.Equal(clk.t.Add(24*time.Hour)) {
		t.Fatalf("skipped route %+v", p)
	}
	if len(au.entries) != 1 || au.entries[0].Action != "route.run" {
		t.Fatalf("audit %+v", au.entries)
	}
	if again, _ := newRouter(t, st, clk, au).RoutePending(context.Background()); again.Routed != 0 {
		t.Fatal("nothing left to route")
	}
}

func TestEscalationWithFakeClock(t *testing.T) {
	clk := &fakeClock{t: time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC)}
	st := &fakeRouteStore{pending: []store.RouteCandidate{
		classified("gen", "a@b.com", "Question", "What are your hours?", "General Inquiry", "medium"), // P3, 24h
	}}
	au := &memAudit{}
	r := newRouter(t, st, clk, au)
	if _, err := r.RoutePending(context.Background()); err != nil {
		t.Fatal(err)
	}

	clk.Advance(23 * time.Hour) // not yet due
	if s, _ := r.EscalateBreached(context.Background()); s.Breached != 0 {
		t.Fatalf("escalated early: %+v", s)
	}
	clk.Advance(2 * time.Hour) // 25h: breached -> P2, due now + 4h
	s, err := r.EscalateBreached(context.Background())
	if err != nil || s.Escalated != 1 || s.AtTop != 0 {
		t.Fatalf("first escalation %+v %v", s, err)
	}
	if o := st.routes[0]; o.Priority != 2 || !o.SLADueAt.Equal(clk.t.Add(4*time.Hour)) || o.Escalations != 1 {
		t.Fatalf("after first escalation %+v", o)
	}
	last := au.entries[len(au.entries)-1]
	if last.Action != "sla.escalated" || last.EntityID != "gen" || last.Details["old_priority"] != 3 || last.Details["new_priority"] != 2 {
		t.Fatalf("audit %+v", last)
	}
	if s, _ := r.EscalateBreached(context.Background()); s.Breached != 0 {
		t.Fatal("re-escalated within the new deadline")
	}

	clk.Advance(5 * time.Hour) // P2 -> P1 (1h)
	r.EscalateBreached(context.Background())
	clk.Advance(2 * time.Hour) // P1 breached: re-armed, recorded
	s, _ = r.EscalateBreached(context.Background())
	if s.AtTop != 1 || st.routes[0].Priority != 1 || st.routes[0].Escalations != 3 {
		t.Fatalf("at top: %+v route %+v", s, st.routes[0])
	}
	if au.entries[len(au.entries)-1].Action != "sla.breached_p1" {
		t.Fatalf("audit %+v", au.entries[len(au.entries)-1])
	}
}

func TestEscalationSkipsRoutesChangedMeanwhile(t *testing.T) {
	clk := &fakeClock{t: time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC)}
	st := &fakeRouteStore{routes: []store.OpenRoute{{ID: 7, TicketID: "x", Priority: 4, SLADueAt: clk.t.Add(-time.Hour)}}}
	r := newRouter(t, st, clk, &memAudit{})
	// The store reports the breach, but the row's deadline moved before the update.
	st2 := &staleStore{fakeRouteStore: st}
	r.Store = st2
	s, err := r.EscalateBreached(context.Background())
	if err != nil || s.Breached != 1 || s.Escalated != 0 {
		t.Fatalf("stats %+v err %v", s, err)
	}
}

type staleStore struct{ *fakeRouteStore }

func (s *staleStore) Escalate(context.Context, store.OpenRoute, int, time.Time, time.Time) (bool, error) {
	return false, nil
}
