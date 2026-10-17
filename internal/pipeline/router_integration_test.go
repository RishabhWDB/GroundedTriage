//go:build integration

package pipeline_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/pipeline"
	"github.com/RishabhWDB/GroundedTriage/internal/routing"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

type clock struct{ t time.Time }

func (c *clock) Now() time.Time { return c.t }

func TestRouterRoutesAndEscalatesInDB(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "routes", "audit_log")
	ctx := context.Background()
	tickets := store.Tickets{Pool: pool}
	cls := store.Classifications{Pool: pool}
	mk := func(ext, sender, subject, status string) string {
		id, err := tickets.Insert(ctx, model.Ticket{Source: model.SourceCSV, ExternalID: ext, Sender: sender,
			Subject: subject, Body: subject + " body", ReceivedAt: time.Now(), ContentHash: "h-" + ext,
			IngestStatus: status, SkipReason: map[bool]string{true: "non_english"}[status == model.IngestSkipped]})
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	bill := mk("bill", "priya@acme.com", "Charged twice", model.IngestAccepted)
	news := mk("news", "noreply@newsletter.io", "Weekly digest", model.IngestAccepted)
	skipped := mk("pt", "lucas@example.com.br", "Problema", model.IngestSkipped)
	waiting := mk("wait", "x@y.com", "Not classified yet", model.IngestAccepted)
	for _, c := range []model.Classification{
		{TicketID: bill, Category: "Billing", CategoryConfidence: 0.9, Tier: "fast", Urgency: "high", UrgencySource: "heuristic"},
		{TicketID: news, Category: "General Inquiry", CategoryConfidence: 0.8, Tier: "fast", Urgency: "low", UrgencySource: "heuristic"},
	} {
		if err := cls.Insert(ctx, c); err != nil {
			t.Fatal(err)
		}
	}
	engine, err := routing.NewEngine("../../config/routing_rules.yaml")
	if err != nil {
		t.Fatal(err)
	}
	sla, _ := routing.LoadSLA("../../config/sla.yaml")
	clk := &clock{t: time.Now().UTC().Truncate(time.Second)}
	repo := store.Routes{Pool: pool}
	r := &pipeline.Router{Store: repo, Rules: engine, SLA: sla, Clock: clk, Audit: store.Audit{Pool: pool}}

	st, err := r.RoutePending(ctx)
	if err != nil || st.Routed != 3 {
		t.Fatalf("routed %+v err %v (want bill, news, skipped; not the unclassified one)", st, err)
	}
	if _, ok, _ := repo.Latest(ctx, waiting); ok {
		t.Fatal("unclassified accepted ticket was routed")
	}
	b, _, _ := repo.Latest(ctx, bill)
	if b.Action != "reply" || b.Queue != "billing" || b.Priority != 2 || b.SLADueAt == nil {
		t.Fatalf("billing %+v", b)
	}
	n, _, _ := repo.Latest(ctx, news)
	if n.Action != "mute" || n.SLADueAt != nil || !strings.Contains(n.Reason, "no-reply sender") {
		t.Fatalf("newsletter %+v", n)
	}
	s, _, _ := repo.Latest(ctx, skipped)
	if s.Queue != "human_review" || !strings.Contains(s.Reason, "non_english") {
		t.Fatalf("skipped %+v", s)
	}

	clk.t = clk.t.Add(5 * time.Hour) // billing P2 (4h) is breached; skipped P3 (24h) isn't
	es, err := r.EscalateBreached(ctx)
	if err != nil || es.Breached != 1 || es.Escalated != 1 {
		t.Fatalf("escalation %+v err %v", es, err)
	}
	b, _, _ = repo.Latest(ctx, bill)
	if b.Priority != 1 || !b.SLADueAt.Equal(clk.t.Add(time.Hour)) {
		t.Fatalf("after escalation %+v", b)
	}
	var audits int
	_ = pool.QueryRow(ctx, `select count(*) from audit_log where action = 'sla.escalated' and entity_id = $1`, bill).Scan(&audits)
	if audits != 1 {
		t.Fatalf("sla.escalated audit rows = %d", audits)
	}
	// The same breach can't be escalated twice by a stale read.
	stale := store.OpenRoute{ID: 0, SLADueAt: clk.t.Add(-time.Hour)}
	_ = pool.QueryRow(ctx, `select id from routes where ticket_id = $1`, bill).Scan(&stale.ID)
	if ok, err := repo.Escalate(ctx, stale, 1, clk.t, clk.t); ok || err != nil {
		t.Fatalf("stale escalate ok=%v err=%v", ok, err)
	}
}
