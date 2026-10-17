//go:build integration

package store_test

import (
	"context"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func TestMonitoringDayInputsAndDaily(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "routes", "reply_drafts", "stage_timings", "metrics_daily")
	ctx := context.Background()
	m := store.Monitoring{Pool: pool}
	now := time.Now().UTC()
	day := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)

	a := seedTicket(t, ctx, "a@x.com", "Server down")
	b := seedTicket(t, ctx, "b@x.com", "Charged twice")
	cls := store.Classifications{Pool: pool}
	// a is reclassified: only its latest category counts.
	_ = cls.Insert(ctx, model.Classification{TicketID: a, Category: "Billing", Tier: "fast", Urgency: "low", UrgencySource: "heuristic"})
	_ = cls.Insert(ctx, model.Classification{TicketID: a, Category: "Technical Issue", Tier: "deep", Urgency: "low", UrgencySource: "heuristic"})
	_ = cls.Insert(ctx, model.Classification{TicketID: b, Category: "Billing", Tier: "fast", Urgency: "low", UrgencySource: "heuristic"})

	past, future := now.Add(-time.Hour), now.Add(time.Hour)
	routes := store.Routes{Pool: pool}
	_, _ = routes.Insert(ctx, model.Route{TicketID: a, Action: "queue", Queue: "q", Priority: 1, SLADueAt: &past})
	_, _ = routes.Insert(ctx, model.Route{TicketID: b, Action: "reply", Queue: "q", Priority: 3, SLADueAt: &future})
	_, _ = routes.Insert(ctx, model.Route{TicketID: b, Action: "mute"}) // no SLA: not counted

	drafts := store.ReplyDrafts{Pool: pool}
	_, _ = drafts.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: a, Status: "abstained"}, AbstainReason: "floor"})
	_, _ = drafts.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: b, Status: "grounded"}})

	var ts []store.StageTiming
	for i := 1; i <= 20; i++ {
		ts = append(ts, store.StageTiming{Stage: "classify", Ms: float64(i * 10), OK: i != 20, At: now})
	}
	ts = append(ts, store.StageTiming{Stage: "classify", Ms: 5, OK: true, At: day.Add(-time.Minute)}) // yesterday
	if err := m.InsertTimings(ctx, ts); err != nil {
		t.Fatal(err)
	}

	in, err := m.DayInputs(ctx, day, now)
	if err != nil {
		t.Fatal(err)
	}
	if in.Categories["Technical Issue"] != 1 || in.Categories["Billing"] != 1 || len(in.Categories) != 2 {
		t.Fatalf("categories %v", in.Categories)
	}
	if in.Drafts != 2 || in.Abstained != 1 || in.SLARoutes != 2 || in.SLABreached != 1 {
		t.Fatalf("inputs %+v", in)
	}
	stages := map[string]store.StageStat{}
	for _, s := range in.Stages {
		stages[s.Stage] = s
	}
	c := stages["classify"]
	if c.N != 20 || c.Errors != 1 || c.P50 != 105 || c.P95 < 190 || c.P95 > 200 {
		t.Fatalf("classify %+v", c)
	}
	if stages["ingest_to_classified"].N != 3 || stages["ingest_to_draft"].N != 2 {
		t.Fatalf("stages %+v", in.Stages)
	}

	v1, v2 := 0.1, 0.25
	rows := []store.DailyMetric{{Day: day, Metric: "category_psi", Value: &v1, Detail: map[string]any{"alert": false}},
		{Day: day.AddDate(0, 0, -20), Metric: "category_psi", Value: &v1, Detail: map[string]any{}}}
	if err := m.UpsertDaily(ctx, rows); err != nil {
		t.Fatal(err)
	}
	rows[0].Value, rows[0].Detail = &v2, map[string]any{"alert": true}
	if err := m.UpsertDaily(ctx, rows[:1]); err != nil {
		t.Fatal(err)
	}
	got, err := m.Daily(ctx, 14, now)
	if err != nil || len(got) != 1 || *got[0].Value != 0.25 || got[0].Detail["alert"] != true || !got[0].Day.Equal(day) {
		t.Fatalf("daily %+v err %v", got, err)
	}
}
