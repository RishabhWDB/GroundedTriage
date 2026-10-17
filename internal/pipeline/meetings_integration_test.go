//go:build integration

package pipeline_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/pipeline"
	"github.com/RishabhWDB/GroundedTriage/internal/scheduler"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func TestMeetingSchedulerStoresOutcome(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "routes", "meetings", "audit_log")
	ctx := context.Background()
	id, err := store.Tickets{Pool: pool}.Insert(ctx, model.Ticket{Source: model.SourceCSV, ExternalID: "meet",
		Sender: "sarah@acme.com", Subject: "Q4 rollout", Body: "Can we meet Thursday at 3pm?",
		ReceivedAt: time.Date(2026, 10, 13, 9, 0, 0, 0, time.UTC), ContentHash: "h-meet"})
	if err != nil {
		t.Fatal(err)
	}
	other, _ := store.Tickets{Pool: pool}.Insert(ctx, model.Ticket{Source: model.SourceCSV, ExternalID: "bill",
		Body: "refund", ReceivedAt: time.Now(), ContentHash: "h-bill"})
	routes := store.Routes{Pool: pool}
	_, _ = routes.Insert(ctx, model.Route{TicketID: id, Action: "schedule", Queue: "meetings", Priority: 3})
	_, _ = routes.Insert(ctx, model.Route{TicketID: other, Action: "reply", Queue: "billing", Priority: 3})

	start := time.Date(2026, 10, 15, 15, 0, 0, 0, time.UTC)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req mlclient.MeetingRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if r.URL.Path != "/extract_meeting" || req.TicketID != id || req.DefaultTimezone != "UTC" {
			t.Errorf("request %s %+v", r.URL.Path, req)
		}
		_ = json.NewEncoder(w).Encode(mlclient.MeetingExtraction{TicketID: id, IsMeetingRequest: true, Start: &start,
			Timezone: "UTC", Venue: "video"})
	}))
	defer srv.Close()
	cal := &scheduler.FakeCalendar{}
	ms := &pipeline.MeetingScheduler{Store: store.Meetings{Pool: pool}, ML: mlclient.New(srv.URL), Timezone: "UTC",
		Audit: store.Audit{Pool: pool}, Scheduler: &scheduler.Scheduler{Calendar: cal, Config: scheduler.DefaultConfig(time.UTC),
			Now: func() time.Time { return time.Date(2026, 10, 13, 9, 5, 0, 0, time.UTC) }}}
	st, err := ms.SchedulePending(ctx)
	if err != nil || st.Pending != 1 || st.ByStatus["booked"] != 1 {
		t.Fatalf("stats %+v err %v", st, err)
	}
	m, ok, err := store.Meetings{Pool: pool}.Get(ctx, id)
	if err != nil || !ok || m.Status != "booked" || m.EventID != "fake-event-1" || !m.SlotStart.Equal(start) {
		t.Fatalf("meeting %+v ok=%v err=%v", m, ok, err)
	}
	if again, _ := ms.SchedulePending(ctx); again.Pending != 0 {
		t.Fatal("ticket scheduled twice")
	}
}
