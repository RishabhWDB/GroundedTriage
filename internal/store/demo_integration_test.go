//go:build integration

package store_test

import (
	"context"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

func TestDemoResetDeletesOnlyDemoRows(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "routes", "reply_drafts", "meetings", "audit_log")
	ctx := context.Background()
	tickets := store.Tickets{Pool: pool}
	insert := func(source, extID string) string {
		id, err := tickets.Insert(ctx, model.Ticket{Source: source, ExternalID: extID, Sender: "a@x.com",
			Subject: extID, Body: extID + " body", ReceivedAt: time.Now(), ContentHash: storetest.UniqueID("h")})
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	demo1 := insert(model.SourceCSV, "demo-01")
	demo2 := insert(model.SourceCSV, "demo-02")
	keepCSV := insert(model.SourceCSV, "csv-123")    // a real CSV row
	keepZD := insert(model.SourceZendesk, "demo-77") // demo- prefix but not a CSV row
	keepLike := insert(model.SourceCSV, "xdemo-01")  // prefix must be at the start
	for _, id := range []string{demo1, demo2, keepCSV} {
		_ = store.Classifications{Pool: pool}.Insert(ctx, model.Classification{TicketID: id, Category: "Billing",
			Tier: "fast", Urgency: "low", UrgencySource: "heuristic"})
		_, _ = store.Routes{Pool: pool}.Insert(ctx, model.Route{TicketID: id, Action: "reply", Queue: "billing", Priority: 3})
	}
	draftID, _ := store.ReplyDrafts{Pool: pool}.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: demo1, Status: "grounded"}})
	audit := store.Audit{Pool: pool}
	_ = audit.Write(ctx, store.AuditEntry{EntityType: "ticket", EntityID: demo1, Action: "reply.grounded"})
	_ = audit.Write(ctx, store.AuditEntry{EntityType: "draft", EntityID: draftID, Action: "draft.approved",
		Details: map[string]any{"ticket_id": demo1}})
	_ = audit.Write(ctx, store.AuditEntry{EntityType: "draft", EntityID: draftID, Action: "reply.send_unknown"})
	_ = audit.Write(ctx, store.AuditEntry{EntityType: "ticket", EntityID: keepCSV, Action: "reply.grounded"})
	_ = audit.Write(ctx, store.AuditEntry{EntityType: "pipeline", EntityID: "classify", Action: "classify.run"})

	d := store.Demo{Pool: pool}
	rows, err := d.Rows(ctx)
	if err != nil || len(rows) != 2 || rows[0].ExternalID != "demo-01" || *rows[0].DraftStatus != "grounded" || *rows[1].Action != "reply" {
		t.Fatalf("rows %+v err %v", rows, err)
	}
	c, err := d.Counts(ctx)
	want := store.DemoCounts{Tickets: 2, Classifications: 2, Routes: 2, Drafts: 1, Meetings: 0, Audit: 3}
	if err != nil || c != want {
		t.Fatalf("counts %+v err %v, want %+v", c, err, want)
	}
	if _, err := d.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	var left []string
	r, _ := pool.Query(ctx, `select external_id from tickets order by external_id`)
	for r.Next() {
		var s string
		_ = r.Scan(&s)
		left = append(left, s)
	}
	r.Close()
	if len(left) != 3 || left[0] != "csv-123" || left[1] != "demo-77" || left[2] != "xdemo-01" {
		t.Fatalf("left %v", left)
	}
	var classifications, auditRows int
	_ = pool.QueryRow(ctx, `select count(*) from classifications`).Scan(&classifications)
	_ = pool.QueryRow(ctx, `select count(*) from audit_log`).Scan(&auditRows)
	if classifications != 1 || auditRows != 2 {
		t.Fatalf("classifications %d audit %d, want 1 and 2", classifications, auditRows)
	}
	_ = keepZD
	_ = keepLike
	if c, _ := d.Counts(ctx); c.Tickets != 0 {
		t.Fatalf("after reset %+v", c)
	}
}
