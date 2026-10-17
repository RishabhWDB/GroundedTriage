//go:build integration

package store_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

func seedTicket(t *testing.T, ctx context.Context, sender, subject string) string {
	t.Helper()
	id, err := store.Tickets{Pool: testDB.Pool}.Insert(ctx, model.Ticket{Source: model.SourceCSV,
		ExternalID: storetest.UniqueID("rv"), Sender: sender, Subject: subject, Body: subject + " body",
		ReceivedAt: time.Now(), ContentHash: storetest.UniqueID("h")})
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func TestReviewQueriesAndDraftTransitions(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "routes", "reply_drafts", "meetings", "kb_chunks", "kb_documents")
	ctx := context.Background()
	rv := store.Review{Pool: pool}

	urgent := seedTicket(t, ctx, "a@x.com", "Server down")
	bill := seedTicket(t, ctx, "a@x.com", "Charged twice")
	unrouted := seedTicket(t, ctx, "b@y.com", "Hello")
	cls := store.Classifications{Pool: pool}
	_ = cls.Insert(ctx, model.Classification{TicketID: urgent, Category: "Technical Issue", CategoryConfidence: 0.9,
		Tier: "fast", Urgency: "critical", UrgencySource: "heuristic"})
	_ = cls.Insert(ctx, model.Classification{TicketID: bill, Category: "Billing", CategoryConfidence: 0.8,
		Tier: "deep", Urgency: "medium", UrgencySource: "model"})
	due := time.Now().Add(-time.Hour) // already breached
	routes := store.Routes{Pool: pool}
	_, _ = routes.Insert(ctx, model.Route{TicketID: urgent, Action: "queue", Queue: "human_review", Priority: 1, SLADueAt: &due})
	_, _ = routes.Insert(ctx, model.Route{TicketID: bill, Action: "reply", Queue: "billing", Priority: 3})

	_, _ = pool.Exec(ctx, `insert into kb_documents (id, title, url) values ('kb1', 'Refunds', 'https://help/refunds')`)
	_, _ = pool.Exec(ctx, `insert into kb_chunks (id, doc_id, ordinal, title, heading, text) values
		('kb1:0', 'kb1', 0, 'Refunds', 'Policy', 'Refunds take 5 days.')`)
	score := 1.0
	drafts := store.ReplyDrafts{Pool: pool}
	if _, err := drafts.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: bill, Subject: "Re: charge",
		Body: "Refunds take 5 days [1].", Citations: []string{"kb1:0"}, FaithfulnessScore: &score, Status: "grounded"},
		RetrievedChunkIDs: []string{"kb1:0", "kb9:9"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := drafts.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: urgent, Status: "abstained"},
		AbstainReason: "best retrieval score 0.4 < floor 0.7"}); err != nil {
		t.Fatal(err)
	}

	// Inbox order: priority 1 first; filters.
	list, total, err := rv.ListTickets(ctx, store.TicketFilter{})
	if err != nil || total != 3 || list[0].ID != urgent || list[2].ID != unrouted {
		t.Fatalf("list %+v total %d err %v", list, total, err)
	}
	if l, n, _ := rv.ListTickets(ctx, store.TicketFilter{Category: "Billing"}); n != 1 || l[0].ID != bill || *l[0].DraftStatus != "grounded" {
		t.Fatalf("category filter %+v", l)
	}
	if l, n, _ := rv.ListTickets(ctx, store.TicketFilter{Status: "unrouted"}); n != 1 || l[0].ID != unrouted {
		t.Fatalf("unrouted filter %+v", l)
	}
	if _, n, _ := rv.ListTickets(ctx, store.TicketFilter{Queue: "billing", Urgency: "medium", Status: "open"}); n != 1 {
		t.Fatalf("queue+urgency+status = %d", n)
	}

	// Detail and citations.
	det, err := rv.GetTicket(ctx, bill)
	if err != nil || det.Classification.Tier != "deep" || det.Route.Queue != "billing" || det.Draft == nil || det.Meeting != nil {
		t.Fatalf("detail %+v err %v", det, err)
	}
	ch, _ := rv.Chunks(ctx, det.Draft.RetrievedChunkIDs)
	if len(ch) != 1 || ch[0].URL != "https://help/refunds" || ch[0].Heading != "Policy" {
		t.Fatalf("chunks %+v", ch)
	}
	if _, err := rv.GetTicket(ctx, "00000000-0000-0000-0000-000000000000"); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("missing ticket err %v", err)
	}

	// Transitions.
	g := det.Draft.ID
	ab, _ := rv.LatestDraft(ctx, urgent)
	if _, err := rv.ApproveDraft(ctx, ab.ID, "agent", ""); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("approving an unedited abstained draft: %v", err)
	}
	if e, err := rv.EditDraft(ctx, ab.ID, "agent", "Re: outage", "We're on it."); err != nil || !e.Edited {
		t.Fatalf("edit abstained: %+v %v", e, err)
	}
	if a, err := rv.ApproveDraft(ctx, ab.ID, "agent", "manual reply"); err != nil || a.Status != "approved" || *a.IdempotencyKey != "draft-"+ab.ID {
		t.Fatalf("approve edited abstained: %+v %v", a, err)
	}
	if a, err := rv.ApproveDraft(ctx, g, "agent", "ok"); err != nil || a.Status != "approved" || a.Reviewer != "agent" {
		t.Fatalf("approve grounded: %+v %v", a, err)
	}
	if _, err := rv.EditDraft(ctx, g, "agent", "s", "b"); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("editing an approved draft: %v", err)
	}
	if r, err := rv.RejectDraft(ctx, g, "lead", "tone"); err != nil || r.Status != "rejected" {
		t.Fatalf("reject approved-unsent: %+v %v", r, err)
	}
	if rt, _, _ := routes.Latest(ctx, bill); rt.Queue != "human_review" || rt.Action != "queue" ||
		!strings.HasSuffix(rt.Reason, "; draft_rejected") {
		t.Fatalf("route after reject %+v", rt)
	}
	if _, err := rv.RejectDraft(ctx, g, "lead", "again"); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("rejecting twice: %v", err)
	}
	if _, err := rv.ApproveDraft(ctx, "00000000-0000-0000-0000-000000000000", "x", ""); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("missing draft: %v", err)
	}

	sum, err := rv.Summary(ctx, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if sum.Tickets != 3 || sum.ByCategory["unclassified"] != 1 || sum.ByTier["deep"] != 1 || sum.OpenRoutes != 2 ||
		sum.SLABreachedOpen != 1 || sum.ByAction["queue"] != 2 || *sum.AbstainRate != 0.5 || sum.TopSenders[0].Sender != "a@x.com" ||
		sum.TopSenders[0].Tickets != 2 || sum.DraftsByStatus["rejected"] != 1 {
		t.Fatalf("summary %+v", sum)
	}
}
