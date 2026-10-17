//go:build integration

package outbound_test

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/outbound"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

var testDB *storetest.DB

func TestMain(m *testing.M) { os.Exit(storetest.Run(m, &testDB)) }

// Two senders racing over the same approved drafts in the real DB send each exactly once.
func TestConcurrentSendersSendEachApprovalOnce(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "routes", "reply_drafts", "audit_log")
	ctx := context.Background()
	review := store.Review{Pool: pool}
	var drafts []string
	for i := 0; i < 6; i++ {
		id, err := store.Tickets{Pool: pool}.Insert(ctx, model.Ticket{Source: "csv", ExternalID: storetest.UniqueID("o"),
			Sender: "a@b.com", Subject: "s", Body: "b", ReceivedAt: time.Now(), ContentHash: storetest.UniqueID("h")})
		if err != nil {
			t.Fatal(err)
		}
		_, _ = store.Routes{Pool: pool}.Insert(ctx, model.Route{TicketID: id, Action: "reply", Queue: "billing", Priority: 3})
		did, err := store.ReplyDrafts{Pool: pool}.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: id,
			Subject: "Re", Body: "Hello [1]", Status: "grounded"}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := review.ApproveDraft(ctx, did, "ann", ""); err != nil {
			t.Fatal(err)
		}
		drafts = append(drafts, did)
	}
	src := ingest.NewFakeSource("csv")
	var wg sync.WaitGroup
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := &outbound.Sender{Queue: store.Outbound{Pool: pool}, Sources: ingest.Sources{"csv": src},
				Audit: store.Audit{Pool: pool}, BatchSize: 2}
			for i := 0; i < 5; i++ {
				if _, err := s.SendApproved(ctx); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
	sent := map[string]int{}
	for _, r := range src.Replies() {
		sent[r.Draft.IdempotencyKey]++
	}
	if len(sent) != 6 {
		t.Fatalf("sent %d distinct drafts", len(sent))
	}
	for k, n := range sent {
		if n != 1 {
			t.Fatalf("%s sent %d times", k, n)
		}
	}
	var sentRows, resolved int
	_ = pool.QueryRow(ctx, `select count(*) from reply_drafts where status = 'sent' and sent_at is not null and send_started_at is null`).Scan(&sentRows)
	_ = pool.QueryRow(ctx, `select count(*) from routes where status = 'resolved'`).Scan(&resolved)
	if sentRows != 6 || resolved != 6 {
		t.Fatalf("sent rows %d, resolved routes %d", sentRows, resolved)
	}
	d, _ := review.GetDraft(ctx, drafts[0])
	if d.Status != "sent" || *d.IdempotencyKey != "draft-"+drafts[0] {
		t.Fatalf("draft %+v", d)
	}

	// A stale claim (sender died mid-send) is flagged and never resent.
	id, _ := store.Tickets{Pool: pool}.Insert(ctx, model.Ticket{Source: "csv", ExternalID: storetest.UniqueID("o"),
		Sender: "a@b.com", Body: "b", ReceivedAt: time.Now(), ContentHash: storetest.UniqueID("h")})
	did, _ := store.ReplyDrafts{Pool: pool}.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: id,
		Body: "x", Status: "grounded"}})
	_, _ = review.ApproveDraft(ctx, did, "ann", "")
	_, _ = pool.Exec(ctx, `update reply_drafts set send_started_at = now() - interval '1 hour', send_attempts = 1 where id = $1`, did)
	s := &outbound.Sender{Queue: store.Outbound{Pool: pool}, Sources: ingest.Sources{"csv": src}}
	st, err := s.SendApproved(ctx)
	if err != nil || st.Stale != 1 || st.Claimed != 0 {
		t.Fatalf("stale run %+v %v", st, err)
	}
	if again, _ := s.SendApproved(ctx); again.Claimed != 0 || again.Stale != 0 {
		t.Fatalf("second run %+v", again)
	}
	if n := len(src.Replies()); n != 6 {
		t.Fatalf("replies %d", n)
	}
}
