package outbound

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// memQueue mirrors store.Outbound's rules in memory.
type memQueue struct {
	mu     sync.Mutex
	drafts map[string]*row
}

type row struct {
	c                   store.Claimed
	status              string
	claimedAt           *time.Time
	attempts            int
	sendErr             string
	routeResolved, sent bool
}

func newQueue(ids ...string) *memQueue {
	q := &memQueue{drafts: map[string]*row{}}
	for _, id := range ids {
		q.drafts[id] = &row{status: "approved", c: store.Claimed{DraftID: id,
			Ticket: model.Ticket{ID: "t-" + id, Source: "csv", Sender: "a@b.com"},
			Draft:  model.ReplyDraft{TicketID: "t-" + id, Body: "Hi [1]", Status: "approved", IdempotencyKey: "draft-" + id}}}
	}
	return q
}

func (q *memQueue) Claim(_ context.Context, now time.Time, limit, maxAttempts int) ([]store.Claimed, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	var out []store.Claimed
	for _, r := range q.drafts {
		if len(out) >= limit {
			break
		}
		if r.status == "approved" && !r.sent && r.claimedAt == nil && r.attempts < maxAttempts &&
			!strings.HasPrefix(r.sendErr, store.UnknownOutcome) {
			t := now
			r.claimedAt, r.attempts = &t, r.attempts+1
			c := r.c
			c.Attempt = r.attempts
			out = append(out, c)
		}
	}
	return out, nil
}

func (q *memQueue) MarkSent(_ context.Context, id string, _ time.Time) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	r := q.drafts[id]
	r.status, r.sent, r.claimedAt, r.sendErr, r.routeResolved = "sent", true, nil, "", true
	return nil
}

func (q *memQueue) MarkFailed(_ context.Context, id, msg string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.drafts[id].sendErr, q.drafts[id].claimedAt = msg, nil
	return nil
}

func (q *memQueue) FlagStale(_ context.Context, before time.Time) ([]string, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	var ids []string
	for id, r := range q.drafts {
		if r.status == "approved" && !r.sent && r.claimedAt != nil && r.claimedAt.Before(before) &&
			!strings.HasPrefix(r.sendErr, store.UnknownOutcome) {
			r.sendErr = store.UnknownOutcome + ": check the channel"
			ids = append(ids, id)
		}
	}
	return ids, nil
}

type memAudit struct {
	mu      sync.Mutex
	entries []store.AuditEntry
}

func (m *memAudit) Write(_ context.Context, e store.AuditEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, e)
	return nil
}

func TestExactlyOneSendPerApproval(t *testing.T) {
	src := ingest.NewFakeSource("csv")
	q := newQueue("d1", "d2")
	au := &memAudit{}
	s := &Sender{Queue: q, Sources: ingest.Sources{"csv": src}, Audit: au}
	for i := 0; i < 3; i++ { // repeated worker runs
		if _, err := s.SendApproved(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	got := src.Replies()
	if len(got) != 2 {
		t.Fatalf("sent %d replies for 2 approvals", len(got))
	}
	for _, r := range got {
		if !strings.HasPrefix(r.Draft.IdempotencyKey, "draft-") || r.Ticket.Source != "csv" {
			t.Fatalf("reply %+v", r)
		}
	}
	if !q.drafts["d1"].sent || !q.drafts["d1"].routeResolved {
		t.Fatal("draft not marked sent / route not resolved")
	}
	if len(au.entries) != 2 || au.entries[0].Action != "reply.sent" {
		t.Fatalf("audit %+v", au.entries)
	}
}

func TestFailedSendIsRetriedThenSentOnce(t *testing.T) {
	src := ingest.NewFakeSource("csv")
	src.ReplyErr = errors.New("smtp: 421 try later")
	q := newQueue("d1")
	s := &Sender{Queue: q, Sources: ingest.Sources{"csv": src}, Audit: &memAudit{}}
	st, _ := s.SendApproved(context.Background())
	if st.Failed != 1 || q.drafts["d1"].sendErr == "" || q.drafts["d1"].claimedAt != nil {
		t.Fatalf("after failure: %+v %+v", st, q.drafts["d1"])
	}
	src.ReplyErr = nil
	st, _ = s.SendApproved(context.Background())
	if st.Sent != 1 || len(src.Replies()) != 1 || q.drafts["d1"].attempts != 2 {
		t.Fatalf("after retry: %+v replies %d", st, len(src.Replies()))
	}
}

func TestAttemptsAreCapped(t *testing.T) {
	src := ingest.NewFakeSource("csv")
	src.ReplyErr = errors.New("down")
	q := newQueue("d1")
	s := &Sender{Queue: q, Sources: ingest.Sources{"csv": src}, MaxAttempts: 2}
	for i := 0; i < 5; i++ {
		_, _ = s.SendApproved(context.Background())
	}
	if q.drafts["d1"].attempts != 2 {
		t.Fatalf("attempts %d, want 2", q.drafts["d1"].attempts)
	}
}

func TestStaleClaimIsNeverResent(t *testing.T) {
	src := ingest.NewFakeSource("csv")
	q := newQueue("d1")
	past := time.Now().Add(-time.Hour)
	q.drafts["d1"].claimedAt, q.drafts["d1"].attempts = &past, 1 // a sender died mid-send
	au := &memAudit{}
	s := &Sender{Queue: q, Sources: ingest.Sources{"csv": src}, Audit: au, Lease: 10 * time.Minute}
	for i := 0; i < 2; i++ {
		st, _ := s.SendApproved(context.Background())
		if i == 0 && st.Stale != 1 {
			t.Fatalf("stale = %d", st.Stale)
		}
	}
	if len(src.Replies()) != 0 || !strings.HasPrefix(q.drafts["d1"].sendErr, store.UnknownOutcome) {
		t.Fatalf("resent or not flagged: replies %d, err %q", len(src.Replies()), q.drafts["d1"].sendErr)
	}
	if au.entries[0].Action != "reply.send_unknown" {
		t.Fatalf("audit %+v", au.entries)
	}
}

func TestMissingSourceIsAFailure(t *testing.T) {
	q := newQueue("d1")
	q.drafts["d1"].c.Ticket.Source = "zendesk"
	st, _ := (&Sender{Queue: q, Sources: ingest.Sources{}}).SendApproved(context.Background())
	if st.Failed != 1 || !strings.Contains(q.drafts["d1"].sendErr, `no enabled "zendesk" source`) {
		t.Fatalf("%+v %q", st, q.drafts["d1"].sendErr)
	}
}
