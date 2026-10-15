package pipeline

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

type memCheckpoints struct {
	mu sync.Mutex
	m  map[string]time.Time
}

func (c *memCheckpoints) Get(_ context.Context, s string) (time.Time, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t, ok := c.m[s]
	return t, ok, nil
}

func (c *memCheckpoints) Set(_ context.Context, s string, t time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[s] = t
	return nil
}

type memAudit struct {
	mu      sync.Mutex
	entries []store.AuditEntry
}

func (a *memAudit) Write(_ context.Context, e store.AuditEntry) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.entries = append(a.entries, e)
	return nil
}

// setIngestor inserts each (source, external_id) once.
type setIngestor struct {
	mu   sync.Mutex
	seen map[string]bool
}

func (s *setIngestor) Process(_ context.Context, ts []model.Ticket) ([]preprocess.Result, preprocess.Stats, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var res []preprocess.Result
	var st preprocess.Stats
	for _, t := range ts {
		k := t.Source + "/" + t.ExternalID
		if s.seen[k] {
			res = append(res, preprocess.Result{Ticket: t, Outcome: preprocess.OutcomeDuplicateID})
			st.DuplicateID++
			continue
		}
		s.seen[k] = true
		t.ID = k
		res = append(res, preprocess.Result{Ticket: t, Outcome: preprocess.OutcomeInserted})
		st.Inserted++
	}
	return res, st, nil
}

var t0 = time.Date(2026, 10, 1, 9, 0, 0, 0, time.UTC)

func tk(id string, at time.Time) model.Ticket {
	return model.Ticket{Source: "csv", ExternalID: id, ReceivedAt: at}
}

func newPoller(srcs ingest.Sources) (*Poller, *memCheckpoints, *memAudit) {
	cp := &memCheckpoints{m: map[string]time.Time{}}
	au := &memAudit{}
	return &Poller{Sources: srcs, Ingestor: &setIngestor{seen: map[string]bool{}},
		Checkpoints: cp, Audit: au, Overlap: 10 * time.Minute}, cp, au
}

func TestPollOnceAdvancesCheckpointAndIsIdempotent(t *testing.T) {
	src := ingest.NewFakeSource("csv", tk("a", t0), tk("b", t0.Add(time.Hour)))
	p, cp, au := newPoller(ingest.Sources{"csv": src})
	ctx := context.Background()

	r := p.PollOnce(ctx)["csv"]
	if r.Err != nil || r.Fetched != 2 || r.Stats.Inserted != 2 {
		t.Fatalf("first poll = %+v", r)
	}
	if got := cp.m["csv"]; !got.Equal(t0.Add(time.Hour)) {
		t.Fatalf("checkpoint = %v", got)
	}
	// Second poll: overlap re-reads "b" (within 10 min of the checkpoint), but nothing is new.
	r = p.PollOnce(ctx)["csv"]
	if r.Err != nil || r.Stats.Inserted != 0 || r.Fetched != 1 || r.Stats.DuplicateID != 1 {
		t.Fatalf("second poll = %+v", r)
	}
	// A late item inside the overlap window is still picked up.
	src.Add(tk("late", t0.Add(55*time.Minute)), tk("c", t0.Add(2*time.Hour)))
	r = p.PollOnce(ctx)["csv"]
	if r.Stats.Inserted != 2 || !cp.m["csv"].Equal(t0.Add(2*time.Hour)) {
		t.Fatalf("third poll = %+v checkpoint=%v", r, cp.m["csv"])
	}
	if len(au.entries) != 3 || au.entries[0].Action != "ingest.poll" ||
		au.entries[0].Details["inserted"] != 2 || len(au.entries[0].Details["inserted_ticket_ids"].([]string)) != 2 {
		t.Fatalf("audit = %+v", au.entries)
	}
}

func TestFailingSourceDoesNotBlockOthers(t *testing.T) {
	bad := ingest.NewFakeSource("email")
	bad.FetchErr = errors.New("imap auth failed")
	good := ingest.NewFakeSource("csv", tk("a", t0))
	p, cp, au := newPoller(ingest.Sources{"email": bad, "csv": good})

	res := p.PollOnce(context.Background())
	if res["email"].Err == nil || res["csv"].Err != nil || res["csv"].Stats.Inserted != 1 {
		t.Fatalf("results = %+v", res)
	}
	if _, ok := cp.m["email"]; ok {
		t.Fatal("failing source's checkpoint advanced")
	}
	var sawError bool
	for _, e := range au.entries {
		sawError = sawError || (e.Action == "ingest.error" && e.EntityID == "email")
	}
	if !sawError {
		t.Fatalf("no ingest.error audit: %+v", au.entries)
	}
}

func TestRunStopsOnCancel(t *testing.T) {
	src := ingest.NewFakeSource("csv", tk("a", t0))
	p, _, au := newPoller(ingest.Sources{"csv": src})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { p.Run(ctx, 10*time.Millisecond); close(done) }()
	time.Sleep(35 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Run did not return after cancel")
	}
	au.mu.Lock()
	defer au.mu.Unlock()
	if len(au.entries) < 2 {
		t.Fatalf("expected several polls, got %d", len(au.entries))
	}
}
