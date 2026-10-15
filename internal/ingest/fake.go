package ingest

import (
	"context"
	"sync"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// FakeSource is an in-memory Source for tests. Fetch returns the tickets
// received after since; Reply records each call.
type FakeSource struct {
	SourceName string
	FetchErr   error
	ReplyErr   error

	mu      sync.Mutex
	tickets []model.Ticket
	replies []FakeReply
}

// FakeReply is one recorded Reply call.
type FakeReply struct {
	Ticket model.Ticket
	Draft  model.ReplyDraft
}

// NewFakeSource returns a FakeSource named name holding tickets.
func NewFakeSource(name string, tickets ...model.Ticket) *FakeSource {
	return &FakeSource{SourceName: name, tickets: tickets}
}

func (f *FakeSource) Name() string { return f.SourceName }

// Add appends tickets that later Fetch calls can return.
func (f *FakeSource) Add(tickets ...model.Ticket) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tickets = append(f.tickets, tickets...)
}

func (f *FakeSource) Fetch(ctx context.Context, since time.Time) ([]model.Ticket, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if f.FetchErr != nil {
		return nil, f.FetchErr
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []model.Ticket
	for _, t := range f.tickets {
		if t.ReceivedAt.After(since) {
			out = append(out, t)
		}
	}
	return out, nil
}

func (f *FakeSource) Reply(ctx context.Context, t model.Ticket, d model.ReplyDraft) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if f.ReplyErr != nil {
		return f.ReplyErr
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.replies = append(f.replies, FakeReply{Ticket: t, Draft: d})
	return nil
}

// Replies returns a copy of the recorded Reply calls.
func (f *FakeSource) Replies() []FakeReply {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]FakeReply(nil), f.replies...)
}
