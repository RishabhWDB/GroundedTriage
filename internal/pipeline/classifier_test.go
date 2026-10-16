package pipeline

import (
	"context"
	"errors"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

type fakeClsStore struct {
	pending  []model.Ticket
	inserted []model.Classification
	limit    int
}

func (f *fakeClsStore) Unclassified(_ context.Context, limit int) ([]model.Ticket, error) {
	f.limit = limit
	return f.pending, nil
}

func (f *fakeClsStore) Insert(_ context.Context, c model.Classification) error {
	f.inserted = append(f.inserted, c)
	return nil
}

type fakeML struct {
	fail  map[string]error
	calls []string
	reply func(req mlclient.ClassifyRequest) model.Classification
}

func (f *fakeML) Classify(_ context.Context, req mlclient.ClassifyRequest) (model.Classification, error) {
	f.calls = append(f.calls, req.TicketID)
	if err := f.fail[req.TicketID]; err != nil {
		return model.Classification{}, err
	}
	if f.reply != nil {
		return f.reply(req), nil
	}
	return model.Classification{TicketID: req.TicketID, Category: "Billing", Tier: "fast",
		Urgency: "medium", UrgencySource: "heuristic"}, nil
}

func tickets(ids ...string) []model.Ticket {
	out := make([]model.Ticket, len(ids))
	for i, id := range ids {
		out[i] = model.Ticket{ID: id, Subject: "s-" + id, Body: "b-" + id}
	}
	return out
}

func TestClassifyPendingStoresResults(t *testing.T) {
	st := &fakeClsStore{pending: tickets("a", "b")}
	ml := &fakeML{}
	audit := &memAudit{}
	c := &Classifier{Store: st, ML: ml, Audit: audit}
	stats, err := c.ClassifyPending(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats != (ClassifyStats{Pending: 2, Classified: 2}) {
		t.Fatalf("stats = %+v", stats)
	}
	if st.limit != 100 || len(st.inserted) != 2 || st.inserted[1].TicketID != "b" {
		t.Fatalf("limit %d inserted %+v", st.limit, st.inserted)
	}
	if len(audit.entries) != 1 || audit.entries[0].Action != "classify.run" {
		t.Fatalf("audit = %+v", audit.entries)
	}
	d := audit.entries[0].Details
	if d["classified"] != 2 || d["by_category"].(map[string]int)["Billing"] != 2 {
		t.Fatalf("audit details = %+v", d)
	}
}

func TestClassifyPendingNothingToDo(t *testing.T) {
	audit := &memAudit{}
	stats, err := (&Classifier{Store: &fakeClsStore{}, ML: &fakeML{}, Audit: audit}).ClassifyPending(context.Background())
	if err != nil || stats.Pending != 0 || len(audit.entries) != 0 {
		t.Fatalf("stats %+v err %v audit %d", stats, err, len(audit.entries))
	}
}

func TestClassifyPendingSkipsFailuresAndBadAnswers(t *testing.T) {
	st := &fakeClsStore{pending: tickets("a", "b", "c", "d")}
	ml := &fakeML{fail: map[string]error{"a": errors.New("boom")}}
	ml.reply = func(req mlclient.ClassifyRequest) model.Classification {
		switch req.TicketID {
		case "b":
			return model.Classification{TicketID: "other", Category: "Billing"}
		case "c":
			return model.Classification{TicketID: "c", Category: "Complaint"}
		}
		return model.Classification{TicketID: req.TicketID, Category: "Spam"}
	}
	stats, err := (&Classifier{Store: st, ML: ml, MaxConsecutiveFailures: 5}).ClassifyPending(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Classified != 1 || stats.Failed != 3 || stats.Stopped {
		t.Fatalf("stats = %+v", stats)
	}
	if len(st.inserted) != 1 || st.inserted[0].TicketID != "d" {
		t.Fatalf("inserted = %+v", st.inserted)
	}
}

func TestClassifyPendingStopsWhenMLDown(t *testing.T) {
	down := errors.New("connection refused")
	st := &fakeClsStore{pending: tickets("a", "b", "c", "d", "e")}
	ml := &fakeML{fail: map[string]error{"a": down, "b": down, "c": down, "d": down, "e": down}}
	stats, err := (&Classifier{Store: st, ML: ml, MaxConsecutiveFailures: 2, BatchSize: 5}).ClassifyPending(context.Background())
	if !errors.Is(err, ErrMLUnavailable) {
		t.Fatalf("err = %v", err)
	}
	if !stats.Stopped || stats.Failed != 2 || len(ml.calls) != 2 || st.limit != 5 {
		t.Fatalf("stats %+v calls %v limit %d", stats, ml.calls, st.limit)
	}
}
