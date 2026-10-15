//go:build integration

package store_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

func TestTicketInsertAndGet(t *testing.T) {
	pool := testDB.Require(t)
	ctx := context.Background()
	repo := store.Tickets{Pool: pool}

	thread := "thread-1"
	in := model.Ticket{
		Source:      model.SourceCSV,
		ExternalID:  storetest.UniqueID("row"),
		Sender:      "alice@example.com",
		Subject:     "Invoice question",
		Body:        "Why was I charged twice?",
		ReceivedAt:  time.Date(2026, 10, 15, 9, 30, 0, 0, time.UTC),
		ThreadID:    &thread,
		RawRef:      "sample.csv#1",
		ContentHash: "abc123",
	}
	id, err := repo.Insert(ctx, in)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}
	got, err := repo.Get(ctx, id)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	in.ID = id
	if got.ThreadID == nil || *got.ThreadID != thread {
		t.Fatalf("thread_id = %v, want %q", got.ThreadID, thread)
	}
	got.ThreadID, in.ThreadID = nil, nil
	if !got.ReceivedAt.Equal(in.ReceivedAt) {
		t.Fatalf("received_at = %v, want %v", got.ReceivedAt, in.ReceivedAt)
	}
	got.ReceivedAt = in.ReceivedAt
	if got != in {
		t.Fatalf("round trip mismatch:\n got  %+v\n want %+v", got, in)
	}

	if _, err := repo.Insert(ctx, in); !errors.Is(err, store.ErrDuplicate) {
		t.Fatalf("second insert err = %v, want ErrDuplicate", err)
	}
}
