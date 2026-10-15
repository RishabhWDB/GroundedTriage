//go:build integration

package preprocess_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

var testDB *storetest.DB

func TestMain(m *testing.M) { os.Exit(storetest.Run(m, &testDB)) }

func TestIngestorRejectsDuplicatesInDB(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets")
	ctx := context.Background()
	in := &preprocess.Ingestor{Store: store.Tickets{Pool: pool}}

	tickets, err := (&ingest.CSVSource{Path: "../../datasets/sample_emails.csv"}).Fetch(ctx, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	_, first, err := in.Process(ctx, tickets)
	if err != nil {
		t.Fatal(err)
	}
	// 22 rows: 1 Portuguese skipped, 1 repeated complaint caught by content hash.
	if first != (preprocess.Stats{Inserted: 20, DuplicateContent: 1, Skipped: 1}) {
		t.Fatalf("first run = %+v", first)
	}

	_, second, err := in.Process(ctx, tickets)
	if err != nil {
		t.Fatal(err)
	}
	if second.Inserted != 0 || second.Skipped != 1 || second.DuplicateContent+second.DuplicateID != 21 {
		t.Fatalf("second run = %+v", second)
	}

	// Same (source, external_id) with new content is rejected by the unique key.
	dup := tickets[0]
	dup.Body = "Completely different follow-up text that has never been seen before."
	res, stats, err := in.Process(ctx, []model.Ticket{dup})
	if err != nil {
		t.Fatal(err)
	}
	if stats.DuplicateID != 1 || res[0].Outcome != preprocess.OutcomeDuplicateID {
		t.Fatalf("external-id duplicate not rejected: %+v", res)
	}

	var n int
	if err := pool.QueryRow(ctx, `select count(*) from tickets`).Scan(&n); err != nil || n != 20 {
		t.Fatalf("tickets in db = %d (%v), want 20", n, err)
	}
	var body string
	if err := pool.QueryRow(ctx, `select body from tickets where subject = 'Mobile app crashes on startup'`).Scan(&body); err != nil {
		t.Fatal(err)
	}
	if body == "" || body[0] == '<' {
		t.Fatalf("stored body not cleaned: %q", body)
	}
}
