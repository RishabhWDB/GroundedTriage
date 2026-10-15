//go:build integration

package pipeline_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/pipeline"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

var testDB *storetest.DB

func TestMain(m *testing.M) { os.Exit(storetest.Run(m, &testDB)) }

func TestWorkerPollWithCSVIsIdempotent(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "ingest_checkpoints", "audit_log")
	ctx := context.Background()

	csv := &ingest.CSVSource{Path: "../../datasets/sample_emails.csv"}
	p := &pipeline.Poller{
		Sources:     ingest.Sources{"csv": csv},
		Ingestor:    &preprocess.Ingestor{Store: store.Tickets{Pool: pool}},
		Checkpoints: store.Checkpoints{Pool: pool},
		Audit:       store.Audit{Pool: pool},
		Overlap:     10 * time.Minute,
	}

	first := p.PollOnce(ctx)["csv"]
	if first.Err != nil || first.Stats.Inserted != 20 {
		t.Fatalf("first poll = %+v", first)
	}
	second := p.PollOnce(ctx)["csv"]
	if second.Err != nil || second.Stats.Inserted != 0 {
		t.Fatalf("second poll = %+v", second)
	}

	// Even with the checkpoint reset, dedupe keeps the run idempotent.
	if _, err := pool.Exec(ctx, `delete from ingest_checkpoints`); err != nil {
		t.Fatal(err)
	}
	third := p.PollOnce(ctx)["csv"]
	if third.Err != nil || third.Stats.Inserted != 0 || third.Fetched != 22 {
		t.Fatalf("poll after checkpoint reset = %+v", third)
	}

	var tickets, polls int
	var cp time.Time
	_ = pool.QueryRow(ctx, `select count(*) from tickets`).Scan(&tickets)
	_ = pool.QueryRow(ctx, `select count(*) from audit_log where action = 'ingest.poll' and entity_id = 'csv'`).Scan(&polls)
	_ = pool.QueryRow(ctx, `select since from ingest_checkpoints where source = 'csv'`).Scan(&cp)
	if tickets != 20 || polls != 3 || !cp.Equal(time.Date(2026, 10, 9, 6, 0, 0, 0, time.UTC)) {
		t.Fatalf("tickets=%d polls=%d checkpoint=%v", tickets, polls, cp)
	}
}
