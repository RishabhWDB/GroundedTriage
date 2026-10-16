//go:build integration

package pipeline_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/pipeline"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// stubML is a stand-in ml-service: POST /classify answers by keyword.
func stubML(t *testing.T) (*httptest.Server, *[]string) {
	var mu sync.Mutex
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/classify" {
			http.NotFound(w, r)
			return
		}
		var req mlclient.ClassifyRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		seen = append(seen, req.TicketID)
		mu.Unlock()
		cat, tier, low := "General Inquiry", "fast", true
		text := strings.ToLower(req.Subject + " " + req.Body)
		if strings.Contains(text, "charge") || strings.Contains(text, "invoice") {
			cat, low = "Billing", false
		}
		_ = json.NewEncoder(w).Encode(model.Classification{
			TicketID: req.TicketID, Category: cat, CategoryConfidence: 0.8, Tier: tier,
			LowConfidence: low, Urgency: "medium", UrgencyConfidence: 0.4, UrgencySource: "heuristic",
			ModelsLoaded:  model.ModelsLoaded{},
			ModelVersions: map[string]string{"fast_tier": "fast-stub", "urgency": "heuristic"},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

func TestClassifierClassifiesIngestedTickets(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "ingest_checkpoints", "audit_log")
	ctx := context.Background()

	// Ingest the sample CSV: 20 accepted + 1 skipped (Portuguese) ticket.
	p := &pipeline.Poller{
		Sources:     ingest.Sources{"csv": &ingest.CSVSource{Path: "../../datasets/sample_emails.csv"}},
		Ingestor:    &preprocess.Ingestor{Store: store.Tickets{Pool: pool}},
		Checkpoints: store.Checkpoints{Pool: pool},
		Audit:       store.Audit{Pool: pool},
		Overlap:     10 * time.Minute,
	}
	if r := p.PollOnce(ctx)["csv"]; r.Err != nil || r.Stats.Inserted != 20 || r.Stats.InsertedSkipped != 1 {
		t.Fatalf("ingest = %+v", r)
	}

	srv, seen := stubML(t)
	repo := store.Classifications{Pool: pool}
	c := &pipeline.Classifier{Store: repo, ML: mlclient.New(srv.URL), Audit: store.Audit{Pool: pool}}
	stats, err := c.ClassifyPending(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Pending != 20 || stats.Classified != 20 || stats.Failed != 0 {
		t.Fatalf("stats = %+v", stats)
	}
	if len(*seen) != 20 {
		t.Fatalf("ml-service saw %d tickets", len(*seen))
	}

	// The skipped ticket was never sent; every accepted one has a row.
	var skipped string
	if err := pool.QueryRow(ctx, `select id::text from tickets where ingest_status = 'skipped'`).Scan(&skipped); err != nil {
		t.Fatal(err)
	}
	for _, id := range *seen {
		if id == skipped {
			t.Fatal("skipped ticket was classified")
		}
	}
	var n int
	if err := pool.QueryRow(ctx, `select count(*) from classifications`).Scan(&n); err != nil || n != 20 {
		t.Fatalf("classifications = %d (%v)", n, err)
	}
	got, ok, err := repo.Latest(ctx, (*seen)[0])
	if err != nil || !ok {
		t.Fatalf("latest: ok=%v err=%v", ok, err)
	}
	if got.ModelVersions["fast_tier"] != "fast-stub" || got.UrgencySource != "heuristic" || got.Tier != "fast" {
		t.Fatalf("stored = %+v", got)
	}
	var billing int
	if err := pool.QueryRow(ctx, `select count(*) from classifications where category = 'Billing' and not low_confidence`).Scan(&billing); err != nil || billing == 0 {
		t.Fatalf("billing rows = %d (%v)", billing, err)
	}
	if _, ok, _ := repo.Latest(ctx, skipped); ok {
		t.Fatal("skipped ticket has a classification")
	}

	// A second run finds nothing pending, so nothing is re-sent.
	again, err := c.ClassifyPending(ctx)
	if err != nil || again.Pending != 0 || len(*seen) != 20 {
		t.Fatalf("second run = %+v, err %v, seen %d", again, err, len(*seen))
	}
	var audits int
	if err := pool.QueryRow(ctx, `select count(*) from audit_log where action = 'classify.run'`).Scan(&audits); err != nil || audits != 1 {
		t.Fatalf("classify.run audit rows = %d (%v)", audits, err)
	}
}

func TestClassifierStopsWhenMLServiceDown(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "ingest_checkpoints", "audit_log")
	ctx := context.Background()
	if _, err := (store.Tickets{Pool: pool}).Insert(ctx, model.Ticket{Source: model.SourceCSV,
		ExternalID: "down-1", Body: "hello", ReceivedAt: time.Now(), ContentHash: "h-down-1"}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "models/fast_tier/ missing", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	ml := mlclient.New(srv.URL)
	ml.MaxRetries, ml.Backoff = 0, time.Millisecond
	c := &pipeline.Classifier{Store: store.Classifications{Pool: pool}, ML: ml, MaxConsecutiveFailures: 1}
	stats, err := c.ClassifyPending(ctx)
	if err == nil || !stats.Stopped || stats.Classified != 0 {
		t.Fatalf("stats %+v err %v", stats, err)
	}
	var n int
	_ = pool.QueryRow(ctx, `select count(*) from classifications`).Scan(&n)
	if n != 0 {
		t.Fatalf("classifications = %d", n)
	}
}
