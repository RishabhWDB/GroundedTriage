//go:build integration

package pipeline_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/pipeline"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/routing"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// stubFullML answers /classify, /retrieve, /draft and /verify. Billing tickets
// about a double charge find KB chunks (grounded); everything else retrieves
// below the floor (abstained).
func stubFullML(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var in map[string]any
		_ = json.NewDecoder(r.Body).Decode(&in)
		id, _ := in["ticket_id"].(string)
		text := strings.ToLower(fmtStr(in["subject"]) + " " + fmtStr(in["body"]))
		enc := json.NewEncoder(w)
		switch r.URL.Path {
		case "/classify":
			cat := "General Inquiry"
			if strings.Contains(text, "charged") || strings.Contains(text, "invoice") {
				cat = "Billing"
			}
			_ = enc.Encode(model.Classification{TicketID: id, Category: cat, CategoryConfidence: 0.9, Tier: "fast",
				Urgency: "medium", UrgencySource: "heuristic"})
		case "/retrieve":
			if strings.Contains(text, "charged twice") {
				_ = enc.Encode(mlclient.RetrieveResponse{TicketID: id, Mode: "dense", MaxScore: 0.81, RetrievalFloor: 0.6,
					Chunks: []model.RetrievedChunk{{ChunkID: "kb1:0", DocID: "kb1", Title: "Refunds",
						Text: "Duplicate charges are refunded within 5 days.", Score: 0.81}}})
				return
			}
			_ = enc.Encode(mlclient.RetrieveResponse{TicketID: id, Mode: "dense", MaxScore: 0.42, RetrievalFloor: 0.6,
				BelowFloor: true, Chunks: []model.RetrievedChunk{{ChunkID: "kb9:0", DocID: "kb9", Text: "x", Score: 0.42}}})
		case "/draft":
			_ = enc.Encode(mlclient.GeneratedDraft{TicketID: id, Subject: "Re: charge",
				Body: "Duplicate charges are refunded within 5 days [1].", Citations: []string{"kb1:0"}})
		case "/verify":
			score := 1.0
			_ = enc.Encode(mlclient.Verification{
				Draft: model.ReplyDraft{TicketID: id, Subject: fmtStr(in["subject"]), Body: fmtStr(in["body"]),
					Citations: []string{"kb1:0"}, FaithfulnessScore: &score, Status: "grounded"},
				Claims:            []mlclient.Claim{{Claim: "Duplicate charges are refunded within 5 days.", Cited: []int{1}, Supported: true}},
				RetrievalMaxScore: 0.81})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func fmtStr(v any) string {
	s, _ := v.(string)
	return s
}

func TestCSVTicketToGroundedDraftAndAnotherToAbstention(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "routes", "reply_drafts", "ingest_checkpoints", "audit_log")
	ctx := context.Background()

	// Layer 1: ingest the sample CSV.
	poller := &pipeline.Poller{
		Sources:     ingest.Sources{"csv": &ingest.CSVSource{Path: "../../datasets/sample_emails.csv"}},
		Ingestor:    &preprocess.Ingestor{Store: store.Tickets{Pool: pool}},
		Checkpoints: store.Checkpoints{Pool: pool},
		Overlap:     10 * time.Minute,
	}
	if r := poller.PollOnce(ctx)["csv"]; r.Err != nil {
		t.Fatal(r.Err)
	}
	ml := mlclient.New(stubFullML(t).URL)
	// Layer 2 and 3: classify and route.
	if _, err := (&pipeline.Classifier{Store: store.Classifications{Pool: pool}, ML: ml}).ClassifyPending(ctx); err != nil {
		t.Fatal(err)
	}
	engine, _ := routing.NewEngine("../../config/routing_rules.yaml")
	sla, _ := routing.LoadSLA("../../config/sla.yaml")
	if _, err := (&pipeline.Router{Store: store.Routes{Pool: pool}, Rules: engine, SLA: sla}).RoutePending(ctx); err != nil {
		t.Fatal(err)
	}
	// Layer 4: grounded reply.
	drafts := store.ReplyDrafts{Pool: pool}
	rep := &pipeline.Replier{Drafts: drafts, Routes: store.Routes{Pool: pool}, ML: ml, Audit: store.Audit{Pool: pool}, BatchSize: 50}
	st, err := rep.ReplyPending(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if st.Grounded < 1 || st.Abstained < 1 || st.Failed != 0 {
		t.Fatalf("stats %+v", st)
	}

	var charged, library string
	_ = pool.QueryRow(ctx, `select id::text from tickets where subject = 'Charged twice for October invoice'`).Scan(&charged)
	_ = pool.QueryRow(ctx, `select id::text from tickets where subject = 'Library hours during the holiday weekend'`).Scan(&library)
	g, ok, err := drafts.Latest(ctx, charged)
	if err != nil || !ok || g.Status != "grounded" || g.Citations[0] != "kb1:0" || *g.FaithfulnessScore != 1 ||
		!strings.Contains(g.Body, "[1]") || g.RetrievalMode != "dense" {
		t.Fatalf("grounded draft %+v ok=%v err=%v", g, ok, err)
	}
	a, ok, _ := drafts.Latest(ctx, library)
	if !ok || a.Status != "abstained" || !strings.Contains(a.AbstainReason, "< floor 0.6") || a.Body != "" {
		t.Fatalf("abstained draft %+v", a)
	}
	route, _, _ := store.Routes{Pool: pool}.Latest(ctx, library)
	if route.Queue != "human_review" || route.Action != "queue" || !strings.Contains(route.Reason, "draft abstained") {
		t.Fatalf("abstained ticket's route %+v", route)
	}
	if again, _ := rep.ReplyPending(ctx); again.Pending != 0 {
		t.Fatalf("drafted twice: %+v", again)
	}
}
