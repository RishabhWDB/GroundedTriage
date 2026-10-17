//go:build integration

package api_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/RishabhWDB/GroundedTriage/internal/api"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
	"github.com/RishabhWDB/GroundedTriage/internal/ui"
)

var testDB *storetest.DB

func TestMain(m *testing.M) { os.Exit(storetest.Run(m, &testDB)) }

// The review UI's calls, in order, against seeded data and the real API + UI server.
func TestReviewFlowAgainstSeededData(t *testing.T) {
	pool := testDB.Require(t)
	testDB.Truncate(t, "tickets", "classifications", "routes", "reply_drafts", "kb_chunks", "kb_documents", "audit_log")
	ctx := t.Context()
	id, err := store.Tickets{Pool: pool}.Insert(ctx, model.Ticket{Source: model.SourceCSV, ExternalID: storetest.UniqueID("ui"),
		Sender: "priya@acme.com", Subject: "Charged twice <script>alert(1)</script>", Body: "Please refund.",
		ReceivedAt: time.Now(), ContentHash: storetest.UniqueID("h")})
	if err != nil {
		t.Fatal(err)
	}
	_ = store.Classifications{Pool: pool}.Insert(ctx, model.Classification{TicketID: id, Category: "Billing",
		CategoryConfidence: 0.91, Tier: "fast", Urgency: "high", UrgencySource: "heuristic"})
	due := time.Now().Add(4 * time.Hour)
	_, _ = store.Routes{Pool: pool}.Insert(ctx, model.Route{TicketID: id, Action: "reply", Queue: "billing", Priority: 2, SLADueAt: &due})
	_, _ = pool.Exec(ctx, `insert into kb_documents (id, title, url) values ('kb1', 'Refunds', 'https://help.example/refunds')`)
	_, _ = pool.Exec(ctx, `insert into kb_chunks (id, doc_id, ordinal, title, heading, text) values
		('kb1:0', 'kb1', 0, 'Refunds', 'Duplicate charges', 'Duplicate charges are refunded within 5 days.')`)
	one := 1.0
	_, err = store.ReplyDrafts{Pool: pool}.Insert(ctx, store.StoredDraft{ReplyDraft: model.ReplyDraft{TicketID: id,
		Subject: "Re: Charged twice", Body: "Duplicate charges are refunded within 5 days [1].", Citations: []string{"kb1:0"},
		FaithfulnessScore: &one, Status: "grounded"}, RetrievedChunkIDs: []string{"kb1:0"}, RetrievalMode: "dense",
		Claims: []map[string]any{{"claim": "Duplicate charges are refunded within 5 days.", "cited": []int{1}, "supported": true}}})
	if err != nil {
		t.Fatal(err)
	}

	gin.SetMode(gin.TestMode)
	r := gin.New()
	(&api.Server{Store: store.Review{Pool: pool}, Ingest: api.FileIngester{Ingestor: &preprocess.Ingestor{Store: store.Tickets{Pool: pool}}},
		Audit: store.Audit{Pool: pool}, APIKey: "k", UploadDir: t.TempDir()}).Register(r)
	ui.Register(r)
	srv := httptest.NewServer(r)
	defer srv.Close()
	call := func(method, path, body string, out any) int {
		req, _ := http.NewRequest(method, srv.URL+path, strings.NewReader(body))
		req.Header.Set("X-API-Key", "k")
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		if out != nil {
			if err := json.Unmarshal(data, out); err != nil {
				t.Fatalf("%s %s: %s", method, path, data)
			}
		}
		return resp.StatusCode
	}

	// The page itself.
	resp, err := http.Get(srv.URL + "/ui/")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("ui: %v %v", resp, err)
	}
	resp.Body.Close()

	var inbox struct {
		Items []store.TicketSummary
		Total int
	}
	if call("GET", "/tickets?status=open", "", &inbox) != 200 || inbox.Total != 1 || *inbox.Items[0].DraftStatus != "grounded" ||
		*inbox.Items[0].Priority != 2 || !strings.Contains(inbox.Items[0].Subject, "<script>") {
		t.Fatalf("inbox %+v", inbox)
	}
	var det store.TicketDetail
	if call("GET", "/tickets/"+id, "", &det) != 200 || det.Classification.UrgencySource != "heuristic" || det.Route.Queue != "billing" {
		t.Fatalf("detail %+v", det)
	}
	var dr struct {
		Draft  store.DraftView
		Chunks []struct {
			ID, URL, Text string
			Number        int
			Cited         bool
		}
	}
	if call("GET", "/tickets/"+id+"/draft", "", &dr) != 200 || len(dr.Chunks) != 1 || dr.Chunks[0].Number != 1 || !dr.Chunks[0].Cited ||
		dr.Chunks[0].URL != "https://help.example/refunds" {
		t.Fatalf("draft %+v", dr)
	}
	var edited store.DraftView
	if call("POST", "/drafts/"+dr.Draft.ID+"/edit", `{"reviewer":"ann","subject":"Re: Charged twice","body":"Sorry! Duplicate charges are refunded within 5 days [1]."}`, &edited) != 200 || !edited.Edited {
		t.Fatalf("edit %+v", edited)
	}
	var approved store.DraftView
	if call("POST", "/drafts/"+dr.Draft.ID+"/approve", `{"reviewer":"ann","note":"ok"}`, &approved) != 200 ||
		approved.Status != "approved" || approved.IdempotencyKey == nil {
		t.Fatalf("approve %+v", approved)
	}
	if code := call("POST", "/drafts/"+dr.Draft.ID+"/edit", `{"reviewer":"ann","body":"x"}`, nil); code != 409 {
		t.Fatalf("edit after approve: %d", code)
	}
	var n int
	_ = pool.QueryRow(ctx, `select count(*) from audit_log where entity_id = $1 and actor = 'ann'`, dr.Draft.ID).Scan(&n)
	if n != 2 {
		t.Fatalf("audit rows %d", n)
	}
}
