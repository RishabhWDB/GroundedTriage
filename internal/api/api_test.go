package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

const (
	key      = "test-key"
	ticketID = "11111111-1111-1111-1111-111111111111"
	draftID  = "22222222-2222-2222-2222-222222222222"
)

type fakeStore struct {
	filter store.TicketFilter
	drafts map[string]store.DraftView
}

func (f *fakeStore) ListTickets(_ context.Context, fl store.TicketFilter) ([]store.TicketSummary, int, error) {
	f.filter = fl
	return []store.TicketSummary{{ID: ticketID, Subject: "Charged twice"}}, 1, nil
}

func (f *fakeStore) GetTicket(_ context.Context, id string) (store.TicketDetail, error) {
	if id != ticketID {
		return store.TicketDetail{}, store.ErrNotFound
	}
	return store.TicketDetail{Ticket: model.Ticket{ID: id, Subject: "Charged twice"}}, nil
}

func (f *fakeStore) LatestDraft(_ context.Context, id string) (store.DraftView, error) {
	if id != ticketID {
		return store.DraftView{}, store.ErrNotFound
	}
	return store.DraftView{ID: draftID, TicketID: id, Body: "Refunds take 5 days [2].", Citations: []string{"kb2:0"},
		RetrievedChunkIDs: []string{"kb1:0", "kb2:0"}, Status: "grounded"}, nil
}

func (f *fakeStore) Chunks(_ context.Context, ids []string) ([]store.ChunkView, error) {
	out := []store.ChunkView{}
	for _, id := range ids {
		out = append(out, store.ChunkView{ID: id, Text: "text of " + id})
	}
	return out, nil
}

func (f *fakeStore) transition(id, status, reviewer string) (store.DraftView, error) {
	d, ok := f.drafts[id]
	if !ok {
		return d, store.ErrNotFound
	}
	if d.Status != "grounded" {
		return d, store.ErrConflict
	}
	d.Status, d.Reviewer = status, reviewer
	f.drafts[id] = d
	return d, nil
}

func (f *fakeStore) ApproveDraft(_ context.Context, id, reviewer, _ string) (store.DraftView, error) {
	return f.transition(id, "approved", reviewer)
}

func (f *fakeStore) RejectDraft(_ context.Context, id, reviewer, _ string) (store.DraftView, error) {
	return f.transition(id, "rejected", reviewer)
}

func (f *fakeStore) EditDraft(_ context.Context, id, reviewer, subject, body string) (store.DraftView, error) {
	d, err := f.transition(id, "grounded", reviewer)
	d.Subject, d.Body, d.Edited = subject, body, true
	return d, err
}

func (f *fakeStore) Summary(context.Context, time.Time) (store.Summary, error) {
	return store.Summary{Tickets: 3, ByCategory: map[string]int{"Billing": 1}}, nil
}

type fakeIngest struct{ path string }

func (f *fakeIngest) IngestCSV(_ context.Context, path string) (int, preprocess.Stats, error) {
	f.path = path
	data, _ := os.ReadFile(path)
	if !strings.HasPrefix(string(data), "Sender,") {
		return 0, preprocess.Stats{}, errors.New("missing Sender column")
	}
	return 2, preprocess.Stats{Inserted: 1, DuplicateContent: 1}, nil
}

type fakeHealth struct{ err error }

func (f fakeHealth) Health(context.Context) (mlclient.Health, error) {
	return mlclient.Health{Status: "ok", FastTierLoaded: true}, f.err
}

type memAudit struct{ entries []store.AuditEntry }

func (m *memAudit) Write(_ context.Context, e store.AuditEntry) error {
	m.entries = append(m.entries, e)
	return nil
}

func newServer(t *testing.T) (*gin.Engine, *fakeStore, *fakeIngest, *memAudit) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	st := &fakeStore{drafts: map[string]store.DraftView{draftID: {ID: draftID, Status: "grounded"}}}
	in := &fakeIngest{}
	au := &memAudit{}
	s := &Server{Store: st, Ingest: in, ML: fakeHealth{}, Audit: au, APIKey: key, UploadDir: t.TempDir(), MaxUpload: 1024}
	r := gin.New()
	s.Register(r)
	return r, st, in, au
}

func do(r http.Handler, method, path, body string, hdr map[string]string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("X-API-Key", key)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec
}

func errCode(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var b struct {
		Error struct{ Code, Message string } `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &b); err != nil || b.Error.Code == "" {
		t.Fatalf("not a JSON error: %s", rec.Body.String())
	}
	return b.Error.Code
}

func TestAPIKey(t *testing.T) {
	r, _, _, _ := newServer(t)
	if rec := do(r, "GET", "/tickets", "", map[string]string{"X-API-Key": ""}); rec.Code != 401 || errCode(t, rec) != CodeUnauthorized {
		t.Fatalf("no key: %d", rec.Code)
	}
	if rec := do(r, "GET", "/tickets", "", map[string]string{"X-API-Key": "wrong"}); rec.Code != 401 {
		t.Fatalf("wrong key: %d", rec.Code)
	}
	if rec := do(r, "GET", "/tickets", "", map[string]string{"X-API-Key": "", "Authorization": "Bearer " + key}); rec.Code != 200 {
		t.Fatalf("bearer: %d", rec.Code)
	}
	gin.SetMode(gin.TestMode)
	empty := gin.New()
	(&Server{Store: &fakeStore{}}).Register(empty) // no API key configured: everything is refused
	if rec := do(empty, "GET", "/tickets", "", map[string]string{"X-API-Key": ""}); rec.Code != 401 {
		t.Fatalf("unconfigured key: %d", rec.Code)
	}
}

func TestListTicketsValidationAndFilters(t *testing.T) {
	r, st, _, _ := newServer(t)
	for _, q := range []string{"category=Complaint", "urgency=urgent", "status=closed", "limit=0", "limit=x", "offset=-1"} {
		if rec := do(r, "GET", "/tickets?"+q, "", nil); rec.Code != 400 || errCode(t, rec) != CodeBadRequest {
			t.Fatalf("%s: %d", q, rec.Code)
		}
	}
	rec := do(r, "GET", "/tickets?category=Billing&urgency=high&status=open&queue=billing&limit=10&offset=5", "", nil)
	if rec.Code != 200 {
		t.Fatalf("code %d %s", rec.Code, rec.Body)
	}
	want := store.TicketFilter{Category: "Billing", Urgency: "high", Status: "open", Queue: "billing", Limit: 10, Offset: 5}
	if st.filter != want {
		t.Fatalf("filter %+v", st.filter)
	}
	var b struct {
		Items []store.TicketSummary
		Total int
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &b)
	if b.Total != 1 || b.Items[0].ID != ticketID {
		t.Fatalf("body %s", rec.Body)
	}
}

func TestTicketAndDraftEndpoints(t *testing.T) {
	r, _, _, _ := newServer(t)
	if rec := do(r, "GET", "/tickets/not-a-uuid", "", nil); rec.Code != 400 {
		t.Fatalf("bad id: %d", rec.Code)
	}
	if rec := do(r, "GET", "/tickets/33333333-3333-3333-3333-333333333333", "", nil); rec.Code != 404 || errCode(t, rec) != CodeNotFound {
		t.Fatalf("missing: %d", rec.Code)
	}
	if rec := do(r, "GET", "/tickets/"+ticketID, "", nil); rec.Code != 200 || !strings.Contains(rec.Body.String(), "Charged twice") {
		t.Fatalf("detail: %d %s", rec.Code, rec.Body)
	}
	rec := do(r, "GET", "/tickets/"+ticketID+"/draft", "", nil)
	var b struct {
		Draft  store.DraftView
		Chunks []struct {
			ID     string
			Number int
			Cited  bool
		}
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &b); err != nil || rec.Code != 200 {
		t.Fatalf("draft: %d %s", rec.Code, rec.Body)
	}
	if len(b.Chunks) != 2 || b.Chunks[1].ID != "kb2:0" || b.Chunks[1].Number != 2 || !b.Chunks[1].Cited || b.Chunks[0].Cited {
		t.Fatalf("chunks %+v", b.Chunks)
	}
}

func TestReviewActions(t *testing.T) {
	r, st, _, au := newServer(t)
	url := "/drafts/" + draftID
	if rec := do(r, "POST", url+"/approve", `{"note":"x"}`, nil); rec.Code != 400 {
		t.Fatalf("missing reviewer: %d", rec.Code)
	}
	if rec := do(r, "POST", url+"/approve", `not json`, nil); rec.Code != 400 {
		t.Fatalf("bad json: %d", rec.Code)
	}
	if rec := do(r, "POST", url+"/edit", `{"reviewer":"ann","body":"  "}`, nil); rec.Code != 400 {
		t.Fatalf("edit without body: %d", rec.Code)
	}
	if rec := do(r, "POST", url+"/edit", `{"reviewer":"ann","subject":"Re","body":"New text"}`, nil); rec.Code != 200 {
		t.Fatalf("edit: %d %s", rec.Code, rec.Body)
	}
	if rec := do(r, "POST", url+"/approve", `{"reviewer":"ann","note":"lgtm"}`, nil); rec.Code != 200 ||
		st.drafts[draftID].Status != "approved" {
		t.Fatalf("approve: %d %s", rec.Code, rec.Body)
	}
	if rec := do(r, "POST", url+"/reject", `{"reviewer":"bob"}`, nil); rec.Code != 409 || errCode(t, rec) != CodeConflict {
		t.Fatalf("reject approved (fake disallows): %d", rec.Code)
	}
	if rec := do(r, "POST", "/drafts/33333333-3333-3333-3333-333333333333/approve", `{"reviewer":"ann"}`, nil); rec.Code != 404 {
		t.Fatalf("missing draft: %d", rec.Code)
	}
	if len(au.entries) != 2 || au.entries[0].Action != "draft.edited" || au.entries[1].Action != "draft.approved" ||
		au.entries[1].Actor != "ann" {
		t.Fatalf("audit %+v", au.entries)
	}
}

func upload(r http.Handler, field, name, content string) *httptest.ResponseRecorder {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, _ := w.CreateFormFile(field, name)
	_, _ = fw.Write([]byte(content))
	_ = w.Close()
	req := httptest.NewRequest("POST", "/ingest/csv", &buf)
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("X-API-Key", key)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec
}

func TestIngestCSVUpload(t *testing.T) {
	r, _, in, _ := newServer(t)
	rec := upload(r, "file", "../../etc/passwd.csv", "Sender,Subject,Body\na@b.com,Hi,Hello\n")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"inserted":1`) {
		t.Fatalf("upload: %d %s", rec.Code, rec.Body)
	}
	if strings.Contains(in.path, "passwd") || filepath.Ext(in.path) != ".csv" {
		t.Fatalf("stored at %q: the client's file name must not be used", in.path)
	}
	if rec := upload(r, "file", "notes.txt", "x"); rec.Code != 400 {
		t.Fatalf("txt: %d", rec.Code)
	}
	if rec := upload(r, "other", "a.csv", "x"); rec.Code != 400 {
		t.Fatalf("missing field: %d", rec.Code)
	}
	if rec := upload(r, "file", "big.csv", strings.Repeat("x", 2048)); rec.Code != 413 || errCode(t, rec) != CodeTooLarge {
		t.Fatalf("too big: %d", rec.Code)
	}
	if rec := upload(r, "file", "bad.csv", "Name,Text\n"); rec.Code != 400 || !strings.Contains(rec.Body.String(), "Sender") {
		t.Fatalf("bad csv: %d %s", rec.Code, rec.Body)
	}
}

func TestSummaryIncludesModels(t *testing.T) {
	r, _, _, _ := newServer(t)
	rec := do(r, "GET", "/metrics/summary", "", nil)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"fast_tier_loaded":true`) ||
		!strings.Contains(rec.Body.String(), `"tickets":3`) {
		t.Fatalf("summary: %s", rec.Body)
	}
	gin.SetMode(gin.TestMode)
	down := gin.New()
	(&Server{Store: &fakeStore{}, ML: fakeHealth{err: errors.New("refused")}, APIKey: key}).Register(down)
	if rec := do(down, "GET", "/metrics/summary", "", nil); !strings.Contains(rec.Body.String(), "unreachable") {
		t.Fatalf("ml down: %s", rec.Body)
	}
}

type fakeDaily struct{ days int }

func (f *fakeDaily) Daily(_ context.Context, days int, _ time.Time) ([]store.DailyMetric, error) {
	f.days = days
	v := 0.31
	return []store.DailyMetric{{Metric: "category_psi", Value: &v, Detail: map[string]any{"alert": true}}}, nil
}

func TestDailyMetrics(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	fd := &fakeDaily{}
	(&Server{Store: &fakeStore{}, Daily: fd, APIKey: key}).Register(r)
	rec := do(r, "GET", "/metrics/daily", "", nil)
	if rec.Code != 200 || fd.days != 14 || !strings.Contains(rec.Body.String(), `"value":0.31`) {
		t.Fatalf("daily: %d %s (days %d)", rec.Code, rec.Body, fd.days)
	}
	if rec := do(r, "GET", "/metrics/daily?days=30", "", nil); rec.Code != 200 || fd.days != 30 {
		t.Fatalf("days=30: %d, got %d", rec.Code, fd.days)
	}
	for _, bad := range []string{"0", "91", "x"} {
		if rec := do(r, "GET", "/metrics/daily?days="+bad, "", nil); rec.Code != 400 || errCode(t, rec) != CodeBadRequest {
			t.Fatalf("days=%s: %d", bad, rec.Code)
		}
	}
	none := gin.New()
	(&Server{Store: &fakeStore{}, APIKey: key}).Register(none)
	if rec := do(none, "GET", "/metrics/daily", "", nil); rec.Code != 200 || !strings.Contains(rec.Body.String(), `"metrics":[]`) {
		t.Fatalf("no store: %s", rec.Body)
	}
}

func TestRequireHTTPS(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequireHTTPS())
	r.GET("/x", func(c *gin.Context) { c.String(200, "ok") })
	req := httptest.NewRequest("GET", "http://demo.example/x?a=1", nil)
	req.Header.Set("X-Forwarded-Proto", "http")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusPermanentRedirect || rec.Header().Get("Location") != "https://demo.example/x?a=1" {
		t.Fatalf("http: %d %q", rec.Code, rec.Header().Get("Location"))
	}
	for _, proto := range []string{"https", ""} {
		req := httptest.NewRequest("GET", "/x", nil)
		if proto != "" {
			req.Header.Set("X-Forwarded-Proto", proto)
		}
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		if rec.Code != 200 || !strings.Contains(rec.Header().Get("Strict-Transport-Security"), "max-age=") {
			t.Fatalf("proto %q: %d %v", proto, rec.Code, rec.Header())
		}
	}
}
