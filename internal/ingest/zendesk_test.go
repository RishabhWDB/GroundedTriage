package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

var zdNow = time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)

type zdMock struct {
	mu       sync.Mutex
	requests []*http.Request
	bodies   []string
	handler  func(n int, r *http.Request) (int, http.Header, string)
}

func (m *zdMock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	m.mu.Lock()
	m.requests = append(m.requests, r)
	m.bodies = append(m.bodies, string(b))
	n := len(m.requests)
	m.mu.Unlock()
	code, hdr, body := m.handler(n, r)
	for k, v := range hdr {
		w.Header()[k] = v
	}
	w.WriteHeader(code)
	_, _ = w.Write([]byte(body))
}

func newZD(t *testing.T, m *zdMock) (*ZendeskSource, *[]time.Duration) {
	t.Helper()
	srv := httptest.NewServer(m)
	t.Cleanup(srv.Close)
	var sleeps []time.Duration
	return &ZendeskSource{
		BaseURL: srv.URL, Email: "agent@example.com", APIToken: "tok",
		Now:   func() time.Time { return zdNow },
		Sleep: func(_ context.Context, d time.Duration) error { sleeps = append(sleeps, d); return nil },
	}, &sleeps
}

func page(t *testing.T, tickets []zdTicket, users []zdUser, cursor string, end bool) string {
	t.Helper()
	b, err := json.Marshal(zdPage{Tickets: tickets, Users: users, AfterCursor: cursor, EndOfStream: end})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestZendeskFetchPaginatesAndMaps(t *testing.T) {
	created := zdNow.Add(-2 * time.Hour)
	m := &zdMock{}
	m.handler = func(n int, r *http.Request) (int, http.Header, string) {
		switch n {
		case 1:
			return 200, nil, page(t, []zdTicket{
				{ID: 101, Subject: "Refund", Description: "Charged twice", Status: "new", RequesterID: 7, CreatedAt: created},
				{ID: 102, Subject: "Old", Description: "done", Status: "solved", RequesterID: 7, CreatedAt: created},
			}, []zdUser{{ID: 7, Email: "Priya@Acme.com"}}, "c2", false)
		case 2:
			return 200, nil, page(t, []zdTicket{
				{ID: 103, Subject: "Login", Description: "Locked out", Status: "open", RequesterID: 8, CreatedAt: created},
				{ID: 104, Subject: "Gone", Status: "deleted", RequesterID: 8, CreatedAt: created},
			}, nil, "c3", true)
		}
		t.Errorf("unexpected request %d", n)
		return 500, nil, ""
	}
	z, _ := newZD(t, m)
	since := zdNow.Add(-24 * time.Hour)
	got, err := z.Fetch(context.Background(), since)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(m.requests))
	}
	q1, q2 := m.requests[0].URL.Query(), m.requests[1].URL.Query()
	if m.requests[0].URL.Path != "/api/v2/incremental/tickets/cursor.json" ||
		q1.Get("start_time") != fmt.Sprint(since.Unix()) || q1.Get("include") != "users" ||
		q2.Get("cursor") != "c2" || q2.Get("start_time") != "" {
		t.Fatalf("queries: %v / %v", q1, q2)
	}
	if user, pass, _ := m.requests[0].BasicAuth(); user != "agent@example.com/token" || pass != "tok" {
		t.Fatalf("auth = %q/%q", user, pass)
	}
	if len(got) != 2 {
		t.Fatalf("tickets = %+v (solved/deleted should be skipped)", got)
	}
	if got[0].ExternalID != "101" || got[0].Sender != "priya@acme.com" || got[0].Source != "zendesk" ||
		got[0].Body != "Charged twice" || !got[0].ReceivedAt.Equal(created) || *got[0].ThreadID != "101" {
		t.Fatalf("ticket 0 = %+v", got[0])
	}
	if got[1].Sender != "zendesk-user:8" {
		t.Fatalf("missing user fallback: %q", got[1].Sender)
	}
}

func TestZendeskStartTimeClamping(t *testing.T) {
	z := &ZendeskSource{Now: func() time.Time { return zdNow }}
	if got := z.startTime(time.Time{}); got != zdNow.Add(-30*24*time.Hour).Unix() {
		t.Fatalf("zero since -> %d", got)
	}
	if got := z.startTime(zdNow); got != zdNow.Add(-61*time.Second).Unix() {
		t.Fatalf("recent since not clamped: %d", got)
	}
}

func TestZendeskMaxPages(t *testing.T) {
	m := &zdMock{handler: func(n int, r *http.Request) (int, http.Header, string) {
		return 200, nil, page(t, nil, nil, fmt.Sprintf("c%d", n+1), false)
	}}
	z, _ := newZD(t, m)
	z.MaxPages = 3
	if _, err := z.Fetch(context.Background(), zdNow.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if len(m.requests) != 3 {
		t.Fatalf("requests = %d, want 3", len(m.requests))
	}
}

func TestZendesk429RetryHonoursRetryAfter(t *testing.T) {
	m := &zdMock{handler: func(n int, r *http.Request) (int, http.Header, string) {
		switch n {
		case 1:
			return 429, http.Header{"Retry-After": {"7"}}, `{"error":"rate limited"}`
		case 2:
			return 503, nil, "busy"
		}
		return 200, nil, page(t, nil, nil, "", true)
	}}
	z, sleeps := newZD(t, m)
	if _, err := z.Fetch(context.Background(), zdNow.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if len(m.requests) != 3 || len(*sleeps) != 2 || (*sleeps)[0] != 7*time.Second || (*sleeps)[1] != 2*time.Second {
		t.Fatalf("requests=%d sleeps=%v", len(m.requests), *sleeps)
	}
}

func TestZendeskGivesUp(t *testing.T) {
	m := &zdMock{handler: func(int, *http.Request) (int, http.Header, string) {
		return 429, http.Header{"Retry-After": {"1"}}, "slow down"
	}}
	z, _ := newZD(t, m)
	z.MaxRetries = 2
	_, err := z.Fetch(context.Background(), zdNow.Add(-time.Hour))
	if err == nil || !strings.Contains(err.Error(), "giving up after 3 attempts") || len(m.requests) != 3 {
		t.Fatalf("err=%v requests=%d", err, len(m.requests))
	}
}

func TestZendeskReply(t *testing.T) {
	m := &zdMock{handler: func(n int, r *http.Request) (int, http.Header, string) {
		if n == 1 {
			return 429, http.Header{"Retry-After": {"0"}}, ""
		}
		return 200, nil, `{"ticket":{"id":101}}`
	}}
	z, _ := newZD(t, m)
	err := z.Reply(context.Background(), model.Ticket{ExternalID: "101"}, model.ReplyDraft{Body: "We refunded it [1]."})
	if err != nil {
		t.Fatal(err)
	}
	r := m.requests[1]
	if len(m.requests) != 2 || r.Method != http.MethodPut || r.URL.Path != "/api/v2/tickets/101.json" ||
		r.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("request = %s %s (n=%d)", r.Method, r.URL.Path, len(m.requests))
	}
	var body struct {
		Ticket struct {
			Comment struct {
				Body   string `json:"body"`
				Public bool   `json:"public"`
			} `json:"comment"`
		} `json:"ticket"`
	}
	if err := json.Unmarshal([]byte(m.bodies[1]), &body); err != nil {
		t.Fatal(err)
	}
	if body.Ticket.Comment.Body != "We refunded it [1]." || !body.Ticket.Comment.Public {
		t.Fatalf("body = %s", m.bodies[1])
	}
}

func TestZendeskReplyDoesNotRetry5xx(t *testing.T) {
	m := &zdMock{handler: func(int, *http.Request) (int, http.Header, string) { return 500, nil, "oops" }}
	z, _ := newZD(t, m)
	err := z.Reply(context.Background(), model.Ticket{ExternalID: "101"}, model.ReplyDraft{Body: "x"})
	if err == nil || len(m.requests) != 1 {
		t.Fatalf("err=%v requests=%d (a failed PUT must not be retried)", err, len(m.requests))
	}
	if err := z.Reply(context.Background(), model.Ticket{ExternalID: "csv-abc"}, model.ReplyDraft{}); err == nil {
		t.Fatal("non-numeric id accepted")
	}
}

func TestZendeskFactory(t *testing.T) {
	t.Setenv("ZENDESK_SUBDOMAIN", "")
	t.Setenv("ZENDESK_EMAIL", "")
	t.Setenv("ZENDESK_API_TOKEN", "")
	if _, err := NewZendeskFactory(SourceConfig{Type: "zendesk"}); err == nil {
		t.Fatal("missing env accepted")
	}
	t.Setenv("ZENDESK_SUBDOMAIN", "acme")
	t.Setenv("ZENDESK_EMAIL", "a@x.com")
	t.Setenv("ZENDESK_API_TOKEN", "t")
	s, err := NewZendeskFactory(SourceConfig{Type: "zendesk"})
	if err != nil || s.(*ZendeskSource).BaseURL != "https://acme.zendesk.com" {
		t.Fatalf("s=%+v err=%v", s, err)
	}
}
