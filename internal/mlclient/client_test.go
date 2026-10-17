package mlclient

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

const healthJSON = `{"status":"ok","service":"ml-service","version":"0.1.0",
 "models_loaded":{"deep_tier":false,"urgency":true},"fast_tier_loaded":false,"torch_installed":false}`

func newTestClient(url string) *Client {
	c := New(url)
	c.Backoff = time.Millisecond
	c.Timeout = 200 * time.Millisecond
	return c
}

func TestHealthDecodes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("path = %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(healthJSON))
	}))
	defer srv.Close()

	h, err := newTestClient(srv.URL).Health(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if h.Status != "ok" || h.ModelsLoaded.DeepTier || !h.ModelsLoaded.Urgency {
		t.Fatalf("unexpected health %+v", h)
	}
}

func TestRetriesThenSucceeds(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch n.Add(1) {
		case 1:
			w.WriteHeader(http.StatusServiceUnavailable)
		case 2:
			w.WriteHeader(http.StatusTooManyRequests)
		default:
			_, _ = w.Write([]byte(healthJSON))
		}
	}))
	defer srv.Close()

	if _, err := newTestClient(srv.URL).Health(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := n.Load(); got != 3 {
		t.Fatalf("attempts = %d, want 3", got)
	}
}

func TestNoRetryOn4xx(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		http.Error(w, "bad input", http.StatusUnprocessableEntity)
	}))
	defer srv.Close()

	_, err := newTestClient(srv.URL).Health(context.Background())
	var se *StatusError
	if !errors.As(err, &se) || se.Code != http.StatusUnprocessableEntity {
		t.Fatalf("err = %v, want 422 StatusError", err)
	}
	if n.Load() != 1 {
		t.Fatalf("attempts = %d, want 1", n.Load())
	}
}

func TestPerAttemptTimeoutAndGiveUp(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		select {
		case <-time.After(time.Second):
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()

	c := newTestClient(srv.URL)
	c.Timeout = 30 * time.Millisecond
	c.MaxRetries = 2
	start := time.Now()
	_, err := c.Health(context.Background())
	if err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
	if n.Load() != 3 {
		t.Fatalf("attempts = %d, want 3", n.Load())
	}
	if time.Since(start) > 900*time.Millisecond {
		t.Fatalf("took %v; per-attempt timeout not applied", time.Since(start))
	}
}

func TestClassifyPostsTicket(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/classify" {
			t.Errorf("%s %s", r.Method, r.URL.Path)
		}
		var req ClassifyRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.TicketID != "t1" || req.Body != "refund" {
			t.Errorf("request %+v err %v", req, err)
		}
		_, _ = w.Write([]byte(`{"ticket_id":"t1","category":"Billing","category_confidence":0.9,"tier":"fast",
		 "low_confidence":false,"urgency":"high","urgency_confidence":0.7,"urgency_source":"heuristic",
		 "models_loaded":{"deep_tier":false,"urgency":false},"model_versions":{"fast_tier":"fast-x"}}`))
	}))
	defer srv.Close()
	c, err := newTestClient(srv.URL).Classify(context.Background(), ClassifyRequest{TicketID: "t1", Body: "refund"})
	if err != nil {
		t.Fatal(err)
	}
	if c.Category != "Billing" || c.Urgency != "high" || c.ModelVersions["fast_tier"] != "fast-x" {
		t.Fatalf("decoded %+v", c)
	}
}

func TestObserveOncePerCallWithRetries(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_, _ = w.Write([]byte(healthJSON))
			return
		}
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(`{"chunks":[],"max_score":0}`))
	}))
	defer srv.Close()

	c := newTestClient(srv.URL)
	var stages []string
	var lastErr error
	c.Observe = func(stage string, d time.Duration, err error) {
		stages = append(stages, stage)
		lastErr = err
		if d <= 0 {
			t.Errorf("duration %v", d)
		}
	}
	if _, err := c.Health(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Retrieve(context.Background(), RetrieveRequest{Body: "q"}); err != nil {
		t.Fatal(err)
	}
	if len(stages) != 1 || stages[0] != "retrieve" || lastErr != nil {
		t.Fatalf("stages = %v, err = %v (health is not observed; one call despite a retry)", stages, lastErr)
	}
}

func TestNewAddsSchemeToHostPort(t *testing.T) {
	if got := New("groundedtriage-ml:10000").BaseURL; got != "http://groundedtriage-ml:10000" {
		t.Fatalf("BaseURL = %q", got)
	}
	if got := New("https://ml.example/").BaseURL; got != "https://ml.example" {
		t.Fatalf("BaseURL = %q", got)
	}
}
