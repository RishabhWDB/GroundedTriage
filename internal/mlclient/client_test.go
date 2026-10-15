package mlclient

import (
	"context"
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
