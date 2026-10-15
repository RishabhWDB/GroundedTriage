//go:build live

package mlclient

import (
	"context"
	"os"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
)

// TestLiveHealth hits a running ml-service (make ml / make dev):
//
//	go test -tags live -count=1 ./internal/mlclient -run TestLiveHealth
func TestLiveHealth(t *testing.T) {
	_ = envfile.Load()
	url := os.Getenv("ML_SERVICE_URL")
	if url == "" {
		url = "http://127.0.0.1:8001"
	}
	h, err := New(url).Health(context.Background())
	if err != nil {
		t.Fatalf("ml-service at %s: %v", url, err)
	}
	if h.Status != "ok" {
		t.Fatalf("status = %q", h.Status)
	}
	t.Logf("ml-service %s %s models_loaded=%+v", url, h.Version, h.ModelsLoaded)
}
