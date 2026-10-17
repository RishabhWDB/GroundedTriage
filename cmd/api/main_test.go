package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
)

func TestHealthReportsMLService(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ml := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok","service":"ml-service","models_loaded":{"deep_tier":false,"urgency":false}}`))
	}))
	defer ml.Close()

	for _, tc := range []struct {
		name, url, want string
	}{
		{"up", ml.URL, "ok"},
		{"down", "http://127.0.0.1:1", "unreachable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := mlclient.New(tc.url)
			c.MaxRetries = 0
			rec := httptest.NewRecorder()
			newRouter(c, nil, false).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
			if rec.Code != http.StatusOK {
				t.Fatalf("code = %d", rec.Code)
			}
			var body struct {
				Status    string `json:"status"`
				MLService struct {
					Status string `json:"status"`
				} `json:"ml_service"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body.Status != "ok" || body.MLService.Status != tc.want {
				t.Fatalf("body = %s", rec.Body.String())
			}
		})
	}
}
