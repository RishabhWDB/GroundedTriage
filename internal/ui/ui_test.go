package ui

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestServesUIWithSecurityHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	Register(r)
	get := func(path string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		return rec
	}
	if rec := get("/"); rec.Code != http.StatusFound || rec.Header().Get("Location") != "/ui/" {
		t.Fatalf("root: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	idx := get("/ui/")
	if idx.Code != 200 || !strings.Contains(idx.Body.String(), `<script src="/ui/app.js" defer></script>`) {
		t.Fatalf("index: %d", idx.Code)
	}
	csp := idx.Header().Get("Content-Security-Policy")
	if !strings.Contains(csp, "script-src 'self'") || strings.Contains(csp, "unsafe-inline") {
		t.Fatalf("csp %q", csp)
	}
	if idx.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatal("nosniff missing")
	}
	js := get("/ui/app.js")
	if js.Code != 200 || !strings.Contains(js.Header().Get("Content-Type"), "javascript") {
		t.Fatalf("app.js: %d %s", js.Code, js.Header().Get("Content-Type"))
	}
	// Untrusted ticket text must never reach an HTML sink.
	for _, sink := range []string{".innerHTML", "insertAdjacentHTML", ".outerHTML", "document.write"} {
		if strings.Contains(js.Body.String(), sink) {
			t.Fatalf("app.js uses %s", sink)
		}
	}
	if get("/ui/style.css").Code != 200 || get("/ui/missing.js").Code != 404 {
		t.Fatal("static file serving")
	}
}
