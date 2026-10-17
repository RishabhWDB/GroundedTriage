// Package ui serves the agent review UI (P6.2) and dashboard (P6.4): static
// HTML/CSS/JS embedded in the binary, calling the REST API with the API key
// the agent enters (kept in the browser's localStorage).
package ui

import (
	"embed"
	"io/fs"
	"net/http"

	"github.com/gin-gonic/gin"
)

//go:embed static
var static embed.FS

// csp allows only this origin's scripts and styles: ticket text is untrusted
// and is only ever inserted with textContent.
const csp = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; " +
	"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

// Register serves the UI at /ui/ and redirects / there.
func Register(r *gin.Engine) {
	sub, _ := fs.Sub(static, "static")
	files := http.FileServer(http.FS(sub))
	r.GET("/", func(c *gin.Context) { c.Redirect(http.StatusFound, "/ui/") })
	r.GET("/ui/*file", func(c *gin.Context) {
		c.Header("Content-Security-Policy", csp)
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("Referrer-Policy", "no-referrer")
		http.StripPrefix("/ui", files).ServeHTTP(c.Writer, c.Request)
	})
}
