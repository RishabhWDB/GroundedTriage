// Package api is the REST API (P6.1) behind the agent review UI (P6.2) and
// dashboard (P6.4).
//
// Every endpoint except GET /health needs the API key (X-API-Key or
// "Authorization: Bearer <key>"). Errors are always
// {"error": {"code": "...", "message": "..."}}.
package api

import (
	"context"
	"crypto/subtle"
	"errors"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// Store is the data the API reads and changes (store.Review).
type Store interface {
	ListTickets(ctx context.Context, f store.TicketFilter) ([]store.TicketSummary, int, error)
	GetTicket(ctx context.Context, id string) (store.TicketDetail, error)
	LatestDraft(ctx context.Context, ticketID string) (store.DraftView, error)
	Chunks(ctx context.Context, ids []string) ([]store.ChunkView, error)
	ApproveDraft(ctx context.Context, id, reviewer, note string) (store.DraftView, error)
	RejectDraft(ctx context.Context, id, reviewer, note string) (store.DraftView, error)
	EditDraft(ctx context.Context, id, reviewer, subject, body string) (store.DraftView, error)
	Summary(ctx context.Context, now time.Time) (store.Summary, error)
}

// CSVIngester ingests an uploaded CSV file (layer-1 ingestion for one file).
type CSVIngester interface {
	IngestCSV(ctx context.Context, path string) (fetched int, stats preprocess.Stats, err error)
}

// HealthChecker is the ml-service's /health (mlclient.Client).
type HealthChecker interface {
	Health(ctx context.Context) (mlclient.Health, error)
}

// AuditWriter appends audit_log rows (store.Audit).
type AuditWriter interface {
	Write(ctx context.Context, e store.AuditEntry) error
}

// Server holds the API's dependencies.
type Server struct {
	Store     Store
	Ingest    CSVIngester
	ML        HealthChecker
	Audit     AuditWriter
	APIKey    string
	UploadDir string // where uploaded CSVs are kept (gitignored datasets/out/uploads)
	MaxUpload int64  // bytes; default 5 MiB
	Now       func() time.Time
	Log       *slog.Logger
}

// Error codes in the JSON error body.
const (
	CodeBadRequest   = "bad_request"
	CodeUnauthorized = "unauthorized"
	CodeNotFound     = "not_found"
	CodeConflict     = "conflict"
	CodeTooLarge     = "payload_too_large"
	CodeInternal     = "internal"
)

func fail(c *gin.Context, status int, code, msg string) {
	c.AbortWithStatusJSON(status, gin.H{"error": gin.H{"code": code, "message": msg}})
}

func (s *Server) log() *slog.Logger {
	if s.Log != nil {
		return s.Log
	}
	return slog.Default()
}

func (s *Server) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

// storeErr maps store errors to HTTP responses.
func (s *Server) storeErr(c *gin.Context, err error, what string) {
	switch {
	case errors.Is(err, store.ErrNotFound):
		fail(c, http.StatusNotFound, CodeNotFound, what+" not found")
	case errors.Is(err, store.ErrConflict):
		fail(c, http.StatusConflict, CodeConflict, what+" is not in a state that allows this")
	default:
		s.log().Error("api: store error", "what", what, "err", err)
		fail(c, http.StatusInternalServerError, CodeInternal, "internal error")
	}
}

// RequireAPIKey checks X-API-Key or a Bearer token against key.
func RequireAPIKey(key string) gin.HandlerFunc {
	return func(c *gin.Context) {
		got := c.GetHeader("X-API-Key")
		if got == "" {
			got = strings.TrimPrefix(c.GetHeader("Authorization"), "Bearer ")
		}
		if key == "" || subtle.ConstantTimeCompare([]byte(got), []byte(key)) != 1 {
			fail(c, http.StatusUnauthorized, CodeUnauthorized, "missing or wrong API key")
			return
		}
		c.Next()
	}
}

// Register adds the API routes to r (GET /health is the caller's).
func (s *Server) Register(r gin.IRouter) {
	g := r.Group("/", RequireAPIKey(s.APIKey))
	g.GET("/tickets", s.listTickets)
	g.GET("/tickets/:id", s.getTicket)
	g.GET("/tickets/:id/draft", s.getDraft)
	g.POST("/drafts/:id/approve", s.approve)
	g.POST("/drafts/:id/reject", s.reject)
	g.POST("/drafts/:id/edit", s.edit)
	g.POST("/ingest/csv", s.ingestCSV)
	g.GET("/metrics/summary", s.summary)
}

func isUUID(id string) bool {
	if len(id) != 36 {
		return false
	}
	for i, r := range id {
		switch {
		case i == 8 || i == 13 || i == 18 || i == 23:
			if r != '-' {
				return false
			}
		case !strings.ContainsRune("0123456789abcdefABCDEF", r):
			return false
		}
	}
	return true
}

func (s *Server) idParam(c *gin.Context) (string, bool) {
	id := c.Param("id")
	if !isUUID(id) {
		fail(c, http.StatusBadRequest, CodeBadRequest, "id must be a UUID")
		return "", false
	}
	return id, true
}

func (s *Server) listTickets(c *gin.Context) {
	f := store.TicketFilter{Category: c.Query("category"), Urgency: c.Query("urgency"),
		Status: c.Query("status"), Queue: c.Query("queue")}
	if f.Category != "" && !slices.Contains(model.Categories, f.Category) {
		fail(c, http.StatusBadRequest, CodeBadRequest, "unknown category")
		return
	}
	if f.Urgency != "" && !slices.Contains(model.UrgencyScale, f.Urgency) {
		fail(c, http.StatusBadRequest, CodeBadRequest, "urgency must be one of low, medium, high, critical")
		return
	}
	if f.Status != "" && !slices.Contains([]string{"open", "resolved", "unrouted"}, f.Status) {
		fail(c, http.StatusBadRequest, CodeBadRequest, "status must be open, resolved or unrouted")
		return
	}
	var err error
	if f.Limit, err = intQuery(c, "limit", 50, 1, 200); err != nil {
		fail(c, http.StatusBadRequest, CodeBadRequest, err.Error())
		return
	}
	if f.Offset, err = intQuery(c, "offset", 0, 0, 1_000_000); err != nil {
		fail(c, http.StatusBadRequest, CodeBadRequest, err.Error())
		return
	}
	items, total, err := s.Store.ListTickets(c.Request.Context(), f)
	if err != nil {
		s.storeErr(c, err, "tickets")
		return
	}
	c.JSON(http.StatusOK, gin.H{"items": items, "total": total, "limit": f.Limit, "offset": f.Offset})
}

func intQuery(c *gin.Context, key string, def, lo, hi int) (int, error) {
	v := c.Query(key)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < lo || n > hi {
		return 0, errors.New(key + " must be an integer in [" + strconv.Itoa(lo) + ", " + strconv.Itoa(hi) + "]")
	}
	return n, nil
}

func (s *Server) getTicket(c *gin.Context) {
	id, ok := s.idParam(c)
	if !ok {
		return
	}
	d, err := s.Store.GetTicket(c.Request.Context(), id)
	if err != nil {
		s.storeErr(c, err, "ticket")
		return
	}
	c.JSON(http.StatusOK, d)
}

// citedChunk is a retrieved chunk with its [n] number and whether the draft cites it.
type citedChunk struct {
	store.ChunkView
	Number int  `json:"number"`
	Cited  bool `json:"cited"`
}

func (s *Server) getDraft(c *gin.Context) {
	id, ok := s.idParam(c)
	if !ok {
		return
	}
	d, err := s.Store.LatestDraft(c.Request.Context(), id)
	if err != nil {
		s.storeErr(c, err, "draft")
		return
	}
	chunks, err := s.Store.Chunks(c.Request.Context(), d.RetrievedChunkIDs)
	if err != nil {
		s.storeErr(c, err, "chunks")
		return
	}
	// The drafter numbered the retrieved chunks [1..k] in retrieval order.
	pos := map[string]int{}
	for i, cid := range d.RetrievedChunkIDs {
		pos[cid] = i + 1
	}
	out := make([]citedChunk, 0, len(chunks))
	for _, ch := range chunks {
		out = append(out, citedChunk{ChunkView: ch, Number: pos[ch.ID], Cited: slices.Contains(d.Citations, ch.ID)})
	}
	c.JSON(http.StatusOK, gin.H{"draft": d, "chunks": out})
}

type reviewBody struct {
	Reviewer string `json:"reviewer"`
	Note     string `json:"note"`
	Subject  string `json:"subject"`
	Body     string `json:"body"`
}

func (s *Server) reviewInput(c *gin.Context, needBody bool) (string, reviewBody, bool) {
	id, ok := s.idParam(c)
	if !ok {
		return "", reviewBody{}, false
	}
	var b reviewBody
	if err := c.ShouldBindJSON(&b); err != nil {
		fail(c, http.StatusBadRequest, CodeBadRequest, "body must be JSON: "+err.Error())
		return "", b, false
	}
	b.Reviewer = strings.TrimSpace(b.Reviewer)
	if b.Reviewer == "" {
		fail(c, http.StatusBadRequest, CodeBadRequest, "reviewer is required")
		return "", b, false
	}
	if needBody && strings.TrimSpace(b.Body) == "" {
		fail(c, http.StatusBadRequest, CodeBadRequest, "body is required")
		return "", b, false
	}
	return id, b, true
}

func (s *Server) afterReview(c *gin.Context, action string, d store.DraftView, err error) {
	if err != nil {
		s.storeErr(c, err, "draft")
		return
	}
	if s.Audit != nil {
		if aerr := s.Audit.Write(c.Request.Context(), store.AuditEntry{Actor: d.Reviewer, EntityType: "draft",
			EntityID: d.ID, Action: action, Details: map[string]any{"ticket_id": d.TicketID, "status": d.Status,
				"note": d.ReviewNote, "edited": d.Edited}}); aerr != nil {
			s.log().Warn("api: audit write failed", "err", aerr)
		}
	}
	c.JSON(http.StatusOK, d)
}

func (s *Server) approve(c *gin.Context) {
	if id, b, ok := s.reviewInput(c, false); ok {
		d, err := s.Store.ApproveDraft(c.Request.Context(), id, b.Reviewer, b.Note)
		s.afterReview(c, "draft.approved", d, err)
	}
}

func (s *Server) reject(c *gin.Context) {
	if id, b, ok := s.reviewInput(c, false); ok {
		d, err := s.Store.RejectDraft(c.Request.Context(), id, b.Reviewer, b.Note)
		s.afterReview(c, "draft.rejected", d, err)
	}
}

func (s *Server) edit(c *gin.Context) {
	if id, b, ok := s.reviewInput(c, true); ok {
		d, err := s.Store.EditDraft(c.Request.Context(), id, b.Reviewer, b.Subject, b.Body)
		s.afterReview(c, "draft.edited", d, err)
	}
}

func (s *Server) summary(c *gin.Context) {
	sum, err := s.Store.Summary(c.Request.Context(), s.now())
	if err != nil {
		s.storeErr(c, err, "summary")
		return
	}
	resp := gin.H{"summary": sum}
	if s.ML != nil {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()
		if h, err := s.ML.Health(ctx); err == nil {
			resp["models"] = gin.H{"fast_tier_loaded": h.FastTierLoaded, "deep_tier": h.ModelsLoaded.DeepTier,
				"urgency": h.ModelsLoaded.Urgency, "ml_service": "ok"}
		} else {
			resp["models"] = gin.H{"ml_service": "unreachable"}
		}
	}
	c.JSON(http.StatusOK, resp)
}
