package mlclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Client calls the ml-service. Each attempt is bounded by Timeout; failed
// attempts (transport errors, 429, 5xx) are retried up to MaxRetries times
// with exponential backoff starting at Backoff. Other 4xx fail immediately.
type Client struct {
	BaseURL    string
	HTTP       *http.Client
	Timeout    time.Duration
	MaxRetries int
	Backoff    time.Duration
	// Observe, when set, is called once per call (retries included) with the
	// stage name (the path without its slash, e.g. "classify"), the
	// wall-clock duration and the final error. Used for monitoring (P7.3).
	Observe func(stage string, d time.Duration, err error)
}

// New returns a Client with defaults suited to a local ml-service.
func New(baseURL string) *Client {
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		HTTP:       &http.Client{},
		Timeout:    10 * time.Second,
		MaxRetries: 3,
		Backoff:    200 * time.Millisecond,
	}
}

// StatusError is a non-2xx response from the ml-service.
type StatusError struct {
	Code int
	Body string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("ml-service: HTTP %d: %s", e.Code, e.Body)
}

func retryable(err error) bool {
	var se *StatusError
	if errors.As(err, &se) {
		return se.Code == http.StatusTooManyRequests || se.Code >= 500
	}
	return true // transport error or per-attempt timeout
}

// Health is the ml-service /health response.
type Health struct {
	Status         string             `json:"status"`
	Service        string             `json:"service"`
	Version        string             `json:"version"`
	ModelsLoaded   model.ModelsLoaded `json:"models_loaded"`
	FastTierLoaded bool               `json:"fast_tier_loaded"`
	TorchInstalled bool               `json:"torch_installed"`
}

// Health calls GET /health.
func (c *Client) Health(ctx context.Context) (Health, error) {
	var h Health
	err := c.do(ctx, http.MethodGet, "/health", nil, &h)
	return h, err
}

func (c *Client) do(ctx context.Context, method, path string, in, out any) (err error) {
	if c.Observe != nil && path != "/health" {
		start := time.Now()
		defer func() { c.Observe(strings.TrimPrefix(path, "/"), time.Since(start), err) }()
	}
	var body []byte
	if in != nil {
		if body, err = json.Marshal(in); err != nil {
			return err
		}
	}
	for attempt := 0; attempt <= c.MaxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return fmt.Errorf("%w (last error: %v)", ctx.Err(), err)
			case <-time.After(c.Backoff << (attempt - 1)):
			}
		}
		err = c.once(ctx, method, path, body, out)
		if err == nil || !retryable(err) || ctx.Err() != nil {
			return err
		}
	}
	return fmt.Errorf("ml-service %s %s: giving up after %d attempts: %w", method, path, c.MaxRetries+1, err)
}

func (c *Client) once(ctx context.Context, method, path string, body []byte, out any) error {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		return &StatusError{Code: resp.StatusCode, Body: strings.TrimSpace(string(data))}
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(data, out)
}

// ClassifyRequest is the POST /classify body.
type ClassifyRequest struct {
	TicketID string `json:"ticket_id"`
	Subject  string `json:"subject"`
	Body     string `json:"body"`
}

// Classify calls POST /classify. A 503 means the ml-service has no fast
// tier (models/fast_tier/ missing); it is retried like any 5xx.
func (c *Client) Classify(ctx context.Context, req ClassifyRequest) (model.Classification, error) {
	var out model.Classification
	err := c.do(ctx, http.MethodPost, "/classify", req, &out)
	return out, err
}

// MeetingRequest is the POST /extract_meeting body.
type MeetingRequest struct {
	TicketID        string    `json:"ticket_id"`
	Subject         string    `json:"subject"`
	Body            string    `json:"body"`
	ReceivedAt      time.Time `json:"received_at"`
	DefaultTimezone string    `json:"default_timezone"`
}

// MeetingExtraction mirrors ml-service app.meeting.MeetingExtraction.
type MeetingExtraction struct {
	TicketID         string     `json:"ticket_id"`
	IsMeetingRequest bool       `json:"is_meeting_request"`
	Start            *time.Time `json:"start"`
	Timezone         string     `json:"timezone"`
	DurationMinutes  *int       `json:"duration_minutes"`
	Venue            string     `json:"venue"`
	VenueDetail      *string    `json:"venue_detail"`
	TimeText         *string    `json:"time_text"`
	Action           string     `json:"action"`
	Notes            []string   `json:"notes"`
}

// ExtractMeeting calls POST /extract_meeting (an LLM call: allow a longer
// Timeout than for /classify).
func (c *Client) ExtractMeeting(ctx context.Context, req MeetingRequest) (MeetingExtraction, error) {
	var out MeetingExtraction
	err := c.do(ctx, http.MethodPost, "/extract_meeting", req, &out)
	return out, err
}

// RetrieveRequest is the POST /retrieve body.
type RetrieveRequest struct {
	TicketID string `json:"ticket_id"`
	Subject  string `json:"subject"`
	Body     string `json:"body"`
	K        int    `json:"k"`
	Hybrid   bool   `json:"hybrid"`
}

// RetrieveResponse is the POST /retrieve result.
type RetrieveResponse struct {
	TicketID string                 `json:"ticket_id"`
	Mode     string                 `json:"mode"`
	MaxScore float64                `json:"max_score"`
	Chunks   []model.RetrievedChunk `json:"chunks"`
	// The faithfulness gate's floor (config/grounding.yaml) and whether
	// MaxScore is below it (then the draft would abstain anyway).
	RetrievalFloor float64 `json:"retrieval_floor"`
	BelowFloor     bool    `json:"below_floor"`
}

// Retrieve calls POST /retrieve.
func (c *Client) Retrieve(ctx context.Context, req RetrieveRequest) (RetrieveResponse, error) {
	var out RetrieveResponse
	err := c.do(ctx, http.MethodPost, "/retrieve", req, &out)
	return out, err
}

// DraftRequest is the POST /draft body.
type DraftRequest struct {
	TicketID string                 `json:"ticket_id"`
	Subject  string                 `json:"subject"`
	Body     string                 `json:"body"`
	Category string                 `json:"category"`
	Urgency  string                 `json:"urgency"`
	Chunks   []model.RetrievedChunk `json:"chunks"`
}

// GeneratedDraft mirrors ml-service rag.generator.GeneratedDraft.
type GeneratedDraft struct {
	TicketID         string   `json:"ticket_id"`
	Subject          string   `json:"subject"`
	Body             string   `json:"body"`
	Citations        []string `json:"citations"`
	InvalidCitations []int    `json:"invalid_citations"`
	Refined          bool     `json:"refined"`
	Notes            []string `json:"notes"`
}

// Draft calls POST /draft (two LLM calls).
func (c *Client) Draft(ctx context.Context, req DraftRequest) (GeneratedDraft, error) {
	var out GeneratedDraft
	err := c.do(ctx, http.MethodPost, "/draft", req, &out)
	return out, err
}

// VerifyRequest is the POST /verify body.
type VerifyRequest struct {
	TicketID  string                 `json:"ticket_id"`
	Subject   string                 `json:"subject"`
	Body      string                 `json:"body"`
	Citations []string               `json:"citations"`
	Chunks    []model.RetrievedChunk `json:"chunks"`
}

// Claim is one judged atomic claim.
type Claim struct {
	Claim     string `json:"claim"`
	Cited     []int  `json:"cited"`
	Supported bool   `json:"supported"`
	Reason    string `json:"reason"`
}

// Verification mirrors ml-service rag.faithfulness.Verification.
type Verification struct {
	Draft             model.ReplyDraft `json:"draft"`
	Claims            []Claim          `json:"claims"`
	AbstainReason     string           `json:"abstain_reason"`
	RetrievalMaxScore float64          `json:"retrieval_max_score"`
}

// Verify calls POST /verify (one judge LLM call, cached by the ml-service).
func (c *Client) Verify(ctx context.Context, req VerifyRequest) (Verification, error) {
	var out Verification
	err := c.do(ctx, http.MethodPost, "/verify", req, &out)
	return out, err
}
