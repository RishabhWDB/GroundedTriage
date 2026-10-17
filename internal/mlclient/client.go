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

func (c *Client) do(ctx context.Context, method, path string, in, out any) error {
	var body []byte
	if in != nil {
		var err error
		if body, err = json.Marshal(in); err != nil {
			return err
		}
	}
	var err error
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
