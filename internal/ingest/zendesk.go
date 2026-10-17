package ingest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// ZendeskSource ingests tickets through the cursor-based incremental export
// API (/api/v2/incremental/tickets/cursor.json, users sideloaded) and replies
// with a public comment.
type ZendeskSource struct {
	BaseURL  string // https://<subdomain>.zendesk.com
	Email    string
	APIToken string
	HTTP     *http.Client

	MaxRetries      int           // for 429 (and 5xx on reads)
	MaxPages        int           // per Fetch; the rest is picked up next poll
	InitialLookback time.Duration // used when since is zero
	Now             func() time.Time
	Sleep           func(context.Context, time.Duration) error
}

func (z *ZendeskSource) Name() string { return model.SourceZendesk }

type zdTicket struct {
	ID          int64     `json:"id"`
	Subject     string    `json:"subject"`
	Description string    `json:"description"`
	Status      string    `json:"status"`
	RequesterID int64     `json:"requester_id"`
	CreatedAt   time.Time `json:"created_at"`
}

type zdUser struct {
	ID    int64  `json:"id"`
	Email string `json:"email"`
}

type zdPage struct {
	Tickets     []zdTicket `json:"tickets"`
	Users       []zdUser   `json:"users"`
	AfterCursor string     `json:"after_cursor"`
	EndOfStream bool       `json:"end_of_stream"`
}

// Statuses that still need triage; solved/closed/deleted tickets are skipped.
var zdActive = map[string]bool{"new": true, "open": true, "pending": true, "hold": true}

func (z *ZendeskSource) now() time.Time {
	if z.Now != nil {
		return z.Now()
	}
	return time.Now()
}

func (z *ZendeskSource) sleep(ctx context.Context, d time.Duration) error {
	if z.Sleep != nil {
		return z.Sleep(ctx, d)
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(d):
		return nil
	}
}

// startTime clamps since into what the API accepts: not newer than one
// minute ago, and a bounded lookback on the first poll.
func (z *ZendeskSource) startTime(since time.Time) int64 {
	now := z.now()
	if since.IsZero() {
		lb := z.InitialLookback
		if lb == 0 {
			lb = 30 * 24 * time.Hour
		}
		since = now.Add(-lb)
	}
	if latest := now.Add(-61 * time.Second); since.After(latest) {
		since = latest
	}
	return since.Unix()
}

// Fetch pages through the incremental export starting at since.
func (z *ZendeskSource) Fetch(ctx context.Context, since time.Time) ([]model.Ticket, error) {
	maxPages := z.MaxPages
	if maxPages == 0 {
		maxPages = 50
	}
	q := url.Values{"start_time": {strconv.FormatInt(z.startTime(since), 10)}, "include": {"users"}}
	var out []model.Ticket
	for page := 0; page < maxPages; page++ {
		var p zdPage
		if err := z.do(ctx, http.MethodGet, "/api/v2/incremental/tickets/cursor.json?"+q.Encode(), nil, &p, true); err != nil {
			return nil, err
		}
		emails := make(map[int64]string, len(p.Users))
		for _, u := range p.Users {
			emails[u.ID] = strings.ToLower(u.Email)
		}
		for _, t := range p.Tickets {
			if !zdActive[t.Status] {
				continue
			}
			sender := emails[t.RequesterID]
			if sender == "" {
				sender = fmt.Sprintf("zendesk-user:%d", t.RequesterID)
			}
			id := strconv.FormatInt(t.ID, 10)
			out = append(out, model.Ticket{
				Source:     model.SourceZendesk,
				ExternalID: id,
				Sender:     sender,
				Subject:    t.Subject,
				Body:       t.Description,
				ReceivedAt: t.CreatedAt.UTC(),
				ThreadID:   &id,
				RawRef:     "zendesk:ticket/" + id,
			})
		}
		if p.EndOfStream || p.AfterCursor == "" {
			break
		}
		q = url.Values{"cursor": {p.AfterCursor}, "include": {"users"}}
	}
	return out, nil
}

// Reply posts d.Body as a public comment on the ticket.
func (z *ZendeskSource) Reply(ctx context.Context, t model.Ticket, d model.ReplyDraft) error {
	if _, err := strconv.ParseInt(t.ExternalID, 10, 64); err != nil {
		return fmt.Errorf("zendesk: external id %q is not a ticket id", t.ExternalID)
	}
	body := map[string]any{"ticket": map[string]any{
		"comment": map[string]any{"body": d.Body, "public": true},
	}}
	var hdr map[string]string
	if d.IdempotencyKey != "" { // lets Zendesk drop a retried request
		hdr = map[string]string{"Idempotency-Key": d.IdempotencyKey}
	}
	return z.do(ctx, http.MethodPut, "/api/v2/tickets/"+t.ExternalID+".json", body, nil, false, hdr)
}

// ZendeskError is a non-2xx API response.
type ZendeskError struct {
	Status int
	Body   string
}

func (e *ZendeskError) Error() string { return fmt.Sprintf("zendesk: HTTP %d: %s", e.Status, e.Body) }

// do sends one API request. 429 is always retried (honouring Retry-After);
// 5xx and transport errors only when idempotent (a failed PUT may have
// applied, and retrying could post the comment twice).
func (z *ZendeskSource) do(ctx context.Context, method, path string, in, out any, idempotent bool,
	headers ...map[string]string) error {
	var payload []byte
	if in != nil {
		var err error
		if payload, err = json.Marshal(in); err != nil {
			return err
		}
	}
	client := z.HTTP
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	retries := z.MaxRetries
	if retries == 0 {
		retries = 5
	}
	var lastErr error
	for attempt := 0; attempt <= retries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(z.BaseURL, "/")+path, bytes.NewReader(payload))
		if err != nil {
			return err
		}
		req.SetBasicAuth(z.Email+"/token", z.APIToken)
		req.Header.Set("Accept", "application/json")
		for _, h := range headers {
			for k, v := range h {
				req.Header.Set(k, v)
			}
		}
		if payload != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := client.Do(req)
		wait := time.Duration(1<<attempt) * time.Second
		switch {
		case err != nil:
			if !idempotent || ctx.Err() != nil {
				return fmt.Errorf("zendesk: %w", err)
			}
			lastErr = err
		default:
			data, readErr := io.ReadAll(io.LimitReader(resp.Body, 20<<20))
			resp.Body.Close()
			if readErr != nil {
				return readErr
			}
			if resp.StatusCode/100 == 2 {
				if out == nil {
					return nil
				}
				return json.Unmarshal(data, out)
			}
			lastErr = &ZendeskError{Status: resp.StatusCode, Body: strings.TrimSpace(string(data))}
			retryable := resp.StatusCode == http.StatusTooManyRequests || (idempotent && resp.StatusCode >= 500)
			if !retryable {
				return lastErr
			}
			if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s >= 0 {
				wait = time.Duration(s) * time.Second
			}
		}
		if attempt < retries {
			if err := z.sleep(ctx, wait); err != nil {
				return err
			}
		}
	}
	return fmt.Errorf("zendesk: giving up after %d attempts: %w", retries+1, lastErr)
}

// NewZendeskFactory builds a ZendeskSource from ZENDESK_SUBDOMAIN,
// ZENDESK_EMAIL and ZENDESK_API_TOKEN (options.base_url overrides the URL).
func NewZendeskFactory(c SourceConfig) (Source, error) {
	var missing []string
	for _, k := range []string{"ZENDESK_EMAIL", "ZENDESK_API_TOKEN"} {
		if os.Getenv(k) == "" {
			missing = append(missing, k)
		}
	}
	base := c.Options["base_url"]
	if base == "" {
		sub := os.Getenv("ZENDESK_SUBDOMAIN")
		if sub == "" {
			missing = append(missing, "ZENDESK_SUBDOMAIN")
		}
		base = "https://" + sub + ".zendesk.com"
	}
	if len(missing) > 0 {
		return nil, errors.New("zendesk: missing env vars " + strings.Join(missing, ", "))
	}
	return &ZendeskSource{BaseURL: base, Email: os.Getenv("ZENDESK_EMAIL"), APIToken: os.Getenv("ZENDESK_API_TOKEN")}, nil
}
