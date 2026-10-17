package scheduler

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
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// CalendarScope lets the app read free/busy and create events.
const CalendarScope = "https://www.googleapis.com/auth/calendar.events https://www.googleapis.com/auth/calendar.freebusy"

// DefaultCalendarAPI is the Google Calendar v3 REST base URL.
const DefaultCalendarAPI = "https://www.googleapis.com/calendar/v3"

// GoogleCalendar implements Calendar over the Calendar v3 REST API with an
// OAuth2 HTTP client (see NewGoogleCalendar / RunInstalledAppFlow).
type GoogleCalendar struct {
	HTTP       *http.Client
	CalendarID string // "primary" or an id
	BaseURL    string // DefaultCalendarAPI unless testing
}

// OAuthConfig reads the installed-app ("Desktop app") client secrets file.
func OAuthConfig(credentialsFile string) (*oauth2.Config, error) {
	data, err := os.ReadFile(credentialsFile)
	if err != nil {
		return nil, fmt.Errorf("google credentials: %w (download the OAuth client JSON for a Desktop app)", err)
	}
	return google.ConfigFromJSON(data, strings.Fields(CalendarScope)...)
}

// NewGoogleCalendar builds a client from the credentials and a saved token
// (written by `go run ./cmd/calendar-auth`). Refreshed tokens are saved back.
func NewGoogleCalendar(ctx context.Context, credentialsFile, tokenFile, calendarID string) (*GoogleCalendar, error) {
	cfg, err := OAuthConfig(credentialsFile)
	if err != nil {
		return nil, err
	}
	tok, err := LoadToken(tokenFile)
	if err != nil {
		return nil, fmt.Errorf("google token: %w (run `go run ./cmd/calendar-auth` first)", err)
	}
	ts := &savingTokenSource{base: cfg.TokenSource(ctx, tok), path: tokenFile, last: tok}
	return &GoogleCalendar{HTTP: oauth2.NewClient(ctx, ts), CalendarID: calendarID, BaseURL: DefaultCalendarAPI}, nil
}

// LoadToken reads a saved OAuth token.
func LoadToken(path string) (*oauth2.Token, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var t oauth2.Token
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

// SaveToken writes a token readable only by the owner.
func SaveToken(path string, t *oauth2.Token) error {
	data, err := json.Marshal(t)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

// savingTokenSource persists refreshed tokens so the next run reuses them.
type savingTokenSource struct {
	mu   sync.Mutex
	base oauth2.TokenSource
	path string
	last *oauth2.Token
}

func (s *savingTokenSource) Token() (*oauth2.Token, error) {
	t, err := s.base.Token()
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.last == nil || t.AccessToken != s.last.AccessToken {
		_ = SaveToken(s.path, t)
		s.last = t
	}
	return t, nil
}

type apiError struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (g *GoogleCalendar) post(ctx context.Context, path string, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(g.BaseURL, "/")+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := g.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		var ae apiError
		_ = json.Unmarshal(data, &ae)
		return fmt.Errorf("google calendar %s: HTTP %d: %s", path, resp.StatusCode, ae.Error.Message)
	}
	return json.Unmarshal(data, out)
}

// FreeBusy implements Calendar.
func (g *GoogleCalendar) FreeBusy(ctx context.Context, from, to time.Time) ([]Interval, error) {
	in := map[string]any{
		"timeMin": from.UTC().Format(time.RFC3339),
		"timeMax": to.UTC().Format(time.RFC3339),
		"items":   []map[string]string{{"id": g.CalendarID}},
	}
	var out struct {
		Calendars map[string]struct {
			Busy []struct {
				Start time.Time `json:"start"`
				End   time.Time `json:"end"`
			} `json:"busy"`
			Errors []struct {
				Reason string `json:"reason"`
			} `json:"errors"`
		} `json:"calendars"`
	}
	if err := g.post(ctx, "/freeBusy", in, &out); err != nil {
		return nil, err
	}
	cal, ok := out.Calendars[g.CalendarID]
	if !ok && len(out.Calendars) == 1 { // "primary" comes back keyed by the account's address
		for _, c := range out.Calendars {
			cal, ok = c, true
		}
	}
	if !ok {
		return nil, fmt.Errorf("freebusy: calendar %q missing from the response", g.CalendarID)
	}
	if len(cal.Errors) > 0 {
		return nil, fmt.Errorf("freebusy: calendar %q: %s", g.CalendarID, cal.Errors[0].Reason)
	}
	busy := make([]Interval, 0, len(cal.Busy))
	for _, b := range cal.Busy {
		busy = append(busy, Interval{Start: b.Start, End: b.End})
	}
	return busy, nil
}

// CreateEvent implements Calendar; attendees get Google's invitation email.
func (g *GoogleCalendar) CreateEvent(ctx context.Context, e Event) (string, error) {
	attendees := make([]map[string]string, 0, len(e.Attendees))
	for _, a := range e.Attendees {
		attendees = append(attendees, map[string]string{"email": a})
	}
	in := map[string]any{
		"summary":     e.Summary,
		"description": e.Description,
		"location":    e.Location,
		"start":       map[string]string{"dateTime": e.Start.Format(time.RFC3339)},
		"end":         map[string]string{"dateTime": e.End.Format(time.RFC3339)},
		"attendees":   attendees,
	}
	var out struct {
		ID string `json:"id"`
	}
	path := "/calendars/" + url.PathEscape(g.CalendarID) + "/events?sendUpdates=all"
	if err := g.post(ctx, path, in, &out); err != nil {
		return "", err
	}
	if out.ID == "" {
		return "", errors.New("create event: response has no id")
	}
	return out.ID, nil
}
