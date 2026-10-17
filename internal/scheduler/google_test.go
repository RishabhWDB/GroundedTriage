package scheduler

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

func TestFreeBusyParsesPrimaryKeyedByEmail(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/freeBusy" || r.Method != http.MethodPost {
			t.Errorf("%s %s", r.Method, r.URL.Path)
		}
		var in map[string]any
		_ = json.NewDecoder(r.Body).Decode(&in)
		if in["timeMin"] != "2026-10-13T08:00:00Z" {
			t.Errorf("timeMin %v", in["timeMin"])
		}
		_, _ = w.Write([]byte(`{"calendars":{"arya@example.com":{"busy":[
			{"start":"2026-10-15T14:00:00Z","end":"2026-10-15T15:00:00Z"}]}}}`))
	}))
	defer srv.Close()
	g := &GoogleCalendar{HTTP: srv.Client(), CalendarID: "primary", BaseURL: srv.URL}
	busy, err := g.FreeBusy(context.Background(), time.Date(2026, 10, 13, 8, 0, 0, 0, time.UTC), time.Date(2026, 10, 27, 8, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(busy) != 1 || !busy[0].Start.Equal(time.Date(2026, 10, 15, 14, 0, 0, 0, time.UTC)) {
		t.Fatalf("busy %+v", busy)
	}
}

func TestFreeBusyErrors(t *testing.T) {
	for name, body := range map[string]string{
		"calendar error": `{"calendars":{"team@x":{"errors":[{"reason":"notFound"}]}}}`,
		"missing":        `{"calendars":{"a@x":{},"b@x":{}}}`,
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(body)) }))
		g := &GoogleCalendar{HTTP: srv.Client(), CalendarID: "team@x", BaseURL: srv.URL}
		if _, err := g.FreeBusy(context.Background(), time.Now(), time.Now().Add(time.Hour)); err == nil {
			t.Errorf("%s: no error", name)
		}
		srv.Close()
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"code":401,"message":"Invalid Credentials"}}`))
	}))
	defer srv.Close()
	g := &GoogleCalendar{HTTP: srv.Client(), CalendarID: "primary", BaseURL: srv.URL}
	_, err := g.FreeBusy(context.Background(), time.Now(), time.Now().Add(time.Hour))
	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "Invalid Credentials") {
		t.Fatalf("err %v", err)
	}
}

func TestCreateEventPayload(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/calendars/team@x/events" || r.URL.Query().Get("sendUpdates") != "all" {
			t.Errorf("path %s query %s", r.URL.EscapedPath(), r.URL.RawQuery)
		}
		var in struct {
			Summary   string            `json:"summary"`
			Location  string            `json:"location"`
			Start     map[string]string `json:"start"`
			Attendees []map[string]string
		}
		_ = json.NewDecoder(r.Body).Decode(&in)
		if in.Summary != "Q4 review" || in.Location != "Zoom" || in.Start["dateTime"] != "2026-10-15T15:00:00+01:00" ||
			len(in.Attendees) != 1 || in.Attendees[0]["email"] != "a@b.com" {
			t.Errorf("payload %+v", in)
		}
		_, _ = w.Write([]byte(`{"id":"evt123"}`))
	}))
	defer srv.Close()
	g := &GoogleCalendar{HTTP: srv.Client(), CalendarID: "team@x", BaseURL: srv.URL}
	start := time.Date(2026, 10, 15, 15, 0, 0, 0, london)
	id, err := g.CreateEvent(context.Background(), Event{Summary: "Q4 review", Location: "Zoom", Start: start,
		End: start.Add(30 * time.Minute), Attendees: []string{"a@b.com"}})
	if err != nil || id != "evt123" {
		t.Fatalf("id %q err %v", id, err)
	}
}

func TestCallbackHandler(t *testing.T) {
	codes, errs := make(chan string, 1), make(chan error, 1)
	h := CallbackHandler("s1", codes, errs)
	get := func(q string) int {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/callback?"+q, nil))
		return rec.Code
	}
	if get("state=wrong&code=c") != http.StatusBadRequest || !strings.Contains((<-errs).Error(), "state mismatch") {
		t.Fatal("state mismatch accepted")
	}
	if get("error=access_denied") != http.StatusBadRequest || !strings.Contains((<-errs).Error(), "access_denied") {
		t.Fatal("error param accepted")
	}
	if get("state=s1") != http.StatusBadRequest {
		t.Fatal("missing code accepted")
	}
	if get("state=s1&code=good") != http.StatusOK || <-codes != "good" {
		t.Fatal("valid callback rejected")
	}
}

// The whole loopback flow against a fake Google token endpoint.
func TestInstalledAppFlowSavesToken(t *testing.T) {
	tokenSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		form, _ := url.ParseQuery(string(body))
		if form.Get("code") != "the-code" || form.Get("code_verifier") == "" {
			t.Errorf("exchange form %v", form)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"at","refresh_token":"rt","token_type":"Bearer","expires_in":3600}`))
	}))
	defer tokenSrv.Close()
	cfg := &oauth2.Config{ClientID: "cid", ClientSecret: "cs", Scopes: []string{"s"},
		Endpoint: oauth2.Endpoint{AuthURL: "https://accounts.example/auth", TokenURL: tokenSrv.URL}}
	tokenFile := filepath.Join(t.TempDir(), "token.json")
	show := func(authURL string) {
		u, _ := url.Parse(authURL)
		q := u.Query()
		if q.Get("code_challenge_method") != "S256" || q.Get("access_type") != "offline" {
			t.Errorf("auth url %s", authURL)
		}
		go func() { // the browser redirect
			resp, err := http.Get(q.Get("redirect_uri") + "?state=" + q.Get("state") + "&code=the-code")
			if err == nil {
				resp.Body.Close()
			}
		}()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tok, err := RunInstalledAppFlow(ctx, cfg, tokenFile, show)
	if err != nil || tok.AccessToken != "at" || tok.RefreshToken != "rt" {
		t.Fatalf("token %+v err %v", tok, err)
	}
	info, err := os.Stat(tokenFile)
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("token file %v err %v", info, err)
	}
	saved, err := LoadToken(tokenFile)
	if err != nil || saved.RefreshToken != "rt" {
		t.Fatalf("saved %+v err %v", saved, err)
	}
}

func TestNewGoogleCalendarNeedsFiles(t *testing.T) {
	dir := t.TempDir()
	if _, err := NewGoogleCalendar(context.Background(), filepath.Join(dir, "none.json"), filepath.Join(dir, "t.json"), "primary"); err == nil {
		t.Fatal("missing credentials accepted")
	}
	creds := filepath.Join(dir, "credentials.json")
	_ = os.WriteFile(creds, []byte(`{"installed":{"client_id":"c","client_secret":"s","auth_uri":"https://a","token_uri":"https://t","redirect_uris":["http://localhost"]}}`), 0o600)
	_, err := NewGoogleCalendar(context.Background(), creds, filepath.Join(dir, "t.json"), "primary")
	if err == nil || !strings.Contains(err.Error(), "calendar-auth") {
		t.Fatalf("missing token: %v", err)
	}
}

func TestCreateEventEscapesCalendarID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.EscapedPath() != "/calendars/en.uk%23holiday@group/events" {
			t.Errorf("escaped path %s", r.URL.EscapedPath())
		}
		_, _ = w.Write([]byte(`{"id":"e"}`))
	}))
	defer srv.Close()
	g := &GoogleCalendar{HTTP: srv.Client(), CalendarID: "en.uk#holiday@group", BaseURL: srv.URL}
	if _, err := g.CreateEvent(context.Background(), Event{Start: time.Now(), End: time.Now()}); err != nil {
		t.Fatal(err)
	}
}
