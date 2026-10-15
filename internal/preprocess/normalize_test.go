package preprocess

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func TestNormalizeWhitespace(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"crlf and cr", "a\r\nb\rc", "a\nb\nc"},
		{"inline runs and tabs", "a  \t b\u00a0\u00a0c", "a b c"},
		{"trim lines", "  a  \n   b ", "a\nb"},
		{"collapse blank lines", "a\n\n\n\n b", "a\n\nb"},
		{"zero-width and soft hyphen", "pass\u200bword re\u00adset\ufeff", "password reset"},
		{"nfc", "Cafe\u0301", "Café"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := NormalizeWhitespace(tc.in); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
	if got := NormalizeSubject("  Re:\tBilling \n issue  "); got != "Re: Billing issue" {
		t.Fatalf("subject = %q", got)
	}
}

func TestCleanBodyIsIdempotentAndHandlesHTML(t *testing.T) {
	in := "<p>App <b>crashes</b>.</p><blockquote>old</blockquote>"
	once := CleanBody(in)
	if once != "App crashes." || CleanBody(once) != once {
		t.Fatalf("once=%q twice=%q", once, CleanBody(once))
	}
	quoted := "Still broken.\n\n-- \nSig\n"
	if got := CleanBody(quoted); got != "Still broken." {
		t.Fatalf("got %q", got)
	}
}

func TestContentHash(t *testing.T) {
	base := ContentHash("A@x.com", "Charged twice", "I was  billed\n twice.")
	same := []struct{ name, sender, subject, body string }{
		{"case and whitespace", "a@x.com ", "charged  twice", "I was billed twice."},
		{"reply prefix", "a@x.com", "Re: Fwd: Charged twice", "i was billed twice."},
		{"numbered prefix", "a@x.com", "RE[2]: Charged twice", "I was billed twice."},
	}
	for _, tc := range same {
		if got := ContentHash(tc.sender, tc.subject, tc.body); got != base {
			t.Errorf("%s: hash differs", tc.name)
		}
	}
	diff := []struct{ name, sender, subject, body string }{
		{"other sender", "b@x.com", "Charged twice", "I was billed twice."},
		{"other body", "a@x.com", "Charged twice", "I was billed three times."},
	}
	for _, tc := range diff {
		if ContentHash(tc.sender, tc.subject, tc.body) == base {
			t.Errorf("%s: hash should differ", tc.name)
		}
	}
	if len(base) != 64 {
		t.Fatalf("hash length %d", len(base))
	}
}

func TestDetectLanguage(t *testing.T) {
	cases := []struct {
		name, text string
		nonEnglish bool
	}{
		{"english", "I was billed twice for my invoice this month, can you refund the duplicate charge please?", false},
		{"portuguese", "Olá, a fatura de setembro veio com um valor diferente do contratado. Podem verificar, por favor? Obrigado.", true},
		{"german", "Hallo, ich kann mich seit gestern nicht mehr in mein Konto einloggen. Bitte helfen Sie mir schnell.", true},
		{"short non-english is not checked", "Obrigado!", false},
		{"short english", "ok thanks", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, non := DetectLanguage(tc.text); non != tc.nonEnglish {
				t.Fatalf("nonEnglish = %v, want %v", non, tc.nonEnglish)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	when := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	cases := []struct {
		name     string
		in       model.Ticket
		skip     string
		wantBody string
	}{
		{"cleans and hashes", model.Ticket{Sender: " Priya@Acme.com ", Subject: " Help ", Body: "<p>Hi there, my invoice is wrong.</p>", ReceivedAt: when}, "", "Hi there, my invoice is wrong."},
		{"empty", model.Ticket{Sender: "a@x.com", Subject: "  ", Body: "\n \u200b "}, SkipEmpty, ""},
		{"non-english", model.Ticket{Sender: "a@x.com", Subject: "Problema com a fatura", Body: "Olá, a fatura de setembro veio com um valor diferente do contratado. Podem verificar?"}, SkipNonEnglish, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, skip, detail := Normalize(tc.in)
			if skip != tc.skip {
				t.Fatalf("skip = %q (%s), want %q", skip, detail, tc.skip)
			}
			if out.ContentHash == "" {
				t.Fatal("content hash not set")
			}
			if tc.skip == "" && (out.Body != tc.wantBody || out.Sender != "priya@acme.com" || out.Subject != "Help") {
				t.Fatalf("out = %+v", out)
			}
			if tc.skip == SkipNonEnglish && !strings.Contains(detail, "por") {
				t.Fatalf("detail = %q", detail)
			}
		})
	}
}

// memStore is an in-memory TicketStore with store.Tickets' duplicate semantics.
type memStore struct {
	byKey   map[string]bool
	byHash  map[string]bool
	n       int
	failErr error
}

func newMemStore() *memStore { return &memStore{byKey: map[string]bool{}, byHash: map[string]bool{}} }

func (m *memStore) Insert(_ context.Context, t model.Ticket) (string, error) {
	if m.failErr != nil {
		return "", m.failErr
	}
	key := t.Source + "/" + t.ExternalID
	if m.byKey[key] {
		return "", store.ErrDuplicate
	}
	m.byKey[key], m.byHash[t.ContentHash] = true, true
	m.n++
	return key, nil
}

func (m *memStore) ContentHashExists(_ context.Context, h string) (bool, error) {
	return m.byHash[h], nil
}

func TestIngestorOutcomes(t *testing.T) {
	tk := func(id, sender, body string) model.Ticket {
		return model.Ticket{Source: "csv", ExternalID: id, Sender: sender, Subject: "Billing question", Body: body}
	}
	batch := []model.Ticket{
		tk("1", "a@x.com", "Why was I charged twice this month?"),
		tk("1", "a@x.com", "Different body, same external id."),
		tk("2", "A@X.com", "Why was I  charged twice this month?"), // same content, new id
		tk("3", "b@x.com", ""),
		{Source: "csv", ExternalID: "4", Sender: "c@x.com"},
		tk("5", "d@x.com", "Olá, a fatura de setembro veio com um valor diferente do contratado. Podem verificar?"),
	}
	ms := newMemStore()
	res, stats, err := (&Ingestor{Store: ms}).Process(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{OutcomeInserted, OutcomeDuplicateID, OutcomeDuplicateContent, OutcomeInserted, "skipped", "skipped"}
	for i, w := range want {
		if res[i].Outcome != w {
			t.Errorf("ticket %d outcome = %q (%s), want %q", i, res[i].Outcome, res[i].Reason, w)
		}
	}
	if stats != (Stats{Inserted: 2, DuplicateID: 1, DuplicateContent: 1, Skipped: 2}) {
		t.Fatalf("stats = %+v", stats)
	}
	if res[0].Ticket.ID == "" {
		t.Fatal("inserted ticket id not set")
	}

	ms.failErr = errors.New("db down")
	if _, _, err := (&Ingestor{Store: ms}).Process(context.Background(), []model.Ticket{tk("9", "z@x.com", "new")}); err == nil {
		t.Fatal("storage error swallowed")
	}
}
