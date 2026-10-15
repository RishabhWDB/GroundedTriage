package ingest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func parseFixture(t *testing.T, name string) ParsedEmail {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "email", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := ParseEmail(f)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return p
}

func TestParsePlain(t *testing.T) {
	p := parseFixture(t, "plain.eml")
	if p.Subject != "Cannot log in – password reset" {
		t.Fatalf("RFC 2047 subject not decoded: %q", p.Subject)
	}
	if p.From != "tom.becker@fastmail.com" || p.FromName != "Tom Becker" || p.MessageID != "plain-001@fastmail.com" {
		t.Fatalf("headers = %+v", p)
	}
	if !p.Date.Equal(time.Date(2026, 10, 3, 10, 40, 0, 0, time.UTC)) {
		t.Fatalf("date = %v", p.Date)
	}
	body := p.CleanBody()
	if !strings.HasSuffix(body, "My username is tbecker.") || strings.Contains(body, "Senior Analyst") {
		t.Fatalf("signature not stripped: %q", body)
	}
	tk := p.Ticket("imap:INBOX/1", time.Time{})
	if tk.Source != "email" || tk.ExternalID != "plain-001@fastmail.com" || *tk.ThreadID != tk.ExternalID {
		t.Fatalf("ticket = %+v", tk)
	}
}

func TestParseHTMLOnlyLatin1(t *testing.T) {
	p := parseFixture(t, "html.eml")
	if p.Text != "" || p.HTML == "" {
		t.Fatalf("text=%q html=%q", p.Text, p.HTML)
	}
	want := "Hello,\n\nSince the update to version 4.2 the iOS app crashes immediately on startup at our Café kiosks.\n\nThanks,\nRetail Co IT"
	if got := p.CleanBody(); got != want {
		t.Fatalf("body:\n got %q\nwant %q", got, want)
	}
	if p.FromName != "Retail Co IT" {
		t.Fatalf("from name = %q", p.FromName)
	}
}

func TestParseMultipartPrefersPlainAndSkipsAttachment(t *testing.T) {
	p := parseFixture(t, "multipart.eml")
	if !strings.Contains(p.Text, ".xlsx?") || !strings.Contains(p.HTML, "<b>.xlsx</b>") {
		t.Fatalf("parts not both captured: text=%q html=%q", p.Text, p.HTML)
	}
	if strings.Contains(p.Text+p.HTML, "JVBER") {
		t.Fatal("attachment leaked into body")
	}
	if got := p.CleanBody(); got != "Would it be possible to export the monthly usage report as .xlsx?\nOur finance team prefers Excel." {
		t.Fatalf("body = %q", got)
	}
}

func TestParseReplyChain(t *testing.T) {
	p := parseFixture(t, "reply_chain.eml")
	if p.InReplyTo != "support-ack-77@example.com" ||
		len(p.References) != 2 || p.References[0] != "orig-001@acme-corp.com" {
		t.Fatalf("threading headers: in-reply-to=%q refs=%v", p.InReplyTo, p.References)
	}
	if p.ThreadID() != "orig-001@acme-corp.com" {
		t.Fatalf("thread id = %q", p.ThreadID())
	}
	if got := p.CleanBody(); got != "Following up on this - I still see both charges on my statement." {
		t.Fatalf("quotes/footer not stripped: %q", got)
	}
}

func TestTicketWithoutMessageIDOrDate(t *testing.T) {
	p := ParsedEmail{From: "a@x.com", Subject: " Hi ", Text: "Body"}
	fallback := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	tk := p.Ticket("ref", fallback)
	if !strings.HasPrefix(tk.ExternalID, "nomsgid-") || tk.Subject != "Hi" || !tk.ReceivedAt.Equal(fallback) || tk.ThreadID != nil {
		t.Fatalf("ticket = %+v", tk)
	}
	if again := p.Ticket("ref", fallback); again.ExternalID != tk.ExternalID {
		t.Fatal("hash id not stable")
	}
}
