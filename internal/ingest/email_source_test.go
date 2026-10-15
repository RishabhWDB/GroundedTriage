package ingest

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

type fakeFetcher struct {
	raws []RawEmail
	err  error
}

func (f *fakeFetcher) FetchUnseen(context.Context, time.Time) ([]RawEmail, error) {
	return f.raws, f.err
}

type sentMail struct {
	from string
	to   []string
	msg  []byte
}

type fakeMailer struct{ sent []sentMail }

func (m *fakeMailer) Send(_ context.Context, from string, to []string, msg []byte) error {
	m.sent = append(m.sent, sentMail{from, to, msg})
	return nil
}

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "email", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestEmailSourceFetch(t *testing.T) {
	day := func(d int) time.Time { return time.Date(2026, 10, d, 12, 0, 0, 0, time.UTC) }
	ownReply := []byte("From: support@example.com\r\nSubject: Re: x\r\nMessage-ID: <own@example.com>\r\n\r\nour reply\r\n")
	src := &EmailSource{
		Fetcher: &fakeFetcher{raws: []RawEmail{
			{UID: 1, InternalDate: day(3), Raw: fixture(t, "plain.eml")},
			{UID: 2, InternalDate: day(7), Raw: fixture(t, "reply_chain.eml")},
			{UID: 3, InternalDate: day(8), Raw: ownReply},
			{UID: 4, InternalDate: day(8), Raw: []byte("Content-Type: multipart/mixed; boundary=\"x\"\r\n\r\n--x\r\nbroken")},
		}},
		From:    "Support@Example.com",
		Mailbox: "INBOX",
	}
	got, err := src.Fetch(context.Background(), day(5))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d tickets, want 1 (since filter, own mail and broken mail skipped): %+v", len(got), got)
	}
	tk := got[0]
	if tk.ExternalID != "reply-004@acme-corp.com" || tk.RawRef != "imap:INBOX/2" ||
		!tk.ReceivedAt.Equal(day(7)) || *tk.ThreadID != "orig-001@acme-corp.com" ||
		tk.Body != "Following up on this - I still see both charges on my statement." {
		t.Fatalf("ticket = %+v", tk)
	}

	src.Fetcher = &fakeFetcher{err: errors.New("auth failed")}
	if _, err := src.Fetch(context.Background(), time.Time{}); err == nil {
		t.Fatal("fetch error swallowed")
	}
}

func TestEmailReplyThreadsAndRoundTrips(t *testing.T) {
	mailer := &fakeMailer{}
	src := &EmailSource{Mailer: mailer, From: "support@example.com",
		Now: func() time.Time { return time.Date(2026, 10, 15, 8, 0, 0, 0, time.UTC) }}
	thread := "orig-001@acme-corp.com"
	tk := model.Ticket{ExternalID: "reply-004@acme-corp.com", ThreadID: &thread,
		Sender: "priya.n@acme-corp.com", Subject: "Re: Charged twice"}
	body := "Hi Priya,\nWe've refunded the duplicate charge — see [1]."
	if err := src.Reply(context.Background(), tk, model.ReplyDraft{Body: body}); err != nil {
		t.Fatal(err)
	}
	if len(mailer.sent) != 1 || mailer.sent[0].to[0] != "priya.n@acme-corp.com" {
		t.Fatalf("sent = %+v", mailer.sent)
	}
	p, err := ParseEmail(bytes.NewReader(mailer.sent[0].msg))
	if err != nil {
		t.Fatal(err)
	}
	if p.Subject != "Re: Charged twice" { // no "Re: Re:"
		t.Fatalf("subject = %q", p.Subject)
	}
	if p.InReplyTo != "reply-004@acme-corp.com" || len(p.References) != 2 ||
		p.References[0] != thread || !strings.HasSuffix(p.MessageID, "@example.com") {
		t.Fatalf("threading: %+v", p)
	}
	if strings.ReplaceAll(p.Text, "\r\n", "\n") != body {
		t.Fatalf("body round trip: %q", p.Text)
	}
}

func TestBuildReplyEdgeCases(t *testing.T) {
	now := time.Now()
	tk := model.Ticket{ExternalID: "nomsgid-abc", Sender: "a@x.com", Subject: "Help\r\nBcc: evil@x.com"}
	msg, err := BuildReply("support@example.com", tk, model.ReplyDraft{Body: "ok"}, now)
	if err != nil {
		t.Fatal(err)
	}
	s := string(msg)
	if strings.Contains(s, "In-Reply-To") || strings.Contains(s, "References") {
		t.Fatal("hash-based id must not be used as In-Reply-To")
	}
	if strings.Contains(s, "\r\nBcc:") {
		t.Fatal("header injection via subject")
	}
	if _, err := BuildReply("", tk, model.ReplyDraft{}, now); err == nil {
		t.Fatal("empty From accepted")
	}
	if err := (&EmailSource{}).Reply(context.Background(), model.Ticket{}, model.ReplyDraft{}); err == nil {
		t.Fatal("reply without sender accepted")
	}
}

func TestEmailFactoryRequiresEnv(t *testing.T) {
	for _, k := range []string{"IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD", "SMTP_HOST"} {
		t.Setenv(k, "")
	}
	if _, err := NewEmailFactory(SourceConfig{Type: "email"}); err == nil || !strings.Contains(err.Error(), "IMAP_HOST") {
		t.Fatalf("err = %v", err)
	}
	t.Setenv("IMAP_HOST", "imap.example.com")
	t.Setenv("IMAP_USER", "support@example.com")
	t.Setenv("IMAP_PASSWORD", "pw")
	t.Setenv("SMTP_HOST", "smtp.example.com")
	t.Setenv("SMTP_FROM", "")
	s, err := NewEmailFactory(SourceConfig{Type: "email", Options: map[string]string{"mailbox": "Support"}})
	if err != nil {
		t.Fatal(err)
	}
	es := s.(*EmailSource)
	if es.Mailbox != "Support" || es.From != "support@example.com" || es.Fetcher.(*IMAPClient).Addr != "imap.example.com:993" {
		t.Fatalf("source = %+v", es)
	}
}

// fakeSMTPServer speaks just enough SMTP (no TLS, no auth) to accept one message.
func fakeSMTPServer(t *testing.T) (addr string, got <-chan string) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	ch := make(chan string, 1)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		r, w := bufio.NewReader(conn), bufio.NewWriter(conn)
		say := func(s string) { w.WriteString(s + "\r\n"); w.Flush() }
		say("220 fake ESMTP")
		var data strings.Builder
		for {
			line, err := r.ReadString('\n')
			if err != nil {
				return
			}
			cmd := strings.ToUpper(strings.TrimSpace(line))
			switch {
			case strings.HasPrefix(cmd, "EHLO"), strings.HasPrefix(cmd, "HELO"):
				say("250 fake")
			case strings.HasPrefix(cmd, "MAIL"), strings.HasPrefix(cmd, "RCPT"):
				data.WriteString(strings.TrimSpace(line) + "\n")
				say("250 ok")
			case cmd == "DATA":
				say("354 go ahead")
				for {
					l, err := r.ReadString('\n')
					if err != nil || l == ".\r\n" {
						break
					}
					data.WriteString(l)
				}
				say("250 queued")
			case cmd == "QUIT":
				say("221 bye")
				ch <- data.String()
				return
			default:
				say("502 unsupported")
			}
		}
	}()
	return ln.Addr().String(), ch
}

func TestSMTPMailerAgainstFakeServer(t *testing.T) {
	addr, got := fakeSMTPServer(t)
	host, port, _ := net.SplitHostPort(addr)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	m := &SMTPMailer{Host: host, Port: port}
	if err := m.Send(ctx, "support@example.com", []string{"a@x.com"}, []byte("Subject: hi\r\n\r\nhello\r\n")); err != nil {
		t.Fatal(err)
	}
	select {
	case s := <-got:
		if !strings.Contains(s, "MAIL FROM:<support@example.com>") || !strings.Contains(s, "RCPT TO:<a@x.com>") || !strings.Contains(s, "hello") {
			t.Fatalf("server saw %q", s)
		}
	case <-ctx.Done():
		t.Fatal("server did not receive message")
	}
}
