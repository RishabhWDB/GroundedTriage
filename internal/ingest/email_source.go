package ingest

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"mime"
	"mime/quotedprintable"
	"os"
	"strings"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// RawEmail is one message fetched from the mailbox.
type RawEmail struct {
	UID          uint32
	InternalDate time.Time // server receipt time
	Raw          []byte
}

// IMAPFetcher returns unseen messages received on or after since's date,
// without marking them seen.
type IMAPFetcher interface {
	FetchUnseen(ctx context.Context, since time.Time) ([]RawEmail, error)
}

// Mailer sends a raw RFC 5322 message.
type Mailer interface {
	Send(ctx context.Context, from string, to []string, msg []byte) error
}

// EmailSource ingests from IMAP and replies via SMTP.
type EmailSource struct {
	Fetcher IMAPFetcher
	Mailer  Mailer
	From    string // our address: replies are sent from it, mail from it is ignored
	Mailbox string
	Now     func() time.Time
	Log     *slog.Logger
}

func (s *EmailSource) Name() string { return model.SourceEmail }

func (s *EmailSource) log() *slog.Logger {
	if s.Log != nil {
		return s.Log
	}
	return slog.Default()
}

// Fetch parses unseen messages and returns those received after since.
// ReceivedAt is the server's internal date (the Date header is sender-
// controlled and only a fallback). Unparseable messages are logged and skipped.
func (s *EmailSource) Fetch(ctx context.Context, since time.Time) ([]model.Ticket, error) {
	raws, err := s.Fetcher.FetchUnseen(ctx, since)
	if err != nil {
		return nil, fmt.Errorf("imap: %w", err)
	}
	own := strings.ToLower(s.From)
	var out []model.Ticket
	for _, raw := range raws {
		ref := fmt.Sprintf("imap:%s/%d", s.Mailbox, raw.UID)
		p, err := ParseEmail(bytes.NewReader(raw.Raw))
		if err != nil {
			s.log().Warn("email: skipping unparseable message", "ref", ref, "err", err)
			continue
		}
		if own != "" && p.From == own {
			continue // our own outbound reply
		}
		t := p.Ticket(ref, raw.InternalDate)
		if !raw.InternalDate.IsZero() {
			t.ReceivedAt = raw.InternalDate.UTC()
		}
		if t.ReceivedAt.After(since) {
			out = append(out, t)
		}
	}
	return out, nil
}

// Reply sends d to the ticket's sender, threaded onto the original message.
func (s *EmailSource) Reply(ctx context.Context, t model.Ticket, d model.ReplyDraft) error {
	if t.Sender == "" {
		return errors.New("email: ticket has no sender")
	}
	now := time.Now
	if s.Now != nil {
		now = s.Now
	}
	msg, err := BuildReply(s.From, t, d, now())
	if err != nil {
		return err
	}
	return s.Mailer.Send(ctx, s.From, []string{t.Sender}, msg)
}

func headerSafe(v string) string {
	return strings.NewReplacer("\r", " ", "\n", " ").Replace(strings.TrimSpace(v))
}

func replySubject(t model.Ticket, d model.ReplyDraft) string {
	if d.Subject != "" {
		return d.Subject
	}
	if strings.HasPrefix(strings.ToLower(t.Subject), "re:") {
		return t.Subject
	}
	return "Re: " + t.Subject
}

// BuildReply renders a text/plain (UTF-8, quoted-printable) reply with
// In-Reply-To/References pointing at the original message and thread root.
func BuildReply(from string, t model.Ticket, d model.ReplyDraft, now time.Time) ([]byte, error) {
	if from == "" {
		return nil, errors.New("email: no From address configured (SMTP_FROM)")
	}
	domain := "localhost"
	if i := strings.LastIndex(from, "@"); i >= 0 {
		domain = from[i+1:]
	}
	idBytes := make([]byte, 12)
	_, _ = rand.Read(idBytes)

	var refs []string
	if t.ThreadID != nil && *t.ThreadID != "" && *t.ThreadID != t.ExternalID {
		refs = append(refs, "<"+*t.ThreadID+">")
	}
	original := ""
	if t.ExternalID != "" && !strings.HasPrefix(t.ExternalID, "nomsgid-") {
		original = "<" + t.ExternalID + ">"
		refs = append(refs, original)
	}

	var b bytes.Buffer
	hdr := func(k, v string) { fmt.Fprintf(&b, "%s: %s\r\n", k, headerSafe(v)) }
	hdr("From", from)
	hdr("To", t.Sender)
	hdr("Subject", mime.QEncoding.Encode("utf-8", headerSafe(replySubject(t, d))))
	hdr("Date", now.UTC().Format(time.RFC1123Z))
	hdr("Message-ID", "<"+hex.EncodeToString(idBytes)+"@"+domain+">")
	if original != "" {
		hdr("In-Reply-To", original)
	}
	if len(refs) > 0 {
		hdr("References", strings.Join(refs, " "))
	}
	hdr("MIME-Version", "1.0")
	hdr("Content-Type", "text/plain; charset=utf-8")
	hdr("Content-Transfer-Encoding", "quoted-printable")
	b.WriteString("\r\n")
	qp := quotedprintable.NewWriter(&b)
	if _, err := qp.Write([]byte(strings.ReplaceAll(d.Body, "\n", "\r\n"))); err != nil {
		return nil, err
	}
	if err := qp.Close(); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// NewEmailFactory builds an EmailSource from IMAP_* / SMTP_* env vars
// (options.mailbox overrides IMAP_MAILBOX).
func NewEmailFactory(c SourceConfig) (Source, error) {
	var missing []string
	for _, k := range []string{"IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD", "SMTP_HOST"} {
		if os.Getenv(k) == "" {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("email: missing env vars %v", missing)
	}
	mailbox := c.Options["mailbox"]
	if mailbox == "" {
		mailbox = envOr("IMAP_MAILBOX", "INBOX")
	}
	from := envOr("SMTP_FROM", os.Getenv("IMAP_USER"))
	return &EmailSource{
		Fetcher: &IMAPClient{
			Addr: os.Getenv("IMAP_HOST") + ":" + envOr("IMAP_PORT", "993"),
			User: os.Getenv("IMAP_USER"), Password: os.Getenv("IMAP_PASSWORD"), Mailbox: mailbox,
		},
		Mailer: &SMTPMailer{
			Host: os.Getenv("SMTP_HOST"), Port: envOr("SMTP_PORT", "587"),
			User: envOr("SMTP_USER", os.Getenv("IMAP_USER")), Password: envOr("SMTP_PASSWORD", os.Getenv("IMAP_PASSWORD")),
		},
		From:    from,
		Mailbox: mailbox,
	}, nil
}
