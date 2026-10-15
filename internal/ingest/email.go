package ingest

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/emersion/go-message"
	_ "github.com/emersion/go-message/charset" // non-UTF-8 charsets (iso-8859-1, windows-1252, ...)
	"github.com/emersion/go-message/mail"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
)

// ParsedEmail is the subset of an RFC 5322 message the adapter uses.
// Message ids are stored without angle brackets.
type ParsedEmail struct {
	MessageID  string
	InReplyTo  string
	References []string
	From       string // address only
	FromName   string
	Subject    string
	Date       time.Time
	Text       string // first inline text/plain part
	HTML       string // first inline text/html part
}

// ParseEmail reads a raw message, decoding transfer encodings and charsets.
// It walks nested multiparts and ignores attachments.
func ParseEmail(r io.Reader) (ParsedEmail, error) {
	var p ParsedEmail
	mr, err := mail.CreateReader(r)
	if err != nil && !message.IsUnknownCharset(err) {
		return p, fmt.Errorf("email: %w", err)
	}
	defer mr.Close()

	h := mr.Header
	if p.Subject, err = h.Subject(); err != nil {
		p.Subject = h.Get("Subject")
	}
	if from, err := h.AddressList("From"); err == nil && len(from) > 0 {
		p.From, p.FromName = strings.ToLower(from[0].Address), from[0].Name
	} else {
		p.From = strings.TrimSpace(h.Get("From"))
	}
	p.Date, _ = h.Date()
	p.MessageID, _ = h.MessageID()
	if ids, err := h.MsgIDList("In-Reply-To"); err == nil && len(ids) > 0 {
		p.InReplyTo = ids[0]
	}
	p.References, _ = h.MsgIDList("References")

	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if message.IsUnknownCharset(err) {
				continue
			}
			return p, fmt.Errorf("email: read part: %w", err)
		}
		ih, ok := part.Header.(*mail.InlineHeader)
		if !ok {
			continue // attachment
		}
		ct, _, _ := ih.ContentType()
		if ct != "text/plain" && ct != "text/html" && ct != "" {
			continue
		}
		body, err := io.ReadAll(io.LimitReader(part.Body, 5<<20))
		if err != nil {
			return p, fmt.Errorf("email: read body: %w", err)
		}
		switch {
		case ct == "text/html" && p.HTML == "":
			p.HTML = string(body)
		case ct != "text/html" && p.Text == "":
			p.Text = string(body)
		}
	}
	return p, nil
}

// CleanBody returns the plain-text body: text/plain if present, else the
// HTML converted to text; then quoted replies and signatures are stripped.
func (p ParsedEmail) CleanBody() string {
	body := p.Text
	if strings.TrimSpace(body) == "" && p.HTML != "" {
		body = preprocess.HTMLToText(p.HTML)
	}
	return preprocess.StripSignature(preprocess.StripQuoted(body))
}

// ThreadID is the thread root: the first References id, else In-Reply-To,
// else the message's own id.
func (p ParsedEmail) ThreadID() string {
	switch {
	case len(p.References) > 0:
		return p.References[0]
	case p.InReplyTo != "":
		return p.InReplyTo
	default:
		return p.MessageID
	}
}

// Ticket converts the email into a Ticket. ExternalID is the Message-ID (or a
// content hash when missing); fallbackDate is used when Date is absent.
func (p ParsedEmail) Ticket(rawRef string, fallbackDate time.Time) model.Ticket {
	received := p.Date
	if received.IsZero() {
		received = fallbackDate
	}
	body := p.CleanBody()
	id := p.MessageID
	if id == "" {
		sum := sha256.Sum256([]byte(p.From + "\x1f" + p.Subject + "\x1f" + body))
		id = "nomsgid-" + hex.EncodeToString(sum[:8])
	}
	t := model.Ticket{
		Source:     model.SourceEmail,
		ExternalID: id,
		Sender:     p.From,
		Subject:    strings.TrimSpace(p.Subject),
		Body:       body,
		ReceivedAt: received.UTC(),
		RawRef:     rawRef,
	}
	if th := p.ThreadID(); th != "" {
		t.ThreadID = &th
	}
	return t
}
