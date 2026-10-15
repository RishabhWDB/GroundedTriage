package ingest

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/smtp"
	"time"

	"github.com/emersion/go-imap"
	"github.com/emersion/go-imap/client"
)

// IMAPClient is the real IMAPFetcher (implicit TLS, e.g. port 993). It opens
// a connection per poll, selects the mailbox read-only and fetches with
// BODY.PEEK[] so messages are not marked seen.
type IMAPClient struct {
	Addr, User, Password, Mailbox string
	Timeout                       time.Duration
}

func (c *IMAPClient) FetchUnseen(ctx context.Context, since time.Time) ([]RawEmail, error) {
	timeout := c.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}
	dialer := &net.Dialer{Timeout: timeout}
	cl, err := client.DialWithDialerTLS(dialer, c.Addr, nil)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", c.Addr, err)
	}
	cl.Timeout = timeout
	defer cl.Logout()
	stop := context.AfterFunc(ctx, func() { _ = cl.Terminate() })
	defer stop()

	if err := cl.Login(c.User, c.Password); err != nil {
		return nil, fmt.Errorf("login: %w", err)
	}
	if _, err := cl.Select(c.Mailbox, true); err != nil {
		return nil, fmt.Errorf("select %s: %w", c.Mailbox, err)
	}
	crit := imap.NewSearchCriteria()
	crit.WithoutFlags = []string{imap.SeenFlag}
	if !since.IsZero() {
		crit.Since = since // IMAP SINCE is day-granular; Fetch filters precisely
	}
	uids, err := cl.UidSearch(crit)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	if len(uids) == 0 {
		return nil, nil
	}
	seq := new(imap.SeqSet)
	seq.AddNum(uids...)
	section := &imap.BodySectionName{Peek: true}
	items := []imap.FetchItem{imap.FetchUid, imap.FetchInternalDate, section.FetchItem()}

	msgs := make(chan *imap.Message, 16)
	done := make(chan error, 1)
	go func() { done <- cl.UidFetch(seq, items, msgs) }()
	var out []RawEmail
	for m := range msgs {
		body := m.GetBody(section)
		if body == nil {
			continue
		}
		raw, err := io.ReadAll(body)
		if err != nil {
			return nil, err
		}
		out = append(out, RawEmail{UID: m.Uid, InternalDate: m.InternalDate, Raw: raw})
	}
	if err := <-done; err != nil {
		return nil, fmt.Errorf("fetch: %w", err)
	}
	return out, ctx.Err()
}

// SMTPMailer is the real Mailer: implicit TLS on port 465, otherwise
// STARTTLS when offered (e.g. 587), with PLAIN auth when a user is set.
type SMTPMailer struct {
	Host, Port, User, Password string
}

func (m *SMTPMailer) Send(ctx context.Context, from string, to []string, msg []byte) error {
	addr := net.JoinHostPort(m.Host, m.Port)
	tlsCfg := &tls.Config{ServerName: m.Host}
	var conn net.Conn
	var err error
	d := &net.Dialer{Timeout: 30 * time.Second}
	if m.Port == "465" {
		conn, err = (&tls.Dialer{NetDialer: d, Config: tlsCfg}).DialContext(ctx, "tcp", addr)
	} else {
		conn, err = d.DialContext(ctx, "tcp", addr)
	}
	if err != nil {
		return fmt.Errorf("smtp dial %s: %w", addr, err)
	}
	if dl, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(dl)
	}
	c, err := smtp.NewClient(conn, m.Host)
	if err != nil {
		conn.Close()
		return err
	}
	defer c.Close()
	if m.Port != "465" {
		if ok, _ := c.Extension("STARTTLS"); ok {
			if err := c.StartTLS(tlsCfg); err != nil {
				return fmt.Errorf("smtp starttls: %w", err)
			}
		}
	}
	if m.User != "" {
		if err := c.Auth(smtp.PlainAuth("", m.User, m.Password, m.Host)); err != nil {
			return fmt.Errorf("smtp auth: %w", err)
		}
	}
	if err := c.Mail(from); err != nil {
		return err
	}
	for _, r := range to {
		if err := c.Rcpt(r); err != nil {
			return err
		}
	}
	w, err := c.Data()
	if err != nil {
		return err
	}
	if _, err := w.Write(msg); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	return c.Quit()
}
