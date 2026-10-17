package ingest

import (
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// CSVSource reads SCRIBE-compatible CSV files: required columns Sender,
// Subject, Body; optional Label, Date (RFC 3339 or "2006-01-02 15:04:05", UTC)
// and Id. Header names are case-insensitive. Reply appends to ReplyPath.
type CSVSource struct {
	Path      string
	ReplyPath string

	mu sync.Mutex // serialises writes to ReplyPath
}

// CSVRow is one parsed row; Label is kept for evaluation tooling and is not
// part of the Ticket contract.
type CSVRow struct {
	Ticket model.Ticket
	Label  string
}

// NewCSVFactory builds CSV sources from config options "path" and "reply_path".
func NewCSVFactory(c SourceConfig) (Source, error) {
	path := c.Options["path"]
	if path == "" {
		return nil, errors.New("csv: options.path is required")
	}
	reply := c.Options["reply_path"]
	if reply == "" {
		reply = "datasets/out/replies.csv"
	}
	return &CSVSource{Path: path, ReplyPath: reply}, nil
}

func (s *CSVSource) Name() string { return model.SourceCSV }

var csvDateLayouts = []string{time.RFC3339, "2006-01-02 15:04:05", "2006-01-02"}

func parseCSVDate(v string) (time.Time, error) {
	for _, layout := range csvDateLayouts {
		if t, err := time.Parse(layout, v); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised date %q", v)
}

// ReadRows parses the whole file. Rows without a Date column get the file's
// modification time.
func (s *CSVSource) ReadRows() ([]CSVRow, error) {
	f, err := os.Open(s.Path)
	if err != nil {
		return nil, fmt.Errorf("csv: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	r.LazyQuotes = true

	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("csv: read header of %s: %w", s.Path, err)
	}
	col := map[string]int{}
	for i, h := range header {
		h = strings.ToLower(strings.TrimSpace(strings.TrimPrefix(h, "\ufeff")))
		col[h] = i
	}
	for _, req := range []string{"sender", "subject", "body"} {
		if _, ok := col[req]; !ok {
			return nil, fmt.Errorf("csv: %s: missing required column %q (have %v)", s.Path, req, header)
		}
	}
	get := func(rec []string, name string) string {
		if i, ok := col[name]; ok && i < len(rec) {
			return strings.TrimSpace(rec[i])
		}
		return ""
	}

	var rows []CSVRow
	for line := 2; ; line++ {
		rec, err := r.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("csv: %s: %w", s.Path, err)
		}
		received := info.ModTime().UTC()
		if d := get(rec, "date"); d != "" {
			if received, err = parseCSVDate(d); err != nil {
				return nil, fmt.Errorf("csv: %s line %d: %w", s.Path, line, err)
			}
		}
		t := model.Ticket{
			Source:     model.SourceCSV,
			Sender:     get(rec, "sender"),
			Subject:    get(rec, "subject"),
			Body:       get(rec, "body"),
			ReceivedAt: received,
			RawRef:     fmt.Sprintf("%s#line=%d", filepath.Base(s.Path), line),
		}
		t.ExternalID = get(rec, "id")
		if t.ExternalID == "" {
			sum := sha256.Sum256([]byte(strings.Join(
				[]string{t.Sender, t.Subject, t.Body, received.Format(time.RFC3339)}, "\x1f")))
			t.ExternalID = "csv-" + hex.EncodeToString(sum[:8])
		}
		rows = append(rows, CSVRow{Ticket: t, Label: get(rec, "label")})
	}
	return rows, nil
}

// Fetch returns rows received after since. Content hashing and cleaning
// happen in preprocess.
func (s *CSVSource) Fetch(ctx context.Context, since time.Time) ([]model.Ticket, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	rows, err := s.ReadRows()
	if err != nil {
		return nil, err
	}
	var out []model.Ticket
	for _, r := range rows {
		if r.Ticket.ReceivedAt.After(since) {
			out = append(out, r.Ticket)
		}
	}
	return out, nil
}

var replyHeader = []string{"ExternalID", "To", "Subject", "Body", "Citations", "SentAt", "IdempotencyKey"}

// Reply appends the draft to ReplyPath (header written on first use). A
// draft whose IdempotencyKey is already in the file is not written again.
func (s *CSVSource) Reply(ctx context.Context, t model.Ticket, d model.ReplyDraft) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(s.ReplyPath), 0o755); err != nil {
		return err
	}
	if d.IdempotencyKey != "" {
		sent, err := s.keyWritten(d.IdempotencyKey)
		if err != nil {
			return err
		}
		if sent {
			return nil
		}
	}
	_, statErr := os.Stat(s.ReplyPath)
	f, err := os.OpenFile(s.ReplyPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if errors.Is(statErr, os.ErrNotExist) {
		if err := w.Write(replyHeader); err != nil {
			return err
		}
	}
	subject := d.Subject
	if subject == "" {
		subject = "Re: " + t.Subject
	}
	if err := w.Write([]string{
		t.ExternalID, t.Sender, subject, d.Body,
		strings.Join(d.Citations, " "), time.Now().UTC().Format(time.RFC3339), d.IdempotencyKey,
	}); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}

// keyWritten reports whether a reply with this idempotency key is in ReplyPath.
func (s *CSVSource) keyWritten(key string) (bool, error) {
	f, err := os.Open(s.ReplyPath)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	header, err := r.Read()
	if err != nil {
		return false, nil // empty file
	}
	col := -1
	for i, h := range header {
		if h == "IdempotencyKey" {
			col = i
		}
	}
	if col < 0 {
		return false, fmt.Errorf("csv: %s has no IdempotencyKey column (written by an older version); move it aside", s.ReplyPath)
	}
	for {
		rec, err := r.Read()
		if errors.Is(err, io.EOF) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		if col < len(rec) && rec[col] == key {
			return true, nil
		}
	}
}
