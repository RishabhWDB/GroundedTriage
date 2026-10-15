package ingest

import (
	"context"
	"encoding/csv"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

const samplePath = "../../datasets/sample_emails.csv"

func TestCSVIngestsSampleFile(t *testing.T) {
	src := &CSVSource{Path: samplePath}
	rows, err := src.ReadRows()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 22 {
		t.Fatalf("rows = %d, want 22", len(rows))
	}
	first := rows[0].Ticket
	if first.Source != "csv" || first.Sender != "priya.n@acme-corp.com" ||
		first.Subject != "Charged twice for October invoice" ||
		!strings.Contains(first.Body, "INV-2291") ||
		!first.ReceivedAt.Equal(time.Date(2026, 10, 3, 9, 12, 0, 0, time.UTC)) ||
		first.RawRef != "sample_emails.csv#line=2" || rows[0].Label != "Billing" {
		t.Fatalf("first row = %+v label=%q", first, rows[0].Label)
	}
	// Multi-line quoted body (row with signature) survives intact.
	if !strings.Contains(rows[2].Ticket.Body, "Platform Ops, Northwind") {
		t.Fatalf("multi-line body truncated: %q", rows[2].Ticket.Body)
	}
	ids := map[string]bool{}
	for _, r := range rows {
		if !strings.HasPrefix(r.Ticket.ExternalID, "csv-") || ids[r.Ticket.ExternalID] {
			t.Fatalf("bad or duplicate external id %q", r.Ticket.ExternalID)
		}
		ids[r.Ticket.ExternalID] = true
	}
	// External ids are stable across reads.
	again, _ := src.ReadRows()
	if again[5].Ticket.ExternalID != rows[5].Ticket.ExternalID {
		t.Fatal("external id not stable")
	}

	got, err := src.Fetch(context.Background(), time.Date(2026, 10, 8, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("Fetch since Oct 8 = %d tickets, want 4", len(got))
	}
}

func writeFile(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "in.csv")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCSVHeaderVariants(t *testing.T) {
	// BOM, lowercase headers, no Label/Date, explicit Id column.
	p := writeFile(t, "\ufeffid,sender,subject,body\nT-1,a@x.com,Hi,Hello there\n")
	rows, err := (&CSVSource{Path: p}).ReadRows()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Ticket.ExternalID != "T-1" || rows[0].Label != "" ||
		rows[0].Ticket.ReceivedAt.IsZero() {
		t.Fatalf("rows = %+v", rows)
	}
}

func TestCSVErrors(t *testing.T) {
	for name, body := range map[string]string{
		"missing body column": "Sender,Subject\na,b\n",
		"bad date":            "Sender,Subject,Body,Date\na,b,c,yesterday\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := (&CSVSource{Path: writeFile(t, body)}).ReadRows(); err == nil {
				t.Fatal("expected error")
			}
		})
	}
	if _, err := (&CSVSource{Path: "nope.csv"}).ReadRows(); err == nil {
		t.Fatal("missing file accepted")
	}
}

func TestCSVReplyAppends(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out", "replies.csv")
	src := &CSVSource{Path: samplePath, ReplyPath: out}
	tk := model.Ticket{ExternalID: "csv-1", Sender: "a@x.com", Subject: "Help"}
	ctx := context.Background()
	for _, body := range []string{"first, with comma", "second\nmultiline"} {
		if err := src.Reply(ctx, tk, model.ReplyDraft{Body: body, Citations: []string{"d:1", "d:2"}}); err != nil {
			t.Fatal(err)
		}
	}
	f, err := os.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 3 || recs[0][0] != "ExternalID" || recs[1][2] != "Re: Help" ||
		recs[2][3] != "second\nmultiline" || recs[1][4] != "d:1 d:2" {
		t.Fatalf("records = %q", recs)
	}
}

func TestCSVFactory(t *testing.T) {
	if _, err := NewCSVFactory(SourceConfig{Type: "csv"}); err == nil {
		t.Fatal("missing path accepted")
	}
	srcs, err := DefaultRegistry().Build([]SourceConfig{{Type: "csv", Enabled: true, Options: map[string]string{"path": samplePath}}})
	if err != nil || srcs["csv"] == nil {
		t.Fatalf("build csv: %v", err)
	}
}
