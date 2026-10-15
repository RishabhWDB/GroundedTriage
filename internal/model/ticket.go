// Package model holds the shared data contracts (mirrored by
// ml-service/app/schemas.py). Keep the two in sync.
package model

import "time"

// Source identifiers for Ticket.Source.
const (
	SourceEmail   = "email"
	SourceZendesk = "zendesk"
	SourceCSV     = "csv"
)

// Ingest statuses for Ticket.IngestStatus. Skipped tickets are stored for
// visibility (and routed to a human) but never classified.
const (
	IngestAccepted = "accepted"
	IngestSkipped  = "skipped"
)

// Ticket is one inbound message from any ingestion source.
type Ticket struct {
	ID          string    `json:"id"`
	Source      string    `json:"source"`
	ExternalID  string    `json:"external_id"`
	Sender      string    `json:"sender"`
	Subject     string    `json:"subject"`
	Body        string    `json:"body"`
	ReceivedAt  time.Time `json:"received_at"`
	ThreadID    *string   `json:"thread_id,omitempty"`
	RawRef      string    `json:"raw_ref"`
	ContentHash string    `json:"content_hash"`
	// IngestStatus is "accepted" (default when empty) or "skipped";
	// SkipReason explains a skip (e.g. "non_english").
	IngestStatus string `json:"ingest_status"`
	SkipReason   string `json:"skip_reason,omitempty"`
}
