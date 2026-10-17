package api

import (
	"context"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
)

// FileIngester ingests one CSV file through the same normalisation and
// dedupe as the CSV source (preprocess.Ingestor).
type FileIngester struct {
	Ingestor *preprocess.Ingestor
}

// IngestCSV implements CSVIngester.
func (f FileIngester) IngestCSV(ctx context.Context, path string) (int, preprocess.Stats, error) {
	tickets, err := (&ingest.CSVSource{Path: path}).Fetch(ctx, time.Time{})
	if err != nil {
		return 0, preprocess.Stats{}, err
	}
	_, stats, err := f.Ingestor.Process(ctx, tickets)
	return len(tickets), stats, err
}
