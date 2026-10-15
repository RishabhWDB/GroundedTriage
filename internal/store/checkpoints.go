package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Checkpoints stores the per-source ingestion high-water mark.
type Checkpoints struct{ Pool *pgxpool.Pool }

// Get returns the checkpoint for source; ok is false when none is stored.
func (r Checkpoints) Get(ctx context.Context, source string) (t time.Time, ok bool, err error) {
	err = r.Pool.QueryRow(ctx, `select since from ingest_checkpoints where source = $1`, source).Scan(&t)
	if errors.Is(err, pgx.ErrNoRows) {
		return time.Time{}, false, nil
	}
	if err != nil {
		return time.Time{}, false, fmt.Errorf("get checkpoint %s: %w", source, err)
	}
	return t.UTC(), true, nil
}

// Set upserts the checkpoint for source.
func (r Checkpoints) Set(ctx context.Context, source string, since time.Time) error {
	_, err := r.Pool.Exec(ctx, `
		insert into ingest_checkpoints (source, since, updated_at) values ($1, $2, now())
		on conflict (source) do update set since = excluded.since, updated_at = now()`, source, since)
	if err != nil {
		return fmt.Errorf("set checkpoint %s: %w", source, err)
	}
	return nil
}
