package store

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/pressly/goose/v3"

	"github.com/RishabhWDB/GroundedTriage/migrations"
)

func newProvider(db *sql.DB) (*goose.Provider, error) {
	return goose.NewProvider(goose.DialectPostgres, db, migrations.FS)
}

// MigrateUp applies all pending migrations and returns how many were applied.
// Only "up" is exposed: down-migrations are never run from code paths that
// could target DATABASE_URL.
func MigrateUp(ctx context.Context, db *sql.DB) (int, error) {
	p, err := newProvider(db)
	if err != nil {
		return 0, err
	}
	results, err := p.Up(ctx)
	if err != nil {
		return len(results), fmt.Errorf("migrate up: %w", err)
	}
	return len(results), nil
}

// MigrationStatus returns "<version> <applied|pending> <file>" lines.
func MigrationStatus(ctx context.Context, db *sql.DB) ([]string, error) {
	p, err := newProvider(db)
	if err != nil {
		return nil, err
	}
	statuses, err := p.Status(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(statuses))
	for _, s := range statuses {
		out = append(out, fmt.Sprintf("%05d %-8s %s", s.Source.Version, s.State, s.Source.Path))
	}
	return out, nil
}
