package store

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
)

// Options configures a connection. Schema, when set, pins search_path to
// "<schema>, public, extensions" on every connection (used by per-run test
// schemas). public and extensions are kept so the pgvector type resolves:
// Supabase pre-installs pgvector in "extensions"; plain Postgres puts it in
// public. Nonexistent schemas in search_path are ignored by Postgres.
type Options struct {
	Schema string
}

func searchPathSQL(schema string) string {
	return fmt.Sprintf("SET search_path TO %s, public, extensions", pgx.Identifier{schema}.Sanitize())
}

// OpenPool opens a pgx pool for application queries.
func OpenPool(ctx context.Context, url string, opts Options) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return nil, fmt.Errorf("parse database url: %w", err)
	}
	if opts.Schema != "" {
		stmt := searchPathSQL(opts.Schema)
		cfg.AfterConnect = func(ctx context.Context, c *pgx.Conn) error {
			_, err := c.Exec(ctx, stmt)
			return err
		}
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping database (if this is Supabase, check the project isn't paused): %w", err)
	}
	return pool, nil
}

// OpenSQL opens a database/sql handle (needed by goose) with the same
// search_path handling as OpenPool.
func OpenSQL(url string, opts Options) (*sql.DB, error) {
	cfg, err := pgx.ParseConfig(url)
	if err != nil {
		return nil, fmt.Errorf("parse database url: %w", err)
	}
	var options []stdlib.OptionOpenDB
	if opts.Schema != "" {
		stmt := searchPathSQL(opts.Schema)
		options = append(options, stdlib.OptionAfterConnect(func(ctx context.Context, c *pgx.Conn) error {
			_, err := c.Exec(ctx, stmt)
			return err
		}))
	}
	return stdlib.OpenDB(*cfg, options...), nil
}
