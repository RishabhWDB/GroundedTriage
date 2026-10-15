package store

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
)

// Options configures a connection.
type Options struct {
	// Schema, when set, is put first in search_path (per-run test schemas).
	Schema string
}

// SearchPath returns the search_path every connection uses. It is always set
// explicitly instead of relying on the Supabase role default: pgvector is
// pre-installed in "extensions" on Supabase and in public on plain Postgres,
// so both are listed (Postgres ignores schemas that don't exist).
func SearchPath(schema string) string {
	base := "public, extensions"
	if schema == "" {
		return base
	}
	return pgx.Identifier{schema}.Sanitize() + ", " + base
}

func afterConnect(schema string) func(context.Context, *pgx.Conn) error {
	stmt := "SET search_path TO " + SearchPath(schema)
	return func(ctx context.Context, c *pgx.Conn) error {
		_, err := c.Exec(ctx, stmt)
		return err
	}
}

// OpenPool opens a pgx pool for application queries.
func OpenPool(ctx context.Context, url string, opts Options) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return nil, fmt.Errorf("parse database url: %w", err)
	}
	cfg.AfterConnect = afterConnect(opts.Schema)
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
	return stdlib.OpenDB(*cfg, stdlib.OptionAfterConnect(afterConnect(opts.Schema))), nil
}
