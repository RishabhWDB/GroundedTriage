// Package storetest provides a throwaway, migrated schema in TEST_DATABASE_URL
// for integration tests. It never touches DATABASE_URL.
package storetest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// SchemaName returns a unique schema name like "test_20261015t171500_ab12cd34".
func SchemaName() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return "test_" + time.Now().UTC().Format("20060102t150405") + "_" + hex.EncodeToString(b)
}

// NewSchema creates a uniquely named schema in TEST_DATABASE_URL, runs all
// migrations into it, and returns a pool pinned to it. The schema is dropped
// in t.Cleanup. The test is skipped when TEST_DATABASE_URL is unset.
func NewSchema(t *testing.T) *pgxpool.Pool {
	t.Helper()
	if err := envfile.Load(); err != nil {
		t.Fatalf("load .env: %v", err)
	}
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("TEST_DATABASE_URL is unset")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	schema := SchemaName()
	admin, err := pgx.Connect(ctx, url)
	if err != nil {
		t.Fatalf("connect TEST_DATABASE_URL (Supabase paused?): %v", err)
	}
	if _, err := admin.Exec(ctx, "create schema "+pgx.Identifier{schema}.Sanitize()); err != nil {
		admin.Close(ctx)
		t.Fatalf("create schema: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if _, err := admin.Exec(ctx, "drop schema "+pgx.Identifier{schema}.Sanitize()+" cascade"); err != nil {
			t.Errorf("drop schema %s: %v", schema, err)
		}
		admin.Close(ctx)
	})

	opts := store.Options{Schema: schema}
	db, err := store.OpenSQL(url, opts)
	if err != nil {
		t.Fatalf("open sql: %v", err)
	}
	defer db.Close()
	if _, err := store.MigrateUp(ctx, db); err != nil {
		t.Fatalf("migrate test schema: %v", err)
	}

	pool, err := store.OpenPool(ctx, url, opts)
	if err != nil {
		t.Fatalf("open pool: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}
