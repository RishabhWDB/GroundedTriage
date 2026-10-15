// Package storetest provides one throwaway, migrated schema in
// TEST_DATABASE_URL per test package. It never touches DATABASE_URL.
//
// Usage in a package's integration tests:
//
//	var testDB *storetest.DB
//
//	func TestMain(m *testing.M) { os.Exit(storetest.Run(m, &testDB)) }
//
//	func TestX(t *testing.T) {
//		pool := testDB.Require(t)           // skips if TEST_DATABASE_URL unset
//		id := storetest.UniqueID("ticket")  // isolate by unique ids ...
//		testDB.Truncate(t, "tickets")       // ... or truncate what the test owns
//	}
package storetest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

const schemaPrefix = "test_"

// DB is a migrated schema shared by one test package.
type DB struct {
	Pool       *pgxpool.Pool
	Schema     string
	skipReason string
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// SchemaName returns a unique schema name like "test_20261015t171500_ab12cd34".
func SchemaName() string {
	return schemaPrefix + time.Now().UTC().Format("20060102t150405") + "_" + randHex(4)
}

// UniqueID returns prefix plus a random suffix, for isolating rows between
// tests that share a schema.
func UniqueID(prefix string) string { return prefix + "-" + randHex(6) }

// Run sets up the shared schema, runs the package's tests and drops the schema.
// Use it from TestMain: os.Exit(storetest.Run(m, &testDB)).
func Run(m *testing.M, out **DB) int {
	db, cleanup, err := setup()
	if err != nil {
		log.Printf("storetest: %v", err)
		return 1
	}
	*out = db
	code := m.Run()
	cleanup()
	return code
}

func setup() (*DB, func(), error) {
	if err := envfile.Load(); err != nil {
		return nil, nil, fmt.Errorf("load .env: %w", err)
	}
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		return &DB{skipReason: "TEST_DATABASE_URL is unset"}, func() {}, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	schema := SchemaName()
	ident := pgx.Identifier{schema}.Sanitize()
	admin, err := pgx.Connect(ctx, url)
	if err != nil {
		return nil, nil, fmt.Errorf("connect TEST_DATABASE_URL (Supabase paused?): %w", err)
	}
	if _, err := admin.Exec(ctx, "create schema "+ident); err != nil {
		admin.Close(ctx)
		return nil, nil, fmt.Errorf("create schema: %w", err)
	}
	drop := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if _, err := admin.Exec(ctx, "drop schema "+ident+" cascade"); err != nil {
			log.Printf("storetest: drop schema %s: %v", schema, err)
		}
		admin.Close(ctx)
	}

	opts := store.Options{Schema: schema}
	sqlDB, err := store.OpenSQL(url, opts)
	if err != nil {
		drop()
		return nil, nil, err
	}
	_, err = store.MigrateUp(ctx, sqlDB)
	sqlDB.Close()
	if err != nil {
		drop()
		return nil, nil, fmt.Errorf("migrate test schema: %w", err)
	}
	pool, err := store.OpenPool(ctx, url, opts)
	if err != nil {
		drop()
		return nil, nil, err
	}
	return &DB{Pool: pool, Schema: schema}, func() { pool.Close(); drop() }, nil
}

// Require returns the shared pool, or skips the test when no test database
// is configured.
func (d *DB) Require(t *testing.T) *pgxpool.Pool {
	t.Helper()
	if d == nil {
		t.Fatal("storetest: DB not initialised; call storetest.Run from TestMain")
	}
	if d.Pool == nil {
		t.Skip(d.skipReason)
	}
	return d.Pool
}

// Truncate empties the given tables (restart identity, cascade) in the test
// schema. It refuses to run on anything but a storetest-created schema.
func (d *DB) Truncate(t *testing.T, tables ...string) {
	t.Helper()
	pool := d.Require(t)
	if !strings.HasPrefix(d.Schema, schemaPrefix) {
		t.Fatalf("storetest: refusing to truncate in schema %q", d.Schema)
	}
	idents := make([]string, len(tables))
	for i, tbl := range tables {
		idents[i] = pgx.Identifier{d.Schema, tbl}.Sanitize()
	}
	stmt := "truncate " + strings.Join(idents, ", ") + " restart identity cascade"
	if _, err := pool.Exec(context.Background(), stmt); err != nil {
		t.Fatalf("truncate: %v", err)
	}
}
