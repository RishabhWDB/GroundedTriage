//go:build integration

package store_test

import (
	"context"
	"os"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func TestVerifySchemaOnMigratedSchema(t *testing.T) {
	testDB.Require(t)
	db, err := store.OpenSQL(os.Getenv("TEST_DATABASE_URL"), store.Options{Schema: testDB.Schema})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	r, err := store.VerifySchema(context.Background(), db)
	if err != nil {
		t.Fatal(err)
	}
	if !r.OK() {
		t.Fatalf("schema not OK: %+v", r)
	}
	if r.VectorSchema != "extensions" && r.VectorSchema != "public" {
		t.Errorf("vector in unexpected schema %q", r.VectorSchema)
	}
}
