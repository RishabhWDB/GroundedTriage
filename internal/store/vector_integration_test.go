//go:build integration

package store_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

// vec returns a 384-dim pgvector literal with value v at index hot, 0 elsewhere.
func vec(hot int, v float64) string {
	parts := make([]string, 384)
	for i := range parts {
		parts[i] = "0"
	}
	parts[hot] = fmt.Sprint(v)
	return "[" + strings.Join(parts, ",") + "]"
}

// TestCosineQueryResolvesVector guards against search_path regressions: the
// vector type and the <=> operator must resolve on an app connection.
func TestCosineQueryResolvesVector(t *testing.T) {
	pool := testDB.Require(t)
	ctx := context.Background()
	doc := storetest.UniqueID("doc")

	if _, err := pool.Exec(ctx, `insert into kb_documents (id, title) values ($1, 'Doc')`, doc); err != nil {
		t.Fatalf("insert doc: %v", err)
	}
	for i, hot := range []int{0, 1} {
		_, err := pool.Exec(ctx, `
			insert into kb_chunks (id, doc_id, ordinal, text, embedding)
			values ($1, $2, $3, $4, $5::vector)`,
			fmt.Sprintf("%s:%d", doc, i), doc, i, fmt.Sprintf("chunk %d", i), vec(hot, 1))
		if err != nil {
			t.Fatalf("insert chunk %d: %v", i, err)
		}
	}

	var id string
	var score float64
	err := pool.QueryRow(ctx, `
		select id, 1 - (embedding <=> $1::vector) as score
		from kb_chunks where doc_id = $2
		order by embedding <=> $1::vector
		limit 1`, vec(1, 0.5), doc).Scan(&id, &score)
	if err != nil {
		t.Fatalf("cosine query: %v", err)
	}
	if id != doc+":1" || score < 0.999 {
		t.Fatalf("top hit = %s (score %.4f), want %s:1 with score ~1", id, score, doc)
	}
}
