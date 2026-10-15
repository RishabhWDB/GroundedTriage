package store

import (
	"context"
	"database/sql"
	"fmt"
)

// RequiredTables are the tables the migrations create; VerifySchema checks each.
var RequiredTables = []string{
	"tickets", "classifications", "routes", "kb_documents", "kb_chunks",
	"reply_drafts", "audit_log", "ingest_checkpoints",
}

// HNSWIndex is the vector index on kb_chunks.embedding (migration 00002).
const HNSWIndex = "kb_chunks_embedding_hnsw_idx"

// SchemaReport is the read-only result of VerifySchema.
type SchemaReport struct {
	VectorSchema  string // schema holding the vector extension ("" if missing)
	VectorVersion string
	MissingTables []string
	HNSWMethod    string // access method of HNSWIndex ("" if missing)
	// Tickets per "source/ingest_status[/skip_reason]" (empty if the table is missing).
	TicketCounts map[string]int
}

// OK reports whether everything the migrations should create is present.
func (r SchemaReport) OK() bool {
	return r.VectorSchema != "" && len(r.MissingTables) == 0 && r.HNSWMethod == "hnsw"
}

// VerifySchema checks, read-only, that the vector extension, every required
// table and the HNSW index exist, resolving names via the connection's
// search_path.
func VerifySchema(ctx context.Context, db *sql.DB) (SchemaReport, error) {
	var r SchemaReport
	err := db.QueryRowContext(ctx, `
		select n.nspname, e.extversion
		from pg_extension e join pg_namespace n on n.oid = e.extnamespace
		where e.extname = 'vector'`).Scan(&r.VectorSchema, &r.VectorVersion)
	if err != nil && err != sql.ErrNoRows {
		return r, fmt.Errorf("vector extension: %w", err)
	}
	for _, t := range RequiredTables {
		var found sql.NullString
		if err := db.QueryRowContext(ctx, `select to_regclass($1)::text`, t).Scan(&found); err != nil {
			return r, fmt.Errorf("table %s: %w", t, err)
		}
		if !found.Valid {
			r.MissingTables = append(r.MissingTables, t)
		}
	}
	var method sql.NullString
	err = db.QueryRowContext(ctx, `
		select am.amname
		from pg_class c join pg_am am on am.oid = c.relam
		where c.oid = to_regclass($1)`, HNSWIndex).Scan(&method)
	if err != nil && err != sql.ErrNoRows {
		return r, fmt.Errorf("index %s: %w", HNSWIndex, err)
	}
	r.HNSWMethod = method.String
	if len(r.MissingTables) > 0 {
		return r, nil
	}
	rows, err := db.QueryContext(ctx, `
		select source, ingest_status, skip_reason, count(*)
		from tickets group by 1, 2, 3 order by 1, 2, 3`)
	if err != nil {
		return r, fmt.Errorf("ticket counts: %w", err)
	}
	defer rows.Close()
	r.TicketCounts = map[string]int{}
	for rows.Next() {
		var src, status, reason string
		var n int
		if err := rows.Scan(&src, &status, &reason, &n); err != nil {
			return r, err
		}
		key := src + "/" + status
		if reason != "" {
			key += "/" + reason
		}
		r.TicketCounts[key] = n
	}
	return r, rows.Err()
}
