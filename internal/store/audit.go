package store

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

// AuditEntry is one audit_log row.
type AuditEntry struct {
	Actor      string // defaults to "system"
	EntityType string
	EntityID   string
	Action     string
	Details    map[string]any
}

// Audit writes to audit_log.
type Audit struct{ Pool *pgxpool.Pool }

// Write appends e to audit_log.
func (r Audit) Write(ctx context.Context, e AuditEntry) error {
	if e.Actor == "" {
		e.Actor = "system"
	}
	details, err := json.Marshal(e.Details)
	if err != nil {
		return err
	}
	if e.Details == nil {
		details = []byte("{}")
	}
	_, err = r.Pool.Exec(ctx, `
		insert into audit_log (actor, entity_type, entity_id, action, details)
		values ($1, $2, $3, $4, $5)`, e.Actor, e.EntityType, e.EntityID, e.Action, details)
	if err != nil {
		return fmt.Errorf("write audit %s: %w", e.Action, err)
	}
	return nil
}
