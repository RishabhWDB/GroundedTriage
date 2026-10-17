package store

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

// DemoIDPrefix tags demo tickets: CSV rows whose Id (external_id) starts
// with it. make demo-reset deletes only these (P8.1).
const DemoIDPrefix = "demo-"

// demoTickets selects the demo-tagged tickets; nothing else is ever touched.
const demoTickets = `select id from tickets where source = 'csv' and external_id like 'demo-%'`

// Demo reads and resets the demo-tagged tickets.
type Demo struct{ Pool *pgxpool.Pool }

// DemoCounts is what a reset deletes.
type DemoCounts struct {
	Tickets, Classifications, Routes, Drafts, Meetings, Audit int
}

// auditOfDemo matches audit rows about a demo ticket: by entity id, by a
// ticket_id in the details, or a draft row about one of its drafts. Run-level rows
// (e.g. classify.run listing many ids) are history and are kept.
const auditOfDemo = `from audit_log where entity_id in (select id::text from demo)
	or details->>'ticket_id' in (select id::text from demo)
	or (entity_type = 'draft' and entity_id in
	    (select id::text from reply_drafts where ticket_id in (select id from demo)))`

// Counts reports the demo rows per table.
func (d Demo) Counts(ctx context.Context) (DemoCounts, error) {
	var c DemoCounts
	err := d.Pool.QueryRow(ctx, `with demo as (`+demoTickets+`)
		select (select count(*) from demo),
		       (select count(*) from classifications where ticket_id in (select id from demo)),
		       (select count(*) from routes where ticket_id in (select id from demo)),
		       (select count(*) from reply_drafts where ticket_id in (select id from demo)),
		       (select count(*) from meetings where ticket_id in (select id from demo)),
		       (select count(*) `+auditOfDemo+`)`).
		Scan(&c.Tickets, &c.Classifications, &c.Routes, &c.Drafts, &c.Meetings, &c.Audit)
	return c, err
}

// Reset deletes the demo tickets (their classifications, routes, drafts and
// meetings cascade) and their audit rows, in one transaction.
func (d Demo) Reset(ctx context.Context) (DemoCounts, error) {
	c, err := d.Counts(ctx)
	if err != nil {
		return c, err
	}
	tx, err := d.Pool.Begin(ctx)
	if err != nil {
		return c, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, `with demo as (`+demoTickets+`) delete `+auditOfDemo); err != nil {
		return c, fmt.Errorf("demo reset audit: %w", err)
	}
	tag, err := tx.Exec(ctx, `delete from tickets where source = 'csv' and external_id like 'demo-%'`)
	if err != nil {
		return c, fmt.Errorf("demo reset tickets: %w", err)
	}
	if int(tag.RowsAffected()) != c.Tickets {
		return c, fmt.Errorf("demo reset: expected %d tickets, would delete %d; nothing changed", c.Tickets, tag.RowsAffected())
	}
	return c, tx.Commit(ctx)
}

// DemoRow is one demo ticket's current pipeline state.
type DemoRow struct {
	ExternalID  string
	IngestState string
	Category    *string
	Tier        *string
	Action      *string
	Queue       *string
	DraftStatus *string
	Meeting     *string
}

// Rows lists the demo tickets with their latest classification, route, draft
// and meeting.
func (d Demo) Rows(ctx context.Context) ([]DemoRow, error) {
	rows, err := d.Pool.Query(ctx, `
		select t.external_id, t.ingest_status, c.category, c.tier, r.action, r.queue, dr.status, m.status
		from tickets t
		left join lateral (select category, tier from classifications where ticket_id = t.id
		                   order by created_at desc, id desc limit 1) c on true
		left join lateral (select action, queue from routes where ticket_id = t.id
		                   order by created_at desc, id desc limit 1) r on true
		left join lateral (select status from reply_drafts where ticket_id = t.id
		                   order by created_at desc limit 1) dr on true
		left join meetings m on m.ticket_id = t.id
		where t.source = 'csv' and t.external_id like 'demo-%'
		order by t.external_id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []DemoRow
	for rows.Next() {
		var r DemoRow
		if err := rows.Scan(&r.ExternalID, &r.IngestState, &r.Category, &r.Tier, &r.Action, &r.Queue,
			&r.DraftStatus, &r.Meeting); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
