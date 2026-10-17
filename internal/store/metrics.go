package store

import (
	"context"
	"fmt"
	"time"
)

// Summary is the dashboard data behind GET /metrics/summary (P6.4).
type Summary struct {
	Tickets         int            `json:"tickets"`
	BySource        map[string]int `json:"by_source"`
	ByCategory      map[string]int `json:"by_category"` // latest classification; "unclassified" otherwise
	ByUrgency       map[string]int `json:"by_urgency"`
	ByTier          map[string]int `json:"by_tier"`
	ByAction        map[string]int `json:"by_action"`
	OpenRoutes      int            `json:"open_routes"`
	SLABreachedOpen int            `json:"sla_breached_open"` // open and past the deadline now
	EscalatedRoutes int            `json:"escalated_routes"`  // escalated at least once
	DraftsByStatus  map[string]int `json:"drafts_by_status"`
	AbstainRate     *float64       `json:"abstain_rate"`     // abstained / drafts, nil without drafts
	AvgFaithfulness *float64       `json:"avg_faithfulness"` // over drafts that were judged
	TopSenders      []SenderCount  `json:"top_senders"`
	GeneratedAt     time.Time      `json:"generated_at"`
}

// SenderCount is one top-senders row (SCRIBE-style analytics).
type SenderCount struct {
	Sender  string `json:"sender"`
	Tickets int    `json:"tickets"`
}

func (r Review) groupCount(ctx context.Context, sql string) (map[string]int, error) {
	rows, err := r.Pool.Query(ctx, sql)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]int{}
	for rows.Next() {
		var k string
		var n int
		if err := rows.Scan(&k, &n); err != nil {
			return nil, err
		}
		out[k] = n
	}
	return out, rows.Err()
}

// Summary computes the dashboard counts at now.
func (r Review) Summary(ctx context.Context, now time.Time) (Summary, error) {
	s := Summary{GeneratedAt: now}
	latest := `from tickets t
		left join lateral (select category, urgency, tier from classifications c where c.ticket_id = t.id
		                   order by c.created_at desc, c.id desc limit 1) c on true`
	groups := []struct {
		dst *map[string]int
		sql string
	}{
		{&s.BySource, "select source, count(*) from tickets group by 1"},
		{&s.ByCategory, "select coalesce(c.category, 'unclassified'), count(*) " + latest + " group by 1"},
		{&s.ByUrgency, "select c.urgency, count(*) " + latest + " where c.urgency is not null group by 1"},
		{&s.ByTier, "select c.tier, count(*) " + latest + " where c.tier is not null group by 1"},
		{&s.ByAction, `select r.action, count(*) from tickets t join lateral (select action from routes r
			where r.ticket_id = t.id order by r.created_at desc, r.id desc limit 1) r on true group by 1`},
		{&s.DraftsByStatus, `select d.status, count(*) from tickets t join lateral (select status from reply_drafts d
			where d.ticket_id = t.id order by d.created_at desc limit 1) d on true group by 1`},
	}
	for _, g := range groups {
		m, err := r.groupCount(ctx, g.sql)
		if err != nil {
			return s, fmt.Errorf("summary: %w", err)
		}
		*g.dst = m
	}
	for _, n := range s.BySource {
		s.Tickets += n
	}
	err := r.Pool.QueryRow(ctx, `
		select count(*) filter (where status = 'open'),
		       count(*) filter (where status = 'open' and sla_due_at < $1),
		       count(*) filter (where escalations > 0)
		from routes`, now).Scan(&s.OpenRoutes, &s.SLABreachedOpen, &s.EscalatedRoutes)
	if err != nil {
		return s, fmt.Errorf("summary routes: %w", err)
	}
	var drafts, abstained int
	var avg *float64
	err = r.Pool.QueryRow(ctx, `
		select count(*), count(*) filter (where abstain_reason <> ''),
		       avg(faithfulness_score)::float8
		from reply_drafts`).Scan(&drafts, &abstained, &avg)
	if err != nil {
		return s, fmt.Errorf("summary drafts: %w", err)
	}
	if drafts > 0 {
		rate := float64(abstained) / float64(drafts)
		s.AbstainRate = &rate
	}
	s.AvgFaithfulness = avg
	rows, err := r.Pool.Query(ctx, `select sender, count(*) from tickets where sender <> ''
		group by 1 order by 2 desc, 1 limit 10`)
	if err != nil {
		return s, err
	}
	defer rows.Close()
	s.TopSenders = []SenderCount{}
	for rows.Next() {
		var sc SenderCount
		if err := rows.Scan(&sc.Sender, &sc.Tickets); err != nil {
			return s, err
		}
		s.TopSenders = append(s.TopSenders, sc)
	}
	return s, rows.Err()
}
