package store

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// StageTiming is one ml-service call timed by the worker (stage_timings).
type StageTiming struct {
	Stage string
	Ms    float64
	OK    bool
	At    time.Time
}

// StageStat is one stage's latency over a day.
type StageStat struct {
	Stage  string
	N      int
	P50    float64
	P95    float64
	Errors int
}

// DayInputs is everything the monitoring job needs for one UTC day.
type DayInputs struct {
	// Categories counts tickets whose latest classification was made that day.
	Categories map[string]int
	// Stages: ml-service calls (classify, retrieve, draft, verify, extract_meeting)
	// plus ingest_to_classified and ingest_to_draft (from row timestamps).
	Stages []StageStat
	// Drafts made that day and how many of them abstained.
	Drafts, Abstained int
	// Routes created that day with an SLA, and how many breached it: resolved
	// after the deadline, or still open past it at now.
	SLARoutes, SLABreached int
}

// DailyMetric is one metrics_daily row.
type DailyMetric struct {
	Day        time.Time      `json:"day"`
	Metric     string         `json:"metric"`
	Value      *float64       `json:"value"`
	Detail     map[string]any `json:"detail"`
	ComputedAt time.Time      `json:"computed_at"`
}

// Monitoring reads and writes the P7.3 tables.
type Monitoring struct {
	Pool *pgxpool.Pool
}

// InsertTimings writes timings in one COPY.
func (m Monitoring) InsertTimings(ctx context.Context, ts []StageTiming) error {
	rows := make([][]any, len(ts))
	for i, t := range ts {
		rows[i] = []any{t.Stage, t.Ms, t.OK, t.At}
	}
	_, err := m.Pool.CopyFrom(ctx, pgx.Identifier{"stage_timings"},
		[]string{"stage", "ms", "ok", "created_at"}, pgx.CopyFromRows(rows))
	return err
}

const stageStatsSQL = `
with calls as (
	select stage, ms, ok from stage_timings where created_at >= $1 and created_at < $2
	union all
	select 'ingest_to_classified', extract(epoch from c.created_at - t.created_at) * 1000, true
	from classifications c join tickets t on t.id = c.ticket_id
	where c.created_at >= $1 and c.created_at < $2
	union all
	select 'ingest_to_draft', extract(epoch from d.created_at - t.created_at) * 1000, true
	from reply_drafts d join tickets t on t.id = d.ticket_id
	where d.created_at >= $1 and d.created_at < $2
)
select stage, count(*),
       percentile_cont(0.5) within group (order by ms)::float8,
       percentile_cont(0.95) within group (order by ms)::float8,
       count(*) filter (where not ok)
from calls group by stage order by stage`

// DayInputs collects one day's inputs; day is the UTC midnight starting it.
func (m Monitoring) DayInputs(ctx context.Context, day, now time.Time) (DayInputs, error) {
	from, to := day, day.AddDate(0, 0, 1)
	in := DayInputs{Categories: map[string]int{}}
	rows, err := m.Pool.Query(ctx, `
		select category, count(*) from (
			select distinct on (ticket_id) category, created_at from classifications
			order by ticket_id, created_at desc, id desc) c
		where created_at >= $1 and created_at < $2 group by 1`, from, to)
	if err != nil {
		return in, fmt.Errorf("categories: %w", err)
	}
	for rows.Next() {
		var k string
		var n int
		if err := rows.Scan(&k, &n); err != nil {
			rows.Close()
			return in, err
		}
		in.Categories[k] = n
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return in, err
	}
	rows, err = m.Pool.Query(ctx, stageStatsSQL, from, to)
	if err != nil {
		return in, fmt.Errorf("stages: %w", err)
	}
	for rows.Next() {
		var s StageStat
		if err := rows.Scan(&s.Stage, &s.N, &s.P50, &s.P95, &s.Errors); err != nil {
			rows.Close()
			return in, err
		}
		in.Stages = append(in.Stages, s)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return in, err
	}
	err = m.Pool.QueryRow(ctx, `
		select count(*), count(*) filter (where abstain_reason <> '')
		from reply_drafts where created_at >= $1 and created_at < $2`, from, to).Scan(&in.Drafts, &in.Abstained)
	if err != nil {
		return in, fmt.Errorf("drafts: %w", err)
	}
	err = m.Pool.QueryRow(ctx, `
		select count(*), count(*) filter (where coalesce(resolved_at, $3) > sla_due_at)
		from routes where sla_due_at is not null and created_at >= $1 and created_at < $2`,
		from, to, now).Scan(&in.SLARoutes, &in.SLABreached)
	if err != nil {
		return in, fmt.Errorf("sla: %w", err)
	}
	return in, nil
}

// UpsertDaily writes metric rows, replacing a day's earlier values.
func (m Monitoring) UpsertDaily(ctx context.Context, ms []DailyMetric) error {
	b := &pgx.Batch{}
	for _, d := range ms {
		detail, err := json.Marshal(d.Detail)
		if err != nil {
			return err
		}
		b.Queue(`insert into metrics_daily (day, metric, value, detail, computed_at)
			values ($1, $2, $3, $4, now())
			on conflict (day, metric) do update
			set value = excluded.value, detail = excluded.detail, computed_at = excluded.computed_at`,
			d.Day, d.Metric, d.Value, detail)
	}
	return m.Pool.SendBatch(ctx, b).Close()
}

// Daily returns the rows of the last `days` days up to now, newest day first.
func (m Monitoring) Daily(ctx context.Context, days int, now time.Time) ([]DailyMetric, error) {
	rows, err := m.Pool.Query(ctx, `
		select day, metric, value, detail, computed_at from metrics_daily
		where day > ($1::timestamptz at time zone 'UTC')::date - $2::int
		order by day desc, metric`, now, days)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []DailyMetric{}
	for rows.Next() {
		var d DailyMetric
		var detail []byte
		if err := rows.Scan(&d.Day, &d.Metric, &d.Value, &detail, &d.ComputedAt); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(detail, &d.Detail); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}
