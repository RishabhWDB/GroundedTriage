// Package monitoring computes the P7.3 daily metrics: category drift (PSI of
// the day's category distribution vs the reference in config/monitoring.yaml),
// p50/p95 latency per pipeline stage, abstain rate and SLA breach rate. The
// worker records ml-service call timings with a Recorder and runs the Job
// after every poll; results land in metrics_daily (one row per day and
// metric) and are served by GET /metrics/daily.
package monitoring

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// Config is config/monitoring.yaml.
type Config struct {
	Version         int                `yaml:"version"`
	PSIAlert        float64            `yaml:"psi_alert"`
	MinTickets      int                `yaml:"min_tickets"`
	ReferenceSource string             `yaml:"reference_source"`
	Reference       map[string]float64 `yaml:"reference"`
}

// LoadConfig reads and validates the monitoring config.
func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	var c Config
	if err := dec.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("monitoring config: %w", err)
	}
	if len(c.Reference) == 0 || c.PSIAlert <= 0 {
		return Config{}, fmt.Errorf("monitoring config: reference and psi_alert are required")
	}
	if c.MinTickets <= 0 {
		c.MinTickets = 30
	}
	return c, nil
}

// PSIFloor is the minimum share used for a category, so an empty category
// on either side doesn't make the PSI infinite.
const PSIFloor = 1e-4

// PSI is the population stability index of counts vs the reference shares:
// sum over categories of (p - q) * ln(p / q). Categories missing from the
// reference count with share 0 (floored). It returns NaN for empty counts.
func PSI(reference map[string]float64, counts map[string]int) float64 {
	total := 0
	for _, n := range counts {
		total += n
	}
	if total == 0 {
		return math.NaN()
	}
	keys := map[string]bool{}
	for k := range reference {
		keys[k] = true
	}
	for k := range counts {
		keys[k] = true
	}
	psi := 0.0
	for k := range keys {
		p := math.Max(float64(counts[k])/float64(total), PSIFloor)
		q := math.Max(reference[k], PSIFloor)
		psi += (p - q) * math.Log(p/q)
	}
	return psi
}

// Recorder buffers ml-service call timings (mlclient.Client.Observe) until
// the next Flush. Safe for concurrent use.
type Recorder struct {
	mu  sync.Mutex
	buf []store.StageTiming
	Now func() time.Time
}

// Observe matches mlclient.Client.Observe.
func (r *Recorder) Observe(stage string, d time.Duration, err error) {
	now := time.Now
	if r.Now != nil {
		now = r.Now
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.buf = append(r.buf, store.StageTiming{Stage: stage, Ms: float64(d.Microseconds()) / 1000, OK: err == nil, At: now()})
}

// TimingWriter persists timings (store.Monitoring).
type TimingWriter interface {
	InsertTimings(ctx context.Context, ts []store.StageTiming) error
}

// Flush writes the buffered timings in one batch. On error they stay buffered.
func (r *Recorder) Flush(ctx context.Context, w TimingWriter) (int, error) {
	r.mu.Lock()
	batch := r.buf
	r.buf = nil
	r.mu.Unlock()
	if len(batch) == 0 {
		return 0, nil
	}
	if err := w.InsertTimings(ctx, batch); err != nil {
		r.mu.Lock()
		r.buf = append(batch, r.buf...)
		r.mu.Unlock()
		return 0, err
	}
	return len(batch), nil
}

// Store is what the Job needs (store.Monitoring).
type Store interface {
	TimingWriter
	DayInputs(ctx context.Context, day, now time.Time) (store.DayInputs, error)
	UpsertDaily(ctx context.Context, ms []store.DailyMetric) error
}

// Job computes and stores the daily metrics.
type Job struct {
	Store    Store
	Config   Config
	Recorder *Recorder // optional; flushed first so today's timings count
	Now      func() time.Time
}

// Metric names in metrics_daily.
const (
	MetricPSI        = "category_psi"
	MetricAbstain    = "abstain_rate"
	MetricSLABreach  = "sla_breach_rate"
	MetricLatencyPre = "latency_p95_ms."
)

// Day returns the UTC calendar day containing t.
func Day(t time.Time) time.Time {
	y, m, d := t.UTC().Date()
	return time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
}

// Run flushes timings, then recomputes yesterday (to close it out) and today.
func (j *Job) Run(ctx context.Context) ([]store.DailyMetric, error) {
	now := time.Now()
	if j.Now != nil {
		now = j.Now()
	}
	if j.Recorder != nil {
		if _, err := j.Recorder.Flush(ctx, j.Store); err != nil {
			return nil, fmt.Errorf("monitoring: flush timings: %w", err)
		}
	}
	var all []store.DailyMetric
	for _, day := range []time.Time{Day(now).AddDate(0, 0, -1), Day(now)} {
		in, err := j.Store.DayInputs(ctx, day, now)
		if err != nil {
			return nil, fmt.Errorf("monitoring: %s: %w", day.Format("2006-01-02"), err)
		}
		all = append(all, Compute(j.Config, day, in)...)
	}
	if err := j.Store.UpsertDaily(ctx, all); err != nil {
		return nil, fmt.Errorf("monitoring: store: %w", err)
	}
	return all, nil
}

func ratio(num, den int) *float64 {
	if den == 0 {
		return nil
	}
	v := float64(num) / float64(den)
	return &v
}

// Compute turns one day's inputs into metric rows (pure; unit-tested).
func Compute(cfg Config, day time.Time, in store.DayInputs) []store.DailyMetric {
	classified := 0
	for _, n := range in.Categories {
		classified += n
	}
	psiDetail := map[string]any{"classified": classified, "counts": in.Categories,
		"min_tickets": cfg.MinTickets, "psi_alert": cfg.PSIAlert}
	var psi *float64
	if classified >= cfg.MinTickets {
		v := PSI(cfg.Reference, in.Categories)
		psi = &v
		psiDetail["alert"] = v > cfg.PSIAlert
	} else {
		psiDetail["alert"] = false
		psiDetail["insufficient"] = true
	}
	out := []store.DailyMetric{
		{Day: day, Metric: MetricPSI, Value: psi, Detail: psiDetail},
		{Day: day, Metric: MetricAbstain, Value: ratio(in.Abstained, in.Drafts),
			Detail: map[string]any{"drafts": in.Drafts, "abstained": in.Abstained}},
		{Day: day, Metric: MetricSLABreach, Value: ratio(in.SLABreached, in.SLARoutes),
			Detail: map[string]any{"routes_with_sla": in.SLARoutes, "breached": in.SLABreached}},
	}
	stages := append([]store.StageStat(nil), in.Stages...)
	sort.Slice(stages, func(a, b int) bool { return stages[a].Stage < stages[b].Stage })
	for _, s := range stages {
		p95 := s.P95
		out = append(out, store.DailyMetric{Day: day, Metric: MetricLatencyPre + s.Stage, Value: &p95,
			Detail: map[string]any{"n": s.N, "p50_ms": s.P50, "errors": s.Errors}})
	}
	return out
}
