package monitoring

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

var ref = map[string]float64{"A": 0.5, "B": 0.3, "C": 0.2}

func TestPSIIdenticalIsZero(t *testing.T) {
	if got := PSI(ref, map[string]int{"A": 50, "B": 30, "C": 20}); math.Abs(got) > 1e-12 {
		t.Fatalf("PSI = %v, want 0", got)
	}
}

func TestPSIKnownValue(t *testing.T) {
	// p = (0.3, 0.3, 0.4) vs q = (0.5, 0.3, 0.2)
	want := (0.3-0.5)*math.Log(0.3/0.5) + (0.4-0.2)*math.Log(0.4/0.2)
	if got := PSI(ref, map[string]int{"A": 30, "B": 30, "C": 40}); math.Abs(got-want) > 1e-12 {
		t.Fatalf("PSI = %v, want %v", got, want)
	}
	if want < 0.2 || want > 0.3 {
		t.Fatalf("sanity: this shift should be ~0.24, got %v", want)
	}
}

func TestPSIFloorsEmptyAndUnknownCategories(t *testing.T) {
	got := PSI(ref, map[string]int{"A": 10, "New": 10})
	if math.IsInf(got, 0) || math.IsNaN(got) || got <= 1 {
		t.Fatalf("PSI = %v, want a large finite value", got)
	}
	if !math.IsNaN(PSI(ref, map[string]int{})) {
		t.Fatal("empty counts should give NaN")
	}
}

func TestPSIIsSymmetricInSwap(t *testing.T) {
	a := PSI(map[string]float64{"A": 0.7, "B": 0.3}, map[string]int{"A": 40, "B": 60})
	b := PSI(map[string]float64{"A": 0.4, "B": 0.6}, map[string]int{"A": 70, "B": 30})
	if math.Abs(a-b) > 1e-12 {
		t.Fatalf("PSI(p,q)=%v != PSI(q,p)=%v", a, b)
	}
}

func cfg() Config { return Config{PSIAlert: 0.2, MinTickets: 30, Reference: ref} }

func metric(ms []store.DailyMetric, name string) store.DailyMetric {
	for _, m := range ms {
		if m.Metric == name {
			return m
		}
	}
	return store.DailyMetric{}
}

func TestComputeFlagsDriftAndRates(t *testing.T) {
	day := time.Date(2026, 10, 17, 0, 0, 0, 0, time.UTC)
	ms := Compute(cfg(), day, store.DayInputs{
		Categories: map[string]int{"A": 10, "B": 10, "C": 20},
		Stages:     []store.StageStat{{Stage: "verify", N: 3, P50: 900, P95: 1500}, {Stage: "classify", N: 40, P95: 80}},
		Drafts:     4, Abstained: 1, SLARoutes: 10, SLABreached: 3,
	})
	psi := metric(ms, MetricPSI)
	if psi.Value == nil || psi.Detail["alert"] != true {
		t.Fatalf("psi = %+v, want an alert", psi)
	}
	if v := metric(ms, MetricAbstain).Value; v == nil || *v != 0.25 {
		t.Fatalf("abstain = %v", v)
	}
	if v := metric(ms, MetricSLABreach).Value; v == nil || *v != 0.3 {
		t.Fatalf("sla = %v", v)
	}
	if v := metric(ms, MetricLatencyPre+"verify").Value; v == nil || *v != 1500 {
		t.Fatalf("verify p95 = %v", v)
	}
	if ms[3].Metric != MetricLatencyPre+"classify" {
		t.Fatalf("stages not sorted: %s", ms[3].Metric)
	}
}

func TestComputeSkipsPSIOnFewTicketsAndEmptyRates(t *testing.T) {
	ms := Compute(cfg(), time.Time{}, store.DayInputs{Categories: map[string]int{"A": 29}})
	psi := metric(ms, MetricPSI)
	if psi.Value != nil || psi.Detail["insufficient"] != true || psi.Detail["alert"] != false {
		t.Fatalf("psi = %+v", psi)
	}
	if metric(ms, MetricAbstain).Value != nil || metric(ms, MetricSLABreach).Value != nil {
		t.Fatal("rates without data must be nil")
	}
}

type fakeStore struct {
	timings []store.StageTiming
	days    []time.Time
	upserts []store.DailyMetric
	fail    bool
}

func (f *fakeStore) InsertTimings(_ context.Context, ts []store.StageTiming) error {
	if f.fail {
		return errors.New("db down")
	}
	f.timings = append(f.timings, ts...)
	return nil
}

func (f *fakeStore) DayInputs(_ context.Context, day, _ time.Time) (store.DayInputs, error) {
	f.days = append(f.days, day)
	return store.DayInputs{}, nil
}

func (f *fakeStore) UpsertDaily(_ context.Context, ms []store.DailyMetric) error {
	f.upserts = append(f.upserts, ms...)
	return nil
}

func TestRecorderFlushKeepsTimingsOnError(t *testing.T) {
	r := &Recorder{}
	r.Observe("classify", 12*time.Millisecond, nil)
	r.Observe("draft", 2*time.Second, errors.New("x"))
	f := &fakeStore{fail: true}
	if _, err := r.Flush(context.Background(), f); err == nil {
		t.Fatal("want error")
	}
	f.fail = false
	n, err := r.Flush(context.Background(), f)
	if err != nil || n != 2 || f.timings[0].Ms != 12 || f.timings[1].OK {
		t.Fatalf("n=%d err=%v timings=%+v", n, err, f.timings)
	}
	if n, _ := r.Flush(context.Background(), f); n != 0 {
		t.Fatal("buffer should be empty")
	}
}

func TestJobRecomputesYesterdayAndToday(t *testing.T) {
	now := time.Date(2026, 10, 17, 0, 30, 0, 0, time.FixedZone("IST", 5*3600+1800))
	f := &fakeStore{}
	j := &Job{Store: f, Config: cfg(), Recorder: &Recorder{}, Now: func() time.Time { return now }}
	if _, err := j.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	// 00:30 IST is 19:00 UTC on the 16th
	if len(f.days) != 2 || f.days[0].Format("2006-01-02") != "2026-10-15" || f.days[1].Format("2006-01-02") != "2026-10-16" {
		t.Fatalf("days = %v", f.days)
	}
	if len(f.upserts) != 6 {
		t.Fatalf("upserts = %d, want 3 metrics x 2 days", len(f.upserts))
	}
}

func TestLoadConfigRepoFile(t *testing.T) {
	c, err := LoadConfig(filepath.Join("..", "..", "config", "monitoring.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	sum := 0.0
	for _, v := range c.Reference {
		sum += v
	}
	if math.Abs(sum-1) > 1e-3 || c.PSIAlert != 0.2 {
		t.Fatalf("reference sums to %v, psi_alert %v", sum, c.PSIAlert)
	}
	if len(c.Reference) != len(model.Categories) {
		t.Fatalf("reference has %d categories, taxonomy %d", len(c.Reference), len(model.Categories))
	}
	for _, cat := range model.Categories {
		if _, ok := c.Reference[cat]; !ok {
			t.Fatalf("reference misses %q (re-run make monitoring-ref)", cat)
		}
	}
	bad := filepath.Join(t.TempDir(), "m.yaml")
	_ = os.WriteFile(bad, []byte("version: 1\npsi_alert: 0.2\nreference: {A: 1}\ntypo: 1\n"), 0o600)
	if _, err := LoadConfig(bad); err == nil {
		t.Fatal("unknown key should fail")
	}
}
