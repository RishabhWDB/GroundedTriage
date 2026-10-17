package routing

import (
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

var t0 = time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC)

func TestShippedSLA(t *testing.T) {
	p, err := LoadSLA("../../config/sla.yaml")
	if err != nil {
		t.Fatal(err)
	}
	want := map[int]time.Duration{1: time.Hour, 2: 4 * time.Hour, 3: 24 * time.Hour, 4: 72 * time.Hour}
	for prio, d := range want {
		if p.Targets[prio] != d {
			t.Fatalf("priority %d target %v, want %v", prio, p.Targets[prio], d)
		}
	}
}

func TestApplyAndEscalate(t *testing.T) {
	p, _ := ParseSLA([]byte("version: 1\ntargets: {1: 1h, 2: 4h, 3: 24h, 4: 72h}\n"))
	r := model.Route{Action: "reply", Priority: 3}
	p.Apply(&r, t0)
	if r.SLADueAt == nil || !r.SLADueAt.Equal(t0.Add(24*time.Hour)) {
		t.Fatalf("due %v", r.SLADueAt)
	}
	m := model.Route{Action: "mute", Priority: 4, SLADueAt: r.SLADueAt}
	p.Apply(&m, t0)
	if m.SLADueAt != nil {
		t.Fatal("muted route got an SLA")
	}
	odd := model.Route{Action: "queue", Priority: 0}
	p.Apply(&odd, t0)
	if !odd.SLADueAt.Equal(t0.Add(time.Hour)) {
		t.Fatalf("priority 0 clamps to 1: %v", odd.SLADueAt)
	}

	e := p.Escalate(3, t0)
	if e.OldPriority != 3 || e.NewPriority != 2 || !e.NewDue.Equal(t0.Add(4*time.Hour)) || e.AtTop {
		t.Fatalf("escalate 3: %+v", e)
	}
	top := p.Escalate(1, t0)
	if top.NewPriority != 1 || !top.AtTop || !top.NewDue.Equal(t0.Add(time.Hour)) {
		t.Fatalf("escalate 1: %+v", top)
	}
}

func TestParseSLARejects(t *testing.T) {
	for name, y := range map[string]string{
		"missing priority": "version: 1\ntargets: {1: 1h, 2: 4h, 3: 24h}\n",
		"extra priority":   "version: 1\ntargets: {1: 1h, 2: 4h, 3: 24h, 4: 72h, 5: 99h}\n",
		"bad duration":     "version: 1\ntargets: {1: soon, 2: 4h, 3: 24h, 4: 72h}\n",
		"negative":         "version: 1\ntargets: {1: -1h, 2: 4h, 3: 24h, 4: 72h}\n",
		"not monotonic":    "version: 1\ntargets: {1: 8h, 2: 4h, 3: 24h, 4: 72h}\n",
		"version":          "version: 2\ntargets: {1: 1h, 2: 4h, 3: 24h, 4: 72h}\n",
		"unknown key":      "version: 1\ntargets: {1: 1h, 2: 4h, 3: 24h, 4: 72h}\nbusiness_hours: true\n",
	} {
		if _, err := ParseSLA([]byte(y)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
