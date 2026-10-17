package routing

import (
	"bytes"
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Clock is the time source (a fake in tests).
type Clock interface{ Now() time.Time }

// SystemClock is the real clock.
type SystemClock struct{}

// Now returns the current time.
func (SystemClock) Now() time.Time { return time.Now() }

// SLAPolicy holds the response target per priority (config/sla.yaml).
type SLAPolicy struct {
	Targets map[int]time.Duration
}

type slaFile struct {
	Version int            `yaml:"version"`
	Targets map[int]string `yaml:"targets"`
}

// ParseSLA decodes and validates an SLA file: version 1 and a positive
// target for each priority 1..4, non-decreasing from 1 to 4.
func ParseSLA(data []byte) (SLAPolicy, error) {
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	var f slaFile
	if err := dec.Decode(&f); err != nil {
		return SLAPolicy{}, fmt.Errorf("sla: %w", err)
	}
	if f.Version != 1 {
		return SLAPolicy{}, fmt.Errorf("sla: version must be 1, got %d", f.Version)
	}
	p := SLAPolicy{Targets: map[int]time.Duration{}}
	for prio := 1; prio <= 4; prio++ {
		s, ok := f.Targets[prio]
		if !ok {
			return SLAPolicy{}, fmt.Errorf("sla: no target for priority %d", prio)
		}
		d, err := time.ParseDuration(s)
		if err != nil || d <= 0 {
			return SLAPolicy{}, fmt.Errorf("sla: priority %d target %q must be a positive duration", prio, s)
		}
		if prio > 1 && d < p.Targets[prio-1] {
			return SLAPolicy{}, fmt.Errorf("sla: priority %d target shorter than priority %d", prio, prio-1)
		}
		p.Targets[prio] = d
	}
	if len(f.Targets) != 4 {
		return SLAPolicy{}, fmt.Errorf("sla: targets must be exactly priorities 1..4")
	}
	return p, nil
}

// LoadSLA reads config/sla.yaml.
func LoadSLA(path string) (SLAPolicy, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SLAPolicy{}, err
	}
	return ParseSLA(data)
}

// Apply sets r.SLADueAt = now + target(priority); muted routes get none.
func (p SLAPolicy) Apply(r *model.Route, now time.Time) {
	if r.Action == "mute" {
		r.SLADueAt = nil
		return
	}
	due := now.Add(p.Targets[clampPriority(r.Priority)])
	r.SLADueAt = &due
}

// Escalation is the outcome of escalating one breached route.
type Escalation struct {
	OldPriority, NewPriority int
	NewDue                   time.Time
	// AtTop is true when the route was already priority 1: the deadline is
	// re-armed and the breach is recorded, but the priority can't rise.
	AtTop bool
}

// Escalate raises a breached route one priority level (4 -> 1) and re-arms
// its deadline for the new priority.
func (p SLAPolicy) Escalate(priority int, now time.Time) Escalation {
	old := clampPriority(priority)
	next := old - 1
	if next < 1 {
		next = 1
	}
	return Escalation{OldPriority: old, NewPriority: next, NewDue: now.Add(p.Targets[next]), AtTop: old == 1}
}

func clampPriority(p int) int {
	switch {
	case p < 1:
		return 1
	case p > 4:
		return 4
	}
	return p
}
