package routing

import (
	"bytes"
	"fmt"
	"os"
	"slices"

	"gopkg.in/yaml.v3"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Actions a rule can take (routes.action check constraint).
var Actions = []string{"mute", "queue", "reply", "schedule"}

// Tiers a classification can come from.
var Tiers = []string{"fast", "deep", "spam_filter"}

// Input is everything the rules may look at. Classification is nil for
// tickets that were never classified (ingest_status=skipped).
type Input struct {
	Ticket         model.Ticket
	Classification *model.Classification
	// NoReply is set by DetectNoReply (P4.2); NoReplyReason explains it.
	NoReply       bool
	NoReplyReason string
}

// When holds a rule's conditions; nil/empty fields match anything.
type When struct {
	IngestStatus  string   `yaml:"ingest_status"`
	Category      []string `yaml:"category"`
	Urgency       []string `yaml:"urgency"`
	Tier          []string `yaml:"tier"`
	LowConfidence *bool    `yaml:"low_confidence"`
	PossibleSpam  *bool    `yaml:"possible_spam"`
	NoReply       *bool    `yaml:"no_reply"`
	MinConfidence *float64 `yaml:"min_confidence"`
	MaxConfidence *float64 `yaml:"max_confidence"`
}

// Then is a rule's outcome. Priority 0 means "from urgency".
type Then struct {
	Action   string `yaml:"action"`
	Queue    string `yaml:"queue"`
	Priority int    `yaml:"priority"`
}

// Rule is one entry of config/routing_rules.yaml.
type Rule struct {
	Name string `yaml:"name"`
	When *When  `yaml:"when"`
	Then Then   `yaml:"then"`
}

// RuleSet is a validated rules file.
type RuleSet struct {
	Version int      `yaml:"version"`
	Queues  []string `yaml:"queues"`
	Rules   []Rule   `yaml:"rules"`
}

// UrgencyPriority maps urgency to the default priority (1 = most urgent).
var UrgencyPriority = map[string]int{"critical": 1, "high": 2, "medium": 3, "low": 4}

// DefaultPriority is used when a rule gives no priority and there is no urgency.
const DefaultPriority = 3

// ParseRules decodes and validates a rules file (unknown keys are errors).
func ParseRules(data []byte) (*RuleSet, error) {
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	var rs RuleSet
	if err := dec.Decode(&rs); err != nil {
		return nil, fmt.Errorf("routing rules: %w", err)
	}
	if err := rs.validate(); err != nil {
		return nil, fmt.Errorf("routing rules: %w", err)
	}
	return &rs, nil
}

// LoadRules reads and validates a rules file.
func LoadRules(path string) (*RuleSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseRules(data)
}

func (rs *RuleSet) validate() error {
	if rs.Version != 1 {
		return fmt.Errorf("version must be 1, got %d", rs.Version)
	}
	if len(rs.Rules) == 0 {
		return fmt.Errorf("no rules")
	}
	seen := map[string]bool{}
	for i, r := range rs.Rules {
		where := fmt.Sprintf("rule %d (%q)", i+1, r.Name)
		if r.Name == "" {
			return fmt.Errorf("rule %d has no name", i+1)
		}
		if seen[r.Name] {
			return fmt.Errorf("%s: duplicate name", where)
		}
		seen[r.Name] = true
		if !slices.Contains(Actions, r.Then.Action) {
			return fmt.Errorf("%s: action %q not in %v", where, r.Then.Action, Actions)
		}
		if !slices.Contains(rs.Queues, r.Then.Queue) {
			return fmt.Errorf("%s: queue %q not in queues %v", where, r.Then.Queue, rs.Queues)
		}
		if r.Then.Priority < 0 || r.Then.Priority > 4 {
			return fmt.Errorf("%s: priority must be 1..4 (or omitted)", where)
		}
		last := i == len(rs.Rules)-1
		if r.When == nil && !last {
			return fmt.Errorf("%s: only the last rule may omit `when` (it is the default)", where)
		}
		if last && r.When != nil {
			return fmt.Errorf("%s: the last rule must be the default (no `when`)", where)
		}
		if r.When == nil {
			continue
		}
		w := r.When
		if w.IngestStatus != "" && w.IngestStatus != model.IngestAccepted && w.IngestStatus != model.IngestSkipped {
			return fmt.Errorf("%s: ingest_status %q", where, w.IngestStatus)
		}
		for _, c := range w.Category {
			if !slices.Contains(model.Categories, c) {
				return fmt.Errorf("%s: unknown category %q", where, c)
			}
		}
		for _, u := range w.Urgency {
			if !slices.Contains(model.UrgencyScale, u) {
				return fmt.Errorf("%s: unknown urgency %q", where, u)
			}
		}
		for _, t := range w.Tier {
			if !slices.Contains(Tiers, t) {
				return fmt.Errorf("%s: unknown tier %q", where, t)
			}
		}
	}
	return nil
}

func (w *When) matches(in Input) bool {
	if w == nil {
		return true
	}
	status := in.Ticket.IngestStatus
	if status == "" {
		status = model.IngestAccepted
	}
	if w.IngestStatus != "" && w.IngestStatus != status {
		return false
	}
	if w.NoReply != nil && *w.NoReply != in.NoReply {
		return false
	}
	c := in.Classification
	needsClass := len(w.Category) > 0 || len(w.Urgency) > 0 || len(w.Tier) > 0 ||
		w.LowConfidence != nil || w.PossibleSpam != nil || w.MinConfidence != nil || w.MaxConfidence != nil
	if c == nil {
		return !needsClass
	}
	if len(w.Category) > 0 && !slices.Contains(w.Category, c.Category) {
		return false
	}
	if len(w.Urgency) > 0 && !slices.Contains(w.Urgency, c.Urgency) {
		return false
	}
	if len(w.Tier) > 0 && !slices.Contains(w.Tier, c.Tier) {
		return false
	}
	if w.LowConfidence != nil && *w.LowConfidence != c.LowConfidence {
		return false
	}
	if w.PossibleSpam != nil && *w.PossibleSpam != c.PossibleSpam {
		return false
	}
	if w.MinConfidence != nil && c.CategoryConfidence < *w.MinConfidence {
		return false
	}
	if w.MaxConfidence != nil && c.CategoryConfidence >= *w.MaxConfidence {
		return false
	}
	return true
}

// Evaluate returns the route of the first matching rule. The RuleSet always
// ends with a default rule, so there is always a match. SLADueAt is set by
// the SLA policy (P4.3), not here.
func (rs *RuleSet) Evaluate(in Input) model.Route {
	for _, r := range rs.Rules {
		if !r.When.matches(in) {
			continue
		}
		prio := r.Then.Priority
		if prio == 0 {
			prio = DefaultPriority
			if in.Classification != nil {
				if p, ok := UrgencyPriority[in.Classification.Urgency]; ok {
					prio = p
				}
			}
		}
		reason := "rule " + r.Name
		if r.When != nil && r.When.NoReply != nil && *r.When.NoReply && in.NoReplyReason != "" {
			reason += ": " + in.NoReplyReason
		}
		if in.Ticket.IngestStatus == model.IngestSkipped && in.Ticket.SkipReason != "" {
			reason += ": skipped (" + in.Ticket.SkipReason + ")"
		}
		return model.Route{TicketID: in.Ticket.ID, Action: r.Then.Action, Queue: r.Then.Queue,
			Priority: prio, Reason: reason}
	}
	panic("routing: rule set without a default rule") // validate() makes this unreachable
}
