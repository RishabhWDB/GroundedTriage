package routing

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

const shippedRules = "../../config/routing_rules.yaml"

func cls(category, urgency string, mods ...func(*model.Classification)) *model.Classification {
	c := &model.Classification{TicketID: "t", Category: category, CategoryConfidence: 0.9,
		Tier: "fast", Urgency: urgency, UrgencySource: "heuristic"}
	for _, m := range mods {
		m(c)
	}
	return c
}

func accepted() model.Ticket { return model.Ticket{ID: "t", IngestStatus: model.IngestAccepted} }

// Every rule in the shipped file, plus the default.
func TestShippedRules(t *testing.T) {
	rs, err := LoadRules(shippedRules)
	if err != nil {
		t.Fatal(err)
	}
	lowConf := func(c *model.Classification) { c.LowConfidence = true; c.CategoryConfidence = 0.4 }
	possible := func(c *model.Classification) { c.PossibleSpam = true }
	spamTier := func(c *model.Classification) { c.Tier = "spam_filter" }
	cases := []struct {
		name                string
		in                  Input
		rule, action, queue string
		priority            int
		reasonContains      string
	}{
		{"skipped non-English", Input{Ticket: model.Ticket{ID: "t", IngestStatus: model.IngestSkipped, SkipReason: "non_english"}},
			"skipped-ticket", "queue", "human_review", 3, "non_english"},
		{"spam from the filter", Input{Ticket: accepted(), Classification: cls("Spam", "low", spamTier)},
			"spam", "mute", "spam", 4, ""},
		{"possible spam beats topic", Input{Ticket: accepted(), Classification: cls("Billing", "high", possible)},
			"possible-spam", "queue", "possible_spam", 4, ""},
		{"no-reply sender", Input{Ticket: accepted(), Classification: cls("General Inquiry", "low"), NoReply: true, NoReplyReason: "no-reply sender"},
			"no-reply", "mute", "muted", 4, "no-reply sender"},
		{"critical overrides category", Input{Ticket: accepted(), Classification: cls("Billing", "critical")},
			"critical", "queue", "human_review", 1, ""},
		{"low confidence to a human", Input{Ticket: accepted(), Classification: cls("Technical Issue", "high", lowConf)},
			"low-confidence", "queue", "human_review", 2, ""},
		{"meeting", Input{Ticket: accepted(), Classification: cls("Meeting Scheduling", "medium")},
			"meeting", "schedule", "meetings", 3, ""},
		{"technical", Input{Ticket: accepted(), Classification: cls("Technical Issue", "high")},
			"technical", "reply", "technical", 2, ""},
		{"billing", Input{Ticket: accepted(), Classification: cls("Billing", "low")},
			"billing", "reply", "billing", 4, ""},
		{"feature request", Input{Ticket: accepted(), Classification: cls("Feature Request", "medium")},
			"feature-request", "queue", "product", 3, ""},
		{"general", Input{Ticket: accepted(), Classification: cls("General Inquiry", "medium")},
			"general", "reply", "general", 3, ""},
		{"default: accepted but not classified", Input{Ticket: accepted()},
			"default", "queue", "human_review", 3, ""},
	}
	covered := map[string]bool{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := rs.Evaluate(tc.in)
			if r.Reason != "rule "+tc.rule && !strings.HasPrefix(r.Reason, "rule "+tc.rule+":") {
				t.Fatalf("reason %q, want rule %s", r.Reason, tc.rule)
			}
			if r.Action != tc.action || r.Queue != tc.queue || r.Priority != tc.priority || r.TicketID != "t" {
				t.Fatalf("route %+v, want %s/%s/p%d", r, tc.action, tc.queue, tc.priority)
			}
			if tc.reasonContains != "" && !strings.Contains(r.Reason, tc.reasonContains) {
				t.Fatalf("reason %q lacks %q", r.Reason, tc.reasonContains)
			}
		})
		covered[tc.rule] = true
	}
	for _, r := range rs.Rules {
		if !covered[r.Name] {
			t.Errorf("rule %q has no test case", r.Name)
		}
	}
}

func TestConfidenceBoundsAndTier(t *testing.T) {
	rs, err := ParseRules([]byte(`
version: 1
queues: [a, b, c]
rules:
  - name: confident-deep
    when: {tier: [deep], min_confidence: 0.8}
    then: {action: reply, queue: a}
  - name: unsure
    when: {max_confidence: 0.5}
    then: {action: queue, queue: b, priority: 2}
  - name: default
    then: {action: queue, queue: c}
`))
	if err != nil {
		t.Fatal(err)
	}
	deep := cls("Billing", "medium", func(c *model.Classification) { c.Tier = "deep"; c.CategoryConfidence = 0.8 })
	if r := rs.Evaluate(Input{Ticket: accepted(), Classification: deep}); r.Queue != "a" {
		t.Fatalf("min_confidence is inclusive: %+v", r)
	}
	unsure := cls("Billing", "medium", func(c *model.Classification) { c.CategoryConfidence = 0.5 })
	if r := rs.Evaluate(Input{Ticket: accepted(), Classification: unsure}); r.Queue != "c" {
		t.Fatalf("max_confidence is exclusive: %+v", r)
	}
	unsure.CategoryConfidence = 0.49
	if r := rs.Evaluate(Input{Ticket: accepted(), Classification: unsure}); r.Queue != "b" || r.Priority != 2 {
		t.Fatalf("got %+v", r)
	}
}

func TestParseRulesRejects(t *testing.T) {
	head := "version: 1\nqueues: [q]\nrules:\n"
	def := "  - name: default\n    then: {action: queue, queue: q}\n"
	cases := map[string]string{
		"unknown key":       head + "  - name: a\n    when: {colour: red}\n    then: {action: queue, queue: q}\n" + def,
		"bad action":        head + "  - name: a\n    when: {urgency: [low]}\n    then: {action: delete, queue: q}\n" + def,
		"unknown queue":     head + "  - name: a\n    when: {urgency: [low]}\n    then: {action: queue, queue: nope}\n" + def,
		"unknown category":  head + "  - name: a\n    when: {category: [Complaint]}\n    then: {action: queue, queue: q}\n" + def,
		"unknown urgency":   head + "  - name: a\n    when: {urgency: [urgent]}\n    then: {action: queue, queue: q}\n" + def,
		"bad priority":      head + "  - name: a\n    when: {urgency: [low]}\n    then: {action: queue, queue: q, priority: 9}\n" + def,
		"no default":        head + "  - name: a\n    when: {urgency: [low]}\n    then: {action: queue, queue: q}\n",
		"default not last":  head + def + "  - name: b\n    when: {urgency: [low]}\n    then: {action: queue, queue: q}\n",
		"duplicate name":    head + "  - name: default\n    when: {urgency: [low]}\n    then: {action: queue, queue: q}\n" + def,
		"wrong version":     "version: 2\nqueues: [q]\nrules:\n" + def,
		"no rules":          "version: 1\nqueues: [q]\nrules: []\n",
		"bad ingest status": head + "  - name: a\n    when: {ingest_status: maybe}\n    then: {action: queue, queue: q}\n" + def,
	}
	for name, yml := range cases {
		if _, err := ParseRules([]byte(yml)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestEngineHotReload(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rules.yaml")
	v1 := "version: 1\nqueues: [one, two]\nrules:\n  - name: default\n    then: {action: queue, queue: one}\n"
	if err := os.WriteFile(path, []byte(v1), 0o644); err != nil {
		t.Fatal(err)
	}
	e, err := NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	in := Input{Ticket: accepted()}
	if e.Route(in).Queue != "one" {
		t.Fatal("initial rules not active")
	}
	if changed, err := e.Reload(); changed || err != nil {
		t.Fatalf("unchanged file: changed=%v err=%v", changed, err)
	}
	// An invalid edit keeps the previous rules.
	_ = os.WriteFile(path, []byte("version: 1\nqueues: [one]\nrules: []\n"), 0o644)
	if changed, err := e.Reload(); changed || err == nil {
		t.Fatalf("invalid file: changed=%v err=%v", changed, err)
	}
	if e.Route(in).Queue != "one" {
		t.Fatal("invalid edit replaced the rules")
	}
	// Watch picks up a valid edit.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go e.Watch(ctx, 5*time.Millisecond)
	_ = os.WriteFile(path, []byte(strings.ReplaceAll(v1, "queue: one}", "queue: two}")), 0o644)
	deadline := time.Now().Add(2 * time.Second)
	for e.Route(in).Queue != "two" {
		if time.Now().After(deadline) {
			t.Fatal("Watch did not reload the rules")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if _, err := NewEngine(filepath.Join(dir, "missing.yaml")); err == nil {
		t.Fatal("missing file accepted")
	}
}
