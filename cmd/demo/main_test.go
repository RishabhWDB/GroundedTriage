package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func TestConfirmNeedsExactPhrase(t *testing.T) {
	for in, want := range map[string]bool{"reset demo\n": true, "  reset demo  \n": true, "yes\n": false, "": false, "reset\n": false} {
		if got := confirm(strings.NewReader(in), &bytes.Buffer{}); got != want {
			t.Errorf("confirm(%q) = %v", in, got)
		}
	}
}

func p(s string) *string { return &s }

func TestActualOutcome(t *testing.T) {
	cases := []struct {
		row  store.DemoRow
		want string
	}{
		{store.DemoRow{}, "(not routed yet)"},
		{store.DemoRow{Action: p("mute")}, "mute"},
		{store.DemoRow{Action: p("schedule")}, "schedule"},
		{store.DemoRow{Action: p("reply"), DraftStatus: p("abstained")}, "abstain"},
		{store.DemoRow{Action: p("reply"), DraftStatus: p("grounded")}, "ground"},
		{store.DemoRow{Action: p("queue"), Queue: p("human_review"), DraftStatus: p("abstained")}, "abstain"},
		{store.DemoRow{Action: p("queue"), Queue: p("product")}, "queue:product"},
		{store.DemoRow{Action: p("reply")}, "reply (no draft yet)"},
	}
	for _, c := range cases {
		if got := actualOutcome(c.row); got != c.want {
			t.Errorf("%+v: got %q want %q", c.row, got, c.want)
		}
	}
}

func TestDemoCSVIsTaggedAndHasExpectations(t *testing.T) {
	exp, err := readExpected("../../datasets/demo/demo_tickets.csv")
	if err != nil {
		t.Fatal(err)
	}
	if len(exp) != 25 {
		t.Fatalf("%d rows, want 25", len(exp))
	}
	allowed := map[string]bool{"ground": true, "abstain": true, "mute": true, "schedule": true,
		"queue:product": true, "draft (gate decides)": true}
	for id, e := range exp {
		if !strings.HasPrefix(id, store.DemoIDPrefix) || !allowed[e.Outcome] || e.Label == "" {
			t.Errorf("row %s: %+v", id, e)
		}
	}
}

func TestPrintStatusCountsMatches(t *testing.T) {
	var buf bytes.Buffer
	printStatus(&buf, []store.DemoRow{
		{ExternalID: "demo-01", Action: p("reply"), DraftStatus: p("grounded")},
		{ExternalID: "demo-24", Action: p("queue"), Queue: p("human_review"), DraftStatus: p("abstained")},
		{ExternalID: "demo-22", Action: p("reply"), DraftStatus: p("grounded")},
	}, map[string]expectation{"demo-01": {"Billing", "ground"}, "demo-24": {"Billing", "draft (gate decides)"},
		"demo-22": {"Spam", "mute"}})
	if !strings.Contains(buf.String(), "3 demo tickets, 2 match") {
		t.Fatal(buf.String())
	}
}
