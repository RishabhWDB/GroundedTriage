package routing

import (
	"strings"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

func TestDetectNoReply(t *testing.T) {
	cases := []struct {
		name, sender, subject, body string
		want                        bool
		reason                      string
	}{
		{"noreply sender", "noreply@newsletter.saasweekly.com", "This week in SaaS", "Read our top stories.", true, "no-reply sender"},
		{"do-not-reply variant", "do-not-reply+abc@bank.com", "Statement ready", "Your statement is ready.", true, "no-reply sender"},
		{"mailer daemon", "mailer-daemon@mx.example.com", "Undeliverable", "Message could not be delivered", true, "no-reply sender"},
		{"automated subject", "ops@vendor.com", "[Automated] Scheduled maintenance", "Maintenance on Oct 20.", true, "automated notification (subject)"},
		{"automated body", "support-bot@status.cloudhost.net", "Maintenance window", "This is an automated message. Do not reply to this email.", true, "automated notification"},
		{"out of office", "j.doe@acme.com", "Automatic reply: Invoice", "I am out of office until Monday.", true, "automated notification"},
		{"newsletter footer", "editor@weekly.io", "10 growth tactics", "Read more... Unsubscribe at any time.", true, "newsletter"},
		{"fyi without a question", "lead@acme.com", "FYI: new org chart", "Sharing the new org chart for reference.", true, "FYI"},
		{"fyi with a question is answered", "lead@acme.com", "FYI: new org chart", "Can you update the CRM roles?", false, ""},
		{"bot sender alone is not enough", "alerts@monitoring.acme-corp.com", "CRITICAL: production database down", "db-prod-1 is unreachable. Immediate action required.", false, ""},
		{"customer", "priya.n@acme-corp.com", "Charged twice", "We were charged twice, please refund.", false, ""},
		{"word inside a name is not a cue", "newsome.k@example.com", "Billing", "Why was I charged twice?", false, ""},
		{"news@ is a newsletter sender", "news@shop.com", "Spring sale", "Our spring sale starts now", true, "no-reply sender"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, reason := DetectNoReply(model.Ticket{Sender: tc.sender, Subject: tc.subject, Body: tc.body})
			if got != tc.want || !strings.Contains(reason, tc.reason) {
				t.Fatalf("got (%v, %q), want (%v, ~%q)", got, reason, tc.want, tc.reason)
			}
		})
	}
}

// A critical automated alert is queued, not muted (critical comes first).
func TestCriticalAlertBeatsNoReply(t *testing.T) {
	rs, err := LoadRules(shippedRules)
	if err != nil {
		t.Fatal(err)
	}
	in := Input{Ticket: accepted(), Classification: cls("Technical Issue", "critical"), NoReply: true, NoReplyReason: "x"}
	if r := rs.Evaluate(in); r.Action != "queue" || r.Priority != 1 {
		t.Fatalf("critical no-reply alert routed %+v", r)
	}
}
