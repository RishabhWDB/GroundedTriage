package routing

import (
	"regexp"
	"strings"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// No-reply detection (P4.2): mail that needs no answer is muted with a
// recorded reason. The checks are deliberately conservative (an unanswered
// customer costs more than a reviewed newsletter) and ordered: the first hit
// gives the reason. Critical tickets are routed before the no-reply rule in
// config/routing_rules.yaml, so an automated "production down" alert is
// still queued.

var (
	// Local parts of addresses that never read replies.
	noReplyLocal = regexp.MustCompile(`^(no-?reply|do-?not-?reply|donotreply|mailer-daemon|postmaster|bounces?|notifications?|newsletters?|news|marketing|mailer)([+._-].*)?$`)
	// Bot-style local parts; only a signal together with an automated body or subject.
	botLocal = regexp.MustCompile(`(^|[._-])(bot|alerts?|automated|system|daemon|status)([._-]|$)`)

	newsletterCues = []string{
		"unsubscribe", "view this email in your browser", "view in browser",
		"you are receiving this email because", "manage your email preferences",
		"update your preferences", "to stop receiving these emails",
	}
	automatedCues = []string{
		"this is an automated message", "this is an automatically generated",
		"do not reply to this email", "please do not reply", "this mailbox is not monitored",
		"auto-generated", "automatic reply", "out of office", "delivery status notification",
	}
	fyiSubject  = regexp.MustCompile(`(?i)^\s*(fyi|for your information|\[fyi\])\b`)
	autoSubject = regexp.MustCompile(`(?i)^\s*(\[automated\]|automatic reply|auto-?reply|out of office|undeliverable|delivery status notification)`)
)

// DetectNoReply reports whether a ticket needs no reply and why. sender is
// the normalised (bare, lower-cased) address from ingestion.
func DetectNoReply(t model.Ticket) (bool, string) {
	local := strings.ToLower(t.Sender)
	if i := strings.LastIndex(local, "@"); i >= 0 {
		local = local[:i]
	}
	subject := strings.ToLower(t.Subject)
	body := strings.ToLower(t.Body)
	text := subject + "\n" + body

	if noReplyLocal.MatchString(local) {
		return true, "no-reply sender (" + t.Sender + ")"
	}
	if autoSubject.MatchString(t.Subject) {
		return true, "automated notification (subject)"
	}
	for _, cue := range automatedCues {
		if strings.Contains(text, cue) {
			return true, "automated notification (" + cue + ")"
		}
	}
	for _, cue := range newsletterCues {
		if strings.Contains(body, cue) {
			return true, "newsletter (" + cue + ")"
		}
	}
	if fyiSubject.MatchString(t.Subject) && !strings.Contains(body, "?") {
		return true, "FYI with no question"
	}
	if botLocal.MatchString(local) && strings.Contains(text, "automated") {
		return true, "automated sender (" + t.Sender + ")"
	}
	return false, ""
}
