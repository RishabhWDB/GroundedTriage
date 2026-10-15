package preprocess

import (
	"regexp"
	"strings"
)

// Reply-header patterns that start the quoted part of a message. Everything
// from the first match onwards is dropped.
var quoteHeaders = []*regexp.Regexp{
	// Gmail/Apple: "On <date>, <name> wrote:" (may wrap onto a second line).
	regexp.MustCompile(`(?m)^On\b[^\n]{0,200}(\n[^\n]{0,200})?\bwrote:\s*$`),
	regexp.MustCompile(`(?m)^-{2,}\s*Original Message\s*-{2,}\s*$`),
	// Outlook: "From: ..." followed by "Sent:"/"Date:" within the next lines.
	regexp.MustCompile(`(?m)^_{10,}\s*\n(From|De|Von):`),
	regexp.MustCompile(`(?m)^From: [^\n]+\n(To|Sent|Date|Cc): [^\n]+`),
}

// Signature markers. The RFC 3676 "-- " delimiter and mobile footers cut
// everything after them.
var signatureMarkers = []*regexp.Regexp{
	regexp.MustCompile(`(?m)^-- ?$`),
	regexp.MustCompile(`(?mi)^(Sent from my (iPhone|iPad|Android|mobile device|Galaxy)[^\n]*|Get Outlook for (iOS|Android)[^\n]*)$`),
}

func cutAtFirst(s string, patterns []*regexp.Regexp) string {
	cut := len(s)
	for _, re := range patterns {
		if loc := re.FindStringIndex(s); loc != nil && loc[0] < cut {
			cut = loc[0]
		}
	}
	return s[:cut]
}

// StripQuoted removes the quoted previous message(s) from a reply: reply
// headers ("On ... wrote:", Outlook headers, "Original Message") and every
// line starting with ">". If nothing would remain, s is returned trimmed.
func StripQuoted(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	cut := cutAtFirst(s, quoteHeaders)
	var kept []string
	for _, line := range strings.Split(cut, "\n") {
		if strings.HasPrefix(strings.TrimLeft(line, " \t"), ">") {
			continue
		}
		kept = append(kept, line)
	}
	out := strings.TrimSpace(strings.Join(kept, "\n"))
	if out == "" {
		return strings.TrimSpace(s)
	}
	return out
}

// StripSignature removes a trailing signature introduced by "-- " or a mobile
// footer. Sign-offs like "Thanks, Priya" are kept (harmless and ambiguous).
// If nothing would remain, s is returned trimmed.
func StripSignature(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	out := strings.TrimSpace(cutAtFirst(s, signatureMarkers))
	if out == "" {
		return strings.TrimSpace(s)
	}
	return out
}
