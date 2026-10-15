package preprocess

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/mail"
	"regexp"
	"strings"
	"unicode"

	"github.com/abadojack/whatlanggo"
	"golang.org/x/text/unicode/norm"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Skip reasons returned by Normalize.
const (
	SkipEmpty      = "empty"
	SkipNonEnglish = "non_english"
)

// MinLettersForLanguageCheck: shorter texts are assumed English, because
// language detection on a few words is unreliable.
const MinLettersForLanguageCheck = 30

var (
	zeroWidth     = strings.NewReplacer("\u200b", "", "\u200c", "", "\u200d", "", "\ufeff", "", "\u00ad", "")
	inlineSpace   = regexp.MustCompile(`[ \t\f\v\x{00a0}]+`)
	replyPrefixes = regexp.MustCompile(`(?i)^\s*((re|fw|fwd|aw|wg)\s*(\[\d+\])?\s*:\s*)+`)
)

// NormalizeWhitespace converts to NFC, drops zero-width characters,
// normalises line endings, collapses runs of spaces/tabs, trims each line and
// keeps at most one blank line between paragraphs.
func NormalizeWhitespace(s string) string {
	s = zeroWidth.Replace(norm.NFC.String(s))
	s = strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\r", "\n")
	s = inlineSpace.ReplaceAllString(s, " ")
	return tidyLines(s)
}

var angleAddr = regexp.MustCompile(`<([^<>@\s]+@[^<>\s]+)>`)

// NormalizeSender reduces a sender to a bare, lower-cased email address:
// display names, angle brackets, quotes and "mailto:" are stripped. Values
// that contain no address (e.g. "zendesk-user:42") are trimmed and lower-cased.
func NormalizeSender(s string) string {
	s = strings.TrimSpace(s)
	if a, err := mail.ParseAddress(s); err == nil {
		return strings.ToLower(a.Address)
	}
	if m := angleAddr.FindStringSubmatch(s); m != nil {
		return strings.ToLower(m[1])
	}
	s = strings.TrimPrefix(strings.ToLower(s), "mailto:")
	return strings.Trim(s, "<>\"' ")
}

// NormalizeSubject collapses all whitespace in a subject to single spaces.
func NormalizeSubject(s string) string {
	return strings.Join(strings.Fields(norm.NFC.String(zeroWidth.Replace(s))), " ")
}

// CleanBody is the body-cleaning chain every adapter's tickets go through:
// HTML to text (when the body is HTML), quoted-reply and signature stripping,
// whitespace normalisation. It is idempotent.
func CleanBody(body string) string {
	if LooksLikeHTML(body) {
		body = HTMLToText(body)
	}
	return NormalizeWhitespace(StripSignature(StripQuoted(body)))
}

// ContentHash identifies a message's content independent of source and
// transport: sha256 over the lower-cased sender, the subject without reply/
// forward prefixes, and the body with all whitespace collapsed.
func ContentHash(sender, subject, body string) string {
	subj := replyPrefixes.ReplaceAllString(strings.ToLower(subject), "")
	parts := []string{
		NormalizeSender(sender),
		strings.Join(strings.Fields(subj), " "),
		strings.Join(strings.Fields(strings.ToLower(body)), " "),
	}
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x1f")))
	return hex.EncodeToString(sum[:])
}

// DetectLanguage returns the ISO 639-3 code and whether the text should be
// treated as non-English: only a reliable, non-English detection on at least
// MinLettersForLanguageCheck letters counts.
func DetectLanguage(text string) (lang string, nonEnglish bool) {
	letters := 0
	for _, r := range text {
		if unicode.IsLetter(r) {
			letters++
		}
	}
	if letters < MinLettersForLanguageCheck {
		return "eng", false
	}
	info := whatlanggo.Detect(text)
	lang = info.Lang.Iso6393()
	return lang, info.Lang != whatlanggo.Eng && info.IsReliable()
}

// Normalize cleans t, sets its ContentHash and IngestStatus. skip is
// SkipEmpty (the ticket must be dropped) or SkipNonEnglish (stored with
// IngestStatus "skipped" so it stays visible but is never classified).
func Normalize(t model.Ticket) (out model.Ticket, skip, detail string) {
	t.Sender = NormalizeSender(t.Sender)
	t.IngestStatus, t.SkipReason = model.IngestAccepted, ""
	t.Subject = NormalizeSubject(t.Subject)
	t.Body = CleanBody(t.Body)
	t.ContentHash = ContentHash(t.Sender, t.Subject, t.Body)
	if t.Subject == "" && t.Body == "" {
		return t, SkipEmpty, "no subject and no body after cleaning"
	}
	if lang, non := DetectLanguage(t.Subject + "\n" + t.Body); non {
		t.IngestStatus, t.SkipReason = model.IngestSkipped, SkipNonEnglish
		return t, SkipNonEnglish, fmt.Sprintf("detected language %s", lang)
	}
	return t, "", ""
}
