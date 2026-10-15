package preprocess

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
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
		strings.ToLower(strings.TrimSpace(sender)),
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

// Normalize cleans t and sets its ContentHash. It returns a non-empty skip
// reason (and a human-readable detail) when the ticket must not be stored.
func Normalize(t model.Ticket) (out model.Ticket, skip, detail string) {
	t.Sender = strings.ToLower(strings.TrimSpace(t.Sender))
	t.Subject = NormalizeSubject(t.Subject)
	t.Body = CleanBody(t.Body)
	t.ContentHash = ContentHash(t.Sender, t.Subject, t.Body)
	if t.Subject == "" && t.Body == "" {
		return t, SkipEmpty, "no subject and no body after cleaning"
	}
	if lang, non := DetectLanguage(t.Subject + "\n" + t.Body); non {
		return t, SkipNonEnglish, fmt.Sprintf("detected language %s", lang)
	}
	return t, "", ""
}
