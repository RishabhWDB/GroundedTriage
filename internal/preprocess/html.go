package preprocess

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

var htmlish = regexp.MustCompile(`(?i)<(html|body|div|p|br|span|table|b|i|a)(\s[^<>]*)?/?>`)

// LooksLikeHTML reports whether s appears to be an HTML document or fragment.
func LooksLikeHTML(s string) bool { return htmlish.MatchString(s) }

var blockAtoms = map[atom.Atom]bool{
	atom.P: true, atom.Div: true, atom.Br: true, atom.Li: true, atom.Tr: true,
	atom.H1: true, atom.H2: true, atom.H3: true, atom.H4: true, atom.H5: true, atom.H6: true,
	atom.Ul: true, atom.Ol: true, atom.Table: true, atom.Hr: true, atom.Pre: true,
}

var skipAtoms = map[atom.Atom]bool{
	atom.Script: true, atom.Style: true, atom.Head: true, atom.Title: true,
	atom.Blockquote: true, // quoted replies in HTML mail
}

// isQuoteContainer reports Gmail/Outlook wrappers around the quoted thread.
func isQuoteContainer(n *html.Node) bool {
	for _, a := range n.Attr {
		if (a.Key == "class" && strings.Contains(a.Val, "gmail_quote")) ||
			(a.Key == "id" && (a.Val == "appendonsend" || a.Val == "divRplyFwdMsg")) {
			return true
		}
	}
	return false
}

// HTMLToText converts HTML to plain text: block elements become line breaks,
// list items get "- ", scripts/styles and quoted-reply containers are dropped,
// and entities are decoded. Unparseable input is returned unchanged.
func HTMLToText(s string) string {
	doc, err := html.Parse(strings.NewReader(s))
	if err != nil {
		return s
	}
	var b strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			if skipAtoms[n.DataAtom] || isQuoteContainer(n) {
				return
			}
			if blockAtoms[n.DataAtom] {
				b.WriteString("\n")
			}
			if n.DataAtom == atom.Li {
				b.WriteString("- ")
			}
		}
		if n.Type == html.TextNode {
			words := strings.Fields(n.Data)
			if len(words) == 0 {
				if n.Data != "" {
					writeSpace(&b)
				}
			} else {
				if unicode.IsSpace(rune(n.Data[0])) {
					writeSpace(&b)
				}
				b.WriteString(strings.Join(words, " "))
				if unicode.IsSpace(rune(n.Data[len(n.Data)-1])) {
					writeSpace(&b)
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
		if n.Type == html.ElementNode && blockAtoms[n.DataAtom] && n.DataAtom != atom.Br && n.DataAtom != atom.Li {
			b.WriteString("\n")
		}
	}
	walk(doc)
	return tidyLines(b.String())
}

// writeSpace adds a single space unless the output already ends in whitespace.
func writeSpace(b *strings.Builder) {
	s := b.String()
	if s != "" && !strings.HasSuffix(s, " ") && !strings.HasSuffix(s, "\n") {
		b.WriteString(" ")
	}
}

var multiBlank = regexp.MustCompile(`\n{3,}`)

// tidyLines trims each line and collapses runs of blank lines to one.
func tidyLines(s string) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimSpace(l)
	}
	return strings.TrimSpace(multiBlank.ReplaceAllString(strings.Join(lines, "\n"), "\n\n"))
}
