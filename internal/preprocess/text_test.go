package preprocess

import "testing"

func TestHTMLToText(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"paragraphs and br", "<p>Hello,</p><p>Line one<br>Line two</p>", "Hello,\n\nLine one\nLine two"},
		{"entities and inline", "<div>Price &amp; tax: <b>5&nbsp;USD</b></div>", "Price & tax: 5 USD"}, // nbsp normalised to a space,
		{"scripts dropped", "<html><head><title>x</title><style>p{}</style></head><body><script>alert(1)</script>Hi</body></html>", "Hi"},
		{"spaces around inline tags", "<p>app <b>crashes</b> on <i>start</i>up</p>", "app crashes on startup"},
		{"lists", "<ul><li>one</li><li>two</li></ul>", "- one\n- two"},
		{"blockquote dropped", "<p>New text</p><blockquote>old text</blockquote>", "New text"},
		{"gmail quote dropped", `<div>Answer</div><div class="gmail_quote">On Mon wrote: old</div>`, "Answer"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := HTMLToText(tc.in); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestLooksLikeHTML(t *testing.T) {
	for in, want := range map[string]bool{
		"<html><body>x</body></html>": true,
		"<p>hi</p>":                   true,
		"use a < b and c > d":         false,
		"plain text":                  false,
	} {
		if got := LooksLikeHTML(in); got != want {
			t.Errorf("LooksLikeHTML(%q) = %v", in, got)
		}
	}
}

func TestStripQuoted(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"gmail one line", "Still broken.\n\nOn Fri, Oct 3, 2026 at 9:12 AM Priya <p@x.com> wrote:\n> old\n> older", "Still broken."},
		{"gmail wrapped", "Ping.\n\nOn Fri, Oct 3, 2026 at 9:12 AM Priya N <\npriya@x.com> wrote:\n> old", "Ping."},
		{"outlook header", "Thanks!\n\nFrom: Support <s@x.com>\nSent: Monday\nSubject: Re: x\n\nold", "Thanks!"},
		{"outlook underscores", "Yes.\n________________________________\nFrom: A\nold", "Yes."},
		{"original message", "See below\n-----Original Message-----\nold", "See below"},
		{"inline > lines dropped", "Top\n> quoted\nmore", "Top\nmore"},
		{"no quote", "Just a question about billing.", "Just a question about billing."},
		{"all quoted keeps original", "> only quoted", "> only quoted"},
		{"mid-sentence 'on' is not a header", "I depend on it, it wrote: nothing", "I depend on it, it wrote: nothing"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := StripQuoted(tc.in); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestStripSignature(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"rfc delimiter", "Help please.\n\n-- \nLena Fischer\nPlatform Ops", "Help please."},
		{"delimiter without space", "Help.\n--\nLena", "Help."},
		{"mobile footer", "On my way.\n\nSent from my iPhone", "On my way."},
		{"outlook mobile", "Ok\nGet Outlook for iOS", "Ok"},
		{"sign-off kept", "Can you refund it?\nThanks,\nPriya", "Can you refund it?\nThanks,\nPriya"},
		{"dashes inside text kept", "Use --force -- carefully", "Use --force -- carefully"},
		{"only signature keeps original", "-- \nLena", "-- \nLena"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := StripSignature(tc.in); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}
