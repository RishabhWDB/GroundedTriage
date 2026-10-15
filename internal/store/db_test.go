package store

import "testing"

func TestSearchPath(t *testing.T) {
	cases := map[string]string{
		"":       "public, extensions",
		"test_x": `"test_x", public, extensions`,
		`we"ird`: `"we""ird", public, extensions`,
	}
	for schema, want := range cases {
		if got := SearchPath(schema); got != want {
			t.Errorf("SearchPath(%q) = %q, want %q", schema, got, want)
		}
	}
}
