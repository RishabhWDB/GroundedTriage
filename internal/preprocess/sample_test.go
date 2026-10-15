package preprocess_test

import (
	"context"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
)

// TestSampleFileNormalisation checks the cleaning rules end to end on the
// committed sample CSV: only the Portuguese row is skipped, and the HTML,
// quoted-reply and signature rows come out clean.
func TestSampleFileNormalisation(t *testing.T) {
	src := &ingest.CSVSource{Path: "../../datasets/sample_emails.csv"}
	tickets, err := src.Fetch(context.Background(), time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	skipped := 0
	bySubject := map[string]string{}
	for _, tk := range tickets {
		out, skip, _ := preprocess.Normalize(tk)
		if skip != "" {
			skipped++
			if skip != preprocess.SkipNonEnglish || out.Subject != "Problema com a fatura de setembro" {
				t.Errorf("unexpected skip %q for %q", skip, out.Subject)
			}
			continue
		}
		bySubject[out.Subject] = out.Body
	}
	if skipped != 1 {
		t.Fatalf("skipped = %d, want 1", skipped)
	}
	checks := map[string]string{
		"Mobile app crashes on startup":               "Hello,\n\nSince the update to version 4.2 the iOS app crashes immediately on startup. iPhone 14, iOS 19.1.\n\nThanks,\nRetail Co IT",
		"Re: Charged twice for October invoice":       "Following up on this - I still see both charges on my statement.",
		"API returning 500 errors since this morning": "Our integration has been getting HTTP 500 from /v2/orders since about 07:00 UTC. This is blocking all our order syncs. Please treat as urgent.",
	}
	for subj, want := range checks {
		if got := bySubject[subj]; got != want {
			t.Errorf("%s:\n got %q\nwant %q", subj, got, want)
		}
	}
}
