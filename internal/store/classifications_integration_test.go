//go:build integration

package store_test

import (
	"context"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

func TestClassificationSpamFilterRoundTrip(t *testing.T) {
	pool := testDB.Require(t)
	ctx := context.Background()
	id, err := store.Tickets{Pool: pool}.Insert(ctx, model.Ticket{Source: model.SourceCSV,
		ExternalID: storetest.UniqueID("spam"), Body: "win money", ReceivedAt: time.Now(),
		ContentHash: storetest.UniqueID("h")})
	if err != nil {
		t.Fatal(err)
	}
	repo := store.Classifications{Pool: pool}
	score := 0.97
	in := model.Classification{TicketID: id, Category: "Spam", CategoryConfidence: score, Tier: "spam_filter",
		Urgency: "medium", UrgencyConfidence: 0.4, UrgencySource: "heuristic", SpamScore: &score,
		ModelVersions: map[string]string{"spam_filter": "spam-x:support"}}
	if err := repo.Insert(ctx, in); err != nil {
		t.Fatal(err)
	}
	got, ok, err := repo.Latest(ctx, id)
	if err != nil || !ok {
		t.Fatalf("latest ok=%v err=%v", ok, err)
	}
	if got.Tier != "spam_filter" || got.SpamScore == nil || *got.SpamScore < 0.969 || got.PossibleSpam {
		t.Fatalf("got %+v", got)
	}

	// A topic classification in the review band, and one without a filter (NULL score).
	in2 := model.Classification{TicketID: id, Category: "Billing", CategoryConfidence: 0.8, Tier: "fast",
		Urgency: "low", UrgencyConfidence: 0.4, UrgencySource: "heuristic", PossibleSpam: true}
	if err := repo.Insert(ctx, in2); err != nil {
		t.Fatal(err)
	}
	got, _, _ = repo.Latest(ctx, id)
	if !got.PossibleSpam || got.SpamScore != nil || got.Tier != "fast" {
		t.Fatalf("got %+v", got)
	}

	bad := in
	bad.Tier = "regex"
	if err := repo.Insert(ctx, bad); err == nil {
		t.Fatal("tier check constraint did not reject 'regex'")
	}
}
