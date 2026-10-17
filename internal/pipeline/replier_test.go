package pipeline

import (
	"context"
	"errors"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

type memDrafts struct {
	pending []store.PendingReply
	saved   []store.StoredDraft
}

func (m *memDrafts) PendingReplies(context.Context, int) ([]store.PendingReply, error) {
	return m.pending, nil
}

func (m *memDrafts) Insert(_ context.Context, d store.StoredDraft) (string, error) {
	m.saved = append(m.saved, d)
	return "d", nil
}

type memReassign struct{ moved map[string]string }

func (m *memReassign) Reassign(_ context.Context, id, queue, reason string) (bool, error) {
	m.moved[id] = queue + "|" + reason
	return true, nil
}

type fakeReplyML struct {
	retrieve map[string]mlclient.RetrieveResponse
	verify   mlclient.Verification
	drafts   int
}

func (f *fakeReplyML) Retrieve(_ context.Context, r mlclient.RetrieveRequest) (mlclient.RetrieveResponse, error) {
	res, ok := f.retrieve[r.TicketID]
	if !ok {
		return res, errors.New("ml-service down")
	}
	return res, nil
}

func (f *fakeReplyML) Draft(_ context.Context, r mlclient.DraftRequest) (mlclient.GeneratedDraft, error) {
	f.drafts++
	return mlclient.GeneratedDraft{TicketID: r.TicketID, Subject: "Re", Body: "x [1]", Citations: []string{"c1"}}, nil
}

func (f *fakeReplyML) Verify(context.Context, mlclient.VerifyRequest) (mlclient.Verification, error) {
	return f.verify, nil
}

func TestReplierPaths(t *testing.T) {
	chunk := []model.RetrievedChunk{{ChunkID: "c1", DocID: "d", Text: "t", Score: 0.8}}
	half := 0.5
	ml := &fakeReplyML{
		retrieve: map[string]mlclient.RetrieveResponse{
			"empty":   {Mode: "dense"},
			"low":     {Mode: "dense", MaxScore: 0.3, RetrievalFloor: 0.6, BelowFloor: true, Chunks: chunk},
			"unfaith": {Mode: "dense", MaxScore: 0.8, RetrievalFloor: 0.6, Chunks: chunk},
		},
		verify: mlclient.Verification{Draft: model.ReplyDraft{Status: "abstained", FaithfulnessScore: &half,
			UnsupportedClaims: []string{"made up"}}, AbstainReason: "faithfulness 0.50 < 0.9"},
	}
	drafts := &memDrafts{pending: []store.PendingReply{
		{Ticket: model.Ticket{ID: "empty"}}, {Ticket: model.Ticket{ID: "low"}},
		{Ticket: model.Ticket{ID: "unfaith"}}, {Ticket: model.Ticket{ID: "down"}},
	}}
	routes := &memReassign{moved: map[string]string{}}
	au := &memAudit{}
	st, err := (&Replier{Drafts: drafts, Routes: routes, ML: ml, Audit: au}).ReplyPending(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if st != (ReplyStats{Pending: 4, Abstained: 3, Failed: 1}) {
		t.Fatalf("stats %+v", st)
	}
	if ml.drafts != 1 {
		t.Fatalf("drafted %d times; empty and below-floor tickets must not be drafted", ml.drafts)
	}
	reasons := map[string]string{}
	for _, d := range drafts.saved {
		reasons[d.TicketID] = d.AbstainReason
	}
	if reasons["empty"] != "no retrieved sources" || reasons["low"] != "best retrieval score 0.300 < floor 0.6" ||
		reasons["unfaith"] != "faithfulness 0.50 < 0.9" {
		t.Fatalf("reasons %+v", reasons)
	}
	if routes.moved["unfaith"] != "human_review|draft abstained: faithfulness 0.50 < 0.9" || len(routes.moved) != 3 {
		t.Fatalf("reassigned %+v", routes.moved)
	}
	if au.entries[0].Action != "reply.abstained" {
		t.Fatalf("audit %+v", au.entries[0])
	}
}
