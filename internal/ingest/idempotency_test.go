package ingest

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

func TestCSVReplyWritesAKeyOnce(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replies.csv")
	s := &CSVSource{ReplyPath: path}
	tk := model.Ticket{ExternalID: "e1", Sender: "a@b.com", Subject: "Hi"}
	d := model.ReplyDraft{Body: "Hello [1]", IdempotencyKey: "draft-1"}
	for i := 0; i < 3; i++ {
		if err := s.Reply(context.Background(), tk, d); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Reply(context.Background(), tk, model.ReplyDraft{Body: "other", IdempotencyKey: "draft-2"}); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	if n := strings.Count(string(data), "draft-1"); n != 1 || !strings.Contains(string(data), "draft-2") {
		t.Fatalf("draft-1 written %d times:\n%s", n, data)
	}
	old := filepath.Join(t.TempDir(), "old.csv")
	_ = os.WriteFile(old, []byte("ExternalID,To,Subject,Body,Citations,SentAt\n"), 0o644)
	if err := (&CSVSource{ReplyPath: old}).Reply(context.Background(), tk, d); err == nil ||
		!strings.Contains(err.Error(), "IdempotencyKey") {
		t.Fatalf("old-format file: %v", err)
	}
}

func TestEmailMessageIDFromKey(t *testing.T) {
	tk := model.Ticket{ExternalID: "orig@x", Sender: "a@b.com", Subject: "Hi"}
	now := time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC)
	a, _ := BuildReply("support@acme.com", tk, model.ReplyDraft{Body: "x", IdempotencyKey: "draft-42"}, now)
	b, _ := BuildReply("support@acme.com", tk, model.ReplyDraft{Body: "x", IdempotencyKey: "draft-42"}, now)
	if !strings.Contains(string(a), "Message-ID: <draft-42@acme.com>") || string(a) != string(b) {
		t.Fatalf("message:\n%s", a)
	}
	c, _ := BuildReply("support@acme.com", tk, model.ReplyDraft{Body: "x"}, now)
	if strings.Contains(string(c), "draft-42") {
		t.Fatal("random Message-ID expected without a key")
	}
}

func TestZendeskReplySendsIdempotencyKey(t *testing.T) {
	m := &zdMock{handler: func(int, *http.Request) (int, http.Header, string) { return 200, nil, `{"ticket":{"id":7}}` }}
	z, _ := newZD(t, m)
	if err := z.Reply(context.Background(), model.Ticket{ExternalID: "7"}, model.ReplyDraft{Body: "b", IdempotencyKey: "draft-7"}); err != nil {
		t.Fatal(err)
	}
	if got := m.requests[0].Header.Get("Idempotency-Key"); got != "draft-7" {
		t.Fatalf("Idempotency-Key = %q", got)
	}
}
