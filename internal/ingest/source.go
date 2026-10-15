package ingest

import (
	"context"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Source is one ingestion channel. Name returns the source type, which is also
// what adapters put in Ticket.Source ("csv", "email", "zendesk"), so outbound
// replies can be routed back through the originating adapter.
type Source interface {
	Name() string
	// Fetch returns tickets received after since. Adapters may return tickets
	// already seen; dedupe happens in preprocess/store.
	Fetch(ctx context.Context, since time.Time) ([]model.Ticket, error)
	// Reply sends d as a reply to t through this source.
	Reply(ctx context.Context, t model.Ticket, d model.ReplyDraft) error
}
