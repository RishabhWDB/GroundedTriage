package pipeline

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// ClassificationStore reads pending tickets and stores results (store.Classifications).
type ClassificationStore interface {
	Unclassified(ctx context.Context, limit int) ([]model.Ticket, error)
	Insert(ctx context.Context, c model.Classification) error
}

// MLClassifier calls the ml-service cascade (mlclient.Client).
type MLClassifier interface {
	Classify(ctx context.Context, req mlclient.ClassifyRequest) (model.Classification, error)
}

// Classifier sends accepted, unclassified tickets to POST /classify and
// persists the result (layer 2). Tickets that fail stay unclassified and are
// retried on the next run.
type Classifier struct {
	Store ClassificationStore
	ML    MLClassifier
	Audit AuditWriter
	// BatchSize caps the tickets handled per run (default 100).
	BatchSize int
	// MaxConsecutiveFailures stops a run early when the ml-service looks down
	// (default 3); the rest wait for the next run.
	MaxConsecutiveFailures int
	Log                    *slog.Logger
}

// ClassifyStats summarises one run.
type ClassifyStats struct {
	Pending    int
	Classified int
	Failed     int
	// Stopped is set when the run gave up after consecutive failures.
	Stopped bool
}

// ErrMLUnavailable is returned when a run stops after consecutive ml-service failures.
var ErrMLUnavailable = errors.New("ml-service unavailable")

func (c *Classifier) log() *slog.Logger {
	if c.Log != nil {
		return c.Log
	}
	return slog.Default()
}

// ClassifyPending classifies up to BatchSize pending tickets.
func (c *Classifier) ClassifyPending(ctx context.Context) (ClassifyStats, error) {
	limit := c.BatchSize
	if limit <= 0 {
		limit = 100
	}
	maxFail := c.MaxConsecutiveFailures
	if maxFail <= 0 {
		maxFail = 3
	}
	tickets, err := c.Store.Unclassified(ctx, limit)
	if err != nil {
		return ClassifyStats{}, err
	}
	stats := ClassifyStats{Pending: len(tickets)}
	if len(tickets) == 0 {
		return stats, nil
	}
	var classified []string
	byCategory := map[string]int{}
	byTier := map[string]int{}
	var lastErr error
	streak := 0
	for _, t := range tickets {
		if ctx.Err() != nil {
			break
		}
		res, err := c.classifyOne(ctx, t)
		if err != nil {
			stats.Failed++
			streak++
			lastErr = err
			c.log().Warn("classify: ticket failed", "ticket", t.ID, "err", err)
			if streak >= maxFail {
				stats.Stopped = true
				break
			}
			continue
		}
		streak = 0
		stats.Classified++
		byCategory[res.Category]++
		byTier[res.Tier]++
		if len(classified) < maxAuditIDs {
			classified = append(classified, t.ID)
		}
	}
	c.log().Info("classify: run", "pending", stats.Pending, "classified", stats.Classified,
		"failed", stats.Failed, "stopped", stats.Stopped)
	details := map[string]any{
		"pending": stats.Pending, "classified": stats.Classified, "failed": stats.Failed,
		"stopped": stats.Stopped, "by_category": byCategory, "by_tier": byTier,
		"classified_ticket_ids": classified,
	}
	if lastErr != nil {
		details["last_error"] = lastErr.Error()
	}
	c.audit(ctx, store.AuditEntry{EntityType: "pipeline", EntityID: "classify", Action: "classify.run", Details: details})
	if stats.Stopped {
		return stats, fmt.Errorf("%w after %d consecutive failures: %v", ErrMLUnavailable, maxFail, lastErr)
	}
	return stats, nil
}

func (c *Classifier) classifyOne(ctx context.Context, t model.Ticket) (model.Classification, error) {
	res, err := c.ML.Classify(ctx, mlclient.ClassifyRequest{TicketID: t.ID, Subject: t.Subject, Body: t.Body})
	if err != nil {
		return res, err
	}
	if res.TicketID != t.ID {
		return res, fmt.Errorf("ml-service answered for ticket %q, want %q", res.TicketID, t.ID)
	}
	if !slices.Contains(model.Categories, res.Category) {
		return res, fmt.Errorf("ml-service returned unknown category %q", res.Category)
	}
	return res, c.Store.Insert(ctx, res)
}

func (c *Classifier) audit(ctx context.Context, e store.AuditEntry) {
	if c.Audit == nil {
		return
	}
	if err := c.Audit.Write(ctx, e); err != nil {
		c.log().Warn("classify: audit write failed", "err", err)
	}
}
