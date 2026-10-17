package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/scheduler"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// MeetingStore reads tickets awaiting scheduling and stores outcomes (store.Meetings).
type MeetingStore interface {
	PendingSchedule(ctx context.Context, limit int) ([]model.Ticket, error)
	Insert(ctx context.Context, m store.Meeting) error
}

// MeetingExtractor calls the ml-service (mlclient.Client).
type MeetingExtractor interface {
	ExtractMeeting(ctx context.Context, req mlclient.MeetingRequest) (mlclient.MeetingExtraction, error)
}

// MeetingScheduler handles tickets routed to the `schedule` action: extract
// the request (ml-service), then book or propose a slot (internal/scheduler).
// A ticket whose extraction or calendar call fails is retried next run.
type MeetingScheduler struct {
	Store     MeetingStore
	ML        MeetingExtractor
	Scheduler *scheduler.Scheduler
	Timezone  string // default timezone for the extraction (GOOGLE_TIMEZONE)
	Audit     AuditWriter
	BatchSize int // default 20 (each ticket is an LLM call)
	Log       *slog.Logger
}

// MeetingStats summarises one run.
type MeetingStats struct {
	Pending  int
	ByStatus map[string]int
	Failed   int
}

func (m *MeetingScheduler) log() *slog.Logger {
	if m.Log != nil {
		return m.Log
	}
	return slog.Default()
}

// SchedulePending processes up to BatchSize tickets.
func (m *MeetingScheduler) SchedulePending(ctx context.Context) (MeetingStats, error) {
	st := MeetingStats{ByStatus: map[string]int{}}
	limit := m.BatchSize
	if limit <= 0 {
		limit = 20
	}
	tickets, err := m.Store.PendingSchedule(ctx, limit)
	if err != nil {
		return st, err
	}
	st.Pending = len(tickets)
	for _, t := range tickets {
		out, err := m.scheduleOne(ctx, t)
		if err != nil {
			st.Failed++
			m.log().Warn("meeting: ticket failed; retrying next run", "ticket", t.ID, "err", err)
			continue
		}
		if err := m.Store.Insert(ctx, out); err != nil {
			return st, err
		}
		st.ByStatus[out.Status]++
		m.audit(ctx, store.AuditEntry{EntityType: "ticket", EntityID: t.ID, Action: "meeting." + out.Status,
			Details: map[string]any{"reason": out.Reason, "slot_start": out.SlotStart, "event_id": out.EventID}})
	}
	if st.Pending > 0 {
		m.log().Info("meeting: run", "pending", st.Pending, "by_status", st.ByStatus, "failed", st.Failed)
	}
	return st, nil
}

func (m *MeetingScheduler) scheduleOne(ctx context.Context, t model.Ticket) (store.Meeting, error) {
	ext, err := m.ML.ExtractMeeting(ctx, mlclient.MeetingRequest{TicketID: t.ID, Subject: t.Subject,
		Body: t.Body, ReceivedAt: t.ReceivedAt, DefaultTimezone: m.Timezone})
	if err != nil {
		return store.Meeting{}, fmt.Errorf("extract: %w", err)
	}
	venue := ext.Venue
	location := ""
	if ext.VenueDetail != nil {
		location = *ext.VenueDetail
	}
	if !ext.IsMeetingRequest {
		return store.Meeting{TicketID: t.ID, Status: "not_a_meeting", Timezone: ext.Timezone, Venue: venue,
			Reason: "extraction found no meeting request", Extraction: ext}, nil
	}
	req := scheduler.Request{TicketID: t.ID, Requester: t.Sender, Start: ext.Start, Location: location,
		Summary: meetingSummary(t)}
	if ext.DurationMinutes != nil {
		req.Duration = time.Duration(*ext.DurationMinutes) * time.Minute
	}
	out, err := m.Scheduler.Schedule(ctx, req)
	if err != nil {
		return store.Meeting{}, err
	}
	rec := store.Meeting{TicketID: t.ID, Status: out.Status, EventID: out.EventID, Timezone: ext.Timezone,
		Venue: venue, Reason: out.Reason, Extraction: ext}
	if out.Requested != nil {
		rec.RequestedStart = &out.Requested.Start
	}
	if out.Slot != nil {
		rec.SlotStart, rec.SlotEnd = &out.Slot.Start, &out.Slot.End
	}
	return rec, nil
}

func meetingSummary(t model.Ticket) string {
	if t.Subject != "" {
		return "Support call: " + t.Subject
	}
	return "Support call"
}

func (m *MeetingScheduler) audit(ctx context.Context, e store.AuditEntry) {
	if m.Audit == nil {
		return
	}
	if err := m.Audit.Write(ctx, e); err != nil {
		m.log().Warn("meeting: audit write failed", "err", err)
	}
}
