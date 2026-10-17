package pipeline

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/model"
	"github.com/RishabhWDB/GroundedTriage/internal/scheduler"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

type memMeetings struct {
	pending []model.Ticket
	saved   []store.Meeting
}

func (m *memMeetings) PendingSchedule(context.Context, int) ([]model.Ticket, error) {
	return m.pending, nil
}

func (m *memMeetings) Insert(_ context.Context, rec store.Meeting) error {
	m.saved = append(m.saved, rec)
	return nil
}

type stubExtractor map[string]mlclient.MeetingExtraction

func (s stubExtractor) ExtractMeeting(_ context.Context, r mlclient.MeetingRequest) (mlclient.MeetingExtraction, error) {
	e, ok := s[r.TicketID]
	if !ok {
		return e, errors.New("ml-service down")
	}
	return e, nil
}

func TestSchedulePendingOutcomes(t *testing.T) {
	loc, _ := time.LoadLocation("Europe/London")
	now := time.Date(2026, 10, 13, 8, 0, 0, 0, loc)
	free := time.Date(2026, 10, 15, 15, 0, 0, 0, loc)
	busy := time.Date(2026, 10, 15, 11, 0, 0, 0, loc)
	cal := &scheduler.FakeCalendar{Busy: []scheduler.Interval{{Start: busy, End: busy.Add(time.Hour)}}}
	thirty, zoom := 30, "Zoom"
	ml := stubExtractor{
		"book":   {IsMeetingRequest: true, Start: &free, DurationMinutes: &thirty, Venue: "video", VenueDetail: &zoom, Timezone: "Europe/London"},
		"busy":   {IsMeetingRequest: true, Start: &busy, Timezone: "Europe/London"},
		"notime": {IsMeetingRequest: true, Timezone: "Europe/London"},
		"nomeet": {IsMeetingRequest: false, Timezone: "Europe/London"},
	}
	st := &memMeetings{pending: []model.Ticket{
		{ID: "book", Sender: "a@b.com", Subject: "Q4 review"}, {ID: "busy"}, {ID: "notime"}, {ID: "nomeet"}, {ID: "down"},
	}}
	au := &memAudit{}
	ms := &MeetingScheduler{Store: st, ML: ml, Timezone: "Europe/London", Audit: au,
		Scheduler: &scheduler.Scheduler{Calendar: cal, Config: scheduler.DefaultConfig(loc), Now: func() time.Time { return now }}}
	stats, err := ms.SchedulePending(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"booked": 1, "counter_offer": 2, "not_a_meeting": 1}
	for k, v := range want {
		if stats.ByStatus[k] != v {
			t.Fatalf("by status %+v", stats.ByStatus)
		}
	}
	if stats.Failed != 1 || stats.Pending != 5 || len(st.saved) != 4 {
		t.Fatalf("stats %+v saved %d", stats, len(st.saved))
	}
	b := st.saved[0]
	if b.Status != "booked" || b.EventID == "" || !b.SlotStart.Equal(free) || b.Venue != "video" {
		t.Fatalf("booked %+v", b)
	}
	if e := cal.Created[0]; e.Location != "Zoom" || e.Attendees[0] != "a@b.com" || e.Summary != "Support call: Q4 review" || e.End.Sub(e.Start) != 30*time.Minute {
		t.Fatalf("event %+v", e)
	}
	if c := st.saved[1]; c.Status != "counter_offer" || !c.RequestedStart.Equal(busy) || !c.SlotStart.Equal(busy.Add(time.Hour)) {
		t.Fatalf("counter %+v", c)
	}
	if n := st.saved[3]; n.Status != "not_a_meeting" || n.SlotStart != nil {
		t.Fatalf("not a meeting %+v", n)
	}
	if len(au.entries) != 4 || au.entries[0].Action != "meeting.booked" {
		t.Fatalf("audit %+v", au.entries)
	}
}
