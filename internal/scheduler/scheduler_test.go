package scheduler

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

var london, _ = time.LoadLocation("Europe/London")

// Tuesday 13 Oct 2026, 08:00 London.
var now = time.Date(2026, 10, 13, 8, 0, 0, 0, london)

func at(day, hour, minute int) time.Time { return time.Date(2026, 10, day, hour, minute, 0, 0, london) }

func sched(cal *FakeCalendar) *Scheduler {
	return &Scheduler{Calendar: cal, Config: DefaultConfig(london), Now: func() time.Time { return now }}
}

func TestFreeSlotIsBooked(t *testing.T) {
	cal := &FakeCalendar{}
	start := at(15, 15, 0) // Thursday 15:00
	out, err := sched(cal).Schedule(context.Background(), Request{TicketID: "t1", Requester: "a@b.com",
		Start: &start, Duration: 45 * time.Minute, Summary: "Q4 review", Location: "Zoom"})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != Booked || out.EventID != "fake-event-1" || !out.Slot.Start.Equal(start) || !out.Slot.End.Equal(start.Add(45*time.Minute)) {
		t.Fatalf("outcome %+v", out)
	}
	e := cal.Created[0]
	if e.Summary != "Q4 review" || e.Location != "Zoom" || e.Attendees[0] != "a@b.com" || !strings.Contains(e.Description, "t1") {
		t.Fatalf("event %+v", e)
	}
}

func TestBusySlotGetsNextFreeSlotAsCounterOffer(t *testing.T) {
	start := at(15, 15, 0)
	cal := &FakeCalendar{Busy: []Interval{
		{Start: at(15, 14, 30), End: at(15, 16, 0)}, // overlaps 15:00-15:30
		{Start: at(15, 16, 0), End: at(15, 16, 30)},
	}}
	out, err := sched(cal).Schedule(context.Background(), Request{TicketID: "t", Start: &start})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != CounterOffer || out.Reason != "requested time is busy" || len(cal.Created) != 0 {
		t.Fatalf("outcome %+v created %d", out, len(cal.Created))
	}
	if !out.Slot.Start.Equal(at(15, 16, 30)) || !out.Requested.Start.Equal(start) {
		t.Fatalf("proposed %v, want Thu 16:30", out.Slot)
	}
}

func TestOutsideHoursAndWeekendsAreCounterOffered(t *testing.T) {
	evening := at(14, 19, 0)  // Wed 19:00
	saturday := at(17, 10, 0) // Sat
	overrun := at(14, 16, 45) // 30 min would end 17:15
	for _, start := range []time.Time{evening, saturday, overrun} {
		out, _ := sched(&FakeCalendar{}).Schedule(context.Background(), Request{Start: &start})
		if out.Status != CounterOffer || out.Reason != "requested time is outside working hours" {
			t.Fatalf("%v: %+v", start, out)
		}
	}
	// The first slot after the 2h lead on Tuesday is 10:00.
	out, _ := sched(&FakeCalendar{}).Schedule(context.Background(), Request{Start: &evening})
	if !out.Slot.Start.Equal(at(13, 10, 0)) {
		t.Fatalf("proposed %v", out.Slot.Start)
	}
}

func TestNoTimeProposedAndPastTime(t *testing.T) {
	out, _ := sched(&FakeCalendar{}).Schedule(context.Background(), Request{})
	if out.Status != CounterOffer || out.Requested != nil || out.Reason != "no time proposed" || !out.Slot.Start.Equal(at(13, 10, 0)) {
		t.Fatalf("no time: %+v", out)
	}
	past := at(12, 10, 0)
	out, _ = sched(&FakeCalendar{}).Schedule(context.Background(), Request{Start: &past})
	if out.Status != CounterOffer || !strings.Contains(out.Reason, "past") {
		t.Fatalf("past: %+v", out)
	}
}

func TestNoSlotWhenFullyBooked(t *testing.T) {
	cal := &FakeCalendar{Busy: []Interval{{Start: now, End: now.Add(30 * 24 * time.Hour)}}}
	start := at(15, 15, 0)
	out, err := sched(cal).Schedule(context.Background(), Request{Start: &start})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != NoSlot || out.Slot != nil || !strings.Contains(out.Reason, "no free working-hours slot") {
		t.Fatalf("outcome %+v", out)
	}
}

func TestCalendarErrorsPropagate(t *testing.T) {
	start := at(15, 15, 0)
	if _, err := sched(&FakeCalendar{Err: errors.New("401")}).Schedule(context.Background(), Request{Start: &start}); err == nil {
		t.Fatal("expected an error")
	}
}

func TestBookedSlotBecomesBusy(t *testing.T) {
	cal := &FakeCalendar{}
	s := sched(cal)
	start := at(15, 11, 0)
	if out, _ := s.Schedule(context.Background(), Request{Start: &start}); out.Status != Booked {
		t.Fatalf("first: %+v", out)
	}
	out, _ := s.Schedule(context.Background(), Request{Start: &start})
	if out.Status != CounterOffer || !out.Slot.Start.Equal(at(15, 11, 30)) {
		t.Fatalf("second request for the same slot: %+v", out)
	}
}

func TestDSTWeekKeepsLocalHours(t *testing.T) {
	// UK clocks go back on Sunday 25 Oct 2026; Monday 26 Oct 09:00 is GMT.
	cal := &FakeCalendar{Busy: []Interval{{Start: now, End: at(26, 0, 0)}}}
	out, _ := sched(cal).Schedule(context.Background(), Request{})
	if want := time.Date(2026, 10, 26, 9, 0, 0, 0, london); !out.Slot.Start.Equal(want) {
		t.Fatalf("proposed %v, want %v", out.Slot.Start, want)
	}
}
