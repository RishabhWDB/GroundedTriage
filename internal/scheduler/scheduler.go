package scheduler

import (
	"context"
	"fmt"
	"sort"
	"time"
)

// Interval is a half-open time range [Start, End).
type Interval struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

func (a Interval) overlaps(b Interval) bool { return a.Start.Before(b.End) && b.Start.Before(a.End) }

// Event is a calendar event to create.
type Event struct {
	Summary     string
	Description string
	Location    string
	Start, End  time.Time
	Attendees   []string
}

// Calendar is the scheduler's view of a calendar.
type Calendar interface {
	// FreeBusy returns the busy intervals overlapping [from, to).
	FreeBusy(ctx context.Context, from, to time.Time) ([]Interval, error)
	// CreateEvent creates the event and returns its id.
	CreateEvent(ctx context.Context, e Event) (string, error)
}

// Config holds the scheduling policy.
type Config struct {
	Location        *time.Location // working hours are in this zone
	DayStart        time.Duration  // e.g. 9h after local midnight
	DayEnd          time.Duration  // e.g. 17h
	Workdays        []time.Weekday
	DefaultDuration time.Duration // when the request doesn't say
	SlotStep        time.Duration // candidate starts are aligned to this
	MinLead         time.Duration // never offer a slot sooner than now + MinLead
	Horizon         time.Duration // how far ahead to search
}

// DefaultConfig is Mon–Fri 09:00–17:00 in loc, 30-minute meetings and
// steps, 2 hours' lead, searched 14 days ahead.
func DefaultConfig(loc *time.Location) Config {
	return Config{
		Location: loc, DayStart: 9 * time.Hour, DayEnd: 17 * time.Hour,
		Workdays:        []time.Weekday{time.Monday, time.Tuesday, time.Wednesday, time.Thursday, time.Friday},
		DefaultDuration: 30 * time.Minute, SlotStep: 30 * time.Minute,
		MinLead: 2 * time.Hour, Horizon: 14 * 24 * time.Hour,
	}
}

// Request is what the extraction (ml-service /extract_meeting) produced.
type Request struct {
	TicketID  string
	Requester string     // attendee email
	Start     *time.Time // nil when the sender proposed no time
	Duration  time.Duration
	Summary   string
	Location  string // venue detail ("Zoom", an address, ...)
}

// Outcome statuses.
const (
	Booked       = "booked"        // requested slot was free: event created
	CounterOffer = "counter_offer" // a different slot is proposed (no event yet)
	NoSlot       = "no_slot"       // nothing free within the horizon
)

// Outcome of one scheduling attempt.
type Outcome struct {
	Status    string    `json:"status"`
	Slot      *Interval `json:"slot,omitempty"`      // booked or proposed slot
	Requested *Interval `json:"requested,omitempty"` // what the sender asked for
	EventID   string    `json:"event_id,omitempty"`
	Reason    string    `json:"reason"`
}

// Scheduler books the requested slot when it is free and inside working
// hours; otherwise it proposes the next free working-hours slot.
type Scheduler struct {
	Calendar Calendar
	Config   Config
	Now      func() time.Time // default time.Now
}

func (s *Scheduler) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

// Schedule decides and, for a free requested slot, creates the event.
func (s *Scheduler) Schedule(ctx context.Context, req Request) (Outcome, error) {
	cfg := s.Config
	dur := req.Duration
	if dur <= 0 {
		dur = cfg.DefaultDuration
	}
	now := s.now()
	earliest := now.Add(cfg.MinLead)
	from, to := earliest, earliest.Add(cfg.Horizon)
	var requested *Interval
	if req.Start != nil {
		requested = &Interval{Start: *req.Start, End: req.Start.Add(dur)}
		if requested.Start.Before(from) {
			from = requested.Start
		}
	}
	busy, err := s.Calendar.FreeBusy(ctx, from, to)
	if err != nil {
		return Outcome{}, fmt.Errorf("freebusy: %w", err)
	}
	sort.Slice(busy, func(i, j int) bool { return busy[i].Start.Before(busy[j].Start) })

	reason := "no time proposed"
	searchFrom := earliest
	if requested != nil {
		switch {
		case requested.Start.Before(earliest):
			reason = "requested time is too soon or in the past"
		case !s.inWorkingHours(*requested):
			reason = "requested time is outside working hours"
		case isBusy(*requested, busy):
			reason = "requested time is busy"
			searchFrom = requested.Start // propose the nearest later slot
		default:
			id, err := s.Calendar.CreateEvent(ctx, Event{Summary: req.Summary, Description: "Ticket " + req.TicketID,
				Location: req.Location, Start: requested.Start, End: requested.End, Attendees: nonEmpty(req.Requester)})
			if err != nil {
				return Outcome{}, fmt.Errorf("create event: %w", err)
			}
			return Outcome{Status: Booked, Slot: requested, Requested: requested, EventID: id,
				Reason: "requested time was free"}, nil
		}
	}
	if slot, ok := s.nextFree(searchFrom, to, dur, busy); ok {
		return Outcome{Status: CounterOffer, Slot: &slot, Requested: requested, Reason: reason}, nil
	}
	return Outcome{Status: NoSlot, Requested: requested,
		Reason: reason + "; no free working-hours slot in the next " + cfg.Horizon.String()}, nil
}

func nonEmpty(s string) []string {
	if s == "" {
		return nil
	}
	return []string{s}
}

func isBusy(iv Interval, busy []Interval) bool {
	for _, b := range busy {
		if iv.overlaps(b) {
			return true
		}
	}
	return false
}

// inWorkingHours: the whole interval falls on one workday inside the hours.
func (s *Scheduler) inWorkingHours(iv Interval) bool {
	cfg := s.Config
	st, en := iv.Start.In(cfg.Location), iv.End.In(cfg.Location)
	day := time.Date(st.Year(), st.Month(), st.Day(), 0, 0, 0, 0, cfg.Location)
	if !containsDay(cfg.Workdays, st.Weekday()) {
		return false
	}
	open, close := addClock(day, cfg.DayStart), addClock(day, cfg.DayEnd)
	return !st.Before(open) && !en.After(close)
}

// nextFree returns the first free slot of length dur starting at or after
// from (aligned to SlotStep within the day) and before to.
func (s *Scheduler) nextFree(from, to time.Time, dur time.Duration, busy []Interval) (Interval, bool) {
	cfg := s.Config
	f := from.In(cfg.Location)
	for day := time.Date(f.Year(), f.Month(), f.Day(), 0, 0, 0, 0, cfg.Location); day.Before(to); day = day.AddDate(0, 0, 1) {
		if !containsDay(cfg.Workdays, day.Weekday()) {
			continue
		}
		for st := addClock(day, cfg.DayStart); !st.Add(dur).After(addClock(day, cfg.DayEnd)); st = st.Add(cfg.SlotStep) {
			if st.Before(from) || !st.Before(to) {
				continue
			}
			iv := Interval{Start: st, End: st.Add(dur)}
			if !isBusy(iv, busy) {
				return iv, true
			}
		}
	}
	return Interval{}, false
}

// addClock adds a wall-clock offset to local midnight (DST-safe for whole hours).
func addClock(midnight time.Time, d time.Duration) time.Time {
	h, m := int(d/time.Hour), int((d%time.Hour)/time.Minute)
	return time.Date(midnight.Year(), midnight.Month(), midnight.Day(), h, m, 0, 0, midnight.Location())
}

func containsDay(days []time.Weekday, d time.Weekday) bool {
	for _, x := range days {
		if x == d {
			return true
		}
	}
	return false
}
