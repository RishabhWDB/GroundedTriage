package scheduler

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// FakeCalendar is an in-memory Calendar for tests and demos: Busy holds
// pre-existing events; created events become busy too.
type FakeCalendar struct {
	mu      sync.Mutex
	Busy    []Interval
	Created []Event
	// Err, when set, is returned by every call.
	Err error
}

// FreeBusy returns the busy intervals overlapping [from, to).
func (f *FakeCalendar) FreeBusy(_ context.Context, from, to time.Time) ([]Interval, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	window := Interval{Start: from, End: to}
	var out []Interval
	for _, b := range f.Busy {
		if b.overlaps(window) {
			out = append(out, b)
		}
	}
	return out, nil
}

// CreateEvent records the event and marks its time busy.
func (f *FakeCalendar) CreateEvent(_ context.Context, e Event) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.Err != nil {
		return "", f.Err
	}
	f.Created = append(f.Created, e)
	f.Busy = append(f.Busy, Interval{Start: e.Start, End: e.End})
	return fmt.Sprintf("fake-event-%d", len(f.Created)), nil
}
