package routing

import (
	"context"
	"crypto/sha256"
	"log/slog"
	"os"
	"sync/atomic"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

// Engine evaluates the active rule set and hot-reloads it when the file
// changes. Polling (not fsnotify) keeps it dependency-free and works with
// editors that replace files; a change is picked up within one interval. An
// invalid edit is logged and the previous rules stay active.
type Engine struct {
	path  string
	rules atomic.Pointer[RuleSet]
	hash  [32]byte
	Log   *slog.Logger
}

// NewEngine loads path; it fails if the initial file is invalid.
func NewEngine(path string) (*Engine, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	rs, err := ParseRules(data)
	if err != nil {
		return nil, err
	}
	e := &Engine{path: path, hash: sha256.Sum256(data)}
	e.rules.Store(rs)
	return e, nil
}

func (e *Engine) log() *slog.Logger {
	if e.Log != nil {
		return e.Log
	}
	return slog.Default()
}

// Rules returns the active rule set.
func (e *Engine) Rules() *RuleSet { return e.rules.Load() }

// Route evaluates the active rules.
func (e *Engine) Route(in Input) model.Route { return e.rules.Load().Evaluate(in) }

// Reload re-reads the file if its content changed. It returns true when new
// rules were activated; an unreadable or invalid file leaves the old rules.
func (e *Engine) Reload() (bool, error) {
	data, err := os.ReadFile(e.path)
	if err != nil {
		return false, err
	}
	h := sha256.Sum256(data)
	if h == e.hash {
		return false, nil
	}
	rs, err := ParseRules(data)
	if err != nil {
		e.hash = h // don't re-log the same broken content every poll
		return false, err
	}
	e.hash = h
	e.rules.Store(rs)
	return true, nil
}

// Watch polls the file every interval until ctx is cancelled.
func (e *Engine) Watch(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			changed, err := e.Reload()
			switch {
			case err != nil:
				e.log().Error("routing: rules reload failed; keeping previous rules", "path", e.path, "err", err)
			case changed:
				e.log().Info("routing: rules reloaded", "path", e.path, "rules", len(e.Rules().Rules))
			}
		}
	}
}
