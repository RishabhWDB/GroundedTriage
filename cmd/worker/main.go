// Command worker polls enabled ingestion sources (config/sources.yaml),
// stores new tickets (layer 1), classifies accepted tickets through the
// ml-service's POST /classify (layer 2, ML_SERVICE_URL), routes classified and
// skipped tickets with config/routing_rules.yaml + config/sla.yaml and
// escalates SLA breaches (layer 3), and books or proposes meetings for
// `schedule` routes (P4.4; MEETING_CALENDAR=google|fake|off), and drafts
// grounded replies for `reply` routes (P5.5: retrieve -> draft -> verify;
// abstentions go to human_review), and sends agent-approved drafts through
// the ticket's source adapter exactly once (P6.3).
//
//	go run ./cmd/worker            # poll every WORKER_POLL_INTERVAL (default 60s)
//	go run ./cmd/worker -once      # one poll, then exit
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/monitoring"
	"github.com/RishabhWDB/GroundedTriage/internal/outbound"
	"github.com/RishabhWDB/GroundedTriage/internal/pipeline"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/routing"
	"github.com/RishabhWDB/GroundedTriage/internal/scheduler"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func main() {
	once := flag.Bool("once", false, "poll every source once and exit")
	sourcesPath := flag.String("sources", "config/sources.yaml", "sources config file")
	rulesPath := flag.String("rules", "config/routing_rules.yaml", "routing rules (hot-reloaded)")
	slaPath := flag.String("sla", "config/sla.yaml", "SLA targets")
	monitoringPath := flag.String("monitoring", "config/monitoring.yaml", "monitoring reference + alert settings (P7.3)")
	flag.Parse()

	if err := envfile.Load(); err != nil {
		log.Fatalf("load .env: %v", err)
	}
	interval := 60 * time.Second
	if v := os.Getenv("WORKER_POLL_INTERVAL"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			log.Fatalf("WORKER_POLL_INTERVAL=%q: want a positive duration like 60s", v)
		}
		interval = d
	}
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		log.Fatal("DATABASE_URL is unset")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cfgs, err := ingest.LoadConfig(*sourcesPath)
	if err != nil {
		log.Fatal(err)
	}
	sources, err := ingest.DefaultRegistry().Build(cfgs)
	if err != nil {
		log.Fatal(err)
	}
	if len(sources) == 0 {
		log.Fatalf("no sources enabled in %s", *sourcesPath)
	}
	pool, err := store.OpenPool(ctx, url, store.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	mlURL := os.Getenv("ML_SERVICE_URL")
	if mlURL == "" {
		mlURL = "http://localhost:8001"
	}
	recorder := &monitoring.Recorder{}
	newML := func(timeout time.Duration) *mlclient.Client {
		c := mlclient.New(mlURL)
		c.Observe = recorder.Observe // P7.3 stage timings
		if timeout > 0 {
			c.Timeout = timeout
		}
		return c
	}
	monCfg, err := monitoring.LoadConfig(*monitoringPath)
	if err != nil {
		log.Fatal(err)
	}
	monitor := &monitoring.Job{Store: store.Monitoring{Pool: pool}, Config: monCfg, Recorder: recorder}
	classifier := &pipeline.Classifier{
		Store: store.Classifications{Pool: pool},
		ML:    newML(0),
		Audit: store.Audit{Pool: pool},
	}
	engine, err := routing.NewEngine(*rulesPath)
	if err != nil {
		log.Fatal(err)
	}
	sla, err := routing.LoadSLA(*slaPath)
	if err != nil {
		log.Fatal(err)
	}
	router := &pipeline.Router{
		Store: store.Routes{Pool: pool},
		Rules: engine,
		SLA:   sla,
		Clock: routing.SystemClock{},
		Audit: store.Audit{Pool: pool},
	}
	meetings, err := meetingScheduler(ctx, pool, newML)
	if err != nil {
		log.Fatal(err)
	}
	replyML := newML(120 * time.Second) // drafting and judging are LLM calls
	replier := &pipeline.Replier{
		Drafts: store.ReplyDrafts{Pool: pool},
		Routes: store.Routes{Pool: pool},
		ML:     replyML,
		Audit:  store.Audit{Pool: pool},
		Hybrid: os.Getenv("RETRIEVAL_HYBRID") == "true",
	}
	sender := &outbound.Sender{Queue: store.Outbound{Pool: pool}, Sources: sources, Audit: store.Audit{Pool: pool}}

	// After each poll: classify, route, escalate, schedule meetings. A failing step is logged and
	// the next one still runs (routing also covers skipped tickets).
	classify := func(ctx context.Context) {
		if _, err := classifier.ClassifyPending(ctx); err != nil {
			slog.Error("worker: classify", "err", err)
		}
		if _, err := router.RoutePending(ctx); err != nil {
			slog.Error("worker: route", "err", err)
		}
		if _, err := router.EscalateBreached(ctx); err != nil {
			slog.Error("worker: escalate", "err", err)
		}
		if meetings != nil {
			if _, err := meetings.SchedulePending(ctx); err != nil {
				slog.Error("worker: meetings", "err", err)
			}
		}
		if _, err := replier.ReplyPending(ctx); err != nil {
			slog.Error("worker: replies", "err", err)
		}
		if _, err := sender.SendApproved(ctx); err != nil {
			slog.Error("worker: outbound", "err", err)
		}
		if _, err := monitor.Run(ctx); err != nil {
			slog.Error("worker: monitoring", "err", err)
		}
	}

	poller := &pipeline.Poller{
		Sources:     sources,
		Ingestor:    &preprocess.Ingestor{Store: store.Tickets{Pool: pool}},
		Checkpoints: store.Checkpoints{Pool: pool},
		Audit:       store.Audit{Pool: pool},
		Overlap:     10 * time.Minute,
		AfterPoll:   classify,
	}

	if *once {
		failed := false
		for name, r := range poller.PollOnce(ctx) {
			slog.Info("worker: result", "source", name, "fetched", r.Fetched, "inserted", r.Stats.Inserted,
				"inserted_skipped", r.Stats.InsertedSkipped, "dup_id", r.Stats.DuplicateID,
				"dup_content", r.Stats.DuplicateContent, "dropped", r.Stats.Dropped, "err", r.Err)
			failed = failed || r.Err != nil
		}
		stats, err := classifier.ClassifyPending(ctx)
		slog.Info("worker: classify", "pending", stats.Pending, "classified", stats.Classified,
			"failed", stats.Failed, "err", err)
		rs, rerr := router.RoutePending(ctx)
		slog.Info("worker: route", "routed", rs.Routed, "by_action", rs.ByAction, "err", rerr)
		es, eerr := router.EscalateBreached(ctx)
		slog.Info("worker: escalate", "breached", es.Breached, "escalated", es.Escalated, "err", eerr)
		var merr error
		if meetings != nil {
			var ms pipeline.MeetingStats
			ms, merr = meetings.SchedulePending(ctx)
			slog.Info("worker: meetings", "pending", ms.Pending, "by_status", ms.ByStatus, "failed", ms.Failed, "err", merr)
		}
		rp, perr := replier.ReplyPending(ctx)
		slog.Info("worker: replies", "pending", rp.Pending, "grounded", rp.Grounded, "abstained", rp.Abstained,
			"failed", rp.Failed, "err", perr)
		ss, serr := sender.SendApproved(ctx)
		slog.Info("worker: outbound", "claimed", ss.Claimed, "sent", ss.Sent, "failed", ss.Failed, "stale", ss.Stale, "err", serr)
		mm, moerr := monitor.Run(ctx)
		slog.Info("worker: monitoring", "metrics", len(mm), "err", moerr)
		if failed || err != nil || rerr != nil || eerr != nil || merr != nil || perr != nil || serr != nil || moerr != nil {
			os.Exit(1)
		}
		return
	}
	go engine.Watch(ctx, 5*time.Second) // hot reload of the routing rules
	slog.Info("worker: polling", "sources", len(sources), "interval", interval)
	poller.Run(ctx, interval)
	slog.Info("worker stopped")
}

// meetingScheduler builds the P4.4 step from MEETING_CALENDAR: "google"
// (needs `make calendar-auth`), "fake" (an in-memory calendar for demos) or
// "off". Unset means google when the token file exists, otherwise off.
func meetingScheduler(ctx context.Context, pool *pgxpool.Pool, newML func(time.Duration) *mlclient.Client) (*pipeline.MeetingScheduler, error) {
	tz := os.Getenv("GOOGLE_TIMEZONE")
	if tz == "" {
		tz = "UTC"
	}
	loc, err := time.LoadLocation(tz)
	if err != nil {
		return nil, fmt.Errorf("GOOGLE_TIMEZONE=%q: %w", tz, err)
	}
	tokenFile := os.Getenv("GOOGLE_TOKEN_FILE")
	if tokenFile == "" {
		tokenFile = "token.json"
	}
	mode := os.Getenv("MEETING_CALENDAR")
	if mode == "" {
		mode = "off"
		if _, err := os.Stat(tokenFile); err == nil {
			mode = "google"
		}
	}
	var cal scheduler.Calendar
	switch mode {
	case "off":
		slog.Info("worker: meeting scheduling off (MEETING_CALENDAR=off or no Google token)")
		return nil, nil
	case "fake":
		slog.Warn("worker: MEETING_CALENDAR=fake: meetings are booked in an in-memory calendar (demo only)")
		cal = &scheduler.FakeCalendar{}
	case "google":
		creds := os.Getenv("GOOGLE_CREDENTIALS_FILE")
		if creds == "" {
			creds = "credentials.json"
		}
		calID := os.Getenv("GOOGLE_CALENDAR_ID")
		if calID == "" {
			calID = "primary"
		}
		g, err := scheduler.NewGoogleCalendar(ctx, creds, tokenFile, calID)
		if err != nil {
			return nil, err
		}
		cal = g
	default:
		return nil, fmt.Errorf("MEETING_CALENDAR=%q: want google, fake or off", mode)
	}
	ml := newML(60 * time.Second) // an LLM call
	return &pipeline.MeetingScheduler{
		Store:     store.Meetings{Pool: pool},
		ML:        ml,
		Scheduler: &scheduler.Scheduler{Calendar: cal, Config: scheduler.DefaultConfig(loc)},
		Timezone:  tz,
		Audit:     store.Audit{Pool: pool},
	}, nil
}
