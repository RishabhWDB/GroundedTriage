// Command worker polls enabled ingestion sources (config/sources.yaml),
// stores new tickets (layer 1), classifies accepted tickets through the
// ml-service's POST /classify (layer 2, ML_SERVICE_URL), routes classified and
// skipped tickets with config/routing_rules.yaml + config/sla.yaml and
// escalates SLA breaches (layer 3). Replies are added in later phases.
//
//	go run ./cmd/worker            # poll every WORKER_POLL_INTERVAL (default 60s)
//	go run ./cmd/worker -once      # one poll, then exit
package main

import (
	"context"
	"flag"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/pipeline"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/routing"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func main() {
	once := flag.Bool("once", false, "poll every source once and exit")
	sourcesPath := flag.String("sources", "config/sources.yaml", "sources config file")
	rulesPath := flag.String("rules", "config/routing_rules.yaml", "routing rules (hot-reloaded)")
	slaPath := flag.String("sla", "config/sla.yaml", "SLA targets")
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
	classifier := &pipeline.Classifier{
		Store: store.Classifications{Pool: pool},
		ML:    mlclient.New(mlURL),
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
	// After each poll: classify, route, escalate. A failing step is logged and
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
		if failed || err != nil || rerr != nil || eerr != nil {
			os.Exit(1)
		}
		return
	}
	go engine.Watch(ctx, 5*time.Second) // hot reload of the routing rules
	slog.Info("worker: polling", "sources", len(sources), "interval", interval)
	poller.Run(ctx, interval)
	slog.Info("worker stopped")
}
