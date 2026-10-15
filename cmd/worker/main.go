// Command worker polls enabled ingestion sources and runs the per-ticket
// pipeline (layers 1–4).
//
// Stub until P1.6: it idles until SIGINT/SIGTERM so `make dev` (honcho) keeps
// the other processes running — honcho stops everything when one exits.
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Println("worker: idle stub (ingestion loop arrives in P1.6); waiting for shutdown signal")
	<-ctx.Done()
	log.Println("worker stopped")
}
