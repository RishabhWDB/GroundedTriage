// Command migrate applies the goose migrations in migrations/ to DATABASE_URL.
//
//	go run ./cmd/migrate up      # apply pending migrations (idempotent)
//	go run ./cmd/migrate status  # list applied/pending migrations
//
// Down-migrations are deliberately not exposed.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

func main() {
	log.SetFlags(0)
	if err := envfile.Load(); err != nil {
		log.Fatalf("load .env: %v", err)
	}
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		log.Fatal("DATABASE_URL is unset (copy .env.example to .env)")
	}
	cmd := "up"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	db, err := store.OpenSQL(url, store.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	if err := db.PingContext(ctx); err != nil {
		log.Fatalf("connect: %v\n(if this is Supabase, check the project isn't paused and that you used the Session pooler string)", err)
	}

	switch cmd {
	case "up":
		n, err := store.MigrateUp(ctx, db)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("migrate: %d migration(s) applied\n", n)
	case "status":
		lines, err := store.MigrationStatus(ctx, db)
		if err != nil {
			log.Fatal(err)
		}
		for _, l := range lines {
			fmt.Println(l)
		}
	default:
		log.Fatalf("unknown command %q (use up|status)", cmd)
	}
}
