// Command calendar-auth runs the Google OAuth2 installed-app flow once and
// saves the token for the meeting scheduler (P4.4).
//
//	go run ./cmd/calendar-auth      # == make calendar-auth
//
// Needs GOOGLE_CREDENTIALS_FILE (an OAuth client of type "Desktop app" from
// Google Cloud Console, with the Calendar API enabled) and writes
// GOOGLE_TOKEN_FILE (default token.json, gitignored, mode 0600).
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/scheduler"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	if err := envfile.Load(); err != nil {
		log.Fatalf("load .env: %v", err)
	}
	creds := env("GOOGLE_CREDENTIALS_FILE", "credentials.json")
	tokenFile := env("GOOGLE_TOKEN_FILE", "token.json")
	cfg, err := scheduler.OAuthConfig(creds)
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	_, err = scheduler.RunInstalledAppFlow(ctx, cfg, tokenFile, func(url string) {
		fmt.Printf("Open this URL in a browser and allow access:\n\n%s\n\nWaiting for Google's redirect...\n", url)
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Saved the token to %s\n", tokenFile)
}
