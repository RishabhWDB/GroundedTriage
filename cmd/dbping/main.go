// Command dbping measures database latency: TCP connect, fresh pgx
// connection, warm-pool round-trips (SELECT 1) and the ingestor's real
// queries, and reports whether the pool reuses connections. It prints the
// Supabase region, never the connection string.
//
//	go run ./cmd/dbping [-n 20] [-env DATABASE_URL]
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/url"
	"os"
	"regexp"
	"sort"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

var regionRe = regexp.MustCompile(`^aws-\d+-([a-z]+-[a-z]+-\d+)\.pooler\.supabase\.com$`)

func stats(ds []time.Duration) string {
	s := append([]time.Duration(nil), ds...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	pct := func(p float64) time.Duration { return s[int(p*float64(len(s)-1))] }
	return fmt.Sprintf("n=%d min=%v p50=%v p95=%v max=%v", len(s),
		s[0].Round(time.Millisecond), pct(0.5).Round(time.Millisecond),
		pct(0.95).Round(time.Millisecond), s[len(s)-1].Round(time.Millisecond))
}

func main() {
	n := flag.Int("n", 20, "samples per measurement")
	envVar := flag.String("env", "DATABASE_URL", "env var holding the connection string")
	flag.Parse()
	_ = envfile.Load()
	dsn := os.Getenv(*envVar)
	if dsn == "" {
		log.Fatalf("%s is unset", *envVar)
	}
	u, err := url.Parse(dsn)
	if err != nil {
		log.Fatal("cannot parse connection string")
	}
	host, port := u.Hostname(), u.Port()
	region := "unknown (not a Supabase pooler host)"
	if m := regionRe.FindStringSubmatch(host); m != nil {
		region = m[1]
	}
	fmt.Printf("host region: %s   port: %s\n", region, port)
	ctx := context.Background()

	var tcp []time.Duration
	for i := 0; i < 5; i++ {
		start := time.Now()
		c, err := net.DialTimeout("tcp", net.JoinHostPort(host, port), 10*time.Second)
		if err != nil {
			log.Fatalf("tcp dial: %v", err)
		}
		tcp = append(tcp, time.Since(start))
		c.Close()
	}
	fmt.Printf("TCP connect (network RTT proxy):   %s\n", stats(tcp))

	var fresh []time.Duration
	for i := 0; i < 5; i++ {
		start := time.Now()
		c, err := pgx.Connect(ctx, dsn)
		if err != nil {
			log.Fatalf("connect: %v", err)
		}
		if _, err := c.Exec(ctx, "select 1"); err != nil {
			log.Fatal(err)
		}
		fresh = append(fresh, time.Since(start))
		c.Close(ctx)
	}
	fmt.Printf("fresh connection + SELECT 1:       %s\n", stats(fresh))

	pool, err := store.OpenPool(ctx, dsn, store.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()
	var warm []time.Duration
	for i := 0; i < *n; i++ {
		start := time.Now()
		var one int
		if err := pool.QueryRow(ctx, "select 1").Scan(&one); err != nil {
			log.Fatal(err)
		}
		warm = append(warm, time.Since(start))
	}
	fmt.Printf("warm pool SELECT 1:                %s\n", stats(warm))

	var hashq []time.Duration
	tickets := store.Tickets{Pool: pool}
	for i := 0; i < *n; i++ {
		start := time.Now()
		if _, err := tickets.ContentHashExists(ctx, fmt.Sprintf("probe-%d", i)); err != nil {
			log.Fatal(err)
		}
		hashq = append(hashq, time.Since(start))
	}
	fmt.Printf("warm pool ContentHashExists:       %s\n", stats(hashq))

	st := pool.Stat()
	fmt.Printf("pool: new connections opened=%d total=%d idle=%d acquires=%d (reuse = acquires >> new connections)\n",
		st.NewConnsCount(), st.TotalConns(), st.IdleConns(), st.AcquireCount())
}
