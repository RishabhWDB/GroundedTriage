// Command demo loads, inspects and resets the P8.1 demo tickets.
//
//	go run ./cmd/demo load     # ingest datasets/demo/demo_tickets.csv (additive)
//	go run ./cmd/demo status   # expected vs actual per demo ticket
//	go run ./cmd/demo reset    # delete ONLY demo-tagged tickets, after typed confirmation
//
// Demo tickets are CSV rows whose Id starts with "demo-". Nothing else in
// the database is read for changes or deleted. -env picks the connection
// variable (DATABASE_URL by default; TEST_DATABASE_URL for a dry run).
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/ingest"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
)

// ConfirmPhrase must be typed to reset (or passed with -yes in scripts).
const ConfirmPhrase = "reset demo"

func main() {
	envVar := flag.String("env", "DATABASE_URL", "environment variable holding the database URL")
	csvPath := flag.String("csv", "datasets/demo/demo_tickets.csv", "demo ticket CSV")
	yes := flag.Bool("yes", false, "skip the typed confirmation for reset")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: demo [-env VAR] [-csv path] [-yes] load|status|reset")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}
	if err := envfile.Load(); err != nil {
		log.Fatalf("load .env: %v", err)
	}
	url := os.Getenv(*envVar)
	if url == "" {
		log.Fatalf("%s is unset", *envVar)
	}
	ctx := context.Background()
	pool, err := store.OpenPool(ctx, url, store.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()
	demo := store.Demo{Pool: pool}

	switch flag.Arg(0) {
	case "load":
		tickets, err := (&ingest.CSVSource{Path: *csvPath}).Fetch(ctx, time.Time{})
		if err != nil {
			log.Fatal(err)
		}
		for _, t := range tickets {
			if !strings.HasPrefix(t.ExternalID, store.DemoIDPrefix) {
				log.Fatalf("%s: row %q has no %q Id; refusing to load untagged rows", *csvPath, t.ExternalID, store.DemoIDPrefix)
			}
		}
		_, stats, err := (&preprocess.Ingestor{Store: store.Tickets{Pool: pool}}).Process(ctx, tickets)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("demo load: %d rows, %d inserted (%d skipped as non-English), %d already present, %d duplicate content, %d dropped\n",
			len(tickets), stats.Inserted, stats.InsertedSkipped, stats.DuplicateID, stats.DuplicateContent, stats.Dropped)
		fmt.Println("next: make ml (and make api), then `go run ./cmd/worker -once` to classify, route, schedule and draft; then `make demo-status`")
	case "status":
		expected, err := readExpected(*csvPath)
		if err != nil {
			log.Fatal(err)
		}
		rows, err := demo.Rows(ctx)
		if err != nil {
			log.Fatal(err)
		}
		printStatus(os.Stdout, rows, expected)
	case "reset":
		c, err := demo.Counts(ctx)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("demo reset on %s would delete ONLY demo-tagged rows: %d tickets, %d classifications, %d routes, %d drafts, %d meetings, %d audit rows\n",
			*envVar, c.Tickets, c.Classifications, c.Routes, c.Drafts, c.Meetings, c.Audit)
		if c.Tickets == 0 {
			fmt.Println("nothing to delete")
			return
		}
		if !*yes && !confirm(os.Stdin, os.Stdout) {
			fmt.Println("aborted; nothing changed")
			os.Exit(1)
		}
		if _, err := demo.Reset(ctx); err != nil {
			log.Fatal(err)
		}
		fmt.Println("demo rows deleted")
	default:
		flag.Usage()
		os.Exit(2)
	}
}

func confirm(in io.Reader, out io.Writer) bool {
	fmt.Fprintf(out, "type %q to continue: ", ConfirmPhrase)
	line, _ := bufio.NewReader(in).ReadString('\n')
	return strings.TrimSpace(line) == ConfirmPhrase
}

// expectation is one row's Label / Expected_Outcome from the demo CSV.
type expectation struct{ Label, Outcome string }

func readExpected(path string) (map[string]expectation, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil || len(recs) == 0 {
		return nil, fmt.Errorf("read %s: %v", path, err)
	}
	col := map[string]int{}
	for i, h := range recs[0] {
		col[h] = i
	}
	for _, need := range []string{"Id", "Label", "Expected_Outcome"} {
		if _, ok := col[need]; !ok {
			return nil, fmt.Errorf("%s: missing column %s", path, need)
		}
	}
	out := map[string]expectation{}
	for _, r := range recs[1:] {
		out[r[col["Id"]]] = expectation{Label: r[col["Label"]], Outcome: r[col["Expected_Outcome"]]}
	}
	return out, nil
}

func str(p *string) string {
	if p == nil {
		return "-"
	}
	return *p
}

// actualOutcome condenses the pipeline state into the Expected_Outcome vocabulary.
func actualOutcome(r store.DemoRow) string {
	switch {
	case r.Action == nil:
		return "(not routed yet)"
	case *r.Action == "mute":
		return "mute"
	case *r.Action == "schedule" || r.Meeting != nil:
		return "schedule"
	case r.DraftStatus != nil && *r.DraftStatus == "abstained":
		return "abstain"
	case r.DraftStatus != nil:
		return "ground"
	case *r.Action == "queue":
		return "queue:" + str(r.Queue)
	}
	return *r.Action + " (no draft yet)"
}

func printStatus(w io.Writer, rows []store.DemoRow, expected map[string]expectation) {
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "id\tgold label\tpredicted\ttier\troute\tdraft\tmeeting\texpected\tactual\tmatch")
	match := 0
	for _, r := range rows {
		e := expected[r.ExternalID]
		act := actualOutcome(r)
		ok := act == e.Outcome || (e.Outcome == "draft (gate decides)" && (act == "ground" || act == "abstain"))
		if ok {
			match++
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s/%s\t%s\t%s\t%s\t%s\t%v\n", r.ExternalID, e.Label, str(r.Category), str(r.Tier),
			str(r.Action), str(r.Queue), str(r.DraftStatus), str(r.Meeting), e.Outcome, act, ok)
	}
	_ = tw.Flush()
	fmt.Fprintf(w, "%d demo tickets, %d match the expected outcome\n", len(rows), match)
}
