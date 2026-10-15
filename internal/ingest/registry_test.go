package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RishabhWDB/GroundedTriage/internal/model"
)

var _ Source = (*FakeSource)(nil)

func fakeFactory(c SourceConfig) (Source, error) {
	if c.Options["fail"] == "yes" {
		return nil, errors.New("boom")
	}
	return NewFakeSource(c.Type), nil
}

func newReg(t *testing.T, types ...string) *Registry {
	t.Helper()
	r := NewRegistry()
	for _, typ := range types {
		if err := r.Register(typ, fakeFactory); err != nil {
			t.Fatal(err)
		}
	}
	return r
}

func TestRegisterRejectsDuplicatesAndEmpty(t *testing.T) {
	r := newReg(t, "csv")
	if err := r.Register("csv", fakeFactory); err == nil {
		t.Fatal("duplicate register succeeded")
	}
	if err := r.Register("", fakeFactory); err == nil {
		t.Fatal("empty type register succeeded")
	}
}

func TestBuild(t *testing.T) {
	cases := []struct {
		name    string
		cfgs    []SourceConfig
		want    []string
		wantErr string
	}{
		{"enabled only", []SourceConfig{{Type: "csv", Enabled: true}, {Type: "email"}}, []string{"csv"}, ""},
		{"disabled unknown type is fine", []SourceConfig{{Type: "nope"}}, nil, ""},
		{"enabled unknown type", []SourceConfig{{Type: "nope", Enabled: true}}, nil, "unknown source type"},
		{"duplicate type", []SourceConfig{{Type: "csv", Enabled: true}, {Type: "csv", Enabled: true}}, nil, "more than once"},
		{"factory error", []SourceConfig{{Type: "csv", Enabled: true, Options: map[string]string{"fail": "yes"}}}, nil, "boom"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := newReg(t, "csv", "email").Build(tc.cfgs)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("built %d sources, want %v", len(got), tc.want)
			}
			for _, name := range tc.want {
				if got[name] == nil || got[name].Name() != name {
					t.Fatalf("missing source %q", name)
				}
			}
		})
	}
}

func TestBuildRejectsMisnamedSource(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("csv", func(SourceConfig) (Source, error) { return NewFakeSource("email"), nil })
	if _, err := r.Build([]SourceConfig{{Type: "csv", Enabled: true}}); err == nil {
		t.Fatal("misnamed source accepted")
	}
}

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	write := func(body string) string {
		p := filepath.Join(dir, "s.yaml")
		if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	cfgs, err := LoadConfig(write("sources:\n  - type: csv\n    enabled: true\n    options: {path: a.csv}\n  - type: email\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfgs) != 2 || !cfgs[0].Enabled || cfgs[0].Options["path"] != "a.csv" || cfgs[1].Enabled {
		t.Fatalf("cfgs = %+v", cfgs)
	}
	if _, err := LoadConfig(write("sources:\n  - enabled: true\n")); err == nil {
		t.Fatal("missing type accepted")
	}
	if _, err := LoadConfig(write("sources:\n  - type: csv\n    enabeld: true\n")); err == nil {
		t.Fatal("unknown field (typo) accepted")
	}
}

func TestRepoSourcesConfigParses(t *testing.T) {
	cfgs, err := LoadConfig("../../config/sources.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(cfgs) != 3 {
		t.Fatalf("got %d sources, want csv/email/zendesk", len(cfgs))
	}
}

func TestFakeSource(t *testing.T) {
	t0 := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	f := NewFakeSource("csv",
		model.Ticket{ExternalID: "old", ReceivedAt: t0},
		model.Ticket{ExternalID: "new", ReceivedAt: t0.Add(time.Hour)},
	)
	ctx := context.Background()
	got, err := f.Fetch(ctx, t0)
	if err != nil || len(got) != 1 || got[0].ExternalID != "new" {
		t.Fatalf("Fetch since t0 = %v, %v", got, err)
	}
	if err := f.Reply(ctx, got[0], model.ReplyDraft{Body: "hi"}); err != nil {
		t.Fatal(err)
	}
	if r := f.Replies(); len(r) != 1 || r[0].Draft.Body != "hi" {
		t.Fatalf("replies = %+v", r)
	}
	f.ReplyErr = errors.New("smtp down")
	if err := f.Reply(ctx, got[0], model.ReplyDraft{}); err == nil || len(f.Replies()) != 1 {
		t.Fatal("ReplyErr not returned or reply recorded")
	}
}
