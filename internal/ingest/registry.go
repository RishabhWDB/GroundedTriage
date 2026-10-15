package ingest

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"sort"

	"gopkg.in/yaml.v3"
)

// SourceConfig is one entry of config/sources.yaml. Options hold non-secret
// settings (paths, mailbox names); secrets come from env vars read by the
// adapter's factory.
type SourceConfig struct {
	Type    string            `yaml:"type"`
	Enabled bool              `yaml:"enabled"`
	Options map[string]string `yaml:"options"`
}

// Factory builds a Source from its config.
type Factory func(cfg SourceConfig) (Source, error)

// Registry maps source types to factories.
type Registry struct {
	factories map[string]Factory
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return &Registry{factories: map[string]Factory{}} }

// Register adds a factory for typ. Registering a type twice is an error.
func (r *Registry) Register(typ string, f Factory) error {
	if typ == "" || f == nil {
		return errors.New("ingest: register needs a type and a factory")
	}
	if _, ok := r.factories[typ]; ok {
		return fmt.Errorf("ingest: source type %q already registered", typ)
	}
	r.factories[typ] = f
	return nil
}

// Types returns the registered source types, sorted.
func (r *Registry) Types() []string {
	out := make([]string, 0, len(r.factories))
	for t := range r.factories {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// Sources is the set of built, enabled sources keyed by type.
type Sources map[string]Source

// Build constructs every enabled source. Disabled entries are skipped without
// needing a registered factory. At most one enabled source per type is allowed
// so Ticket.Source identifies the adapter to reply through.
func (r *Registry) Build(cfgs []SourceConfig) (Sources, error) {
	out := Sources{}
	for _, c := range cfgs {
		if !c.Enabled {
			continue
		}
		f, ok := r.factories[c.Type]
		if !ok {
			return nil, fmt.Errorf("ingest: unknown source type %q (registered: %v)", c.Type, r.Types())
		}
		if _, dup := out[c.Type]; dup {
			return nil, fmt.Errorf("ingest: source type %q enabled more than once", c.Type)
		}
		s, err := f(c)
		if err != nil {
			return nil, fmt.Errorf("ingest: build %q: %w", c.Type, err)
		}
		if s.Name() != c.Type {
			return nil, fmt.Errorf("ingest: factory for %q built a source named %q", c.Type, s.Name())
		}
		out[c.Type] = s
	}
	return out, nil
}

type configFile struct {
	Sources []SourceConfig `yaml:"sources"`
}

// LoadConfig reads a sources YAML file (see config/sources.yaml).
func LoadConfig(path string) ([]SourceConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cf configFile
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&cf); err != nil {
		return nil, fmt.Errorf("ingest: parse %s: %w", path, err)
	}
	for i, c := range cf.Sources {
		if c.Type == "" {
			return nil, fmt.Errorf("ingest: %s: sources[%d] has no type", path, i)
		}
	}
	return cf.Sources, nil
}
