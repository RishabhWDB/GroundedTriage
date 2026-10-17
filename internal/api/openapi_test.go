package api

import (
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Every registered route is documented in docs/openapi.yaml.
func TestOpenAPIDocumentsEveryRoute(t *testing.T) {
	data, err := os.ReadFile("../../docs/openapi.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Paths map[string]map[string]any `yaml:"paths"`
	}
	if err := yaml.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	r, _, _, _ := newServer(t)
	routes := r.Routes()
	if len(routes) < 8 {
		t.Fatalf("only %d routes registered", len(routes))
	}
	for _, rt := range routes {
		p := strings.ReplaceAll(rt.Path, ":id", "{id}")
		ops, ok := doc.Paths[p]
		if !ok {
			t.Errorf("%s %s missing from docs/openapi.yaml", rt.Method, rt.Path)
			continue
		}
		if _, ok := ops[strings.ToLower(rt.Method)]; !ok {
			t.Errorf("%s %s: method not documented", rt.Method, rt.Path)
		}
	}
}
