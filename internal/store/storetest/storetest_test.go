package storetest

import (
	"regexp"
	"testing"
)

func TestSchemaNameIsUniqueAndSafe(t *testing.T) {
	re := regexp.MustCompile(`^test_\d{8}t\d{6}_[0-9a-f]{8}$`)
	a, b := SchemaName(), SchemaName()
	if !re.MatchString(a) || a == b {
		t.Fatalf("schema names %q, %q", a, b)
	}
}
