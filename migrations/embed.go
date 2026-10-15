// Package migrations embeds the goose SQL migrations so cmd/migrate and the
// integration-test helper apply exactly the same files.
package migrations

import "embed"

//go:embed *.sql
var FS embed.FS
