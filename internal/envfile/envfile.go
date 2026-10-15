// Package envfile loads the repo-root .env into the process environment.
// make never reads .env (see CLAUDE.md decisions log); programs call Load.
package envfile

import (
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
)

// Load finds the nearest .env walking up from the working directory (stopping
// at the directory containing go.mod) and loads it without overriding
// variables already set in the environment. A missing .env is not an error.
func Load() error {
	dir, err := os.Getwd()
	if err != nil {
		return err
	}
	for {
		candidate := filepath.Join(dir, ".env")
		if _, err := os.Stat(candidate); err == nil {
			return godotenv.Load(candidate)
		}
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return nil
		}
		dir = parent
	}
}
