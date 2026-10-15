-- +goose Up
-- On Supabase pgvector is already installed in the "extensions" schema, so this
-- is a no-op there; on plain Postgres it installs into public. Either way the
-- per-test schemas resolve it via search_path "<test>, public, extensions".
create extension if not exists vector with schema public;

-- +goose Down
-- Intentionally empty: never drop the extension (shared by every schema).
