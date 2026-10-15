# GroundedTriage

An agentic support-triage system with pluggable ingestion and retrieval-grounded reply generation. It extends SCRIBE (Procedia CS 283, 2026) with:
- IMAP, Zendesk and CSV ingestion;
- a learned fast tier plus a fine-tuned DeBERTa-v3 deep tier;
- an ordinal urgency model;
- pgvector-grounded replies with a faithfulness gate and an abstain path;
- an evaluation harness against the SCRIBE baseline.

> Work in progress. See `CLAUDE.md` for the current status board and `BUILDPROMPT.md` for the full spec.

## Quick start (macOS, native)

```bash
cp .env.example .env   # fill in Supabase + Groq values
make setup
make lint
make test
```

Full documentation is written in P8.3.
