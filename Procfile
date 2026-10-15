api: go run ./cmd/api
worker: go run ./cmd/worker
ml: cd ml-service && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port ${ML_PORT:-8001}
