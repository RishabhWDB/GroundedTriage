#!/usr/bin/env bash
# make demo (P8.1): load the demo tickets, run the pipeline once, then start
# api + worker + ml-service so the review UI opens with tickets and drafts.
#
#   1. go run ./cmd/demo load        (additive; only demo-* rows)
#   2. start the ml-service unless one already answers on ML_PORT
#   3. go run ./cmd/worker -once     (classify, route, schedule, draft ALL pending
#                                      tickets in DATABASE_URL, the 21 sample rows included)
#   4. make demo-status              (expected vs actual)
#   5. stop the ml-service we started; exec make dev (Ctrl-C stops everything)
#
# DEMO_NO_DEV=1 skips step 5 (used to just prepare the data).
set -euo pipefail
cd "$(dirname "$0")/.."
port="${ML_PORT:-8001}"
url="http://127.0.0.1:${port}"

go run ./cmd/demo load

started=""
if curl -fsS "${url}/health" >/dev/null 2>&1; then
  echo "demo: using the ml-service already running on ${url}"
else
  echo "demo: starting the ml-service on ${url}"
  (cd ml-service && exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${port}") >/tmp/groundedtriage-demo-ml.log 2>&1 &
  started=$!
  trap '[ -n "${started}" ] && kill "${started}" 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -fsS "${url}/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS "${url}/health" >/dev/null || { echo "demo: ml-service did not start (see /tmp/groundedtriage-demo-ml.log)"; exit 1; }
fi

# MEETING_CALENDAR=fake books meetings in memory when no Google token exists.
MEETING_CALENDAR="${MEETING_CALENDAR:-fake}" ML_SERVICE_URL="${url}" go run ./cmd/worker -once \
  || echo "demo: a worker step reported an error (see above); continuing so you can inspect the UI"
go run ./cmd/demo status

if [ -n "${started}" ]; then
  kill "${started}" 2>/dev/null || true
  wait "${started}" 2>/dev/null || true
  started=""
fi
if [ "${DEMO_NO_DEV:-}" = "1" ]; then
  exit 0
fi
echo "demo: starting api + worker + ml-service; open http://127.0.0.1:${API_PORT:-8080}/"
exec make dev
