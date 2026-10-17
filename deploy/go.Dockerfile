# Optional containerised demo (P8.1): api or worker. Nothing in the repo
# depends on Docker; the native path (make setup / make dev) is primary.
FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /out/api ./cmd/api && CGO_ENABLED=0 go build -o /out/worker ./cmd/worker \
 && CGO_ENABLED=0 go build -o /out/demo ./cmd/demo && CGO_ENABLED=0 go build -o /out/migrate ./cmd/migrate

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=build /out/ /app/bin/
COPY config/ /app/config/
COPY datasets/sample_emails.csv /app/datasets/sample_emails.csv
COPY datasets/demo/ /app/datasets/demo/
ENTRYPOINT ["/app/bin/api"]
