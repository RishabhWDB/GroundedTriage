// Command api serves the Gin REST API and the agent review UI (Layer 5).
//
// GET /health (public; also reports ml-service health) and the REST API of
// internal/api (P6.1, API key required: API_KEY), plus the agent review UI
// at /ui/ (P6.2; the browser asks for the same key).
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/RishabhWDB/GroundedTriage/internal/api"
	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
	"github.com/RishabhWDB/GroundedTriage/internal/preprocess"
	"github.com/RishabhWDB/GroundedTriage/internal/store"
	"github.com/RishabhWDB/GroundedTriage/internal/ui"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func newRouter(ml *mlclient.Client, srv *api.Server) *gin.Engine {
	r := gin.New()
	r.Use(gin.Recovery())
	if srv != nil {
		srv.Register(r)
		ui.Register(r)
	}
	r.GET("/health", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()
		resp := gin.H{"status": "ok", "service": "api"}
		if h, err := ml.Health(ctx); err != nil {
			resp["ml_service"] = gin.H{"status": "unreachable", "error": err.Error()}
		} else {
			resp["ml_service"] = h
		}
		c.JSON(http.StatusOK, resp)
	})
	return r
}

func main() {
	if err := envfile.Load(); err != nil {
		log.Fatalf("load .env: %v", err)
	}
	gin.SetMode(getenv("GIN_MODE", gin.ReleaseMode))

	ml := mlclient.New(getenv("ML_SERVICE_URL", "http://127.0.0.1:8001"))
	ml.MaxRetries = 0 // health probes should answer fast, not retry
	key := os.Getenv("API_KEY")
	if key == "" || key == "change-me" {
		log.Fatal("API_KEY is unset or still the .env.example placeholder; set a secret value")
	}
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("DATABASE_URL is unset")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	pool, err := store.OpenPool(ctx, dbURL, store.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()
	apiSrv := &api.Server{
		Store:     store.Review{Pool: pool},
		Daily:     store.Monitoring{Pool: pool},
		Ingest:    api.FileIngester{Ingestor: &preprocess.Ingestor{Store: store.Tickets{Pool: pool}}},
		ML:        ml,
		Audit:     store.Audit{Pool: pool},
		APIKey:    key,
		UploadDir: getenv("UPLOAD_DIR", "datasets/out/uploads"),
	}
	addr := "127.0.0.1:" + getenv("API_PORT", "8080")
	srv := &http.Server{Addr: addr, Handler: newRouter(ml, apiSrv), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		log.Printf("api listening on http://%s", addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
	log.Println("api stopped")
}
