// Command api serves the Gin REST API and the agent review UI (Layer 5).
//
// P0.3: only GET /health, which also reports ml-service health via mlclient.
// The full API and UI arrive in P6.x.
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

	"github.com/RishabhWDB/GroundedTriage/internal/envfile"
	"github.com/RishabhWDB/GroundedTriage/internal/mlclient"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func newRouter(ml *mlclient.Client) *gin.Engine {
	r := gin.New()
	r.Use(gin.Recovery())
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
	addr := "127.0.0.1:" + getenv("API_PORT", "8080")
	srv := &http.Server{Addr: addr, Handler: newRouter(ml), ReadHeaderTimeout: 10 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
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
