package api

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
)

const defaultMaxUpload = 5 << 20 // 5 MiB

// ingestCSV handles POST /ingest/csv: a multipart upload (field "file") of a
// SCRIBE-format CSV (Sender, Subject, Body[, Label, Date, Id]). The file is
// stored under UploadDir with a generated name (the client's name is never
// used as a path) and ingested like the CSV source.
func (s *Server) ingestCSV(c *gin.Context) {
	limit := s.MaxUpload
	if limit <= 0 {
		limit = defaultMaxUpload
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, limit+1<<16)
	fh, err := c.FormFile("file")
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			fail(c, http.StatusRequestEntityTooLarge, CodeTooLarge, "upload exceeds the size limit")
			return
		}
		fail(c, http.StatusBadRequest, CodeBadRequest, `expected a multipart form with a "file" field`)
		return
	}
	if fh.Size > limit {
		fail(c, http.StatusRequestEntityTooLarge, CodeTooLarge, "upload exceeds the size limit")
		return
	}
	if !strings.EqualFold(filepath.Ext(fh.Filename), ".csv") {
		fail(c, http.StatusBadRequest, CodeBadRequest, "only .csv files are accepted")
		return
	}
	src, err := fh.Open()
	if err != nil {
		fail(c, http.StatusBadRequest, CodeBadRequest, "cannot read the upload")
		return
	}
	defer src.Close()
	if err := os.MkdirAll(s.UploadDir, 0o755); err != nil {
		s.storeErr(c, err, "upload directory")
		return
	}
	suffix := make([]byte, 6)
	_, _ = rand.Read(suffix)
	path := filepath.Join(s.UploadDir, s.now().UTC().Format("20060102T150405Z")+"-"+hex.EncodeToString(suffix)+".csv")
	dst, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		s.storeErr(c, err, "upload file")
		return
	}
	if _, err := io.Copy(dst, src); err != nil {
		dst.Close()
		s.storeErr(c, err, "upload file")
		return
	}
	if err := dst.Close(); err != nil {
		s.storeErr(c, err, "upload file")
		return
	}
	fetched, stats, err := s.Ingest.IngestCSV(c.Request.Context(), path)
	if err != nil {
		fail(c, http.StatusBadRequest, CodeBadRequest, "cannot ingest the CSV: "+err.Error())
		return
	}
	c.JSON(http.StatusOK, gin.H{"file": filepath.Base(path), "fetched": fetched, "inserted": stats.Inserted,
		"inserted_skipped": stats.InsertedSkipped, "duplicate_external_id": stats.DuplicateID,
		"duplicate_content": stats.DuplicateContent, "dropped": stats.Dropped})
}
