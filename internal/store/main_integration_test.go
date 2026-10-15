//go:build integration

package store_test

import (
	"os"
	"testing"

	"github.com/RishabhWDB/GroundedTriage/internal/store/storetest"
)

var testDB *storetest.DB

func TestMain(m *testing.M) { os.Exit(storetest.Run(m, &testDB)) }
