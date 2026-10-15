package ingest

// DefaultRegistry returns a registry with every built-in adapter registered.
func DefaultRegistry() *Registry {
	r := NewRegistry()
	must := func(err error) {
		if err != nil {
			panic(err)
		}
	}
	must(r.Register("csv", NewCSVFactory))
	must(r.Register("email", NewEmailFactory))
	return r
}
