package model

import "time"

// Urgency scale, ordered low → critical.
var UrgencyScale = []string{"low", "medium", "high", "critical"}

// ModelsLoaded reports which fine-tuned models the ml-service has loaded.
type ModelsLoaded struct {
	DeepTier bool `json:"deep_tier"`
	Urgency  bool `json:"urgency"`
}

// Classification is the output of the classification cascade (Layer 2).
type Classification struct {
	TicketID           string            `json:"ticket_id"`
	Category           string            `json:"category"`
	CategoryConfidence float64           `json:"category_confidence"`
	Tier               string            `json:"tier"` // fast | deep
	LowConfidence      bool              `json:"low_confidence"`
	Urgency            string            `json:"urgency"` // low | medium | high | critical
	UrgencyConfidence  float64           `json:"urgency_confidence"`
	UrgencySource      string            `json:"urgency_source"` // model | heuristic
	ModelsLoaded       ModelsLoaded      `json:"models_loaded"`
	ModelVersions      map[string]string `json:"model_versions"`
}

// Route is the routing decision for a ticket (Layer 3).
type Route struct {
	TicketID string     `json:"ticket_id"`
	Action   string     `json:"action"` // mute | queue | reply | schedule
	Queue    string     `json:"queue"`
	Priority int        `json:"priority"`
	SLADueAt *time.Time `json:"sla_due_at,omitempty"`
	Reason   string     `json:"reason"`
}

// RetrievedChunk is one KB chunk returned by the retriever (Layer 4).
type RetrievedChunk struct {
	ChunkID string  `json:"chunk_id"`
	DocID   string  `json:"doc_id"`
	Title   string  `json:"title"`
	Text    string  `json:"text"`
	Score   float64 `json:"score"`
}

// ReplyDraft is a generated (and verified) reply (Layer 4).
type ReplyDraft struct {
	TicketID          string   `json:"ticket_id"`
	Subject           string   `json:"subject"`
	Body              string   `json:"body"`
	Citations         []string `json:"citations"`
	FaithfulnessScore *float64 `json:"faithfulness_score,omitempty"`
	UnsupportedClaims []string `json:"unsupported_claims"`
	Status            string   `json:"status"` // grounded | abstained | approved | sent | rejected
}
