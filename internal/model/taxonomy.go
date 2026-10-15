package model

// Categories mirrors config/taxonomy.yaml (canonical order). A test keeps the
// two in sync; change the YAML first.
const (
	CategoryTechnicalIssue    = "Technical Issue"
	CategoryBilling           = "Billing"
	CategoryFeatureRequest    = "Feature Request"
	CategoryGeneralInquiry    = "General Inquiry"
	CategorySpam              = "Spam"
	CategoryMeetingScheduling = "Meeting Scheduling"
)

// Categories lists every category in canonical order.
var Categories = []string{
	CategoryTechnicalIssue, CategoryBilling, CategoryFeatureRequest,
	CategoryGeneralInquiry, CategorySpam, CategoryMeetingScheduling,
}

// IsCategory reports whether c is a known category.
func IsCategory(c string) bool {
	for _, k := range Categories {
		if k == c {
			return true
		}
	}
	return false
}
