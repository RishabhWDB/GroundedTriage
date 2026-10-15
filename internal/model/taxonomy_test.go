package model

import (
	"os"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestCategoriesMatchTaxonomyYAML(t *testing.T) {
	data, err := os.ReadFile("../../config/taxonomy.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var tax struct {
		Categories []struct {
			Name string `yaml:"name"`
		} `yaml:"categories"`
		Urgency struct {
			Scale []string `yaml:"scale"`
		} `yaml:"urgency"`
	}
	if err := yaml.Unmarshal(data, &tax); err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, c := range tax.Categories {
		names = append(names, c.Name)
	}
	if !reflect.DeepEqual(names, Categories) {
		t.Fatalf("taxonomy.yaml categories %v != model.Categories %v", names, Categories)
	}
	if !reflect.DeepEqual(tax.Urgency.Scale, UrgencyScale) {
		t.Fatalf("taxonomy.yaml urgency %v != model.UrgencyScale %v", tax.Urgency.Scale, UrgencyScale)
	}
	if !IsCategory("Billing") || IsCategory("Complaint") {
		t.Fatal("IsCategory")
	}
}
