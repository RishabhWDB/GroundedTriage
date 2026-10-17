from app import taxonomy
from app.schemas import URGENCY_SCALE


def test_six_categories_in_canonical_order():
    assert taxonomy.categories() == [
        "Technical Issue",
        "Billing",
        "Feature Request",
        "General Inquiry",
        "Spam",
        "Meeting Scheduling",
    ]
    descs = taxonomy.category_descriptions()
    assert set(descs) == set(taxonomy.categories())
    assert all(len(d) > 20 and "\n" not in d for d in descs.values())


def test_urgency_scale_matches_contract():
    assert tuple(taxonomy.urgency_scale()) == URGENCY_SCALE
    assert taxonomy.trained_urgency_levels() == ["low", "medium", "high"]


def test_mapping_targets_are_known_categories():
    cats = set(taxonomy.categories())
    tb = taxonomy.mapping("tobi_bueck")
    for rule in tb["category_rules"]:
        assert rule.get("drop") or rule["category"] in cats
    assert set(tb["urgency_map"].values()) <= set(taxonomy.urgency_scale())
    assert taxonomy.mapping("spam")["category"] in cats
    assert taxonomy.mapping("synthetic_meetings")["category"] in cats


def test_topic_and_filter_categories():
    from app.taxonomy import categories, filter_categories, topic_categories

    assert filter_categories() == ["Spam"]
    assert topic_categories() == [c for c in categories() if c != "Spam"]
    assert len(topic_categories()) == 5
