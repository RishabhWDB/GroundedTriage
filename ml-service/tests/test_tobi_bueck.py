import pandas as pd

from data.mapping import DROP, apply_category_rules
from data.textclean import clean_body, clean_subject
from data.tobi_bueck import build_frame


def raw(rows):
    cols = ["subject", "body", "type", "queue", "priority", "language"]
    return pd.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])


def test_rule_order_first_match_wins():
    df = pd.DataFrame(
        {
            "type": ["Change", "Incident", "Change", "Incident", "Request", "Problem"],
            "queue": [
                "Billing and Payments",
                "Billing and Payments",
                "Human Resources",
                "Human Resources",
                "Customer Service",
                "Unknown Queue",
            ],
        }
    )
    rules = [
        {"when": {"type": ["Change"]}, "category": "Feature Request"},
        {"when": {"queue": ["Billing and Payments"]}, "category": "Billing"},
        {"when": {"queue": ["Customer Service"]}, "category": "General Inquiry"},
        {"when": {"queue": ["Human Resources"]}, "drop": True},
    ]
    got = apply_category_rules(df, rules).tolist()
    assert got[:5] == ["Feature Request", "Billing", "Feature Request", DROP, "General Inquiry"]
    assert pd.isna(got[5])


def test_build_frame_maps_filters_cleans_and_dedupes():
    df = raw(
        [
            (
                "Refund",
                "I was charged twice.<br>Please refund.",
                "Request",
                "Billing and Payments",
                "high",
                "en",
            ),
            ("Crash", "App crashes on start", "Incident", "Technical Support", "very_low", "en"),
            (
                "Crash again",
                "app  crashes on START",
                "Incident",
                "IT Support",
                "medium",
                "en",
            ),  # dup body
            ("Add SSO", "Please add SSO support", "Change", "Human Resources", "low", "en"),
            ("HR", "Payroll question", "Request", "Human Resources", "low", "en"),
            ("Hallo", "Rechnung falsch", "Request", "Billing and Payments", "critical", "de"),
            (None, "Pricing for enterprise?", "Request", "Sales and Pre-Sales", "medium", "en"),
            ("Empty", "   ", "Request", "General Inquiry", "low", "en"),
        ]
    )
    out, st = build_frame(df)
    assert st.rows_total == 8 and st.rows_english == 7
    assert st.dropped_by_rule == 1  # HR non-Change row
    assert st.empty_body == 1 and st.duplicates_removed == 1
    assert list(out["category"]) == [
        "Billing",
        "Technical Issue",
        "Feature Request",
        "General Inquiry",
    ]
    assert list(out["urgency"]) == ["high", "low", "low", "medium"]  # very_low -> low
    assert out.loc[0, "body"] == "I was charged twice.\nPlease refund."
    assert out.loc[3, "subject"] == ""
    assert set(out["origin"]) == {"real"} and out["id"].is_unique


def test_text_cleaning():
    assert clean_body("<p>Hello&nbsp;there</p><p>Line\\nTwo</p>") == "Hello there\n\nLine\nTwo"
    assert clean_body("a\u200bb   c\r\n\r\n\r\n\r\nd") == "ab c\n\nd"
    assert clean_body(float("nan")) == "" and clean_body(None) == ""
    assert clean_subject(float("nan")) == "" and clean_subject("  Re:  Hi ") == "Re: Hi"
