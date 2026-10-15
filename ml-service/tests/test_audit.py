import pandas as pd

from data.audit import (
    exact_duplicate_pairs,
    leakage_table,
    near_duplicate_pairs,
    subject_openings,
)


def test_exact_and_near_duplicates():
    texts = [
        "please reschedule our onboarding call to friday",
        "Please  reschedule our onboarding call to Friday",  # exact after normalising
        "our invoice shows a double charge for october",
        "the export job fails with a timeout error every night",
    ]
    assert exact_duplicate_pairs(texts) == 1
    pairs = near_duplicate_pairs(texts)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_subject_openings():
    got = subject_openings(["Request to meet", "request to: reschedule", "Demo time"], top=2)
    assert got == [("request to", 2), ("demo time", 1)]


def test_leakage_table_per_stratum():
    pool = pd.DataFrame(
        {
            "text": ["the invoice shows a double charge", "login fails after the update"],
            "category": ["Billing", "Technical Issue"],
        }
    )
    test = pd.DataFrame(
        {
            "text": ["The invoice shows a double charge!", "how do i export reports to csv"],
            "category": ["Billing", "General Inquiry"],
            "origin": ["real", "real"],
        }
    )
    out = leakage_table(pool, test)
    by_cat = {r["category"]: r for r in out["per_stratum"]}
    assert by_cat["Billing"]["test_rows_with_near_dup"] == 1
    assert by_cat["Billing"]["nearest_same_category"] == 1
    assert by_cat["General Inquiry"]["test_rows_with_near_dup"] == 0
    assert out["total_pairs"] == 1
