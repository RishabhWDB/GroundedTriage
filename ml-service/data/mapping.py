"""Applies the ordered label-mapping rules from config/taxonomy.yaml."""

from __future__ import annotations

from typing import Any

import pandas as pd

DROP = "__drop__"


def apply_category_rules(df: pd.DataFrame, rules: list[dict[str, Any]]) -> pd.Series:
    """Return the mapped category per row: first matching rule wins; rows
    matched by a drop rule get DROP; unmatched rows get NA."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    unassigned = pd.Series(True, index=df.index)
    for rule in rules:
        mask = unassigned.copy()
        for column, values in rule["when"].items():
            mask &= df[column].isin(values)
        out[mask] = DROP if rule.get("drop") else rule["category"]
        unassigned &= ~mask
    return out


def map_urgency(priority: pd.Series, urgency_map: dict[str, str]) -> pd.Series:
    return priority.map(urgency_map)
