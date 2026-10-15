"""Loads config/taxonomy.yaml: the six categories, urgency scale and dataset mappings."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "config" / "taxonomy.yaml"


@lru_cache
def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def categories() -> list[str]:
    return [c["name"] for c in load_taxonomy()["categories"]]


def category_descriptions() -> dict[str, str]:
    """Natural-language descriptions (used by the SCRIBE zero-shot baseline)."""
    return {c["name"]: " ".join(c["description"].split()) for c in load_taxonomy()["categories"]}


def urgency_scale() -> list[str]:
    return list(load_taxonomy()["urgency"]["scale"])


def trained_urgency_levels() -> list[str]:
    return list(load_taxonomy()["urgency"]["trained_levels"])


def mapping(dataset: str) -> dict[str, Any]:
    return load_taxonomy()["mappings"][dataset]
