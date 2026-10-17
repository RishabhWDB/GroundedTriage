"""Fixed WixQA question splits (Phase 5 review), defined before P7.2 runs.

- demo: DEMO_N expert-written questions reserved for the P8 demo ticket CSV
  (rewritten as support emails). **P7.2 never evaluates on them.**
- p72_dev / p72_test: the remaining questions (expert-written and simulated),
  P72_DEV_FRACTION / rest. Choices such as the floor sweep (85/90/95/98%)
  use p72_dev; p72_test is reported once and never moves the floor.

Order = sha256(SEED + "|" + kind + "|" + question): deterministic and
independent of file order. A question's split never changes.
"""

from __future__ import annotations

import hashlib

from rag.kb import QAPair, load_wixqa_qa

SEED = "wixqa-splits-v1"
DEMO_N = 20
P72_DEV_FRACTION = 0.4
KINDS = ("expertwritten", "simulated")


def question_id(kind: str, question: str) -> str:
    return hashlib.sha256(f"{kind}|{question}".encode()).hexdigest()[:16]


def _rank(kind: str, question: str) -> str:
    return hashlib.sha256(f"{SEED}|{kind}|{question}".encode()).hexdigest()


def assign(pairs: list[QAPair]) -> dict[str, str]:
    """question_id -> demo | p72_dev | p72_test."""
    out: dict[str, str] = {}
    expert = sorted(
        (p for p in pairs if p.kind == "expertwritten"), key=lambda p: _rank(p.kind, p.question)
    )
    for p in expert[:DEMO_N]:
        out[question_id(p.kind, p.question)] = "demo"
    for kind in KINDS:
        rest = sorted(
            (p for p in pairs if p.kind == kind and question_id(p.kind, p.question) not in out),
            key=lambda p: _rank(p.kind, p.question),
        )
        n_dev = int(round(len(rest) * P72_DEV_FRACTION))
        for i, p in enumerate(rest):
            out[question_id(p.kind, p.question)] = "p72_dev" if i < n_dev else "p72_test"
    return out


def load_split(split: str) -> list[QAPair]:
    pairs = [p for kind in KINDS for p in load_wixqa_qa(kind)]
    a = assign(pairs)
    return [p for p in pairs if a[question_id(p.kind, p.question)] == split]
