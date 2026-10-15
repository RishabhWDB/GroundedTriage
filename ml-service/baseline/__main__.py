"""SCRIBE baseline CLI (its own process: BART-MNLI is ~1.6 GB of RAM).

    python -m baseline classify --split test --limit 10
    python -m baseline classify --csv ../datasets/sample_emails.csv --out preds.jsonl
    python -m baseline draft --csv ../datasets/sample_emails.csv --limit 2

Needs the optional extra: pip install -e ".[baseline]" (make setup-baseline).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from app.interfaces import TicketText
from data.build import OUT_DIR

logger = logging.getLogger("baseline")


def load_tickets(
    split: str | None, csv: str | None, limit: int | None
) -> tuple[list[TicketText], list[str | None]]:
    """Tickets plus their gold category (None when the input has no label)."""
    if csv:
        df = pd.read_csv(csv, dtype=str).fillna("")
        df = df.rename(columns=str.lower)
        df["id"] = df.get("id", pd.Series([f"csv-{i}" for i in range(len(df))]))
        df["category"] = df.get("label", pd.Series([None] * len(df)))
    else:
        path = OUT_DIR / f"{split}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} missing: run `make data` first")
        df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    tickets = [
        TicketText(id=str(r["id"]), subject=r["subject"] or "", body=r["body"] or "")
        for _, r in df.iterrows()
    ]
    gold = [c if isinstance(c, str) and c else None for c in df["category"]]
    return tickets, gold


def cmd_classify(args) -> int:
    from baseline.scribe import ScribeClassifier

    tickets, gold = load_tickets(args.split, args.csv, args.limit)
    clf = ScribeClassifier(device=args.device)
    preds = clf.classify(tickets)
    rows = []
    for t, g, p in zip(tickets, gold, preds, strict=True):
        rows.append({"id": t.id, "gold": g, **p.model_dump()})
        print(
            f"{t.id[:22]:<22} {p.stage:<9} {p.category:<18} {p.confidence:.2f} "
            f"urgency={p.urgency:<6} gold={g}"
        )
    labelled = [r for r in rows if r["gold"]]
    if labelled:
        correct = sum(r["gold"] == r["category"] for r in labelled)
        print(f"\naccuracy on {len(labelled)} labelled rows: {correct / len(labelled):.3f}")
    s = clf.stats
    seconds = {k: round(v, 2) for k, v in s.seconds.items()}
    print(f"rules: {s.by_rule}/{s.tickets}; seconds: {seconds}")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return 0


def cmd_draft(args) -> int:
    from app.llm import build_llm
    from app.settings import get_settings
    from baseline.scribe import ScribeClassifier, ScribeDrafter

    tickets, _ = load_tickets(args.split, args.csv, args.limit)
    preds = ScribeClassifier(device=args.device).classify(tickets)
    llm = build_llm(get_settings(), "draft")
    drafter = ScribeDrafter(llm)
    for t, p in zip(tickets, preds, strict=True):
        d = drafter.draft(t, p)
        print(f"=== {t.id} | {p.category} ({p.stage}) | {t.subject}\n{d.body}\n")
    print(f"token usage: {llm.usage}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseline", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("classify", cmd_classify), ("draft", cmd_draft)):
        p = sub.add_parser(name)
        src = p.add_mutually_exclusive_group()
        src.add_argument("--split", choices=["train", "val", "test"], default="test")
        src.add_argument("--csv", help="SCRIBE-format CSV (Sender,Subject,Body[,Label])")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--device", default=None, help="cpu | mps (default: auto)")
        p.set_defaults(fn=fn)
        if name == "classify":
            p.add_argument("--out", help="write predictions as JSON lines")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
