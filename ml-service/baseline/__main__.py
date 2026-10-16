"""SCRIBE baseline CLI (its own process: BART-MNLI is ~1.6 GB of RAM).

    python -m baseline classify --split test --limit 10
    python -m baseline classify --csv ../datasets/sample_emails.csv --out preds.jsonl
    python -m baseline draft --csv ../datasets/sample_emails.csv --limit 2
    python -m baseline evaluate --split test     # P2.3 -> results/baseline/
    python -m baseline evaluate --stress-only    # after the stress CSV review

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


def cmd_evaluate(args) -> int:
    from baseline.evaluate import RESULTS, evaluate, evaluate_stress_only

    if args.stress_only:
        m = evaluate_stress_only(Path(args.out) if args.out else RESULTS, args.device)
        print(json.dumps(m["stress_set"], indent=1))
        return 0
    m = evaluate(
        args.split,
        limit=args.limit,
        batch_size=args.batch_size,
        latency_n=args.latency_n,
        out=Path(args.out) if args.out else RESULTS,
        device=args.device,
    )
    c = m["classification"]
    print(json.dumps({k: {"macro_f1": v.get("macro_f1"), "accuracy": v.get("accuracy")}
                      for k, v in c.items()}, indent=1))  # fmt: skip
    print(f"rule-handled share: {m['stages']['rule_handled_share']}")
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
    p = sub.add_parser("evaluate", help="P2.3: metrics, confusion matrix, latency")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--limit", type=int, default=None, help="first N rows (smoke runs)")
    p.add_argument("--batch-size", type=int, default=4, help="4 was fastest on M1 MPS")
    p.add_argument("--latency-n", type=int, default=100, help="single-ticket latency sample")
    p.add_argument("--out", default=None, help="output dir (default results/baseline)")
    p.add_argument("--device", default=None, help="cpu | mps (default: auto)")
    p.add_argument(
        "--stress-only",
        action="store_true",
        help="only (re)score the reviewed stress set into an existing metrics.json",
    )
    p.set_defaults(fn=cmd_evaluate)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
