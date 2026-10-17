"""Fill the expected outcome of each demo ticket (P8.1).

    python -m eval.demo_expect      # == make demo-expect (retrieval only, no LLM calls)

datasets/demo/demo_tickets.csv holds ~25 hand-written demo tickets: WixQA
`demo`-split questions rewritten as emails (never used in P7.2), out-of-KB IT
tickets, meeting requests, spam and billing tickets the Wix KB doesn't cover.
This script measures each row's best retrieval score on the live KB
(DATABASE_URL, read-only) and records what the pipeline should do, given the
gold label and the gate's floor:

- Spam -> mute; Meeting Scheduling -> schedule; Feature Request -> product queue;
- otherwise reply: `abstain` if the best top-5 cosine < retrieval_floor;
  above it, `ground` for WixQA rows (the KB has the answer) and
  `draft (gate decides)` for the rest (the faithfulness judge decides live).

The actual outcome also depends on the classifier (the style shortcut, §5), so
the demo shows both and `make demo` prints the comparison.
"""

from __future__ import annotations

import csv

from app.settings import REPO_ROOT, get_settings
from app.textinput import model_input
from rag.faithfulness import load_config
from rag.kb import KB_MAX_LENGTH, connect
from rag.retriever import Retriever

PATH = REPO_ROOT / "datasets" / "demo" / "demo_tickets.csv"


def expected(label: str, kind: str, score: float, floor: float) -> str:
    if label == "Spam":
        return "mute"
    if label == "Meeting Scheduling":
        return "schedule"
    if label == "Feature Request":
        return "queue:product"
    if score < floor:
        return "abstain"
    # a KB answer exists only for the WixQA rows; others reach the judge
    return "ground" if kind == "wixqa" else "draft (gate decides)"


def main() -> int:
    from triage.embedder import Embedder

    with open(PATH, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    floor = load_config().retrieval_floor
    retriever = Retriever(
        connect(get_settings().database_url or None), Embedder(max_length=KB_MAX_LENGTH).encode
    )
    for r in rows:
        res = retriever.search(model_input(r["Subject"], r["Body"]), k=5)
        r["Retrieval_Max_Score"] = f"{res.max_score:.4f}"
        r["Expected_Outcome"] = expected(r["Label"], r["Kind"], res.max_score, floor)
        gold = r["WixQA_Id"]
        hit = ""
        if gold:
            from eval.wixqa_splits import load_split, question_id

            golds = {
                question_id(q.kind, q.question): set(q.article_ids) for q in load_split("demo")
            }
            hit = "gold in top-5" if golds[gold] & {c.doc_id for c in res.chunks} else "gold missed"
        print(f"{r['Id']}  {r['Label']:<18} {res.max_score:.4f}  {r['Expected_Outcome']:<13} {hit}")
    with open(PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"floor {floor}; wrote {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
