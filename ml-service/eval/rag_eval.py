"""RAG evaluation (P7.2): grounded vs ungrounded (SCRIBE-style) replies.

    python -m eval.rag_eval --plan            # sizing + expected days, no LLM calls
    python -m eval.rag_eval --sweep           # floor sweep on p72_dev (retrieval only)
    python -m eval.rag_eval [--limit N]       # == make rag-eval (resumable; run daily)

Items (fixed, see eval/wixqa_splits.py):
- answerable: the 228 WixQA `p72_test` questions (gold article ids + reference answers);
- out_of_kb: OUT_OF_KB Tobi-Bueck val tickets (generic IT support the Wix KB doesn't
  cover), seeded and disjoint from the 400 val tickets used to set the floor.
The demo questions (`demo` split) are never used.

Systems (same drafter, gpt-oss-120b, for both):
- grounded: retrieve top-5 -> best score < floor ? abstain : draft -> refine -> gate (P5.4);
- ungrounded: SCRIBE's draft -> refine with no sources (baseline.scribe.ScribeDrafter),
  then judged against the same retrieved chunks (measures how much of an
  ungrounded answer the KB supports).

Metrics (a documented RAGAS-style equivalent; the `ragas` package needs an
OpenAI-style stack and many judge calls per item, which the 8 GB Mac and
Groq's free caps can't carry):
- faithfulness: supported / all atomic claims (judge = qwen3.8-27b, as the gate);
- hallucination rate: share of non-abstained replies with >= 1 unsupported claim;
- answer relevance: bge cosine(question [query instruction], answer);
- answer similarity: bge cosine(answer, WixQA reference answer) (answerable only);
- context recall: a gold article among the top-5 chunks' documents;
  context precision: share of the top-5 chunks that come from gold articles;
- abstain rate; abstain precision / recall w.r.t. out_of_kb items.

Every LLM call is cached on disk (app.llm.CachingLLM for drafts, JudgeCache for
judgements) and each finished (item, system) row is appended to
datasets/raw/rag_eval/progress.jsonl, so a run stopped by Groq's daily cap
resumes where it left off. Outputs: results/rag/{metrics.json, report.md,
floor_sweep.json}.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.fileio import append_line, open_append, read_jsonl
from app.interfaces import CategoryPrediction, TicketText
from app.llm import CachingLLM, LLMError, build_llm
from app.settings import REPO_ROOT, get_settings
from data.build import OUT_DIR
from eval.runinfo import run_info
from eval.wixqa_splits import load_split, question_id
from rag.faithfulness import JudgeCache, judge, load_config
from rag.generator import DraftRequest, generate
from rag.kb import KB_MAX_LENGTH, connect
from rag.retriever import QUERY_INSTRUCTION, Retriever

logger = logging.getLogger(__name__)

RESULTS = REPO_ROOT / "results" / "rag"
WORK = REPO_ROOT / "datasets" / "raw" / "rag_eval"
DRAFT_CACHE = REPO_ROOT / "datasets" / "raw" / "llm_cache" / "rag_eval_drafts"
OUT_OF_KB = 50
K = 5
SEED = 7
CATEGORY, URGENCY = "General Inquiry", "medium"  # WixQA questions carry no ticket metadata
FLOOR_TICKETS_SEED, FLOOR_TICKETS = 42, 400  # eval/retrieval.py's floor sample (excluded)
SWEEP_KEEP = (0.85, 0.90, 0.95, 0.98)
# A-priori token estimates per call (recomputed from real usage after a run).
EST = {
    "grounded_draft": 2600,
    "grounded_refine": 900,
    "ungrounded_draft": 700,
    "ungrounded_refine": 900,
    "judge": 2600,
}
DAILY_CAP = 200_000  # gpt-oss-120b tokens/day on the free tier (confirmed)


@dataclass
class Item:
    id: str
    kind: str  # answerable | out_of_kb
    question: str
    gold_ids: list[str] = field(default_factory=list)
    reference: str = ""


def out_of_kb_items(n: int, split_seed: int, exclude: set[str]) -> list[Item]:
    val = pd.read_parquet(OUT_DIR / "val.parquet")
    val = val[~val["id"].isin(exclude)]
    pick = val.sample(n=n, random_state=split_seed)
    return [Item(id=f"oob-{r.id}", kind="out_of_kb", question=r.text) for r in pick.itertuples()]


def floor_sample_ids() -> set[str]:
    val = pd.read_parquet(OUT_DIR / "val.parquet")
    return set(val.sample(n=FLOOR_TICKETS, random_state=FLOOR_TICKETS_SEED)["id"])


def build_items(split: str = "p72_test", n_oob: int = OUT_OF_KB, oob_seed: int = 101) -> list[Item]:
    items = [
        Item(
            id=question_id(q.kind, q.question),
            kind="answerable",
            question=q.question,
            gold_ids=q.article_ids,
            reference=q.answer,
        )
        for q in load_split(split)
    ]
    exclude = floor_sample_ids()
    if split != "p72_test":  # the dev out-of-KB set must not overlap the test one
        exclude |= {
            i.id.removeprefix("oob-") for i in out_of_kb_items(OUT_OF_KB, 101, floor_sample_ids())
        }
    items += out_of_kb_items(n_oob, oob_seed, exclude)
    random.Random(SEED).shuffle(items)
    return items


def context_scores(chunks, gold: set[str]) -> dict:
    if not gold:
        return {"context_recall": None, "context_precision": None}
    hits = [c.doc_id in gold for c in chunks]
    return {
        "context_recall": float(any(hits)),
        "context_precision": float(np.mean(hits)) if hits else 0.0,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------- plan / sweep


def plan(items: list[Item], floor: float, retriever: Retriever | None) -> dict:
    n_ans = sum(i.kind == "answerable" for i in items)
    n_oob = len(items) - n_ans
    drafted_share = {"answerable": 0.95, "out_of_kb": 0.12}  # from the floor analysis (P5.2)
    if retriever is not None:  # measure the real below-floor share (retrieval only)
        for kind in ("answerable", "out_of_kb"):
            scores = [retriever.search(i.question, k=K).max_score for i in items if i.kind == kind]
            drafted_share[kind] = float(np.mean(np.array(scores) >= floor)) if scores else 0.0
    grounded = n_ans * drafted_share["answerable"] + n_oob * drafted_share["out_of_kb"]
    ungrounded = n_ans + n_oob
    gpt = grounded * (EST["grounded_draft"] + EST["grounded_refine"]) + ungrounded * (
        EST["ungrounded_draft"] + EST["ungrounded_refine"]
    )
    judge_calls = grounded + ungrounded
    return {
        "items": {"answerable": n_ans, "out_of_kb": n_oob},
        "drafted_share": {k: round(v, 3) for k, v in drafted_share.items()},
        "grounded_drafts": round(grounded),
        "ungrounded_drafts": ungrounded,
        "gpt_oss_calls": round(2 * grounded + 2 * ungrounded),
        "gpt_oss_tokens_est": round(gpt),
        "expected_days_at_free_cap": round(gpt / DAILY_CAP, 1),
        "judge_calls": round(judge_calls),
        "judge_hours_est_at_1_per_min": round(judge_calls / 60, 1),
        "note": "estimates; real usage per call is printed after each run and replaces them",
    }


def floor_sweep(retriever: Retriever, items: list[Item]) -> dict:
    """On p72_dev: for each keep-rate, the floor that keeps that share of
    answerable questions, and the abstain precision/recall it gives on dev."""
    ans = np.array(
        [retriever.search(i.question, k=K).max_score for i in items if i.kind == "answerable"]
    )
    oob = np.array(
        [retriever.search(i.question, k=K).max_score for i in items if i.kind == "out_of_kb"]
    )
    rows = []
    for keep in SWEEP_KEEP:
        floor = float(np.quantile(ans, 1 - keep))
        abst_ans, abst_oob = int((ans < floor).sum()), int((oob < floor).sum())
        abst = abst_ans + abst_oob
        rows.append(
            {
                "keep": keep,
                "floor": round(floor, 4),
                "abstain_precision": round(abst_oob / abst, 4) if abst else None,
                "abstain_recall": round(abst_oob / len(oob), 4) if len(oob) else None,
                "answerable_abstained": round(abst_ans / len(ans), 4),
            }
        )
    return {
        "split": "p72_dev",
        "answerable_n": len(ans),
        "out_of_kb_n": len(oob),
        "rows": rows,
        "chosen_floor": load_config().retrieval_floor,
        "note": "the floor is not moved from these numbers or the final eval (Phase 5 review)",
    }


# ---------------------------------------------------------------- run


class DailyCapReached(RuntimeError):
    pass


def call(fn, *args, **kw):
    """Run an LLM step; wait out per-minute limits, stop on the daily cap."""
    for attempt in range(4):
        try:
            return fn(*args, **kw)
        except LLMError as exc:
            msg = str(exc)
            if "per day" in msg or "TPD" in msg or "RPD" in msg:
                raise DailyCapReached(msg) from exc
            if "429" in msg and attempt < 3:
                logger.warning("rate limited; waiting 65 s (%s)", msg[:120])
                time.sleep(65)
                continue
            raise
    raise RuntimeError("unreachable")


def load_progress(path: Path) -> dict[tuple[str, str], dict]:
    return {(r["item_id"], r["system"]): r for r in read_jsonl(path)}


def run(
    items: list[Item],
    retriever: Retriever,
    embed,
    drafter,
    judge_llm,
    judge_model: str,
    progress_path: Path,
    floor: float,
    limit: int | None = None,
    judge_cache: JudgeCache | None = None,
) -> dict:
    from baseline.scribe import ScribeDrafter

    done = load_progress(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    cache = judge_cache or JudgeCache()
    scribe = ScribeDrafter(drafter)
    status: dict = {"processed": 0, "stopped": None, "errors": []}
    todo = items if limit is None else items[:limit]
    with open_append(progress_path) as f:
        for it in todo:
            if all((it.id, s) in done for s in ("grounded", "ungrounded")):
                continue
            res = retriever.search(it.question, k=K)
            ctx = context_scores(res.chunks, set(it.gold_ids))
            base = {
                "item_id": it.id,
                "kind": it.kind,
                "retrieval_max_score": res.max_score,
                "retrieved": [c.chunk_id for c in res.chunks],
                **ctx,
            }
            try:
                if (it.id, "grounded") not in done:
                    row = {**base, "system": "grounded"}
                    if not res.chunks or res.max_score < floor:
                        row.update(
                            abstained=True, abstain_reason="below floor", answer="", claims=[]
                        )
                    else:
                        d = call(
                            generate,
                            DraftRequest(
                                ticket_id=it.id,
                                body=it.question,
                                category=CATEGORY,
                                urgency=URGENCY,
                                chunks=res.chunks,
                            ),
                            drafter,
                        )
                        claims = call(judge, d.body, res.chunks, judge_llm, judge_model, cache)
                        faith = (
                            1.0 if not claims else sum(c.supported for c in claims) / len(claims)
                        )
                        gate = load_config()
                        row.update(
                            abstained=faith < gate.min_faithfulness,
                            answer=d.body,
                            abstain_reason="faithfulness" if faith < gate.min_faithfulness else "",
                            claims=[c.model_dump() for c in claims],
                        )
                    done[(it.id, "grounded")] = finish(row, it, embed)
                    append_line(f, done[(it.id, "grounded")])
                    f.flush()
                if (it.id, "ungrounded") not in done:
                    pred = CategoryPrediction(
                        category=CATEGORY,
                        confidence=1.0,
                        stage="zero_shot",
                        signals={"sentiment": 0.0},
                    )
                    d = call(scribe.draft, TicketText(id=it.id, subject="", body=it.question), pred)
                    claims = (
                        call(judge, d.body, res.chunks, judge_llm, judge_model, cache)
                        if res.chunks
                        else []
                    )
                    row = {
                        **base,
                        "system": "ungrounded",
                        "abstained": False,
                        "abstain_reason": "",
                        "answer": d.body,
                        "claims": [c.model_dump() for c in claims],
                    }
                    done[(it.id, "ungrounded")] = finish(row, it, embed)
                    append_line(f, done[(it.id, "ungrounded")])
                    f.flush()
            except DailyCapReached as exc:
                status["stopped"] = f"daily cap: {str(exc)[:160]}"
                logger.warning("stopping: Groq daily cap reached; re-run tomorrow to resume")
                break
            except LLMError as exc:  # e.g. judge JSON failed twice: retried on the next run
                logger.warning("item %s skipped this run: %s", it.id, exc)
                status["errors"].append({"item_id": it.id, "error": str(exc)[:200]})
                continue
            status["processed"] += 1
            logger.info("rag-eval: item %s done (%d rows total)", it.id, len(done))
    status["done_rows"] = len(done)
    return status


def finish(row: dict, it: Item, embed) -> dict:
    claims = row["claims"]
    row["faithfulness"] = (
        (1.0 if not claims else sum(c["supported"] for c in claims) / len(claims))
        if row["answer"]
        else None
    )
    row["unsupported_claims"] = sum(not c["supported"] for c in claims)
    if row["answer"]:
        q, a = embed([QUERY_INSTRUCTION + it.question, row["answer"]])
        row["answer_relevance"] = round(cosine(q, a), 4)
        if it.reference:
            ref = embed([it.reference])[0]
            row["answer_similarity"] = round(cosine(a, ref), 4)
    return row


# ---------------------------------------------------------------- metrics


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def aggregate(rows: list[dict], items: list[Item]) -> dict:
    by_kind = {i.id: i.kind for i in items}
    out = {}
    for system in ("grounded", "ungrounded"):
        sys_rows = [r for r in rows if r["system"] == system and r["item_id"] in by_kind]
        m: dict = {"n": len(sys_rows)}
        for kind in ("answerable", "out_of_kb", "all"):
            rs = sys_rows if kind == "all" else [r for r in sys_rows if r["kind"] == kind]
            answered = [r for r in rs if r["answer"] and not r["abstained"]]
            m[kind] = {
                "n": len(rs),
                "answered": len(answered),
                "abstain_rate": _share([r["abstained"] for r in rs]),
                "faithfulness": _mean(answered, "faithfulness"),
                "hallucination_rate": _share([r["unsupported_claims"] > 0 for r in answered]),
                "answer_relevance": _mean(answered, "answer_relevance"),
                "answer_similarity": _mean(answered, "answer_similarity"),
                "context_recall": _mean(rs, "context_recall"),
                "context_precision": _mean(rs, "context_precision"),
            }
        abstained = [r for r in sys_rows if r["abstained"]]
        m["abstain_precision"] = _share([r["kind"] == "out_of_kb" for r in abstained])
        m["abstain_recall"] = _share([r["abstained"] for r in sys_rows if r["kind"] == "out_of_kb"])
        out[system] = m
    return out


COLUMNS = [
    "n",
    "answered",
    "abstain_rate",
    "faithfulness",
    "hallucination_rate",
    "answer_relevance",
    "answer_similarity",
    "context_recall",
    "context_precision",
]


def report_markdown(m: dict) -> str:
    complete = m["rows_done"] >= m["rows_expected"]
    lines = [
        "# RAG evaluation: grounded vs ungrounded (SCRIBE-style)",
        "",
        f"Generated by `make rag-eval`. Items: {m['items']}; rows done: {m['rows_done']} of "
        f"{m['rows_expected']} ({'complete' if complete else 'PARTIAL, resumes daily'}).",
        f"Floor {m['floor']} (not moved). Same drafter (gpt-oss-120b) for both systems; "
        f"judge {m['judge_model']}.",
        "",
        "| system | slice | " + " | ".join(c.replace("_", " ") for c in COLUMNS) + " |",
        "|---|---|" + "---|" * len(COLUMNS),
    ]
    for system, sm in m["metrics"].items():
        for kind in ("answerable", "out_of_kb", "all"):
            cells = " | ".join(str(sm[kind][c]) for c in COLUMNS)
            lines.append(f"| {system} | {kind} | {cells} |")
    lines += ["", "| system | abstain precision | abstain recall (out-of-KB) |", "|---|---|---|"]
    for system, sm in m["metrics"].items():
        lines.append(f"| {system} | {sm['abstain_precision']} | {sm['abstain_recall']} |")
    lines += ["", "Metric definitions: the module docstring of `ml-service/eval/rag_eval.py`.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from triage.embedder import Embedder

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--plan", action="store_true", help="print sizing and expected days; no LLM calls"
    )
    ap.add_argument("--sweep", action="store_true", help="floor sweep on p72_dev (retrieval only)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N items this run")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = get_settings()
    floor = load_config().retrieval_floor
    emb = Embedder(max_length=KB_MAX_LENGTH)
    retriever = Retriever(connect(s.database_url or None), emb.encode,
                          reconnect=lambda: connect(s.database_url or None))  # fmt: skip
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        sweep = floor_sweep(retriever, build_items("p72_dev"))
        (RESULTS / "floor_sweep.json").write_text(json.dumps(sweep, indent=2))
        print(json.dumps(sweep, indent=2))
        return 0
    items = build_items()
    if args.plan:
        p = plan(items, floor, retriever)
        (RESULTS / "plan.json").write_text(json.dumps(p, indent=2))
        print(json.dumps(p, indent=2))
        return 0
    drafter_raw = build_llm(s, "draft")
    drafter = CachingLLM(drafter_raw, DRAFT_CACHE, s.groq_model)
    judge_llm = build_llm(s, "judge")
    progress = WORK / "progress.jsonl"
    status = run(
        items, retriever, emb.encode, drafter, judge_llm, s.judge_model, progress, floor, args.limit
    )
    rows = list(load_progress(progress).values())
    m = {
        "items": {
            "answerable": sum(i.kind == "answerable" for i in items),
            "out_of_kb": sum(i.kind == "out_of_kb" for i in items),
        },
        "rows_done": len(rows),
        "rows_expected": 2 * len(items),
        "floor": floor,
        "judge_model": s.judge_model,
        "metrics": aggregate(rows, items),
        "last_run": {
            **status,
            "drafter_usage": drafter_raw.usage,
            "judge_usage": judge_llm.usage,
            "draft_cache_hits": drafter.hits,
        },
        "run": run_info(),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    (RESULTS / "report.md").write_text(report_markdown(m))
    print(report_markdown(m))
    print(json.dumps(m["last_run"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
