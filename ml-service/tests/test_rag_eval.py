import json

import numpy as np

from app.llm import CachingLLM, FakeLLM, LLMError
from app.schemas import RetrievedChunk
from baseline.scribe import DRAFT_SYSTEM, REFINE_SYSTEM
from eval.rag_eval import Item, aggregate, context_scores, load_progress, report_markdown, run
from rag.faithfulness import JudgeCache
from rag.retriever import RetrievalResult

CHUNKS = [
    RetrievedChunk(chunk_id="a:0", doc_id="a", text="Domains renew yearly.", score=0.9),
    RetrievedChunk(chunk_id="b:0", doc_id="b", text="Refunds take 5 days.", score=0.8),
]


class FakeRetriever:
    def __init__(self, scores):
        self.scores = scores  # question -> max score

    def search(self, query, k=5, hybrid=False):
        s = self.scores[query]
        return RetrievalResult([c.model_copy(update={"score": s}) for c in CHUNKS], "dense")


def embed(texts):
    rng = np.random.default_rng(len(texts[0]))
    return [rng.normal(size=4) for _ in texts]


def drafter_responder(messages):
    if messages[0]["content"] in (DRAFT_SYSTEM, REFINE_SYSTEM):
        return "Your refund was processed today."  # ungrounded (SCRIBE) text
    return json.dumps({"subject": "Re", "body": "Domains renew yearly [1]."})


def judge_responder(messages):
    supported = "Your refund" not in messages[-1]["content"]
    return json.dumps({"claims": [{"claim": "x", "cited": [1], "supported": supported}]})


ITEMS = [
    Item(id="q1", kind="answerable", question="When do domains renew?", gold_ids=["a"],
         reference="Every year."),
    Item(id="o1", kind="out_of_kb", question="My VPN drops."),
]  # fmt: skip


def test_context_scores():
    assert context_scores(CHUNKS, {"a"}) == {"context_recall": 1.0, "context_precision": 0.5}
    assert context_scores(CHUNKS, set())["context_recall"] is None


def test_run_resumes_and_aggregates(tmp_path):
    retriever = FakeRetriever({"When do domains renew?": 0.9, "My VPN drops.": 0.5})
    drafter = CachingLLM(FakeLLM(responder=drafter_responder), tmp_path / "d", "m")
    judge_llm = FakeLLM(responder=judge_responder)
    progress = tmp_path / "progress.jsonl"
    jc = JudgeCache(tmp_path / "j")
    status = run(
        ITEMS, retriever, embed, drafter, judge_llm, "judge", progress, 0.7, judge_cache=jc
    )
    assert status["processed"] == 2 and status["stopped"] is None
    rows = load_progress(progress)
    assert len(rows) == 4
    g_ans, g_oob = rows[("q1", "grounded")], rows[("o1", "grounded")]
    assert not g_ans["abstained"] and g_ans["faithfulness"] == 1.0
    assert "answer_similarity" in g_ans
    assert g_oob["abstained"] and g_oob["abstain_reason"] == "below floor" and not g_oob["answer"]
    u = rows[("q1", "ungrounded")]
    assert u["faithfulness"] == 0.0 and u["unsupported_claims"] == 1

    # a second run does no LLM work
    n_calls = len(judge_llm.calls)
    status = run(
        ITEMS, retriever, embed, drafter, FakeLLM(), "judge", progress, 0.7, judge_cache=jc
    )
    assert status["processed"] == 0 and len(judge_llm.calls) == n_calls

    m = aggregate(list(rows.values()), ITEMS)
    assert m["grounded"]["abstain_precision"] == 1.0 and m["grounded"]["abstain_recall"] == 1.0
    assert m["grounded"]["all"]["hallucination_rate"] == 0.0
    assert m["ungrounded"]["all"]["hallucination_rate"] == 1.0
    assert m["ungrounded"]["abstain_precision"] is None
    md = report_markdown({"items": {}, "rows_done": 4, "rows_expected": 4, "floor": 0.7,
                          "judge_model": "j", "metrics": m})  # fmt: skip
    assert "complete" in md and "| grounded | answerable |" in md


def test_daily_cap_stops_cleanly(tmp_path):
    def capped(messages):
        raise LLMError("HTTP 429: Rate limit reached ... tokens per day (TPD): Limit 200000")

    retriever = FakeRetriever({"When do domains renew?": 0.9, "My VPN drops.": 0.5})
    status = run(ITEMS, retriever, embed, FakeLLM(responder=capped), FakeLLM(), "judge",
                 tmp_path / "p.jsonl", 0.7, judge_cache=JudgeCache(tmp_path / "j"))  # fmt: skip
    assert status["stopped"].startswith("daily cap") and status["processed"] == 0
