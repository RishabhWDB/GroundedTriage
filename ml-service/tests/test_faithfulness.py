import json

import pytest
from fastapi.testclient import TestClient

from app.llm import FakeLLM, LLMError, TruncatedError
from app.main import app
from app.models import Registry, get_registry
from app.schemas import RetrievedChunk
from app.settings import Settings
from rag.faithfulness import (
    GateConfig,
    JudgeCache,
    VerifyRequest,
    judge,
    judge_messages,
    load_config,
    split_halves,
    verify,
)

CHUNKS = [
    RetrievedChunk(chunk_id=f"kb{i}:0", doc_id=f"kb{i}", title=t, text=x, score=s)
    for i, (t, x, s) in enumerate(
        [
            ("Refunds", "Refunds take 5 business days.", 0.82),
            ("Invoices", "Invoices are in Billing.", 0.71),
        ],
        start=1,
    )
]
CFG = GateConfig(min_faithfulness=0.9, retrieval_floor=0.6)


def vreq(chunks=CHUNKS):
    return VerifyRequest(ticket_id="t1", subject="Re: refund", body="Refunds take 5 days [1].",
                         citations=["kb1:0"], chunks=chunks)  # fmt: skip


def claims(*supported):
    return json.dumps({"claims": [{"claim": f"c{i}", "cited": [1], "supported": s, "reason": ""}
                                  for i, s in enumerate(supported)]})  # fmt: skip


def test_all_supported_is_grounded():
    v = verify(vreq(), FakeLLM([claims(True, True)]), CFG)
    assert v.draft.status == "grounded" and v.draft.faithfulness_score == 1.0
    assert (
        v.draft.unsupported_claims == []
        and v.retrieval_max_score == 0.82
        and v.abstain_reason == ""
    )


def test_partly_unsupported_abstains_with_claims():
    v = verify(vreq(), FakeLLM([claims(True, True, False)]), CFG)
    assert v.draft.status == "abstained" and v.draft.faithfulness_score == 0.6667
    assert v.draft.unsupported_claims == ["c2"] and "1 of 3 claims unsupported" in v.abstain_reason


def test_empty_retrieval_and_floor_abstain_without_calling_the_judge():
    llm = FakeLLM([])
    v = verify(vreq(chunks=[]), llm, CFG)
    assert v.draft.status == "abstained" and v.abstain_reason == "no retrieved sources"
    low = [c.model_copy(update={"score": 0.4}) for c in CHUNKS]
    v = verify(vreq(chunks=low), llm, CFG)
    assert v.draft.status == "abstained" and "< floor 0.6" in v.abstain_reason
    assert v.draft.faithfulness_score is None and llm.calls == []


def test_no_claims_counts_as_faithful():
    v = verify(vreq(), FakeLLM([json.dumps({"claims": []})]), CFG)
    assert v.draft.status == "grounded" and v.draft.faithfulness_score == 1.0


def test_bad_judge_output_retries_then_fails():
    v = verify(vreq(), FakeLLM(["not json", claims(True)]), CFG)
    assert v.draft.status == "grounded"
    with pytest.raises(LLMError):
        verify(vreq(), FakeLLM(["x", json.dumps({"nope": 1})]), CFG)
    with pytest.raises(LLMError):
        verify(vreq(), None, CFG)


def test_judge_cache_avoids_repeat_calls(tmp_path):
    cache = JudgeCache(tmp_path)
    llm = FakeLLM([claims(True)])
    verify(vreq(), llm, CFG, "qwen", cache)
    v = verify(vreq(), FakeLLM([]), CFG, "qwen", cache)  # served from the cache
    assert v.draft.status == "grounded" and len(list(tmp_path.iterdir())) == 1


def test_prompt_and_config():
    msgs = judge_messages("Body [1].", CHUNKS)
    assert "atomic factual claims" in msgs[0]["content"] and "[1] Refunds" in msgs[1]["content"]
    cfg = load_config()
    assert cfg.min_faithfulness == 0.9


def test_verify_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.faithfulness.CACHE_DIR", tmp_path / "judge")
    reg = Registry(Settings(models_dir=tmp_path))
    reg._judge = FakeLLM([claims(True, False)])
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        r = TestClient(app).post("/verify", json=vreq().model_dump())
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["draft"]["status"] == "abstained" and body["draft"]["unsupported_claims"] == ["c1"]


def test_json_mode_failure_is_retried():
    from app.llm import JSONModeError

    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise JSONModeError("HTTP 400 json_validate_failed")
        return claims(True)

    v = verify(vreq(), FakeLLM(responder=flaky), CFG)
    assert v.draft.status == "grounded" and calls["n"] == 2


def test_split_halves_prefers_paragraphs():
    assert split_halves("Para one is here.\n\nPara two.\n\nPara three is long.") == (
        "Para one is here.\n\nPara two.",
        "Para three is long.",
    )
    assert split_halves("One sentence. Two sentence") == ("One sentence", "Two sentence")
    assert split_halves("single") is None


def test_judge_splits_a_truncated_draft(tmp_path):
    def responder(messages):
        draft = messages[-1]["content"]
        if "Alpha" in draft and "Beta" in draft:
            raise TruncatedError("qwen: output truncated at max_tokens=1000")
        word = "Alpha" if "Alpha" in draft else "Beta"
        return json.dumps({"claims": [{"claim": word, "cited": [1], "supported": True}]})

    claims = judge("Alpha holds [1].\n\nBeta holds [1].", CHUNKS, FakeLLM(responder=responder),
                   "m", JudgeCache(tmp_path))  # fmt: skip
    assert [c.claim for c in claims] == ["Alpha", "Beta"]


def test_judge_gives_up_when_unsplittable(tmp_path):
    def responder(messages):
        raise TruncatedError("truncated")

    with pytest.raises(TruncatedError):
        judge("one block", CHUNKS, FakeLLM(responder=responder), "m", JudgeCache(tmp_path))
