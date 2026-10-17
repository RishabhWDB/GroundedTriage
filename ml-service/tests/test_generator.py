import json

import pytest
from fastapi.testclient import TestClient

from app.llm import FakeLLM, LLMError
from app.main import app
from app.models import Registry, get_registry
from app.schemas import RetrievedChunk
from app.settings import Settings
from rag.generator import (
    DraftRequest,
    build_draft_messages,
    build_refine_messages,
    generate,
    parse_citations,
)

CHUNKS = [
    RetrievedChunk(chunk_id=cid, doc_id=cid.split(":")[0], title=title, text=text, score=score)
    for cid, title, text, score in [
        ("kb1:0", "Refunds", "Refunds take 5 days.", 0.8),
        ("kb2:3", "Invoices", "Invoices are in Billing.", 0.7),
    ]
]


def req(**kw):
    base = {"ticket_id": "t1", "subject": "Charged twice", "body": "I was charged twice.",
            "category": "Billing", "urgency": "high", "chunks": CHUNKS}  # fmt: skip
    return DraftRequest(**{**base, **kw})


def reply(body, subject="Re: Charged twice"):
    return json.dumps({"subject": subject, "body": body})


def test_prompt_numbers_sources_and_sets_tone():
    sys_msg, user = build_draft_messages(req())
    assert "ONLY the numbered sources" in sys_msg["content"]
    assert "Never claim an action was taken" in sys_msg["content"]
    assert "[1] Refunds\nRefunds take 5 days." in user["content"]
    assert "[2] Invoices" in user["content"]
    assert "quick resolution" in user["content"] and "Be precise about amounts" in user["content"]
    refine = build_refine_messages("S", "B [1]")
    assert "must NOT add, remove or change any fact" in refine[0]["content"]


def test_parse_citations():
    body, valid, invalid = parse_citations("Refunds take 5 days [1]. Invoices [2, 7]. See [9].", 2)
    assert valid == [1, 2] and invalid == [7, 9]
    assert body == "Refunds take 5 days [1]. Invoices [2]. See."
    assert parse_citations("No sources here.", 2) == ("No sources here.", [], [])


def test_generate_draft_and_refine():
    llm = FakeLLM([
        reply("Sorry! Refunds take 5 days [1]. Find invoices in Billing [2]."),
        reply("Sorry for the trouble. Refunds take 5 days [1]; invoices are in Billing [2]."),
    ])  # fmt: skip
    d = generate(req(), llm)
    assert d.refined and d.body.startswith("Sorry for the trouble")
    assert d.citations == ["kb1:0", "kb2:3"] and d.cited_numbers == [1, 2]
    assert len(llm.calls) == 2 and all(
        c["json_mode"] and c["temperature"] == 0.0 for c in llm.calls
    )


def test_refine_that_changes_citations_is_dropped():
    llm = FakeLLM([reply("Refunds take 5 days [1]."), reply("Refunds take 5 days.")])
    d = generate(req(), llm)
    assert (
        not d.refined
        and d.body == "Refunds take 5 days [1]."
        and "refine pass dropped" in d.notes[0]
    )


def test_invalid_citations_and_placeholders_are_reported():
    d = generate(req(refine=False), FakeLLM([reply("Done [3]. Regards, [Your Name]")]))
    assert d.invalid_citations == [3] and d.citations == []
    assert any("non-existent" in n for n in d.notes) and any("placeholder" in n for n in d.notes)


def test_no_sources_and_bad_json():
    with pytest.raises(ValueError):
        generate(req(chunks=[]), FakeLLM([]))
    with pytest.raises(LLMError):
        generate(req(refine=False), FakeLLM(["nope", json.dumps({"subject": "x"})]))


def test_draft_endpoint(tmp_path):
    reg = Registry(Settings(models_dir=tmp_path, groq_api_key=""))
    app.dependency_overrides[get_registry] = lambda: reg
    body = req(refine=False).model_dump()
    try:
        client = TestClient(app)
        assert client.post("/draft", json=body).status_code == 503
        reg._llm = FakeLLM([reply("Refunds take 5 days [1].")])
        r = client.post("/draft", json=body)
        assert r.status_code == 200 and r.json()["citations"] == ["kb1:0"]
        assert client.post("/draft", json={**body, "chunks": []}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_json_mode_failure_is_retried():
    from app.llm import JSONModeError

    seq = iter([JSONModeError("json_validate_failed"), reply("Refunds take 5 days [1].")])

    def responder(messages):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    d = generate(req(refine=False), FakeLLM(responder=responder))
    assert d.citations == ["kb1:0"]
