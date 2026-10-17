import json

from app.llm import FakeLLM
from eval.judge_labels import (
    agreement,
    claim_verdict,
    kappa,
    label,
    read_labels,
    sample,
    score,
    sources_text,
    write_labels,
)
from rag.faithfulness import JudgeCache

CHUNKS = {"a:0": {"title": "Domains", "text": "Domains renew yearly."},
          "b:0": {"title": "Refunds", "text": "Refunds take 5 days."}}  # fmt: skip


def row(item, system, supported):
    claims = [
        {"claim": f"{item}-{i}", "cited": [2], "supported": s} for i, s in enumerate(supported)
    ]
    return {"item_id": item, "system": system, "retrieved": ["a:0", "b:0"], "claims": claims}


def test_sources_text_uses_cited_else_all():
    assert sources_text(["a:0", "b:0"], [2], CHUNKS) == "[2] Refunds\nRefunds take 5 days."
    assert sources_text(["a:0", "b:0"], [], CHUNKS).count("[") == 2
    assert sources_text(["a:0", "b:0"], [9], CHUNKS).startswith("[1] Domains")


def test_sample_is_stratified_and_hides_verdicts():
    rows = [row("q1", "grounded", [True] * 6), row("q2", "ungrounded", [False] * 3 + [True] * 3)]
    out, verdicts = sample(rows, 6, fetch=lambda ids: CHUNKS)
    assert len(out) == 6 and len({r["claim_id"] for r in out}) == 6
    assert sum(not verdicts[r["claim_id"]] for r in out) == 3  # every unsupported claim is taken
    assert all("supported" not in r and r["human_supported"] == "" for r in out)
    assert out == sample(rows, 6, fetch=lambda ids: CHUNKS)[0]  # deterministic


def test_kappa_and_agreement():
    assert kappa([True, False], [True, False]) == 1.0
    assert kappa([True, True, False, False], [True, False, True, False]) == 0.0
    a = agreement([True, True, False], [True, False, False])
    assert a["accuracy"] == round(2 / 3, 4) and a["human_yes_judge_no"] == 1


def test_label_resumes_and_score(tmp_path):
    path = tmp_path / "claims.csv"
    rows, verdicts = sample(
        [row("q1", "grounded", [True, False, True])], 3, fetch=lambda ids: CHUNKS
    )
    write_labels(rows, path)
    answers = iter(["x", "y", "q"])
    label(path, ask=lambda _: next(answers), show=lambda *_: None)
    assert [r["human_supported"] for r in read_labels(path)].count("yes") == 1
    answers = iter(["n", "u"])
    label(path, ask=lambda _: next(answers), show=lambda *_: None)
    labelled = read_labels(path)
    assert all(r["human_supported"] for r in labelled)
    s = score(labelled, verdicts)
    assert s["labelled"] == 2 and s["unsure"] == 1 and s["preliminary"]
    assert s["judges"]["gate (qwen, whole-draft)"]["n"] == 2


def test_claim_verdict_caches(tmp_path):
    cache = JudgeCache(tmp_path)
    llm = FakeLLM([json.dumps({"supported": True}), "not json"])
    assert claim_verdict(llm, "m", "c", "s", cache) is True
    assert claim_verdict(llm, "m", "c", "s", cache) is True  # cached: no second call
    assert len(llm.calls) == 1
    assert claim_verdict(llm, "m", "other", "s", cache) is None
