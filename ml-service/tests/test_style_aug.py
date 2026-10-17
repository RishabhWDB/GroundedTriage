import json
import re

import pandas as pd
import pytest

from app.llm import FakeLLM
from data.style_aug import BATCH_SIZE, build_prompt, generate, select, styles
from data.synthetic_meetings import Cache


def train_frame():
    cats = {"Technical Issue": 40, "Billing": 12, "Meeting Scheduling": 8}
    rows = []
    for cat, n in cats.items():
        for i in range(n):
            rows.append(
                {
                    "id": f"{cat[:4]}-{i:03d}",
                    "category": cat,
                    "urgency": "high" if cat != "Meeting Scheduling" else None,
                    "subject": f"Formal subject {cat} {i}",
                    "body": f"Formal LLM-style ticket number {i} about {cat} with details.",
                    "text": f"Formal subject {cat} {i}\n\nFormal ticket {i} about {cat}.",
                }
            )
    return pd.DataFrame(rows)


def test_select_is_stratified_seeded_and_shuffled():
    tr = train_frame()
    a, b = select(tr), select(tr)
    assert a["id"].tolist() == b["id"].tolist()
    assert a["category"].value_counts().to_dict() == {
        "Technical Issue": 10, "Billing": 3, "Meeting Scheduling": 2
    }  # fmt: skip
    assert a["category"].tolist() != sorted(a["category"].tolist())  # interleaved


def test_prompt_contains_styles_and_tickets():
    tr = train_frame().iloc[:2]
    msgs = build_prompt(tr.to_dict("records"), styles(2))
    user = msgs[1]["content"]
    assert "### Ticket 2" in user and "Style:" in user and "exactly 2 emails" in user
    assert "anonymised tokens" in msgs[0]["content"]


def responder(messages):
    n = int(re.search(r"exactly (\d+) emails", messages[-1]["content"]).group(1))
    emails = [
        {
            "subject": f"hey quick q {i}",
            "body": f"hi, totally different casual words number {i} lol",
        }
        for i in range(n)
    ]
    return json.dumps({"emails": emails})


def test_generate_writes_resumes_and_copies_labels(tmp_path):
    tr = train_frame()
    out = tmp_path / "aug.jsonl"
    llm = FakeLLM(responder=responder)
    st = generate(llm, "m", Cache(tmp_path / "cache"), tr, out=out, limit=BATCH_SIZE + 2)
    assert st["written"] == BATCH_SIZE + 2 and len(llm.calls) == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    src = tr.set_index("id")
    for r in rows:
        assert r["id"] == f"aug-{r['aug_of']}" and r["origin"] == "augmented"
        assert r["category"] == src.loc[r["aug_of"], "category"]
    st2 = generate(llm, "m", Cache(tmp_path / "cache"), tr, out=out)
    assert st2["already_done"] == BATCH_SIZE + 2 and st2["total_done"] == 15  # 25% of 60


def test_generate_rejects_near_copies(tmp_path):
    tr = train_frame()

    def copier(messages):
        bodies = re.findall(r"Subject: [^\n]*\n([^\n]*)", messages[-1]["content"])
        return json.dumps({"emails": [{"subject": "s", "body": b} for b in bodies]})

    st = generate(
        FakeLLM(responder=copier), "m", Cache(tmp_path / "c"), tr, out=tmp_path / "a.jsonl"
    )
    assert st["written"] == 0 and st["rejected"] == 15


def test_generate_retries_json_mode_failures(tmp_path):
    from app.llm import LLMError

    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError('HTTP 400: {"error":{"code":"json_validate_failed"}}')
        return responder(messages)

    st = generate(
        FakeLLM(responder=flaky), "m", Cache(tmp_path / "c"), train_frame(),
        out=tmp_path / "a.jsonl", limit=BATCH_SIZE,
    )  # fmt: skip
    assert st["written"] == BATCH_SIZE and calls["n"] == 2

    def down(messages):
        raise LLMError("giving up after 6 attempts; last: HTTP 503")

    with pytest.raises(LLMError):
        generate(FakeLLM(responder=down), "m", Cache(tmp_path / "d"), train_frame(),
                 out=tmp_path / "b.jsonl", limit=BATCH_SIZE)  # fmt: skip
