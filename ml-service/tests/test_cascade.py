import json

import pytest
import yaml
from fastapi.testclient import TestClient

from app.interfaces import TicketText
from app.main import app
from app.models import Registry, get_registry
from app.settings import Settings
from app.taxonomy import categories
from triage.cascade import Cascade, load_tau
from triage.fast_tier import TierPrediction
from triage.tune_threshold import pick_tau, sweep, tune, write_tau
from triage.urgency import UrgencyPredictor


class StubTier:
    """Predicts from a {text: (category, confidence)} table."""

    def __init__(self, table, version="stub", available=True, fail=False):
        self.table, self.version, self.available, self.fail = table, version, available, fail
        self.calls: list[list[str]] = []

    def predict(self, texts):
        self.calls.append(list(texts))
        if self.fail:  # like HFClassifier: a load failure makes it unavailable
            self.available = False
            raise RuntimeError("deep tier failed to load")
        return [TierPrediction(*self.table[t]) for t in texts]


class StubUrgencyModel:
    def __init__(self, level="medium", available=True):
        self.level, self.available, self.version = level, available, "urg-stub"
        self.labels = ["low", "medium", "high"]

    def predict_proba(self, texts):
        import numpy as np

        row = [0.1, 0.1, 0.1]
        row[self.labels.index(self.level)] = 0.8
        return np.array([row] * len(texts))


FAST = {
    "sure": ("Billing", 0.95),
    "unsure": ("General Inquiry", 0.40),
    "outage": ("Technical Issue", 0.90),
}
DEEP = {"unsure": ("Technical Issue", 0.88), "sure": ("Spam", 0.99)}


def test_cascade_with_deep_tier():
    deep = StubTier(DEEP, "deep-v1")
    c = Cascade(StubTier(FAST, "fast-v1"), deep, UrgencyPredictor(StubUrgencyModel("high")), 0.7)
    r = c.run(["sure", "unsure"])
    assert (r[0].category, r[0].tier, r[0].low_confidence) == ("Billing", "fast", False)
    assert (r[1].category, r[1].tier, r[1].low_confidence) == ("Technical Issue", "deep", False)
    assert r[1].fast.category == "General Inquiry"
    assert deep.calls == [["unsure"]]  # only below-tau tickets reach the deep tier
    assert r[0].urgency.source == "model" and r[0].urgency.urgency == "high"
    assert c.model_versions() == {
        "fast_tier": "fast-v1",
        "tau": "0.7",
        "deep_tier": "deep-v1",
        "urgency": "urg-stub",
    }
    assert c.models_loaded().model_dump() == {"deep_tier": True, "urgency": True}


@pytest.mark.parametrize(
    "deep",
    [None, StubTier(DEEP, available=False), StubTier(DEEP, fail=True)],
    ids=["absent", "unavailable", "fails-at-load"],
)
def test_cascade_without_deep_tier_flags_low_confidence(deep):
    c = Cascade(StubTier(FAST), deep, UrgencyPredictor(None), 0.7)
    r = c.run(["sure", "unsure"])
    assert (r[0].tier, r[0].low_confidence) == ("fast", False)
    assert (r[1].category, r[1].tier, r[1].low_confidence) == ("General Inquiry", "fast", True)
    assert r[1].urgency.source == "heuristic"
    assert "deep_tier" not in c.model_versions()
    assert c.model_versions()["urgency"] == "heuristic"


def test_cascade_no_escalation_skips_deep():
    deep = StubTier(DEEP)
    Cascade(StubTier(FAST), deep, UrgencyPredictor(None), 0.5).run(["sure"])
    assert deep.calls == []
    assert Cascade(StubTier(FAST), deep, UrgencyPredictor(None), 0.5).run([]) == []


def test_classify_tickets_contract_and_critical():
    c = Cascade(StubTier(FAST), None, UrgencyPredictor(StubUrgencyModel("low")), 0.7)
    t = TicketText(id="t-1", subject="", body="outage")
    t_crit = TicketText(id="t-2", subject="", body="production is down for all users, urgent")
    FAST[t_crit.text] = ("Technical Issue", 0.99)
    out = c.classify_tickets([t, t_crit])
    assert (out[0].ticket_id, out[0].urgency, out[0].urgency_source) == ("t-1", "low", "model")
    assert out[1].urgency == "critical" and out[1].urgency_source == "heuristic"
    assert out[0].models_loaded.urgency is True and out[0].models_loaded.deep_tier is False
    preds = c.classify([t])
    assert preds[0].stage == "fast" and preds[0].category == "Technical Issue"


def test_load_tau(tmp_path):
    p = tmp_path / "cascade.yaml"
    p.write_text("tau: 0.65\n")
    assert load_tau(p) == 0.65
    p.write_text("tau: 1.5\n")
    with pytest.raises(ValueError):
        load_tau(p)


# ---- threshold tuning ----


def test_sweep_and_pick_rule():
    gold = ["Billing", "Billing", "Spam", "Spam"]
    fast_pred = ["Billing", "Spam", "Spam", "Billing"]
    fast_conf = [0.9, 0.5, 0.8, 0.3]
    deep_pred = ["Billing", "Billing", "Spam", "Spam"]  # deep is always right
    rows = sweep(gold, fast_pred, fast_conf, deep_pred, taus=[0.0, 0.4, 0.6, 0.85, 1.0])
    share = {r["tau"]: r["deep_share"] for r in rows}
    assert share == {0.0: 0.0, 0.4: 0.25, 0.6: 0.5, 0.85: 0.75, 1.0: 1.0}
    acc = {r["tau"]: r["accuracy"] for r in rows}
    assert acc[0.0] == 0.5 and acc[0.6] == 1.0 and acc[1.0] == 1.0
    choice = pick_tau(rows)
    # tau 0.6 reaches the best macro-F1 with only half the tickets on the deep tier.
    assert choice["tau"] == 0.6 and choice["deep_share"] == 0.5


def test_pick_tau_tolerance():
    rows = [
        {"tau": 0.5, "deep_share": 0.2, "macro_f1": 0.796, "accuracy": 0.8},
        {"tau": 0.7, "deep_share": 0.4, "macro_f1": 0.800, "accuracy": 0.8},
        {"tau": 0.3, "deep_share": 0.1, "macro_f1": 0.790, "accuracy": 0.8},
    ]
    assert pick_tau(rows)["tau"] == 0.5  # 0.796 is within 0.005 of 0.800


def toy_val(tmp_path):
    import pandas as pd

    texts = ["sure", "unsure", "outage"]
    gold = ["Billing", "Technical Issue", "Technical Issue"]
    d = tmp_path / "processed"
    d.mkdir()
    pd.DataFrame({"id": texts, "text": texts, "category": gold}).to_parquet(d / "val.parquet")
    return d


def test_tune_without_deep_tier_exits_cleanly(tmp_path, capsys):
    cfg = tmp_path / "cascade.yaml"
    cfg.write_text("tau: 0.7\ntuned: false\n")
    assert tune(StubTier(FAST), None, tmp_path / "out", toy_val(tmp_path), cfg) is None
    assert "needs the deep tier" in capsys.readouterr().out
    assert yaml.safe_load(cfg.read_text()) == {"tau": 0.7, "tuned": False}
    assert not (tmp_path / "out").exists()  # no fake curve


def test_tune_with_stub_tiers_writes_tau(tmp_path):
    cfg = tmp_path / "cascade.yaml"
    cfg.write_text("# comment\ntau: 0.7\ntuned: false\n")
    deep = StubTier({**DEEP, "outage": ("Technical Issue", 0.9)}, "deep-v1")
    choice = tune(StubTier(FAST, "fast-v1"), deep, tmp_path / "out", toy_val(tmp_path), cfg)
    assert 0.40 < choice["tau"] <= 0.90 and choice["macro_f1"] == 1.0
    saved = yaml.safe_load(cfg.read_text())
    assert saved["tau"] == choice["tau"] and saved["tuned"] is True
    assert saved["models"] == {"fast_tier": "fast-v1", "deep_tier": "deep-v1"}
    sweep_file = json.loads((tmp_path / "out" / "tau_sweep.json").read_text())
    assert len(sweep_file["rows"]) == 101
    assert (tmp_path / "out" / "cost_curve.png").exists()
    write_tau(choice, {}, cfg)  # idempotent rewrite keeps a loadable file
    assert load_tau(cfg) == choice["tau"]


# ---- /classify with only the fast tier ----


@pytest.fixture
def fast_only_client(tmp_path):
    from tests.test_fast_tier import save_toy_model

    save_toy_model(tmp_path / "models" / "fast_tier")
    reg = Registry(Settings(models_dir=tmp_path / "models"))
    app.dependency_overrides[get_registry] = lambda: reg
    yield TestClient(app), reg
    app.dependency_overrides.clear()


def test_classify_endpoint_fast_tier_only(fast_only_client):
    client, reg = fast_only_client
    h = client.get("/health").json()
    assert h["fast_tier_loaded"] is False
    assert h["models_loaded"] == {"deep_tier": False, "urgency": False}
    r = client.post(
        "/classify",
        json={"ticket_id": "t1", "subject": "Refund", "body": "refund my invoice payment ASAP"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticket_id"] == "t1" and body["category"] == "Billing"
    assert body["tier"] == "fast"
    assert body["urgency"] == "high" and body["urgency_source"] == "heuristic"
    assert body["models_loaded"] == {"deep_tier": False, "urgency": False}
    assert body["model_versions"]["fast_tier"] == "fast-test"
    assert body["model_versions"]["urgency"] == "heuristic"
    assert body["low_confidence"] == (body["category_confidence"] < load_tau())
    assert client.get("/health").json()["fast_tier_loaded"] is True
    assert body["category"] in categories()


def test_classify_endpoint_without_fast_tier(tmp_path):
    reg = Registry(Settings(models_dir=tmp_path / "empty"))
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        r = TestClient(app).post("/classify", json={"ticket_id": "t", "body": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and "make fast-tier" in r.json()["detail"]
