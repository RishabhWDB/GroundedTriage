import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models import Registry, get_registry
from app.schemas import Classification, ModelsLoaded, ReplyDraft
from app.settings import Settings


def test_health_reports_no_models_loaded(tmp_path):
    app.dependency_overrides[get_registry] = lambda: Registry(Settings(models_dir=tmp_path))
    try:
        resp = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == {"deep_tier": False, "urgency": False}
    assert body["fast_tier_loaded"] is False
    assert isinstance(body["torch_installed"], bool)


def test_classification_rejects_unknown_urgency():
    base = dict(
        ticket_id="t1",
        category="Billing",
        category_confidence=0.9,
        tier="fast",
        urgency_confidence=0.5,
        urgency_source="heuristic",
        models_loaded=ModelsLoaded(),
    )
    assert Classification(urgency="high", **base).low_confidence is False
    with pytest.raises(ValidationError):
        Classification(urgency="urgent", **base)
    with pytest.raises(ValidationError, match="unknown category"):
        Classification(urgency="high", **{**base, "category": "Complaint"})


def test_reply_draft_status_enum():
    ReplyDraft(ticket_id="t", status="abstained", citations=["doc:1"])
    with pytest.raises(ValidationError):
        ReplyDraft(ticket_id="t", status="pending")
