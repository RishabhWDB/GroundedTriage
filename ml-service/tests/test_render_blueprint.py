import importlib.util

import yaml

from app.settings import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "validate_render", REPO_ROOT / "scripts" / "validate_render.py"
)
vr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vr)


def blueprint():
    return yaml.safe_load((REPO_ROOT / "render.yaml").read_text())


def test_shipped_blueprint_passes_repo_checks():
    assert vr.repo_checks(blueprint()) == []


def test_repo_checks_catch_secrets_refs_and_free_workers():
    bp = blueprint()
    bp["envVarGroups"][0]["envVars"].append({"key": "GROQ_API_KEY", "value": "gsk_x"})
    worker = next(s for s in bp["services"] if s["type"] == "worker")
    worker["plan"] = "free"
    worker["envVars"].append({"key": "X", "fromService": {"type": "pserv", "name": "missing"}})
    worker["buildCommand"] = "go build ./cmd/nope"
    errs = "\n".join(vr.repo_checks(bp))
    for want in (
        "secret GROQ_API_KEY",
        "no free plan",
        "missing pserv missing",
        "./cmd/nope missing",
    ):
        assert want in errs
