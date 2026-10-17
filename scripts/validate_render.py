"""Validate render.yaml (P8.2): Render's published JSON schema + repo checks.

    ml-service/.venv/bin/python scripts/validate_render.py   # == make render-validate

Downloads https://render.com/schema/render.yaml.json (the schema Render's
dashboard and CLI use) and validates the blueprint against it, then checks
what the schema can't: every fromService target exists with the right type,
no secret-looking key has an inline value, and every build/start command
refers to a path that exists in the repo.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_URL = "https://render.com/schema/render.yaml.json"
SECRET_KEYS = {
    "DATABASE_URL",
    "GROQ_API_KEY",
    "API_KEY",
    "JUDGE_API_KEY",
    "MODELS_SHA256",
    "MODELS_URL",
}


def repo_checks(bp: dict) -> list[str]:
    errs = []
    services = {s["name"]: s for s in bp.get("services", [])}
    groups = {g["name"] for g in bp.get("envVarGroups", [])}
    env_lists = [("group " + g["name"], g["envVars"]) for g in bp.get("envVarGroups", [])]
    env_lists += [("service " + n, s.get("envVars", [])) for n, s in services.items()]
    for where, envs in env_lists:
        for e in envs:
            if "fromService" in e:
                ref = e["fromService"]
                if ref["name"] not in services or services[ref["name"]]["type"] != ref["type"]:
                    errs.append(
                        f"{where}: {e['key']} references missing {ref['type']} {ref['name']}"
                    )
            if "fromGroup" in e and e["fromGroup"] not in groups:
                errs.append(f"{where}: unknown group {e['fromGroup']}")
            if e.get("key") in SECRET_KEYS and "value" in e:
                errs.append(f"{where}: secret {e['key']} has an inline value")
    for n, s in services.items():
        root = ROOT / s.get("rootDir", "")
        if not root.is_dir():
            errs.append(f"{n}: rootDir {root} missing")
        for cmd in (s.get("buildCommand", ""), s.get("startCommand", "")):
            for tok in cmd.replace("&&", " ").split():
                if tok.startswith("./cmd/") and not (root / tok).is_dir():
                    errs.append(f"{n}: {tok} missing")
        if s["type"] in ("worker", "pserv") and s.get("plan") == "free":
            errs.append(f"{n}: Render has no free plan for {s['type']} services")
    return errs


def main() -> int:
    bp = yaml.safe_load((ROOT / "render.yaml").read_text())
    with urllib.request.urlopen(SCHEMA_URL, timeout=30) as r:
        schema = json.load(r)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(bp), key=lambda e: list(e.path)
    )
    for e in errors:
        print(f"schema: {'/'.join(map(str, e.path))}: {e.message[:200]}")
    extra = repo_checks(bp)
    for e in extra:
        print(f"repo: {e}")
    if errors or extra:
        return 1
    print(
        f"render.yaml valid: {len(bp['services'])} services against {SCHEMA_URL}; "
        "repo checks passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
