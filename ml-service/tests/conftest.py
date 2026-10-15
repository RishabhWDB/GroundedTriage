"""Shared pytest config.

- ``slow`` tests (large model downloads/loads) run only when ``RUN_SLOW=1``.
- ``integration`` tests run only when ``TEST_DATABASE_URL`` is set.

The repo-root ``.env`` is loaded (without overriding real env vars) so tests see
the same config as ``make`` targets.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def pytest_collection_modifyitems(config, items):
    run_slow = os.environ.get("RUN_SLOW") == "1"
    has_db = bool(os.environ.get("TEST_DATABASE_URL"))
    skip_slow = pytest.mark.skip(reason="slow test; set RUN_SLOW=1 to run")
    skip_db = pytest.mark.skip(reason="TEST_DATABASE_URL is unset")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "integration" in item.keywords and not has_db:
            item.add_marker(skip_db)
