import importlib

import pytest


@pytest.mark.parametrize("pkg", ["app", "triage", "rag", "baseline", "data", "eval"])
def test_packages_import(pkg):
    assert importlib.import_module(pkg) is not None
