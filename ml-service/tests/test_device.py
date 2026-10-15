import sys
import types

import pytest

from app.device import select_device


def fake_torch(mps: bool, cuda: bool = False):
    mod = types.ModuleType("torch")
    mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return mod


def test_no_torch_returns_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert select_device("") == "cpu"
    assert select_device("mps") == "cpu"


@pytest.mark.parametrize(
    ("mps", "override", "want"),
    [
        (True, "", "mps"),
        (False, "", "cpu"),
        (True, "cpu", "cpu"),
        (False, "mps", "cpu"),  # requested but unavailable
        (True, "bogus", "mps"),  # invalid override ignored
    ],
)
def test_select_with_torch(monkeypatch, mps, override, want):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps))
    assert select_device(override) == want


def test_env_override(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True))
    monkeypatch.setenv("TORCH_DEVICE", "CPU")
    assert select_device() == "cpu"
