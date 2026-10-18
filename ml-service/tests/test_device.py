import sys
import types

import pytest

from app.device import select_device, use_fp16


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
    ("mps", "cuda", "override", "want"),
    [
        (True, False, "", "mps"),
        (False, False, "", "cpu"),
        (False, True, "", "cuda"),
        (True, True, "", "cuda"),  # cuda > mps > cpu
        (True, False, "cpu", "cpu"),
        (False, True, "cpu", "cpu"),
        (False, False, "mps", "cpu"),  # requested but unavailable
        (False, False, "cuda", "cpu"),
        (True, True, "mps", "mps"),
        (True, False, "bogus", "mps"),  # invalid override ignored
    ],
)
def test_select_with_torch(monkeypatch, mps, cuda, override, want):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps, cuda))
    assert select_device(override) == want


def test_env_override(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True))
    monkeypatch.setenv("TORCH_DEVICE", "CPU")
    assert select_device() == "cpu"


def test_env_override_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, cuda=True))
    monkeypatch.setenv("TORCH_DEVICE", "cuda")
    assert select_device() == "cuda"


def test_fp16_only_on_accelerators():
    assert use_fp16("mps", True) and use_fp16("cuda", True)
    assert not use_fp16("cpu", True)
    assert not use_fp16("cuda", False)
