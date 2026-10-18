import importlib.util

import pytest

from app.settings import REPO_ROOT
from eval import runinfo

spec = importlib.util.spec_from_file_location("jobguard", REPO_ROOT / "scripts" / "jobguard.py")
jg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jg)

MEMINFO = """MemTotal:        8000000 kB
MemFree:          500000 kB
MemAvailable:    3000000 kB
SwapTotal:       2097152 kB
SwapFree:        1048576 kB
"""


def psi(some: float, full: float) -> str:
    return (
        f"some avg10={some:.2f} avg60=0.00 avg300=0.00 total=1\n"
        f"full avg10={full:.2f} avg60=0.00 avg300=0.00 total=1\n"
    )


def test_linux_memory_state_from_meminfo_and_psi():
    m = jg.linux_memory_state(MEMINFO, psi(0.5, 0.0))
    assert m["free_pct"] == 37  # MemAvailable / MemTotal, not MemFree
    assert m["swap_used_mb"] == 1024.0
    assert m["pressure_level"] == 1
    assert m["psi_avg10"] == {"some": 0.5, "full": 0.0}
    assert jg.memory_refusal(m, 20) is None


@pytest.mark.parametrize(
    ("some", "full", "level"),
    [(0.0, 0.0, 1), (9.99, 0.0, 1), (10.0, 0.0, 2), (40.0, 9.0, 2), (40.0, 10.0, 4)],
)
def test_psi_levels(some, full, level):
    assert jg.psi_level(jg.parse_psi(psi(some, full))) == level


def test_psi_warn_refuses():
    m = jg.linux_memory_state(MEMINFO, psi(25.0, 0.0))
    assert "warn" in jg.memory_refusal(m, 20)


def test_no_psi_falls_back_to_free_memory():
    m = jg.linux_memory_state(MEMINFO, "")
    assert m["pressure_level"] is None and m["psi_avg10"] is None
    assert jg.memory_refusal(m, 20) is None
    low = MEMINFO.replace("MemAvailable:    3000000", "MemAvailable:    1000000")
    assert "12% memory free" in jg.memory_refusal(jg.linux_memory_state(low, ""), 20)


def test_missing_meminfo_checks_nothing():
    m = jg.linux_memory_state("", "")
    assert m == {"pressure_level": None, "free_pct": None, "swap_used_mb": None, "psi_avg10": None}
    assert jg.memory_refusal(m, 20) is None


def test_linux_swap_usage_matches_macos_shape():
    out = runinfo._linux_swap_usage(MEMINFO)
    assert out == "total = 2048.00M  used = 1024.00M  free = 1024.00M"
    assert runinfo._linux_swap_usage("MemTotal: 1 kB\n") == ""
