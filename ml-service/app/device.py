"""Torch device selection: MPS when available, else CPU; overridable via TORCH_DEVICE.

torch is imported lazily and is optional: without it, "cpu" is returned with a
logged explanation, so the service runs before any model-dependent part adds torch.
"""

import logging
import os

logger = logging.getLogger(__name__)

VALID = {"cpu", "mps", "cuda"}


def select_device(override: str | None = None) -> str:
    want = (override if override is not None else os.environ.get("TORCH_DEVICE", "")).strip()
    want = want.lower()
    if want and want not in VALID:
        logger.warning("TORCH_DEVICE=%r is not one of %s; ignoring", want, sorted(VALID))
        want = ""

    try:
        import torch
    except ImportError:
        logger.info("torch is not installed; using cpu (torch is added by the first model part)")
        return "cpu"

    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if want:
        if available[want]:
            return want
        logger.warning("TORCH_DEVICE=%s requested but not available; using cpu", want)
        return "cpu"
    return "mps" if available["mps"] else "cpu"
