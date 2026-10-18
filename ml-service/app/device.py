"""Torch device selection: CUDA, then MPS, then CPU; overridable via TORCH_DEVICE.

torch is imported lazily and is optional: without it, "cpu" is returned with a
logged explanation, so the service runs before any model-dependent part adds torch.
"""

import logging
import os

logger = logging.getLogger(__name__)

VALID = {"cpu", "mps", "cuda"}
# Devices where MODEL_FP16 applies; CPU always runs fp32.
HALF_DEVICES = {"mps", "cuda"}


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
    for dev in ("cuda", "mps"):
        if available[dev]:
            return dev
    return "cpu"


def use_fp16(device: str, fp16: bool) -> bool:
    return fp16 and device in HALF_DEVICES


def empty_cache() -> None:
    """Release cached accelerator memory after a model is dropped (no-op without torch)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
