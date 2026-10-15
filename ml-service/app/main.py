"""ml-service FastAPI app. Endpoints beyond /health arrive with their parts
(/classify P3.4, /retrieve P5.2, /draft P5.3, /verify P5.4)."""

import importlib.util
import logging

from fastapi import FastAPI

from app.schemas import Health, ModelsLoaded

VERSION = "0.1.0"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = FastAPI(title="GroundedTriage ml-service", version=VERSION)


@app.get("/health", response_model=Health)
def health() -> Health:
    # Model loaders land in P3.1-P3.3; until then nothing is loaded. torch is
    # only probed with find_spec so /health never imports it.
    return Health(
        status="ok",
        service="ml-service",
        version=VERSION,
        models_loaded=ModelsLoaded(deep_tier=False, urgency=False),
        fast_tier_loaded=False,
        torch_installed=importlib.util.find_spec("torch") is not None,
    )
