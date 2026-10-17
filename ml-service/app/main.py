"""ml-service FastAPI app. Endpoints arrive with their parts
(/classify P3.4, /extract_meeting P4.4, /retrieve P5.2, /draft P5.3, /verify P5.4)."""

import importlib.util
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from app.interfaces import TicketText
from app.llm import LLMError
from app.meeting import MeetingExtraction, MeetingRequest, extract
from app.models import Registry, get_registry
from app.schemas import Classification, ClassifyRequest, Health, ModelsLoaded

VERSION = "0.1.0"
RegistryDep = Annotated[Registry, Depends(get_registry)]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = FastAPI(title="GroundedTriage ml-service", version=VERSION)


@app.get("/health", response_model=Health)
def health(reg: RegistryDep) -> Health:
    # Never loads weights: deep/urgency report whether a valid artifact is
    # present (file checks); fast_tier_loaded is true once /classify loaded it.
    # torch is only probed with find_spec so /health never imports it.
    return Health(
        status="ok",
        service="ml-service",
        version=VERSION,
        models_loaded=ModelsLoaded(deep_tier=reg.deep.available, urgency=reg.urgency.available),
        fast_tier_loaded=reg.fast_loaded,
        torch_installed=importlib.util.find_spec("torch") is not None,
    )


@app.post("/classify", response_model=Classification)
def classify(req: ClassifyRequest, reg: RegistryDep) -> Classification:
    cascade = reg.cascade()
    if cascade is None:
        raise HTTPException(status_code=503, detail=reg.fast_reason)
    ticket = TicketText(id=req.ticket_id, subject=req.subject, body=req.body)
    return cascade.classify_tickets([ticket])[0]


@app.post("/extract_meeting", response_model=MeetingExtraction)
def extract_meeting(req: MeetingRequest, reg: RegistryDep) -> MeetingExtraction:
    llm = reg.llm()
    if llm is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set")
    try:
        return extract(req, llm)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"meeting extraction failed: {exc}") from exc
