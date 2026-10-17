"""Meeting-request extraction (P4.4a): POST /extract_meeting.

The drafter LLM (gpt-oss-120b, Groq JSON mode, temperature 0) reads a
Meeting Scheduling ticket and returns the requested start, duration, venue
and timezone. Relative dates ("Thursday", "next week at 3") are resolved
against the ticket's received_at, which the prompt states explicitly. The
output is validated here; anything unusable becomes null rather than a
guess, and the Go scheduler (internal/scheduler) then proposes a slot.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from app.llm import LLM, JSONModeError, LLMError

logger = logging.getLogger(__name__)

MIN_MINUTES, MAX_MINUTES = 5, 480
VENUES = ("video", "phone", "in_person", "unspecified")

SYSTEM_PROMPT = (
    "You extract meeting requests from customer-support emails. Resolve relative dates and "
    "times against the email's received time given below. Never invent a time: if the "
    "sender does not propose one, set start to null. Output JSON only."
)
USER_TEMPLATE = """Email received at: {received_at} ({weekday}), default timezone {tz}.

Subject: {subject}

{body}

Return JSON with exactly these keys:
{{"is_meeting_request": true|false,
  "start": "YYYY-MM-DDTHH:MM" local time of the requested start, or null,
  "timezone": IANA timezone the sender means (e.g. "Europe/London"), or null for the default,
  "duration_minutes": integer, or null if not stated,
  "venue": "video" | "phone" | "in_person" | "unspecified",
  "venue_detail": short text such as "Zoom", "Teams link", an address, or null,
  "time_text": the exact phrase that states the time, or null,
  "action": "schedule" | "reschedule" | "confirm" | "cancel" | "other"}}"""


class MeetingRequest(BaseModel):
    ticket_id: str
    subject: str = ""
    body: str = ""
    received_at: datetime
    default_timezone: str = "UTC"


class MeetingExtraction(BaseModel):
    ticket_id: str
    is_meeting_request: bool
    # Requested start as an aware datetime (UTC offset included), or None.
    start: datetime | None = None
    timezone: str
    duration_minutes: int | None = Field(default=None, ge=MIN_MINUTES, le=MAX_MINUTES)
    venue: str = "unspecified"
    venue_detail: str | None = None
    time_text: str | None = None
    action: str = "other"
    # Why a field was dropped during validation (empty when all were usable).
    notes: list[str] = Field(default_factory=list)


def _zone(name: str | None, default: str) -> tuple[ZoneInfo, str, str | None]:
    """(zone, zone name, note): name if it is a valid IANA zone, else the
    default (with a note when a name was given), else UTC."""
    for candidate in (name, default, "UTC"):
        if not candidate:
            continue
        try:
            zone = ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError):
            continue
        note = f"unknown timezone {name!r}" if name and candidate != name else None
        return zone, candidate, note
    return ZoneInfo("UTC"), "UTC", None


def build_messages(req: MeetingRequest) -> list[dict[str, str]]:
    tz, tz_name, _ = _zone(req.default_timezone, "UTC")
    received = req.received_at.astimezone(tz)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                received_at=received.strftime("%Y-%m-%dT%H:%M"),
                weekday=received.strftime("%A"),
                tz=tz_name,
                subject=req.subject or "(none)",
                body=req.body,
            ),
        },
    ]


def validate(raw: dict, req: MeetingRequest) -> MeetingExtraction:
    notes: list[str] = []
    tz, tz_name, note = _zone(raw.get("timezone"), req.default_timezone)
    if note:
        notes.append(note)
    start = None
    if raw.get("start"):
        try:
            naive = datetime.fromisoformat(str(raw["start"]))
            start = naive.replace(tzinfo=tz) if naive.tzinfo is None else naive
            received = req.received_at
            if start < received - timedelta(hours=1):
                notes.append(f"start {raw['start']} is before the email was received; dropped")
                start = None
            elif start > received + timedelta(days=365):
                notes.append(f"start {raw['start']} is more than a year ahead; dropped")
                start = None
        except ValueError:
            notes.append(f"unparseable start {raw['start']!r}")
    duration = raw.get("duration_minutes")
    if duration is not None:
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            duration = None
        if duration is not None and not MIN_MINUTES <= duration <= MAX_MINUTES:
            notes.append(f"duration {duration} min outside {MIN_MINUTES}..{MAX_MINUTES}; dropped")
            duration = None
    venue = raw.get("venue") if raw.get("venue") in VENUES else "unspecified"
    action = raw.get("action") or "other"
    if action not in ("schedule", "reschedule", "confirm", "cancel", "other"):
        action = "other"
    return MeetingExtraction(
        ticket_id=req.ticket_id,
        is_meeting_request=bool(raw.get("is_meeting_request")),
        start=start,
        timezone=tz_name,
        duration_minutes=duration,
        venue=venue,
        venue_detail=(str(raw["venue_detail"])[:200] if raw.get("venue_detail") else None),
        time_text=(str(raw["time_text"])[:200] if raw.get("time_text") else None),
        action=action,
        notes=notes,
    )


def extract(req: MeetingRequest, llm: LLM) -> MeetingExtraction:
    """Raises LLMError when the LLM fails or returns non-JSON twice."""
    messages = build_messages(req)
    for attempt in range(2):
        try:
            text = llm.complete(messages, temperature=0.0, json_mode=True)
        except JSONModeError as exc:
            logger.warning("meeting extraction attempt %d: JSON mode failed: %s", attempt + 1, exc)
            continue
        try:
            raw = json.loads(text)
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            return validate(raw, req)
        except ValueError as exc:
            logger.warning("meeting extraction attempt %d: bad JSON: %s", attempt + 1, exc)
    raise LLMError("meeting extraction: the LLM returned invalid JSON twice")
