"""The ``Event`` object handed to every handler, plus the conventions for
pulling it out of an incoming OpenAI chat-completions request.

The voice layer rides the OpenAI schema and enriches each request with the
audio signals. By convention that enrichment lives under an ``assemblyai`` (or
``metadata``) key — either at the top level of the request body or attached to
the most recent user message. We look in both places so the SDK works no matter
which the caller picks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .signals import Caller, Signals, Speaker, Turn

# Event types dispatched to handlers.
CALL_START = "call_start"
RESPONSE = "response"
INTERRUPTION = "interruption"
SPEAKER_CHANGE = "speaker_change"
CALL_END = "call_end"

_ENRICHMENT_KEYS = ("assemblyai", "metadata")


@dataclass
class Event:
    """What a handler receives. Fields mirror the design doc's ``ev``."""

    type: str
    text: str = ""
    language: Optional[str] = None
    caller: Caller = field(default_factory=Caller)
    speaker: Speaker = field(default_factory=Speaker)
    signals: Signals = field(default_factory=Signals)
    turn: Turn = field(default_factory=Turn)
    model: str = ""
    raw: dict = field(default_factory=dict)  # the full enrichment blob


def _message_text(content: Any) -> str:
    """Flatten a message ``content`` to plain text.

    Voice turns are strings, but the OpenAI schema also allows a list of
    content parts; pull the text out of those so handlers always see a string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def extract_enrichment(body: dict) -> dict:
    """Find the AssemblyAI enrichment blob in a request body."""
    for key in _ENRICHMENT_KEYS:
        val = body.get(key)
        if isinstance(val, dict):
            return val
    # Fall back to the most recent user message.
    for msg in reversed(body.get("messages", []) or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            for key in _ENRICHMENT_KEYS:
                val = msg.get(key)
                if isinstance(val, dict):
                    return val
            break
    return {}


def last_user_text(body: dict) -> str:
    for msg in reversed(body.get("messages", []) or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _message_text(msg.get("content"))
    return ""


def has_user_turn(body: dict) -> bool:
    return any(
        isinstance(m, dict) and m.get("role") == "user" and _message_text(m.get("content")).strip()
        for m in (body.get("messages") or [])
    )


def infer_event_type(body: dict, enrichment: dict) -> str:
    """Pick the event type for a request.

    An explicit ``assemblyai.event`` wins. Otherwise: a request with no spoken
    user turn yet is the connect moment (``call_start``); anything else is a
    finalized user turn (``response``).
    """
    explicit = enrichment.get("event")
    if explicit:
        return explicit
    return RESPONSE if has_user_turn(body) else CALL_START


def build_event(body: dict, enrichment: dict, event_type: str) -> Event:
    return Event(
        type=event_type,
        text=last_user_text(body),
        language=enrichment.get("language"),
        caller=Caller(enrichment.get("caller") or {}),
        speaker=Speaker(enrichment.get("speaker") or {}),
        signals=Signals(enrichment.get("signals") or {}),
        turn=Turn(enrichment.get("turn") or {}),
        model=body.get("model", ""),
        raw=enrichment,
    )
