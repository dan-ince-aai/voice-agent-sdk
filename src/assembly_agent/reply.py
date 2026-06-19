"""``Reply`` — the back half of the loop.

A handler can return a plain string (spoken as-is) or a ``Reply`` to shape how
the line is rendered by TTS: tone, speed, and room to grow (pitch, emphasis,
pauses, a one-off voice swap). The text is what gets said; everything else is
delivery and rides back to the voice layer in the response's ``assemblyai``
extension field.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class Reply:
    text: str
    tone: Optional[str] = None
    speed: Optional[str] = None
    pitch: Optional[str] = None
    emphasis: Optional[str] = None
    pause: Optional[str] = None
    voice: Optional[str] = None  # render this one line in a different voice

    def delivery(self) -> dict:
        """The non-text controls, omitting anything left unset."""
        out = {}
        for f in fields(self):
            if f.name == "text":
                continue
            val = getattr(self, f.name)
            if val is not None:
                out[f.name] = val
        return out
