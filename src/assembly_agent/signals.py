"""Read-only views over the audio-intelligence enrichment the voice layer
attaches to every turn.

These wrap plain dicts so the documented fields are reachable as attributes
(``ev.signals.emotion``) while unknown/forward-compatible fields stay available
via ``.get()`` / ``["key"]`` / ``.to_dict()``. Missing keys read back as
``None`` rather than raising, so handler code can branch on them directly:

    if ev.signals.emotion == "frustrated":
        ...
"""

from __future__ import annotations

from typing import Any, Iterator


class _Bag:
    """Attribute + mapping access over a dict, with ``None`` for misses.

    Nested dicts are wrapped on access, so ``ev.signals.prosody.pitch`` works
    without the caller knowing the shape ahead of time.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", dict(data or {}))

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        val = self._data.get(key)
        return _Bag(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        val = self._data.get(key)
        return _Bag(val) if isinstance(val, dict) else val

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


class Signals(_Bag):
    """Per-turn audio intelligence, precomputed by the voice layer.

    Documented fields: ``emotion``, ``sentiment``, ``prosody`` (a bag with
    ``pitch`` / ``energy`` / ``pace``), ``hesitance``, ``accent``,
    ``confidence``. Any field the voice layer adds later is reachable the same
    way.
    """


class Caller(_Bag):
    """Caller metadata: ``phone_number``, ``direction`` (inbound/outbound),
    ``from_`` / ``to``, etc. Present on ``on_call_start``."""

    @property
    def phone_number(self) -> Any:
        # Tolerate either `phone_number` or `from` as the source number.
        return self._data.get("phone_number") or self._data.get("from")


class Speaker(_Bag):
    """Diarization result: ``id`` and ``confidence``."""


class Turn(_Bag):
    """Turn-taking signals: ``interruption`` flag, ``overlap``, ``latency``."""
