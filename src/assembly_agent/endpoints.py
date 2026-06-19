"""Region → endpoint mapping. One ``region`` switches both the agents REST API
and the LLM Gateway together."""

from __future__ import annotations

_AGENTS = {
    "us": "https://agents.assemblyai.com/v1",
    "eu": "https://agents.eu.assemblyai.com/v1",
}
_LLM = {
    "us": "https://llm-gateway.assemblyai.com/v1",
    "eu": "https://llm-gateway.eu.assemblyai.com/v1",
}


def _norm(region: str | None) -> str:
    r = (region or "us").lower()
    return r if r in _AGENTS else "us"


def agents_base(region: str | None) -> str:
    return _AGENTS[_norm(region)]


def llm_base(region: str | None) -> str:
    return _LLM[_norm(region)]
