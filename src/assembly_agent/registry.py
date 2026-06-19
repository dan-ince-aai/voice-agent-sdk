"""Register the running SDK as an agent's BYO LLM endpoint.

The AssemblyAI agent record carries an ``llm`` array; the voice layer calls
that ``base_url`` as a chat-completions endpoint, presenting the configured
``api_key``. This module upserts that record so the URL where your SDK is
running (a tunnel in dev, your host in prod) is what the voice layer dials.

Contract (per the agent_record PRs):
- ``base_url`` must be HTTPS with a public-DNS host (SSRF guard) — so
  ``http://localhost`` is rejected; use the tunnel or a deployed host.
- ``model`` and ``api_key`` must be non-empty.
- ``llm`` is an array capped at 1 entry in v1.
- ``api_key`` is envelope-encrypted at rest and never returned on reads.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_API_BASE = "https://agents.assemblyai.com/v1"


class RegistrationError(RuntimeError):
    pass


def normalize_base_url(public_url: str) -> str:
    """Coerce to the ``…/v1`` base the voice layer appends
    ``/chat/completions`` to, and enforce the HTTPS / public-DNS rule."""
    base = public_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    parsed = urlparse(base)
    if parsed.scheme != "https":
        raise RegistrationError(
            f"base_url must be HTTPS, got {base!r}. The agent record rejects non-HTTPS "
            "and non-public hosts — use the public tunnel (serve()) or a deployed URL, "
            "not http://localhost."
        )
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
        raise RegistrationError(
            f"base_url host {host!r} is not public — the agent record requires a public-DNS host. "
            "Use the public tunnel (serve()) or a deployed URL."
        )
    return base


def build_record(
    *,
    name: str,
    voice: str,
    base_url: str,
    model: str,
    ingress_key: str,
    greeting: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    if not model:
        raise RegistrationError("llm.model must be non-empty.")
    if not ingress_key:
        raise RegistrationError("llm.api_key must be non-empty.")
    record: dict[str, Any] = {
        "name": name,
        "voice": {"voice_id": voice},
        "llm": [{"base_url": base_url, "model": model, "api_key": ingress_key}],
    }
    if greeting:
        record["greeting"] = greeting
    if system_prompt:
        record["system_prompt"] = system_prompt
    if extra:
        record.update(extra)
    return record


def register_agent(
    *,
    name: str,
    voice: str,
    public_url: str,
    model: str,
    ingress_key: str,
    assemblyai_api_key: str,
    agent_id: Optional[str] = None,
    greeting: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra: Optional[dict] = None,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 30.0,
) -> dict:
    """Create (POST) or update (PUT, when ``agent_id`` is given) the agent
    record so its ``llm`` endpoint points at ``public_url``. Returns the record."""
    import httpx

    if not assemblyai_api_key:
        raise RegistrationError("ASSEMBLYAI_API_KEY is required to register the agent.")

    base_url = normalize_base_url(public_url)
    record = build_record(
        name=name,
        voice=voice,
        base_url=base_url,
        model=model,
        ingress_key=ingress_key,
        greeting=greeting,
        system_prompt=system_prompt,
        extra=extra,
    )
    headers = {"Authorization": assemblyai_api_key, "Content-Type": "application/json"}

    with httpx.Client(timeout=timeout) as client:
        if agent_id:
            resp = client.put(f"{api_base}/agents/{agent_id}", json=record, headers=headers)
        else:
            resp = client.post(f"{api_base}/agents", json=record, headers=headers)

    if resp.status_code >= 400:
        raise RegistrationError(f"register failed {resp.status_code}: {resp.text}")
    return resp.json()
