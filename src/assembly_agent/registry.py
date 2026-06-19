"""Register the running SDK as an agent's BYO LLM endpoint.

The AssemblyAI agent record carries an ``llm`` array; the voice layer calls
that ``base_url`` as a chat-completions endpoint, presenting the configured
``api_key``. This module upserts that record so the URL where your SDK runs
(a tunnel in dev, your host in prod) is what the voice layer dials.

Create vs update:
- No ``agent_id`` → ``POST /v1/agents`` creates a record.
- ``agent_id`` given → ``GET`` the record, swap in the ``llm`` block, and
  ``PUT /v1/agents/{id}``. We merge rather than send a bare body because PUT is
  a full replace — a minimal body would wipe the ``input`` / ``output`` /
  ``tools`` you configured.

Contract (per the agent_record PRs):
- ``base_url`` must be HTTPS with a public-DNS host (SSRF guard) — so
  ``http://localhost`` is rejected; use the tunnel or a deployed host.
- ``model`` and ``api_key`` must be non-empty.
- ``llm`` is an array capped at 1 entry in v1.
- ``api_key`` is envelope-encrypted at rest and never returned on reads.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_API_BASE = "https://agents.assemblyai.com/v1"

# Server-managed fields that a read returns but a write won't accept.
_NON_WRITABLE = {"id", "created_at", "updated_at", "request_id", "object"}


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


def _managed_fields(
    name: str,
    voice: str,
    greeting: Optional[str],
    system_prompt: Optional[str],
    extra: Optional[dict],
) -> dict:
    """The fields the SDK owns on the record (everything except the llm block)."""
    rec: dict[str, Any] = {"name": name, "voice": {"voice_id": voice}}
    if greeting:
        rec["greeting"] = greeting
    if system_prompt:
        rec["system_prompt"] = system_prompt
    if extra:
        rec.update(extra)
    return rec


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
    merge: bool = True,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 30.0,
) -> dict:
    """Create or update the agent record so its ``llm`` endpoint points at
    ``public_url`` with ``ingress_key`` as the shared secret. Returns the record."""
    import httpx

    if not assemblyai_api_key:
        raise RegistrationError("ASSEMBLYAI_API_KEY is required to register the agent.")
    if not model:
        raise RegistrationError("llm.model must be non-empty.")
    if not ingress_key:
        raise RegistrationError("llm.api_key must be non-empty.")

    base_url = normalize_base_url(public_url)
    llm_entry = {"base_url": base_url, "model": model, "api_key": ingress_key}
    headers = {"Authorization": assemblyai_api_key, "Content-Type": "application/json"}

    with httpx.Client(timeout=timeout) as client:
        if agent_id:
            # PUT is a full replace — start from the existing record so we don't
            # drop input/output/tools, then swap in our managed fields + llm.
            record: dict[str, Any] = {}
            if merge:
                got = client.get(f"{api_base}/agents/{agent_id}", headers=headers)
                if got.status_code < 400:
                    record = {k: v for k, v in got.json().items() if k not in _NON_WRITABLE}
            record.update(_managed_fields(name, voice, greeting, system_prompt, extra))
            record["llm"] = [llm_entry]
            resp = client.put(f"{api_base}/agents/{agent_id}", json=record, headers=headers)
        else:
            record = _managed_fields(name, voice, greeting, system_prompt, extra)
            record["llm"] = [llm_entry]
            resp = client.post(f"{api_base}/agents", json=record, headers=headers)

    if resp.status_code >= 400:
        raise RegistrationError(f"register failed {resp.status_code}: {resp.text}")
    return resp.json()
