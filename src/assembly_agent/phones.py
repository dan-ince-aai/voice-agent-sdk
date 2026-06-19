"""Phone-number provisioning client.

Thin wrapper over ``/v1/phone-numbers`` on ``https://agents.assemblyai.com``.
Mirrors ``registry.py``: plain functions, raw API-key auth (no ``Bearer``),
raise on non-2xx. The ergonomic surface is the ``Agent`` methods that wrap
these and fill in the agent id.

Unlike registration, **buying a number costs money and persists** — so the SDK
never provisions implicitly. You call these explicitly.

Four ways to get a number onto an agent:
- ``buy_number(..., agent_id=…)`` — search + buy + assign in one shot (Option A).
- ``purchase_number(number)`` — buy a specific known number (Option B), then assign.
- ``import_number(number, termination_uri)`` — bring your own trunk (Option C), then assign.
- ``assign_number(number, agent_id)`` — bind an owned number to an agent (Option D).
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_API_BASE = "https://agents.assemblyai.com/v1"


class PhoneError(RuntimeError):
    pass


def _request(
    method: str,
    url: str,
    *,
    api_key: str,
    json: Optional[dict] = None,
    idempotency_key: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    import httpx

    if not api_key:
        raise PhoneError("ASSEMBLYAI_API_KEY is required for phone-number operations.")
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    with httpx.Client(timeout=timeout) as client:
        resp = client.request(method, url, json=json, headers=headers)

    if resp.status_code >= 400:
        raise PhoneError(f"{method} {url} failed {resp.status_code}: {resp.text}")
    # 201/204 with empty bodies are normal for several of these endpoints.
    if not resp.content:
        return {}
    try:
        return resp.json()
    except Exception:
        return {}


def buy_number(
    *,
    assemblyai_api_key: str,
    country_code: str = "US",
    number_type: str = "local",
    area_code: Optional[int] = None,
    locality: Optional[str] = None,
    label: Optional[str] = None,
    agent_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 30.0,
) -> dict:
    """Option A — search the provider, buy the first match, optionally assign."""
    body: dict[str, Any] = {"country_code": country_code, "number_type": number_type}
    if area_code is not None:
        body["area_code"] = area_code
    if locality:
        body["locality"] = locality
    if label:
        body["label"] = label
    if agent_id:
        body["agent_id"] = agent_id
    return _request("POST", f"{api_base}/phone-numbers", api_key=assemblyai_api_key,
                    json=body, idempotency_key=idempotency_key, timeout=timeout)


def purchase_number(phone_number: str, *, assemblyai_api_key: str,
                    api_base: str = DEFAULT_API_BASE, timeout: float = 30.0) -> dict:
    """Option B — buy a specific known number (no assignment)."""
    return _request("POST", f"{api_base}/phone-numbers/purchase", api_key=assemblyai_api_key,
                    json={"phone_number": phone_number}, timeout=timeout)


def import_number(phone_number: str, termination_uri: str, *, assemblyai_api_key: str,
                  api_base: str = DEFAULT_API_BASE, timeout: float = 30.0) -> dict:
    """Option C — import a number you already own (BYO trunk)."""
    return _request("POST", f"{api_base}/phone-numbers/import", api_key=assemblyai_api_key,
                    json={"phone_number": phone_number, "termination_uri": termination_uri},
                    timeout=timeout)


def assign_number(phone_number: str, agent_id: str, *, assemblyai_api_key: str,
                  api_base: str = DEFAULT_API_BASE, timeout: float = 30.0) -> dict:
    """Option D — bind an owned number to an agent (also re-assigns)."""
    return _request("PUT", f"{api_base}/phone-numbers/{phone_number}/agent",
                    api_key=assemblyai_api_key, json={"agent_id": agent_id}, timeout=timeout)


def unassign_number(phone_number: str, *, assemblyai_api_key: str,
                    api_base: str = DEFAULT_API_BASE, timeout: float = 30.0) -> dict:
    return _request("DELETE", f"{api_base}/phone-numbers/{phone_number}/agent",
                    api_key=assemblyai_api_key, timeout=timeout)


def list_numbers(*, assemblyai_api_key: str, limit: int = 20, cursor: Optional[str] = None,
                 api_base: str = DEFAULT_API_BASE, timeout: float = 30.0) -> dict:
    url = f"{api_base}/phone-numbers?limit={limit}"
    if cursor:
        url += f"&cursor={cursor}"
    return _request("GET", url, api_key=assemblyai_api_key, timeout=timeout)


def get_number(phone_number: str, *, assemblyai_api_key: str,
               api_base: str = DEFAULT_API_BASE, timeout: float = 30.0) -> dict:
    return _request("GET", f"{api_base}/phone-numbers/{phone_number}",
                    api_key=assemblyai_api_key, timeout=timeout)


def release_number(phone_number: str, *, assemblyai_api_key: str,
                   api_base: str = DEFAULT_API_BASE, timeout: float = 30.0) -> dict:
    return _request("DELETE", f"{api_base}/phone-numbers/{phone_number}",
                    api_key=assemblyai_api_key, timeout=timeout)
