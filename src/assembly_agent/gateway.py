"""The LLM Gateway as a native primitive.

AssemblyAI's LLM Gateway is itself an OpenAI-compatible endpoint — one key
across Claude / GPT / Gemini / Qwen / Kimi, with automatic retries and
fallbacks. Because the agent already authenticates with your AssemblyAI key,
the Gateway needs no extra credential: it reuses ``ASSEMBLYAI_API_KEY``.

In a handler it shows up as ``ctx.llm``:

    @agent.on_response
    async def respond(ev, ctx):
        return await ctx.llm.complete()        # answer the turn with the Gateway
        # or: return ctx.llm.stream()          # stream tokens to TTS

And if you register no ``on_response`` at all, the agent falls back to the
Gateway automatically (the "managed LLM" default) — so a useful voice agent is
just a name, a voice, and a model.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

US_BASE_URL = "https://llm-gateway.assemblyai.com/v1"
EU_BASE_URL = "https://llm-gateway.eu.assemblyai.com/v1"
DEFAULT_MODEL = "claude-sonnet-4-6"


class GatewayError(RuntimeError):
    """Raised when the Gateway returns a non-2xx response."""

    def __init__(self, status: int, body: str, request_id: Optional[str] = None) -> None:
        self.status = status
        self.body = body
        self.request_id = request_id
        msg = f"LLM Gateway {status}"
        if request_id:
            msg += f" (request_id={request_id})"
        msg += f": {body}"
        super().__init__(msg)


class Gateway:
    """Thin async client for the LLM Gateway's ``/chat/completions``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("LLM_GATEWAY_URL") or US_BASE_URL).rstrip("/")
        self.model = model
        self.last_request_id: Optional[str] = None
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _require_key(self) -> None:
        if not self.api_key:
            raise GatewayError(0, "No API key — set ASSEMBLYAI_API_KEY in the environment.")

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        return self._client

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, messages: list[dict], model: Optional[str], stream: bool, params: dict) -> dict:
        body: dict[str, Any] = {"model": model or self.model, "messages": messages, "stream": stream}
        body.update(params)
        return body

    async def complete(self, messages: list[dict], *, model: Optional[str] = None, **params: Any) -> str:
        """One request, full reply as a string."""
        self._require_key()
        resp = await self.client.post(
            self.base_url + "/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, model, False, params),
        )
        if resp.status_code >= 400:
            raise GatewayError(resp.status_code, resp.text, resp.headers.get("x-request-id"))
        data = resp.json()
        self.last_request_id = data.get("request_id") or resp.headers.get("x-request-id")
        return (data["choices"][0]["message"].get("content") or "") if data.get("choices") else ""

    async def stream(self, messages: list[dict], *, model: Optional[str] = None, **params: Any) -> AsyncIterator[str]:
        """Stream the reply token-by-token (for models that support it)."""
        self._require_key()
        async with self.client.stream(
            "POST",
            self.base_url + "/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, model, True, params),
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise GatewayError(resp.status_code, body, resp.headers.get("x-request-id"))
            self.last_request_id = resp.headers.get("x-request-id")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield piece

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class CallLLM:
    """Per-call view of the Gateway, bound to the running history and the
    agent's instructions (used as the system prompt). This is ``ctx.llm``."""

    def __init__(self, gateway: Gateway, history: list, system: Optional[str]) -> None:
        self._gateway = gateway
        self._history = history
        self._system = system

    def _messages(self) -> list[dict]:
        msgs: list[dict] = []
        has_system = any(getattr(m, "role", None) == "system" for m in self._history)
        if self._system and not has_system:
            msgs.append({"role": "system", "content": self._system})
        for m in self._history:
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role in ("system", "user", "assistant") and content:
                msgs.append({"role": role, "content": content if isinstance(content, str) else str(content)})
        return msgs

    async def complete(self, *, model: Optional[str] = None, **params: Any) -> str:
        return await self._gateway.complete(self._messages(), model=model, **params)

    def stream(self, *, model: Optional[str] = None, **params: Any) -> AsyncIterator[str]:
        return self._gateway.stream(self._messages(), model=model, **params)

    async def __call__(self, *, model: Optional[str] = None, **params: Any) -> str:
        return await self.complete(model=model, **params)
