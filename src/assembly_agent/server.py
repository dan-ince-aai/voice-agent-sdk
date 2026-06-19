"""ASGI app exposing the agent as an OpenAI-compatible server.

Built on Starlette (no heavier framework needed — this is just routes, JSON,
and SSE).

Routes:
- ``POST /v1/chat/completions`` — the seam the voice layer talks to.
- ``GET  /v1/models``           — advertises this agent as a model.
- ``GET  /healthz``             — liveness.
"""

from __future__ import annotations

import hmac
import time
from typing import TYPE_CHECKING, AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from .agent import Agent


def _authorized(agent: "Agent", request: Request) -> bool:
    """When an ingress key is configured, the caller (the voice layer) must
    present it — this is the ``api_key`` stored in the agent record's ``llm``
    config. No key configured → open (dev convenience)."""
    key = agent.ingress_key
    if not key:
        return True
    got = request.headers.get("authorization", "")
    if got.startswith("Bearer "):
        got = got[len("Bearer "):]
    return hmac.compare_digest(got.strip(), key)


def create_app(agent: "Agent") -> Starlette:
    async def chat_completions(request: Request):
        if not _authorized(agent, request):
            return JSONResponse(
                {"error": {"message": "Unauthorized", "type": "invalid_request_error",
                           "code": "invalid_api_key"}},
                status_code=401,
            )
        body = await request.json()
        headers = {k.lower(): v for k, v in request.headers.items()}
        result = await agent.runtime.dispatch(body, headers)

        if isinstance(result, dict):
            return JSONResponse(result)

        # Streaming branch: `result` is an async iterator of SSE strings.
        async def event_stream() -> AsyncIterator[bytes]:
            async for chunk in result:
                yield chunk.encode("utf-8")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def models(request: Request):
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": agent.model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "assemblyai-agent-sdk",
                    }
                ],
            }
        )

    async def healthz(request: Request):
        return JSONResponse({"status": "ok", "agent": agent.name})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/healthz", healthz, methods=["GET"]),
        ]
    )
