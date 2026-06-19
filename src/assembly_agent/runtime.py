"""The translation layer: an incoming OpenAI chat-completions request goes in,
your handler runs, and an OpenAI-shaped response (or SSE token stream) comes
out.

Delivery controls (``Reply``), routing (``Transfer``), and the augment/
passthrough signal (returning ``None``) all ride back in an ``assemblyai``
extension field on the choice, alongside a standard OpenAI body so any client
that ignores the extension still gets valid output.
"""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Any, AsyncIterator, Optional

from . import events as ev_mod
from .context import Context, Message, Transfer
from .events import build_event, extract_enrichment, infer_event_type
from .reply import Reply


def _now() -> int:
    return int(time.time())


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class Outcome:
    """Normalized result of a handler, ready to render as OpenAI output."""

    def __init__(
        self,
        *,
        text: str = "",
        token_iter: Optional[AsyncIterator[str]] = None,
        delivery: Optional[dict] = None,
        extension: Optional[dict] = None,
        finish_reason: str = "stop",
    ) -> None:
        self.text = text
        self.token_iter = token_iter
        self.delivery = delivery
        self.extension = extension  # full choice-level `assemblyai` blob
        self.finish_reason = finish_reason

    @property
    def streaming(self) -> bool:
        return self.token_iter is not None

    def choice_extension(self) -> Optional[dict]:
        if self.extension is not None:
            return self.extension
        if self.delivery:
            return {"delivery": self.delivery}
        return None


class Runtime:
    def __init__(self, agent: "Any") -> None:
        self.agent = agent

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    async def dispatch(self, body: dict, headers: Optional[dict] = None):
        """Run the right handler for a request body.

        Returns either a dict (non-streaming completion) or an async iterator
        of SSE strings (streaming).
        """
        headers = headers or {}
        enrichment = extract_enrichment(body)
        call_id = (
            enrichment.get("call_id")
            or headers.get("x-call-id")
            or headers.get("x-assemblyai-call-id")
            or "default"
        )
        event_type = infer_event_type(body, enrichment)
        event = build_event(body, enrichment, event_type)

        store = self.agent.store_for(call_id)
        history = [Message(m if isinstance(m, dict) else dict(m)) for m in (body.get("messages") or [])]
        ctx = Context(call_id, store, history, self.agent)

        handler = self.agent.handler_for(event_type)
        model = body.get("model") or self.agent.model
        stream = bool(body.get("stream"))

        # Side-channel events (interruption / speaker_change / call_end) don't
        # produce speech; run the handler if present and return an empty body.
        if event_type in (ev_mod.INTERRUPTION, ev_mod.SPEAKER_CHANGE, ev_mod.CALL_END):
            if handler is not None:
                await _maybe_await(handler(event, ctx))
            if event_type == ev_mod.CALL_END:
                self.agent.drop_call(call_id)
            outcome = Outcome(text="", finish_reason="stop")
            return await self._render(outcome, model=model, stream=stream)

        # call_start / response produce a spoken turn.
        result = None
        if handler is not None:
            result = await _maybe_await(handler(event, ctx))
        elif event_type == ev_mod.CALL_START and self.agent.greeting:
            result = self.agent.greeting
        elif event_type == ev_mod.RESPONSE and self.agent.has_gateway():
            # "Managed LLM" default: no on_response handler, but a Gateway key
            # is configured — answer the turn through the Gateway.
            result = await self._augment(ctx)

        outcome = self._normalize(result)
        return await self._render(outcome, model=model, stream=stream)

    async def _augment(self, ctx) -> Any:
        """Proxy a turn to the LLM Gateway. Falls back to passthrough (let the
        voice layer's managed LLM handle it) if the Gateway errors."""
        try:
            return await ctx.llm.complete()
        except Exception as exc:  # noqa: BLE001
            import sys

            print(f"[assembly_agent] Gateway augment failed, passing through: {exc}", file=sys.stderr)
            return None

    # ------------------------------------------------------------------ #
    # normalize a handler return value
    # ------------------------------------------------------------------ #
    def _normalize(self, result: Any) -> Outcome:
        # None -> augment / passthrough: let a managed LLM handle this turn.
        if result is None:
            return Outcome(text="", extension={"action": "passthrough"}, finish_reason="stop")

        if isinstance(result, Transfer):
            return Outcome(text="", extension=result.to_dict(), finish_reason="stop")

        if isinstance(result, Reply):
            return Outcome(text=result.text, delivery=result.delivery() or None)

        if isinstance(result, str):
            return Outcome(text=result)

        # Streaming: async generator or sync generator of token strings.
        if inspect.isasyncgen(result):
            return Outcome(token_iter=self._wrap_async(result))
        if inspect.isgenerator(result):
            return Outcome(token_iter=self._wrap_sync(result))

        # Anything else: stringify (best effort).
        return Outcome(text=str(result))

    async def _wrap_async(self, gen: AsyncIterator[Any]) -> AsyncIterator[str]:
        async for tok in gen:
            yield self._token_text(tok)

    async def _wrap_sync(self, gen) -> AsyncIterator[str]:
        for tok in gen:
            yield self._token_text(tok)

    @staticmethod
    def _token_text(tok: Any) -> str:
        if isinstance(tok, Reply):
            return tok.text
        return tok if isinstance(tok, str) else str(tok)

    # ------------------------------------------------------------------ #
    # render to OpenAI shape
    # ------------------------------------------------------------------ #
    async def _render(self, outcome: Outcome, *, model: str, stream: bool):
        if stream:
            # Returns the async-generator object (not awaited) for the server
            # to iterate as SSE.
            return self._render_stream(outcome, model=model)
        return await self._render_full(outcome, model=model)

    async def _render_full(self, outcome: Outcome, *, model: str) -> dict:
        if outcome.streaming:
            # Caller wants a single body but the handler streamed: collect it.
            parts = []
            async for tok in outcome.token_iter:  # type: ignore[union-attr]
                parts.append(tok)
            outcome = Outcome(
                text="".join(parts),
                delivery=outcome.delivery,
                extension=outcome.extension,
                finish_reason=outcome.finish_reason,
            )

        message: dict[str, Any] = {"role": "assistant", "content": outcome.text}
        ext = outcome.choice_extension()
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": outcome.finish_reason,
        }
        if ext is not None:
            choice["assemblyai"] = ext
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": _now(),
            "model": model,
            "choices": [choice],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _render_stream(self, outcome: Outcome, *, model: str) -> AsyncIterator[str]:
        cid = _completion_id()
        created = _now()
        ext = outcome.choice_extension()

        def chunk(delta: dict, finish_reason: Optional[str] = None, assemblyai: Optional[dict] = None) -> str:
            choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            if assemblyai is not None:
                choice["assemblyai"] = assemblyai
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
            }
            import json

            return f"data: {json.dumps(payload)}\n\n"

        # Opening chunk establishes the assistant role (and carries the
        # extension up front so the voice layer can shape delivery from token 0).
        yield chunk({"role": "assistant", "content": ""}, assemblyai=ext)

        if outcome.streaming:
            async for tok in outcome.token_iter:  # type: ignore[union-attr]
                if tok:
                    yield chunk({"content": tok})
        elif outcome.text:
            yield chunk({"content": outcome.text})

        yield chunk({}, finish_reason=outcome.finish_reason, assemblyai=ext)
        yield "data: [DONE]\n\n"
