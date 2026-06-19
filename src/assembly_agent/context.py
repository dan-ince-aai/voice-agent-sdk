"""Per-call context and the ``Transfer`` signal.

``ctx`` lives for the whole call. The OpenAI chat-completions protocol is
stateless — each request carries the full message history — so anything you
``ctx.set()`` is keyed on the call id and held server-side between turns, then
dropped on ``on_call_end``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .signals import _Bag

if TYPE_CHECKING:
    from .agent import Agent


class Transfer:
    """Returned by ``ctx.transfer(...)`` to route the call to another agent."""

    def __init__(self, agent_name: str, reason: Optional[str] = None) -> None:
        self.agent_name = agent_name
        self.reason = reason

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"action": "transfer", "transfer_to": self.agent_name}
        if self.reason:
            out["reason"] = self.reason
        return out

    def __repr__(self) -> str:
        return f"Transfer(agent_name={self.agent_name!r}, reason={self.reason!r})"


class Message(_Bag):
    """One entry in ``ctx.history`` — ``.role`` and ``.content`` (plus any
    extra fields the request carried)."""


class Context:
    def __init__(self, call_id: str, store: dict, history: list[Message], agent: "Agent") -> None:
        self.call_id = call_id
        self.history = history
        self._store = store
        self._agent = agent
        self._cancelled = False
        self._llm = None

    @property
    def llm(self):
        """The LLM Gateway, bound to this call's history and the agent's
        instructions. ``await ctx.llm.complete()`` or ``ctx.llm.stream()``."""
        if self._llm is None:
            from .gateway import CallLLM

            self._llm = CallLLM(self._agent.gateway, self.history, self._agent.instructions)
        return self._llm

    # --- key/value store, lives for the whole call ---
    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    # --- routing & cancellation ---
    def transfer(self, agent_name: str, reason: Optional[str] = None) -> Transfer:
        return Transfer(agent_name, reason)

    def cancel_pending(self) -> None:
        """Mark in-flight work for this call as cancelled. Handlers can check
        ``ctx.cancelled`` between steps to bail out early (usually on a
        barge-in)."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # --- tools ---
    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Run a registered ``@agent.tool`` and return its result."""
        return await self._agent.run_tool(name, kwargs)
