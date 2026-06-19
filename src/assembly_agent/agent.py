"""The ``Agent`` — config + handler registry in one object.

    from assembly_agent import Agent, Reply

    agent = Agent(name="Support Assistant", voice="ivy")

    @agent.on_response
    async def respond(ev, ctx):
        return "Hello"

    agent.serve()

``serve()`` stands up an OpenAI-compatible chat-completions server. Point your
voice agent's LLM ``base_url`` at it and your handlers generate every turn —
with an LLM, a decision tree, a lookup table, whatever you write.
"""

from __future__ import annotations

import os
import secrets
import threading
from typing import Any, Callable, Optional

from . import events as ev_mod
from .reply import Reply
from .runtime import Runtime
from .tools import Tool, build_tool

__all__ = ["Agent", "Reply"]

_DEFAULT_MODEL = "assemblyai-agent"
_PLAYGROUND_URL = "https://www.assemblyai.com/dashboard/playground/voice-agent"


class Agent:
    def __init__(
        self,
        name: str,
        *,
        voice: str = "ivy",
        greeting: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        ingress_key: Optional[str] = None,
    ) -> None:
        self.name = name
        self.voice = voice
        self.greeting = greeting
        self.prompt = prompt
        # Stable id for upsert; falls back to a slug of the name.
        self.agent_id = agent_id or _slug(name)
        self.model = model or self.agent_id or _DEFAULT_MODEL

        # Ingress secret: the `api_key` the voice layer presents when calling
        # this SDK as the agent's BYO LLM endpoint. When set, the server rejects
        # requests without it. An explicit key (constructor or env) is kept as-is;
        # otherwise a fresh one is minted on each registration.
        provided = ingress_key or os.environ.get("ASSEMBLY_AGENT_INGRESS_KEY")
        self.ingress_key = provided
        self._ingress_explicit = provided is not None
        self.remote_agent_id: Optional[str] = None

        # LLM Gateway connection only — auth (ASSEMBLYAI_API_KEY by default) and
        # region. The *model* is chosen per request in ctx.llm.complete(model=…),
        # not here: agent config is identity/senses, not response generation.
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self._gateway = None

        self._handlers: dict[str, Callable] = {}
        self._tools: dict[str, Tool] = {}
        self._stores: dict[str, dict] = {}
        self.runtime = Runtime(self)
        self._app = None

    @property
    def gateway(self):
        """Lazily-built LLM Gateway client (see ``ctx.llm``)."""
        if self._gateway is None:
            from .gateway import Gateway

            self._gateway = Gateway(api_key=self.api_key, base_url=self.llm_base_url)
        return self._gateway

    def has_gateway(self) -> bool:
        """True when a Gateway key is available (env or constructor)."""
        return self.gateway.configured

    # ------------------------------------------------------------------ #
    # handler registration (decorators)
    # ------------------------------------------------------------------ #
    def _register(self, event_type: str, fn: Callable) -> Callable:
        self._handlers[event_type] = fn
        return fn

    def on_call_start(self, fn: Callable) -> Callable:
        return self._register(ev_mod.CALL_START, fn)

    def on_response(self, fn: Callable) -> Callable:
        return self._register(ev_mod.RESPONSE, fn)

    def on_interrupt(self, fn: Callable) -> Callable:
        return self._register(ev_mod.INTERRUPTION, fn)

    # `on_interruption` kept as an alias for the longer name.
    def on_interruption(self, fn: Callable) -> Callable:
        return self.on_interrupt(fn)

    def on_speaker_change(self, fn: Callable) -> Callable:
        return self._register(ev_mod.SPEAKER_CHANGE, fn)

    def on_call_end(self, fn: Callable) -> Callable:
        return self._register(ev_mod.CALL_END, fn)

    def handler_for(self, event_type: str) -> Optional[Callable]:
        return self._handlers.get(event_type)

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def tool(self, fn: Callable) -> Callable:
        """Register a function as a tool. Schema is inferred from the
        signature; the first paragraph of the docstring is the description."""
        t = build_tool(fn)
        self._tools[t.name] = t
        return fn

    @property
    def tools(self) -> dict[str, Tool]:
        return self._tools

    def tool_schemas(self) -> list[dict]:
        """All registered tools in OpenAI ``tools`` format."""
        return [t.openai_schema() for t in self._tools.values()]

    async def run_tool(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            raise KeyError(f"No tool registered named {name!r}")
        return await self._tools[name].run(args)

    # ------------------------------------------------------------------ #
    # per-call state
    # ------------------------------------------------------------------ #
    def store_for(self, call_id: str) -> dict:
        return self._stores.setdefault(call_id, {})

    def drop_call(self, call_id: str) -> None:
        self._stores.pop(call_id, None)

    # ------------------------------------------------------------------ #
    # serving
    # ------------------------------------------------------------------ #
    @property
    def app(self):
        """The ASGI (Starlette) app. Build lazily so importing the SDK doesn't
        require a server runtime until you serve."""
        if self._app is None:
            from .server import create_app

            self._app = create_app(self)
        return self._app

    def register(
        self,
        public_url: str,
        *,
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        assemblyai_api_key: Optional[str] = None,
        extra: Optional[dict] = None,
        rotate: bool = True,
    ) -> dict:
        """Point an AssemblyAI agent record's BYO LLM endpoint at ``public_url``
        (the SDK), and return the record.

        ``public_url`` must be HTTPS/public-DNS (a tunnel or deployed host —
        localhost is rejected). Unless you set an explicit ``ingress_key``, a
        fresh shared secret is minted here (``rotate=True``) and written to the
        record as the LLM ``api_key`` — a new key per registration.

        The record is identified by **name**: if an agent with this name already
        exists it's updated (PUT), otherwise one is created (POST). Pass
        ``agent_id`` to target a specific record instead."""
        from .registry import register_agent

        if self._ingress_explicit:
            pass  # respect the user's key, never rotate it
        elif rotate or not self.ingress_key:
            self.ingress_key = secrets.token_urlsafe(24)

        key = assemblyai_api_key or self.api_key or os.environ.get("ASSEMBLYAI_API_KEY", "")
        record = register_agent(
            name=self.name,
            voice=self.voice,
            public_url=public_url,
            model=model or self.model,
            ingress_key=self.ingress_key,
            assemblyai_api_key=key,
            agent_id=agent_id or self.remote_agent_id,
            greeting=self.greeting,
            system_prompt=system_prompt,
            extra=extra,
        )
        self.remote_agent_id = record.get("id") or agent_id or self.remote_agent_id
        return record

    # ------------------------------------------------------------------ #
    # phone numbers (control-plane; buying is always explicit)
    # ------------------------------------------------------------------ #
    def _aai_key(self, override: Optional[str]) -> str:
        return override or self.api_key or os.environ.get("ASSEMBLYAI_API_KEY", "")

    def _require_remote_id(self, agent_id: Optional[str]) -> str:
        aid = agent_id or self.remote_agent_id
        if not aid:
            raise RuntimeError(
                "This agent has no record id yet — register it first (serve() or "
                "agent.register(url)), or pass agent_id=."
            )
        return aid

    def buy_phone_number(
        self,
        *,
        country_code: str = "US",
        number_type: str = "local",
        area_code: Optional[int] = None,
        locality: Optional[str] = None,
        label: Optional[str] = None,
        assign: bool = True,
        agent_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        assemblyai_api_key: Optional[str] = None,
    ) -> dict:
        """Buy a fresh number and (by default) assign it to this agent — one
        call. Set ``assign=False`` to buy without assigning."""
        from .phones import buy_number

        aid = self._require_remote_id(agent_id) if assign else None
        return buy_number(
            assemblyai_api_key=self._aai_key(assemblyai_api_key),
            country_code=country_code, number_type=number_type, area_code=area_code,
            locality=locality, label=label, agent_id=aid, idempotency_key=idempotency_key,
        )

    def import_phone_number(
        self, phone_number: str, termination_uri: str, *,
        assign: bool = True, agent_id: Optional[str] = None,
        assemblyai_api_key: Optional[str] = None,
    ) -> dict:
        """Import a number you already own (BYO trunk), then assign it here."""
        from .phones import assign_number, import_number

        key = self._aai_key(assemblyai_api_key)
        result = import_number(phone_number, termination_uri, assemblyai_api_key=key)
        if assign:
            assign_number(phone_number, self._require_remote_id(agent_id), assemblyai_api_key=key)
        return result

    def assign_phone_number(self, phone_number: str, *, agent_id: Optional[str] = None,
                            assemblyai_api_key: Optional[str] = None) -> dict:
        """Bind an already-owned number to this agent (also re-assigns)."""
        from .phones import assign_number

        return assign_number(phone_number, self._require_remote_id(agent_id),
                             assemblyai_api_key=self._aai_key(assemblyai_api_key))

    def unassign_phone_number(self, phone_number: str, *,
                              assemblyai_api_key: Optional[str] = None) -> dict:
        from .phones import unassign_number

        return unassign_number(phone_number, assemblyai_api_key=self._aai_key(assemblyai_api_key))

    def phone_numbers(self, *, limit: int = 20, cursor: Optional[str] = None,
                      assemblyai_api_key: Optional[str] = None) -> dict:
        """List the numbers owned by this account."""
        from .phones import list_numbers

        return list_numbers(assemblyai_api_key=self._aai_key(assemblyai_api_key),
                            limit=limit, cursor=cursor)

    def release_phone_number(self, phone_number: str, *,
                             assemblyai_api_key: Optional[str] = None) -> dict:
        """Release a number back to the provider."""
        from .phones import release_number

        return release_number(phone_number, assemblyai_api_key=self._aai_key(assemblyai_api_key))

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        tunnel: bool = True,
        tunnel_provider: str = "auto",
        register: bool = True,
        agent_id: Optional[str] = None,
        **uvicorn_kwargs: Any,
    ) -> None:
        """Run the OpenAI-compatible server (blocking).

        By default this opens a public HTTPS URL (via ``cloudflared`` or
        ``ngrok``) and, if ``ASSEMBLYAI_API_KEY`` is set, wires that URL onto
        your agent record and prints a playground link so you can test it in the
        browser immediately. Pass ``tunnel=False`` for a local-only endpoint, or
        ``register=False`` to skip touching the agent record.
        """
        import uvicorn

        if not tunnel:
            shown_host = "localhost" if host in ("0.0.0.0", "") else host
            self._print_banner(f"http://{shown_host}:{port}/v1")
            uvicorn.run(self.app, host=host, port=port, **uvicorn_kwargs)
            return

        self._serve_with_tunnel(host, port, tunnel_provider, register, agent_id, uvicorn_kwargs)

    def _serve_with_tunnel(self, host, port, provider, register, agent_id, uvicorn_kwargs) -> None:
        import time

        import uvicorn

        from .tunnel import open_tunnel

        # We only register (and lock the endpoint with a rotating key) when a
        # key is actually available — otherwise a local dev run would lock its
        # own endpoint with a secret nobody knows, breaking curl/call.py.
        api_key = self.api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        will_register = register and bool(api_key)

        # Mint the rotating ingress key *before* the server starts, so the
        # endpoint is never open in the window before registration.
        if will_register and not self._ingress_explicit:
            self.ingress_key = secrets.token_urlsafe(24)

        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning", **uvicorn_kwargs)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.05)

        print(f"\n  AssemblyAI Agent SDK — {self.name!r}")
        print(f"  Local:  http://localhost:{port}/v1")
        print("  Opening public tunnel…")
        tun = open_tunnel(port, provider=provider)

        if tun is None:
            print("  Could not open a tunnel. Install cloudflared (recommended) or ngrok:")
            print("    brew install cloudflared")
            print(f"  The server is still running locally at http://localhost:{port}/v1\n")
        else:
            self._print_banner(f"{tun.url}/v1", public=True, via=tun.provider)
            if will_register:
                try:
                    # Key already minted above; don't rotate again here.
                    record = self.register(tun.url, agent_id=agent_id, rotate=False)
                    aid = record.get("id")
                    if aid:
                        print(f"  Agent id: {aid}")
                        print(f"  Test it in your browser:\n    {_PLAYGROUND_URL}/{aid}\n")
                    else:
                        print("  Registered the agent record (no id returned).\n")
                except Exception as exc:  # noqa: BLE001
                    print(f"  Registration failed (serving anyway): {exc}\n")
            elif register:
                print("  Set ASSEMBLYAI_API_KEY to auto-register this URL on your agent "
                      "and get a browser test link.\n")

        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.should_exit = True
            if tun is not None:
                tun.close()

    def _print_banner(self, base_url: str, *, public: bool = False, via: str = "") -> None:
        label = f"Public endpoint (via {via})" if public else "OpenAI-compatible endpoint"
        print(f"\n  AssemblyAI Agent SDK — {self.name!r}")
        print(f"  {label}: {base_url}")
        print(f"  Point your voice agent's LLM base_url here (model: {self.model!r}).\n")


def _slug(name: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in name)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or _DEFAULT_MODEL
