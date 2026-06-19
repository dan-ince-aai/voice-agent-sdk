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

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        tunnel: bool = True,
        tunnel_provider: str = "auto",
        register: bool = False,
        agent_id: Optional[str] = None,
        **uvicorn_kwargs: Any,
    ) -> None:
        """Run the OpenAI-compatible server (blocking).

        By default this opens a public HTTPS URL (via ``cloudflared`` or
        ``ngrok``) so the voice layer can reach a server running on your laptop
        with no deploy — that's the point: run it and it's callable. Pass
        ``tunnel=False`` for a local-only endpoint (e.g. when you're deploying
        to a host that already has a public address).

        ``register=True`` upserts the AssemblyAI agent record to point its BYO
        LLM endpoint at the public URL once it's up (requires
        ASSEMBLYAI_API_KEY).
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

        # Mint the rotating ingress key *before* the server starts, so the
        # endpoint is never open — even in the window before registration.
        if register and not self._ingress_explicit:
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
            if register:
                try:
                    # Key already minted above; don't rotate again here.
                    record = self.register(tun.url, agent_id=agent_id, rotate=False)
                    print(f"  Agent record set (id: {record.get('id')!r}); the voice "
                          f"layer will call this URL with a freshly-rotated key.\n")
                except Exception as exc:  # noqa: BLE001
                    print(f"  Registration failed: {exc}\n")

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
