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
        region: Optional[str] = None,
        api_key: Optional[str] = None,
        phone_number: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.name = name
        self.voice = voice
        self.greeting = greeting
        self.prompt = prompt
        # Stable id for upsert; falls back to a slug of the name.
        self.agent_id = agent_id or _slug(name)
        self.model = model or self.agent_id or _DEFAULT_MODEL

        # A number you already bought (via the CLI). serve() assigns it to this
        # agent at boot — free and idempotent, so it's safe every run. You only
        # ever handle the number; the agent id is resolved internally.
        self.phone_number = phone_number or os.environ.get("ASSEMBLY_AGENT_PHONE_NUMBER")

        # Shared secret the voice layer presents when calling this endpoint.
        # Auto-rotated on every registration; the server rejects requests that
        # don't present the current one. Not user-configurable.
        self.ingress_key: Optional[str] = None
        self.remote_agent_id: Optional[str] = None

        # Region picks both the agents REST endpoint and the LLM Gateway. US by
        # default; "eu" switches both to the EU endpoints. (llm_base_url / api_base
        # are advanced per-endpoint overrides.)
        self.region = (region or os.environ.get("ASSEMBLY_AGENT_REGION") or "us").lower()
        self._llm_base_url = llm_base_url
        self._api_base = api_base
        self.api_key = api_key
        self._gateway = None

        self._handlers: dict[str, Callable] = {}
        self._stores: dict[str, dict] = {}
        self.runtime = Runtime(self)
        self._app = None

    @property
    def llm_base_url(self) -> str:
        """LLM Gateway endpoint for this agent's region (or an explicit override)."""
        from .endpoints import llm_base

        return self._llm_base_url or os.environ.get("LLM_GATEWAY_URL") or llm_base(self.region)

    @property
    def api_base(self) -> str:
        """Agents REST endpoint for this agent's region (or an explicit override)."""
        from .endpoints import agents_base

        return self._api_base or agents_base(self.region)

    @property
    def gateway(self):
        """Lazily-built LLM Gateway client (see ``ctx.llm``)."""
        if self._gateway is None:
            from .gateway import Gateway

            self._gateway = Gateway(api_key=self.api_key, base_url=self.llm_base_url)
        return self._gateway

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

    def on_call_end(self, fn: Callable) -> Callable:
        return self._register(ev_mod.CALL_END, fn)

    def handler_for(self, event_type: str) -> Optional[Callable]:
        return self._handlers.get(event_type)

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

        ``public_url`` must be HTTPS/public-DNS (a tunnel or deployed host;
        localhost is rejected). A fresh shared secret is minted here and written
        to the record as the LLM ``api_key`` (a new key per registration).

        The record is identified by **name**: if an agent with this name already
        exists it's updated (PUT), otherwise one is created (POST). Pass
        ``agent_id`` to target a specific record instead."""
        from .registry import register_agent

        if rotate or not self.ingress_key:
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
            api_base=self.api_base,
        )
        self.remote_agent_id = record.get("id") or agent_id or self.remote_agent_id
        return record

    def _wire_phone(self, number: str, api_key: str) -> None:
        """Bind an already-owned number to this (registered) agent — Option D.
        Idempotent; safe to run every serve."""
        from .phones import assign_number

        if not self.remote_agent_id:
            raise RuntimeError("agent not registered yet — can't assign a number.")
        assign_number(number, self.remote_agent_id, assemblyai_api_key=api_key)

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        tunnel: bool = True,
        public_url: Optional[str] = None,
        tunnel_provider: str = "auto",
        **uvicorn_kwargs: Any,
    ) -> None:
        """Run the agent's chat-completions endpoint (blocking).

        With ``ASSEMBLYAI_API_KEY`` set, the agent is registered automatically
        and a browser playground link is printed. ``tunnel=True`` (default)
        opens a public URL to your machine. For a deploy, use ``tunnel=False``
        and pass ``public_url`` (or ``ASSEMBLY_AGENT_PUBLIC_URL``) so the record
        points at your host. A ``phone_number`` on the ``Agent`` is assigned on
        boot.
        """
        import time

        import uvicorn

        api_key = self.api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        public_url = public_url or os.environ.get("ASSEMBLY_AGENT_PUBLIC_URL")
        will_register = bool(api_key) and (tunnel or bool(public_url))

        # Mint the rotating ingress key before serving, so the endpoint is never
        # open in the window before registration.
        if will_register:
            self.ingress_key = secrets.token_urlsafe(24)

        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning", **uvicorn_kwargs)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.05)

        tun = None
        url = public_url
        if tunnel:
            from .tunnel import open_tunnel

            print(f"\n  Opening public tunnel for {self.name!r}…")
            tun = open_tunnel(port, provider=tunnel_provider)
            if tun is None:
                print("  Could not open a tunnel. Install cloudflared (brew install cloudflared) or ngrok.")
                print(f"  Serving locally at http://localhost:{port}/v1\n")
            else:
                url = tun.url
                self._print_banner(f"{url}/v1", public=True, via=tun.provider)
        else:
            shown = "localhost" if host in ("0.0.0.0", "") else host
            self._print_banner(f"{public_url or f'http://{shown}:{port}'}/v1")

        try:
            if api_key and url:
                record = self.register(url, rotate=False)
                aid = record.get("id")
                if aid:
                    print(f"  Agent id: {aid}")
                    print(f"  Test it in your browser:\n    {_PLAYGROUND_URL}/{aid}\n")
                if self.phone_number and self.remote_agent_id:
                    try:
                        self._wire_phone(self.phone_number, api_key)
                        print(f"  Phone: calls to {self.phone_number} now reach this agent.\n")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  Could not assign {self.phone_number} — is it purchased? "
                              f"Run: assembly-agent phone buy ...\n    ({exc})\n")
            elif not api_key:
                print("  Set ASSEMBLYAI_API_KEY to register this agent and get a browser link.\n")
            elif not url:
                print("  Pass public_url=… (or ASSEMBLY_AGENT_PUBLIC_URL) to register a deployed URL.\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  Registration failed (serving anyway): {exc}\n")

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
