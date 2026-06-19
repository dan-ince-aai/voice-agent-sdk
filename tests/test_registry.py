import json
import socket
import threading
import time

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from assembly_agent import Agent
from assembly_agent.registry import RegistrationError, normalize_base_url, register_agent


# --- base_url normalization / SSRF guard --------------------------------- #
def test_normalize_appends_v1():
    assert normalize_base_url("https://abc.trycloudflare.com") == "https://abc.trycloudflare.com/v1"
    assert normalize_base_url("https://abc.trycloudflare.com/v1/") == "https://abc.trycloudflare.com/v1"


def test_normalize_rejects_http():
    with pytest.raises(RegistrationError):
        normalize_base_url("http://abc.trycloudflare.com")


def test_normalize_rejects_localhost():
    with pytest.raises(RegistrationError):
        normalize_base_url("https://localhost:8000")
    with pytest.raises(RegistrationError):
        normalize_base_url("http://127.0.0.1:8000")


# --- register_agent against a local mock agents API ---------------------- #
def _mock_agents_api(existing_agents=None):
    """existing_agents: list of full records (each with id + name)."""
    captured = {}
    agents = {a["id"]: a for a in (existing_agents or [])}

    async def list_or_create(request):
        if request.method == "GET":
            return JSONResponse({"agents": list(agents.values())})
        captured["method"] = "POST"
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = await request.json()
        return JSONResponse({"id": "agent_new", **captured["body"]})

    async def by_id(request):
        agent_id = request.path_params["agent_id"]
        if request.method == "GET":
            return JSONResponse({"id": agent_id, **agents.get(agent_id, {})})
        captured["method"] = "PUT"
        captured["agent_id"] = agent_id
        captured["body"] = await request.json()
        return JSONResponse({"id": agent_id, **captured["body"]})

    app = Starlette(routes=[
        Route("/v1/agents", list_or_create, methods=["GET", "POST"]),
        Route("/v1/agents/{agent_id}", by_id, methods=["GET", "PUT"]),
    ])
    return app, captured


def _serve(app):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}/v1", server


def test_register_agent_posts_llm_record():
    app, captured = _mock_agents_api()
    api_base, server = _serve(app)
    try:
        record = register_agent(
            name="Support Assistant",
            voice="ivy",
            public_url="https://abc.trycloudflare.com",
            model="support-assistant",
            ingress_key="secret-123",
            assemblyai_api_key="aai-key",
            greeting="Hi",
            api_base=api_base,
        )
    finally:
        server.should_exit = True

    assert captured["method"] == "POST"
    assert captured["auth"] == "aai-key"  # raw key, AssemblyAI convention
    llm = captured["body"]["llm"]
    assert len(llm) == 1
    assert llm[0] == {
        "base_url": "https://abc.trycloudflare.com/v1",
        "model": "support-assistant",
        "api_key": "secret-123",
    }
    assert captured["body"]["voice"] == {"voice_id": "ivy"}
    assert captured["body"]["greeting"] == "Hi"
    assert record["id"] == "agent_new"


def test_register_agent_puts_when_id_given():
    app, captured = _mock_agents_api()
    api_base, server = _serve(app)
    try:
        register_agent(
            name="X", voice="ivy", public_url="https://abc.trycloudflare.com",
            model="x", ingress_key="k", assemblyai_api_key="aai", agent_id="agent_999",
            api_base=api_base,
        )
    finally:
        server.should_exit = True
    assert captured["method"] == "PUT"
    assert captured["agent_id"] == "agent_999"


def test_update_by_name_when_name_matches():
    # Same name as an existing agent → update that one (PUT), no agent_id needed.
    existing = [{
        "id": "agent_55", "name": "Support Assistant", "voice": {"voice_id": "ivy"},
        "tools": [{"name": "get_weather"}],
        "input": {"type": "audio", "format": {"sample_rate": 24000}},
        "llm": [{"base_url": "https://old.example.com/v1", "model": "old"}],
    }]
    app, captured = _mock_agents_api(existing)
    api_base, server = _serve(app)
    try:
        register_agent(
            name="Support Assistant", voice="ivy", public_url="https://new.trycloudflare.com",
            model="m", ingress_key="freshkey", assemblyai_api_key="aai", api_base=api_base,
        )
    finally:
        server.should_exit = True

    assert captured["method"] == "PUT"
    assert captured["agent_id"] == "agent_55"          # found by name
    body = captured["body"]
    assert body["tools"] == [{"name": "get_weather"}]  # preserved (PUT is full replace)
    assert body["input"] == {"type": "audio", "format": {"sample_rate": 24000}}
    assert body["llm"] == [{"base_url": "https://new.trycloudflare.com/v1",
                            "model": "m", "api_key": "freshkey"}]  # llm swapped


def test_create_when_name_not_found():
    existing = [{"id": "agent_99", "name": "Some Other Agent"}]
    app, captured = _mock_agents_api(existing)
    api_base, server = _serve(app)
    try:
        register_agent(
            name="Brand New", voice="ivy", public_url="https://abc.trycloudflare.com",
            model="m", ingress_key="k", assemblyai_api_key="aai", api_base=api_base,
        )
    finally:
        server.should_exit = True
    assert captured["method"] == "POST"   # no name match → create


def test_register_requires_api_key():
    with pytest.raises(RegistrationError):
        register_agent(name="X", voice="ivy", public_url="https://abc.trycloudflare.com",
                       model="x", ingress_key="k", assemblyai_api_key="")


def test_agent_register_generates_ingress_key(monkeypatch):
    seen = {}

    def fake_register_agent(**kwargs):
        seen.update(kwargs)
        return {"id": "agent_abc"}

    monkeypatch.setattr("assembly_agent.registry.register_agent", fake_register_agent)

    agent = Agent(name="Gen")
    assert agent.ingress_key is None
    record = agent.register("https://abc.trycloudflare.com", assemblyai_api_key="aai")

    # A key was generated (endpoint not left open) and passed as the llm api_key.
    assert agent.ingress_key
    assert seen["ingress_key"] == agent.ingress_key
    assert agent.remote_agent_id == "agent_abc"
    assert record["id"] == "agent_abc"
    # Identity is by name — no explicit id sent; register_agent resolves it.
    assert seen["agent_id"] is None
    assert seen["name"] == "Gen"


def test_register_rotates_key_each_call(monkeypatch):
    calls = []
    monkeypatch.setattr("assembly_agent.registry.register_agent",
                        lambda **kw: calls.append(kw) or {"id": "agent_x"})

    agent = Agent(name="Rotate")
    agent.register("https://x.trycloudflare.com", assemblyai_api_key="aai")
    agent.register("https://x.trycloudflare.com", assemblyai_api_key="aai")

    assert calls[0]["ingress_key"] != calls[1]["ingress_key"]  # fresh key each time


# --- server-side ingress auth -------------------------------------------- #
def _client(ingress_key):
    agent = Agent(name="Auth")
    agent.ingress_key = ingress_key   # auto-set on registration; set directly for the test

    @agent.on_response
    async def respond(ev, ctx):
        return "ok"

    return TestClient(agent.app)


def _post(client, headers=None):
    return client.post("/v1/chat/completions", headers=headers or {}, json={
        "model": "x", "messages": [{"role": "user", "content": "hi"}],
        "assemblyai": {"call_id": "a"},
    })


def test_no_ingress_key_is_open():
    assert _post(_client(None)).status_code == 200


def test_ingress_key_required_when_set():
    client = _client("topsecret")
    assert _post(client).status_code == 401                                   # missing
    assert _post(client, {"Authorization": "Bearer wrong"}).status_code == 401  # wrong
    assert _post(client, {"Authorization": "Bearer topsecret"}).status_code == 200  # bearer
    assert _post(client, {"Authorization": "topsecret"}).status_code == 200        # raw
