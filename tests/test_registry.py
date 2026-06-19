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
def _mock_agents_api(existing=None):
    captured = {}
    state = {"record": existing or {}}

    async def create(request):
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = await request.json()
        return JSONResponse({"id": "agent_123", **captured["body"]})

    async def by_id(request):
        agent_id = request.path_params["agent_id"]
        if request.method == "GET":
            return JSONResponse({"id": agent_id, **state["record"]})
        captured["method"] = "PUT"
        captured["agent_id"] = agent_id
        captured["body"] = await request.json()
        return JSONResponse({"id": agent_id, **captured["body"]})

    app = Starlette(routes=[
        Route("/v1/agents", create, methods=["POST"]),
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
    assert record["id"] == "agent_123"


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


def test_update_merges_and_preserves_other_fields():
    # PUT is a full replace, so updating must keep input/output/tools intact
    # and only swap the llm block + the SDK-managed fields.
    existing = {
        "name": "Old Name",
        "voice": {"voice_id": "ivy"},
        "tools": [{"name": "get_weather"}],
        "input": {"type": "audio", "format": {"sample_rate": 24000}},
        "llm": [{"base_url": "https://old.example.com/v1", "model": "old"}],
    }
    app, captured = _mock_agents_api(existing)
    api_base, server = _serve(app)
    try:
        register_agent(
            name="New Name", voice="ivy", public_url="https://new.trycloudflare.com",
            model="m", ingress_key="freshkey", assemblyai_api_key="aai",
            agent_id="agent_1", api_base=api_base,
        )
    finally:
        server.should_exit = True

    body = captured["body"]
    assert body["tools"] == [{"name": "get_weather"}]          # preserved
    assert body["input"] == {"type": "audio", "format": {"sample_rate": 24000}}  # preserved
    assert body["name"] == "New Name"                          # managed field updated
    assert body["llm"] == [{"base_url": "https://new.trycloudflare.com/v1",
                            "model": "m", "api_key": "freshkey"}]  # llm swapped


def test_register_requires_api_key():
    with pytest.raises(RegistrationError):
        register_agent(name="X", voice="ivy", public_url="https://abc.trycloudflare.com",
                       model="x", ingress_key="k", assemblyai_api_key="")


def test_agent_register_generates_ingress_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
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


def test_register_rotates_key_each_call(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("assembly_agent.registry.register_agent",
                        lambda **kw: calls.append(kw) or {"id": "agent_x"})

    agent = Agent(name="Rotate")
    agent.register("https://x.trycloudflare.com", assemblyai_api_key="aai")
    agent.register("https://x.trycloudflare.com", assemblyai_api_key="aai")

    assert calls[0]["ingress_key"] != calls[1]["ingress_key"]  # fresh key each time


def test_explicit_key_is_not_rotated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("assembly_agent.registry.register_agent",
                        lambda **kw: calls.append(kw) or {"id": "agent_x"})

    agent = Agent(name="Fixed", ingress_key="my-fixed-key")
    agent.register("https://x.trycloudflare.com", assemblyai_api_key="aai")
    agent.register("https://x.trycloudflare.com", assemblyai_api_key="aai")

    assert calls[0]["ingress_key"] == "my-fixed-key"
    assert calls[1]["ingress_key"] == "my-fixed-key"


def test_register_persists_and_reuses_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("assembly_agent.registry.register_agent",
                        lambda **kw: calls.append(kw) or {"id": "agent_persist"})

    Agent(name="Persist").register("https://x.trycloudflare.com", assemblyai_api_key="aai")
    assert calls[0]["agent_id"] is None  # first run creates

    # A fresh instance (new process) reuses the persisted id → updates via PUT.
    Agent(name="Persist").register("https://x.trycloudflare.com", assemblyai_api_key="aai")
    assert calls[1]["agent_id"] == "agent_persist"


# --- server-side ingress auth -------------------------------------------- #
def _client(ingress_key):
    agent = Agent(name="Auth", ingress_key=ingress_key)

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
