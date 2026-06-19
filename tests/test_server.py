import json

import pytest
from starlette.testclient import TestClient

from assembly_agent import Agent, Reply


def build_agent():
    # api_key="" -> no Gateway, so returning None means passthrough (not augment),
    # independent of whatever ASSEMBLYAI_API_KEY may be in the environment.
    agent = Agent(name="Test Agent", voice="ivy", greeting="Hi there, you're connected.", api_key="")

    @agent.on_response
    async def respond(ev, ctx):
        if ev.text == "frustrated":
            return Reply("I hear you.", tone="reassuring", speed="slow")
        if ev.text == "spanish":
            return ctx.transfer("support-es")
        if ev.text == "augment":
            return None
        if ev.text == "remember":
            ctx.set("seen", True)
            return "ok"
        if ev.text == "recall":
            return "yes" if ctx.get("seen") else "no"
        if ev.text == "stream":
            async def toks():
                for t in ["one ", "two ", "three"]:
                    yield t
            return toks()
        return f"echo: {ev.text}"

    @agent.tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    return agent


@pytest.fixture
def client():
    return TestClient(build_agent().app)


def chat(text, *, stream=False, call_id="c1", enrichment=None, messages=None):
    body = {
        "model": "test-agent",
        "stream": stream,
        "messages": messages or [{"role": "user", "content": text}],
        "assemblyai": {"call_id": call_id, **(enrichment or {})},
    }
    return body


def test_plain_string_response(client):
    r = client.post("/v1/chat/completions", json=chat("hello"))
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    msg = data["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "echo: hello"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_call_start_greeting_when_no_user_turn(client):
    body = {"model": "test-agent", "messages": [{"role": "system", "content": "be nice"}],
            "assemblyai": {"call_id": "g1"}}
    r = client.post("/v1/chat/completions", json=body)
    assert r.json()["choices"][0]["message"]["content"] == "Hi there, you're connected."


def test_reply_delivery_controls(client):
    r = client.post("/v1/chat/completions", json=chat("frustrated"))
    choice = r.json()["choices"][0]
    assert choice["message"]["content"] == "I hear you."
    assert choice["assemblyai"]["delivery"] == {"tone": "reassuring", "speed": "slow"}


def test_transfer(client):
    r = client.post("/v1/chat/completions", json=chat("spanish"))
    ext = r.json()["choices"][0]["assemblyai"]
    assert ext["action"] == "transfer"
    assert ext["transfer_to"] == "support-es"


def test_passthrough_on_none(client):
    r = client.post("/v1/chat/completions", json=chat("augment"))
    ext = r.json()["choices"][0]["assemblyai"]
    assert ext["action"] == "passthrough"
    assert r.json()["choices"][0]["message"]["content"] == ""


def test_ctx_persists_across_turns(client):
    client.post("/v1/chat/completions", json=chat("remember", call_id="same"))
    r = client.post("/v1/chat/completions", json=chat("recall", call_id="same"))
    assert r.json()["choices"][0]["message"]["content"] == "yes"
    # A different call id does not see the stashed state.
    r2 = client.post("/v1/chat/completions", json=chat("recall", call_id="other"))
    assert r2.json()["choices"][0]["message"]["content"] == "no"


def test_call_end_drops_state(client):
    client.post("/v1/chat/completions", json=chat("remember", call_id="ends"))
    end = {"model": "x", "messages": [{"role": "user", "content": "bye"}],
           "assemblyai": {"call_id": "ends", "event": "call_end"}}
    client.post("/v1/chat/completions", json=end)
    r = client.post("/v1/chat/completions", json=chat("recall", call_id="ends"))
    assert r.json()["choices"][0]["message"]["content"] == "no"


def test_streaming_sse(client):
    with client.stream("POST", "/v1/chat/completions", json=chat("stream", stream=True)) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = "".join(r.iter_text())

    lines = [ln for ln in body.split("\n\n") if ln.startswith("data: ")]
    assert lines[-1].strip() == "data: [DONE]"
    contents = []
    role_seen = False
    for ln in lines:
        payload = ln[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        delta = json.loads(payload)["choices"][0]["delta"]
        if delta.get("role") == "assistant":
            role_seen = True
        if "content" in delta and delta["content"]:
            contents.append(delta["content"])
    assert role_seen
    assert "".join(contents) == "one two three"


def test_generator_handler_collected_for_nonstream(client):
    # Handler yields tokens but the client did not ask for a stream: the SDK
    # collects them into a single completion body.
    r = client.post("/v1/chat/completions", json=chat("stream", stream=False))
    assert r.json()["choices"][0]["message"]["content"] == "one two three"


def test_signals_reach_handler():
    agent = Agent(name="S")

    @agent.on_response
    async def respond(ev, ctx):
        return f"{ev.signals.emotion}/{ev.signals.prosody.pace}/{ev.language}"

    client = TestClient(agent.app)
    body = chat("hi", enrichment={
        "signals": {"emotion": "calm", "prosody": {"pace": "fast"}},
        "language": "en",
    })
    r = client.post("/v1/chat/completions", json=body)
    assert r.json()["choices"][0]["message"]["content"] == "calm/fast/en"


def test_on_interrupt_fires_and_returns_empty():
    agent = Agent(name="I")
    hits = {"n": 0}

    @agent.on_interrupt
    async def stopped(ev, ctx):
        hits["n"] += 1
        ctx.cancel_pending()

    client = TestClient(agent.app)
    body = {"model": "x", "messages": [{"role": "user", "content": "wait"}],
            "assemblyai": {"call_id": "z", "event": "interruption"}}
    r = client.post("/v1/chat/completions", json=body)
    assert hits["n"] == 1
    assert r.json()["choices"][0]["message"]["content"] == ""


def test_models_endpoint(client):
    r = client.get("/v1/models")
    data = r.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "test-agent"


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_enrichment_on_user_message():
    """Enrichment attached to the last user message (not request-level)."""
    agent = Agent(name="S")

    @agent.on_response
    async def respond(ev, ctx):
        return ev.signals.emotion or "none"

    client = TestClient(agent.app)
    body = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "hi", "assemblyai": {"signals": {"emotion": "happy"}}}
        ],
    }
    r = client.post("/v1/chat/completions", json=body)
    assert r.json()["choices"][0]["message"]["content"] == "happy"
