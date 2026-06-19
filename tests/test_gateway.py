import asyncio

from starlette.testclient import TestClient

from assembly_agent import Agent
from assembly_agent.gateway import CallLLM, Gateway


class FakeGateway:
    """Stand-in for the real Gateway: records the messages it was given and
    returns canned output."""

    def __init__(self, *, reply="canned answer", tokens=None):
        self.api_key = "fake-key"
        self.reply = reply
        self.tokens = tokens or ["can", "ned ", "answer"]
        self.seen = None
        self.seen_model = None

    @property
    def configured(self):
        return True

    async def complete(self, messages, *, model=None, **params):
        self.seen = messages
        self.seen_model = model
        return self.reply

    async def stream(self, messages, *, model=None, **params):
        self.seen = messages
        self.seen_model = model
        for t in self.tokens:
            yield t


def test_callllm_builds_messages_with_system_and_history():
    class M:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    fake = FakeGateway()
    history = [M("user", "hi"), M("assistant", "hello"), M("user", "what's up")]
    llm = CallLLM(fake, history)
    asyncio.run(llm.complete(system="be nice"))  # system is per-call, not config
    assert fake.seen[0] == {"role": "system", "content": "be nice"}
    assert {"role": "user", "content": "hi"} in fake.seen
    assert fake.seen[-1] == {"role": "user", "content": "what's up"}


def test_callllm_tolerates_dict_history_items():
    # Handlers naturally append plain dicts to ctx.history — they must be sent.
    fake = FakeGateway()
    history = [{"role": "user", "content": "hi"}, {"role": "system", "content": "extra fact"}]
    llm = CallLLM(fake, history)
    asyncio.run(llm.complete())
    assert {"role": "user", "content": "hi"} in fake.seen
    assert {"role": "system", "content": "extra fact"} in fake.seen


def test_handler_uses_ctx_llm():
    agent = Agent(name="G")
    fake = FakeGateway(reply="from the gateway")
    agent._gateway = fake  # inject

    @agent.on_response
    async def respond(ev, ctx):
        return await ctx.llm.complete()

    client = TestClient(agent.app)
    r = client.post("/v1/chat/completions", json={
        "model": "g", "messages": [{"role": "user", "content": "hello"}],
        "assemblyai": {"call_id": "a"},
    })
    assert r.json()["choices"][0]["message"]["content"] == "from the gateway"
    # the user turn made it into the messages sent to the gateway
    assert {"role": "user", "content": "hello"} in fake.seen


def test_model_is_chosen_per_request():
    agent = Agent(name="G")
    fake = FakeGateway(reply="ok")
    agent._gateway = fake

    @agent.on_response
    async def respond(ev, ctx):
        return await ctx.llm.complete(model="gpt-4o")

    client = TestClient(agent.app)
    client.post("/v1/chat/completions", json={
        "model": "g", "messages": [{"role": "user", "content": "hi"}],
        "assemblyai": {"call_id": "a"},
    })
    assert fake.seen_model == "gpt-4o"


def test_agent_takes_no_llm_model_config():
    import pytest as _pytest

    # The model is per-request now; the agent constructor must not accept `llm=`.
    with _pytest.raises(TypeError):
        Agent(name="X", llm="claude-sonnet-4-6")


def test_no_handler_is_passthrough_not_managed():
    # With no on_response the SDK does NOT secretly proxy to the Gateway — a
    # handler-less agent is pointless, so it just passes through.
    agent = Agent(name="Bare")
    agent._gateway = FakeGateway(reply="should not be used")

    client = TestClient(agent.app)
    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "assemblyai": {"call_id": "a"},
    })
    assert r.json()["choices"][0]["assemblyai"]["action"] == "passthrough"


def test_handler_returning_none_is_passthrough_even_with_gateway():
    # Returning None never implicitly calls the Gateway — it's passthrough.
    # To have the Gateway answer, the handler returns ctx.llm.complete().
    agent = Agent(name="Aug")
    agent._gateway = FakeGateway(reply="should not be used")

    @agent.on_response
    async def respond(ev, ctx):
        return None

    client = TestClient(agent.app)
    r = client.post("/v1/chat/completions", json={
        "model": "a", "messages": [{"role": "user", "content": "hi"}],
        "assemblyai": {"call_id": "a"},
    })
    choice = r.json()["choices"][0]
    assert choice["assemblyai"]["action"] == "passthrough"
    assert choice["message"]["content"] == ""


def test_ctx_llm_stream_through_sdk():
    import json

    agent = Agent(name="S")
    agent._gateway = FakeGateway(tokens=["one ", "two ", "three"])

    @agent.on_response
    async def respond(ev, ctx):
        return ctx.llm.stream()

    client = TestClient(agent.app)
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "s", "stream": True, "messages": [{"role": "user", "content": "go"}],
        "assemblyai": {"call_id": "a"},
    }) as s:
        body = "".join(s.iter_text())
    contents = []
    for ln in body.split("\n\n"):
        if ln.startswith("data: ") and "[DONE]" not in ln:
            d = json.loads(ln[6:])["choices"][0]["delta"]
            if d.get("content"):
                contents.append(d["content"])
    assert "".join(contents) == "one two three"


def test_gateway_reads_env_key(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "env-key-123")
    gw = Gateway()
    assert gw.api_key == "env-key-123"
    assert gw.configured
    assert gw.base_url.endswith("/v1")
