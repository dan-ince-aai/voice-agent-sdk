from starlette.testclient import TestClient

from assembly_agent import Agent


class FakeGateway:
    """Records the messages it last received."""

    def __init__(self):
        self.api_key = "k"
        self.seen = None

    @property
    def configured(self):
        return True

    async def complete(self, messages, *, model=None, **params):
        self.seen = messages
        return "ok"

    async def stream(self, messages, *, model=None, **params):
        self.seen = messages
        if False:
            yield ""


def _agent():
    agent = Agent(name="Mem")
    agent._gateway = FakeGateway()

    @agent.on_response
    async def respond(ev, ctx):
        return await ctx.llm.complete(model="m")

    return agent


def _say(client, text, call_id):
    # Each request carries ONLY the current turn — the ephemeral case.
    return client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": text}],
        "assemblyai": {"call_id": call_id},
    })


def test_history_accumulates_across_turns_even_when_requests_are_ephemeral():
    agent = _agent()
    client = TestClient(agent.app)

    _say(client, "my name is Sam", "c1")
    _say(client, "what's my name?", "c1")

    # By the second turn the gateway sees the whole conversation, not just the
    # latest turn — proving the SDK carries history.
    contents = [m["content"] for m in agent._gateway.seen]
    assert "my name is Sam" in contents
    assert "ok" in contents                 # the first assistant reply was recorded
    assert contents[-1] == "what's my name?"


def test_separate_calls_dont_share_history():
    agent = _agent()
    client = TestClient(agent.app)

    _say(client, "turn from call A", "A")
    _say(client, "turn from call B", "B")

    contents = [m["content"] for m in agent._gateway.seen]
    assert "turn from call A" not in contents
    assert contents == ["turn from call B"]


def test_history_dropped_on_call_end():
    agent = _agent()
    client = TestClient(agent.app)

    _say(client, "remember this", "c1")
    client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "bye"}],
        "assemblyai": {"call_id": "c1", "event": "call_end"},
    })
    _say(client, "fresh start", "c1")

    contents = [m["content"] for m in agent._gateway.seen]
    assert "remember this" not in contents
    assert contents == ["fresh start"]
