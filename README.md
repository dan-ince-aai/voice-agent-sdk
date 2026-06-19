# Agent SDK

> **Independent prototype — not an official AssemblyAI product or release.** A
> design sketch exploring what a logic-side voice-agent SDK could look like.
> References to AssemblyAI products describe intended integration points.

Write the **logic** side of a voice agent in plain Python. The SDK serves it as
an **OpenAI-compatible chat-completions endpoint** that the AssemblyAI voice
layer talks to.

You own what to say; we own the senses — STT, TTS, turn-taking, barge-in, and
the audio signals (emotion, prosody, speaker, language) computed on every turn
and handed to you with no extra inference on your side.

```python
from assembly_agent import Agent, Reply

agent = Agent(name="Support Assistant", voice="ivy")

@agent.on_response
async def respond(ev, ctx):
    if ev.signals.emotion == "frustrated":
        return Reply("I hear you, let me fix this.", tone="reassuring", speed="slow")
    return await my_llm(ev.text, history=ctx.history)

agent.serve()   # opens a public HTTPS URL you point the voice agent at
```

## The idea

`on_response` is just code. It can call an LLM, walk a decision tree, hit a
lookup table, query a database, run an open-source model — anything. The voice
layer never knows the difference, because the seam between the two is the
universal OpenAI chat-completions schema. `agent.serve()` stands up that
endpoint for you; you never serve a route, parse the schema, or touch a socket.

```
  ┌─────────────────────┐         OpenAI /v1/chat/completions        ┌──────────────────┐
  │  AssemblyAI voice    │  ── messages + audio signals (enriched) ─▶ │  your agent.py    │
  │  layer (the senses)  │                                            │  (the logic)      │
  │  STT · TTS · turns   │ ◀── text / Reply / stream / transfer ────  │  on_response(...) │
  └─────────────────────┘                                            └──────────────────┘
```

## Install & run

```sh
cd agent-sdk
pip install -e .              # or: pip install -e ".[dev]" for tests
python examples/support_assistant.py
```

That opens a public endpoint to point your voice agent's LLM `base_url` at:

```
  Public endpoint (via cloudflared): https://wisconsin-comparable-encounter.trycloudflare.com/v1
  Point your voice agent's LLM base_url here (model: 'support-assistant').
```

(Local-only? `serve(tunnel=False)` serves at `http://localhost:8000/v1`.)

Because it's just an OpenAI endpoint, any OpenAI client works against it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
client.chat.completions.create(model="support-assistant",
                               messages=[{"role": "user", "content": "hi"}])
```

## Reachability — develop locally, get called

The voice layer has to reach your handler. You do **not** need to deploy to
Railway/Fly. `agent.serve()` opens a public HTTPS URL straight to your laptop —
that's the point: run it and it's callable.

```
  Public endpoint (via cloudflared): https://wisconsin-comparable-encounter.trycloudflare.com/v1
  Point your voice agent's LLM base_url here (model: 'support-assistant').
```

Paste that `…/v1` URL into your agent's LLM `base_url` and place a call — it
runs against the code on your machine. It uses [`cloudflared`](https://github.com/cloudflare/cloudflared)
quick tunnels (no account needed); `ngrok` works too if that's what you have.

```sh
brew install cloudflared    # one-time; without it serve() falls back to local-only
```

| Mode                          | When                                            |
| ----------------------------- | ----------------------------------------------- |
| `serve()`                     | **Default.** Public URL to your laptop. Develop locally and take real calls. |
| `serve(tunnel=False)`         | Local only (curl / OpenAI-client testing, or deploying behind your own address). |
| Deploy to Railway/Fly/Render  | A stable production URL (`serve(tunnel=False, host="0.0.0.0", port=$PORT)`). |

> The destination in the design doc is an **outbound-worker** model — `serve()`
> dials out to AssemblyAI and registers as the worker, so there's no URL and no
> tunnel at all (like the Stripe CLI). That needs the worker-dispatch endpoint
> on the AssemblyAI side; until then, the tunnel is the zero-deploy bridge and
> the handler code you write doesn't change when it lands.

## Events

Three events are the whole surface. Handling the response is most of the power.

| Decorator             | Fires when                                                        |
| --------------------- | ----------------------------------------------------------------- |
| `@agent.on_call_start`| Call connects, before anyone speaks. Greet, or skip it.           |
| `@agent.on_response`  | Every finalized user turn. Transcript + signals in, a reply out. Do anything here. |
| `@agent.on_interrupt` | They cut in — drop what you were doing (`ctx.cancel_pending()`).  |

```python
@agent.on_response
async def respond(ev, ctx):
    # simplest version is a proxy:
    return await my_llm(ev.text, history=ctx.history)
```

Two more are there when you need them:

| Decorator             | Fires when                                                        |
| --------------------- | ----------------------------------------------------------------- |
| `@agent.on_speaker_change` | Diarization changed (handoff, third party).                  |
| `@agent.on_call_end`  | Wrap up — write back to the CRM, fire a webhook.                  |
| `@agent.tool`         | Register a plain function; schema inferred from the signature.    |

Handlers may be `async` or plain functions. Each gets the event `ev` and a
per-call context `ctx`.

### What `ev` carries

- `ev.text` — finalized transcript for this turn.
- `ev.language` — detected language, per turn (catch a mid-call switch).
- `ev.caller` — `phone_number`, `direction`, `from`/`to` (on call start).
- `ev.speaker` — diarization `id` and `confidence`.
- `ev.signals` — `emotion`, `sentiment`, `prosody` (`pitch`/`energy`/`pace`),
  `hesitance`, `accent`, `confidence`. Precomputed, no added latency.
- `ev.turn` — `interruption`, `overlap`, `latency`.

Missing fields read back as `None`, so you can branch on them directly. Unknown
forward-compatible fields are reachable via `.get()` / `["key"]` / `.to_dict()`.

### `ctx`

- `ctx.set(key, value)` / `ctx.get(key)` — key/value store for the whole call.
- `ctx.history` — running message list (`.role` / `.content`).
- `ctx.transfer(agent_name, reason=None)` — route the call elsewhere.
- `ctx.cancel_pending()` / `ctx.cancelled` — drop in-flight work on barge-in.
- `await ctx.call_tool(name, **kwargs)` — run a registered tool.
- `ctx.llm` — the LLM Gateway, bound to this call (see below).

## LLM Gateway (built in)

The [AssemblyAI LLM Gateway](https://assemblyai.com/docs/llm-gateway) is a
native primitive — one OpenAI-compatible interface across Claude / GPT / Gemini
/ Qwen / Kimi, with automatic retries and fallbacks. The agent already
authenticates with your AssemblyAI key, so the Gateway **reuses
`ASSEMBLYAI_API_KEY`** — no second credential, nothing in code.

```python
agent = Agent(name="Assistant", voice="ivy")

@agent.on_response
async def respond(ev, ctx):
    if ev.signals.emotion == "frustrated":
        return Reply("I hear you.", tone="reassuring", speed="slow")
    return await ctx.llm.complete(model="claude-sonnet-4-6", system="Be warm and concise.")
```

The **model and system prompt are chosen per request**, in the call — both are
response-generation decisions, not agent config. Agent config stays about
identity and senses (name, voice, greeting, TTS `prompt`) plus connection
(`api_key`, `llm_base_url` for EU residency). `ctx.llm` already carries the call
history, so you pass `model=` (and optionally `system=`); any other Gateway
param goes through as kwargs.

**Managed default:** register *no* `on_response` and, as long as
`ASSEMBLYAI_API_KEY` is set, every turn is answered through the Gateway
automatically with the default model — the whole agent is a name and a voice
(`examples/managed.py`). Add an `on_response` to pick the model per turn. If the
Gateway errors, the turn falls back to passthrough.

## Returning a response

`on_response` (and `on_call_start`) can return:

| Return value             | Effect                                                         |
| ------------------------ | -------------------------------------------------------------- |
| `str`                    | Spoken as-is.                                                  |
| `Reply(text, tone=, speed=, …)` | Same text, shaped delivery. Controls: `tone`, `speed`, `pitch`, `emphasis`, `pause`, `voice` (one-off voice swap). |
| `async`/sync generator   | Tokens streamed to TTS as they arrive (lower time-to-first-audio). |
| `ctx.transfer(...)`      | Hand the call to another agent.                                |
| `None`                   | **Augment** — you didn't generate the reply, so fall back. Identical to having no `on_response` at all: the Gateway answers this turn (with any context you added to `ctx`/history) if a key is configured, otherwise the turn passes through to the voice layer's managed LLM. |

So returning `None` is the "intercept only — add context, redact, then let the
model write the words" pattern, and it routes exactly where a handler-less agent
would.

Delivery controls, transfer, and the passthrough signal ride back in an
`assemblyai` extension field on the response choice, alongside a standard OpenAI
body — so a client that ignores the extension still gets valid output.

## Tools

```python
@agent.tool
async def lookup_order(order_id: str) -> dict:
    "Look up an order's status."
    return await db.get_order(order_id)
```

The JSON schema is inferred from the type hints; the first paragraph of the
docstring is the description. `ctx` (if present as the first param) is excluded.
`agent.tool_schemas()` returns them in OpenAI `tools` format. Call them from
your handlers with `await ctx.call_tool("lookup_order", order_id="123")`.

## Guardrails are just your code

Redaction, refund checks, compliance filters — they run in your process, so your
team can read, version, and audit them, and nothing leaves your environment
except the line you approved:

```python
reply = redact_pii(reply)
if promises_refund(reply) and not ctx.get("customer").refund_eligible:
    return "Let me connect you with someone who can help with that."
return reply
```

## How the enrichment is wired

The voice layer sends a normal OpenAI request and attaches the audio signals
under an `assemblyai` key — either at the top level of the body or on the latest
user message. The SDK reads either location:

```json
{
  "model": "support-assistant",
  "messages": [{ "role": "user", "content": "I want a refund" }],
  "assemblyai": {
    "call_id": "abc-123",
    "language": "en",
    "signals": { "emotion": "frustrated", "prosody": { "pace": "fast" } },
    "caller": { "phone_number": "+1...", "direction": "inbound" },
    "turn": { "interruption": false }
  }
}
```

`call_id` (or an `X-Call-Id` header) is what ties `ctx` state together across
turns; it's dropped on `on_call_end`. Side-channel events
(`interruption` / `speaker_change` / `call_end`) arrive as requests with
`assemblyai.event` set.

## Endpoints

- `POST /v1/chat/completions` — the seam (streaming via SSE when `stream: true`).
- `GET  /v1/models` — advertises this agent as a model id.
- `GET  /healthz` — liveness.

## Examples

- `examples/support_assistant.py` — the reference example (CRM, guardrails, Reply, transfer, tool).
- `examples/decision_tree.py` — bring-your-own-logic with **no LLM** (rules + per-call state).
- `examples/streaming.py` — token streaming via an async generator.
- `examples/llm_gateway.py` — `ctx.llm` proxy to the **LLM Gateway** with an emotion-shaped short-circuit. Needs `ASSEMBLYAI_API_KEY`.
- `examples/managed.py` — **no handlers**: the Gateway answers every turn. Needs `ASSEMBLYAI_API_KEY`.

## Tests

```sh
pip install -e ".[dev]"
pytest
```

## Relationship to the design doc

The design doc sketches a no-URL **outbound worker** (`serve()` dials out and
registers itself) and notes a portable **inbound `url=` mode** as the explicit
opt-in. This package implements that inbound mode: `serve()` stands up the
OpenAI endpoint you point the voice agent at. The handler/event surface (`Agent`,
`on_response`, `ev`, `ctx`, `Reply`, tools) is identical either way, so an
outbound transport can be added later without changing how you write an agent.
