# Agent SDK

> **Independent prototype** — not an official AssemblyAI product. A design
> sketch of a logic-side voice-agent SDK; AssemblyAI references describe
> intended integration points.

Write the **logic** of a voice agent in Python. The SDK serves it as an
**OpenAI-compatible endpoint** that the AssemblyAI voice layer calls on every
turn. You decide what to say; the voice layer owns the senses — speech-to-text,
text-to-speech, turn-taking, barge-in, and the audio signals (emotion, prosody,
speaker, language) handed to you on each turn.

```python
from assembly_agent import Agent, Reply

agent = Agent(name="Assistant", voice="ivy")

@agent.on_response
async def respond(ev, ctx):
    if ev.signals.emotion == "frustrated":
        return Reply("I hear you, let me fix this.", tone="reassuring", speed="slow")
    return await ctx.llm.complete(model="claude-sonnet-4-6")

agent.serve()
```

`on_response` is just code — call an LLM, walk a decision tree, hit a database,
run an open-source model. The voice layer never knows the difference, because
the seam between them is the OpenAI chat-completions schema. You never serve a
route, parse the schema, or touch a socket.

---

## Quickstart

```sh
pip install -e .
python examples/managed.py        # needs ASSEMBLYAI_API_KEY
```

On start it prints a public URL:

```
  Public endpoint (via cloudflared): https://blue-forest-1234.trycloudflare.com/v1
  Point your voice agent's LLM base_url here (model: 'managed-assistant').
```

Paste that `…/v1` into your voice agent's LLM `base_url`, place a call, and it
runs against the code on your machine. To try it without a voice call, talk to
it from your terminal:

```sh
python examples/call.py https://blue-forest-1234.trycloudflare.com/v1
```

---

## The three events

That's the whole surface. Each handler gets the event `ev` and the per-call
context `ctx`, and may be `async` or sync.

```python
@agent.on_call_start      # call connects, before anyone speaks → greet
@agent.on_response        # every finalized user turn → return what to say
@agent.on_interrupt       # caller cut in → ctx.cancel_pending()
@agent.on_call_end        # call is over → clean up, save to CRM, fire a webhook
```

One more when you need it: `@agent.on_speaker_change` (diarization changed —
handoff, third party on the line).

### `ev` — what came in

| Field         | What it is                                                          |
| ------------- | ------------------------------------------------------------------- |
| `ev.text`     | The finalized transcript for this turn.                             |
| `ev.signals`  | `emotion`, `sentiment`, `prosody` (`pitch`/`energy`/`pace`), `hesitance`, `accent`, `confidence`. |
| `ev.language` | Detected language, per turn — catch a mid-call switch.              |
| `ev.caller`   | `phone_number`, `direction`, `from`/`to` (on call start).           |
| `ev.speaker`  | Diarization `id` and `confidence`.                                  |
| `ev.turn`     | `interruption`, `overlap`, `latency`.                               |

Missing fields read back as `None`, so you can branch on them directly:
`if ev.signals.emotion == "frustrated": ...`.

### `ctx` — the call

| Member                              | What it does                                   |
| ----------------------------------- | ---------------------------------------------- |
| `ctx.set(k, v)` / `ctx.get(k)`      | Key/value store that lives for the whole call. |
| `ctx.history`                       | The running message list (`.role` / `.content`). |
| `ctx.llm`                           | The LLM Gateway, bound to this call (below).   |
| `ctx.transfer(name, reason=None)`   | Route the call to another agent.               |
| `ctx.end(text="", reason=None)`     | Speak a goodbye, then hang up.                 |
| `ctx.cancel_pending()`              | Drop in-flight work (on barge-in).             |
| `await ctx.call_tool(name, **kw)`   | Run a registered tool.                          |

---

## Returning a response

| Return                          | Effect                                                       |
| ------------------------------- | ------------------------------------------------------------ |
| `str`                           | Spoken as-is.                                                |
| `Reply(text, tone=, speed=, …)` | Same words, shaped delivery (`tone`, `speed`, `pitch`, `emphasis`, `pause`, `voice`). |
| an async/sync generator         | Tokens streamed to TTS as they arrive (lower latency).       |
| `ctx.transfer(...)`             | Hand the call to another agent.                              |
| `ctx.end("Bye!")`               | Speak the goodbye, then end the call.                        |
| `None`                          | You didn't write the reply — fall back to the LLM (see below). |

The voice layer reads delivery, transfer, end, and fallback hints from an
`assemblyai` field on the response, alongside a standard OpenAI body — so any
OpenAI client still gets valid output.

### Ending the call

Return `ctx.end(...)` to hang up. The goodbye is spoken, the session ends, and
`on_call_end` fires for cleanup:

```python
@agent.on_response
async def respond(ev, ctx):
    if "goodbye" in ev.text.lower():
        return ctx.end("Thanks for calling. Take care!")
    return await ctx.llm.complete(model="claude-sonnet-4-6")

@agent.on_call_end
async def wrap_up(ev, ctx):
    await crm.save(ctx.get("customer"), ctx.history)   # runs however the call ended
```

`on_call_end` fires whenever the call ends — your `ctx.end(...)`, the caller
hanging up, or the line dropping — so it's the one place for cleanup.

---

## The LLM Gateway

[AssemblyAI's LLM Gateway](https://assemblyai.com/docs/llm-gateway) is built in
as `ctx.llm` — one OpenAI-compatible interface across Claude / GPT / Gemini /
Qwen / Kimi, with retries and fallbacks. It reuses your `ASSEMBLYAI_API_KEY`, so
there's no second credential and nothing in code.

```python
@agent.on_response
async def respond(ev, ctx):
    return await ctx.llm.complete(model="claude-sonnet-4-6", system="Be warm and concise.")
    # streaming: return ctx.llm.stream(model="gpt-4o")
```

`ctx.llm` already carries the call history. You choose the **model and system
prompt per call** — both are response-generation decisions, so they live in your
handler, not in the agent config. (Agent config is identity and senses only:
name, voice, greeting, TTS `prompt`, plus `api_key` / `llm_base_url` for EU.)

**No handler? It still works.** With `ASSEMBLYAI_API_KEY` set and no
`on_response`, every turn is answered through the Gateway automatically — the
whole agent is a name and a voice (`examples/managed.py`). Returning `None` from
a handler does the same thing, after you've added context or redacted: the
Gateway writes the words, or the turn passes through to the voice layer's LLM if
no key is set.

---

## Tools

```python
@agent.tool
async def lookup_order(order_id: str) -> dict:
    "Look up an order's status."
    return await db.get_order(order_id)
```

The JSON schema is inferred from the type hints; the first line of the docstring
is the description. Call them with `await ctx.call_tool("lookup_order",
order_id="123")`, or get OpenAI-format schemas via `agent.tool_schemas()`.

---

## Going live

`agent.serve()` opens a public HTTPS tunnel to your laptop by default — run it
and it's callable, no deploy. It uses
[`cloudflared`](https://github.com/cloudflare/cloudflared) (no account needed);
`ngrok` works too.

```sh
brew install cloudflared        # one-time; without it serve() stays local
```

| Call                                              | Use                                   |
| ------------------------------------------------- | ------------------------------------- |
| `serve()`                                         | Public URL to your laptop. Take real calls while you develop. |
| `serve(tunnel=False)`                             | Local only (`http://localhost:8000/v1`). |
| `serve(tunnel=False, host="0.0.0.0", port=$PORT)` | Behind your own address, for a deploy. |

> Tunnel URLs are random and disappear when you stop the process. For a stable
> address, deploy or use a named cloudflared tunnel.

---

## Connecting it to a voice agent (BYO LLM)

The voice layer reaches the SDK through the agent record's `llm` config — point
its `base_url` at the SDK and the voice layer calls it as a chat-completions
endpoint:

```json
"llm": [{ "base_url": "https://abc.trycloudflare.com/v1", "model": "support-assistant", "api_key": "•••" }]
```

Let `serve()` wire it for you — needs `ASSEMBLYAI_API_KEY`:

```python
agent.serve(register=True)
```

Each run it: mints a **fresh ingress key**, points the record's `base_url` at
the current tunnel URL, and **updates the agent with the same name** — it lists
your agents (`GET /v1/agents`), and if one already has this name it `PUT`s that
record, otherwise it creates one. So the name is the identity; re-running just
updates "the agent called X". You never copy the ephemeral URL or manage the
secret by hand. Do it explicitly with `agent.register("https://…/v1")` (pass
`agent_id=` to target a specific record, e.g. after a rename), or the
`POST/PUT /v1/agents` curl.

What the SDK handles for you, from the agent-record contract:

- **HTTPS, public-DNS host.** `register()` rejects `http://localhost` — use the
  tunnel (`serve()`) or a deployed URL. `model` and `api_key` must be non-empty.
- **The `api_key` is a rotating shared secret**, not your AssemblyAI key — it's
  what the voice layer presents when calling the SDK. A new one is minted each
  `serve()` (set `Agent(..., ingress_key=...)` to pin your own instead), and the
  server **rejects requests that don't present it**, so your public URL isn't
  open. It's encrypted at rest and never returned on reads.
- **Updates merge.** `PUT` is a full replace, so the SDK fetches the record
  first and swaps in only the `llm` block — your `input` / `output` / `tools`
  config is preserved.

The record's `model` is sent in each request's `model` field; your handler can
use it or ignore it (the model you actually call is `ctx.llm.complete(model=…)`).

---

## Examples

| File                            | Shows                                                  |
| ------------------------------- | ------------------------------------------------------ |
| `examples/managed.py`           | No handlers — the Gateway answers every turn.          |
| `examples/llm_gateway.py`       | `ctx.llm` with an emotion-shaped short-circuit.        |
| `examples/support_assistant.py` | CRM lookup, guardrails, `Reply`, transfer, a tool.     |
| `examples/decision_tree.py`     | Bring-your-own-logic with **no LLM** (rules + state).  |
| `examples/streaming.py`         | Token streaming via an async generator.                |
| `examples/call.py`              | Terminal client to talk to any of them.                |

Run tests with `pip install -e ".[dev]" && pytest`.

---

## Under the hood

The voice layer sends a normal OpenAI request and attaches the audio signals
under an `assemblyai` key (top-level or on the last user message):

```json
{
  "model": "assistant",
  "messages": [{ "role": "user", "content": "I want a refund" }],
  "assemblyai": {
    "call_id": "abc-123",
    "language": "en",
    "signals": { "emotion": "frustrated", "prosody": { "pace": "fast" } },
    "caller": { "phone_number": "+1...", "direction": "inbound" }
  }
}
```

`call_id` (or an `X-Call-Id` header) ties `ctx` state across turns and is
dropped on `on_call_end`. Side-channel events arrive as requests with
`assemblyai.event` set. The server exposes `POST /v1/chat/completions`,
`GET /v1/models`, and `GET /healthz`.

The signal field names and the `assemblyai` envelope are a convention defined
here — confirm the exact wire contract when integrating with the live voice
layer.
