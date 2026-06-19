"""Put a voice front-end on the text agent you already have.

Most teams already run a text agent — an LLM app with tools, RAG, and business
logic. You don't rewrite it to add voice; you call it from `on_response` and let
the SDK add the *voice reflexes* on top:

  • reuse the brain: delegate the words to your existing agent
  • add reflexes the text stack can't do: react to emotion, hand off on a
    language switch, end the call, drop work on barge-in
  • stream tokens straight to TTS for low time-to-first-audio

`your_text_agent` below stands in for your real thing (LangChain, your own
OpenAI loop, whatever). It streams tokens; we forward them.

    python examples/proxy_existing_agent.py
"""

import asyncio

from assembly_agent import Agent, Reply

agent = Agent(name="Wrapped Assistant", voice="ivy")


# --- your existing text agent (stand-in) --------------------------------- #
class YourTextAgent:
    async def stream(self, text, history):
        # Replace with your real agent. Yields tokens.
        for token in f"You asked about '{text}'. ".split():
            yield token + " "
        for token in "Here's what I found, and I can help further.".split():
            await asyncio.sleep(0.01)
            yield token + " "


your_text_agent = YourTextAgent()
# ------------------------------------------------------------------------- #


@agent.on_response
async def respond(ev, ctx):
    # Reflexes first — these are the voice value the text agent never had.
    if ev.language and ev.language != "en":
        return ctx.transfer(f"support-{ev.language}")
    if ev.signals.emotion in ("frustrated", "angry"):
        # Acknowledge instantly, in a warm tone, before the model even runs.
        return Reply("I hear you — let me sort this out.", tone="reassuring", speed="slow")

    # Reuse the brain: stream your existing agent's tokens to TTS as they arrive.
    async def tokens():
        async for token in your_text_agent.stream(ev.text, ctx.history):
            yield token

    return tokens()


@agent.on_interrupt
async def barge_in(ev, ctx):
    ctx.cancel_pending()


if __name__ == "__main__":
    agent.serve()
