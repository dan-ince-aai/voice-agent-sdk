"""Bring-your-own-LLM, where the LLM is AssemblyAI's **LLM Gateway** — now a
native primitive of the SDK.

The Gateway is an OpenAI-compatible endpoint with one key across Claude / GPT /
Gemini / Qwen / Kimi, plus automatic retries and fallbacks. The agent already
authenticates with your AssemblyAI key, so the Gateway reuses it — no second
credential, nothing hardcoded. In a handler it's just ``ctx.llm``.

    export ASSEMBLYAI_API_KEY=...        # the only secret, from the environment
    python examples/llm_gateway.py

Optional env:
    LLM_GATEWAY_MODEL   default "claude-sonnet-4-6"   (or pass llm=... below)
    LLM_GATEWAY_URL     default US; set to the EU base_url for EU data residency
"""

import os

from assembly_agent import Agent, Reply

agent = Agent(
    name="Gateway Assistant",
    voice="ivy",
    # The model is config; the key comes from ASSEMBLYAI_API_KEY automatically.
    llm=os.environ.get("LLM_GATEWAY_MODEL", "claude-sonnet-4-6"),
    instructions=(
        "You are a warm, concise voice assistant. Speak in one or two short "
        "sentences. Never use lists or formatting — this is spoken aloud."
    ),
    greeting="Hey, I'm running through the LLM Gateway. What can I help with?",
)


@agent.on_response
async def respond(ev, ctx):
    # React before we ever hit the model — slow and warm when they're upset,
    # no LLM round-trip needed. This is the signal loop wrapping the LLM.
    if ev.signals.emotion == "frustrated":
        return Reply("I hear you — let me help with that.", tone="reassuring", speed="slow")

    # Answer the turn through the Gateway. ctx.llm already has the call history
    # and the agent's instructions as the system prompt.
    return await ctx.llm.complete()

    # Streaming variant (for models that support it), lower time-to-first-audio:
    #   return ctx.llm.stream()


@agent.on_interrupt
async def stopped(ev, ctx):
    ctx.cancel_pending()


if __name__ == "__main__":
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        print("Set ASSEMBLYAI_API_KEY first:  export ASSEMBLYAI_API_KEY=...")
    agent.serve()
