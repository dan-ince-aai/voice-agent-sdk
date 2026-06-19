"""Use case 1 — the LLM Gateway as your model.

The simplest real agent: every turn answered by a model on AssemblyAI's LLM
Gateway (Claude / GPT / Gemini / Qwen / … behind one key), with the model and
system prompt chosen in your handler. No second credential — it reuses
ASSEMBLYAI_API_KEY.

    export ASSEMBLYAI_API_KEY=...
    export LLM_MODEL=claude-sonnet-4-6        # optional
    python examples/gateway_agent.py
"""

import os

from assembly_agent import Agent

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
SYSTEM = "You are a friendly, concise voice assistant. Reply in one or two spoken sentences."

agent = Agent(name="Gateway Assistant", voice="ivy", greeting="Hi, how can I help?")


@agent.on_response
async def respond(ev, ctx):
    # ctx.llm already carries the call history; you pick the model + system here.
    return await ctx.llm.complete(model=MODEL, system=SYSTEM)
    # Lower time-to-first-audio (models that support it):
    #   return ctx.llm.stream(model=MODEL, system=SYSTEM)


if __name__ == "__main__":
    agent.serve()
