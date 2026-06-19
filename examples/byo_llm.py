"""Use case 2 — your own LLM, or a LangChain / LlamaIndex / custom workflow.

The SDK doesn't care how you produce the words. Call your own model or agent
framework in `on_response`, then return the text (or stream it). This is the
"reuse the brain you already built, add voice reflexes" path — the AssemblyAI
Gateway (`ctx.llm`) is available too, but here the brain is *yours*.

`your_workflow` stands in for your real thing — a LangChain chain, an OpenAI
call, a multi-step agent, whatever.

    python examples/byo_llm.py
"""

import asyncio

from assembly_agent import Agent, Reply


# --- your existing LLM workflow (stand-in) ------------------------------- #
class YourWorkflow:
    async def run(self, text: str, history: list) -> str:
        # e.g. return (await chain.ainvoke({"input": text, "history": history}))["output"]
        return f"(your workflow's answer to: {text})"

    async def stream(self, text: str, history: list):
        for chunk in f"Streaming answer to: {text}".split():
            await asyncio.sleep(0.01)
            yield chunk + " "


your_workflow = YourWorkflow()
# ------------------------------------------------------------------------- #

agent = Agent(name="BYO Assistant", voice="ivy", greeting="Hi, how can I help?")


@agent.on_response
async def respond(ev, ctx):
    # Voice reflexes the text stack never had — react before the model runs.
    if ev.language and ev.language != "en":
        return ctx.transfer(f"support-{ev.language}")
    if ev.signals.emotion in ("frustrated", "angry"):
        return Reply("I hear you — let me help.", tone="reassuring", speed="slow")

    # Reuse your brain. Return the text…
    return await your_workflow.run(ev.text, ctx.history)

    # …or stream it to TTS for lower latency:
    #   return your_workflow.stream(ev.text, ctx.history)


@agent.on_interrupt
async def barge_in(ev, ctx):
    ctx.cancel_pending()


if __name__ == "__main__":
    agent.serve()
