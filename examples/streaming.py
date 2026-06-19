"""Streaming: yield tokens and they're forwarded to TTS as they arrive, for
lower time-to-first-audio.

`on_response` returns an async generator; the SDK turns it into an OpenAI SSE
stream automatically.

    python examples/streaming.py
"""

import asyncio

from assembly_agent import Agent

agent = Agent(name="Storyteller", voice="ivy")


@agent.on_response
async def respond(ev, ctx):
    async def tokens():
        for word in f"Here is a thought about {ev.text}: ".split():
            yield word + " "
        for word in "patience and momentum tend to win in the end.".split():
            await asyncio.sleep(0.02)  # stand in for model latency
            yield word + " "

    return tokens()


if __name__ == "__main__":
    agent.serve()
