"""The whole agent, when you just want a managed LLM.

No handlers at all. With ``ASSEMBLYAI_API_KEY`` set, any user turn is answered
through the LLM Gateway automatically — name, voice, model, done. This is the
"land" end of the design doc's land-and-expand: start managed, add handlers
(guardrails, tools, your own logic) when you outgrow it.

    export ASSEMBLYAI_API_KEY=...
    python examples/managed.py
"""

from assembly_agent import Agent

agent = Agent(
    name="Managed Assistant",
    voice="ivy",
    greeting="Hi, how can I help?",
)

# That's it — no on_response. The Gateway handles every turn using the default
# model. To choose the model or a system prompt yourself, add an on_response and
# call ctx.llm.complete(model=..., system=...) (see examples/llm_gateway.py).

if __name__ == "__main__":
    # Opens a public URL and, with ASSEMBLYAI_API_KEY set, registers it on your
    # agent record and prints a playground link to test in the browser.
    agent.serve()
