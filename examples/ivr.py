"""Deterministic, no-LLM voice flow — a real production case.

Not every voice agent should be an LLM. Order status, store hours, appointment
confirmation, compliance-bound scripts — these want a predictable state machine
you can audit and test, with zero model latency or variance. The SDK handles
the voice; your logic is plain code.

This is a pharmacy refill line: it walks a fixed flow, tracks state per call,
and reads back a confirmation. No `ctx.llm`, no API key needed to run.

    python examples/ivr.py
"""

from assembly_agent import Agent

agent = Agent(
    name="Riverside Pharmacy",
    voice="ivy",
    greeting="Riverside Pharmacy refill line. What's your prescription number?",
)


@agent.on_response
async def respond(ev, ctx):
    s = ctx.get("state") or {"step": "rx"}
    text = ev.text.strip()
    digits = "".join(c for c in text if c.isdigit())

    if s["step"] == "rx":
        if len(digits) >= 5:
            s["rx"] = digits
            s["step"] = "confirm"
            ctx.set("state", s)
            return f"Got it, prescription {digits[-4:]}. Should I send it to your usual pharmacy? Yes or no."
        return "I didn't catch a prescription number. Could you read it digit by digit?"

    if s["step"] == "confirm":
        if "yes" in text.lower():
            s["step"] = "done"
            ctx.set("state", s)
            return ctx.end("Your refill is on the way and will be ready in about an hour. Goodbye.")
        if "no" in text.lower():
            s["step"] = "rx"
            ctx.set("state", s)
            return "No problem. What prescription number would you like to refill instead?"
        return "Sorry, was that a yes or a no?"

    return "Is there anything else I can help with?"


if __name__ == "__main__":
    agent.serve()
