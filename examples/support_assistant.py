"""The reference example from the design doc, made runnable.

    python examples/support_assistant.py
    # -> OpenAI endpoint at http://localhost:8000/v1

Stand-ins for the CRM / LLM are inlined so it runs with no external services.
Swap them for the real thing.
"""

from assembly_agent import Agent, Reply

agent = Agent(name="Support Assistant", voice="ivy")


# --- stand-ins you'd replace with real services -------------------------- #
class _Customer:
    def __init__(self, first_name, city, refund_eligible):
        self.first_name = first_name
        self.city = city
        self.refund_eligible = refund_eligible


async def crm_lookup(phone_number):
    return _Customer("Sam", "Austin", refund_eligible=False)


async def my_llm(text, history, customer):
    return f"You said: {text}. Happy to help with that."


def redact_pii(text):
    return text


def promises_refund(text):
    return "refund" in text.lower()


# --- handlers ------------------------------------------------------------- #
@agent.on_call_start
async def greet(ev, ctx):
    customer = await crm_lookup(ev.caller.phone_number)
    ctx.set("customer", customer)
    return (
        f"Hi {customer.first_name}, how's the weather over in "
        f"{customer.city}? What can I do for you?"
    )


@agent.on_response
async def respond(ev, ctx):
    # Say goodbye and hang up when they're done.
    if any(w in ev.text.lower() for w in ("bye", "goodbye", "that's all", "thank you")):
        return ctx.end("Glad I could help. Take care, goodbye.")

    # Language is detected per turn, so a mid-call switch is catchable.
    if ev.language and ev.language != "en":
        return ctx.transfer(f"support-{ev.language}")

    # Shape delivery, not just words — slow and warm when they're upset.
    if ev.signals.emotion == "frustrated":
        return Reply("I hear you, let me fix this.", tone="reassuring", speed="slow")

    reply = await my_llm(ev.text, history=ctx.history, customer=ctx.get("customer"))

    # Your own guardrails — just code in your process.
    reply = redact_pii(reply)
    customer = ctx.get("customer")
    if promises_refund(reply) and customer and not customer.refund_eligible:
        return "Let me connect you with someone who can help with that."
    return reply


@agent.on_interrupt
async def stopped(ev, ctx):
    # They cut in — drop whatever we were doing.
    ctx.cancel_pending()


@agent.on_call_end
async def wrap_up(ev, ctx):
    # Fires however the call ended (ctx.end, caller hung up, line dropped).
    # Write back to the CRM here, fire a webhook, etc.
    customer = ctx.get("customer")
    print(f"call ended for {getattr(customer, 'first_name', 'unknown')}; {len(ctx.history)} turns")


@agent.tool
async def lookup_order(order_id: str) -> dict:
    "Look up an order's status."
    return {"order_id": order_id, "status": "shipped"}


if __name__ == "__main__":
    # serve() opens a public URL by default so the voice layer can reach your
    # laptop with no deploy. Pass tunnel=False for a local-only endpoint.
    agent.serve()
