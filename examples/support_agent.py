"""Flagship example — a production-shaped customer-support voice agent.

Everything wired the way you'd actually ship it:

  • personalized greeting from a CRM lookup on connect
  • your own model via the LLM Gateway (ctx.llm), model + system per turn
  • a backend lookup (order status) the handler does in plain code when needed
  • guardrails that run in *your* process: PII redaction + a refund-policy check
    (note: guardrails need the full reply, so this path uses complete(), not
    streaming — see gateway_agent.py / byo_llm.py for the low-latency streaming case)
  • emotion-aware delivery (slow + reassuring when the caller is upset)
  • mid-call language hand-off to a localized agent
  • a clean spoken goodbye, then CRM write-back when the call ends

Swap the stubbed `crm` / `orders` calls for your real systems — they're plain
functions, so it's your code, your dependencies, your network.

    export ASSEMBLYAI_API_KEY=...
    export LLM_MODEL=claude-sonnet-4-6           # optional
    export SUPPORT_PHONE_NUMBER=+14155550100     # optional, buy via the CLI first
    python examples/support_agent.py             # prints a browser playground link
"""

import os
import re
from dataclasses import dataclass

from assembly_agent import Agent, Reply

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
SYSTEM = (
    "You are Aria, a warm and efficient support agent for Acme. Answer in one or "
    "two short, spoken sentences — no lists, no formatting. If you don't know "
    "something, say you'll find out rather than guessing."
)
ORDER_ID = re.compile(r"[A-Z]{2}-\d{5}")

agent = Agent(
    name="Acme Support",
    voice="ivy",
    # A number you bought once with `assembly-agent phone buy`; assigned at boot.
    phone_number=os.environ.get("SUPPORT_PHONE_NUMBER"),
)


# --- replace these stubs with your real systems -------------------------- #
@dataclass
class Customer:
    id: str
    first_name: str
    tier: str
    refund_eligible: bool


async def crm_lookup(phone_number: str | None) -> Customer:
    return Customer(id="c_1042", first_name="Sam", tier="pro", refund_eligible=False)


async def crm_write_summary(customer: Customer, summary: str) -> None:
    ...  # POST to your CRM


async def fetch_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "out for delivery", "eta": "tomorrow"}


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def redact_pii(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[redacted]", text)
    return text
# ------------------------------------------------------------------------- #


@agent.on_call_start
async def greet(ev, ctx):
    customer = await crm_lookup(ev.caller.phone_number)
    ctx.set("customer", customer)
    return f"Hi {customer.first_name}, thanks for calling Acme. How can I help?"


@agent.on_response
async def respond(ev, ctx):
    text = ev.text.strip()

    # Caller switched languages mid-call — hand off to a localized agent.
    if ev.language and ev.language != "en":
        return ctx.transfer(f"acme-support-{ev.language}")

    # Natural end of the call.
    if any(p in text.lower() for p in ("that's all", "no thanks", "goodbye", "bye now")):
        return ctx.end("Glad I could help. Take care!")

    # If they referenced an order, look it up and feed the facts to the model.
    system = SYSTEM
    match = ORDER_ID.search(text)
    if match:
        order = await fetch_order(match.group())   # your backend — just call it
        system += (f" Known fact: order {order['order_id']} is {order['status']}, "
                   f"ETA {order['eta']}. Use it if relevant.")

    reply = await ctx.llm.complete(model=MODEL, system=system)
    reply = redact_pii(reply)

    # Guardrail: never promise a refund the customer isn't eligible for.
    customer = ctx.get("customer")
    if "refund" in reply.lower() and customer and not customer.refund_eligible:
        return "Let me bring in a specialist who can look into that for you."

    # Shape delivery from the audio signal — slow and warm when they're upset.
    if ev.signals.emotion in ("frustrated", "angry"):
        return Reply(reply, tone="reassuring", speed="slow")
    return reply


@agent.on_interrupt
async def barge_in(ev, ctx):
    ctx.cancel_pending()


@agent.on_call_end
async def wrap_up(ev, ctx):
    customer = ctx.get("customer")
    if customer:
        await crm_write_summary(customer, summary=f"Handled support call ({len(ctx.history)} turns).")


if __name__ == "__main__":
    agent.serve()
