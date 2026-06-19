"""Use case 4 — your knowledge base (RAG) + the LLM Gateway.

Retrieve from your own vector store / search index, ground the answer in the
hits, and let the Gateway phrase it. Retrieval is your code (swap the stub for
your real search); generation is `ctx.llm`. The handler is the RAG pipeline:
retrieve → stuff into the system prompt → answer.

    export ASSEMBLYAI_API_KEY=...
    python examples/rag.py
"""

import os

from assembly_agent import Agent

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

agent = Agent(
    name="Docs Assistant",
    voice="ivy",
    greeting="Hi, ask me anything about our product.",
)


# --- your retrieval (stand-in for a vector DB / search index) ------------ #
KB = {
    "refund": "Refunds are available within 30 days with proof of purchase.",
    "hours": "Support is open 9am–6pm Pacific, Monday through Friday.",
    "shipping": "Standard shipping takes 3–5 business days; express is next-day.",
    "warranty": "All hardware carries a 1-year limited warranty.",
}


async def search_kb(query: str, k: int = 3) -> list[str]:
    # Replace with your embeddings/search. Returns the top-k passages.
    q = query.lower()
    hits = [text for key, text in KB.items() if key in q]
    return hits[:k] or list(KB.values())[:k]
# ------------------------------------------------------------------------- #


@agent.on_response
async def respond(ev, ctx):
    passages = await search_kb(ev.text)
    facts = " ".join(f"- {p}" for p in passages)
    system = (
        "You are a product support assistant. Answer using ONLY the facts below. "
        "If they don't cover the question, say you'll check and follow up. "
        "Keep it to one or two spoken sentences.\n" + facts
    )
    return await ctx.llm.complete(model=MODEL, system=system)


if __name__ == "__main__":
    agent.serve()
