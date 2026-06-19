"""Run the voice agent *inside* your existing web service.

The agent is just an ASGI app, so mount it next to your other routes — it
shares your process, your dependencies, your database connections. No separate
service to deploy, and `on_response` can call straight into the rest of your app.

Here we mount into a small Starlette app; FastAPI is identical
(`app.mount("/voice", agent.app)`).

    uvicorn examples.mounted:app --port 8000
    # the voice agent now lives at  /voice/v1/chat/completions

Because the host runs the server (not serve()), wire it to your AssemblyAI agent
with your public URL — once, from the CLI or a one-off script:

    assembly-agent agents list                       # find/confirm the agent
    python -c "from examples.mounted import agent; agent.register('https://your-host/voice')"
"""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from assembly_agent import Agent

agent = Agent(name="Embedded Support", voice="ivy")


@agent.on_response
async def respond(ev, ctx):
    # Same process as the rest of your app — call your services directly here.
    return await ctx.llm.complete(model="claude-sonnet-4-6", system="Be helpful and brief.")


# --- your existing application ------------------------------------------- #
async def home(request):
    return JSONResponse({"service": "my-app", "voice": "/voice/v1"})


app = Starlette(routes=[Route("/", home)])

# One line to add the voice agent to your service.
app.mount("/voice", agent.app)
