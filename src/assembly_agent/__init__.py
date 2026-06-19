"""AssemblyAI Agent SDK.

Write the logic side of a voice agent in plain code. The SDK serves it as an
OpenAI-compatible chat-completions endpoint that the AssemblyAI voice layer
talks to — you own what to say, we own the senses (STT, TTS, turn-taking,
barge-in, and the audio signals on every turn).
"""

from .agent import Agent
from .context import Context, EndCall, Message, Transfer
from .events import Event
from .gateway import Gateway, GatewayError
from .phones import PhoneError
from .reply import Reply
from .signals import Caller, Signals, Speaker, Turn
from .tools import Tool

__all__ = [
    "Agent",
    "Reply",
    "Event",
    "Context",
    "Transfer",
    "EndCall",
    "Message",
    "Signals",
    "Caller",
    "Speaker",
    "Turn",
    "Tool",
    "Gateway",
    "GatewayError",
    "PhoneError",
]

__version__ = "0.1.0"
