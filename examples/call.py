#!/usr/bin/env python3
"""Demo client — hold a multi-turn "call" against an Agent SDK endpoint from
your terminal. It plays the part of the voice layer: keeps one ``call_id`` for
the whole call, sends the running history each turn, and streams the reply
token-by-token the way TTS would receive it.

    python examples/call.py                         # -> http://localhost:8000/v1
    python examples/call.py <BASE_URL> [--model NAME]
    python examples/call.py https://xxxx.trycloudflare.com/v1 --model pizza-line

Type a line to speak a turn. Commands:
    /emotion <name>   attach an emotion signal to your NEXT turn (e.g. frustrated)
    /lang <code>      attach a detected language to your next turn (e.g. es)
    /new              start a fresh call (new call_id, clears history)
    /quit             exit
"""

import json
import sys
import urllib.request
import uuid


def parse_args(argv):
    base = "http://localhost:8000/v1"
    model = "agent"
    rest = list(argv)
    if "--model" in rest:
        i = rest.index("--model")
        model = rest[i + 1]
        del rest[i : i + 2]
    positional = [a for a in rest if not a.startswith("-")]
    if positional:
        base = positional[0]
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base, model


class Call:
    def __init__(self, base, model):
        self.base = base
        self.model = model
        self.call_id = uuid.uuid4().hex[:8]
        self.history = []

    def reset(self):
        self.call_id = uuid.uuid4().hex[:8]
        self.history = []

    def _post_stream(self, messages, enrichment):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "assemblyai": {"call_id": self.call_id, **enrichment},
        }
        req = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=60)

    def turn(self, messages, enrichment):
        """Stream one reply. Returns (text, extension_dict)."""
        text = ""
        ext = None
        print("  agent: ", end="", flush=True)
        try:
            resp = self._post_stream(messages, enrichment)
        except Exception as e:  # noqa: BLE001
            print(f"[request failed: {e}]")
            return "", None
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choice = chunk["choices"][0]
            if choice.get("assemblyai"):
                ext = choice["assemblyai"]
            piece = choice.get("delta", {}).get("content")
            if piece:
                text += piece
                print(piece, end="", flush=True)
        print()
        return text, ext

    @staticmethod
    def show_extension(ext):
        if not ext:
            return
        action = ext.get("action")
        if action == "transfer":
            print(f"  ↳ [transfer to {ext.get('transfer_to')}]")
        elif action == "passthrough":
            print("  ↳ [passthrough: a managed LLM would generate this turn]")
        elif ext.get("delivery"):
            print(f"  ↳ [delivery: {ext['delivery']}]")


def main():
    base, model = parse_args(sys.argv[1:])
    call = Call(base, model)

    print(f"connected to {base}")
    print(f"call_id={call.call_id}  model={model}")
    print("type a turn, or:  /emotion <e>   /lang <code>   /new   /quit\n")

    # Empty messages = the call connecting -> greeting (on_call_start).
    text, ext = call.turn([], {})
    if text:
        call.history.append({"role": "assistant", "content": text})
    call.show_extension(ext)

    pending = {}  # one-shot enrichment for the next turn
    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/new":
            call.reset()
            print(f"-- new call ({call.call_id}) --")
            text, ext = call.turn([], {})
            if text:
                call.history.append({"role": "assistant", "content": text})
            call.show_extension(ext)
            pending = {}
            continue
        if line.startswith("/emotion"):
            parts = line.split(maxsplit=1)
            pending.setdefault("signals", {})["emotion"] = parts[1] if len(parts) > 1 else "neutral"
            print(f"-- next turn: signals={pending['signals']} --")
            continue
        if line.startswith("/lang"):
            parts = line.split(maxsplit=1)
            pending["language"] = parts[1] if len(parts) > 1 else "en"
            print(f"-- next turn: language={pending['language']} --")
            continue

        call.history.append({"role": "user", "content": line})
        text, ext = call.turn(call.history, pending)
        if text:
            call.history.append({"role": "assistant", "content": text})
        call.show_extension(ext)
        pending = {}


if __name__ == "__main__":
    main()
