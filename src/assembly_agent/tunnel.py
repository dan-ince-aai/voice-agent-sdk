"""Local-dev public URL.

Develop on your laptop and still let the voice layer reach you: ``serve()`` can
open a public HTTPS tunnel to the local server. We shell out to ``cloudflared``
(zero-auth "quick tunnels", no account needed) and fall back to ``ngrok`` if
that's what's installed.

This is the bridge that makes "run it and call the agent" work today without a
deploy. The destination is the outbound-worker model (no tunnel at all), but
this needs nothing server-side beyond the local process.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from typing import Optional

_TRYCF_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")


class Tunnel:
    def __init__(self, url: str, proc: Optional[subprocess.Popen], provider: str) -> None:
        self.url = url
        self.proc = proc
        self.provider = provider

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


def open_cloudflared(port: int, timeout: float = 30.0) -> Optional[Tunnel]:
    if not shutil.which("cloudflared"):
        return None
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    found: dict[str, str] = {}

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = _TRYCF_RE.search(line)
            if m and "url" not in found:
                found["url"] = m.group(0)

    threading.Thread(target=reader, daemon=True).start()

    deadline = time.time() + timeout
    while time.time() < deadline:
        if "url" in found:
            return Tunnel(found["url"], proc, "cloudflared")
        if proc.poll() is not None:  # cloudflared exited early
            break
        time.sleep(0.2)

    if proc.poll() is None:
        proc.terminate()
    return None


def open_ngrok(port: int, timeout: float = 30.0) -> Optional[Tunnel]:
    if not shutil.which("ngrok"):
        return None
    proc = subprocess.Popen(
        ["ngrok", "http", str(port), "--log", "stdout"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # ngrok exposes a local API listing active tunnels.
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as r:
                data = json.loads(r.read())
            for t in data.get("tunnels", []):
                if t.get("proto") == "https":
                    return Tunnel(t["public_url"], proc, "ngrok")
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.3)

    if proc.poll() is None:
        proc.terminate()
    return None


def open_tunnel(port: int, provider: str = "auto", timeout: float = 30.0) -> Optional[Tunnel]:
    """Open a public tunnel to ``port``. ``provider`` is ``auto`` (cloudflared
    then ngrok), ``cloudflared``, or ``ngrok``. Returns ``None`` if no tunnel
    tool is available or it didn't come up in time."""
    if provider in ("auto", "cloudflared"):
        t = open_cloudflared(port, timeout)
        if t:
            return t
    if provider in ("auto", "ngrok"):
        t = open_ngrok(port, timeout)
        if t:
            return t
    return None
