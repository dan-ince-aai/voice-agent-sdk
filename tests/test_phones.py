import socket
import threading
import time

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from assembly_agent import Agent
from assembly_agent.phones import (
    PhoneError,
    assign_number,
    buy_number,
    import_number,
    list_numbers,
    release_number,
)


def _mock_phones_api():
    seen = []

    async def purchase(request):
        seen.append({"method": "POST", "path": request.url.path, "body": await request.json()})
        return Response(status_code=201)

    async def imp(request):
        seen.append({"method": "POST", "path": request.url.path, "body": await request.json()})
        return Response(status_code=201)

    async def agent_binding(request):
        entry = {"method": request.method, "number": request.path_params["number"]}
        if request.method == "PUT":
            entry["body"] = await request.json()
            seen.append(entry)
            return JSONResponse({}, status_code=200)
        seen.append(entry)
        return Response(status_code=204)

    async def one(request):
        if request.method == "DELETE":
            seen.append({"method": "DELETE", "number": request.path_params["number"]})
            return Response(status_code=204)
        return JSONResponse({"phone_number": request.path_params["number"]})

    async def collection(request):
        if request.method == "GET":
            seen.append({"method": "GET", "path": request.url.path})
            return JSONResponse({"data": [{"phone_number": "+14155550100", "agent_id": "agent_xyz"}]})
        body = await request.json()
        seen.append({"method": "POST", "path": request.url.path,
                     "auth": request.headers.get("authorization"),
                     "idem": request.headers.get("idempotency-key"), "body": body})
        return JSONResponse({"phone_number": "+14155550100", "label": body.get("label"),
                             "agent_id": body.get("agent_id")}, status_code=201)

    # Specific routes before the {number} catch-all.
    app = Starlette(routes=[
        Route("/v1/phone-numbers/purchase", purchase, methods=["POST"]),
        Route("/v1/phone-numbers/import", imp, methods=["POST"]),
        Route("/v1/phone-numbers/{number}/agent", agent_binding, methods=["PUT", "DELETE"]),
        Route("/v1/phone-numbers/{number}", one, methods=["GET", "DELETE"]),
        Route("/v1/phone-numbers", collection, methods=["GET", "POST"]),
    ])
    return app, seen


def _serve(app):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}/v1", server


# --- client functions ---------------------------------------------------- #
def test_buy_number_sends_body_auth_and_idempotency():
    app, seen = _mock_phones_api()
    api_base, server = _serve(app)
    try:
        rec = buy_number(assemblyai_api_key="aai", country_code="US", number_type="local",
                         area_code=415, label="Support line", agent_id="agent_xyz",
                         idempotency_key="idem-1", api_base=api_base)
    finally:
        server.should_exit = True
    call = seen[-1]
    assert call["method"] == "POST" and call["path"] == "/v1/phone-numbers"
    assert call["auth"] == "aai"            # raw key, no Bearer
    assert call["idem"] == "idem-1"
    assert call["body"] == {"country_code": "US", "number_type": "local",
                            "area_code": 415, "label": "Support line", "agent_id": "agent_xyz"}
    assert rec["agent_id"] == "agent_xyz"


def test_assign_number_puts_to_agent_subpath():
    app, seen = _mock_phones_api()
    api_base, server = _serve(app)
    try:
        assign_number("+14155550100", "agent_xyz", assemblyai_api_key="aai", api_base=api_base)
    finally:
        server.should_exit = True
    call = seen[-1]
    assert call["method"] == "PUT"
    assert call["number"] == "+14155550100"
    assert call["body"] == {"agent_id": "agent_xyz"}


def test_import_number_posts_trunk():
    app, seen = _mock_phones_api()
    api_base, server = _serve(app)
    try:
        import_number("+14155550132", "my-trunk.pstn.twilio.com",
                      assemblyai_api_key="aai", api_base=api_base)
    finally:
        server.should_exit = True
    call = seen[-1]
    assert call["path"] == "/v1/phone-numbers/import"
    assert call["body"] == {"phone_number": "+14155550132",
                            "termination_uri": "my-trunk.pstn.twilio.com"}


def test_list_numbers():
    app, seen = _mock_phones_api()
    api_base, server = _serve(app)
    try:
        result = list_numbers(assemblyai_api_key="aai", api_base=api_base)
    finally:
        server.should_exit = True
    assert result["data"][0]["phone_number"] == "+14155550100"


def test_release_returns_empty_on_204():
    app, seen = _mock_phones_api()
    api_base, server = _serve(app)
    try:
        result = release_number("+14155550100", assemblyai_api_key="aai", api_base=api_base)
    finally:
        server.should_exit = True
    assert result == {}
    assert seen[-1] == {"method": "DELETE", "number": "+14155550100"}


def test_requires_api_key():
    with pytest.raises(PhoneError):
        buy_number(assemblyai_api_key="", country_code="US")


# --- Agent methods (monkeypatched, no network) --------------------------- #
def test_agent_buy_uses_registered_id(monkeypatch):
    seen = {}
    monkeypatch.setattr("assembly_agent.phones.buy_number",
                        lambda **kw: seen.update(kw) or {"phone_number": "+1..."})
    agent = Agent(name="P")
    agent.remote_agent_id = "agent_999"  # as if registered
    agent.buy_phone_number(area_code=415, assemblyai_api_key="aai")
    assert seen["agent_id"] == "agent_999"
    assert seen["area_code"] == 415


def test_agent_buy_requires_registration_when_assigning(monkeypatch):
    monkeypatch.setattr("assembly_agent.phones.buy_number", lambda **kw: {})
    agent = Agent(name="P")  # no remote_agent_id
    with pytest.raises(RuntimeError):
        agent.buy_phone_number(area_code=415, assemblyai_api_key="aai")


def test_agent_buy_without_assign_needs_no_id(monkeypatch):
    seen = {}
    monkeypatch.setattr("assembly_agent.phones.buy_number",
                        lambda **kw: seen.update(kw) or {})
    agent = Agent(name="P")
    agent.buy_phone_number(assign=False, assemblyai_api_key="aai")
    assert seen["agent_id"] is None
