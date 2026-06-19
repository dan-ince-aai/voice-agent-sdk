import asyncio
from typing import Optional

import pytest

from assembly_agent import Agent
from assembly_agent.tools import build_tool


def test_schema_inference_types_and_required():
    def search(query: str, limit: int = 10, fuzzy: bool = False, tags: list[str] = None) -> dict:
        "Search the catalog."
        return {}

    tool = build_tool(search)
    assert tool.name == "search"
    assert tool.description == "Search the catalog."
    props = tool.parameters["properties"]
    assert props["query"] == {"type": "string"}
    assert props["limit"] == {"type": "integer"}
    assert props["fuzzy"] == {"type": "boolean"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    # Only `query` has no default and is non-optional.
    assert tool.parameters["required"] == ["query"]


def test_optional_not_required():
    def f(a: str, b: Optional[int] = None):
        "f"
        return None

    tool = build_tool(f)
    assert tool.parameters["required"] == ["a"]
    assert tool.parameters["properties"]["b"] == {"type": "integer"}


def test_ctx_param_excluded():
    def f(ctx, name: str):
        "f"
        return name

    tool = build_tool(f)
    assert "ctx" not in tool.parameters["properties"]
    assert tool.parameters["required"] == ["name"]


def test_openai_schema_shape():
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    schema = build_tool(add).openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert schema["function"]["parameters"]["type"] == "object"


def test_run_tool_sync_and_async():
    agent = Agent(name="T")

    @agent.tool
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    @agent.tool
    async def mul(a: int, b: int) -> int:
        "Multiply."
        return a * b

    assert asyncio.run(agent.run_tool("add", {"a": 2, "b": 3})) == 5
    assert asyncio.run(agent.run_tool("mul", {"a": 2, "b": 3})) == 6
    assert len(agent.tool_schemas()) == 2


def test_run_unknown_tool_raises():
    agent = Agent(name="T")
    with pytest.raises(KeyError):
        asyncio.run(agent.run_tool("nope", {}))
