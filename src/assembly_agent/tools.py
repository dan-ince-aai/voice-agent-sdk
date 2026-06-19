"""``@agent.tool`` — register a plain function and infer its JSON schema from
the signature.

The schema is built from type hints and the docstring so you never hand-write
it. Tools can be called directly from your handlers (``await
ctx.call_tool(...)``) and are exposed in OpenAI tool format via
``agent.tool_schemas()`` so a managed LLM can be told about them.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> dict:
    """Map a Python annotation to a JSON-schema fragment. Best-effort: unknown
    types fall back to ``string``."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)

    # Optional[X] / Union[X, None] -> schema of X
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {}  # genuine multi-type union: leave open

    if origin in (list, tuple):
        args = get_args(annotation)
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    return {"type": _PY_TO_JSON.get(annotation, "string")}


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict  # JSON schema (OpenAI "parameters" object)

    async def run(self, args: dict) -> Any:
        result = self.func(**args)
        if inspect.isawaitable(result):
            return await result
        return result

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def build_tool(func: Callable[..., Any]) -> Tool:
    name = func.__name__
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n\n")[0].strip() or name

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = getattr(func, "__annotations__", {}) or {}

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls", "ctx"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, param.annotation)
        properties[pname] = _json_type(annotation)
        has_default = param.default is not inspect.Parameter.empty
        if not has_default and not _is_optional(annotation):
            required.append(pname)

    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return Tool(name=name, description=description, func=func, parameters=parameters)
