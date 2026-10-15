"""Deterministic tool fixtures for scripted conversations (OpenAI tool_calls envelope; arguments are JSON text).

A fixture spec is JSON:
  {"tools": {NAME: {"rules": [{"match": {ARG: VALUE, ...}, "result": ANY}, ...], "default": ANY}
                  | {"builtin": "calculator" | "read_file" | "list_directory" | "write_file"}},
   "files": {PATH: TEXT, ...}}
"""
from __future__ import annotations

import ast
import json
import math
import operator
from pathlib import Path


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POWER = 1000


_MAX_BITS = 10000


def _bits(value) -> int:
    return value.bit_length() if isinstance(value, int) else 0


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POWER or _bits(left) * abs(right) > _MAX_BITS:
                raise ValueError("result too large")
        elif isinstance(node.op, ast.Mult) and _bits(left) + _bits(right) > _MAX_BITS:
            raise ValueError("result too large")
        return _BINARY[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    try:
        value = _evaluate(ast.parse(expression.strip(), mode="eval"))
        if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
            value = int(value)
        if isinstance(value, int) and abs(value) > 10 ** 100:
            raise ValueError("result too large")
        return canonical({"result": value})
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError, AttributeError, RecursionError) as e:
        return canonical({"error": f"cannot evaluate: {type(e).__name__}"})


def normalize_path(path: str) -> str:
    """Strips leading `/` and `./` so `/a`, `./a` and `a` name one file; `.` and `/` name the root."""
    parts = [p for p in path.split("/") if p not in ("", ".")]
    return "/".join(parts)


def _normal(value):
    """Integral floats compare as ints (3.0 matches 3)."""
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normal(v) for v in value]
    return value


BUILTINS = ("calculator", "read_file", "list_directory", "write_file")


def validate_spec(spec, tool_names=()) -> list[str]:
    """Errors in a fixture spec; `tool_names` are the tools the conversation declares, each of which needs an entry."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["fixture is not a JSON object"]
    tools = spec.get("tools")
    if not isinstance(tools, dict):
        return ["fixture needs a tools object"]
    files = spec.get("files", {})
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        errors.append("files must map paths to text")
    for name in tool_names:
        if name not in tools:
            errors.append(f"tool {name} has no fixture entry")
    for name, tool in tools.items():
        if not isinstance(tool, dict):
            errors.append(f"tool {name}: entry is not an object")
        elif "builtin" in tool:
            if tool["builtin"] not in BUILTINS:
                errors.append(f"tool {name}: unknown builtin {tool['builtin']!r}")
        else:
            rules = tool.get("rules", [])
            if not isinstance(rules, list):
                errors.append(f"tool {name}: rules is not a list")
                rules = []
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict) or not isinstance(rule.get("match"), dict) or "result" not in rule:
                    errors.append(f"tool {name}: rule {i} needs a match object and a result")
            if not rules and "default" not in tool:
                errors.append(f"tool {name}: no rules and no default")
    return errors


class Fixture:
    """One conversation's tools. A call returns the first rule whose `match` items all equal the arguments
    (canonical JSON equality), else `default`, else an error; results are canonical JSON text. The file tools act
    on a copy of `files` that lives as long as the Fixture."""

    def __init__(self, spec: dict) -> None:
        self._tools = spec.get("tools", {})
        self._files = {normalize_path(k): v for k, v in spec.get("files", {}).items()}

    def call(self, name: str, arguments) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return canonical({"error": f"unknown tool {name}"})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if not isinstance(arguments, dict):
            return canonical({"error": "arguments are not a JSON object"})
        if "builtin" in tool:
            return self._builtin(tool["builtin"], arguments)
        for rule in tool.get("rules", []):
            match = rule.get("match", {})
            if all(key in arguments and canonical(_normal(arguments[key])) == canonical(_normal(value))
                   for key, value in match.items()):
                return canonical(rule["result"])
        if "default" in tool:
            return canonical(tool["default"])
        return canonical({"error": "no result for these arguments"})

    def _builtin(self, kind: str, arguments: dict) -> str:
        if kind == "calculator":
            expression = arguments.get("expression")
            if not isinstance(expression, str):
                return canonical({"error": "missing expression"})
            return calculate(expression)
        raw = arguments.get("path")
        if not isinstance(raw, str):
            return canonical({"error": "missing path"})
        path = normalize_path(raw)
        if kind == "read_file":
            if path not in self._files:
                return canonical({"error": f"no such file {raw}"})
            return canonical({"path": path, "content": self._files[path]})
        if kind == "write_file":
            content = arguments.get("content")
            if not isinstance(content, str):
                return canonical({"error": "missing content"})
            self._files[path] = content
            return canonical({"path": path, "bytes": len(content.encode("utf-8"))})
        if kind == "list_directory":
            prefix = path + "/" if path else ""
            entries = set()
            for file in self._files:
                if file.startswith(prefix):
                    head, slash, _ = file[len(prefix):].partition("/")
                    entries.add(head + slash)
            if not entries and prefix:
                return canonical({"error": f"no such directory {raw}"})
            return canonical({"path": path or ".", "entries": sorted(entries)})
        return canonical({"error": f"unknown builtin {kind}"})


def load_fixture(path: Path) -> Fixture:
    return Fixture(json.loads(Path(path).read_text(encoding="utf-8")))
