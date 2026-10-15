"""Deterministic tool fixtures for scripted conversations (OpenAI tool_calls envelope; arguments are JSON text).

A fixture spec is JSON:
  {"tools": {NAME: {"rules": [{"match": {ARG: VALUE, ...}, "result": ANY}, ...], "default": ANY}
                  | {"builtin": "calculator" | "read_file" | "list_directory" | "write_file"}},
   "files": {PATH: TEXT, ...}}
"""
from __future__ import annotations

import ast
import json
import operator
from pathlib import Path


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POWER = 1000


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
            raise ValueError("exponent too large")
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


class Fixture:
    """One conversation's tools. A call returns the first rule whose `match` items all equal the arguments
    (canonical JSON equality), else `default`, else an error; results are canonical JSON text. The file tools act
    on a copy of `files` that lives as long as the Fixture."""

    def __init__(self, spec: dict) -> None:
        self._tools = spec.get("tools", {})
        self._files = dict(spec.get("files", {}))

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
            if all(key in arguments and canonical(arguments[key]) == canonical(value)
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
        path = arguments.get("path")
        if not isinstance(path, str):
            return canonical({"error": "missing path"})
        if kind == "read_file":
            if path not in self._files:
                return canonical({"error": f"no such file {path}"})
            return canonical({"path": path, "content": self._files[path]})
        if kind == "write_file":
            content = arguments.get("content")
            if not isinstance(content, str):
                return canonical({"error": "missing content"})
            self._files[path] = content
            return canonical({"path": path, "bytes": len(content.encode("utf-8"))})
        if kind == "list_directory":
            prefix = "" if path in ("", ".", "/") else path.strip("/") + "/"
            entries = set()
            for file in self._files:
                if file.startswith(prefix):
                    head, slash, _ = file[len(prefix):].partition("/")
                    entries.add(head + slash)
            if not entries and prefix:
                return canonical({"error": f"no such directory {path}"})
            return canonical({"path": path, "entries": sorted(entries)})
        return canonical({"error": f"unknown builtin {kind}"})


def load_fixture(path: Path) -> Fixture:
    return Fixture(json.loads(Path(path).read_text(encoding="utf-8")))
