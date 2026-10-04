"""Regenerates api/src/test/resources/qwen-chat-template-golden.json from the checkpoint chat template.

Renders with jinja2 the way Hugging Face `apply_chat_template` does: an immutable sandbox with
`trim_blocks`/`lstrip_blocks`, the `loopcontrols` extension, `raise_exception`, and a `tojson` filter that
calls `json.dumps(ensure_ascii=False)` with default separators. Every case uses `add_generation_prompt=True`.
`cases`, `tool_cases` and `errors` render with `enable_thinking=False`; each `thinking_cases` entry names the thinking
variables it sets (`enable_thinking`, `reasoning_effort`, `preserve_thinking`) and leaves the others undefined.

Usage: python render_qwen_chat_template_golden.py TEMPLATE.jinja OUTPUT.json  (requires jinja2 >= 3.1)
"""

import json
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def compile_template(source):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    return env.from_string(source)


def render(template, messages, tools=None, thinking=None):
    if thinking is None:
        thinking = {"enable_thinking": False}
    return template.render(messages=messages, tools=tools, add_generation_prompt=True, **thinking)


CASES = [
    ("user_only", [{"role": "user", "content": "Hello"}]),
    ("system_and_user", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name a prime."}]),
    ("blank_system", [{"role": "system", "content": "  \n "}, {"role": "user", "content": "Hi"}]),
    (
        "multi_turn",
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
    ),
    (
        "assistant_prefill_history",
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ],
    ),
    (
        "text_parts",
        [{"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]}],
    ),
    ("whitespace_trim", [{"role": "user", "content": "\u00a0 padded \n\t\u0085"}]),
    ("unicode", [{"role": "user", "content": "Translate: café → 日本語 \U0001f600"}]),
    (
        "assistant_null_content",
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": None}, {"role": "user", "content": "y"}],
    ),
    ("assistant_last", [{"role": "user", "content": "Say hi"}, {"role": "assistant", "content": "Hi"}]),
]

ERRORS = [
    ("no_user", [{"role": "system", "content": "s"}]),
    ("system_not_first", [{"role": "user", "content": "u"}, {"role": "system", "content": "s"}]),
    ("tool_response_only", [{"role": "user", "content": "<tool_response>x</tool_response>"}]),
]

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. \"Paris\"."},
                "days": {"type": "integer", "minimum": 1, "maximum": 16},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["city"],
        },
    },
}

SEARCH = {
    "type": "function",
    "function": {
        "name": "search-docs_v2",
        "description": "Search über docs → \U0001f50e\ttab\u0001ctl \\ back/slash </tools>",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "threshold": {"type": "number", "default": 0.5, "exclusiveMinimum": 1e-05, "maximum": 1e16},
                "limits": {"type": "array", "items": {"type": "integer"}, "default": [1, -2, 3.25, -0.0, 1e300]},
                "filters": {"type": ["object", "null"], "additionalProperties": True},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

NO_PARAMS = {"type": "function", "function": {"name": "ping"}}

TOOL_CASES = [
    ("tools_user_only", [WEATHER], [{"role": "user", "content": "Weather in Paris?"}]),
    (
        "tools_with_system",
        [WEATHER, SEARCH, NO_PARAMS],
        [{"role": "system", "content": "  Be brief.  "}, {"role": "user", "content": "Hi"}],
    ),
    ("tools_blank_system", [NO_PARAMS], [{"role": "system", "content": " \n"}, {"role": "user", "content": "Hi"}]),
    (
        "assistant_single_call_and_response",
        [WEATHER],
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "  {\"temp\": 21}  "},
        ],
    ),
    (
        "assistant_content_multi_call_grouped_responses",
        [WEATHER, SEARCH, NO_PARAMS],
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Do things."},
            {
                "role": "assistant",
                "content": "  Let me check.\n",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Line one\nline two", "units": "metric"}},
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {
                            "name": "search-docs_v2",
                            "arguments": {
                                "query": "",
                                "threshold": 0.25,
                                "limits": [1, 2],
                                "filters": {"kéy": "v\"q", "n": None, "t": True},
                            },
                        },
                    },
                    {"id": "c", "type": "function", "function": {"name": "ping", "arguments": {}}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "sunny"},
            {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "r1 "}, {"type": "text", "text": "r2"}]},
            {"role": "tool", "tool_call_id": "c", "content": ""},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Thanks"},
        ],
    ),
    (
        "tool_history_without_tools",
        None,
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": {"a": [1.5, "s"]}}}]},
            {"role": "tool", "tool_call_id": "x", "content": "r"},
        ],
    ),
]

REPLAYED_REASONING = [
    {"role": "system", "content": "Be careful."},
    {"role": "user", "content": "First question?"},
    {"role": "assistant", "reasoning_content": "\n  Think about the first.\n", "content": "First answer."},
    {"role": "user", "content": "Second question?"},
    {
        "role": "assistant",
        "reasoning_content": "Plan a lookup.",
        "content": "",
        "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}],
    },
    {"role": "tool", "tool_call_id": "t1", "content": "cold"},
]

# (name, tools, messages, thinking variables)
THINKING_CASES = [
    ("default_user_only", None, [{"role": "user", "content": "Hello"}], {}),
    ("enabled_explicit", None, [{"role": "user", "content": "Hello"}], {"enable_thinking": True}),
    ("xhigh_with_system", None, CASES[1][1], {"reasoning_effort": "xhigh"}),
    ("medium_user_only", None, [{"role": "user", "content": "Hello"}], {"reasoning_effort": "medium"}),
    ("medium_with_system", None, CASES[1][1], {"reasoning_effort": "medium"}),
    ("low_with_system", None, CASES[1][1], {"reasoning_effort": "low"}),
    ("low_blank_system", None, CASES[2][1], {"reasoning_effort": "low"}),
    ("disabled_with_effort", None, CASES[1][1], {"enable_thinking": False, "reasoning_effort": "low"}),
    ("default_multi_turn", None, CASES[3][1], {}),
    ("replayed_reasoning", [WEATHER], REPLAYED_REASONING, {}),
    ("replayed_reasoning_not_preserved", [WEATHER], REPLAYED_REASONING, {"preserve_thinking": False}),
    (
        "not_preserved_after_last_query",
        None,
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "reasoning_content": "r1", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "reasoning_content": "r2", "content": "d"},
        ],
        {"preserve_thinking": False, "reasoning_effort": "low"},
    ),
    ("tools_low", [WEATHER, NO_PARAMS], TOOL_CASES[1][2], {"reasoning_effort": "low"}),
    ("tools_medium", [WEATHER], TOOL_CASES[0][2], {"reasoning_effort": "medium"}),
    ("tools_default_blank_system", [NO_PARAMS], TOOL_CASES[2][2], {}),
]

PYTHON_JSON = [
    "plain",
    "quote\" back\\ slash/ nl\n cr\r tab\t bs\b ff\f nul\u0000 us\u001f del\u007f nbsp\u00a0 ls\u2028 emoji\U0001f600",
    0,
    -7,
    12345678901234567890,
    0.0,
    -0.0,
    1.0,
    0.1,
    -2.5,
    1e-05,
    0.0001,
    123456789012345.6,
    1e16,
    9999999999999998.0,
    1.7976931348623157e308,
    5e-324,
    True,
    False,
    None,
    [],
    {},
    [1, "a", None, [True]],
    {"b": 1, "a": {"z": [], "y": {}}},
]


def main():
    template_path, output_path = sys.argv[1:3]
    with open(template_path, encoding="utf-8") as source:
        template = compile_template(source.read())
    golden = {
        "cases": [{"name": name, "messages": messages, "expected": render(template, messages)} for name, messages in CASES],
        "errors": [],
        "tool_cases": [
            {"name": name, "tools": tools, "messages": messages, "expected": render(template, messages, tools)}
            for name, tools, messages in TOOL_CASES
        ],
        "thinking_cases": [
            {"name": name, "tools": tools, "messages": messages, "thinking": thinking,
             "expected": render(template, messages, tools, thinking)}
            for name, tools, messages, thinking in THINKING_CASES
        ],
        "python_json": [{"value": value, "expected": _tojson(value)} for value in PYTHON_JSON],
    }
    for name, messages in ERRORS:
        try:
            render(template, messages)
        except jinja2.exceptions.TemplateError as error:
            golden["errors"].append({"name": name, "messages": messages, "message": str(error)})
        else:
            raise AssertionError("expected " + name + " to be rejected")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(golden, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
