#!/usr/bin/env python3
"""Send one fixed request set to the Java server and to the Rust server, and report every difference in the replies.

Usage:
  tools/api_diff.py --java-config FILE.yaml --rust-config FILE.yaml --out-dir DIR --java-jar JAR --rust-binary BIN
                    --cuda-library LIB [--who NAME] [--launcher system-run] [--only java|rust|compare]
  tools/api_diff.py --out-dir DIR --only compare            compare the replies saved earlier (no device)
  tools/api_diff.py --self-test                             the tests of the normaliser and the stream parser

One server runs at a time, each under one launcher grant (`system-run --gpu shared:14 ...`): the harness starts it on a
free local port (never the port of a running service), waits for `/health`, sends the requests of
`api_diff_requests.py` one after the other, saves every raw reply under `OUT/<side>/<request>.json`, stops the server
with SIGTERM (its pid is written by a small `sh` wrapper, so the signal goes to the server and nothing else) and records
its exit code. Both servers read the given settings file; the harness overrides, by environment, only the port, the
context length (so a too-long prompt is cheap) and the body limit.

The comparison reads the saved replies and sorts what differs into two classes. MUST MATCH, the model's input and
output: the prompt token count, the generated text, the reasoning text, the names and arguments of tool calls, the
finish reason and the completion token count, of the whole and the streamed reply on all three surfaces (a request that
one server answers and the other refuses counts too). Everything else is informational: headers, error wording and
status, event framing, field order, optional fields; ids and times are normalised first. Informational differences go
to `OUT/informational.txt`, never to the verdict. Separately every Rust reply is checked on its own terms: it came at
all, it is no server error, and its body or stream has the shape its API documents (required fields, the terminal
event, `[DONE]`, `text/event-stream`).

The verdict per request is `same`, `differs` (a must-match difference, first paths and both values), `invalid` (the Rust
reply is wrong on its own terms), `refused` (the request is marked as one the Rust server refuses by design for now, and
it answered `unsupported_parameter`) or `missing`. The result is written to `OUT/differences.txt` and
`OUT/differences.json`.

Exit code: 0 when no request differs, is invalid or is missing and both servers ended with 143 (SIGTERM) or 0, 1
otherwise, 2 when the arguments are wrong or a server could not run.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_diff_requests import requests_for  # noqa: E402

SIDES = ("java", "rust")
LAUNCHER = "system-run"
MAX_EST_SECONDS = 600
CONTEXT_TOKENS = 4096
BODY_LIMIT = 64 * 1024
REQUEST_TIMEOUT = 120.0

# Headers that carry no API content: the date, framing and connection management, lengths that follow from the body.
IGNORED_HEADERS = {"date", "content-length", "transfer-encoding", "connection", "keep-alive"}
ID_PREFIX = re.compile(r"^(chatcmpl|cmpl|msg|resp|toolu|call|rs|fc|fco|item|req|run|msgr)[-_][A-Za-z0-9_-]{6,}$")
TIME_KEY = re.compile(r"(^|_)(created|created_at|completed_at|started_at|expires_at|timestamp)$|(_ms|_seconds|_time|duration|latency|elapsed)$")
ID_KEY = re.compile(r"(^|_)(id)$|^signature$|^request-id$|request-id$")


# ---------------------------------------------------------------------------------------------------- normalising


def id_marker(value: str) -> str:
    """An id as its prefix, its length and its alphabet: two servers agree when they build ids the same way."""
    match = re.match(r"^([A-Za-z]+[-_])(.+)$", value)
    prefix, rest = (match.group(1), match.group(2)) if match else ("", value)
    if re.fullmatch(r"[0-9a-f]+", rest):
        alphabet = "hex"
    elif re.fullmatch(r"[0-9a-f-]+", rest):
        alphabet = "uuid"
    elif re.fullmatch(r"[A-Za-z0-9]+", rest):
        alphabet = "alnum"
    else:
        alphabet = "other"
    return f"<id {prefix}{len(rest)} {alphabet}>"


def normalise(value, key: str = ""):
    """The value with ids, times and timings replaced by markers. `key` is the name of the member that holds it."""
    if isinstance(value, dict):
        return {k: normalise(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [normalise(v, key) for v in value]
    if isinstance(value, str):
        if (ID_KEY.search(key) and value) or ID_PREFIX.match(value):
            return id_marker(value)
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and TIME_KEY.search(key):
        return "<time>"
    return value


# ---------------------------------------------------------------------------------------------------- SSE


def parse_sse(body: bytes) -> list[dict]:
    """The events of a server-sent event stream: {event, data, id, retry, comments}, in order. `data` is the JSON value
    when it parses, else the text (`[DONE]`)."""
    text = body.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    events = []
    for block in text.split("\n\n"):
        if not block.strip("\n"):
            continue
        event: dict = {}
        data_lines: list[str] = []
        comments: list[str] = []
        for line in block.split("\n"):
            if line.startswith(":"):
                comments.append(line[1:].lstrip(" "))
                continue
            name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if name == "data":
                data_lines.append(value)
            elif name in ("event", "id", "retry"):
                event[name] = value
        if data_lines:
            data = "\n".join(data_lines)
            try:
                event["data"] = json.loads(data)
            except ValueError:
                event["data"] = data
        if comments:
            event["comments"] = comments
        events.append(event)
    return events


# ---------------------------------------------------------------------------------------------------- comparing


def first_diffs(a, b, path: str = "$", limit: int = 5) -> list[tuple[str, object, object]]:
    """Up to `limit` differing paths of two JSON values, as (path, a, b)."""
    found: list[tuple[str, object, object]] = []

    def walk(x, y, where):
        if len(found) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in x:
                if k not in y:
                    found.append((f"{where}.{k}", x[k], "<absent>"))
                else:
                    walk(x[k], y[k], f"{where}.{k}")
            for k in y:
                if k not in x:
                    found.append((f"{where}.{k}", "<absent>", y[k]))
        elif isinstance(x, list) and isinstance(y, list):
            for i in range(min(len(x), len(y))):
                walk(x[i], y[i], f"{where}[{i}]")
            if len(x) != len(y):
                found.append((f"{where}[len]", len(x), len(y)))
        elif type(x) is not type(y) or x != y:
            found.append((where, x, y))

    walk(a, b, path)
    return found[:limit]


def header_map(headers: list) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for name, value in headers:
        lower = name.lower()
        if lower in IGNORED_HEADERS:
            continue
        out.setdefault(lower, []).append(value)
    return {k: ", ".join(v) for k, v in out.items()}


def normalise_header(name: str, value: str) -> str:
    if ID_KEY.search(name.lower()):
        return id_marker(value)
    return re.sub(r"\s+", " ", value.strip())


def body_view(reply: dict):
    """The comparable form of a reply's body: a JSON value, the event list of a stream, or text."""
    body = base64.b64decode(reply.get("body_b64", ""))
    kind = (dict(header_map(reply["headers"])).get("content-type") or "").split(";")[0].strip().lower()
    if kind == "text/event-stream":
        return "events", [normalise(e) for e in parse_sse(body)]
    if kind.endswith("json") or body[:1] in (b"{", b"["):
        try:
            return "json", normalise(json.loads(body))
        except ValueError:
            pass
    return "text", body.decode("utf-8", errors="replace")


def refused_by_design(reply: dict) -> bool:
    status = reply.get("status") or 0
    return 400 <= status < 500 and (
        b"unsupported_parameter" in base64.b64decode(reply.get("body_b64", ""))
        or b"Unsupported parameter" in base64.b64decode(reply.get("body_b64", ""))
    )


def compare_reply(java: dict, rust: dict) -> list[str]:
    """The differences of two saved replies, one line each (first ones first)."""
    lines: list[str] = []
    for side, reply in (("java", java), ("rust", rust)):
        if reply.get("error"):
            lines.append(f"{side}: no reply: {reply['error']}")
    if lines:
        return lines
    if java["status"] != rust["status"]:
        lines.append(f"status: java {java['status']} rust {rust['status']}")
    jh, rh = header_map(java["headers"]), header_map(rust["headers"])
    for name in sorted(set(jh) | set(rh)):
        a = normalise_header(name, jh[name]) if name in jh else "<absent>"
        b = normalise_header(name, rh[name]) if name in rh else "<absent>"
        if a != b:
            lines.append(f"header {name}: java {a!r} rust {b!r}")
    (ja, jv), (ra, rv) = body_view(java), body_view(rust)
    if ja != ra:
        lines.append(f"body kind: java {ja} rust {ra}")
    elif ja == "text":
        if jv != rv:
            lines.append(f"body text: java {jv[:200]!r} rust {rv[:200]!r}")
    else:
        for path, a, b in first_diffs(jv, rv, "$" if ja == "json" else "events"):
            lines.append(f"{'body' if ja == 'json' else 'stream'} {path}: java {short(a)} rust {short(b)}")
    return lines


def short(value, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_args_json(text):
    try:
        return json.loads(text) if isinstance(text, str) and text else text
    except ValueError:
        return text


def sse_data(events: list[dict]) -> list[dict]:
    return [e["data"] for e in events if isinstance(e.get("data"), dict)]


def reply_events(reply: dict) -> list[dict]:
    return parse_sse(base64.b64decode(reply.get("body_b64", "")))


def reply_json(reply: dict):
    try:
        return json.loads(base64.b64decode(reply.get("body_b64", "")))
    except ValueError:
        return None


def usage_of(usage: dict | None, keys: tuple[str, ...]) -> dict:
    return {k: (usage or {}).get(k) for k in keys}


def chat_view(body: dict) -> dict:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = [(c.get("function", {}).get("name"), parse_args_json(c.get("function", {}).get("arguments")))
             for c in message.get("tool_calls") or []]
    return {"text": message.get("content") or "", "reasoning": message.get("reasoning_content") or "", "tool_calls": calls,
            "finish": choice.get("finish_reason"), **usage_of(body.get("usage"), ("prompt_tokens", "completion_tokens"))}


def chat_stream_view(events: list[dict]) -> dict:
    text, reasoning, finish, usage, calls = "", "", None, None, {}
    for chunk in sse_data(events):
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                entry = calls.setdefault(call.get("index", 0), ["", ""])
                entry[0] = (call.get("function") or {}).get("name") or entry[0]
                entry[1] += (call.get("function") or {}).get("arguments") or ""
            finish = choice.get("finish_reason") or finish
        usage = chunk.get("usage") or usage
    return {"text": text, "reasoning": reasoning, "tool_calls": [(n, parse_args_json(a)) for n, a in calls.values()],
            "finish": finish, **usage_of(usage, ("prompt_tokens", "completion_tokens"))}


def messages_view(body: dict) -> dict:
    text = thinking = ""
    calls = []
    for block in body.get("content") or []:
        if block.get("type") == "text":
            text += block.get("text", "")
        elif block.get("type") == "thinking":
            thinking += block.get("thinking", "")
        elif block.get("type") == "tool_use":
            calls.append((block.get("name"), block.get("input")))
    return {"text": text, "reasoning": thinking, "tool_calls": calls, "finish": body.get("stop_reason"),
            **usage_of(body.get("usage"), ("input_tokens", "output_tokens"))}


def messages_stream_view(events: list[dict]) -> dict:
    text = thinking = ""
    blocks: dict = {}
    finish, input_tokens, output_tokens = None, None, None
    for e in sse_data(events):
        kind = e.get("type")
        if kind == "message_start":
            input_tokens = (e.get("message", {}).get("usage") or {}).get("input_tokens")
        elif kind == "content_block_start":
            block = e.get("content_block") or {}
            if block.get("type") == "tool_use":
                blocks[e.get("index")] = [block.get("name"), ""]
        elif kind == "content_block_delta":
            delta = e.get("delta") or {}
            text += delta.get("text", "") if delta.get("type") == "text_delta" else ""
            thinking += delta.get("thinking", "") if delta.get("type") == "thinking_delta" else ""
            if delta.get("type") == "input_json_delta" and e.get("index") in blocks:
                blocks[e["index"]][1] += delta.get("partial_json", "")
        elif kind == "message_delta":
            finish = (e.get("delta") or {}).get("stop_reason") or finish
            output_tokens = (e.get("usage") or {}).get("output_tokens", output_tokens)
            input_tokens = (e.get("usage") or {}).get("input_tokens", input_tokens)
    return {"text": text, "reasoning": thinking, "tool_calls": [(n, parse_args_json(a or "{}")) for n, a in blocks.values()],
            "finish": finish, "input_tokens": input_tokens, "output_tokens": output_tokens}


def responses_view(body: dict) -> dict:
    text = reasoning = ""
    calls = []
    for item in body.get("output") or []:
        if item.get("type") == "message":
            text += "".join(c.get("text", "") for c in item.get("content") or [])
        elif item.get("type") == "reasoning":
            reasoning += "".join(c.get("text", "") for c in (item.get("summary") or []) + (item.get("content") or []))
        elif item.get("type") == "function_call":
            calls.append((item.get("name"), parse_args_json(item.get("arguments"))))
    finish = body.get("status")
    if body.get("incomplete_details"):
        finish = f"{finish}:{body['incomplete_details'].get('reason')}"
    return {"text": text, "reasoning": reasoning, "tool_calls": calls, "finish": finish,
            **usage_of(body.get("usage"), ("input_tokens", "output_tokens"))}


def responses_stream_view(events: list[dict]) -> dict | None:
    final = [e["response"] for e in sse_data(events) if isinstance(e.get("response"), dict)
             and e.get("type") in ("response.completed", "response.incomplete", "response.failed")]
    return responses_view(final[-1]) if final else None


def semantic_view(request: dict, reply: dict):
    """What the model saw and said, from a successful reply: None when there is nothing to compare (an error reply, a
    request that is not a generation, a stream cut by the client)."""
    path, status = request["path"], reply.get("status") or 0
    if request["method"] != "POST" or not 200 <= status < 300 or request.get("disconnect_after"):
        return None
    stream = request.get("stream")
    if path == "/v1/chat/completions":
        return chat_stream_view(reply_events(reply)) if stream else chat_view(reply_json(reply) or {})
    if path == "/v1/messages":
        return messages_stream_view(reply_events(reply)) if stream else messages_view(reply_json(reply) or {})
    if path == "/v1/responses":
        return responses_stream_view(reply_events(reply)) if stream else responses_view(reply_json(reply) or {})
    if path == "/v1/messages/count_tokens":
        return {"input_tokens": (reply_json(reply) or {}).get("input_tokens")}
    return None


def error_shaped(body) -> bool:
    return isinstance(body, dict) and (isinstance(body.get("error"), (dict, str)))


def validity_problems(request: dict, reply: dict) -> list[str]:
    """Why a reply is wrong on its own terms (crash, hang, a shape its API does not document); empty when it is fine."""
    if reply.get("error"):
        return [f"no reply: {reply['error']}"]
    status, path, problems = reply["status"], request["path"], []
    if status >= 500:
        problems.append(f"server error status {status}")
    generation = request["method"] == "POST" and path in ("/v1/chat/completions", "/v1/messages", "/v1/responses")
    if not generation:
        return problems
    content_type = (header_map(reply["headers"]).get("content-type") or "").split(";")[0].strip().lower()
    if status >= 400:
        body = reply_json(reply)
        if not error_shaped(body):
            problems.append("an error reply that is not an error object")
        return problems
    if request.get("stream"):
        if content_type != "text/event-stream":
            problems.append(f"stream content-type {content_type!r}, not text/event-stream")
        if request.get("disconnect_after"):
            return problems
        events = reply_events(reply)
        if not events:
            return problems + ["a stream without events"]
        raw = base64.b64decode(reply.get("body_b64", ""))
        if not raw.endswith(b"\n\n") and not raw.endswith(b"\r\n\r\n"):
            problems.append("the stream ends in the middle of an event")
        data = sse_data(events)
        if path == "/v1/chat/completions":
            if events[-1].get("data") != "[DONE]":
                problems.append("the stream does not end with [DONE]")
            if any("choices" not in d for d in data):
                problems.append("a chunk without choices")
            if not any((c.get("finish_reason")) for d in data for c in d.get("choices") or []):
                problems.append("no chunk carries a finish_reason")
        elif path == "/v1/messages":
            types = [e.get("event") or (e.get("data") or {}).get("type") for e in events if isinstance(e.get("data"), dict)]
            if not types or types[0] != "message_start" or types[-1] != "message_stop":
                problems.append(f"events do not run message_start..message_stop: {types[:1]}..{types[-1:]}")
            if any(e.get("event") != (e.get("data") or {}).get("type") for e in events if isinstance(e.get("data"), dict)):
                problems.append("an event name differs from its payload type")
        else:
            types = [d.get("type") for d in data]
            if not types or types[0] != "response.created":
                problems.append("the first event is not response.created")
            if not set(types) & {"response.completed", "response.incomplete", "response.failed"}:
                problems.append("no terminal response event")
        return problems
    body = reply_json(reply)
    if not isinstance(body, dict):
        return problems + ["the body is not a JSON object"]
    required = {"/v1/chat/completions": ("id", "object", "model", "choices", "usage"),
                "/v1/messages": ("id", "type", "role", "content", "stop_reason", "usage"),
                "/v1/responses": ("id", "object", "status", "output", "usage")}[path]
    missing = [k for k in required if k not in body]
    if missing:
        problems.append(f"required fields missing: {missing}")
    if path == "/v1/chat/completions" and body.get("choices") and "finish_reason" not in body["choices"][0]:
        problems.append("a choice without finish_reason")
    return problems


def informational_request(request: dict) -> bool:
    return not (request["method"] == "POST" and "_err_" not in request["name"] and (request["name"] == "http_chunked_body" or not request["name"].startswith("http_"))
                and request["path"] in ("/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/messages/count_tokens"))


def judge(request: dict, java: dict | None, rust: dict | None) -> dict:
    """The verdict of one request: `same`, `differs` (a must-match difference), `invalid` (the Rust reply is wrong on its
    own terms), `refused` or `missing`, with the lines of each class: `must`, `invalid`, `info`."""
    if java is None or rust is None:
        absent = [s for s, r in (("java", java), ("rust", rust)) if r is None]
        return {"verdict": "missing", "must": [f"no saved reply of {', '.join(absent)}"], "invalid": [], "info": []}
    info = compare_reply(java, rust) if not (java.get("error") or rust.get("error")) else []
    invalid = validity_problems(request, rust)
    must: list[str] = []
    if request.get("refusal") and refused_by_design(rust) and not java.get("error"):
        return {"verdict": "refused", "must": [], "invalid": [], "info": [f"rust refuses: status {rust['status']}"]}
    if not informational_request(request) and not java.get("error"):
        jv, rv = semantic_view(request, java), semantic_view(request, rust)
        if jv is not None and rv is None and not rust.get("error"):
            must.append(f"java answered {java['status']}, rust answered {rust['status']}")
        elif jv is not None and rv is not None:
            must = [f"{path}: java {short(a)} rust {short(b)}" for path, a, b in first_diffs(jv, rv, "view", limit=8)]
    verdict = "differs" if must else "invalid" if invalid else "same"
    return {"verdict": verdict, "must": must, "invalid": invalid, "info": info}


# ---------------------------------------------------------------------------------------------------- the client


def send(port: int, request: dict, timeout: float) -> dict:
    """One request; the raw reply as a dict (status, headers, body_b64, and for a stream the event list)."""
    headers = dict(request.get("headers") or {})
    body = None
    if "json" in request:
        body = json.dumps(request["json"]).encode()
        headers.setdefault("content-type", "application/json")
    elif "raw" in request:
        body = request["raw"]
    reply: dict = {"request": request["name"]}
    started = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        if request.get("chunked"):
            pieces = [body[i : i + 1024] for i in range(0, len(body or b""), 1024)] or [b""]
            conn.request(request["method"], request["path"], body=iter(pieces), headers=headers, encode_chunked=True)
        else:
            conn.request(request["method"], request["path"], body=body, headers=headers)
        response = conn.getresponse()
        reply["status"] = response.status
        reply["reason"] = response.reason
        reply["headers"] = [[k, v] for k, v in response.getheaders()]
        limit = request.get("disconnect_after")
        if limit:
            seen, data = 0, b""
            while seen < limit:
                line = response.readline()
                if not line:
                    break
                data += line
                if line in (b"\n", b"\r\n"):
                    seen += 1
            reply["body_b64"] = base64.b64encode(data).decode()
            reply["disconnected_after_events"] = limit
        else:
            reply["body_b64"] = base64.b64encode(response.read()).decode()
    except (OSError, http.client.HTTPException) as error:
        reply["error"] = f"{type(error).__name__}: {error}"
    finally:
        conn.close()
    reply["seconds"] = round(time.monotonic() - started, 3)
    raw = base64.b64decode(reply.get("body_b64", ""))
    reply["body_text"] = raw.decode("utf-8", errors="replace")
    if any(k.lower() == "content-type" and v.lower().startswith("text/event-stream") for k, v in reply.get("headers", [])):
        reply["events"] = parse_sse(raw)
    return reply


# ---------------------------------------------------------------------------------------------------- the servers


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def model_name(config: Path) -> str:
    """`model.name` of a settings file (a two-level read; the harness needs no YAML library)."""
    inside = False
    for line in config.read_text().splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            inside = line.split("#")[0].strip() == "model:"
        elif inside:
            match = re.match(r"^\s+name:\s*(\S+)", line)
            if match:
                return match.group(1).strip("'\"")
    sys.exit(f"{config} has no model.name")


def server_command(side: str, args, config: Path) -> list[str]:
    if side == "java":
        return ["java", "--enable-native-access=ALL-UNNAMED", "-jar", str(args.java_jar), "--config", str(config)]
    return [str(args.rust_binary), "--config", str(config)]


def server_env(side: str, args, port: int, native_cache: Path) -> list[str]:
    env = [
        f"EUHEDRAL_SERVER_PORT={port}",
        f"PORT={port}",
        f"EUHEDRAL_CONTEXT_MAX_TOKENS={CONTEXT_TOKENS}",
        f"EUHEDRAL_SERVER_REQUESTS_MAX_BODY_SIZE={BODY_LIMIT}",
    ]
    if side == "rust":
        env.append(f"EUHEDRAL_HOST_NATIVE_CACHE={native_cache}")
    return env


def health(port: int) -> bool:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/health")
        return conn.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False  # the first connection after the port opens may be reset: ask again
    finally:
        conn.close()


def run_side(side: str, args, requests_dir: Path) -> int:
    """Start the server of `side` under the launcher, send the requests, stop it. Returns 0, or 2 when it did not run."""
    config = (args.java_config if side == "java" else args.rust_config).resolve()
    out = args.out_dir / side
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    port = free_port()
    native_cache = args.out_dir / "native-cache"
    if side == "rust":
        native_cache.mkdir(parents=True, exist_ok=True)
        # the native product directory (lib, share, include) is the native cache, as the jar unpacks it
        for source in args.cuda_library.resolve().parent.parent.iterdir():
            link = native_cache / source.name
            if link.is_symlink() or link.exists():
                link.unlink() if link.is_symlink() else shutil.rmtree(link)
            link.symlink_to(source)
    model = args.model or model_name(config)
    requests = requests_for(model, BODY_LIMIT, CONTEXT_TOKENS)
    pidfile = out / "server.pid"
    command = [
        args.launcher, "--gpu", f"shared:{args.gpu_gb}", "--mem", args.memory_max, "--est", str(args.est), "--who", args.who, "--",
        "sh", "-c", 'echo $$ > "$0"; exec env "$@"', str(pidfile),
        *server_env(side, args, port, native_cache), *server_command(side, args, config),
    ]
    log = out / "server.log"
    print(f"{side}: starting on port {port} (log {log})", flush=True)
    with log.open("wb") as handle:
        launcher = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    pid = None
    code = None
    try:
        deadline = time.monotonic() + args.queue_timeout
        while pid is None:  # the grant may queue behind other jobs: the pid file appears when the job starts
            if launcher.poll() is not None or time.monotonic() > deadline:
                break
            if pidfile.exists() and pidfile.read_text().strip():
                pid = int(pidfile.read_text().strip())
            else:
                time.sleep(0.5)
        if pid is None:
            print(f"{side}: the server did not start (launcher exit {launcher.poll()})", file=sys.stderr)
            return 2
        began = time.monotonic()
        while not health(port):
            if launcher.poll() is not None:
                print(f"{side}: the server exited before it was healthy (exit {launcher.poll()}); see {log}", file=sys.stderr)
                return 2
            if time.monotonic() - began > args.start_timeout:
                print(f"{side}: not healthy after {args.start_timeout:.0f} s; see {log}", file=sys.stderr)
                return 2
            time.sleep(1.0)
        print(f"{side}: healthy after {time.monotonic() - began:.0f} s; sending {len(requests)} requests", flush=True)
        for request in requests:
            reply = send(port, request, args.request_timeout)
            (out / f"{request['name']}.json").write_text(json.dumps(reply, indent=1) + "\n")
            outcome = reply.get("error") or reply["status"]
            print(f"  {request['name']}: {outcome} ({reply['seconds']} s)", flush=True)
            if launcher.poll() is not None:
                print(f"{side}: the server exited while the requests ran", file=sys.stderr)
                return 2
    finally:
        if pid is not None and launcher.poll() is None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            code = launcher.wait(timeout=60)
        except subprocess.TimeoutExpired:
            launcher.kill()  # the launcher alone: its job scope ends with it
            code = launcher.wait()
    (out / "exit.json").write_text(json.dumps({"exit_code": code, "pid": pid, "port": port}) + "\n")
    print(f"{side}: server exit code {code}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- the report


def load_replies(side_dir: Path) -> dict[str, dict]:
    replies = {}
    if side_dir.is_dir():
        for path in side_dir.glob("*.json"):
            if path.name != "exit.json":
                replies[path.stem] = json.loads(path.read_text())
    return replies


def compare_saved(args) -> int:
    model = args.model or (model_name(args.java_config) if args.java_config else "model")
    requests = requests_for(model, BODY_LIMIT, CONTEXT_TOKENS)
    java, rust = (load_replies(args.out_dir / s) for s in SIDES)
    exits = {}
    for side in SIDES:
        path = args.out_dir / side / "exit.json"
        exits[side] = json.loads(path.read_text())["exit_code"] if path.is_file() else None
    rows, counts = [], {"same": 0, "differs": 0, "invalid": 0, "refused": 0, "missing": 0}
    for request in requests:
        result = judge(request, java.get(request["name"]), rust.get(request["name"]))
        counts[result["verdict"]] += 1
        rows.append({"request": request["name"], **result})
    text = [f"{'request':<44} verdict"]
    info = ["differences that do not need to match (headers, wording, framing, optional fields), per request"]
    for row in rows:
        text.append(f"{row['request']:<44} {row['verdict']}")
        text.extend(f"    must match: {line}" for line in row["must"])
        text.extend(f"    rust reply wrong on its own terms: {line}" for line in row["invalid"])
        if row["info"]:
            info.append(f"{row['request']}:")
            info.extend(f"    {line}" for line in row["info"])
    bad_exit = [s for s in SIDES if exits[s] not in (0, 143)]
    text.append("")
    text.append(f"{len(rows)} requests: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
    text.append(f"server exit codes: java {exits['java']}, rust {exits['rust']} (SIGTERM ends a server with 143)")
    report = "\n".join(text) + "\n"
    print(report, end="")
    (args.out_dir / "differences.txt").write_text(report)
    (args.out_dir / "informational.txt").write_text("\n".join(info) + "\n")
    (args.out_dir / "differences.json").write_text(json.dumps({"rows": rows, "exit_codes": exits}, indent=1) + "\n")
    return 1 if counts["differs"] or counts["invalid"] or counts["missing"] or bad_exit else 0


# ---------------------------------------------------------------------------------------------------- self-test


def self_test() -> int:
    failures: list[str] = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # ids, times, timings
    a = {"id": "chatcmpl-abcdef123456", "created": 1700000000, "model": "m", "usage": {"total_ms": 12.5, "n": 3}}
    b = {"id": "chatcmpl-ffffff654321", "created": 1700000999, "model": "m", "usage": {"total_ms": 99.1, "n": 3}}
    check("normalise ids and times", normalise(a), normalise(b))
    check("id marker", normalise(a)["id"], "<id chatcmpl-12 hex>")
    check("a different id shape differs", normalise({"id": "chatcmpl-abc"}) != normalise({"id": "chatcmpl-abcdef123456"}), True)
    check("tool id by value", normalise({"x": ["toolu_01ABCdef23456789"]}), {"x": ["<id toolu_16 alnum>"]})
    check("plain text is kept", normalise({"content": "call me"}), {"content": "call me"})
    check("numbers elsewhere are kept", normalise({"completion_tokens": 7}), {"completion_tokens": 7})
    check("a bool under a time key", normalise({"created": True}), {"created": True})
    # SSE parser
    stream = b'event: message_start\ndata: {"a": 1}\n\n: keep-alive\n\ndata: {"b": [1,2]}\n\ndata: [DONE]\n\n'
    check("sse events", parse_sse(stream), [
        {"event": "message_start", "data": {"a": 1}},
        {"comments": ["keep-alive"]},
        {"data": {"b": [1, 2]}},
        {"data": "[DONE]"},
    ])
    check("sse crlf", parse_sse(b"data: x\r\n\r\ndata: y\r\n\r\n"), [{"data": "x"}, {"data": "y"}])
    check("sse multi-line data", parse_sse(b"data: a\ndata: b\n\n"), [{"data": "a\nb"}])
    check("sse unfinished block", parse_sse(b'data: {"a":1}\n\ndata: {"b"'), [{"data": {"a": 1}}, {"data": '{"b"'}])
    # diffs
    check("no diff", first_diffs({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}), [])
    check("first diff path", first_diffs({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}), [("$.a[1].b", 2, 3)])
    check("absent member", first_diffs({"a": 1}, {"b": 1}), [("$.a", 1, "<absent>"), ("$.b", "<absent>", 1)])
    check("length", first_diffs([1, 2], [1, 2, 3]), [("$[len]", 2, 3)])
    check("type", first_diffs({"a": 1}, {"a": 1.0}), [("$.a", 1, 1.0)])
    # replies
    def reply(status, headers, body):
        return {"status": status, "headers": headers, "body_b64": base64.b64encode(body).decode()}
    json_type = ["Content-Type", "application/json"]
    j = reply(200, [json_type, ["Date", "x"], ["Content-Length", "5"]], b'{"id":"resp_aaaaaaaaaa","created_at":1}')
    r = reply(200, [json_type, ["Date", "y"], ["Content-Length", "6"]], b'{"id":"resp_bbbbbbbbbb","created_at":2}')
    check("same replies", compare_reply(j, r), [])
    r2 = reply(404, [["Content-Type", "application/json"], ["X-Extra", "1"]], b'{"id":"resp_bbbbbbbbbb","created_at":2}')
    check("status and header", compare_reply(j, r2), ["status: java 200 rust 404", "header x-extra: java '<absent>' rust '1'"])
    sse = ["Content-Type", "text/event-stream;charset=UTF-8"]
    s1 = reply(200, [sse], b'data: {"id":"chatcmpl-aaaaaaaa","n":1}\n\ndata: [DONE]\n\n')
    s2 = reply(200, [["content-type", "text/event-stream"]], b'data: {"id":"chatcmpl-bbbbbbbb","n":1}\n\ndata: [DONE]\n\n')
    check("stream content type differs", compare_reply(s1, s2), ["header content-type: java 'text/event-stream;charset=UTF-8' rust 'text/event-stream'"])
    s3 = reply(200, [sse], b'data: {"id":"chatcmpl-bbbbbbbb","n":2}\n\ndata: [DONE]\n\n')
    check("stream payload", compare_reply(s1, s3), ["stream events[0].data.n: java 1 rust 2"])
    s4 = reply(200, [sse], b'data: {"id":"chatcmpl-bbbbbbbb","n":1}\n\n')
    check("stream length", compare_reply(s1, s4), ["stream events[len]: java 2 rust 1"])
    check("error reply", compare_reply({"error": "timeout"}, j), ["java: no reply: timeout"])
    # verdicts
    refusal = reply(400, [json_type], b'{"error":{"code":"unsupported_parameter","param":"tool_choice"}}')
    ok = reply(200, [json_type], b'{"a":1}')
    chat = {"method": "POST", "path": "/v1/chat/completions", "name": "chat_x"}
    def chat_reply(text, prompt=5, finish="stop", status=200):
        body = {"id": "chatcmpl-aaaaaaaa", "object": "chat.completion", "model": "m", "usage": {"prompt_tokens": prompt, "completion_tokens": 2},
                "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}]}
        return reply(status, [json_type], json.dumps(body).encode())
    check("same text", judge(chat, chat_reply("hi"), chat_reply("hi"))["verdict"], "same")
    check("other text", judge(chat, chat_reply("hi"), chat_reply("ho"))["verdict"], "differs")
    check("other prompt tokens", judge(chat, chat_reply("hi", 5), chat_reply("hi", 6))["verdict"], "differs")
    check("other finish", judge(chat, chat_reply("hi"), chat_reply("hi", finish="length"))["verdict"], "differs")
    check("java answers, rust errors", judge(chat, chat_reply("hi"), reply(400, [json_type], b'{"error":{"message":"x"}}'))["verdict"], "differs")
    check("rust server error", judge(chat, chat_reply("hi"), reply(500, [json_type], b'{"error":{}}'))["verdict"], "differs")
    refusal = reply(400, [json_type], b'{"error":{"code":"unsupported_parameter","param":"tool_choice"}}')
    check("refused by design", judge({**chat, "refusal": True}, chat_reply("hi"), refusal)["verdict"], "refused")
    check("not marked", judge(chat, chat_reply("hi"), refusal)["verdict"], "differs")
    err = {"method": "POST", "path": "/v1/chat/completions", "name": "chat_err_x"}
    e400 = reply(400, [json_type], b'{"error":{"message":"a"}}')
    e422 = reply(422, [json_type], b'{"error":{"message":"b"}}')
    check("error wording is informational", judge(err, e400, e422)["verdict"], "same")
    check("error wording is listed", len(judge(err, e400, e422)["info"]) > 0, True)
    check("error that is no error object", judge(err, e400, reply(400, [json_type], b"[]"))["verdict"], "invalid")
    check("missing", judge({}, chat_reply("a"), None)["verdict"], "missing")
    check("rust did not answer", validity_problems(chat, {"error": "timeout"}), ["no reply: timeout"])
    # views
    chunks = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: {"choices":[{"delta":{"content":"a"}}]}\n\n'
              b'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}\n\n'
              b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\ndata: [DONE]\n\n')
    check("chat stream view", chat_stream_view(parse_sse(chunks)),
          {"text": "ab", "reasoning": "", "tool_calls": [], "finish": "stop", "prompt_tokens": 4, "completion_tokens": 2})
    stream_request = {**chat, "stream": True}
    check("chat stream is valid", validity_problems(stream_request, reply(200, [sse], chunks)),
          [])
    check("chat stream without DONE", validity_problems(stream_request, reply(200, [sse], chunks[:-len(b"data: [DONE]\n\n")])),
          ["the stream does not end with [DONE]"])
    anth = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\n\n'
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"f"}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":"}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}\n\n'
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    check("messages stream view", messages_stream_view(parse_sse(anth)),
          {"text": "", "reasoning": "", "tool_calls": [("f", {"a": 1})], "finish": "tool_use", "input_tokens": 7, "output_tokens": 9})
    check("messages stream is valid", validity_problems({"method": "POST", "path": "/v1/messages", "name": "m", "stream": True},
                                                       reply(200, [sse], anth)), [])
    check("responses view", responses_view({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
          "output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}], "usage": {"input_tokens": 3, "output_tokens": 1}}),
          {"text": "x", "reasoning": "", "tool_calls": [], "finish": "incomplete:max_output_tokens", "input_tokens": 3, "output_tokens": 1})
    # header ids
    check("request id header", normalise_header("X-Request-Id", "abcxyz"), "<id 6 alnum>")
    # a chunked body request is well formed
    check("model name read", model_name_of("model:\n  artifact: /a\n  name: 'm1'  # c\nserver:\n  name: other\n"), "m1")

    for failure in failures:
        print("FAIL", failure)
    print("api_diff self-test:", "failed" if failures else "ok")
    return 1 if failures else 0


def model_name_of(text: str) -> str:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
        handle.write(text)
        handle.flush()
        return model_name(Path(handle.name))


# ---------------------------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--java-config", type=Path)
    parser.add_argument("--rust-config", type=Path)
    parser.add_argument("--java-jar", type=Path, help="the server jar of the `api` module (`bootJar`)")
    parser.add_argument("--rust-binary", type=Path, help="the release binary `euhedral-server`")
    parser.add_argument("--cuda-library", type=Path, help="libeuhedral_cuda.so for the Rust server (the product directory that holds its lib/ is linked as the native cache)")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--only", choices=("java", "rust", "compare"), help="run one side, or only compare the saved replies")
    parser.add_argument("--who", default="api-diff", help="the name given to the arbiter")
    parser.add_argument("--launcher", default=LAUNCHER)
    parser.add_argument("--model", help="the model name clients send (default: model.name of the Java config)")
    parser.add_argument("--gpu-gb", type=int, default=14, help="device memory asked of the launcher, in GB")
    parser.add_argument("--memory-max", default="24G", help="host memory cap of a server (launcher --mem)")
    parser.add_argument("--est", type=int, default=MAX_EST_SECONDS, help="seconds the launcher is told a server runs (at most 600)")
    parser.add_argument("--queue-timeout", type=float, default=3600.0, help="seconds to wait for the grant")
    parser.add_argument("--start-timeout", type=float, default=300.0, help="seconds from the start of the job to a healthy server")
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.out_dir is None:
        parser.error("--out-dir is required")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.est <= MAX_EST_SECONDS:
        parser.error(f"--est must be between 1 and {MAX_EST_SECONDS}")
    sides = [s for s in SIDES if args.only in (None, s)]
    for side in sides:
        config = args.java_config if side == "java" else args.rust_config
        needs = {"java": [("--java-config", config), ("--java-jar", args.java_jar)],
                 "rust": [("--rust-config", config), ("--rust-binary", args.rust_binary), ("--cuda-library", args.cuda_library)]}[side]
        for flag, value in needs:
            if value is None or not Path(value).exists():
                parser.error(f"{flag} must name an existing file for the {side} side")
    if sides and shutil.which(args.launcher) is None:
        parser.error(f"the launcher {args.launcher!r} is not found: give its path with --launcher")
    if args.only != "compare":
        for side in sides:
            if run_side(side, args, args.out_dir) != 0:
                return 2
    if args.only in (None, "compare"):
        if args.model is None and args.java_config is None:
            parser.error("--java-config (or --model) is needed to name the requests")
        return compare_saved(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
