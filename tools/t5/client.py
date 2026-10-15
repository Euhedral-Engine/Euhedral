"""Chat client for the inference server's OpenAI-compatible endpoint (standard library only).

Decision: tokens come from the server (`return_token_ids`), not from re-tokenization, for four reasons.
1. Sampled and multilingual text often decodes from non-canonical token sequences that re-encoding does not
   reproduce.
2. The server reshapes the text (reasoning split, tool-call parsing and re-serialization), so the bytes the client
   sees are not the bytes the model emitted.
3. The engine's tokenizer and template port are already checked against the checkpoint (`QwenTokenizerTest`,
   `QwenChatTemplateTest` goldens).
4. The ids the model actually conditioned on and produced are exactly what the scorer must use.

Request fields: `model`, `messages`, `max_tokens`, `stream` false, `return_token_ids` true, the decoding fields
(`temperature`; sampled adds `top_p`, `top_k`, `min_p`, `seed`), and with tools `tools`, `tool_choice` "auto" and
`parallel_tool_calls` true. Response fields: `choices[0].message` (`content`, `reasoning_content`, `tool_calls`),
`choices[0].finish_reason`, top-level `prompt_token_ids` and `choices[0].token_ids` (every generated token,
thinking and tool-call envelope and terminator included).
"""
from __future__ import annotations

import dataclasses
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from t5.records import Decoding
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from t5.records import Decoding

RETRYABLE_STATUS = frozenset({502, 503, 504})
BACKOFF_CAP = 300.0


@dataclasses.dataclass(frozen=True)
class ChatResult:
    message: dict  # assistant message for the history: role, content, reasoning_content, tool_calls
    finish_reason: str
    prompt_token_ids: list[int]
    token_ids: list[int]


def request_body(model: str, messages: list[dict], tools: list[dict], decoding: Decoding, max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False,
            "return_token_ids": True, **decoding.request_fields()}
    if tools:
        body.update({"tools": tools, "tool_choice": "auto", "parallel_tool_calls": True})
    return body


def parse_response(payload: dict) -> ChatResult:
    choice = payload["choices"][0]
    wire = choice["message"]
    message = {"role": "assistant", "content": wire.get("content")}
    if wire.get("reasoning_content") is not None:
        message["reasoning_content"] = wire["reasoning_content"]
    if wire.get("tool_calls"):
        message["tool_calls"] = wire["tool_calls"]
    prompt_ids, token_ids = payload.get("prompt_token_ids"), choice.get("token_ids")
    if not isinstance(prompt_ids, list) or not isinstance(token_ids, list):
        raise ValueError("response carries no token ids; the server needs return_token_ids support")
    return ChatResult(message, choice.get("finish_reason") or "", prompt_ids, token_ids)


class ChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 6 * 3600, retries: int = 8,
                 sleep=time.sleep) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self._sleep = sleep

    def complete(self, messages: list[dict], tools: list[dict], decoding: Decoding, max_tokens: int) -> ChatResult:
        data = json.dumps(request_body(self.model, messages, tools, decoding, max_tokens),
                          ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return parse_response(json.loads(response.read().decode("utf-8")))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code not in RETRYABLE_STATUS or attempt == self.retries:
                    raise RuntimeError(f"server answered {e.code}: {detail}") from e
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"server unreachable: {e}") from e
            self._sleep(min(2.0 ** attempt, BACKOFF_CAP))
        raise AssertionError("unreachable")
