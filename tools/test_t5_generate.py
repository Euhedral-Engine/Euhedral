from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t5 import client as chat_client, fixtures, generate, progress, records  # noqa: E402

TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
FIXTURE = {"tools": {"lookup": {"rules": [{"match": {"q": 1}, "result": {"answer": 42}}]}}}


def render(messages):
    """Prompt ids: assistant turns re-render as the ids they were generated with, so prompts extend each other."""
    ids = []
    for i, message in enumerate(messages):
        ids += [1000 + i, 2000 + i] if message["role"] == "assistant" else [i * 3, i * 3 + 1, i * 3 + 2]
    return ids


class FakeServer:
    """Scripts replies from the request: ids derive from the message count; with tools, the first turn calls `lookup`."""

    def __init__(self, fail_first: int = 0) -> None:
        self.bodies: list[dict] = []
        self.fail_first = fail_first
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                outer.bodies.append(body)
                messages = body["messages"]
                message = {"role": "assistant", "content": "answer", "reasoning_content": "thought"}
                if body.get("tools") and messages[-1]["role"] == "user":
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{\"q\": 1}"}}]}
                payload = {"choices": [{"index": 0, "message": message, "finish_reason": "stop",
                                        "token_ids": [1000 + len(messages), 2000 + len(messages)]}],
                           "prompt_token_ids": render(messages)}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def users(self) -> list[str]:
        return [next(m["content"] for m in b["messages"] if m["role"] == "user") for b in self.bodies]


def row(name, domain="conversation", order=0, **extra):
    return {"id": name, "domain": domain, "order": order, "language": "en", "turns": [f"hello {name}"], **extra}


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Path(self.temp.name)
        self.server = FakeServer()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.server.close)

    def write(self, rows):
        (self.store / "conversations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (self.store / "fx.json").write_text(json.dumps(FIXTURE))

    def run_main(self, *extra, seed="7", domain="conversation"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = generate.main([self.server.url, str(self.store), "--model", "m", "--base-seed", seed,
                                  "--parallel", "2", *extra],
                                 validation={"chunks": {domain: 1}, "languages": set()})
        return code, out.getvalue()

    def test_request_bodies(self):
        greedy = chat_client.request_body("m", [], [], records.Decoding("greedy", None), 10)
        self.assertEqual(greedy["temperature"], 0)
        self.assertNotIn("seed", greedy)
        sampled = chat_client.request_body("m", [], [TOOL], records.Decoding("sampled", 99), 10)
        for key, value in {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 99,
                           "tool_choice": "auto", "parallel_tool_calls": True}.items():
            self.assertEqual(sampled[key], value)
        for body in (greedy, sampled):
            self.assertIs(body["return_token_ids"], True)
            self.assertIs(body["stream"], False)
            self.assertNotIn("stop", body)
        self.assertNotIn("tools", greedy)

    def test_tool_round_trip(self):
        self.write([row("a", "tool_use", tools=[TOOL], fixture="fx.json")])
        code, _ = self.run_main(domain="tool_use")
        self.assertEqual(code, 0)
        second = self.server.bodies[1]["messages"]
        expected = fixtures.Fixture(FIXTURE).call("lookup", "{\"q\": 1}")
        self.assertEqual(second[-1], {"role": "tool", "tool_call_id": "call_1", "content": expected})
        self.assertEqual(second[-2]["tool_calls"][0]["id"], "call_1")
        record = json.loads((self.store / "generation" / "progress.json").read_text())["conversations"]["a"]
        self.assertEqual(record["tokens"], render(second) + [1000 + len(second), 2000 + len(second)])
        self.assertEqual(record["length"], len(record["tokens"]))
        self.assertEqual(record["ended"], "turns")
        self.assertEqual(len(record["requests"]), 2)
        self.assertTrue(all(r["prefix_consistent"] for r in record["requests"]))
        self.assertEqual(record["decoding"], {"mode": "greedy", "seed": None})

    def test_resume_skips_finished(self):
        self.write([row(n, order=i) for i, n in enumerate("abc")])
        self.assertEqual(self.run_main("--limit", "2")[0], 0)
        self.assertEqual(sorted(self.server.users()), ["hello a", "hello b"])
        self.server.bodies.clear()
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.server.users(), ["hello c"])

    def test_progress_is_one_file_rewritten_in_place(self):
        self.write([row(n, order=i) for i, n in enumerate("abc")])
        self.run_main()
        self.assertEqual([p.name for p in (self.store / "generation").iterdir()], ["progress.json"])
        data = json.loads((self.store / "generation" / "progress.json").read_text())
        self.assertEqual(sorted(data["conversations"]), ["a", "b", "c"])
        self.assertEqual(data["settings"]["base_seed"], 7)

    def test_settings_mismatch_refuses(self):
        self.write([row("a")])
        self.run_main()
        with self.assertRaises(ValueError):
            self.run_main(seed="8")

    def test_capacity_503_is_retried(self):
        self.server.fail_first = 1
        sleeps = []
        client = chat_client.ChatClient(self.server.url, "m", sleep=sleeps.append)
        result = client.complete([{"role": "user", "content": "x"}], [], records.Decoding("greedy", None), 8)
        self.assertEqual(sleeps, [1.0])
        self.assertEqual(result.token_ids, [1001, 2001])
        self.assertEqual(result.message["reasoning_content"], "thought")

    def test_error_is_recorded_and_retried_on_resume(self):
        self.write([row("a")])
        self.server.fail_first = 10 ** 6
        client = chat_client.ChatClient(self.server.url, "m", retries=0, sleep=lambda s: None)
        record = generate.run_conversation(records.load_conversations(self.store / "conversations.jsonl")[0],
                                           records.Decoding("greedy", None), client, None, generate.Limits())
        self.assertEqual(record["ended"], "error")
        saved = progress.Progress.open(self.store / "g" / "p.json", {"x": 1})
        saved.record("a", record)
        self.assertEqual(saved.done(), set())


if __name__ == "__main__":
    unittest.main()
