"""Runs the scripted conversations against the server and writes STORE/generation/progress.json.

Usage (from tools/): python3 -m t5.generate SERVER_URL STORE --model ID --base-seed N [--parallel N]
    [--domains D ...] [--limit N] [--max-tokens N] [--target-tokens N] [--artifact-manifest PATH]

A conversation's corpus sequence is its last request's `prompt_token_ids + token_ids`: the deployed rendering of
the whole conversation, with reasoning replayed through `reasoning_content`. Each request also records whether its
prompt ids start with the previous request's prompt and completion ids (`prefix_consistent`), so template trimming
is visible. See client.py for why the ids come from the server. `--parallel` should match the server's maximum
number of sequences.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from t5 import records
    from t5.client import ChatClient
    from t5.fixtures import Fixture, canonical, load_fixture
    from t5.progress import Progress
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from t5 import records
    from t5.client import ChatClient
    from t5.fixtures import Fixture, canonical, load_fixture
    from t5.progress import Progress


@dataclasses.dataclass(frozen=True)
class Limits:
    max_tokens: int = 4096  # per request
    target_tokens: int = 3072  # stop a conversation once its rendered length reaches this
    max_tool_rounds: int = 8


def run_conversation(conversation: records.Conversation, decoding: records.Decoding, client: ChatClient,
                     fixture: Fixture | None, limits: Limits) -> dict:
    messages = [{"role": "system", "content": conversation.system}] if conversation.system else []
    requests: list[dict] = []
    state = {"last": None, "error": None}

    def finish(ended: str) -> dict:
        last = state["last"]
        tokens = last.prompt_token_ids + last.token_ids if last else []
        return {"id": conversation.id, "domain": conversation.domain, "language": conversation.language,
                "decoding": {"mode": decoding.mode, "seed": decoding.seed}, "requests": requests,
                "messages": messages, "tokens": tokens, "length": len(tokens), "ended": ended,
                "error": state["error"]}

    previous = None
    try:
        for turn in conversation.turns:
            messages.append({"role": "user", "content": turn})
            for _ in range(limits.max_tool_rounds + 1):
                result = client.complete(messages, list(conversation.tools), decoding, limits.max_tokens)
                consistent = previous is None or result.prompt_token_ids[:len(previous)] == previous
                requests.append({"prompt_tokens": len(result.prompt_token_ids),
                                 "completion_tokens": len(result.token_ids),
                                 "finish_reason": result.finish_reason, "prefix_consistent": consistent})
                previous = result.prompt_token_ids + result.token_ids
                state["last"] = result
                messages.append(result.message)
                if result.finish_reason == "length":
                    return finish("length")
                calls = result.message.get("tool_calls") or []
                if not calls:
                    break
                for call in calls:
                    if fixture:
                        content = fixture.call(call["function"]["name"], call["function"].get("arguments"))
                    else:
                        content = canonical({"error": "no tools in this conversation"})
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            else:
                return finish("tool_rounds")
            if len(previous) >= limits.target_tokens:
                return finish("target")
        return finish("turns")
    except Exception as e:  # recorded, retried on resume; the other conversations continue
        state["error"] = f"{type(e).__name__}: {e}"
        return finish("error")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_settings(args, store: Path, limits: Limits, conversations: list[records.Conversation]) -> dict:
    manifest = manifest_file = None
    if args.artifact_manifest:
        manifest = _sha256(args.artifact_manifest)
        data = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
        manifest_file = data.get("sha256") or data.get("file_sha256") if isinstance(data, dict) else None
    fixture_files = sorted({c.fixture for c in conversations if c.fixture})
    return {"server": args.server_url, "model": args.model, "base_seed": args.base_seed,
            "sampling": records.SAMPLING,
            "limits": {"max_tokens": limits.max_tokens, "target_tokens": limits.target_tokens,
                       "max_tool_rounds": limits.max_tool_rounds},
            "artifact_manifest_sha256": manifest, "artifact_file_sha256": manifest_file,
            "git_commit": _git_commit(),
            "conversations_sha256": _sha256(store / "conversations.jsonl"),
            "fixtures_sha256": {name: _sha256(store / name) for name in fixture_files}}


def main(argv: list[str] | None = None, validation: dict | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t5.generate")
    parser.add_argument("server_url")
    parser.add_argument("store", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=Limits.max_tokens)
    parser.add_argument("--target-tokens", type=int, default=Limits.target_tokens)
    parser.add_argument("--artifact-manifest", type=Path)
    args = parser.parse_args(argv)
    limits = Limits(args.max_tokens, args.target_tokens)
    store = args.store
    conversations = records.load_conversations(store / "conversations.jsonl")
    errors = records.validate_conversations(conversations, store, records.load_excluded(),
                                            base_seed=args.base_seed, **(validation or {}))
    if errors:
        for error in errors:
            print(error)
        return 2
    plan = records.plan_decoding(conversations, args.base_seed)
    progress = Progress.open(store / "generation" / "progress.json", build_settings(args, store, limits, conversations))
    finished = progress.done()
    pending = sorted((c for c in conversations if c.id not in finished
                      and (not args.domains or c.domain in args.domains)),
                     key=lambda c: (c.domain, c.order, c.id))
    if args.limit is not None:
        pending = pending[:args.limit]
    client = ChatClient(args.server_url, args.model)
    totals = {"tokens": 0}
    lock = threading.Lock()
    started = time.monotonic()

    def work(conversation: records.Conversation) -> None:
        fixture = load_fixture(store / conversation.fixture) if conversation.fixture else None
        record = run_conversation(conversation, plan[conversation.id], client, fixture, limits)
        progress.record(conversation.id, record)
        with lock:
            totals["tokens"] += sum(r["completion_tokens"] for r in record["requests"])
            rate = totals["tokens"] / max(time.monotonic() - started, 1e-9)
            print(f"{record['id']} length {record['length']} ended {record['ended']} {rate:.1f} tok/s", flush=True)

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max(1, args.parallel)) as pool:
        for conversation, future in [(c, pool.submit(work, c)) for c in pending]:
            try:
                future.result()
            except Exception as e:  # fixture load or progress failure: report, keep going
                failed += 1
                print(f"{conversation.id} failed: {type(e).__name__}: {e}", flush=True)
    errored = sum(1 for c in pending if progress.conversations.get(c.id, {}).get("ended") == "error")
    print(f"{len(pending)} conversations run, {errored + failed} failed, {len(progress.done())} done")
    return 1 if errored or failed else 0


if __name__ == "__main__":
    sys.exit(main())
