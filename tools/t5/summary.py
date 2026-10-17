"""Selects each generated domain's chunks from the finished conversations and writes STORE/generation/summary.json.

Usage (from tools/): python3 -m t5.summary STORE --server TEXT [--chunk 2560]

Per domain, the conversations in `order` (spares last) that ended without an error and reached `--chunk` tokens are
usable; the first records.CHUNKS[domain] of them are selected. The summary also records how each selected
conversation was decoded, the conversations that failed, and the share of follow-up requests whose prompt did not
continue the previous prompt and completion token for token (the server replaying history differently from what the
model generated).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from . import records


def summarize(store: Path, server: str, chunk: int) -> dict:
    conversations = records.load_conversations(store / "conversations.jsonl")
    done = json.loads((store / "generation" / "progress.json").read_text(encoding="utf-8"))["conversations"]
    summary = {"date": datetime.date.today().isoformat(), "server": server, "recorded": len(done),
               "ended": dict(Counter(v["ended"] for v in done.values())), "domains": {}}
    for domain, need in records.CHUNKS.items():
        ordered = sorted((c for c in conversations if c.domain == domain), key=lambda c: (c.order, c.id))
        missing = [c.id for c in ordered if c.id not in done]
        if missing:
            raise ValueError(f"{domain}: not generated: {', '.join(missing)}")
        usable = [c.id for c in ordered if done[c.id]["ended"] != "error" and done[c.id]["length"] >= chunk]
        if len(usable) < need:
            raise ValueError(f"{domain}: {len(usable)} usable conversations, need {need}")
        requests = [r for c in ordered for r in done[c.id]["requests"]]
        summary["domains"][domain] = {
            "chunks": need,
            "usable": len(usable),
            "selected": usable[:need],
            "decoding_of_selected": dict(Counter(done[i]["decoding"]["mode"] for i in usable[:need])),
            "errors": [c.id for c in ordered if done[c.id]["ended"] == "error"],
            "prefix_inconsistent_share": round(
                sum(not r["prefix_consistent"] for r in requests) / max(1, len(requests)), 3),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", type=Path)
    parser.add_argument("--server", required=True, help="the machine and engine settings the text was generated on")
    parser.add_argument("--chunk", type=int, default=2560)
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.store, args.server, args.chunk)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    directory = args.store / "generation"
    handle, temporary = tempfile.mkstemp(dir=directory)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(json.dumps(summary, indent=1) + "\n")
    os.replace(temporary, directory / "summary.json")
    for domain, s in summary["domains"].items():
        print(f"{domain}: {s['usable']} usable of {s['chunks']} needed, {s['decoding_of_selected']}, errors "
              f"{s['errors']}, prefix-inconsistent {s['prefix_inconsistent_share']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
