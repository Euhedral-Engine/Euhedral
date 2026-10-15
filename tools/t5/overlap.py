"""N-gram overlap between the evaluation set's prompts and documents and the data it must stay clear of.

Usage (from tools/): python3 -m t5.overlap STORE --against FILE... [--n 13] [--limit 0.05]

Every conversation turn (and system prompt) and every document is compared with the union of the reference files;
STORE/checks/overlap.json records the fraction of each item's n-word windows found among the references. Reference
files ending in .jsonl contribute their string values, other files are read as text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    from t5 import records
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from t5 import records

# Ideographs and kana are words of their own: those scripts do not separate words with spaces.
_WORD = re.compile(r"[぀-ヿ㐀-鿿]|[^\W぀-ヿ㐀-鿿]+")


def shingles(text: str, n: int = 13) -> set[int]:
    words = _WORD.findall(text.lower())
    return {int.from_bytes(hashlib.sha1(" ".join(words[i:i + n]).encode("utf-8")).digest()[:8], "big")
            for i in range(len(words) - n + 1)}


def overlap(candidate: str, reference: set[int], n: int = 13) -> float:
    found = shingles(candidate, n)
    return len(found & reference) / len(found) if found else 0.0


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def reference_texts(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield from _strings(json.loads(line))
    else:
        yield path.read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t5.overlap")
    parser.add_argument("store", type=Path)
    parser.add_argument("--against", type=Path, nargs="+", required=True)
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--limit", type=float, default=0.05)
    args = parser.parse_args(argv)
    reference: set[int] = set()
    hashes = []
    for path in args.against:
        reference |= {h for text in reference_texts(path) for h in shingles(text, args.n)}
        hashes.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result: dict = {}
    conversations = args.store / "conversations.jsonl"
    if conversations.exists():
        for c in records.load_conversations(conversations):
            texts = list(c.turns) + ([c.system] if c.system else [])
            result[c.id] = max((overlap(t, reference, args.n) for t in texts), default=0.0)
    documents = args.store / "documents.jsonl"
    if documents.exists():
        for d in records.load_documents(documents):
            text = (args.store / d.path).read_text(encoding="utf-8", errors="replace")
            result[d.id] = overlap(text, reference, args.n)
    over = sorted(k for k, v in result.items() if v > args.limit)
    out = args.store / "checks"
    out.mkdir(exist_ok=True)
    (out / "overlap.json").write_text(
        json.dumps({**result, "references": hashes, "n": args.n}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for item in over:
        print(f"{item}: overlap {result[item]:.3f} exceeds {args.limit}")
    print(f"{len(result)} items, {len(over)} over the limit")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
