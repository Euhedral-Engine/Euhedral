"""N-gram overlap between the evaluation set's prompts and documents and the data it must stay clear of.

Usage (from tools/): python3 -m t5.overlap STORE --against FILE... [--n 13] [--limit 0.05] [--min-shared 5]
                                            [--containment 0.5]

Every conversation turn, system prompt, tool definition and fixture text, and every document, is compared with the
union of the reference files. STORE/checks/overlap.json records, per item (under "items"), three measures over
n-word windows (n = 13):
- fraction: the share of the item's windows found in the references (--limit, flagged above it). A long turn
  dilutes it, so it is not enough alone;
- shared: the number of such windows (--min-shared, flagged at or above it);
- containment: the largest share of one reference item (a JSONL string value, or a whole text file) found in the
  item (--containment, flagged at or above it; reference items under 3 windows are ignored).
The exit status is 1 when any item is flagged.

Standard licence and header texts (tools/t5/boilerplate/*.txt: MIT, BSD, Apache, GPL notices, Go/Rust/Python
headers) and copyright lines are removed from references and candidates before measuring, so shared boilerplate
is not an overlap.

Reference files (--against) to pass: the T1-T4 texts (t2.txt, t3.txt from tools/holdout/texts.py and the T1 text),
the T4 prompts JSONL (tools/holdout/t4_prompts.py), and the calibration corpus text: write the GSM8K, MBPP and
Hermes training-split prompts with `python -m gptq.calibration prompts OUT/prompts.jsonl T4_PROMPTS.jsonl --chat N`
(with UltraChat), the model answers with `generate` (OUT/generations.jsonl), and the prose and code files `tokens`
reads (wikitext-2 train and the calibration source files). Files ending in .jsonl contribute their string values,
one item each; other files are one text. Memory: every distinct window of every reference stays in memory (about
100 bytes each), so a multi-gigabyte reference needs a machine to match; split very large corpora and run the check
once per part, any part flagging an item failing the set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    from t5 import fixtures, records
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from t5 import fixtures, records

# Ideographs and kana are words of their own: those scripts do not separate words with spaces. Other scripts'
# words are runs of letters, digits and combining marks (Indic vowel signs and viragas, Arabic diacritics).
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f"
_MARKS = "\u0300-\u036f\u064b-\u065f\u0900-\u0963\u0966-\u0dff\u200c\u200d"
_WORD = re.compile(f"[{_CJK}]|(?:(?![{_CJK}])[\\w{_MARKS}])+")
# Reference items shorter than this many windows do not count towards containment (one shared window is no copy).
MIN_ITEM_SHINGLES = 3


def shingles(text: str, n: int = 13) -> set[int]:
    words = _WORD.findall(text.lower())
    return {int.from_bytes(hashlib.sha1(" ".join(words[i:i + n]).encode("utf-8")).digest()[:8], "big")
            for i in range(len(words) - n + 1)}


def overlap(candidate: str, reference: set[int], n: int = 13) -> float:
    """Fraction of the candidate's windows found in the reference."""
    found = shingles(candidate, n)
    return len(found & reference) / len(found) if found else 0.0


_NOTICE = re.compile(r"(?im)^[\s/#*;-]*(?:copyright|\(c\)).*$")
BOILERPLATE_DIR = Path(__file__).with_name("boilerplate")


def _strip(text: str) -> str:
    return _NOTICE.sub("", text)


def load_boilerplate(n: int = 13) -> set[int]:
    found: set[int] = set()
    for file in sorted(BOILERPLATE_DIR.glob("*.txt")):
        for line in _strip(file.read_text(encoding="utf-8")).splitlines():
            found |= shingles(line, n)
        found |= shingles(_strip(file.read_text(encoding="utf-8")), n)
    return found


class Reference:
    """The union of the reference files' windows, with each reference item (a JSONL string value or a whole text
    file) indexed so a candidate can be tested for containing most of one item."""

    def __init__(self, n: int = 13, boilerplate: set[int] | None = None) -> None:
        self.n = n
        self.boilerplate = load_boilerplate(n) if boilerplate is None else boilerplate
        self.union: set[int] = set()
        self._sizes: list[int] = []
        self._index: dict[int, list[int]] = {}

    def add(self, text: str) -> None:
        found = shingles(_strip(text), self.n) - self.boilerplate
        self.union |= found
        if len(found) >= MIN_ITEM_SHINGLES:
            item = len(self._sizes)
            self._sizes.append(len(found))
            for h in found:
                self._index.setdefault(h, []).append(item)

    def measure(self, text: str) -> dict:
        """fraction: share of the text's windows in the references (diluted by long text); shared: number of such
        windows; containment: the largest share of one reference item's windows found in the text."""
        found = shingles(_strip(text), self.n) - self.boilerplate
        shared = found & self.union
        counts: dict[int, int] = {}
        for h in shared:
            for item in self._index.get(h, ()):
                counts[item] = counts.get(item, 0) + 1
        containment = max((c / self._sizes[i] for i, c in counts.items()), default=0.0)
        return {"fraction": len(shared) / len(found) if found else 0.0, "shared": len(shared),
                "containment": containment}


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


def _worst(measures: list[dict]) -> dict:
    return {key: max((m[key] for m in measures), default=0) for key in ("fraction", "shared", "containment")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t5.overlap")
    parser.add_argument("store", type=Path)
    parser.add_argument("--against", type=Path, nargs="+", required=True)
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--limit", type=float, default=0.05, help="flag when the shared fraction exceeds this")
    parser.add_argument("--min-shared", type=int, default=5, help="flag when this many windows are shared")
    parser.add_argument("--containment", type=float, default=0.5,
                        help="flag when this share of one reference item is contained")
    args = parser.parse_args(argv)
    reference = Reference(args.n)
    hashes = []
    for path in args.against:
        for text in reference_texts(path):
            reference.add(text)
        hashes.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    items: dict = {}
    conversations = args.store / "conversations.jsonl"
    if conversations.exists():
        for c in records.load_conversations(conversations):
            texts = list(c.turns) + ([c.system] if c.system else [])
            if c.tools:
                texts.append(fixtures.canonical(c.tools))
            if c.fixture and (args.store / c.fixture).is_file():
                texts.append((args.store / c.fixture).read_text(encoding="utf-8"))
            items[c.id] = _worst([reference.measure(t) for t in texts])
    documents = args.store / "documents.jsonl"
    if documents.exists():
        for d in records.load_documents(documents):
            text = (args.store / d.path).read_text(encoding="utf-8", errors="replace")
            items[d.id] = reference.measure(text)
    flagged = sorted(k for k, v in items.items()
                     if v["fraction"] > args.limit or v["shared"] >= args.min_shared
                     or v["containment"] >= args.containment)
    out = args.store / "checks"
    out.mkdir(exist_ok=True)
    report = {"items": items, "references": hashes, "n": args.n,
              "limits": {"fraction": args.limit, "min_shared": args.min_shared, "containment": args.containment}}
    (out / "overlap.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for item in flagged:
        v = items[item]
        print(f"{item}: fraction {v['fraction']:.3f}, shared {v['shared']}, containment {v['containment']:.3f}")
    print(f"{len(items)} items, {len(flagged)} flagged")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
