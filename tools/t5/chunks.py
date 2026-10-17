"""Cuts the evaluation set into its scored chunks and writes the frozen corpus.

Usage (from tools/): python3 -m t5.chunks STORE --tokenizer DIR [--chunk 2560]

Needs the `tokenizers` package. Reads STORE/generation/progress.json and summary.json (the generated conversations and
the selection per domain) and STORE/documents.jsonl (the human-written documents; per kind the first
records.DOCUMENT_CHUNKS[kind] by `order`). Writes, under STORE/frozen/:
- chunks.jsonl: one line per chunk in scoring order (id, domain, source, decoding, tokens, sha256 of the token ids,
  and how the chunk was cut). Each document chunk follows a generated chunk whose end joins it without a token
  spanning the join, and precedes another generated chunk, which starts with a special token (`arrange`);
- tokens.bin: every chunk's token ids, int32 little-endian, chunk after chunk;
- text.txt: the chunks' text, which llama.cpp's `llama-perplexity --parse-special -c CHUNK` splits back into exactly
  these chunks (the reference logits are computed from it).

A chunk is the first CHUNK tokens of a conversation as the model saw and wrote it (chat template, thinking and tool
calls included), or of a document. The reference implementation tokenizes text, so a chunk must be the tokenization of
its own text: a conversation whose first CHUNK ids do not survive decoding and encoding again (a sampled token sequence
the tokenizer would split differently, or a cut inside a character) is cut at the nearest prefix whose text encodes to
exactly CHUNK tokens, and that encoding is the chunk. The whole text must encode to the chunks laid end to end;
the command fails otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from . import records

DOMAINS = ("conversation", "coding", "tool_use", "multilingual")
KINDS = ("prose", "code")
SEARCH = 256  # prefix lengths tried on either side of CHUNK when a cut does not round-trip


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


def cuts(tokenizer, ids: list[int], chunk: int):
    """Candidate cuts, nearest prefix length first: (chunk ids, text, prefix length of `ids`)."""
    if len(ids) < chunk:
        raise ValueError(f"{len(ids)} tokens, need {chunk}")
    for distance in range(0, SEARCH + 1):
        for length in ((chunk,) if distance == 0 else (chunk - distance, chunk + distance)):
            if length > len(ids):
                continue
            text = tokenizer.decode(ids[:length], skip_special_tokens=False)
            if text.endswith("\ufffd"):  # cut inside a character
                continue
            encoded = tokenizer.encode(text, add_special_tokens=False).ids
            if len(encoded) == chunk:
                yield encoded, text, length


def joins(tokenizer, left: dict, right: dict) -> bool:
    """Whether two chunks' texts laid end to end encode to their ids end to end (no token spans the join)."""
    return tokenizer.encode(left["text"] + right["text"], add_special_tokens=False).ids == left["ids"] + right["ids"]


def sources(store: Path, tokenizer) -> list[dict]:
    """Every chunk's source in scoring order, with its full token ids."""
    progress = json.loads((store / "generation" / "progress.json").read_text(encoding="utf-8"))["conversations"]
    summary = json.loads((store / "generation" / "summary.json").read_text(encoding="utf-8"))["domains"]
    out: list[dict] = []
    for domain in DOMAINS:
        selected = summary[domain]["selected"]
        if len(selected) != records.CHUNKS[domain]:
            raise ValueError(f"{domain}: {len(selected)} selected, need {records.CHUNKS[domain]}")
        for cid in selected:
            record = progress[cid]
            out.append({"id": cid, "domain": domain, "source": "generated", "decoding": record["decoding"]["mode"],
                        "all": record["tokens"]})
    documents = records.load_documents(store / "documents.jsonl")
    for kind in KINDS:
        used = sorted((d for d in documents if d.kind == kind), key=lambda d: d.order)[:records.DOCUMENT_CHUNKS[kind]]
        for document in used:
            text = (store / document.path).read_text(encoding="utf-8")
            out.append({"id": document.id, "domain": kind, "source": "document", "decoding": None,
                        "all": tokenizer.encode(text, add_special_tokens=False).ids})
    return out


def arrange(tokenizer, rows: list[dict]) -> list[dict]:
    """Orders the chunks so that no token spans a join. A generated chunk starts with a special token, which joins
    any text; a document chunk goes right after a generated chunk whose end joins it, and a generated chunk follows it.
    The pairing is a bipartite matching (augmenting paths); the other generated chunks come last, in their order."""
    generated = [r for r in rows if r["source"] == "generated"]
    documents = [r for r in rows if r["source"] == "document"]
    fits = [[g for g, left in enumerate(generated) if joins(tokenizer, left, d)] for d in documents]
    partner: dict[int, int] = {}  # generated index -> document index

    def augment(d: int, seen: set[int]) -> bool:
        for g in fits[d]:
            if g not in seen:
                seen.add(g)
                if g not in partner or augment(partner[g], seen):
                    partner[g] = d
                    return True
        return False

    for d in range(len(documents)):
        if not augment(d, set()):
            raise ValueError(f"{documents[d]['id']}: no generated chunk's end joins its start")
    if len(generated) <= len(documents):
        raise ValueError("a generated chunk must follow every document chunk")
    ordered: list[dict] = []
    for g, left in enumerate(generated):
        ordered.append(left)
        if g in partner:
            ordered.append(documents[partner[g]])
    # The text's end is not followed by anything, and llama.cpp splits trailing whitespace there differently: the
    # corpus ends on an unpaired generated chunk whose text ends in a visible character.
    last = next((i for i in range(len(ordered) - 1, -1, -1) if ordered[i]["source"] == "generated"
                 and (i + 1 == len(ordered) or ordered[i + 1]["source"] == "generated")
                 and not ordered[i]["text"][-1:].isspace()), None)
    if last is None:
        raise ValueError("no unpaired generated chunk ends in a visible character")
    ordered.append(ordered.pop(last))
    return ordered


def build(store: Path, tokenizer, chunk: int) -> list[dict]:
    rows = sources(store, tokenizer)
    for row in rows:
        try:
            row["ids"], row["text"], row["prefix"] = next(cuts(tokenizer, row["all"], chunk))
        except (ValueError, StopIteration) as error:
            raise ValueError(f"{row['id']}: {error or f'no prefix within {SEARCH} tokens of {chunk} encodes to exactly {chunk} tokens'}") from None
        row["differing"] = sum(a != b for a, b in zip(row["ids"], row.pop("all")[:chunk]))
    rows = arrange(tokenizer, rows)
    whole = tokenizer.encode("".join(r["text"] for r in rows), add_special_tokens=False).ids
    expected = [i for r in rows for i in r["ids"]]
    if whole != expected:
        first = next((i for i, (a, b) in enumerate(zip(whole, expected)) if a != b), min(len(whole), len(expected)))
        raise ValueError(f"the text does not encode to the chunks end to end: first difference at token {first} "
                         f"(chunk {rows[min(first // chunk, len(rows) - 1)]['id']}), {len(whole)} tokens against "
                         f"{len(expected)}")
    return rows


def write(store: Path, rows: list[dict]) -> Path:
    out = store / "frozen"
    out.mkdir(exist_ok=True)
    with (out / "tokens.bin").open("wb") as handle:
        for r in rows:
            handle.write(struct.pack(f"<{len(r['ids'])}i", *r["ids"]))
    (out / "text.txt").write_text("".join(r["text"] for r in rows), encoding="utf-8")
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for index, r in enumerate(rows):
            handle.write(json.dumps({"index": index, "id": r["id"], "domain": r["domain"], "source": r["source"],
                                     "decoding": r["decoding"], "tokens": len(r["ids"]),
                                     "ids_sha256": ids_sha256(r["ids"]), "prefix": r["prefix"],
                                     "differing": r["differing"]}) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True, help="directory holding tokenizer.json")
    parser.add_argument("--chunk", type=int, default=2560)
    args = parser.parse_args(argv)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    try:
        rows = build(args.store, tokenizer, args.chunk)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    out = write(args.store, rows)
    recut = [r["id"] for r in rows if r["prefix"] != args.chunk or r["differing"]]
    print(f"{len(rows)} chunks of {args.chunk} tokens written to {out}; cut elsewhere or re-encoded: "
          f"{len(recut)} ({', '.join(recut) if recut else 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
