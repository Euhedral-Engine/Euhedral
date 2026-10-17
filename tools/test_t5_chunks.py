from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t5 import chunks, run  # noqa: E402


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class CharTokenizer:
    """One token per character, except that "`<" is one token and "<s>" is a special token, so joins can merge."""

    SPECIAL = "<s>"

    def encode(self, text, add_special_tokens=False):
        ids = []
        for index, part in enumerate(text.split(self.SPECIAL)):  # special tokens split the text first
            if index:
                ids.append(1)
            i = 0
            while i < len(part):
                if part.startswith("`<", i):
                    ids.append(2)
                    i += 2
                else:
                    ids.append(1000 + ord(part[i]))
                    i += 1
        return _Encoding(ids)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.SPECIAL if i == 1 else "`<" if i == 2 else chr(i - 1000) for i in ids)


def row(id, source, text, tokenizer):
    return {"id": id, "domain": "d", "source": source, "decoding": None, "text": text,
            "ids": tokenizer.encode(text).ids}


class ChunkTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer()

    def test_a_cut_that_re_encodes_differently_takes_the_nearest_prefix_of_the_right_length(self):
        ids = self.tokenizer.encode("ab`<cd").ids  # a, b, `<, c, d: five tokens
        cut_ids, text, length = next(chunks.cuts(self.tokenizer, ids, 3))
        self.assertEqual((cut_ids, text, length), (ids[:3], "ab`<", 3))

    def test_a_document_goes_after_a_generated_chunk_whose_end_does_not_merge_with_it(self):
        t = self.tokenizer
        rows = [row("g1", "generated", "<s>x`", t), row("g2", "generated", "<s>yz", t),
                row("g3", "generated", "<s>w.", t), row("doc", "document", "<p", t)]
        ordered = chunks.arrange(t, rows)
        ids = [r["id"] for r in ordered]
        self.assertEqual(ids[ids.index("doc") - 1], "g2")  # "x`" + "<p" would merge into "`<"
        text = "".join(r["text"] for r in ordered)
        self.assertEqual(t.encode(text).ids, [i for r in ordered for i in r["ids"]])

    def test_the_corpus_ends_on_a_chunk_whose_text_ends_in_a_visible_character(self):
        t = self.tokenizer
        rows = [row("g1", "generated", "<s>a.", t), row("g2", "generated", "<s>b ", t)]
        self.assertEqual([r["id"] for r in chunks.arrange(t, rows)], ["g2", "g1"])

    def test_a_document_no_generated_chunk_can_precede_is_refused(self):
        t = self.tokenizer
        rows = [row("g1", "generated", "<s>`", t), row("g2", "generated", "<s>a", t),
                row("doc", "document", "<p", t)]
        rows[1]["text"] = "<s>`"
        rows[1]["ids"] = t.encode(rows[1]["text"]).ids
        with self.assertRaises(ValueError):
            chunks.arrange(t, rows)


class RunTest(unittest.TestCase):
    def test_values_written_by_score_read_back_per_chunk_and_token(self):
        values = [[(1.0, 0.5, 0.25, 1.0, 0.0), (2.0, 1.5, 0.5, 0.0, 0.1)], [(0.1, 0.1, 0.0, 1.0, 0.0)] * 2]
        flat = [v for chunk in values for token in chunk for v in token]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.values"
            path.write_bytes(struct.pack("<3i", 2, 2, 5) + struct.pack(f"<{len(flat)}d", *flat))
            self.assertEqual(run.read_values(path), values)

    def test_summary_and_paired_difference(self):
        a = [[(1.0, 0.5, 0.2, 1.0, 0.0)] * 4, [(1.0, 1.0, 0.4, 0.0, 0.0)] * 4]
        s = run.summary(a)
        self.assertAlmostEqual(s["kl"], 0.3)
        self.assertAlmostEqual(s["top1"], 0.5)
        self.assertAlmostEqual(s["dnll"], 0.25)
        p = run.paired([0.3, 0.5], [0.2, 0.4], ["x", "y"], resamples=200)
        self.assertAlmostEqual(p["difference"], 0.1)
        self.assertAlmostEqual(p["low"], 0.1)  # every chunk differs by the same amount
        self.assertAlmostEqual(p["relative"], 0.1 / 0.3)

    def test_the_report_names_every_domain_and_compares_with_the_baseline(self):
        chunk = [(1.0, 0.5, 0.2, 1.0, 0.0)] * 3
        text = run.report({"base": [chunk, chunk], "new": [chunk, chunk]}, ["x", "y"], "base", resamples=50)
        self.assertIn("overall (2 chunks)", text)
        self.assertIn("y (1 chunks)", text)
        self.assertIn("KL vs base", text)


if __name__ == "__main__":
    unittest.main()
