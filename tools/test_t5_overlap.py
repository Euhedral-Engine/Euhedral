from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t5 import overlap  # noqa: E402

REFERENCE = " ".join(f"ref{i}" for i in range(60))
UNRELATED = " ".join(f"other{i}" for i in range(60))


class OverlapTest(unittest.TestCase):
    def test_copy_is_one_and_unrelated_is_zero(self):
        ref = overlap.shingles(REFERENCE)
        self.assertEqual(overlap.overlap(REFERENCE, ref), 1.0)
        self.assertEqual(overlap.overlap(UNRELATED, ref), 0.0)

    def test_partial_copy(self):
        ref = overlap.shingles(REFERENCE)
        mixed = " ".join(f"ref{i}" for i in range(20)) + " " + UNRELATED
        self.assertTrue(0.0 < overlap.overlap(mixed, ref) < 0.5)

    def test_short_text_is_zero(self):
        ref = overlap.shingles(REFERENCE)
        self.assertEqual(overlap.overlap("ref1 ref2 ref3", ref), 0.0)
        self.assertEqual(overlap.shingles("a b c", 13), set())

    def test_case_and_punctuation_insensitive(self):
        self.assertEqual(overlap.shingles("Alpha, beta! gamma", 3), overlap.shingles("alpha beta gamma", 3))

    def test_cjk_has_shingles(self):
        self.assertTrue(overlap.shingles("今日は天気がとても良いので散歩に行きます", 13))

    def test_hindi_tokenizes_to_words(self):
        self.assertEqual(overlap._WORD.findall("हिन्दी भाषा"), ["हिन्दी", "भाषा"])
        self.assertEqual(len(overlap.shingles("हिन्दी भाषा बहुत सुंदर है और लोग इसे बोलते हैं", 3)), 8)

    def test_embedded_question_in_long_turn_is_flagged(self):
        question = " ".join(f"gsm{i}" for i in range(30))
        turn = " ".join(f"filler{i}" for i in range(400)) + " " + question + " " + \
            " ".join(f"tail{i}" for i in range(400))
        ref = overlap.Reference()
        ref.add(question)
        m = ref.measure(turn)
        self.assertLess(m["fraction"], 0.05)
        self.assertEqual(m["shared"], 18)
        self.assertEqual(m["containment"], 1.0)
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            (store / "ref.jsonl").write_text(json.dumps({"question": question}))
            row = {"id": "a", "domain": "conversation", "order": 0, "language": "en", "turns": [turn]}
            (store / "conversations.jsonl").write_text(json.dumps(row))
            self.assertEqual(overlap.main([str(store), "--against", str(store / "ref.jsonl")]), 1)

    def test_tools_and_fixture_text_are_checked(self):
        phrase = " ".join(f"word{i}" for i in range(40))
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            (store / "ref.txt").write_text(phrase)
            (store / "fx.json").write_text(json.dumps({"tools": {"t": {"default": phrase}}}))
            row = {"id": "a", "domain": "tool_use", "order": 0, "language": "en", "turns": ["hello there"],
                   "tools": [{"type": "function", "function": {"name": "t", "description": phrase}}],
                   "fixture": "fx.json"}
            (store / "conversations.jsonl").write_text(json.dumps(row))
            self.assertEqual(overlap.main([str(store), "--against", str(store / "ref.txt")]), 1)
            self.assertGreaterEqual(json.loads((store / "checks" / "overlap.json").read_text())["items"]["a"]["shared"], 5)

    def test_main_writes_report_and_fails_over_limit(self):
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            ref = store / "ref.txt"
            ref.write_text(REFERENCE)
            rows = [{"id": "a", "domain": "conversation", "order": 0, "language": "en", "turns": [REFERENCE]},
                    {"id": "b", "domain": "conversation", "order": 1, "language": "en", "turns": [UNRELATED]}]
            (store / "conversations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            self.assertEqual(overlap.main([str(store), "--against", str(ref)]), 1)
            report = json.loads((store / "checks" / "overlap.json").read_text())
            self.assertEqual(report["items"]["a"]["fraction"], 1.0)
            self.assertEqual(report["items"]["b"]["fraction"], 0.0)
            self.assertEqual(report["n"], 13)
            self.assertEqual(len(report["references"][0]["sha256"]), 64)
            (store / "conversations.jsonl").write_text(json.dumps(rows[1]))
            self.assertEqual(overlap.main([str(store), "--against", str(ref)]), 0)


if __name__ == "__main__":
    unittest.main()
