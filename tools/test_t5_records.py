from __future__ import annotations

import dataclasses
import datetime
import hashlib
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t5 import records  # noqa: E402

HUMAN = records.Source("human", "https://github.com/example/project/issues/1", datetime.date(2026, 10, 2),
                       "MIT", "github created_at")
CHUNKS = {"conversation": 2, "coding": 2, "tool_use": 2, "multilingual": 2}
DOC_CHUNKS = {"prose": 1, "code": 1}
LANGS = frozenset({"es", "zh"})
CODE_LANGS = frozenset({"python"})


def conv(id, domain="conversation", order=0, language="en", **kw):
    fields = dict(system=None, turns=("hello",), tools=(), fixture=None, source=records.Source("authored"))
    fields.update(kw)
    return records.Conversation(id, domain, order, language, **fields)


class RecordsTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store = Path(self._dir.name)
        (self.store / "fx.json").write_text(
            '{"tools": {"lookup": {"rules": [{"match": {"q": 1}, "result": 2}]}}}')
        self.excluded = records.load_excluded()

    def tearDown(self):
        self._dir.cleanup()

    def valid(self):
        tool = dict(tools=({"type": "function", "function": {"name": "lookup"}},), fixture="fx.json")
        return [conv("c0", order=0), conv("c1", order=1), conv("k0", "coding", 0), conv("k1", "coding", 1),
                conv("t0", "tool_use", 0, **tool), conv("t1", "tool_use", 1, **tool),
                conv("m0", "multilingual", 0, "es"), conv("m1", "multilingual", 1, "zh")]

    def check(self, items):
        return records.validate_conversations(items, self.store, self.excluded, chunks=CHUNKS, languages=LANGS)

    def test_valid_store_passes(self):
        self.assertEqual(self.check(self.valid()), [])

    def test_duplicate_id(self):
        items = self.valid() + [conv("c0", order=5)]
        self.assertTrue(any("c0" in e and "duplicate" in e for e in self.check(items)))

    def test_missing_fixture_and_tools(self):
        items = [dataclasses.replace(c, fixture="gone.json", tools=()) if c.id == "t1" else c for c in self.valid()]
        errors = self.check(items)
        self.assertTrue(any("t1" in e and "fixture" in e for e in errors))
        self.assertTrue(any("t1" in e and "tools" in e for e in errors))

    def test_date_on_the_cutoff(self):
        bad = dataclasses.replace(HUMAN, published=records.CUTOFF)
        items = [dataclasses.replace(c, source=bad) if c.id == "k0" else c for c in self.valid()]
        self.assertTrue(any("k0" in e and "not after" in e for e in self.check(items)))

    def test_human_source_after_cutoff_passes(self):
        items = [dataclasses.replace(c, source=HUMAN) if c.id == "k0" else c for c in self.valid()]
        self.assertEqual(self.check(items), [])

    def test_excluded_repository(self):
        bad = dataclasses.replace(HUMAN, url="https://github.com/python/cpython/issues/1")
        items = [dataclasses.replace(c, source=bad) if c.id == "k1" else c for c in self.valid()]
        self.assertTrue(any("k1" in e and "python/cpython" in e for e in self.check(items)))

    def test_excluded_dataset(self):
        url = "https://huggingface.co/datasets/openai/gsm8k/viewer"
        self.assertIn("openai/gsm8k", records.excluded_reason(url, self.excluded))

    def test_short_domain_duplicate_order_and_languages(self):
        items = [c for c in self.valid() if c.id != "m1"]
        items = [dataclasses.replace(c, order=0) if c.id == "c1" else c for c in items]
        errors = self.check(items)
        self.assertTrue(any("multilingual" in e and "need at least" in e for e in errors))
        self.assertTrue(any("conversation" in e and "order" in e for e in errors))
        self.assertTrue(any("missing languages" in e for e in errors))

    def test_empty_turn(self):
        items = [dataclasses.replace(c, turns=("",)) if c.id == "c0" else c for c in self.valid()]
        self.assertTrue(any("c0" in e and "turns" in e for e in self.check(items)))

    def test_plan_decoding(self):
        items = self.valid() + [conv(f"x{i}", order=10 + i) for i in range(6)]
        plan = records.plan_decoding(items, 7)
        self.assertEqual(plan, records.plan_decoding(list(reversed(items)), 7))
        for domain in records.SAMPLED_DOMAINS:
            members = [c for c in items if c.domain == domain]
            sampled = [c for c in members if plan[c.id].mode == "sampled"]
            self.assertEqual(len(sampled), len(members) // 2)
        self.assertTrue(all(plan[c.id].mode == "greedy" for c in items if c.domain == "tool_use"))
        sampled = next(d for d in plan.values() if d.mode == "sampled")
        self.assertEqual(sampled.request_fields()["seed"], sampled.seed)
        self.assertEqual(sampled.request_fields()["top_k"], 20)
        self.assertEqual(records.Decoding("greedy", None).request_fields(), {"temperature": 0})
        self.assertNotEqual(plan, records.plan_decoding(items, 8))

    def document(self, text=b"print(1)\n", **kw):
        (self.store / "d.py").write_bytes(text)
        fields = dict(id="d0", kind="code", language="python", path="d.py",
                      sha256=hashlib.sha256(b"print(1)\n").hexdigest(), source=HUMAN)
        fields.update(kw)
        return records.Document(**fields)

    def check_docs(self, docs):
        prose = records.Document("p0", "prose", "en", "d.py", hashlib.sha256(b"print(1)\n").hexdigest(), HUMAN)
        return records.validate_documents(docs + [prose], self.store, self.excluded, chunks=DOC_CHUNKS,
                                          code_languages=CODE_LANGS)

    def test_document_valid(self):
        self.assertEqual(self.check_docs([self.document()]), [])

    def test_document_sha_mismatch_and_missing(self):
        errors = self.check_docs([self.document(text=b"changed\n")])
        self.assertTrue(any("d0" in e and "sha256" in e for e in errors))
        errors = self.check_docs([self.document(path="none.py")])
        self.assertTrue(any("d0" in e and "missing file" in e for e in errors))

    def test_document_date_and_exclusion(self):
        old = dataclasses.replace(HUMAN, published=datetime.date(2026, 9, 1))
        self.assertTrue(any("not after" in e for e in self.check_docs([self.document(source=old)])))
        bad = dataclasses.replace(HUMAN, url="https://raw.githubusercontent.com/golang/go/master/x.go")
        self.assertTrue(any("golang/go" in e for e in self.check_docs([self.document(source=bad)])))

    def test_excluded_site_by_host_suffix(self):
        for url in ("https://en.wikipedia.org/wiki/Oslo", "https://wikipedia.org/x"):
            self.assertIn("wikipedia.org", records.excluded_reason(url, self.excluded))
        self.assertIsNone(records.excluded_reason("https://notwikipedia.org/x", self.excluded))
        self.assertIn("Salesforce/wikitext",
                      records.excluded_reason("https://huggingface.co/datasets/Salesforce/wikitext", self.excluded))
        bad = dataclasses.replace(HUMAN, url="https://en.wikipedia.org/wiki/Oslo")
        items = [dataclasses.replace(c, source=bad) if c.id == "k0" else c for c in self.valid()]
        self.assertTrue(any("k0" in e and "wikipedia.org" in e for e in self.check(items)))

    def test_sampled_tool_use_fails(self):
        with mock.patch.object(records, "SAMPLED_DOMAINS", records.SAMPLED_DOMAINS | {"tool_use"}):
            errors = self.check(self.valid())
        self.assertTrue(any("t1" in e and "sampled tool_use" in e for e in errors))

    def test_fixture_must_cover_declared_tools(self):
        (self.store / "fx.json").write_text('{"tools": {"other": {"default": 1}}}')
        errors = self.check(self.valid())
        self.assertTrue(any("t0" in e and "tool lookup has no fixture entry" in e for e in errors))
        (self.store / "fx.json").write_text('{"tools": {"lookup": {"rules": [{"match": []}]}}}')
        self.assertTrue(any("t0" in e and "rule 0" in e for e in self.check(self.valid())))

    def test_order_type_and_regional_tags(self):
        items = [dataclasses.replace(c, order="1") if c.id == "c0" else c for c in self.valid()]
        self.assertTrue(any("c0" in e and "order" in e for e in self.check(items)))
        items = [dataclasses.replace(c, language="es-MX") if c.id == "m0" else c for c in self.valid()]
        self.assertEqual(self.check(items), [])

    def test_paths_stay_in_the_store(self):
        self.assertTrue(any("leaves the store" in e for e in self.check_docs([self.document(path="../x.py")])))
        items = [dataclasses.replace(c, fixture="../fx.json") if c.id == "t0" else c for c in self.valid()]
        self.assertTrue(any("t0" in e and "fixture" in e for e in self.check(items)))

    def test_json_round_trip(self):
        path = self.store / "c.jsonl"
        path.write_text('{"id":"a","domain":"coding","order":3,"language":"en","turns":["hi"],'
                        '"source":{"kind":"human","url":"u","published":"2026-10-02","license":"MIT","checked":"x"}}\n')
        [c] = records.load_conversations(path)
        self.assertEqual(c.source.published, datetime.date(2026, 10, 2))
        self.assertEqual(c.turns, ("hi",))


if __name__ == "__main__":
    unittest.main()
