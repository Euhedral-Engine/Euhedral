from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t5 import fixtures  # noqa: E402

SPEC = {
    "tools": {
        "weather": {"rules": [{"match": {"city": "Oslo", "unit": "c"}, "result": {"temp": 3}},
                              {"match": {"city": "Oslo"}, "result": {"temp": 99}}],
                    "default": {"temp": 0}},
        "bare": {"rules": []},
        "calc": {"builtin": "calculator"},
        "read": {"builtin": "read_file"},
        "ls": {"builtin": "list_directory"},
        "write": {"builtin": "write_file"},
    },
    "files": {"src/a.txt": "alpha", "src/b.txt": "beta", "top.txt": "top"},
}


class FixtureTest(unittest.TestCase):
    def test_match_ignores_key_order(self):
        f = fixtures.Fixture(SPEC)
        self.assertEqual(f.call("weather", '{"unit":"c","city":"Oslo"}'), '{"temp":3}')
        self.assertEqual(f.call("weather", {"unit": "c", "city": "Oslo"}), '{"temp":3}')

    def test_first_match_wins_and_default(self):
        f = fixtures.Fixture(SPEC)
        self.assertEqual(f.call("weather", {"city": "Oslo", "unit": "f"}), '{"temp":99}')
        self.assertEqual(f.call("weather", {"city": "Rome"}), '{"temp":0}')

    def test_no_default(self):
        self.assertEqual(fixtures.Fixture(SPEC).call("bare", {"x": 1}),
                         '{"error":"no result for these arguments"}')

    def test_unknown_tool_and_bad_arguments(self):
        f = fixtures.Fixture(SPEC)
        self.assertEqual(f.call("nope", {}), '{"error":"unknown tool nope"}')
        for bad in ("{not json", "[1]", "3"):
            self.assertEqual(f.call("weather", bad), '{"error":"arguments are not a JSON object"}')

    def test_calculator(self):
        f = fixtures.Fixture(SPEC)
        self.assertEqual(f.call("calc", {"expression": "2*(3+4)"}), '{"result":14}')
        self.assertEqual(fixtures.calculate("7 // 2 + 2 ** 3 - -1"), '{"result":12}')
        self.assertIn("error", json.loads(f.call("calc", {"expression": "__import__('os')"})))
        self.assertIn("error", json.loads(fixtures.calculate("1/0")))
        self.assertIn("error", json.loads(fixtures.calculate("9**9**9")))
        self.assertIn("error", json.loads(f.call("calc", {})))

    def test_file_write_then_read_is_per_fixture(self):
        f = fixtures.Fixture(SPEC)
        f.call("write", {"path": "src/a.txt", "content": "changed"})
        self.assertEqual(json.loads(f.call("read", {"path": "src/a.txt"}))["content"], "changed")
        fresh = fixtures.Fixture(SPEC)
        self.assertEqual(json.loads(fresh.call("read", {"path": "src/a.txt"}))["content"], "alpha")
        self.assertEqual(fresh.call("read", {"path": "gone"}), '{"error":"no such file gone"}')

    def test_list_directory(self):
        f = fixtures.Fixture(SPEC)
        f.call("write", {"path": "src/c.txt", "content": "x"})
        self.assertEqual(json.loads(f.call("ls", {"path": "src"}))["entries"], ["a.txt", "b.txt", "c.txt"])
        self.assertEqual(json.loads(f.call("ls", {"path": "."}))["entries"], ["src/", "top.txt"])

    def test_identical_sequences_are_byte_identical(self):
        calls = [("weather", {"city": "Oslo"}), ("write", {"path": "n", "content": "1"}),
                 ("read", {"path": "n"}), ("ls", {"path": "."}), ("calc", {"expression": "1/3"})]
        runs = []
        for _ in range(2):
            f = fixtures.Fixture(SPEC)
            runs.append([f.call(name, args) for name, args in calls])
        self.assertEqual(runs[0], runs[1])

    def test_validate_spec(self):
        self.assertEqual(fixtures.validate_spec({"tools": {k: v for k, v in SPEC["tools"].items() if k != "bare"}}, ["weather", "calc"]), [])
        bad = {"tools": {"a": {"builtin": "shell"}, "b": {"rules": [{"match": {}}]}, "c": {"rules": []}, "d": 3}}
        errors = fixtures.validate_spec(bad, ["missing"])
        for needle in ("missing has no fixture", "unknown builtin", "rule 0", "no rules and no default", "not an object"):
            self.assertTrue(any(needle in e for e in errors), needle)
        self.assertTrue(fixtures.validate_spec([]))

    def test_paths_are_normalized(self):
        f = fixtures.Fixture({"tools": SPEC["tools"], "files": {"/src/a.txt": "alpha"}})
        for path in ("src/a.txt", "./src/a.txt", "/src/a.txt"):
            self.assertEqual(json.loads(f.call("read", {"path": path}))["content"], "alpha")
        self.assertEqual(json.loads(f.call("ls", {"path": "./src/"}))["entries"], ["a.txt"])

    def test_integral_floats_match_ints(self):
        f = fixtures.Fixture({"tools": {"t": {"rules": [{"match": {"n": 3}, "result": "three"}]}}})
        self.assertEqual(f.call("t", '{"n": 3.0}'), '"three"')

    def test_non_finite_arguments(self):
        f = fixtures.Fixture({"tools": {"t": {"rules": [{"match": {"n": 3}, "result": 1}], "default": 0}}})
        for text in ('{"n": Infinity}', '{"n": -Infinity}', '{"n": NaN}', '{"n": [1e999]}'):
            self.assertEqual(f.call("t", text), "0")

    def test_calculator_bounds(self):
        for expression in ("9**999", "(10**900)*(10**900)*(10**900)*(10**900)", "2**-100000", "10**10**3"):
            result = json.loads(fixtures.calculate(expression))
            self.assertTrue("error" in result or len(str(result["result"])) < 5000, expression)
        self.assertIn("error", json.loads(fixtures.calculate("7**7777")))

    def test_load_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.json"
            path.write_text(json.dumps(SPEC))
            self.assertEqual(fixtures.load_fixture(path).call("weather", {"city": "Rome"}), '{"temp":0}')


if __name__ == "__main__":
    unittest.main()
