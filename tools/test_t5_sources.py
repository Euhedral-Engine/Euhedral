from __future__ import annotations

import datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t5 import sources  # noqa: E402


def commit(date):
    return {"commit": {"author": {"date": date}, "committer": {"date": date}}}


class SourcesTest(unittest.TestCase):
    def test_issue_date_from_created_at(self):
        seen = []

        def fetch(url):
            seen.append(url)
            return {"created_at": "2026-10-02T23:30:00Z"}, {}

        self.assertEqual(sources.github_issue_created("o", "r", 7, fetch), datetime.date(2026, 10, 2))
        self.assertTrue(seen[0].endswith("/repos/o/r/issues/7"))

    def test_first_commit_follows_last_page(self):
        pages = {
            "https://api.github.com/repos/o/r/commits?path=a/b.py&per_page=100": (
                [commit("2026-11-05T00:00:00Z"), commit("2026-10-20T00:00:00Z")],
                {"link": '<https://api.github.com/p?page=2>; rel="next", <https://api.github.com/p?page=2>; rel="last"'}),
            "https://api.github.com/p?page=2": ([commit("2026-10-03T00:00:00Z"), commit("2026-10-01T12:00:00Z")], {}),
        }
        self.assertEqual(sources.github_path_first_commit("o", "r", "a/b.py", lambda url: pages[url]),
                         datetime.date(2026, 10, 1))

    def test_first_commit_single_page(self):
        fetch = lambda url: ([commit("2026-10-09T00:00:00Z"), commit("2026-10-04T00:00:00Z")], {})  # noqa: E731
        self.assertEqual(sources.github_path_first_commit("o", "r", "x", fetch), datetime.date(2026, 10, 4))

    def test_published_date_dispatch(self):
        fetch = lambda url: ({"created_at": "2026-10-02T00:00:00Z"}, {})  # noqa: E731
        self.assertEqual(sources.published_date("https://github.com/o/r/pull/3", fetch), datetime.date(2026, 10, 2))
        self.assertIsNone(sources.published_date("https://example.com/x/y/z/w", fetch))

    def test_check_flags_mismatch_and_old_dates(self):
        rows = []
        for i, published in enumerate(["2026-10-02", "2026-10-03", "2026-09-01"]):
            rows.append({"id": f"c{i}", "domain": "coding", "order": i, "language": "en", "turns": ["x"],
                         "source": {"kind": "human", "url": f"https://github.com/o/r/issues/{i}",
                                    "published": published, "license": "MIT", "checked": "t"}})
        dates = {"0": "2026-10-02T00:00:00Z", "1": "2026-10-02T00:00:00Z", "2": "2026-09-01T00:00:00Z"}
        fetch = lambda url: ({"created_at": dates[url.rsplit("/", 1)[1]]}, {})  # noqa: E731
        with tempfile.TemporaryDirectory() as d:
            store = Path(d)
            (store / "conversations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            result, failed = sources.check(store, fetch)
        self.assertTrue(failed)
        self.assertNotIn("error", result["c0"])
        self.assertEqual(result["c1"]["error"], "date differs from the record")
        self.assertEqual(result["c2"]["error"], "date is not after the cutoff")


if __name__ == "__main__":
    unittest.main()
