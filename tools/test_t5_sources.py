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
        self.assertEqual(sources.github_path_first_commit("o", "r", "a/b.py", fetch=lambda url: pages[url]),
                         datetime.date(2026, 10, 1))

    def test_first_commit_single_page(self):
        fetch = lambda url: ([commit("2026-10-09T00:00:00Z"), commit("2026-10-04T00:00:00Z")], {})  # noqa: E731
        self.assertEqual(sources.github_path_first_commit("o", "r", "x", fetch=fetch), datetime.date(2026, 10, 4))

    def test_published_date_dispatch(self):
        fetch = lambda url: ({"created_at": "2026-10-02T00:00:00Z"}, {})  # noqa: E731
        self.assertEqual(sources.published_date("https://github.com/o/r/pull/3", fetch),
                         (datetime.date(2026, 10, 2), None))
        self.assertIsNone(sources.published_date("https://example.com/x/y/z/w", fetch))

    def test_blob_with_slash_ref_and_repository_date(self):
        seen = []

        def fetch(url):
            seen.append(url)
            if url.endswith("/repos/o/r"):
                return {"created_at": "2026-09-30T00:00:00Z"}, {}
            if "sha=feature%2Fx&path=src/m.py" in url:
                return [commit("2026-10-05T00:00:00Z")], {}
            raise OSError("404")

        got = sources.published_date("https://github.com/o/r/blob/feature/x/src/m.py", fetch)
        self.assertEqual(got, (datetime.date(2026, 10, 5), datetime.date(2026, 9, 30)))
        self.assertIn("sha=feature%2Fx%2Fsrc&path=m.py", seen[0])
        self.assertTrue(any("sha=feature%2Fx&path=src/m.py" in u for u in seen))

    def test_longest_ref_is_tried_first(self):
        seen = []

        def fetch(url):
            seen.append(url)
            if url.endswith("/repos/o/r"):
                return {"created_at": "2026-09-30T00:00:00Z"}, {}
            return [commit("2026-10-05T00:00:00Z")], {}

        sources.published_date("https://github.com/o/r/blob/a/b/c/f.py", fetch)
        self.assertIn("sha=a%2Fb%2Fc&path=f.py", seen[0])

    def _file_store(self, d, fetch):
        row = {"id": "c", "domain": "coding", "order": 0, "language": "en", "turns": ["x"],
               "source": {"kind": "human", "url": "https://github.com/o/r/blob/main/a.py",
                          "published": "2026-10-05", "license": "MIT", "checked": "t"}}
        (Path(d) / "conversations.jsonl").write_text(json.dumps(row))
        return sources.check(Path(d), fetch)

    def test_file_added_after_the_cutoff_to_an_old_repository_passes(self):
        def fetch(url):
            if url.endswith("/repos/o/r"):
                return {"created_at": "2026-01-01T00:00:00Z"}, {}
            if "/commits/" in url:
                return {"parents": [{"sha": "p"}], "files": [{"filename": "a.py", "status": "added"}]}, {}
            return [{"sha": "s1", **commit("2026-10-05T00:00:00Z")}], {}
        with tempfile.TemporaryDirectory() as d:
            result, failed = self._file_store(d, fetch)
        self.assertFalse(failed)
        self.assertNotIn("error", result["c"])
        self.assertEqual(result["c"]["repository_created"], "2026-01-01")

    def test_new_repository_passes(self):
        def fetch(url):
            if url.endswith("/repos/o/r"):
                return {"created_at": "2026-09-30T00:00:00Z"}, {}
            if "/commits/" in url:
                return {"parents": [], "files": [{"filename": "a.py", "status": "added"}]}, {}
            return [{"sha": "s1", **commit("2026-10-05T00:00:00Z")}], {}
        with tempfile.TemporaryDirectory() as d:
            _, failed = self._file_store(d, fetch)
        self.assertFalse(failed)

    def _renamed(self, status, old_date):
        def fetch(url):
            if url.endswith("/repos/o/r"):
                return {"created_at": "2026-01-01T00:00:00Z"}, {}
            if url.endswith("/commits/s1"):
                return {"parents": [{"sha": "p1"}], "files": [
                    {"filename": "a.py", "status": status, "previous_filename": "old/a.py"}]}, {}
            if url.endswith("/commits/s0"):
                return {"parents": [], "files": [{"filename": "old/a.py", "status": "added"}]}, {}
            if "sha=p1&path=old/a.py" in url:
                return [{"sha": "s0", **commit(old_date)}], {}
            return [{"sha": "s1", **commit("2026-10-05T00:00:00Z")}], {}
        return fetch

    def test_file_renamed_after_the_cutoff_from_an_earlier_path_fails(self):
        for status in ("renamed", "copied"):
            with tempfile.TemporaryDirectory() as d:
                result, failed = self._file_store(d, self._renamed(status, "2026-03-01T00:00:00Z"))
            self.assertTrue(failed, status)
            self.assertEqual(result["c"]["verified"], "2026-03-01")
            self.assertEqual(result["c"]["error"], "date is not after the cutoff")

    def test_rename_from_a_path_first_added_after_the_cutoff_passes(self):
        with tempfile.TemporaryDirectory() as d:
            result, failed = self._file_store(d, self._renamed("renamed", "2026-10-05T00:00:00Z"))
        self.assertFalse(failed)

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
