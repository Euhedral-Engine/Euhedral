"""ci-landed proves a commit pushed to an integration branch is the merge of a pull request head that passed its land run."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ci_landed", ROOT / "scripts/ci-landed.py")
ci_landed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_landed)

HEAD, MERGE, BASE = "h" * 40, "m" * 40, "b" * 40


def run(**fields):
    base = {"head_sha": HEAD, "path": "ci.yaml@refs/pull/7/head", "event": "pull_request", "status": "completed", "conclusion": "success"}
    return {**base, **fields}


def check(runs, merge_tree="t1", head_tree="t1", parents=(BASE, HEAD), on_main=False):
    trees = {MERGE: merge_tree, HEAD: head_tree}
    return ci_landed.verdict(MERGE, on_main, list(parents), trees.__getitem__, lambda commit: runs if commit == HEAD else [])


class LandedTest(unittest.TestCase):
    def test_the_merge_of_a_head_with_a_green_land_run_and_its_tree_is_landed(self):
        self.assertIsNone(check([run()]))

    def test_a_commit_on_main_is_landed_without_a_run(self):
        self.assertIsNone(check([], parents=(BASE,), on_main=True))

    def test_a_commit_that_is_not_a_merge_is_refused(self):
        self.assertIn("not a merge commit", check([run()], parents=(BASE,)))

    def test_a_merge_with_another_tree_is_refused(self):
        self.assertIn("tree", check([run()], head_tree="t2"))

    def test_a_head_without_a_successful_land_run_is_refused(self):
        for runs in (
            [],
            [run(conclusion="failure")],
            [run(status="running", conclusion=None)],
            [run(event="push")],
            [run(path="other.yaml@refs/pull/7/head")],
            [run(head_sha="x" * 40)],
        ):
            self.assertIn("no successful land run", check(runs), runs)

    def test_a_green_run_among_failed_ones_counts(self):
        self.assertIsNone(check([run(conclusion="failure"), run()]))


class CheckTest(unittest.TestCase):
    def run_check(self, reply, on_main=lambda sha: False):
        trees = {MERGE: "t", HEAD: "t"}
        return ci_landed.check(MERGE, on_main, lambda sha: [BASE, HEAD], trees.__getitem__, lambda commit, limit: reply)

    def test_exit_codes(self):
        self.assertEqual(self.run_check({"workflow_runs": [run()]}), 0)
        self.assertEqual(self.run_check({"workflow_runs": []}), 1)

    def test_a_reply_without_the_list_cannot_be_checked(self):
        self.assertEqual(self.run_check({"message": "denied"}), 2)
        self.assertEqual(self.run_check({"workflow_runs": None}), 2)

    def test_a_full_page_cannot_be_checked(self):
        self.assertEqual(self.run_check({"workflow_runs": [run(conclusion="failure")] * ci_landed.PAGE}), 2)

    def test_a_failing_git_cannot_be_checked(self):
        def broken(sha):
            raise ValueError("git merge-base failed (128)")

        self.assertEqual(self.run_check({"workflow_runs": [run()]}, on_main=broken), 2)


if __name__ == "__main__":
    unittest.main()
