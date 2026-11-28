"""ci-landed proves a commit pushed to an integration branch is the merge of a pull request head that passed its land run."""

import importlib.util
import subprocess
import unittest
from unittest import mock
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
    trees = {MERGE: merge_tree, HEAD: head_tree, BASE: "t0"}
    return ci_landed.verdict(MERGE, on_main, list(parents), trees.__getitem__, lambda commit: runs if commit == HEAD else [])


class LandedTest(unittest.TestCase):
    def test_the_merge_of_a_head_with_a_green_land_run_and_its_tree_is_landed(self):
        self.assertIsNone(check([run()]))

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


class OnMainTest(unittest.TestCase):
    """A commit on main is landed only when its tree is one that a run saw."""

    def on_main(self, runs_by_commit, merge_tree="t1", parents=(BASE, HEAD), trees=None):
        trees = trees or {MERGE: merge_tree, HEAD: "t1", BASE: "t0"}
        return ci_landed.verdict(MERGE, True, list(parents), trees.__getitem__, lambda commit: runs_by_commit.get(commit, []))

    def test_a_merge_made_on_main_with_an_untested_tree_is_refused(self):
        reason = self.on_main({HEAD: [run()], BASE: [run(head_sha=BASE)]}, merge_tree="t9")
        self.assertIn("never tested", reason)

    def test_a_promoted_head_with_a_green_push_run_is_landed(self):
        self.assertIsNone(self.on_main({MERGE: [run(head_sha=MERGE, event="push")]}, merge_tree="t9", parents=(BASE,)))

    def test_a_parent_with_the_same_tree_and_a_green_push_run_is_landed(self):
        self.assertIsNone(self.on_main({HEAD: [run(event="push")]}, merge_tree="t1", parents=(BASE, HEAD)))

    def test_a_parent_with_another_tree_does_not_vouch_for_the_commit(self):
        self.assertIn("never tested", self.on_main({BASE: [run(head_sha=BASE)]}, merge_tree="t9"))

    def test_a_pull_request_or_dispatched_run_does_not_vouch_for_a_commit_on_main(self):
        for event in ("pull_request", "workflow_dispatch"):
            runs = {MERGE: [run(head_sha=MERGE, event=event)]}
            self.assertIn("never tested", self.on_main(runs, parents=(BASE,)), event)

    def test_a_failed_run_does_not_vouch_for_the_commit(self):
        self.assertIn("never tested", self.on_main({MERGE: [run(head_sha=MERGE, event="push", conclusion="failure")]}, parents=(BASE,)))

    def test_a_commit_that_is_the_merge_of_a_tested_head_is_landed(self):
        self.assertIsNone(self.on_main({HEAD: [run()]}))


class IsOnMainTest(unittest.TestCase):
    def exit_code(self, code):
        with mock.patch.object(ci_landed.subprocess, "run", return_value=subprocess.CompletedProcess([], code)):
            return ci_landed.is_on_main(MERGE)

    def test_exit_0_is_on_main_and_1_is_not(self):
        self.assertTrue(self.exit_code(0))
        self.assertFalse(self.exit_code(1))

    def test_exit_128_is_an_error_not_a_no(self):
        with self.assertRaises(ValueError):
            self.exit_code(128)

    def test_exit_128_makes_the_check_exit_2(self):
        def is_on_main(sha):
            with mock.patch.object(ci_landed.subprocess, "run", return_value=subprocess.CompletedProcess([], 128)):
                return ci_landed.is_on_main(sha)

        trees = {MERGE: "t", HEAD: "t"}
        code = ci_landed.check(MERGE, is_on_main, lambda sha: [BASE, HEAD], trees.__getitem__, lambda commit, limit: {"workflow_runs": []})
        self.assertEqual(code, 2)


class TestedTreeTest(unittest.TestCase):
    """The mode that asks whether the head of main is a tested tree: 0 tested, 1 untested, 2 could not be checked."""

    def exit_code(self, runs_by_commit, merge_tree="t9", reply=None):
        trees = {MERGE: merge_tree, HEAD: "t1", BASE: "t0"}
        pages = lambda commit, limit: reply or {"workflow_runs": runs_by_commit.get(commit, [])}
        return ci_landed.check(MERGE, lambda sha: True, lambda sha: [BASE, HEAD], trees.__getitem__, pages, "notice")

    def test_an_untested_merge_made_on_main_is_1(self):
        self.assertEqual(self.exit_code({HEAD: [run()]}), 1)

    def test_a_head_with_a_green_push_run_is_0(self):
        self.assertEqual(self.exit_code({MERGE: [run(head_sha=MERGE, event="push")]}), 0)

    def test_the_merge_of_a_green_head_with_its_tree_is_0(self):
        self.assertEqual(self.exit_code({HEAD: [run()]}, merge_tree="t1"), 0)

    def test_an_unreadable_reply_is_2(self):
        self.assertEqual(self.exit_code({}, reply={"message": "denied"}), 2)


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
