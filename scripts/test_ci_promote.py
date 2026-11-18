"""Tests of scripts/ci-promote.sh against local bare repositories.

The GitLab API is not called: PROMOTE_API_LOG makes the script write its calls to a file. Run:
    python3 -m unittest scripts/test_ci_promote.py
It needs git 2.38 or newer, jq and bash.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("ci-promote.sh")
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@localhost",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@localhost",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}
MUTATING = ("POST ", "PUT ", "DELETE ")


def git(cwd, *args, check=True):
    env = {**os.environ, **GIT_ENV}
    done = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and done.returncode != 0:
        raise AssertionError(f"git {args} failed: {done.stderr}")
    return done.stdout.strip()


class PromoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.srv = self.root / "srv"
        self.srv.mkdir()
        self.bare = self.srv / "proj.git"
        git(self.root, "init", "-q", "--bare", "-b", "main", str(self.bare))
        git(self.bare, "config", "receive.advertisePushOptions", "true")  # the script pushes with -o ci.skip
        self.seed = self.root / "seed"
        git(self.root, "clone", "-q", str(self.bare), str(self.seed))
        git(self.seed, "checkout", "-q", "-b", "main")
        self.commit(self.seed, "base.txt", "base\n")
        git(self.seed, "push", "-q", "origin", "main")
        self.log = self.root / "api.log"

    def commit(self, repo, name, text):
        (repo / name).write_text(text)
        git(repo, "add", name)
        git(repo, "commit", "-q", "-m", f"edit {name}")
        return git(repo, "rev-parse", "HEAD")

    def start_integration(self, branch="rust/next", file="work.txt", text="work\n"):
        git(self.seed, "checkout", "-q", "-B", branch, "main")
        sha = self.commit(self.seed, file, text)
        git(self.seed, "push", "-q", "origin", branch)
        return sha

    def advance_main(self, file, text):
        git(self.seed, "fetch", "-q", "origin", "main")
        git(self.seed, "checkout", "-q", "-B", "main", "FETCH_HEAD")
        self.commit(self.seed, file, text)
        git(self.seed, "push", "-q", "origin", "main")

    def head(self, branch):
        return git(self.bare, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False)

    def heads(self):
        return git(self.bare, "for-each-ref", "refs/heads")

    def run_promote(self, old, sha, *args, extra=None, cwd=None):
        """Run the script in a fresh full clone, as the job does."""
        clone = self.root / f"job{len(list(self.root.glob('job*')))}"
        git(self.root, "clone", "-q", str(self.bare), str(clone))
        env = {
            **os.environ,
            **GIT_ENV,
            "CI_COMMIT_BRANCH": old,
            "CI_COMMIT_SHA": sha,
            "CI_API_V4_URL": "http://gitlab.invalid/api/v4",
            "CI_PROJECT_ID": "6",
            "CI_SERVER_URL": f"file://{self.srv}",
            "CI_PROJECT_PATH": "proj",
            "CI_DEFAULT_BRANCH": "main",
            "PROMOTE_TOKEN": "not-a-token",
            "PROMOTE_API_LOG": str(self.log),
            "INTEGRATION_BRANCH": old,
        }
        for key in ("NEXT_BRANCH", "PROMOTE_CHECKOUT", "MIRROR_REMOTE"):
            env.pop(key, None)
        env.update(extra or {})
        done = subprocess.run(
            ["bash", str(SCRIPT), *args], cwd=cwd or clone, env=env, capture_output=True, text=True, timeout=120
        )
        self.clone = clone
        return done

    def api_calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def mutating_calls(self):
        return [c for c in self.api_calls() if c.startswith(MUTATING)]

    def test_fast_forward(self):
        sha = self.start_integration()
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.head("main"), sha)
        self.assertEqual(self.head("rust/next-2"), sha)
        self.assertEqual(self.head("rust/next"), "")
        calls = "\n".join(self.api_calls())
        self.assertIn("POST /protected_branches --data-urlencode name=rust/next-2", calls)
        self.assertIn("PUT /variables/INTEGRATION_BRANCH --data-urlencode value=rust/next-2", calls)
        self.assertIn("DELETE /protected_branches/rust%2Fnext", calls)
        self.assertIn("PROMOTED rust/next -> main", done.stdout)

    def test_merge_commit_when_main_has_other_commits(self):
        sha = self.start_integration()
        self.advance_main("other.txt", "other\n")
        main_before = self.head("main")
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        merged = self.head("main")
        self.assertEqual(git(self.bare, "rev-list", "--parents", "-n1", merged).split()[1:], [main_before, sha])
        files = git(self.bare, "ls-tree", "--name-only", merged).split()
        self.assertEqual(sorted(files), ["base.txt", "other.txt", "work.txt"])
        self.assertEqual(self.head("rust/next-2"), merged)
        self.assertEqual(self.head("rust/next"), "")

    def test_conflict_stops_before_any_change(self):
        sha = self.start_integration(file="base.txt", text="integration\n")
        self.advance_main("base.txt", "main\n")
        before = self.heads()
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("without conflicts", done.stderr)
        self.assertEqual(self.heads(), before)
        self.assertEqual(self.mutating_calls(), [])

    def test_moved_integration_branch_stops(self):
        sha = self.start_integration()
        self.commit(self.seed, "later.txt", "later\n")
        git(self.seed, "push", "-q", "origin", "rust/next")
        before = self.heads()
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("has moved on", done.stderr)
        self.assertEqual(self.heads(), before)

    def test_dry_run_changes_nothing(self):
        sha = self.start_integration()
        self.advance_main("other.txt", "other\n")
        before = self.heads()
        done = self.run_promote("rust/next", sha, "--dry-run")
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertIn("dry run: nothing was changed", done.stdout)
        self.assertIn("plan  main: a merge commit", done.stdout)
        self.assertIn("no checkout switch", done.stdout)
        self.assertEqual(self.heads(), before)
        self.assertEqual(self.mutating_calls(), [])

    def test_rerun_after_main_and_next_were_done(self):
        sha = self.start_integration()
        git(self.seed, "push", "-q", "origin", f"{sha}:refs/heads/main", f"{sha}:refs/heads/rust/next-2")
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertIn("already holds", done.stdout)
        self.assertEqual(self.head("main"), sha)
        self.assertEqual(self.head("rust/next-2"), sha)
        self.assertEqual(self.head("rust/next"), "")

    def test_rerun_after_the_old_branch_was_deleted(self):
        sha = self.start_integration()
        self.assertEqual(self.run_promote("rust/next", sha).returncode, 0)
        self.advance_main("later.txt", "later\n")  # main moves on between the two runs
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.head("rust/next"), "")
        self.assertEqual(self.head("rust/next-2"), sha)

    def test_rerun_refuses_a_next_branch_that_lacks_the_commit(self):
        sha = self.start_integration()
        git(self.seed, "push", "-q", "origin", "main:refs/heads/rust/next-2")
        before = self.heads()
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("NEXT_BRANCH", done.stderr)
        self.assertEqual(self.heads(), before)

    def plan_next(self, old, **extra):
        sha = self.start_integration(old)
        done = self.run_promote(old, sha, "--dry-run", extra=extra)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        return next(line for line in done.stdout.splitlines() if "created at" in line).split()[1].rstrip(":")

    def test_default_naming(self):
        self.assertEqual(self.plan_next("rust/next"), "rust/next-2")

    def test_name_ending_in_a_number(self):
        self.assertEqual(self.plan_next("integration-9"), "integration-10")

    def test_next_branch_override(self):
        self.assertEqual(self.plan_next("rust/next", NEXT_BRANCH="rust/other"), "rust/other")

    def test_checkout_and_mirror(self):
        sha = self.start_integration()
        mirror = self.root / "mirror.git"
        git(self.root, "clone", "-q", "--bare", str(self.bare), str(mirror))
        checkout = self.root / "desk"
        git(self.root, "clone", "-q", "-b", "rust/next", str(self.bare), str(checkout))
        git(checkout, "remote", "add", "mirror", str(mirror))
        done = self.run_promote(
            "rust/next", sha, extra={"PROMOTE_CHECKOUT": str(checkout), "MIRROR_REMOTE": "mirror"}, cwd=self.root
        )
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(git(checkout, "branch", "--show-current"), "rust/next-2")
        self.assertEqual(git(checkout, "rev-parse", "-q", "--verify", "refs/heads/rust/next", check=False), "")
        self.assertEqual(git(mirror, "rev-parse", "-q", "--verify", "refs/heads/rust/next", check=False), "")

    def test_checkout_with_uncommitted_changes_stops(self):
        sha = self.start_integration()
        checkout = self.root / "desk"
        git(self.root, "clone", "-q", "-b", "rust/next", str(self.bare), str(checkout))
        (checkout / "work.txt").write_text("dirty\n")
        before = self.heads()
        done = self.run_promote("rust/next", sha, extra={"PROMOTE_CHECKOUT": str(checkout)}, cwd=self.root)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("uncommitted", done.stderr)
        self.assertEqual(self.heads(), before)


if __name__ == "__main__":
    unittest.main()
