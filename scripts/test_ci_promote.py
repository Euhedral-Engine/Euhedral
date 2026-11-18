"""Tests of scripts/ci-promote.sh against local bare repositories.

The GitLab API is not called: PROMOTE_API_LOG makes the script write its calls to a file. Run:
    python3 -m unittest scripts/test_ci_promote.py
It needs git 2.38 or newer, jq and bash.
"""

import json
import os
import stat
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
TOKEN = "tok-SECRET-0123456789"
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
            "PROMOTE_TOKEN": TOKEN,
            "PROMOTE_API_LOG": str(self.log),
            "INTEGRATION_BRANCH": old,
        }
        for key in ("NEXT_BRANCH", "PROMOTE_CHECKOUT", "MIRROR_URL"):
            env.pop(key, None)
        for key, value in (extra or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
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
            "rust/next", sha, extra={"PROMOTE_CHECKOUT": str(checkout), "MIRROR_URL": f"file://{mirror}"}, cwd=self.root
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

    # -- helpers for the cases below --
    def state(self):
        path = Path(str(self.log) + ".state")
        path.mkdir(exist_ok=True)
        return path

    def seed_mrs(self, *mrs):
        (self.state() / "mrs.json").write_text(json.dumps([{"iid": i, "target_branch": t} for i, t in mrs]))

    def mr_targets(self):
        return {m["iid"]: m["target_branch"] for m in json.loads((self.state() / "mrs.json").read_text())}

    def hook(self, name, body):
        path = self.bare / "hooks" / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def reject_pushes_to(self, ref):
        """A pre-receive hook that refuses the ref while the file reject-<ref> exists in the bare repository."""
        self.hook("pre-receive", 'while read o n ref; do [ -f "./reject/$(echo $ref | tr / _)" ] && exit 1; done; exit 0\n')
        (self.bare / "reject").mkdir(exist_ok=True)
        marker = self.bare / "reject" / ref.replace("/", "_")
        marker.touch()
        return marker

    def make_upstream(self):
        self.up = self.root / "up.git"
        git(self.root, "clone", "-q", "--bare", str(self.bare), str(self.up))

    def advance(self, target, file, text):
        """A commit on main of the given bare repository (GitLab's or the upstream one)."""
        git(self.seed, "fetch", "-q", str(target), "main")
        git(self.seed, "checkout", "-q", "-B", "main", "FETCH_HEAD")
        self.commit(self.seed, file, text)
        git(self.seed, "push", "-q", str(target), "main")
        return git(self.seed, "rev-parse", "HEAD")

    def author_of(self, rev):
        return git(self.bare, "log", "-1", "--format=%an <%ae>", rev)

    def is_ancestor(self, a, b):
        return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=self.bare).returncode == 0

    # -- the cases --
    def test_merge_requests_are_moved_page_by_page_and_late_ones_too(self):
        sha = self.start_integration()
        self.seed_mrs(*[(i, "rust/next") for i in range(1, 6)], (7, "other"))
        done = self.run_promote("rust/next", sha, extra={"PROMOTE_MR_PAGE": "2", "PROMOTE_API_FAKE_LATE_MR": "99"})
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        targets = self.mr_targets()
        for iid in (1, 2, 3, 4, 5, 99):
            self.assertEqual(targets[iid], "rust/next-2", iid)
        self.assertEqual(targets[7], "other")
        self.assertIn("moved from rust/next to rust/next-2: 1 2 3 4 5", done.stdout)

    def test_a_commit_reaching_the_old_branch_during_the_run_is_not_deleted_and_a_rerun_finishes(self):
        sha = self.start_integration()
        (self.bare / "late").touch()
        self.hook(
            "post-receive",
            'while read o n ref; do if [ "$ref" = refs/heads/rust/next-2 ] && [ -f ./late ]; then rm ./late; '
            'h=$(git rev-parse refs/heads/rust/next); c=$(git commit-tree "$h^{tree}" -p "$h" -m late); '
            'git update-ref refs/heads/rust/next "$c"; fi; done\n',
        )
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("gained commits", done.stderr)
        late = self.head("rust/next")
        self.assertNotEqual(late, sha)
        self.assertEqual(self.head("main"), sha)
        # the old branch is not finished with until the new one holds what it gained
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("has moved on", done.stderr)
        git(self.seed, "fetch", "-q", "origin")
        git(self.seed, "push", "-q", "origin", f"{late}:refs/heads/rust/next-2")
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.head("rust/next"), "")
        self.assertEqual(self.head("rust/next-2"), late)

    def test_the_old_branch_is_frozen_first(self):
        sha = self.start_integration()
        self.reject_pushes_to("refs/heads/main")
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        frozen = json.loads((self.state() / "prot" / "rust%2Fnext").read_text())
        self.assertEqual(frozen["push_access_levels"][0]["access_level"], 0)
        self.assertEqual(frozen["merge_access_levels"][0]["access_level"], 0)

    def test_rejected_main_push_changes_nothing_else_and_a_rerun_finishes(self):
        sha = self.start_integration()
        base = self.head("main")
        marker = self.reject_pushes_to("refs/heads/main")
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual((self.head("main"), self.head("rust/next"), self.head("rust/next-2")), (base, sha, ""))
        marker.unlink()
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual((self.head("main"), self.head("rust/next-2"), self.head("rust/next")), (sha, sha, ""))
        # the new branch took the old one's rules, not the frozen ones
        made = json.loads((self.state() / "prot" / "rust%2Fnext-2").read_text())
        self.assertEqual(made["push_access_levels"][0]["access_level"], 40)

    def test_rerun_of_a_half_done_merge_commit_run(self):
        sha = self.start_integration()
        self.advance_main("other.txt", "other\n")
        marker = self.reject_pushes_to("refs/heads/rust/next-2")
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        merged = self.head("main")
        self.assertEqual(len(git(self.bare, "rev-list", "--parents", "-n1", merged).split()), 3)
        marker.unlink()
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.head("main"), merged)  # no second merge commit
        self.assertEqual(self.head("rust/next-2"), merged)
        self.assertEqual(self.head("rust/next"), "")

    def test_existing_next_branch_without_mains_commits_is_refused_in_the_checks(self):
        sha = self.start_integration()
        self.advance_main("other.txt", "other\n")
        git(self.seed, "push", "-q", "origin", f"{sha}:refs/heads/rust/next-2")
        before = self.heads()
        done = self.run_promote("rust/next", sha)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("not the commits main has", done.stderr)
        self.assertEqual(self.heads(), before)
        self.assertEqual(self.mutating_calls(), [])

    def test_upstream_main_ahead_is_the_base(self):
        sha = self.start_integration()
        self.make_upstream()
        up_head = self.advance(self.up, "upstream.txt", "upstream\n")
        done = self.run_promote("rust/next", sha, extra={"MIRROR_URL": f"file://{self.up}"})
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        merged = self.head("main")
        self.assertEqual(git(self.bare, "rev-list", "--parents", "-n1", merged).split()[1:], [up_head, sha])
        self.assertTrue(self.is_ancestor(up_head, self.head("rust/next-2")))

    def test_upstream_main_behind_keeps_gitlabs_main_as_the_base(self):
        sha = self.start_integration()
        self.make_upstream()
        gl_head = self.advance(self.bare, "gitlab.txt", "gitlab\n")
        done = self.run_promote("rust/next", sha, extra={"MIRROR_URL": f"file://{self.up}"})
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(git(self.bare, "rev-list", "--parents", "-n1", self.head("main")).split()[1:], [gl_head, sha])

    def test_diverged_upstream_main_stops_in_the_checks(self):
        sha = self.start_integration()
        self.make_upstream()
        self.advance(self.up, "upstream.txt", "upstream\n")
        self.advance(self.bare, "gitlab.txt", "gitlab\n")
        before = self.heads()
        done = self.run_promote("rust/next", sha, extra={"MIRROR_URL": f"file://{self.up}"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("have diverged", done.stderr)
        self.assertEqual(self.heads(), before)
        self.assertEqual(self.mutating_calls(), [])

    def test_integration_branch_variable_naming_another_branch_stops(self):
        sha = self.start_integration()
        before = self.heads()
        done = self.run_promote("rust/next", sha, extra={"INTEGRATION_BRANCH": "elsewhere"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("names elsewhere", done.stderr)
        self.assertEqual(self.heads(), before)

    def test_dry_run_variable(self):
        sha = self.start_integration()
        before = self.heads()
        done = self.run_promote("rust/next", sha, extra={"PROMOTE_DRY_RUN": "1"})
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertIn("dry run", done.stdout)
        self.assertEqual(self.heads(), before)
        self.assertEqual(self.mutating_calls(), [])

    def test_the_token_never_appears_in_any_output(self):
        sha = self.start_integration()
        self.advance_main("other.txt", "other\n")
        marker = self.reject_pushes_to("refs/heads/rust/next-2")
        failed = self.run_promote("rust/next", sha)
        marker.unlink()
        done = self.run_promote("rust/next", sha)
        self.assertEqual(done.returncode, 0)
        seen = failed.stdout + failed.stderr + done.stdout + done.stderr + self.log.read_text()
        self.assertNotIn(TOKEN, seen)
        self.assertNotIn("SECRET", seen)

    def test_a_real_server_url_is_refused_in_test_mode(self):
        sha = self.start_integration()
        done = self.run_promote("rust/next", sha, extra={"CI_SERVER_URL": "https://gitlab.example"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("file:// URL", done.stderr)

    def test_a_configured_identity_is_kept_and_a_missing_one_is_supplied(self):
        sha = self.start_integration()
        self.advance_main("other.txt", "other\n")
        config = self.root / "gitconfig"
        config.write_text("[user]\n\tname = Configured\n\temail = configured@localhost\n")
        names = {key: None for key in GIT_ENV if key.startswith("GIT_AUTHOR") or key.startswith("GIT_COMMITTER")}
        done = self.run_promote("rust/next", sha, extra={**names, "GIT_CONFIG_GLOBAL": str(config)})
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.author_of(self.head("main")), "Configured <configured@localhost>")

    def test_a_missing_identity_is_supplied(self):
        sha = self.start_integration()
        self.advance_main("other.txt", "other\n")
        names = {key: None for key in GIT_ENV if key.startswith("GIT_AUTHOR") or key.startswith("GIT_COMMITTER")}
        done = self.run_promote("rust/next", sha, extra=names)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.author_of(self.head("main")), "GitLab promote <promote@localhost>")

    def make_mirror(self):
        self.mirror = self.root / "mirror.git"
        git(self.root, "clone", "-q", "--bare", str(self.bare), str(self.mirror))
        return {"MIRROR_URL": f"file://{self.mirror}"}

    def mirror_head(self, branch):
        return git(self.mirror, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False)

    def test_the_old_branch_is_deleted_on_the_mirror(self):
        sha = self.start_integration()
        extra = self.make_mirror()
        self.assertEqual(self.mirror_head("rust/next"), sha)
        done = self.run_promote("rust/next", sha, extra=extra)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertIn("deleted on the mirror", done.stdout)
        self.assertEqual(self.mirror_head("rust/next"), "")
        self.assertEqual(self.head("rust/next"), "")

    def test_a_mirror_branch_that_moved_after_the_checks_is_left_alone_with_a_message(self):
        sha = self.start_integration()
        extra = self.make_mirror()
        late = self.commit(self.seed, "elsewhere.txt", "elsewhere\n")
        git(self.seed, "push", "-q", str(self.mirror), f"{late}:refs/heads/rust/next", "--force")
        # a hook on GitLab's side of the run: when the old branch is deleted there, someone pushes to the mirror
        (self.bare / "hooks").mkdir(exist_ok=True)
        self.hook(
            "post-receive",
            'while read o n ref; do if [ "$ref" = refs/heads/rust/next ] && [ "$n" = 0000000000000000000000000000000000000000 ]; then '
            f'c=$(git --git-dir={self.mirror} commit-tree "{late}^{{tree}}" -p {late} -m later); '
            f'git --git-dir={self.mirror} update-ref refs/heads/rust/next "$c"; fi; done\n',
        )
        done = self.run_promote("rust/next", sha, extra=extra)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertIn("was left there", done.stdout)
        self.assertNotEqual(self.mirror_head("rust/next"), "")
        self.assertEqual(self.head("rust/next"), "")

    def test_an_unreachable_mirror_fails_in_the_checks(self):
        sha = self.start_integration()
        before = self.heads()
        done = self.run_promote("rust/next", sha, extra={"MIRROR_URL": f"file://{self.root}/nowhere.git"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("mirror cannot be reached", done.stderr)
        self.assertEqual(self.heads(), before)
        self.assertEqual(self.mutating_calls(), [])

    def test_a_rerun_after_the_mirror_delete_finishes(self):
        sha = self.start_integration()
        extra = self.make_mirror()
        self.assertEqual(self.run_promote("rust/next", sha, extra=extra).returncode, 0)
        done = self.run_promote("rust/next", sha, extra=extra)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.mirror_head("rust/next"), "")
        self.assertEqual(self.head("rust/next-2"), sha)

    def test_a_rerun_after_a_stop_before_the_mirror_delete_finishes_it(self):
        sha = self.start_integration()
        extra = self.make_mirror()
        self.assertEqual(self.run_promote("rust/next", sha, extra=extra).returncode, 0)
        git(self.seed, "push", "-q", str(self.mirror), f"{sha}:refs/heads/rust/next")  # as if the delete never happened
        done = self.run_promote("rust/next", sha, extra=extra)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertEqual(self.mirror_head("rust/next"), "")


if __name__ == "__main__":
    unittest.main()
