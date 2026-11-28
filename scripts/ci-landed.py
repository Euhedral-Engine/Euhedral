#!/usr/bin/env python3
"""Check that a commit pushed to an integration branch is a landed pull request.

A pull request is tested once, by the run its label `land` starts, on the head that is then merged with the target already in
it. The merge commit therefore has the head's tree, and this check proves it: the commit is a merge commit whose second parent
had a successful pull request run of ci.yaml and has the same tree. A commit on main (the branch `rollover promote` starts at
main) passes only when its tree is a tested one: the commit passes that rule, or it or a parent with its tree had a successful
push run of ci.yaml (the promoted integration head). A pull request run does not vouch for a commit on main: its diff against
main may be empty, so it may have run no group. A merge commit that promotion made on main has a tree no run saw and fails;
landing a pull request tests it, because its CI runs every group while main's head is untested. A skipped run, a dispatched
run, another label's run (its `land-guard` fails) and a run of another head do not count.

usage: ci-landed.py SHA   (environment: GITEA_TOKEN, API_URL such as http://host/api/v1, REPOSITORY owner/name)
       ci-landed.py --tested-tree SHA   the same rule for a commit of main (the head of main): is its tree a tested one?
exit 0: landed (tested), 1: not landed (untested; one line says why), 2: the check could not be made
"""

import json
import os
import subprocess
import sys
import urllib.request

WORKFLOW = "ci.yaml"


def green_run_on(runs, sha, events=("pull_request",)):
    """True when `runs` (the `workflow_runs` of the Actions API) has a completed successful run of ci.yaml on `sha` whose event
    is one of `events` (a dispatched run is never in the default or in either use)."""
    return any(
        run.get("head_sha") == sha
        and run.get("path", "").split("@")[0] == WORKFLOW
        and run.get("event") in events
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        for run in runs
    )


def verdict(sha, on_main, parents, trees, runs_of):
    """The reason `sha` is not a tested tree, or None when it is: the merge of a pull request head with a green land run and
    that head's tree, or (when `on_main`) a commit that, or a same-tree parent that, had a green push run of ci.yaml (the
    promoted integration head). A pull request run does not vouch for a commit on main: its diff against main may be empty,
    so it may have run no group.

    `parents` are the parent commits of `sha`, `trees(commit)` is a commit's tree and `runs_of(commit)` its Actions runs."""
    reason = merge_verdict(sha, parents, trees, runs_of)
    if reason is None or not on_main:
        return reason
    tested = (sha, *(parent for parent in parents if trees(parent) == trees(sha)))
    if any(green_run_on(runs_of(commit), commit, ("push",)) for commit in tested):
        return None
    return f"{sha} is on main but its tree was never tested: land a pull request; its CI runs every group because main's head is untested"


def merge_verdict(sha, parents, trees, runs_of):
    """The reason `sha` is not the merge of a pull request head with a green land run and its tree, or None."""
    if len(parents) != 2:
        return f"{sha} is not a merge commit (it has {len(parents)} parents) and is not on main"
    head = parents[1]
    if trees(sha) != trees(head):
        return f"{sha} does not have the tree of the pull request head {head}: the target moved after the land run"
    if not green_run_on(runs_of(head), head):
        return f"the pull request head {head} has no successful land run of {WORKFLOW}"
    return None


PAGE = 50


def unmade(error, level):
    """The annotation for a check that could not be made: an error, or a warning for the head-of-main question."""
    if level == "error":
        return f"::error::the check could not be made: {error}"
    return f"::warning::the check could not be made: {error}; every group runs"


def check(sha, is_on_main, parents_of, tree_of, runs_page, level="error"):
    """Exit code for `sha`: 0 landed, 1 not landed, 2 the check could not be made (a ValueError from a collaborator, or an
    Actions reply that has no `workflow_runs` or is a full page, which may hide the run). `level` is the annotation of a
    refusal; any level but "error" is the head-of-main question, whose exit 2 is a `::warning::` that every group runs."""
    def runs_of(commit):
        reply = runs_page(commit, PAGE)
        if not isinstance(reply, dict) or not isinstance(reply.get("workflow_runs"), list):
            raise ValueError(f"the Actions reply for {commit} has no workflow_runs")
        if len(reply["workflow_runs"]) >= PAGE:
            raise ValueError(f"{commit} has {PAGE} runs or more: the list may be cut off")
        return reply["workflow_runs"]

    try:
        reason = verdict(sha, is_on_main(sha), parents_of(sha), tree_of, runs_of)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(unmade(error, level), file=sys.stderr)
        return 2
    if reason:
        print(f"::{level}::{reason}", file=sys.stderr)
        return 1
    print(f"{sha} is a tested tree")
    return 0


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def is_on_main(sha):
    """Whether `sha` is an ancestor of origin/main; any exit but 0 and 1 (128: an unknown commit or ref) is an error, not a no."""
    code = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], capture_output=True).returncode
    if code not in (0, 1):
        raise ValueError(f"git merge-base failed ({code})")
    return code == 0


def main(argv):
    main_head = argv[:1] == ["--tested-tree"]
    if len(argv) != (2 if main_head else 1):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        token, api, repository = (os.environ[name] for name in ("GITEA_TOKEN", "API_URL", "REPOSITORY"))
    except KeyError as error:
        print(unmade(f"{error} is not set", "notice" if argv[:1] == ["--tested-tree"] else "error"), file=sys.stderr)
        return 2

    def runs_page(commit, limit):
        request = urllib.request.Request(
            f"{api}/repos/{repository}/actions/runs?head_sha={commit}&limit={limit}", headers={"Authorization": f"token {token}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    return check(
        argv[-1],
        (lambda sha: True) if main_head else is_on_main,
        lambda sha: git("rev-list", "--parents", "-n", "1", sha).split()[1:],
        lambda commit: git("rev-parse", f"{commit}^{{tree}}"),
        runs_page,
        "notice" if main_head else "error",
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
