#!/usr/bin/env python3
"""Check that a commit pushed to an integration branch is a landed pull request.

A pull request is tested once, by the run its label `land` starts, on the head that is then merged with the target already in
it. The merge commit therefore has the head's tree, and this check proves it: the commit is already on main (the branch
`rollover promote` starts at main), or it is a merge commit whose second parent had a successful pull request run of ci.yaml
and has the same tree. A skipped run, another label's run (its `land-guard` fails) and a run of another head do not count.

usage: ci-landed.py SHA   (environment: GITEA_TOKEN, API_URL such as http://host/api/v1, REPOSITORY owner/name)
exit 0: landed, 1: not landed (one line says why), 2: the check could not be made
"""

import json
import os
import subprocess
import sys
import urllib.request

WORKFLOW = "ci.yaml"


def land_run_succeeded(runs, sha):
    """True when `runs` (the `workflow_runs` of the Actions API) has a successful pull request run of ci.yaml on `sha`."""
    return any(
        run.get("head_sha") == sha
        and run.get("path", "").split("@")[0] == WORKFLOW
        and run.get("event") == "pull_request"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        for run in runs
    )


def verdict(sha, on_main, parents, trees, runs_of):
    """The reason `sha` is not a landed pull request, or None when it is.

    `parents` are the parent commits of `sha`, `trees(commit)` is a commit's tree and `runs_of(commit)` its Actions runs."""
    if on_main:
        return None
    if len(parents) != 2:
        return f"{sha} is not a merge commit (it has {len(parents)} parents) and is not on main"
    head = parents[1]
    if trees(sha) != trees(head):
        return f"{sha} does not have the tree of the pull request head {head}: the target moved after the land run"
    if not land_run_succeeded(runs_of(head), head):
        return f"the pull request head {head} has no successful land run of {WORKFLOW}"
    return None


PAGE = 50


def check(sha, is_on_main, parents_of, tree_of, runs_page):
    """Exit code for `sha`: 0 landed, 1 not landed, 2 the check could not be made (a ValueError from a collaborator, or an
    Actions reply that has no `workflow_runs` or is a full page, which may hide the run)."""
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
        print(f"::error::the check could not be made: {error}", file=sys.stderr)
        return 2
    if reason:
        print(f"::error::{reason}", file=sys.stderr)
        return 1
    print(f"{sha} is on main or the merge of a pull request head with a green land run")
    return 0


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def is_on_main(sha):
    """Whether `sha` is an ancestor of origin/main; git's exit 128 (an unknown commit or ref) is an error, not a no."""
    code = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], capture_output=True).returncode
    if code not in (0, 1):
        raise ValueError(f"git merge-base failed ({code})")
    return code == 0


def main(argv):
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        token, api, repository = (os.environ[name] for name in ("GITEA_TOKEN", "API_URL", "REPOSITORY"))
    except KeyError as error:
        print(f"::error::the check could not be made: {error} is not set", file=sys.stderr)
        return 2

    def runs_page(commit, limit):
        request = urllib.request.Request(
            f"{api}/repos/{repository}/actions/runs?head_sha={commit}&limit={limit}", headers={"Authorization": f"token {token}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    return check(
        argv[0],
        is_on_main,
        lambda sha: git("rev-list", "--parents", "-n", "1", sha).split()[1:],
        lambda commit: git("rev-parse", f"{commit}^{{tree}}"),
        runs_page,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
