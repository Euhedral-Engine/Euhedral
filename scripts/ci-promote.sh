#!/usr/bin/env bash
# Promotion: the integration branch goes into main, the next integration branch starts there, and the old one is deleted.
# Run by the manual job `promote` of .gitlab-ci.yml, in the job's own clone (full history), from the pipeline of the commit to
# promote. Every check runs before anything is changed; --dry-run stops after the checks and prints the plan. It only runs git
# and the GitLab API: it never deploys anything and never touches a service.
#
#   scripts/ci-promote.sh [--dry-run]
#
# Reads (set by GitLab for the job, or by hand for a dry run):
#   CI_COMMIT_BRANCH, CI_COMMIT_SHA   the integration branch and the commit to promote (its head)
#   CI_API_V4_URL, CI_PROJECT_ID, CI_SERVER_URL, CI_PROJECT_PATH, CI_DEFAULT_BRANCH
#   PROMOTE_TOKEN        a project access token (Maintainer; api, write_repository): a protected, masked CI/CD variable
#   INTEGRATION_BRANCH   the project's CI/CD variable naming the integration branch; this script sets it to the new name
#   NEXT_BRANCH          optional: the new branch's name (default: the old name with its trailing number plus one, or "-2"
#                        appended to a name without one)
#   PROMOTE_CHECKOUT     optional: a working checkout of the project; it is checked (no uncommitted changes, refs/ci/gated),
#                        switched to the new branch, and its copy of the old branch is deleted. Unset in CI: all of that is skipped
#   MIRROR_REMOTE        optional: a remote of the checkout that the old branch is also deleted from by hand. Unset in CI: the
#                        GitLab push mirror (protected branches) is left to delete it
#   PROMOTE_API_LOG      the script's own test: the API calls are written to this file in place of being sent
#
# A run that stopped half way can be started again from the same pipeline: every step that is already done is passed over.
# The steps, in order: main, the new branch (protected), the open merge requests' target, the variable, the checkout (if any),
# then the old branch (GitLab, the mirror remote if any, the checkout's copy if any).
set -euo pipefail

dry=0
case "${1:-}" in
  --dry-run) dry=1 ;;
  "") ;;
  *) echo "usage: scripts/ci-promote.sh [--dry-run]" >&2; exit 2 ;;
esac
fail() { echo "FAIL  $*" >&2; exit 1; }

old="${CI_COMMIT_BRANCH:?run it from a pipeline of the integration branch}"
checkout="${PROMOTE_CHECKOUT:-}"
if [ -n "$checkout" ]; then cd "$checkout" || fail "PROMOTE_CHECKOUT $checkout is not a directory"; fi
# merge commits need an identity; the job's clone has none
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-GitLab promote}" GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-promote@localhost}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}" GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"
sha="${CI_COMMIT_SHA:?}"
main="${CI_DEFAULT_BRANCH:-main}"
: "${CI_API_V4_URL:?}" "${CI_PROJECT_ID:?}" "${CI_SERVER_URL:?}" "${CI_PROJECT_PATH:?}"
: "${PROMOTE_TOKEN:?set the protected CI/CD variable PROMOTE_TOKEN (a project access token: Maintainer; api, write_repository)}"
mirror="${MIRROR_REMOTE:-}"
repo="$CI_SERVER_URL/$CI_PROJECT_PATH.git"

next="${NEXT_BRANCH:-}"
if [ -z "$next" ]; then
  if [[ "$old" =~ ^(.*[^0-9])([0-9]+)$ ]]; then
    next="${BASH_REMATCH[1]}$((10#${BASH_REMATCH[2]} + 1))"
  else
    next="$old-2"
  fi
fi
[ "$next" != "$old" ] && [ "$next" != "$main" ] || fail "the new branch cannot be $next"
git check-ref-format "refs/heads/$next" || fail "$next is not a branch name"
case "${INTEGRATION_BRANCH:-$old}" in
  "$old" | "$next") ;;
  *) fail "the variable INTEGRATION_BRANCH names ${INTEGRATION_BRANCH}, and this pipeline is $old's" ;;
esac

# The token goes to git in the environment and to curl through a file descriptor, never on a command line.
auth=$(printf 'promote:%s' "$PROMOTE_TOKEN" | base64 -w0)
g() {
  GIT_TERMINAL_PROMPT=0 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraHeader GIT_CONFIG_VALUE_0="Authorization: Basic $auth" git "$@"
}
enc() { printf %s "$1" | jq -sRr @uri; }
# api METHOD PATH [curl data arguments]: prints the body; a status that is not 2xx fails unless it is listed in $also.
# PROMOTE_API_LOG (the script's own test) writes the calls to that file, in place of sending them.
api() {
  local m=$1 p=$2 out code nl=$'\n'
  shift 2
  if [ -n "${PROMOTE_API_LOG:-}" ]; then
    echo "$m $p $*" >> "$PROMOTE_API_LOG"
    case "$m $p" in
      "GET /protected_branches/"*) echo '{"push_access_levels":[{"access_level":40}],"merge_access_levels":[{"access_level":40}]}' ;;
      "GET /merge_requests"*) echo '[]' ;;
    esac
    return 0
  fi
  out=$(curl -sS -w '\n%{http_code}' -X "$m" --config <(printf 'header = "PRIVATE-TOKEN: %s"\n' "$PROMOTE_TOKEN") \
    "$CI_API_V4_URL/projects/$CI_PROJECT_ID$p" "$@")
  code=${out##*"$nl"}
  printf '%s\n' "${out%"$nl"*}"
  case "$code" in 2??) return 0 ;; esac
  case " ${also:-} " in *" $code "*) return 0 ;; esac
  echo "FAIL  $m $p: HTTP $code" >&2
  return 1
}
head_of() { g ls-remote "$repo" "refs/heads/$1" | cut -f1; }

# ---- checks: nothing is changed before the line "plan" ----
[ "$(git rev-parse --is-shallow-repository)" = false ] || fail "the clone is shallow: the job needs the full history (GIT_DEPTH: 0)"
if [ -n "$checkout" ]; then
  git diff --quiet && git diff --cached --quiet || fail "the integration checkout has uncommitted changes"
fi
old_head=$(head_of "$old")
main_head=$(head_of "$main")
next_head=$(head_of "$next")
[ -n "$main_head" ] || fail "$main is not on $CI_SERVER_URL"
if [ -n "$old_head" ]; then
  [ "$old_head" = "$sha" ] || fail "$old has moved on to ${old_head:0:8}: play the job of that commit's pipeline"
  if [ -n "$checkout" ]; then
    git merge-base --is-ancestor "$sha" HEAD || fail "the integration checkout is at $(git rev-parse --short HEAD), which does not hold ${sha:0:8}: its gate has not run here"
    if gated=$(git rev-parse -q --verify refs/ci/gated); then
      [ "$gated" = "$sha" ] || fail "the gate's last pass was ${gated:0:8}, not ${sha:0:8}"
    fi
  fi
else
  # A second start after the old branch was deleted: only the clean-up is left.
  [ -n "$next_head" ] || fail "neither $old nor $next is on $CI_SERVER_URL"
fi

g fetch -q "$repo" "$main"
main_sha=$(git rev-parse FETCH_HEAD)
how=""
if git merge-base --is-ancestor "$sha" "$main_sha"; then
  new=$main_sha
  how="already holds ${sha:0:8}"
elif git merge-base --is-ancestor "$main_sha" "$sha"; then
  new=$sha
  how="fast-forward from ${main_sha:0:8}"
else
  # main has commits the integration branch lacks: a merge commit, made without a checkout. A conflict stops here.
  tree=$(git merge-tree --write-tree "$main_sha" "$sha") || fail "$old does not merge into $main without conflicts: merge $main into $old first, through a merge request"
  if [ "$dry" = 1 ]; then
    new=""
  else
    new=$(git commit-tree "$tree" -p "$main_sha" -p "$sha" -m "Merge $old into $main")
  fi
  how="a merge commit on ${main_sha:0:8} (main has commits $old lacks: the next gate run tests and deploys them)"
fi
if [ -n "$next_head" ]; then
  g fetch -q "$repo" "refs/heads/$next"
  git merge-base --is-ancestor "$sha" "$next_head" ||
    fail "$next is already on $CI_SERVER_URL at ${next_head:0:8}, which does not hold the promoted commit: give another name as NEXT_BRANCH"
fi

levels=$(api GET "/protected_branches/$(enc "$old")" 2> /dev/null) || levels=$(api GET "/protected_branches/$(enc "$main")") ||
  fail "the token cannot read the branch protections: it needs the Maintainer role and the api scope"
push_level=$(jq -r '[.push_access_levels[].access_level] | max // 40' <<< "$levels")
merge_level=$(jq -r '[.merge_access_levels[].access_level] | max // 40' <<< "$levels")

on_mirror=""
if [ -n "$mirror" ]; then
  git remote get-url "$mirror" > /dev/null 2>&1 || fail "MIRROR_REMOTE $mirror is not a remote of this clone"
  on_mirror=$(GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes" timeout 60 git ls-remote "$mirror" "refs/heads/$old" | cut -f1) ||
    fail "the mirror ($mirror) cannot be reached from here, so $old could not be deleted there"
fi
moving=$(api GET "/merge_requests?state=opened&target_branch=$(enc "$old")&per_page=100" | jq -r '[.[].iid | tostring] | join(" ")')

echo "plan  $main: $how"
echo "plan  $next: created at $main's new head and protected (push $push_level, merge $merge_level)"
echo "plan  open merge requests moved from $old to $next: ${moving:-none}"
if [ -n "$checkout" ]; then
  echo "plan  INTEGRATION_BRANCH=$next; the checkout $checkout switches to $next"
else
  echo "plan  INTEGRATION_BRANCH=$next; no checkout switch (PROMOTE_CHECKOUT is unset)"
fi
echo "plan  $old deleted: $CI_SERVER_URL${on_mirror:+, the mirror remote ($mirror)}${checkout:+, the checkout}"
[ -n "$mirror" ] || echo "plan  the push mirror is left to delete $old on its side (no MIRROR_REMOTE)"
if [ "$dry" = 1 ]; then
  echo "dry run: nothing was changed"
  exit 0
fi

# ---- the promotion ----
if [ "$main_sha" != "$new" ]; then g push -q "$repo" "$new:refs/heads/$main"; fi
echo "ok    $main is at ${new:0:8}"

# Protected before it exists, so that its first push already goes to the mirror (the mirror sends protected branches only).
# ci.skip: the new branch starts at a commit that is already gated and deployed.
also="409 422" api POST "/protected_branches" --data-urlencode "name=$next" --data "push_access_level=$push_level" \
  --data "merge_access_level=$merge_level" > /dev/null
if [ -z "$next_head" ]; then g push -q -o ci.skip "$repo" "$new:refs/heads/$next"; fi
echo "ok    $next is protected and holds ${new:0:8}"

for iid in $moving; do
  api PUT "/merge_requests/$iid" --data-urlencode "target_branch=$next" > /dev/null
  echo "ok    merge request !$iid now targets $next"
done

if ! also="404" api PUT "/variables/INTEGRATION_BRANCH" --data-urlencode "value=$next" | jq -e '.key' > /dev/null; then
  api POST "/variables" --data "key=INTEGRATION_BRANCH" --data-urlencode "value=$next" > /dev/null
fi
echo "ok    INTEGRATION_BRANCH=$next"

# A second start must not take the checkout back: the gate may have moved it on along the new branch since.
if [ -z "$checkout" ]; then
  echo "ok    no checkout to switch (PROMOTE_CHECKOUT is unset)"
elif [ "$(git branch --show-current)" = "$next" ] && git merge-base --is-ancestor "$new" HEAD; then
  :
elif ! git rev-parse -q --verify "refs/heads/$next" > /dev/null || git merge-base --is-ancestor "refs/heads/$next" "$new"; then
  git checkout -q -B "$next" "$new"
else
  fail "this checkout has a branch $next at $(git rev-parse --short "refs/heads/$next") that is not part of ${new:0:8}: look at it before anything else is done"
fi
if [ -n "$checkout" ]; then echo "ok    the checkout is on $next at $(git rev-parse --short HEAD)"; fi

also="404" api DELETE "/protected_branches/$(enc "$old")" > /dev/null
# With a lease: a merge that reached the old branch after the checks is not deleted with it (the push fails, nothing is lost).
still=$(head_of "$old")
if [ -n "$still" ]; then g push -q --force-with-lease="refs/heads/$old:$sha" "$repo" ":refs/heads/$old"; fi
still=$(head_of "$old")
[ -z "$still" ] || fail "$old is still on $CI_SERVER_URL"
echo "ok    $old deleted on $CI_SERVER_URL"
if [ -n "$on_mirror" ]; then
  GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes" timeout 60 git push -q "$mirror" --delete "$old"
  echo "ok    $old deleted on the mirror ($mirror)"
fi
if [ -n "$checkout" ] && git rev-parse -q --verify "refs/heads/$old" > /dev/null; then
  git merge-base --is-ancestor "refs/heads/$old" "$new" ||
    fail "the checkout's branch $old has commits that are not in $next: it is kept here (everything else is done)"
  git branch -q -D "$old"
  echo "ok    $old deleted here"
fi

# main may have moved on since a first run (other pushes), so a finished promotion is checked against what it made
want_next=$new
[ "$new" != "$main_sha" ] || want_next=$sha
for b in "$main" "$next"; do
  want=$new
  [ "$b" != "$next" ] || want=$want_next
  g fetch -q "$repo" "refs/heads/$b"
  git merge-base --is-ancestor "$want" FETCH_HEAD || fail "$b on $CI_SERVER_URL does not hold ${want:0:8}"
done
echo "PROMOTED $old -> $main at ${new:0:8}; the integration branch is now $next"
