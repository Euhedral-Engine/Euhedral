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
#   UPSTREAM_MAIN_URL    optional: a repository read without credentials whose main is also written to by others (GitHub).
#                        The new main is built on the newer of the two mains, so the mirror can fast-forward the upstream one;
#                        mains that have diverged stop the promotion in the checks
#   PROMOTE_DRY_RUN      1: the same as --dry-run
#   PROMOTE_CHECKOUT     optional: a working checkout of the project; it is checked (no uncommitted changes, refs/ci/gated),
#                        switched to the new branch, and its copy of the old branch is deleted. Unset in CI: all of that is skipped
#   MIRROR_REMOTE        optional: a remote of the checkout that the old branch is also deleted from by hand. Unset in CI
#   PROMOTE_API_LOG      the script's own test: the API calls are written to this file in place of being sent (git still pushes,
#                        so only a file:// server is accepted)
#
# The GitHub push mirror sends protected branches only and keeps refs that diverged: it never deletes the old branch there. That
# branch stays on GitHub and must be deleted there by hand.
#
# The first change, after the checks, freezes the old branch (push and merge: No one), so nothing can reach it during the run. A
# run that stopped half way can be started again (retry the job): every step that is already done is passed over. The steps, in
# order: freeze the old branch, main, the new branch (protected), the open merge requests' target, the variable, the checkout (if
# any), then the old branch (GitLab, the mirror remote if any, the checkout's copy if any).
set -euo pipefail

dry=0
[ "${PROMOTE_DRY_RUN:-}" != 1 ] || dry=1
case "${1:-}" in
  --dry-run) dry=1 ;;
  "") ;;
  *) echo "usage: scripts/ci-promote.sh [--dry-run]" >&2; exit 2 ;;
esac
fail() { echo "FAIL  $*" >&2; exit 1; }

old="${CI_COMMIT_BRANCH:?run it from a pipeline of the integration branch}"
checkout="${PROMOTE_CHECKOUT:-}"
if [ -n "$checkout" ]; then cd "$checkout" || fail "PROMOTE_CHECKOUT $checkout is not a directory"; fi
# merge commits need an identity: the job's clone has none, a configured one is kept
[ -n "${GIT_COMMITTER_NAME:-}${GIT_AUTHOR_NAME:-}" ] || git config user.name > /dev/null || export GIT_COMMITTER_NAME="GitLab promote" GIT_AUTHOR_NAME="GitLab promote"
[ -n "${GIT_COMMITTER_EMAIL:-}${GIT_AUTHOR_EMAIL:-}" ] || git config user.email > /dev/null || export GIT_COMMITTER_EMAIL="promote@localhost" GIT_AUTHOR_EMAIL="promote@localhost"
sha="${CI_COMMIT_SHA:?}"
main="${CI_DEFAULT_BRANCH:-main}"
: "${CI_API_V4_URL:?}" "${CI_PROJECT_ID:?}" "${CI_SERVER_URL:?}" "${CI_PROJECT_PATH:?}"
: "${PROMOTE_TOKEN:?set the protected CI/CD variable PROMOTE_TOKEN (a project access token: Maintainer; api, write_repository)}"
mirror="${MIRROR_REMOTE:-}"
upstream="${UPSTREAM_MAIN_URL:-}"
repo="$CI_SERVER_URL/$CI_PROJECT_PATH.git"
if [ -n "${PROMOTE_API_LOG:-}" ]; then
  case "$CI_SERVER_URL" in file://*) ;; *) fail "PROMOTE_API_LOG fakes only the API: CI_SERVER_URL must be a file:// URL, not $CI_SERVER_URL" ;; esac
  case "$upstream" in "" | file://*) ;; *) fail "PROMOTE_API_LOG fakes only the API: UPSTREAM_MAIN_URL must be a file:// URL" ;; esac
fi

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

# The fake API of the script's own test: protections and merge requests are kept as files next to the log.
fake_api() {
  local m=$1 p=$2 state="$PROMOTE_API_LOG.state" a name="" push="" merge="" target="" file q per page iid
  shift 2
  mkdir -p "$state/prot"
  echo "$m $p $*" >> "$PROMOTE_API_LOG"
  [ -f "$state/mrs.json" ] || echo '[]' > "$state/mrs.json"
  for a in "$@"; do
    case "$a" in
      name=*) name=${a#name=} ;;
      push_access_level=*) push=${a#push_access_level=} ;;
      merge_access_level=*) merge=${a#merge_access_level=} ;;
      target_branch=*) target=${a#target_branch=} ;;
    esac
  done
  case "$m $p" in
    "POST /protected_branches")
      file="$state/prot/$(enc "$name")"
      if [ -e "$file" ]; then case " ${also:-} " in *" 409 "*) return 0 ;; *) return 1 ;; esac; fi
      printf '{"push_access_levels":[{"access_level":%s}],"merge_access_levels":[{"access_level":%s}]}\n' "$push" "$merge" > "$file"
      ;;
    "DELETE /protected_branches/"*)
      file="$state/prot/${p#/protected_branches/}"
      if [ ! -e "$file" ]; then case " ${also:-} " in *" 404 "*) return 0 ;; *) return 1 ;; esac; fi
      rm "$file"
      ;;
    "GET /protected_branches/"*)
      file="$state/prot/${p#/protected_branches/}"
      if [ -e "$file" ]; then cat "$file"; elif [ "${p#/protected_branches/}" = "$(enc "${CI_DEFAULT_BRANCH:-main}")" ]; then
        echo '{"push_access_levels":[{"access_level":40}],"merge_access_levels":[{"access_level":40}]}'
      else return 1; fi
      ;;
    "GET /merge_requests"*)
      q=$(sed -n 's/.*target_branch=\([^&]*\).*/\1/p' <<< "$p")
      per=$(sed -n 's/.*per_page=\([0-9]*\).*/\1/p' <<< "$p")
      page=$(sed -n 's/.*[?&]page=\([0-9]*\).*/\1/p' <<< "$p")
      jq -c --arg t "$q" --argjson per "${per:-100}" --argjson page "${page:-1}" \
        '[.[] | select((.target_branch | @uri) == $t)] | .[($page - 1) * $per : $page * $per]' "$state/mrs.json"
      ;;
    "PUT /merge_requests/"*)
      iid=${p#/merge_requests/}
      jq -c --arg i "$iid" --arg t "$target" 'map(if (.iid | tostring) == $i then .target_branch = $t else . end)' "$state/mrs.json" > "$state/mrs.new"
      mv "$state/mrs.new" "$state/mrs.json"
      ;;
    "PUT /variables/"* | "POST /variables")
      if [ -n "${PROMOTE_API_FAKE_LATE_MR:-}" ]; then # a merge request opened while the job runs
        jq -c --argjson i "$PROMOTE_API_FAKE_LATE_MR" --arg t "$old" '. + [{iid: $i, target_branch: $t}]' "$state/mrs.json" > "$state/mrs.new"
        mv "$state/mrs.new" "$state/mrs.json"
      fi
      echo '{"key":"INTEGRATION_BRANCH"}'
      ;;
  esac
  return 0
}
# api METHOD PATH [curl data arguments]: prints the body; a status that is not 2xx fails unless it is listed in $also.
api() {
  local m=$1 p=$2 out code nl=$'\n'
  shift 2
  if [ -n "${PROMOTE_API_LOG:-}" ]; then fake_api "$m" "$p" "$@"; return; fi
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
# levels_of BRANCH: "push merge" access levels of its protection; fails when it has none
levels_of() {
  api GET "/protected_branches/$(enc "$1")" 2> /dev/null |
    jq -er '"\([.push_access_levels[].access_level] | max // 40) \([.merge_access_levels[].access_level] | max // 40)"'
}
# list_mrs BRANCH: the iids of the open merge requests that target it, every page
list_mrs() {
  local page=1 per=${PROMOTE_MR_PAGE:-100} body
  while :; do
    body=$(api GET "/merge_requests?state=opened&target_branch=$(enc "$1")&per_page=$per&page=$page") || fail "the merge requests cannot be listed"
    jq -r '.[].iid' <<< "$body"
    [ "$(jq length <<< "$body")" -ge "$per" ] || break
    page=$((page + 1))
  done
}
move_mrs() {
  local iid
  for iid in $(list_mrs "$old"); do
    api PUT "/merge_requests/$iid" --data-urlencode "target_branch=$next" > /dev/null
    echo "ok    merge request !$iid now targets $next"
  done
}
# protect BRANCH PUSH MERGE: (re)create the protection with these levels
protect() {
  also="404" api DELETE "/protected_branches/$(enc "$1")" > /dev/null
  api POST "/protected_branches" --data-urlencode "name=$1" --data "push_access_level=$2" --data "merge_access_level=$3" > /dev/null
}

# ---- checks: nothing is changed before the line "plan" ----
[ "$(git rev-parse --is-shallow-repository)" = false ] || fail "the clone is shallow: the job needs the full history (GIT_DEPTH: 0)"
if [ -n "$checkout" ]; then
  git diff --quiet && git diff --cached --quiet || fail "the integration checkout has uncommitted changes"
fi
old_head=$(head_of "$old")
main_head=$(head_of "$main")
next_head=$(head_of "$next")
[ -n "$main_head" ] || fail "$main is not on $CI_SERVER_URL"
if [ -n "$next_head" ]; then g fetch -q "$repo" "refs/heads/$next"; fi
if [ -n "$old_head" ]; then
  if [ "$old_head" != "$sha" ]; then
    # Only a rerun may find the old branch beyond the commit: when the new branch already holds what it gained.
    g fetch -q "$repo" "refs/heads/$old"
    git merge-base --is-ancestor "$sha" "$old_head" && [ -n "$next_head" ] && git merge-base --is-ancestor "$old_head" "$next_head" ||
      fail "$old has moved on to ${old_head:0:8}: play the job of the newest pipeline, or (after a stopped run) merge $old into $next and retry this job"
  fi
  if [ -n "$checkout" ]; then
    git merge-base --is-ancestor "$sha" HEAD || fail "the integration checkout is at $(git rev-parse --short HEAD), which does not hold ${sha:0:8}"
    if gated=$(git rev-parse -q --verify refs/ci/gated); then
      [ "$gated" = "$sha" ] || fail "the gate's last pass was ${gated:0:8}, not ${sha:0:8}"
    fi
  fi
else
  # A second start after the old branch was deleted: only the clean-up is left.
  [ -n "$next_head" ] || fail "neither $old nor $next is on $CI_SERVER_URL"
fi

g fetch -q "$repo" "$main"
gl_main=$(git rev-parse FETCH_HEAD)
main_sha=$gl_main
base_note=""
if [ -n "$upstream" ]; then
  GIT_TERMINAL_PROMPT=0 git fetch -q "$upstream" "refs/heads/$main" || fail "$main cannot be read from UPSTREAM_MAIN_URL"
  up_sha=$(git rev-parse FETCH_HEAD)
  if git merge-base --is-ancestor "$up_sha" "$gl_main"; then
    :
  elif git merge-base --is-ancestor "$gl_main" "$up_sha"; then
    main_sha=$up_sha
    base_note=" (the upstream $main is ahead of $main here)"
  else
    fail "$main here (${gl_main:0:8}) and upstream (${up_sha:0:8}) have diverged: bring one into the other by hand first"
  fi
fi
how=""
if git merge-base --is-ancestor "$sha" "$main_sha"; then
  new=$main_sha
  how="already holds ${sha:0:8}$base_note"
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
  how="a merge commit on ${main_sha:0:8} (main has commits $old lacks)$base_note"
  if [ -n "$next_head" ]; then
    git merge-base --is-ancestor "$main_sha" "$next_head" ||
      fail "$next is already on $CI_SERVER_URL at ${next_head:0:8}, which holds the promoted commit but not the commits $main has: give another name as NEXT_BRANCH"
  fi
fi
if [ -n "$next_head" ]; then
  git merge-base --is-ancestor "$sha" "$next_head" ||
    fail "$next is already on $CI_SERVER_URL at ${next_head:0:8}, which does not hold the promoted commit: give another name as NEXT_BRANCH"
fi

# The new branch gets the old one's protection (a frozen old branch has none to give: then the new one's own, or main's).
old_levels=$(levels_of "$old") || old_levels=""
push_level="" merge_level=""
for src in "${old_levels:+old}" next main; do
  [ -n "$src" ] || continue
  case $src in old) lv=$old_levels ;; next) lv=$(levels_of "$next") || continue ;; main) lv=$(levels_of "$main") || continue ;; esac
  if [ "${lv% *}" != 0 ] || [ "${lv#* }" != 0 ]; then push_level=${lv% *} merge_level=${lv#* }; break; fi
done
[ -n "$push_level" ] || fail "the token cannot read the branch protections: it needs the Maintainer role and the api scope"

on_mirror=""
if [ -n "$mirror" ]; then
  git remote get-url "$mirror" > /dev/null 2>&1 || fail "MIRROR_REMOTE $mirror is not a remote of this clone"
  on_mirror=$(GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes" timeout 60 git ls-remote "$mirror" "refs/heads/$old" | cut -f1) ||
    fail "the mirror ($mirror) cannot be reached from here, so $old could not be deleted there"
fi
moving=$(list_mrs "$old" | tr '\n' ' ')
moving=${moving% }

echo "plan  $old frozen (push and merge: No one) before anything else changes"
echo "plan  $main: $how"
echo "plan  $next: created at $main's new head and protected (push $push_level, merge $merge_level)"
echo "plan  open merge requests moved from $old to $next: ${moving:-none}"
if [ -n "$checkout" ]; then
  echo "plan  INTEGRATION_BRANCH=$next; the checkout $checkout switches to $next"
else
  echo "plan  INTEGRATION_BRANCH=$next; no checkout switch (PROMOTE_CHECKOUT is unset)"
fi
echo "plan  $old deleted: $CI_SERVER_URL${on_mirror:+, the mirror remote ($mirror)}${checkout:+, the checkout}"
echo "plan  $old stays on the GitHub mirror: delete it there by hand"
if [ "$dry" = 1 ]; then
  echo "dry run: nothing was changed"
  exit 0
fi

# ---- the promotion ----
# Frozen first: a merge that reaches the old branch from here on is refused, so what is promoted is what was checked.
if [ -n "$old_head" ]; then
  if [ "$old_levels" != "0 0" ]; then protect "$old" 0 0; fi
  if [ "$(head_of "$old")" != "$old_head" ]; then
    if [ -n "$old_levels" ]; then protect "$old" "${old_levels% *}" "${old_levels#* }"; else also="404" api DELETE "/protected_branches/$(enc "$old")" > /dev/null; fi
    fail "$old moved while it was being frozen: nothing was changed, retry the job"
  fi
  echo "ok    $old is frozen at ${old_head:0:8}"
fi

if [ "$gl_main" != "$new" ]; then g push -q "$repo" "$new:refs/heads/$main"; fi
echo "ok    $main is at ${new:0:8}"

# Protected before it exists, so that its first push already goes to the mirror (the mirror sends protected branches only).
# ci.skip when the new branch starts at the very commit whose pipeline passed.
also="409" api POST "/protected_branches" --data-urlencode "name=$next" --data "push_access_level=$push_level" \
  --data "merge_access_level=$merge_level" > /dev/null
[ "$(levels_of "$next" || true)" = "$push_level $merge_level" ] || fail "the protection of $next is not push $push_level, merge $merge_level: look at it before anything else is done"
if [ -z "$next_head" ]; then
  skip=()
  if [ "$new" = "$sha" ]; then skip=(-o ci.skip); fi
  g push -q "${skip[@]}" "$repo" "$new:refs/heads/$next"
fi
echo "ok    $next is protected and holds ${new:0:8}"

move_mrs

if ! also="404" api PUT "/variables/INTEGRATION_BRANCH" --data-urlencode "value=$next" | jq -e '.key' > /dev/null; then
  api POST "/variables" --data "key=INTEGRATION_BRANCH" --data-urlencode "value=$next" > /dev/null
fi
echo "ok    INTEGRATION_BRANCH=$next"

# A second start must not take the checkout back: the next branch may have moved on along the new branch since.
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

# Merge requests opened while the job ran are moved too, then the old branch goes, with a lease on the head that was checked.
move_mrs
if [ -n "$old_head" ]; then
  also="404" api DELETE "/protected_branches/$(enc "$old")" > /dev/null
  if ! g push -q --force-with-lease="refs/heads/$old:$old_head" "$repo" ":refs/heads/$old"; then
    protect "$old" 0 0
    fail "$old gained commits after the checks, so it was not deleted: merge it into $next, then retry this job"
  fi
fi
still=$(head_of "$old")
[ -z "$still" ] || fail "$old is still on $CI_SERVER_URL"
also="404" api DELETE "/protected_branches/$(enc "$old")" > /dev/null
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
echo "note  $old stays on GitHub: delete it there by hand"
