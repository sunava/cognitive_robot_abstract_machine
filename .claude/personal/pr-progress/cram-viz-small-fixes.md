# cram-viz-small-fixes (PR #22, fix-my-pr plan, item viz-small-fixes)

## Plan
Implement Group C of the fix-my-pr triage (roadmap.md): T14, T9, T10, T3, T7,
T8, T21, T42, T38, T32, T30/T47 - renames, dead-code deletions, one-line
corrections. One commit per fix, test-first where behavior changes. Branch
off cram-viz-integration, draft PR on sunava fork. Full plan text was in
/root/.claude/plans/glittery-wobbling-lake.md (this session).

## Done
All 10 fixes implemented and committed (10 commits), suite green (139 passed,
was 137 baseline + 2 new tests) after every commit. Pushed cram-viz-small-fixes
and opened draft PR #22 against cram-viz-integration. Subscribed to PR
activity. CI just started (pending, 0 checks yet as of first check).

## Flags carried into the PR description
- viz-bugs (sibling item, not started) touches the same PLAN_GROUPS dict
  (kb.py:1638-1644) as this PR's T42 - whichever lands first on
  cram-viz-integration forces a rebase of the other.
- T21 scope: only link()'s src/dst renamed; the EQL query-namespace shorthand
  (obj/ep/rob/pkg/sub/cls) was deliberately left alone as documented DSL
  convenience - flagged in the PR body for the reviewer/reply-sheet to confirm.

## Next
- Watch PR #22 for CI results and any review comments (subscribed).
- Once CI is green and the user has reviewed, mark ready-for-review if asked.
- viz-reply-sheet (later item) will post the actual T-thread replies to the
  upstream cram2 PR; this PR does not touch that PR.
