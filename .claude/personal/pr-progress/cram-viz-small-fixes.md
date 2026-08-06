# cram-viz-small-fixes (PR #22, fix-my-pr plan, item viz-small-fixes)

## Plan
Implement Group C of the fix-my-pr triage (roadmap.md): T14, T9, T10, T3, T7,
T8, T21, T42, T38, T32, T30/T47 - renames, dead-code deletions, one-line
corrections. One commit per fix, test-first where behavior changes. Branch
off cram-viz-integration, draft PR on sunava fork. Full plan text was in
/root/.claude/plans/glittery-wobbling-lake.md (this session).

## Done
All 10 fixes implemented and committed, suite green after every commit.
Pushed cram-viz-small-fixes and opened draft PR #22 against
cram-viz-integration. Subscribed to PR activity.

CI failure round 1: bundle_urdf's narrowed excepts (T9/T10) didn't cover
OSError, which ament_index_python raises (via EnvironmentError) when
installed-but-unsourced (CI runs in a real ROS container; my local env had
no ament_index_python at all, so I only saw the ImportError path). Fixed by
widening both excepts to include OSError (commit d734a54e), verified locally
by injecting a fake ament_index_python module that raises exactly that
OSError.

Merge conflict round 1: sibling viz-bugs (PR #20) merged into
cram-viz-integration, landing BUG-1/BUG-2 fixes on the same PLAN_GROUPS dict
and get_presets() this branch's T42/T30/T47 touch. Resolved via merge commit
f0d7da86: kept viz-bugs' corrected dict keys (AttachNode/DetachNode) with
this branch's PlanNodeGroup enum values, renamed the new BUG-1/BUG-2 tests'
kb.reset_kb() calls to kb.reset_knowledge_base(). mergeable_state now clean,
144 passed. PR description updated to reflect both fixes.

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
