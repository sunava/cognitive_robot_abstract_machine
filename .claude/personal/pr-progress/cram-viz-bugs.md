# viz-bugs (fix-my-pr plan) — fix mis-keyed attach/detach plan nodes + EQL preset splicing

Full plan: /root/.claude/plans/clever-wondering-panda.md (approved this session).

## Plan
1. Branch `cram-viz-bugs` off `cram-viz-integration`. [done]
2. BUG-1: failing tests (`test_kb.py::TestPlanGroups`, new `test_graph_panel.js`) →
   fix `kb.py`'s `PLAN_GROUPS` + `panel.js`'s `PLAN_GROUP` (both key on
   `AttachmentNode`/`DetachmentNode`, real classes are `AttachNode`/`DetachNode`). [done]
3. BUG-2: failing tests (`test_kb.py::TestPresetSafety`) → fix `get_presets()`'s
   3 unescaped `'%s' % name` splices in EQL source, using `repr(name)`. [done]
4. Run `python -m pytest test/cram_viz_test -q` + JS suite; `scripts/format_docstrings.py` on `kb.py`. [done — 142 passed]
5. Open draft PR on sunava fork, base `cram-viz-integration`, label `bug`. [done — PR #20]

## Status
PR #20 opened (draft), labelled `bug`, subscribed to activity. CI's `cram_viz`
job (the relevant one) is green; a few unrelated jobs were still in progress
right after push. No review comments yet (fresh draft). Note: caught and fixed
a branch-mix-up mid-session — local `cram-viz-integration` was stale (missing
the latest cram2:main merge); `cram-viz-bugs` was reset onto
`origin/cram-viz-integration` before any commits landed, so no impact on the
final PR.

## Next
Scheduled a ~1h self check-in (send_later) to re-check CI/reviews. Nothing
else pending unless new activity arrives. Item is otherwise complete pending
review.
