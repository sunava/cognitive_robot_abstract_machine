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
MERGED. PR #20 went green (all 21 checks), was marked ready for review by
sunava, and merged with no review comments. This session was auto-unsubscribed
from PR activity on merge. `plan.yaml`'s `viz-bugs` item updated to
`pull_request_number: 20`, `status: done`.

## Next
Nothing — item complete. Next plan item to pick up (per `depends_on`):
`viz-kb-characterization` (depends on `viz-bugs`) or `viz-small-fixes`
(no dependencies), whichever the user wants to kick off next.
