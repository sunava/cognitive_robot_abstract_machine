# viz-bugs (fix-my-pr plan) — fix mis-keyed attach/detach plan nodes + EQL preset splicing

Full plan: /root/.claude/plans/clever-wondering-panda.md (approved this session).

## Plan
1. Branch `cram-viz-bugs` off `cram-viz-integration`.
2. BUG-1: failing tests (`test_kb.py::TestPlanGroups`, new `test_graph_panel.js`) →
   fix `kb.py`'s `PLAN_GROUPS` + `panel.js`'s `PLAN_GROUP` (both key on
   `AttachmentNode`/`DetachmentNode`, real classes are `AttachNode`/`DetachNode`).
3. BUG-2: failing tests (`test_kb.py::TestPresetSafety`) → fix `get_presets()`'s
   3 unescaped `'%s' % name` splices in EQL source, using `repr(name)`.
4. Run `python -m pytest test/cram_viz_test -q` + JS suite; `scripts/format_docstrings.py` on `kb.py`.
5. Open draft PR on sunava fork, base `cram-viz-integration`, label `bug`.

## Status
Not yet started — about to create the branch.

## Next
Create `cram-viz-bugs` branch, write the BUG-1 Python tests first.
