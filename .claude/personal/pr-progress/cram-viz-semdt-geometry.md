# `viz-semdt-geometry` (fix-my-pr plan, item T17)

Draft PR: https://github.com/sunava/cognitive_robot_abstract_machine/pull/29
Branch: `cram-viz-semdt-geometry`, based on `cram-viz-integration`.
Plan approved and bootstrapped (branch + draft PR created, item recorded
`in_progress`, roadmap section appended). Full plan lives in the roadmap
section on `claude/personal-notes` — summary below.

## Plan
- Fix `cram_viz/src/cram_viz/body_geometry.py`'s `BodyExtent.of`: replace the
  Box/Mesh-only `isinstance` scan with a call to `ShapeCollection.scale`
  (checks `body.visual` then `body.collision`, first non-empty wins; `None`
  only when both are empty). Drop the now-unused `Box`/`Mesh` import.
- New `test/cram_viz_test/test_body_geometry.py` (TDD: write first, confirm
  Sphere/Cylinder cases fail against current code, then apply the fix):
  Box, Mesh, Sphere, Cylinder, no-shapes-at-all, visual-preferred-over-
  collision, and `rounded()`.
- Update two `test_live_bridge.py` tests
  (`test_an_object_without_a_mesh_is_catalogued_as_a_sized_box`,
  `test_an_object_with_unscaled_shapes_falls_back_to_the_default_size`) to
  use a real `World`+`Body` instead of the world-less `PublishedBody`/
  `ShapeSet` mimic — `ShapeCollection.scale` needs a shape's reference frame
  wired into a real `World` (`BoundingBox.transform_to_origin` reads
  `origin.reference_frame._world`), which the mimic never had.
- No change needed at either call site (`live/bridge.py`'s `_box_size`,
  `onboard/demo.py`'s onboarder height lookup) — both already handle `None`.
- Run `scripts/format_docstrings.py` on touched files; run
  `python -m pytest test/cram_viz_test -q`, report the pass count in the PR
  description.

## Done so far
- Plan approved via plan-item-kickoff.
- Branch `cram-viz-semdt-geometry` created off `cram-viz-integration`,
  pushed, draft PR #29 opened.
- `fix-my-pr` plan.yaml: item flipped to `in_progress` with branch/session/PR
  number recorded; roadmap.md section appended.

## Next
- Implement the plan above (test-first), starting with
  `test_body_geometry.py`'s Sphere/Cylinder cases against the current code
  to confirm they fail.
- Update the PR description once implementation lands, matching sibling PR
  #27's structure (implements-item / thread-closed / suite-result).
- Republish the `fix-my-pr` dashboard after implementation.

Flag (not part of this item, noted so it isn't lost): fork PR #18
(`warehouse-viz-features`) is closed unmerged as of 2026-08-07, contradicting
roadmap.md's earlier "MERGEABLE" note and the "refactor first, rebase #18
afterward" decision — relevant to `viz-kb-split` later, not to this item.
