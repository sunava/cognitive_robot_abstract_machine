# `viz-semdt-geometry` (fix-my-pr plan, item T17)

Draft PR: https://github.com/sunava/cognitive_robot_abstract_machine/pull/29
Branch: `cram-viz-semdt-geometry`, based on `cram-viz-integration`.
**Implementation complete**, suite green (189 passed), PR description
updated. Still in draft, awaiting the user's own review per personal-notes
convention.

## What landed
- `cram_viz/src/cram_viz/body_geometry.py`'s `BodyExtent.of`: replaced the
  Box/Mesh-only `isinstance` scan with `ShapeCollection.scale` (checks
  `body.visual` then `body.collision`, first non-empty wins; `None` only
  when both are empty). Dropped the now-unused `Box`/`Mesh` import.
- New `test/cram_viz_test/test_body_geometry.py` (TDD: Sphere/Cylinder cases
  confirmed failing against pre-fix code first): Box, Mesh, Sphere,
  Cylinder, no-shapes-at-all, visual-preferred-over-collision, `rounded()`.
- `test_live_bridge.py`: only
  `test_an_object_without_a_mesh_is_catalogued_as_a_sized_box` needed
  updating, to a real `World`+`Body` fixture (`ShapeCollection.scale` needs
  a shape's reference frame wired into a real `World`). The sibling
  `test_an_object_with_unscaled_shapes_falls_back_to_the_default_size`
  needed no change — an empty `ShapeSet` never reaches `.scale`.
- No call-site changes: `live/bridge.py`'s `_box_size` and
  `onboard/demo.py`'s onboarder height lookup already handle `None`.
- Ran `scripts/format_docstrings.py` on all three touched files (installing
  `docformatter` via `uv add --dev` unintentionally modified `pyproject.toml`
  with a dependency-group entry; reverted before committing since it wasn't
  part of this item's scope).
- `uv run pytest test/cram_viz_test -q` — 189 passed.
- PR #29 description updated to match sibling PR #27's structure.

## Next
- Republish the `fix-my-pr` dashboard.
- Nothing else outstanding on this item; it's ready for the user's own
  review pass (per personal-notes convention, stays draft until then).

Flag (not part of this item, noted so it isn't lost): fork PR #18
(`warehouse-viz-features`) is closed unmerged as of 2026-08-07, contradicting
roadmap.md's earlier "MERGEABLE" note and the "refactor first, rebase #18
afterward" decision — relevant to `viz-kb-split` later, not to this item.
