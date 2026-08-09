PR: https://github.com/sunava/cognitive_robot_abstract_machine/pull/35
(draft, base branch `cram-viz-integration`)

Plan: replace cram_viz-local reinventions of data structures that already
exist elsewhere in the workspace (coraplex, semantic_digital_twin) — see
the approved plan mode file for full details, summarized in the PR
description.

Done:
- ArmSide -> coraplex.Arms (Gripper.side/Arm.side) + local JointRegion enum
  for JointMotion.arm_side (BODY/ENVIRONMENT have no Arms equivalent).
- Position -> semantic_digital_twin.Point3, with explicit __eq__/__hash__
  on BenchObject (Point3 has no value-based eq/hash of its own).
- BodyExtent -> semantic_digital_twin.Scale (measure_body/rounded_scale
  free functions replace the BodyExtent class).
- ObjectPalette now stores semantic_digital_twin.Color internally,
  converting to/from hex only at color_for()'s boundary.
- UrdfJoint.type: str -> coraplex.JointType.
- Updated test_knowledge.py/test_body_geometry.py/test_palette.py for the
  new representations; added a kinematics tooltip-text test and a
  Color-storage test. All 197 tests in test/cram_viz_test pass.
- Committed, pushed, opened draft PR #35 against cram-viz-integration.

Next: subscribe to PR activity and address any review feedback/CI
failures per the usual PR-babysitting workflow.
