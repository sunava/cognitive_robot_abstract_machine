# PR #36 — Rerun 3D visualization for the bullet-world demo

PR: https://github.com/sunava/cognitive_robot_abstract_machine/pull/36 (draft)
Branch: rerun-demo-visualization (off main)
Plan file this implements: ~/.claude/plans/enchanted-inventing-sonnet.md

## Status: implementation complete, awaiting review

All plan steps done and verified locally:
1. ✅ RerunAdapter: textures/UVs kept, albedo tint for colorless meshes,
   state_log_stride + log_current_state, default blueprint, prefixed
   entity paths (8 tests green).
2. ✅ Plan.node_callbacks wired: PlanNode.perform() + MotionLifeCycleTracker
   in the Giskard tick loop (test_plan/ 78+1 green).
3. ✅ coraplex.visualization: WorldVisualization + RerunPlanCallback +
   CORAPLEX_* env vars; testing.start_visualization delegates (5 tests).
4. ✅ Demo rewired, CI wrapper pins backend to none; full demo ran
   end-to-end in SAVE mode → 379 world + 14 plan entities in .rrd.
5. ✅ Docs: Rerun section in visualizing_worlds.md (CI-safe cells),
   demo README. Docstrings formatted via pre-commit.
6. ✅ ORM regenerated via scripts/regenerate_all_orm.py (needs argcomplete
   installed, else giskardpy ros2 modules silently drop out of the scan).

## Next steps
- Watch CI on the draft PR; fix anything red.
- On review: address comments, keep PR draft after every push.
- Follow-up initiative (not this PR): roll WorldVisualization out to the
  other demos (Unitree G1, Garmi, tool-based, SAGE-10k, RoboCasa) —
  candidate for /plan-create.
