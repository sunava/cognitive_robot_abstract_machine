# Branch cramera-sorin — montessori demo fixes

## Plan
Fix crashes hit while running the Franka montessori demo
(`experiments/src/experiments/montessori/franka_montessori_demo.py`).

## Done (2026-08-13)
- Fixed `TypeError: unhashable type: 'Plan'` in
  `GiskardExecutable._notify_motion_tick` (coraplex/plans/executables.py:392):
  `Plan` is now `@dataclass(eq=False)` so it hashes/compares by identity,
  matching the `is` comparisons used everywhere else. Covered by new test
  `test_notify_motion_tick_notifies_each_plan_exactly_once` in
  `test/coraplex_test/test_plan/test_executables.py`.
- Regenerated stale ORM interfaces via `scripts/regenerate_all_orm.py`
  (needed `.venv/bin` on PATH for `ruff`); test collection was broken by a
  missing `ShelvingUnit` attribute before that.

## Known pre-existing failures on this branch (not caused by the fix)
- test_graph_parsing.py: test_parse_pick_up, ..._merges_motions_around_model_change,
  test_parse_pick_place, test_parse_transport_plan
- test_plan.py: test_node_expansion (expects 4 children, gets 3)

## Next
- Re-run the montessori demo to confirm insertion proceeds past the crash.
- No PR opened yet for this branch.
