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

## Diagnosed (2026-08-13), fixes not yet applied
- Cramera-viewer segfault ("Aborted/core dumped" in the demo): reproduced headless
  with faulthandler. `WorldStateSync.on_state_change` → `Bridge.snapshot()` →
  `rounded_pose` → `to_position_quaternion_list` constructs casadi SX *on the MuJoCo
  physics thread* (multi_sim `_sim_to_world` → `notify_state_change`), racing the
  executor/event-monitoring threads' own casadi work; casadi SX refcounting is not
  thread-safe. Same class of problem: `/live_scene` (URDF bundle build) and `/info`
  (`RobotPartAnnotation.of_robot`) read the casadi-backed world on HTTP threads, and
  `snapshot()`'s 3-second rebind runs `_build_object_metadata` (SX ops) on the physics
  thread. Fix plan: numpy-only pose reads in snapshot (compute_forward_kinematics_np +
  scipy quaternion), drop periodic rebind (model-change callback already rebinds),
  cache robot part annotations, build live bundle eagerly on attach/model change and
  serve only the cache from HTTP.
- Wrong colors in the live viewer: montessori shapes/board are `Mesh.from_trimesh`
  (OBJ export, no MTL). URDF carries correct `<material rgba>`, but panel.js
  `loadMeshCb` returns OBJLoader's Group and vendored URDFLoader only applies the URDF
  material to a bare `THREE.Mesh` → from_trimesh bodies render default gray (disk and
  sphere are URDF primitives and stay correct). Fix plan: in panel.js `loadMeshCb`,
  when no MTL loaded and the OBJ group has exactly one mesh, hand URDFLoader the mesh
  itself.

## Next
- Await go-ahead, then implement both fixes with tests.
- Re-run the montessori demo to confirm insertion proceeds past the crash.
- No PR opened yet for this branch.
