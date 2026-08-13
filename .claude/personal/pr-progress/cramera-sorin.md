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

## Fixed (2026-08-13) — cramera-caused segfault and gray shapes
- Cramera-viewer segfault: `WorldStateSync.on_state_change` -> `Bridge.snapshot()` ->
  `rounded_pose` -> `to_position_quaternion_list` was constructing casadi SX *on the
  MuJoCo physics thread*, racing the executor/event-monitoring threads' own casadi
  work (casadi SX refcounting is not thread-safe). Also unsafe: `/live_scene` and
  `/info` building/reading casadi-backed state on HTTP threads, and `snapshot()`'s
  periodic rebind doing SX ops on the physics thread. Fixed all four in cramera:
  - `body_geometry.rounded_pose` now numpy-only (`compute_forward_kinematics_np` +
    scipy `Rotation`), tests assert no `SymbolicMathType` gets constructed.
  - `Bridge.snapshot()` no longer rebinds periodically (model-change callback already
    rebinds on the correct thread).
  - `Bridge.status()` serves `robot_parts` captured at bind time
    (`capture_robot_parts()`) instead of walking live annotations on the HTTP thread.
  - `/live_scene` now only reads `Bridge.live_scene()` (set by
    `record_live_scene()`); the actual `build_live_scene()` call moved to
    `LiveVisualization.start()` and `WorldModelSync.on_model_change()`, both demo-owned
    threads.
  - Verified: headless run with sustained HTTP polling (`/info`, `/live_scene`,
    `/state`, `/plan`, `/chart`, `/objects`) now gets past the original crash point
    and completes multiple shape insertions (previously crashed on shape 1, attempt 1).
- Wrong colors in the live viewer: montessori shapes/board are `Mesh.from_trimesh`
  (OBJ export, no MTL). Vendored URDFLoader only applies the URDF `<material>` to a
  bare `THREE.Mesh`, but panel.js's OBJ loader always returns OBJLoader's wrapping
  `Group`, so the color was silently dropped (disk/sphere are URDF primitives and were
  already correct). Fixed: new pure module `web/core/obj-mesh-material.js`
  (`singleMeshChild`) finds the one mesh inside a materials-less OBJ group; panel.js's
  `loadMeshCb` hands that mesh to `done()` instead of the group when no `.mtl` was
  loaded. Covered by `test/cramera_test/js/test_obj_mesh_material.js`, wired into
  `test_web_assets.py::TestJsUnits`.
- All 351 cramera pytest tests pass; all cramera JS unit test files pass under
  `node --test`.

## Found (2026-08-13), NOT fixed — pre-existing race, independent of cramera
- After the above fixes, a longer headless run (3 shapes, sustained polling) still
  segfaulted, but at a different, later point: `franka_montessori_demo.py:337
  _insert_shape` (main thread) calling `body.global_transform` at the same moment
  `experiments/montessori/event_monitoring.py`'s background thread's
  `spatial_relation_detector_nodes.py -> predicates.compute_containment_ratio` also
  calls `global_transform` — both build casadi SX concurrently with no lock. This is
  the same underlying hazard (casadi SX construction is not thread-safe) but it lives
  in `semantic_digital_twin`/`krrood`/`experiments`, not in cramera, and predates this
  branch's cramera work. Started a control run with `CORAPLEX_VISUALIZATION=NONE` to
  check whether it's reproducible without cramera too (cramera's extra thread activity
  may just make the race window much likelier to hit, not be its cause) — check
  `/tmp/.../scratchpad/demo_nocramera.log` for the outcome before deciding whether/how
  to raise this with the user. This needs a decision on approach (e.g. a lock guarding
  casadi construction, or restructuring which thread is allowed to build symbolic
  expressions) before touching cross-package code — do not implement unilaterally.

## Next
- Check the `CORAPLEX_VISUALIZATION=NONE` control run's outcome and report the
  cramera-independent race to the user with that evidence.
- No PR opened yet for this branch.
