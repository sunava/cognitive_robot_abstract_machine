# Branch cramera-sorin-stacking

## Plan
1. Created from cramera-world-visualization (3849c589f). [DONE]
2. Cherry-picked the cramera crash/color fixes from cramera-sorin onto it. [DONE,
   commit 9feb1af2c]
3. User asked to merge remote sorin/segmind_stacking2 (a coraplex_panda_demo
   stacking demo) into this branch so demos/coraplex_panda_demo/inference2.py runs.
   [IN PROGRESS]

## Merging sorin/segmind_stacking2 — conflict resolution log (2026-08-13)
`git merge sorin/segmind_stacking2` produced 18 conflicted files (core simulation
code, not just cramera). Resolving file by file, favoring our (cramera-world-
visualization lineage's) already-refactored mixin-based design when both sides
solve the same problem, but folding in segmind_stacking2's genuinely new
capability (hand-tuned velocity defaults with real tuning rationale in their
docstrings, grasp-retry-with-verification loops, target_opening finger-force
control, per-derivative joint-limit tightening, MobileBase generic typing kept —
load-bearing, used by every robot file).

Resolved so far (all staged):
- .gitignore, coraplex/plans/executables.py, coraplex/robot_plans/{actions/core/
  pick_up.py, actions/core/placing.py, actions/core/robot_body.py,
  motions/gripper.py, motions/robot_body.py}, giskardpy/motion_statechart/
  {monitors/monitors.py, tasks/cartesian_tasks.py, tasks/joint_tasks.py},
  giskardpy/qp/qp_controller_config.py, physics_simulators/mujoco_simulator.py,
  semantic_digital_twin/{adapters/multi_sim.py, robots/robot_parts.py,
  world_description/world_entity.py}.
- Notable non-obvious finds: PickUpAction.execute()/PlaceAction.execute() now use
  segmind_stacking2's retry+verify loops (GraspVerificationFailed on exhaustion);
  ParkArmsAction's velocity field renamed back to `joint_velocity` (not
  `max_joint_velocity`) because inference2.py/demo3.py call it by that exact
  keyword; robot_parts.py had ~30 conflicts that were 28 pure docstring-rewrap
  (resolved to ours) but 2 hid real logic (a Derivatives-based per-derivative
  joint-limit-tightening refactor with a real None-check bugfix, and a
  finger_tip_frame docstring whose "theirs" side depended on now-orphaned shared
  trailing lines — had to hand-fix both, verified via `ast.parse` + real import).
- Deliberately kept our AttachNode/DetachNode active in PickUpAction/PlaceAction
  (rather than segmind_stacking2's disabled-for-friction-only-grasp experiment) to
  preserve default behavior for every other existing caller (e.g. montessori demo)
  — flag this to the user: if the panda demo needs pure-friction holding with no
  kinematic welding, it needs an explicit opt-out added, not a global default flip.

## Still remaining
- Resolve conflicts in: coraplex/orm/ormatic_interface.py, experiments/orm/
  ormatic_interface.py, semantic_digital_twin/orm/ormatic_interface.py (do NOT
  hand-merge — revert to one side then run scripts/regenerate_all_orm.py),
  test/coraplex_test/test_designator/test_motion_designator.py,
  test/giskardpy_test/test_motion_statechart/test_motion_statechart.py,
  test/semantic_digital_twin_test/test_adapters/test_mjcf.py,
  test/semantic_digital_twin_test/test_adapters/test_multi_sim.py.
- Run affected test suites after all conflicts resolve.
- Locate and actually run coraplex/demos/coraplex_panda_demo/inference2.py to
  confirm the user's actual goal.
- Not yet committed — merge is still in progress (`git status` shows `MERGE_HEAD`).

# Branch cramera-sorin — montessori demo fixes (previous branch, for reference)

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
- Control run confirmed the hypothesis: same command, same `--max-shapes 3`, with
  `CORAPLEX_VISUALIZATION=NONE` (cramera off) completed cleanly, 3/3 shapes fell
  through, exit code 0, no crash. So the race predates cramera and isn't caused by
  it, but cramera's extra thread activity (even after the fixes above) makes the
  timing window far more likely to be hit. Reported this to the user with the
  evidence; did not implement a fix — needs a decision on approach (global casadi
  lock vs. restructuring which thread may build symbolic expressions) since it spans
  `experiments`/`semantic_digital_twin`/`krrood`.

## Next
- Awaiting user decision on whether/how to pursue the cross-package casadi
  thread-safety fix (event_monitoring thread vs. main thread vs. physics thread all
  building CasADi SX concurrently with no lock).
- No PR opened yet for this branch.
