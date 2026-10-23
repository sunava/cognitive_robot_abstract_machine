## cramera-isaac-import (tracks sunava/cramera-port) - Isaac/USD -> CRAMERA import

Plan: `cramera-import` turns Isaac Sim/USD stages into CRAMERA scene bundles (environment only by default),
viewer additions for imported scenes, Isaac replay bundles, live CRAM world from a bundle. No PR (works directly
on the fork branch cramera-port).

Done:
- 2026-10-20/21: converter, viewer additions, animation/replay, live world, environment/robot separation,
  reviews; committed by the user as b4b13a85a8 "isac export" and pushed to sunava/cramera-port.
- 2026-10-22: found the 4 texture fixtures in test/cramera_test/dataset/usd_scene/textures/ were left out by
  `.gitignore` (*.png): 13 of 144 usd_scene tests fail without them. Added a .gitignore exception; the PNGs are
  untracked and waiting for the user's OK to commit + push.

- 2026-10-22: fast-forwarded to sunava/cramera-port f6ac5b6a4 (50 commits); fixture fix re-applied on top.
  WarehouseTwin (full shelves, 566 static stock items) re-exported to ~/cramera_scenes/warehouse (gate passed).

In progress (user decision 2026-10-22: "das warehouse soll das genauso machen [wie das medical lab] und alle
zukuenftigen mit dem skill auch"): CRAM-native robot demos for imported twins, like the medical lab.
1. cramera-import: USD colliders -> <collision> boxes in environment.urdf; enabled rigid bodies -> free objects
   (rigidBodyEnabled false / kinematic stay fixed links); rigid bodies nested in a free body ride fixed to it.
   TDD with new usd_scene fixtures; usd_scene_world loads collisions and free objects.
2. WarehouseTwin demo: CRAM Tiago picks tote(s) from row A and places them at the packing station, recorded
   as warehouse_tiago_<task> in ~/.cramera/scenes; publish to ~/cramera_scenes after the user has seen it.
3. isaac-twin skill: Phase 9 + robot task demo become the standard (collisions, free objects, pick/place);
   rsync the skill to the vault.

Progress 2026-10-22 (evening):
- committed locally (not pushed): a65855bda texture fixtures (+ .gitignore exception), 842034d49 collision
  boxes + --free-objects (TDD: stage reader 8, URDF writer 2, bundle 8, world 4 new tests; all 220 usd_scene
  tests pass). Full cramera suite: 35 failed / 20 errors, all outside the USD import (lab ScaleProgramState
  .snapshot missing, MuJoCo MjData.qM, Blender 5.2 use_auto_smooth, offline asset checks) - pre-existing.
- skill wrapper export_cramera.sh passes --free-objects by default (--fixed-bodies opts out) and gates
  collision boxes; warehouse re-exported: 21 free totes, 6596 collision boxes, CRAM world loads in 18 s.
- WarehouseTwin/cramera/demo_tiago_two_totes.py: CRAM Tiago, RailPickUpAction (grip at goods.rail_grasp_height),
  approach pose 0.8 m before each stop + RollAction (straight CartesianPose along the current heading: the
  route planner inflates by the 0.71 m parked turning radius of a differential drive), B00 tote cleared first.
  Steps verified up to the in-feed roll-in; full recorded run in progress.

Progress 2026-10-23:
- coraplex fix 906c53d6b (local, not pushed): PlaceAction takes the grasp of the placed object's own pick-up,
  not the latest pick-up (two totes: the left set-down used the right hand's grasp, ~0.3 m off,
  MotionDidNotFinish). TDD test in test_place_attachment.py; the same 25 plan/motion/multi-robot/ORM tests fail
  with and without the fix (pre-existing). PlanNode.get_previous_node_by_designator_type is now unused (ask).
- demo recorded: warehouse_tiago_two_totes in ~/.cramera/scenes, 2919 frames, A08 3.4 mm / A09 3.1 mm
  (twin commit 271e1d5: RaiseAction over the in-feed, carried place yaw, single-collider carton).
- skill synced to the vault and pushed (sunava/mavaobs c49c67c).
- user 2026-10-23: finish the demo for CRAMERA (not Plan Builder + MuJoCo: the builder only loads
  coraplex/resources/worlds URDFs and always runs simulated_robot_advanced; no MuJoCo executor, lab MuJoCo is PR2-only).
- 376f204d4 viewer: robot Phong/Lambert materials become standard materials in scenes lit by their own
  environment map (TIAGo rendered black); node test test_model_materials.js, all 1019 JS tests pass.
- 41425c5ca cherry-pick of upstream 208cf0e44f (Simon Stelter): collision groups asked bodies_with_collision once per
  body (536 of 568 s of a park + pick); demo now 1 min 48 s instead of 55 min.
- twin fa63d25: recording written at 50 Hz (CONTROL_FREQUENCY) -> 2938 frames in 59 s instead of 48 min.
  Skill vault 109cb7b.

- afternoon: export without the roof (--exclude /World/Site/Roof); Stretch added to the demo
  (WarehouseTwin demo_tiago_stretch.py, recording warehouse_tiago_stretch_totes, R01 -> A10 1 mm); skill vault 6e6481b.
- side sessions: frame rate from simulation time -> cramera 5d0e97a0e on branch claude/infallible-ellis-6967e3
  (not merged here); Plan Builder view beside the replay -> built in sunava/cramera v1 (~/cramera), branch
  replay-plan-steps (on origin, pushed by its side session), monorepo-bundle-compat local. The user told that
  session "ab jetzt immer in sunava cramera" (to confirm with her for this line of work).
- evening: scenes repo AICOR-Solutions/solutions-cramera-scenes (private, clone ~/solutions-cramera-scenes),
  main d078b06: aicor_lab, aicor_lab_r01, warehouse (from ~/cramera_scenes) + warehouse_tiago_two_totes,
  _stretch_totes, _plan_steps (from ~/.cramera/scenes), one merged index.json, README; plain git, no LFS.
  v1 cramera-check-bundle reads warehouse + the two monorepo recordings as unreadable scene.json (pre-v1 format).
- late evening: v1 on another machine failed with "'[object Object]' is not a file.": the two TIAGo recordings
  inlined the lighting in rendering.environment (monorepo ScenePresentation.from_bundle inlines it; v1 wants a
  file name). Scenes repo commit 1ab6bae (pushed to origin main) moves it to lighting.json; all 4 warehouse bundles
  pass cramera-check-bundle and serve every requested file; ~/.cramera/scenes copies updated to match.

Progress 2026-10-23 (night): v1 rerecording (user: "pull AICOR-Solutions/cramera and rerecord the tiago demo";
chose "Full v1 port first", demos "Both"):
- cramera v1 branch usd/colliders-free-objects (local, not pushed; worktree in this session's scratchpad
  cramera-main, commits in ~/cramera/.git), on main 3b26ddb: afed8ea colliders -> <collision> boxes,
  --free-objects (objects/<key>.urdf via ObjectDescription.urdf), BundleEnvironment loads object URDFs,
  PrimEntry.model; 3dc4a45 cramera-export forwards --free-objects; 89e2bb7 RobotTaskDemo.build_plans (each plan
  published right before it performs, else the first plan's steps lose their frames). usd 306 + cram 276 pass;
  the 7 viewer/offline/screenshot failures also fail on pristine main (stale laboratory entry points in the
  cramera-v1 venv; this session used an isolated venv scratchpad/venv-main for the viewer).
- ~/cramera_scenes/warehouse re-exported with v1 cramera-export (format 1.0, 21 free totes, 2900 boxes); the v0
  bundle is in solutions-cramera-scenes 1ab6bae. Old v0 recordings moved out of ~/.cramera/scenes (same commit).
- WarehouseTwin f71c0ae: cramera/warehouse_demos.py + demo_tiago_two_totes.py + demo_tiago_stretch.py on v1
  RobotTaskDemo; recordings in ~/.cramera/scenes: warehouse_tiago_two_totes (2924 frames, 58.5 s, 3.4/3.1 mm),
  warehouse_tiago_stretch_totes (9979 frames, 199.6 s, R01 1.0 mm), both with plan step frames; not published.

Next:
- ask the user: push usd/colliders-free-objects to AICOR-Solutions/cramera and open a draft PR;
- publish both recordings (+ v1 warehouse) to the scenes repo / ~/cramera_scenes once the user has seen them;
- follow-ups found: multi-robot recordings name models/index robot `stretch_<hex>` (picker shows it); recorded
  object keys are body UUIDs, so plan step targets read as UUIDs instead of A08; the isaac-twin skill still
  exports and records with the monorepo cramera (switch it to v1 cramera-export + RobotTaskDemo);
- push the CRAM commits a65855bda, 842034d49, 906c53d6b, 376f204d4, 41425c5ca (on the user's OK);
- open decisions handed to the user: `usd` extra deps (numpy/scipy/Pillow), CI usd-core step, SSAO sRGB fix
  changing every AO scene, exposure 0.72, glass blend approximation, test-only Pose methods;
- known gaps: no generic demo runner (helpers live in the twin's warehouse_demos.py); v1 live recordings estimate
  fps from wall clock (the demo rewrites 50 Hz); the warehouse bundle is 210 MB, recordings 227/303 MB.
