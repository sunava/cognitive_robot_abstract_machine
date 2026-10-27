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

Progress 2026-10-26 (user: plan steps need more parameters, as in the Plan Builder, and short object names instead
of UUIDs; chose "Nur schnell" and the name "A08 · computer mouse box"):
- cramera main needs CRAM main since 766cd33 (10-23 22:22, "Port cramera-cram to CRAM main"); the CRAM fork here is
  2517 commits behind, and the twin demo uses fork-only API (GraspDescription, ViewManager). So the demo stays on
  the old stack: cramera branch viewer/readable-step-cards 080fa36 (records every step parameter, display names
  from sdt:canonicalName, robot named by class) + ~/.virtualenvs/cramera-v1 + this fork + ROS; 217 cram tests pass.
- re-recorded warehouse_tiago_stretch_totes into ~/.cramera/scenes (A08 7.1 mm, A09 9.6 mm, R01 1.0 mm, 203 s);
  the 10-25 recording is in the session scratchpad (previous-recordings). The scenes repo moved to
  ~/cramera/demo/solutions-cramera-scenes (holds the warehouse bundle and the old recordings); a recording whose
  name is in the shared folder is refused at save, so CRAMERA_SCENES pointed at a folder linking only `warehouse`.
- checked in main's viewer (headless Firefox): steps show object designator, arm, grasp, rail height, poses.
  Screenshots ~/Pictures/WarehouseTwin/plan_view_*_rail_pick_up.png. Blemish: the twin's closed_fingers field
  reads "connections RevoluteConnection, ..., name PrefixedName".

- published warehouse_tiago_stretch_totes: scenes repo a4b79d5 (pushed).
- LabAutomationTwin R01 cycle (user: "das soll eine cram demo sein"; FR3 as in Isaac, newest stack, full loop):
  Isaac replay aicor_lab_r01 (5d8ae9c) and CRAM demo aicor_lab_fr3_plate_cycle + env aicor_lab (d3fc3f9), pushed.
  cramera PR AICOR-Solutions/cramera#73 (branch usd/collision-boxes-free-objects: the 3 colliders/free-objects/
  build_plans commits rebased onto main + mesh-beside-URDF fix 1729c77 + changelog), ready for review, bound here.
  Demo code untracked in ~/Documents/LabAutomationTwin/cramera (repo identity is mbeetz: ask before committing).
  isaac-twin skill section 6a, vault 7d5d6e0.
- 2026-10-27 (user: targets in the TIAGo warehouse demo's replayed plan not correct): warehouse_tiago_two_totes
  (10-25) predated the readable names, so main's plan panel showed body UUIDs and no poses. Re-recorded it on the
  same old stack (worktree of viewer/readable-step-cards 080fa36, env bundle from scenes repo 4264188^ since the
  user removed `warehouse` there): A08 7.1 mm, A09 9.6 mm, 62 s; steps now read "A08 · computer mouse box" and
  target locations. The twin's reach/traffic data were stale only because another session moved the catalogue
  path (~/Documents/private_meshes -> ~/private_meshes, uncommitted layout.py/params.py): regenerated
  data/reach.json (38/38) + data/traffic.json (193/193, via ~/LabAutomationTwin/env_cad for shapely), results
  identical, only provenance changed; uncommitted. Old recording backed up in this session's scratchpad.
  Not published (two_totes is no longer in the scenes repo). Headless Firefox shows a grey 3D stage for both
  warehouse recordings in main's viewer (also the published stretch one): unexplained, not checked by eye.
- 2026-10-27 (user: a skill "install cramera main, install cram and test one time" so isaac-twin hands over on the
  newest stack): new personal skill ~/.claude/skills/cramera-stack (update_stack.sh: ff ~/cramera main, CRAM
  checkout ~/cognitive_robot_abstract_machine detached on origin/main, cramera's install_stack.py --with-usd --dev
  into ~/.virtualenvs/cramera, per-package tests with ROS and -p no:launch_testing -p no:launch_ros, hand-over
  check: shelf_with_totes through export_cramera.sh + CRAM world with a PR2; result in
  ~/.local/state/cramera-stack/result.json). isaac-twin now defaults to ~/.virtualenvs/cramera (wrapper,
  SKILL.md table/Phase 9/10, cramera.md 6a/10); template still old stack. First run: cramera a31d493 + CRAM
  4eaae933a, all green except bridge test_demo (DuplicateWorldEntityError with two PR2s, also on CRAM 1bd0c40a8)
  and 2 laboratory MuJoCo tests (CRAM pins mujoco==3.11.0, CI uses 3.17.0).

Progress 2026-10-27 (fork PR #41 cramera_bridge/live/plan.py -> CRAM; user: "do them as single prs"):
- fork main fast-forwarded to cram2 main 4eaae933a; local branch cram2-main = that commit (worktree base).
- 5 sessions, each one draft PR in sunava/cognitive_robot_abstract_machine: action-status (ActionNode status from
  merged motions, bug; then Plan.running_action stacked), readable-parameters (node_info text), perception-result
  (DetectingMotion node keeps detections), plan-root (named exception + is_tree), motion-tick-hook (per-tick
  PlanCallback.on_motion_tick: the bridge's on_motion_tick is never called, live constraints/statechart freeze
  during motions; also in cramera main packages/bridge).
- user: bridge fixes go to AICOR-Solutions/cramera packages/bridge, #41 re-copies later. They need the CRAM PRs
  in cram2 main first (cramera runs on CRAM main); then one cramera PR that drops NodeLifeCycle, running(),
  render, root(), the from_world re-query, uses LifeCycleValues.is_terminal + node_label, and tests the tick.
- draft PRs in the fork (all checked): #42 NodeDetailRenderer (bridge subclasses body_name; re-run here with the
  PR2 overlay: test_plan 216 passed), #43 Plan.has_single_root + PlanHasNoSingleRoot/PlanIsNotATree (needs ORM
  regen), #44 DetectingMotion node result = detected Body (also drops PlanNode.perform's result assignment),
  #45 bug merged-action life cycle, #47 Plan.running_action stacked on #45, #46 ExecutorTickObserver +
  PlanCallback.on_motion_tick. CI: only demo jobs fail (garmi also on cram2 main; real_tracy launch env).
  None covers ExecutionType.REAL (chart runs in the giskard server). Gap from #45: actions with an execution
  boundary (PickUp/Place) stay NOT_STARTED -> proposed card task_c6e0feb3 (not started).

Progress 2026-10-27 (user: "pls make a cramera demo from this git@github.com:yxzhan/cram-vrb-lab.git"):
- clone ~/cram-vrb-lab, local branch cramera-demo 64f9fcf (not pushed, yxzhan's repo): cramera/handover_stage.py
  (world.usda + kitchen objects + MJCF-twin cabinet colliders + handle bodies + drawer slides flipped),
  garmi_apartment_environment.usda, garmi_kitchen_demo.py, demo_garmi_bowl_into_drawer.py, README.
- bundle ~/cramera_scenes/garmi_apartment (gate passes, 350 boxes, 20 free objects); recording
  ~/.cramera/scenes/garmi_apartment_garmi_bowl_into_drawer (164 s, bowl 4.5 mm, drawer 10.6 mm); not published.
  Screenshots ~/Pictures/GarmiApartment/. Viewer: launch.json "cramera-garmi-apartment" (port 8723).
- cramera PR AICOR-Solutions/cramera#78 (branch usd/apartment-materials-carried, ready, bug label, bound here):
  OmniPBR specular_level + preview-surface specular workflow -> KHR_materials_specular, OmniGlass reflection_color
  and cut-out, OmniPBR opacity texture -> ColorOpacityTexture. The export used it before merge (CRAMERA_V1 worktree).
- side session "cramera: carry the GARMI apartment's materials" was redirected by the user to cramera#77
  (old garmi_pick_place as a demo scene, merged); its branch usd/apartment-materials has no own commits.
- isaac-twin skill cramera.md section 6b (vault f3cc5be).

Next:
- after the user has seen garmi_apartment_garmi_bowl_into_drawer: publish it and garmi_apartment
  (demo_garmi_bowl_into_drawer.py --publish / scenes repo), on their OK; after cramera#78 merges, re-export
  garmi_apartment from main;
- side sessions 2026-10-27: bridge test_demo cause is in CRAM coraplex (ParkArmsAction/MoveTorsoAction drop the
  robot namespace), nothing changed, user to decide on a fork fix; laboratory MuJoCo 3.11 fix on cramera branch
  laboratory/mujoco-3-11-physics (f464911, 27d2cdb, not pushed), PR waits for the user's OK; isaac-twin Phase 10
  template ported to CRAM main (vault 385e267), cramera-stack doc updated (vault 6adf6cc);
- ask: commit the LabAutomationTwin cramera/ files (as whom); fix the warehouse closed_fingers text;
- the warehouse demo still runs on the old stack (CRAM fork, cramera_cram); porting it to CRAM main is open;
- push the CRAM commits a65855bda, 842034d49, 906c53d6b, 376f204d4, 41425c5ca (on the user's OK);
- open decisions handed to the user: `usd` extra deps (numpy/scipy/Pillow), CI usd-core step, SSAO sRGB fix
  changing every AO scene, exposure 0.72, glass blend approximation, test-only Pose methods;
- known gaps: no generic demo runner (helpers live in the twin's warehouse_demos.py); v1 live recordings estimate
  fps from wall clock (the demo rewrites 50 Hz); the warehouse bundle is 210 MB, recordings 227/303 MB.
