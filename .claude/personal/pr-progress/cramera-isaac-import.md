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

Next:
- push the two commits (on the user's OK);
- finish the demo recording, show it, publish to ~/cramera_scenes after the user has seen it;
- skill: Phase 9 + CRAM robot demo as the standard for every twin (cramera.md, SKILL.md, template), vault
  rsync + commit/push (new rule: commit and push the vault, author Vanessa Hassouna noreply);
- follow-up idea: move RollAction / grasp-point pick-up / clear into a tested cramera module instead of the
  template;
- open decisions handed to the user: `usd` extra deps (numpy/scipy/Pillow), CI usd-core step, SSAO sRGB fix
  changing every AO scene, exposure 0.72, glass blend approximation, test-only Pose methods;
- known gaps: TIAGo renders dark in the demo recording, no generic demo runner; the warehouse bundle is
  204 MB (no material or mesh sharing between identical stock items).
