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

Next:
- commit + push the fixture fix and the import extension (on the user's OK);
- open decisions handed to the user: `usd` extra deps (numpy/scipy/Pillow), CI usd-core step, SSAO sRGB fix
  changing every AO scene, exposure 0.72, glass blend approximation, test-only Pose methods;
- known gaps: TIAGo renders dark in the demo recording, no generic demo runner; the warehouse bundle is
  204 MB (no material or mesh sharing between identical stock items).
