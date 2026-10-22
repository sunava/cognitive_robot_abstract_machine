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

Next:
- commit + push the fixture fix (on the user's OK);
- open decisions handed to the user: `usd` extra deps (numpy/scipy/Pillow), CI usd-core step, SSAO sRGB fix
  changing every AO scene, exposure 0.72, glass blend approximation, test-only Pose methods;
- known gaps: no collision geometry in bundles, stage rigid bodies are fixed links, TIAGo renders dark in the
  demo recording, no generic demo runner.
