Branch `claude/demo-recordings-scenes-meq6ri` — record the four tool-based demos
(cutting, pouring, mixing, wiping) as cramera scenes.

Plan
- Give cramera a one-command way to onboard several demos at once, since the
  onboarder ends its own process after each bundle.
- Cover it with stdlib-only tests (the CRAM stack is not importable here).

Done
- `cramera/src/cramera/onboard/demo_scenes.py`: `DemoScene` (demo file -> scene
  name, `demo_` prefix dropped), `SceneRecording`, `record_scenes`, and the
  `cramera-onboard-demos` entry point; README + pyproject wired; 14 tests in
  `test/cramera_test/test_demo_scenes.py`, all passing. Pushed, no PR opened.

Next
- The bundles themselves still have to be produced on a machine with the CRAM
  stack/ROS (this container has neither): run the command from the README and
  point `CRAMERA_SCENES` at the scenes checkout that should hold them.
