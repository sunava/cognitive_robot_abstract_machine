# Recorded scene sources

Demo scripts recorded into `defense/scenes/<name>/` with the cramera onboarder.
The bundles themselves are generated artifacts (~150 MB each) and are not meant
to be edited by hand — re-record instead.

## pr2_cooking

A PR2 in the AICOR apartment kitchen performing three state-changing actions in
one continuous session, one arm, one tool at a time:

| phase | action | tool | object |
| ----- | ------ | ---- | ------ |
| separation and division | `CuttingAction`, slice technique, 3 cuts, 3 cm spacing | bread knife | bread on a cutting board |
| material transfer | `PouringAction` | cup | into the bowl |
| aggregation and mixing | `MixingAction` | whisk | in the bowl |

Record it with:

```bash
CRAMERA_SCENES=defense/scenes \
  .venv/bin/cramera-onboard defense/scenes_src/cooking_demo.py --name pr2_cooking
```

### What the recording is, and is not

- It is a **real execution** of the coraplex plans on the simulated robot: the
  joint trajectory in the bundle is what giskardpy produced, not keyframes.
- It runs under `simulated_robot`, i.e. **without collision avoidance**. With
  avoidance enabled (`simulated_robot_advanced`) the pour fails at the cup's
  approach against the cabinet-10 drawer handles — the same failure the
  standalone `experiments…simple_demo.demo_pouring` currently produces on this
  branch. Arms may therefore clip through kitchen geometry during the pour.
- The tool changes between phases are **not executed as actions**. There is no
  bimanual tool exchange in scope, so the script re-parents the tool between the
  counter and the gripper. In the recording a tool moves from the counter into
  the gripper without a grasp motion.
- All three tools are merged into the world *before* the first recorded tick.
  The onboarder binds its object set once at startup, so a tool that appears
  mid-run is invisible for the whole recording.

### Segments

The viewer's segment labels come from object attach/detach windows, which is a
transport-shaped heuristic: it labels this recording `move_big-knife` and
`move_whisk` rather than cut / pour / mix. The action list itself is recorded
correctly in `scene.json` (`actions`, `planTrees`).
