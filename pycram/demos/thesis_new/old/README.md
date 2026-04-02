# thesis_new

This folder contains small demos and helper modules for phase-based motion
profiles (spiral, shear, sweep) and their visualization in RViz or matplotlib.

## Quick start

- RViz demo with a bowl-constrained sequence:
  - `python pycram/demos/thesis_new/Phasenbausteineinwelt.py`
- Matplotlib plots of profiles and sequences:
  - `python pycram/demos/thesis_new/plot_phases.py`
- Simple validation of bowl-constrained sampling:
  - `python pycram/demos/thesis_new/test_bowl_block.py`

## Key modules

- `motion_models.py`
  - Core data types: `Pose`, `MotionSegment`, `MotionSequence`, and sampling logic.
- `motion_profiles.py`
  - Local curve functions (spiral, sweep, shear) and constraint helpers.
- `motion_presets.py`
  - Preset sequences. `build_container_sequence(...)` sizes curves from the
    object's AABB. The default `reference_size=0.10` scales durations.
  - It uses the **visual AABB** by default (`use_visual_aabb=True`) and
    can apply `shape.scale` via `apply_shape_scale=True`.
- `frame_provider.py`
  - Frame providers to map local curves into world/root frames.
- `rviz.py`
  - RViz marker publisher for phase sequences.
- `world_utils.py`
  - AABB helpers, semantic sampling, and pose conversions.

## Demos and scripts

- `Phasenbausteineinwelt.py`
  - Main RViz demo. Builds a world, publishes phase sequences, prints AABB.
- `plot_phases.py`
  - Matplotlib plots for profiles, sequences, and bowl-constrained curves.
- `test_bowl_block.py`
  - Samples a bowl sequence and checks that points stay inside the AABB.
- `demo_robot.py`
  - Robot setup example with semantic annotations.
- `Phasenbausteine.py`
  - Standalone notebook-like script with local plots and examples.
- `cores.py`
  - Small standalone math demo for displacement/oscillation.

## Notes on object sizing

- If the object looks larger in RViz but the AABB does not change, the
  collision geometry might not be scaled. In that case, use visual AABB
  (`use_visual_aabb=True`) until the collision mesh is fixed.
