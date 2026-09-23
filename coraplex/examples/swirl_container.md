# Swirl a held container

`SwirlingAction` mixes a container's contents by keeping a pivot near its mouth or
current grip fixed and moving its bottom around a circle. It preserves the existing
gripper-to-container transform, so an existing front grasp remains a front grasp.
The container's local positive Z axis must point toward its mouth. Acquire the
container, hold it upright and move it clear of the surroundings before swirling.

```python
from coraplex.datastructures.enums import Arms
from coraplex.datastructures.swirling import SwirlProfile
from coraplex.robot_plans.actions.composite.swirling import SwirlingAction

swirl = SwirlingAction(
    container=held_vial,
    arm=Arms.RIGHT,
    profile=SwirlProfile(tilt_angle=0.16, cycles=3.0, duration=12.0),
)
```

Add this action to a plan with its existing world and robot context. Its motion uses
continuous Giskard Cartesian position and rotation constraints, a tilt-angle
inequality, and Cartesian speed constraints. A rigid grasp must already exist in the
CRAM world. The cone is measured from the container's initial upright axis; the
container returns to its initial pose after smoothly ramping down the circular motion.
Trajectory progress pauses when the gripper cannot keep up, so an obstructed arm
cannot finish the action merely by waiting out its duration.

The default pivot is the upper-rim center. Pass `pivot=Point3(...)` to choose a grip
point in the container frame. `SwirlProfile.from_radius(bottom_radius=...,
pivot_to_bottom=...)` derives the tilt from a desired bottom-circle radius, both in
meters. `SwirlTrajectory.pose_at(elapsed)` exposes the same geometry to simulation
adapters without depending on Giskard or MuJoCo.

The action specifies a physical container motion. The liquid model or real liquid
determines the mixing result; the action does not certify chemical homogeneity.

The CRAMERA laboratory demonstration shares `SwirlTrajectory` with this action and
tracks it through its MuJoCo contact controller. Running that browser demonstration
does not execute a complete Coraplex plan or Giskard solver. Running `SwirlingAction`
in a Coraplex plan builds the Giskard constraints described above.
