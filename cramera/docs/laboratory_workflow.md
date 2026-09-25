# Four-sample laboratory workflow

The full demo starts with the clear, amber, teal, and reserve samples in a rack
inside the closed bench drawer. PR2 opens the drawer, carries the complete rack to
the workbench, and closes the drawer before processing the samples through the dual
dispenser, balance, centrifuge, and sample cooler. Each initially empty vial receives
**1 ml of reagent A and 1 ml of reagent B**.

The native Coraplex recipe contains 64 independently executable actions, including
the three drawer-and-rack preparation actions.

## Access and controls

The existing password-protected laboratory address remains
[CRAMERA Laboratory](https://cramera.informatik.uni-bremen.de/laboratory/).
Use the existing laboratory password. Open the PR2 contact-physics scene and find
**Full laboratory demo** in the laboratory controls.
Remote browsers connected to that server operate the same simulation session.
The 3D viewer requires a browser with WebGL support.

| Control | Effect |
| --- | --- |
| **Prepare & run full demo** | Explicitly resets the simulated laboratory, robot, and instruments, places the rack with four empty samples inside the closed drawer, and starts a fresh run. Previous contents and run measurements are replaced. |
| **Pause full demo** | Pauses robot intentions and interrupts an active dose while preserving the quantity already delivered. Gravity, contact dynamics, cooling, and an already running centrifuge cycle continue. |
| **Resume full demo** | Continues the paused recipe with its recorded measurements and remaining dose. It does not prepare or empty the samples again. |
| **Stop dosing** / **Stop rotor** | Stops the selected instrument and pauses the workflow. An interrupted centrifuge cycle cannot count as a completed processing cycle. |
| **Reset laboratory** | Cancels the current run and restores the authored laboratory state. Select **Prepare & run full demo** to prepare four empty samples again. |

Running and paused workflows reserve their samples and instruments. Other robot
demos, manual sample movement, refilling, taring, and device configuration remain
disabled until the workflow finishes, fails, or is explicitly reset. Instrument stop
controls remain available. A failed operation reports its reason; neither a timeout
nor a progress bar reaching its end is sufficient evidence of success.

### Try the robot's lid and door actions separately

The **Centrifuge** and **Sample refrigerator** sections each provide **Robot: open**,
**Robot: close**, and **Robot: open & close**. These controls run the corresponding
native `OpenInstrumentAction` and `CloseInstrumentAction` with physical handle
contacts. They preserve the current laboratory, sample contents, and device state;
the complete processing recipe is not started. An already reached endpoint needs
no extra handle movement.

Each device shows its robot operation's status and any failure. Use **Pause PR2**
to pause; the selected device's action button becomes **Resume: …** and continues
the same action. A running or paused robot program
reserves the instruments, so another operation cannot replace it. The centrifuge
must be stopped before its lid can be manipulated. The ordinary **Open lid** /
**Close lid** and **Open door** / **Close door** controls continue to operate the
device motors directly.

The standalone robot acceptance tests cover a centrifuge close followed by an
open-and-close sequence and a refrigerator open-and-close sequence. Both require
measured handle contacts, zero hinge-motor force during turning, released fingers
at the endpoint, and no unintended robot contact with the surrounding furniture.

## Processing sequence

The preparation stage consists of `OpenInstrumentAction(DRAWER)`,
`RetrieveRackAction(required_sample_keys=...)`, and `CloseInstrumentAction(DRAWER)`.
The drawer motor is disabled while the gripper pulls or pushes its handle. The
carrier is grasped from above at `laboratory_rack_handle`, lifted clear of the open
drawer, transported in front of the workbench, and released onto the table. Lower
guide cups keep the rounded glass bottoms seated; their flared mouths permit
subsequent individual insertion. The rack and all four vials remain separate free
bodies, carried by physical contacts. Base sweeps include every carried body.

Samples are handled in the order **clear → amber → teal → reserve**. The front rack
positions are emptied first so the gripper can approach the remaining samples.
Filling, weighing, and loading are completed for one sample before the next begins.

1. PR2 approaches a vial with a front grasp and verifies opposing fingertip contact.
   It presents the upright vial under the dual dispenser and receives the two doses
   separately. The controller checks both metered delivery and the actual increase
   in the vial's liquid inventory.
2. The robot places the vial on the balance and releases it onto the support. The
   recipe waits for a stable reading attributable to that vial alone. The recorded
   measurement is **gross mass: vial plus liquid**, not liquid mass alone.
3. The robot moves the weighed vial to the free rack position A1 and releases it.
   The centrifuge lid opens for loading. From its supported position at A1,
   the vial is picked up near its upper rim with an inclined top grasp and placed
   vertically into a selected rotor holder. The holder is secured only after seated
   contact is measured. Opposing holders are selected using measured sample masses.
   These steps repeat for all four samples.
4. With four secured samples and a verified balanced load, the lid closes and the
   centrifuge starts. The current recipe requests 300 RPM, limited by the active
   instrument rules, for eight simulation seconds. The timer begins after the rotor
   reaches its accepted speed band. Processing is accepted only after the observed
   timed cycle finishes and the rotor has stopped; interlock or operator interruption
   does not satisfy that condition.
5. The centrifuge lid and cooler door open. Each holder is released in turn. PR2
   withdraws its sample with a top grasp and places it at A1 as a supported
   intermediate location. It then regrips from the front and transports the sample
   through the cooler's open doorway into its assigned storage position.
6. Each stored sample must remain upright with measured supporting contact. The
   cooler door and centrifuge lid close. The workflow only reports success when all
   four processed samples remain in their assigned cooler positions with both
   complete reagent doses retained.

A1 is used as a physical place to change grasp direction and is emptied before the
next sample is picked up. The sample is supported and released before the new grasp;
a change of grasp is not an invisible attachment switch. Top-grasp insertion uses
a slower descent through the rack aperture and still requires measured support
before release.

## Reading the observations

The panel shows the server's current stage, measured A/B doses, recorded gross mass,
and location for each sample. A dash means that the measurement has not yet been
confirmed. **PR2 transfer in progress** includes approach and pickup; it does not
claim that a grasp is already established. Stored locations refer to instrument
contact observations, rather than camera-based recognition.

The displayed `fill`, `weigh`, and `load_centrifuge` stages can alternate while the
four samples are processed. A category is only complete once the required evidence
exists for the entire batch. In particular, one filled or weighed sample does not
mean that all four are ready. Pausing preserves those observations, and resuming
continues from them.

## Physical model and limits

- **Manipulation:** the local controller drives force-limited MuJoCo arm and finger
  actuators. Contacts and friction carry the samples; the running recipe does not
  teleport loose objects to their destinations. Pickup, placement, and final
  supported release require measured evidence. Native Coraplex actions orchestrate
  this fixed recipe; their physical motion backend uses paths calibrated to this
  authored scene. Arm movement is not yet generated by a general-purpose Giskard
  planner, and robot self-collision is excluded from the model.
- **Rack:** the environment describes it with a standard floating URDF joint and
  a separate `assets/tube_rack.glb`. Its measured pose updates the viewer, TF frames,
  and native semantic world. Rack insertion targets follow that measured pose.
  Only explicit demo preparation initializes the rack and vial coordinates in the
  drawer; the running actions never assign their poses.
- **Mobile base:** bounded planar translation and heading joints move the PR2 base
  through force-limited actuators and contact dynamics. The model does not simulate
  wheel steering, tire slip, odometry, or autonomous navigation. Authored cabinet
  panels participate in contact physics. Before base travel, swept checks include
  the whole robot and any carried vial, and reject routes through the environment.
  The measured path is checked again while driving. Cabinet-bound transport first
  retreats into the clear aisle, then travels sideways, keeping the carrying arm
  clear of the centrifuge chamber.
- **Gaze:** a native Giskard `Pointing` constraint continuously follows the measured
  left gripper tool with the PR2's optical camera. It updates at 20 Hz in simulation
  time and controls only head pan and tilt; arm and base targets remain with the
  manipulation controller. The head uses physical servos, the original URDF joint
  and torque limits, and bounded command speed. If a direction cannot be reached,
  the joint limits take priority. Pause holds the last bounded head targets;
  resume continues from measured state without catching up elapsed pause time.
- **Lid and door:** dedicated opening and closing actions approach each handle,
  establish opposing fingertip contact, and drive its hinge through the physical
  hand. The instrument motor is disabled during that movement. At the measured
  endpoint, the hinge is held, the gripper releases, and the hand withdraws clear of
  the moving surface. The centrifuge's stopped-rotor interlock still applies.
  Handle manipulation uses a separate bounded gripper profile; completion,
  cancellation, and failure restore the original sample-grasp actuator settings.
  After a low door grasp, an explicit recovery stage restores torso height before
  the next bench-level sample transfer. Cartesian torso movement is enabled only
  by actual low placement stages, not by that preparatory joint-target stage.
  Starting its processing cycle uses the instrument command; physical button
  pressing is not part of this recipe.
- **Liquids:** a reduced model tracks finite reservoir volumes, transfer, mixing
  color, spills, and sample mass. The dispenser stream is rendered from observed
  delivery. The model does not solve full fluid flow, molecular mixing, reagent
  chemistry, or centrifugation-induced sediment separation. Liquid mass currently
  uses water density: 1 ml corresponds to 1 g.
- **Cooling:** the cooler's temperature is simulated by a simplified thermal model.
  Storage occupancy and door motion use the physical contact model; the display is
  not a validated prediction of a real sample's internal temperature.
- **Time:** motion, dosing, and cycle timing use simulation time. Their elapsed wall
  time depends on the server's simulation speed, including physical grasping,
  placement, and handle-manipulation waits. The validated complete recipe took
  about 35 simulation minutes with the current conservative movement speeds.

## Implementation and verification

The recipe is composed in
[`laboratory_plan.py`](../src/cramera/laboratory_plan.py) using a native Coraplex
`Context`, independent `ActionDescription` classes, `sequential(...)`, and a real
`Plan`. Execution calls `plan.perform()` inside an independent copy of the standard
`simulated_robot_advanced` context manager, so an enclosing caller's execution
environment is restored correctly.
The physical backend and measured observations are implemented in
[`laboratory_workflow.py`](../src/cramera/laboratory_workflow.py). Reusable sample
transfers live in
[`laboratory_sample_transfer.py`](../src/cramera/laboratory_sample_transfer.py),
head tracking in [`laboratory_gaze.py`](../src/cramera/laboratory_gaze.py), and the
browser panel in
[`laboratory-workflow.js`](../src/cramera/web/core/laboratory-workflow.js).

[`laboratory_actions.py`](../src/cramera/laboratory_actions.py) provides separate
transfer, dispensing, weighing, rotor assignment, load verification, securing,
release, centrifugation, opening, closing, and storage-verification actions.
Each action expands into a native execution boundary. Building or expanding a
plan does not move the robot. The boundary submits one immutable operation and
waits for its physical acceptance result; only then may the next action execute.
Native action status, failure, and start/end world records reflect these boundaries.

The server's simulation worker owns MuJoCo. The plan worker never changes its
state directly. Detached measured joints and object poses update the semantic
`Context.world` at action boundaries, including failed actions. Pausing retains
the same pending native action; resetting or stopping cancels and joins its waiting
execution. Expected physical interlock failures fail the active native action and
prevent subsequent actions while the contact simulation remains available.

Individual actions can run on the existing physical scene through
`LaboratoryWorkflowProgram.start_actions([...])`, without emptying samples or
running the full recipe. For example, `WeighSampleAction(sample_key)` records an
already supported specimen. Completion of such a subset does not claim that the
full sample batch has been processed and stored. `VerifyWorkflowAction` is the
explicit final postcondition of the complete recipe.

For example, the simulation owner can request just the two physical handle actions
on its existing `LaboratoryWorkflowProgram`:

```python
from cramera.laboratory_actions import (
    CloseInstrumentAction,
    LaboratoryInstrument,
    OpenInstrumentAction,
)

program.start_actions([
    OpenInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
    CloseInstrumentAction(LaboratoryInstrument.CENTRIFUGE),
])
```

The existing physics loop continues calling `advance()`, stepping MuJoCo, and
calling `observe()`. The request does not reset the laboratory. The program builds
a native sequential plan for these actions and waits at each physical boundary.

A transfer into the cooler selects an observed free storage position without
requiring a previous rotor assignment. Releasing a sample finds its actual secured
holder. `VerifyRotorBalanceAction()` checks the current physical load against the
active manual; the full recipe supplies `required_sample_keys` to additionally
require its entire weighed batch. Required state, such as an open door or an
assigned loading holder, remains an explicit precondition.

The authenticated workflow endpoint is
`POST /api/laboratory/physics/robot/workflow`. It requires exactly
`{"prepare": true}` for explicit preparation or `{"prepare": false}` for resume.
The shared gateway mounts this endpoint below `/laboratory/`. Progress is published
as `robot.workflow` alongside the normal robot state; the browser does not create
sample results locally.

Native context construction requires the standard PR2 ROS description package
`iai_pr2_description` to be discoverable through `AMENT_PREFIX_PATH`. The dedicated
Uni deployment keeps that model in `/srv/cramera-laboratory/ros`, separate from its
other CRAMERA service. Its Python environment must also provide the native
Coraplex and semantic-digital-twin packages used by the laboratory plan. The
laboratory service's dedicated semantic-digital-twin overlay is
`/srv/cramera-laboratory/semantic_digital_twin/src`: it supplies the committed URDF
parser that resolves meshes relative to each URDF and preserves explicit model
prefixes. Other entries use the existing server package through symlinks. This
keeps the separate CRAMERA service and its global dependencies unchanged. A remote
preflight builds the actual laboratory world and PR2 context, constructs the full
recipe, and executes a native simulated action before the laboratory restart.
This check runs as the backend service user inside its existing mount namespace;
`LD_LIBRARY_PATH` also includes `/opt/ros/jazzy/lib` for the native ROS imports.

`robot.workflow.plan` exposes the native plan lifecycle, current operation, and
completed action count. Physical measurements remain the authority for success.

`robot.workflow.gaze` reports the active `Pointing` constraint, measured target and
camera frame names, and the current optical-axis error in radians. It is `null`
until the first manipulation starts. The error is measured from MuJoCo frames,
not from the head controller's predicted positions.

Focused pytest checks passed for native action construction, independent action
reuse, execution lifecycle, measured world records, command validation, paused
ownership, doses, weights, balance/interlocks, and English browser controls.
Physical regressions cover swept base clearance, loaded centrifuge opening,
robot-driven door closure, torso recovery after door handling, and supported rack
insertion without losing fingertip contacts. Head-tracking tests cover moving
targets, pause/resume, joint limits, head-only commands, and measured optical-axis
error during actual dosing and weighing.

The drawer-and-rack integration test exercises the three new preparation actions
and the first individual transfer to the dispenser in one continuous simulation.
It checks measured support, upright empty samples, native action completion,
semantic handle poses, unintended contacts, and that controller calls do not
assign physical coordinates. Separate physical regressions cover pickup from all
four rack positions and supported insertion through the lower guide cups.
Drawer withdrawal must reach its measured retreat waypoint before the next
action; a force-based contact check covers subsequent torso recovery beneath the
worktop edge.

The full four-sample acceptance test additionally checks retained volumes,
measured weights, supported cooler occupants, released rotor holders, and both
closed covers after five seconds of settling. The earlier 61-action recipe passed
this full test at implementation `5b4d64b38`; that result predates the three new
drawer-and-rack actions and is not a fresh full-run validation of the 64-action
recipe. The dedicated deployment verifies committed package and scene hashes,
the native parsing environment, and the protected command boundaries separately.
