# Four-sample laboratory workflow

The full demo processes the clear, amber, teal, and reserve samples through the dual
dispenser, balance, centrifuge, and sample cooler. Each initially empty vial receives
**1 ml of reagent A and 1 ml of reagent B**.

**Reference validation:** a fresh complete four-sample MuJoCo run passed, including
five seconds of settling after both doors closed. All four samples retained their
full 2 ml and occupied R1–R4; the rotor was stopped and all rotor holders were released.

## Access and controls

The existing password-protected laboratory address remains
[CRAMERA Laboratory](https://cramera.informatik.uni-bremen.de/laboratory/).
Use the existing laboratory password. Open the PR2 contact-physics scene and find
**Full laboratory demo** in the laboratory controls.
Remote browsers connected to that server operate the same simulation session.
The 3D viewer requires a browser with WebGL support.

| Control | Effect |
| --- | --- |
| **Prepare & run full demo** | Explicitly resets the simulated laboratory, robot, instruments, and sample poses, empties the four recipe samples, and starts a fresh run. Previous contents and run measurements are replaced. |
| **Pause full demo** | Pauses robot intentions and interrupts an active dose while preserving the quantity already delivered. Gravity, contact dynamics, cooling, and an already running centrifuge cycle continue. |
| **Resume full demo** | Continues the paused recipe with its recorded measurements and remaining dose. It does not prepare or empty the samples again. |
| **Stop dosing** / **Stop rotor** | Stops the selected instrument and pauses the workflow. An interrupted centrifuge cycle cannot count as a completed processing cycle. |
| **Reset laboratory** | Cancels the current run and restores the authored laboratory state. Select **Prepare & run full demo** to prepare four empty samples again. |

Running and paused workflows reserve their samples and instruments. Other robot
demos, manual sample movement, refilling, taring, and device configuration remain
disabled until the workflow finishes, fails, or is explicitly reset. Instrument stop
controls remain available. A failed operation reports its reason; neither a timeout
nor a progress bar reaching its end is sufficient evidence of success.

## Processing sequence

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
a change of grasp is not an invisible attachment switch.

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
  supported release require measured evidence. This is a fixed laboratory recipe,
  not a newly generated general-purpose CRAM or Giskard plan. Paths are calibrated
  to this authored scene, and robot self-collision is excluded from the model.
- **Mobile base:** bounded planar translation and heading joints move the PR2 base
  through force-limited actuators and contact dynamics. The model does not simulate
  wheel steering, tire slip, odometry, or autonomous navigation.
- **Gaze:** a native Giskard `Pointing` constraint continuously follows the measured
  left gripper tool with the PR2's optical camera. It updates at 20 Hz in simulation
  time and controls only head pan and tilt; arm and base targets remain with the
  manipulation controller. The head uses physical servos, the original URDF joint
  and torque limits, and bounded command speed. If a direction cannot be reached,
  the joint limits take priority. Pause holds the last bounded head targets;
  resume continues from measured state without catching up elapsed pause time.
- **Lid and door:** simulated joint actuators open and close the centrifuge lid and
  cooler door, with measured joint state and collision geometry. These operations
  do not currently represent the robot hand grasping a handle or pressing a button.
- **Liquids:** a reduced model tracks finite reservoir volumes, transfer, mixing
  color, spills, and sample mass. The dispenser stream is rendered from observed
  delivery. The model does not solve full fluid flow, molecular mixing, reagent
  chemistry, or centrifugation-induced sediment separation. Liquid mass currently
  uses water density: 1 ml corresponds to 1 g.
- **Cooling:** the cooler's temperature is simulated by a simplified thermal model.
  Storage occupancy and door motion use the physical contact model; the display is
  not a validated prediction of a real sample's internal temperature.
- **Time:** motion, dosing, and cycle timing use simulation time. Their elapsed wall
  time depends on the server's simulation speed. The reference complete run took
  about 31 minutes of simulation time and 48 minutes on the development machine.

## Implementation and verification

The recipe and its observations are implemented in
[`laboratory_workflow.py`](../src/cramera/laboratory_workflow.py). Reusable sample
transfers live in
[`laboratory_sample_transfer.py`](../src/cramera/laboratory_sample_transfer.py),
head tracking in [`laboratory_gaze.py`](../src/cramera/laboratory_gaze.py), and the
browser panel in
[`laboratory-workflow.js`](../src/cramera/web/core/laboratory-workflow.js).

The authenticated workflow endpoint is
`POST /api/laboratory/physics/robot/workflow`. It requires exactly
`{"prepare": true}` for explicit preparation or `{"prepare": false}` for resume.
The shared gateway mounts this endpoint below `/laboratory/`. Progress is published
as `robot.workflow` alongside the normal robot state; the browser does not create
sample results locally.

`robot.workflow.gaze` reports the active `Pointing` constraint, measured target and
camera frame names, and the current optical-axis error in radians. It is `null`
until the first manipulation starts. The error is measured from MuJoCo frames,
not from the head controller's predicted positions.

Focused pytest checks passed for command validation, paused ownership, measured
doses and weights, balance/interlock evidence, mobile and articulated motion, and
browser controls. The reference physical acceptance test passed from a fresh
reset through closed-door storage before head tracking was added. It checks actual
retained volumes and supported occupants, and verifies that controller intentions
and observations do not directly
change body or joint positions. Browser-control tests cover the English controls;
the embedded browser could not provide a WebGL context for visual inspection.

Head tracking has separate tests for moving targets and support frames, continued
tracking after initial alignment, pause/resume, head-only commands, joint limits,
and measured status. A physical integration run also completed the first sample's
two doses and supported weighing with gaze active, preserved both doses and the
measured gross mass, and kept the gripper within five degrees of the camera axis
during dosing and weighing. The complete four-sample run has not been repeated
with head tracking enabled.
