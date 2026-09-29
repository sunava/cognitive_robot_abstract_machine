# Running the expanded medical laboratory

The protected laboratory URL keeps its existing scene name, `precision_lab_pr2_physics`. Its authored source is this independent `aicor_medical_lab_large` bundle.

From the CRAM checkout, select this source when starting CRAMERA:

```sh
CRAMERA_LABORATORY_BUNDLE="$PWD/cramera/src/cramera/resources/aicor_medical_lab_large" \
  .venv/bin/python -m cramera.server 8716 --no-browser
```

Start the PR2 physics session in the viewer. The **AICOR Medical Laboratory** panel offers workstation views, **PR2 sample demos**, and **Instrument mechanisms**.

- **PR2 sample demos** retain the calibrated preparation routines: retrieve the sample rack from the drawer, dispense, weigh, centrifuge and refrigerate, including robot handle actions. The separate instrument-handle demos can also be selected.
- **Instrument mechanisms** runs the selected instrument's inspection and return actions. Select all instruments to execute 24 actions across the 12 articulated models. The three passive accessories have no motor sequence.
- **Inspect position** and **Rest position** operate one selected instrument. **Pause** holds its measured position; resuming continues the pending native action. **Reset** cancels the mechanism plan and drives the mechanisms back to rest.
- **View station** moves the camera to a selected workstation. Existing detailed sample, fluid, scale and centrifuge controls remain available below the new panel.

The new mechanism recipe is `cramera.medical_lab_plan.MedicalLabPlan`. It builds modular `MoveInstrumentJointsAction` steps with `Context`, `sequential` and `simulated_robot_advanced`. `MedicalLabProgram` dispatches the pending action to force-limited MuJoCo drives and acknowledges it only after measured position and velocity settle. Obstructions cause a bounded failure rather than a successful animation. Robot and mechanism plans have exclusive execution ownership, including while paused.

These mechanism demonstrations actuate the devices; they do not make the PR2 manipulate every new instrument. Assay results, incubation temperature, PCR amplification and optical measurements are not simulated. Existing liquid and sample-processing models retain their documented scope.

## Resources and editing

Load `environment.urdf` for the composed room, or a `devices/<instrument>/model.urdf` for an independent resource. `scene.json` includes station cameras, placements, interaction frames, limits and efforts. `laboratory_large.blend` contains the assembled editable scene; `room/model.blend` contains the room architecture. The original standalone instrument sources are retained under `devices/`.

Generate geometry with `cramera/tools/laboratory/build_large_room.py`, compose the bundle with `cramera/tools/laboratory/large_bundle.py`, and render the assembled overview with `cramera/tools/laboratory/render_large_world.py`. Refer to each tool's arguments before rebuilding.

The asset permission notices apply to this bundle and its independent models. No Siemens assets are included.
