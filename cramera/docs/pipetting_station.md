# Pipetting station

The laboratory includes a stationary two-channel liquid dispenser, authored in Blender
and attached to the workbench with MuJoCo collision geometry. Independent amber and
teal reagent lines feed two adjacent nozzles above an open receiving cradle.

## Manual operation

1. Open the contact laboratory and select **Start / resume laboratory** if necessary.
2. Select an available glass in **Object**.
3. Under **Pipetting station**, select **Position selected glass**. Force targets lift
   the glass clear of the rack, approach in front of the station, and slide it under
   the nozzle. **Stop transfer** holds its current position if the path is blocked.
4. Choose **Reagent** and **Dose (ml)**, then select **Dispense dose**.
5. Read delivered volume and remaining reagent in the controls or instrument display.
   **Stop dosing** interrupts the pump and retains the already delivered liquid.
6. Use **Refill selected reagent** to replenish that reservoir deliberately.

New doses range from 0.1 to 5 ml. Each independent reservoir initially contains 50 ml.
The pump delivers 1 ml per simulation second. It requires an aligned upright open
glass, sufficient receiver capacity, and sufficient reagent. It stops if the receiver
moves out of alignment or becomes obstructed. Samples secured in the centrifuge cannot
be dosed. Manual volume editing is unavailable during delivery.

## PR2 demonstration

Select **Start pipetting demo** with the clear glass in A1 and the robot inactive.
Remove secured samples from the centrifuge first. PR2 uses the existing front grasp,
verifies opposing fingertip contacts, holds the receiver upright under the nozzle,
adds 1 ml of amber and 1 ml of teal reagent, and returns the glass to A1. Existing
contents are retained if the glass has capacity for both doses. The program does not
reset other samples. **Pause PR2** stops dosing as well as progression; **Resume
pipetting demo** continues the remaining quantity without repeating the delivered dose.

The robot trajectory uses the local contact controller and MuJoCo actuators. It is
not a newly recorded CRAM plan or a general-purpose autonomous pipetting planner.

## Simulation and rendering

Authoritative simulation time drives conservative reservoir-to-glass volume transfer.
Colors combine by transferred volume and receiver mass follows the existing
water-density model. The browser renders a continuous nozzle stream, measured progress,
and reservoir levels from server observations. The stream is a visual representation;
the station does not solve nozzle flow or reagent chemistry with CFD. Separate fixed
feed lines are modeled; disposable pipette tips and tip changes are not implemented.

The authored metadata lives in `scene.laboratory.pipetting` and is copied into
`scene.physics.pipetting` for the contact viewer. The protected endpoints are
`/api/laboratory/physics/robot/pipetting` for pump operations and
`/api/laboratory/physics/robot/pipette` for the robot recipe. Manual contact scenes use
`/api/laboratory/physics/pipetting`.

## Assets and validation

`cramera/tools/laboratory/pipetting.py` defines the shared station frames and open
cradle geometry. The Blender builder supports `--pipetting-only`; the resulting
`assets/pipetting_station.glb` is packaged with the laboratory resources.

Pytest covers dose validation, finite inventories, volume and color conservation,
receiver interlocks, authenticated API access, measured browser controls, instrument
rendering, and a complete physical PR2 journey including pause and resume.
