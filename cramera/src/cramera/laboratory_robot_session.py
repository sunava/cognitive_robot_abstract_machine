"""Own a physical PR2 and its native motion reference in one contact laboratory."""

from __future__ import annotations

import copy
import importlib
import json
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, ClassVar, Protocol

from cramera import paths
from cramera.laboratory_workflow_command import WorkflowCommand, WorkflowField
from cramera.laboratory_pipetting_command import PipettingAction, PipettingCommand
from cramera.laboratory_refrigerator_command import (
    RefrigeratorCommand,
    RefrigeratorField,
)
from cramera.laboratory_centrifuge_command import (
    CentrifugeAction,
    CentrifugeCommand,
    CentrifugeField,
)
from cramera.laboratory_physics_session import (
    ContactSimulation,
    InvalidPhysicsRequest,
    LaboratoryPhysicsSession,
    PhysicsField,
    PhysicsFile,
    PhysicsState,
)


# %% robot API contract
class RobotPhysicsRoute(StrEnum):
    """Isolated endpoints for the PR2 contact world."""

    START = "/api/laboratory/physics/robot/start"
    """Start the robot contact simulation."""
    STATE = "/api/laboratory/physics/robot/state"
    """Read measured joints, objects and execution state."""
    TARGET = "/api/laboratory/physics/robot/target"
    """Manipulate a loose object while the robot program is inactive."""
    LIQUID = "/api/laboratory/physics/robot/liquid"
    """Set a tube's liquid quantity while the robot program is inactive."""
    PIPETTING = "/api/laboratory/physics/robot/pipetting"
    """Operate the dispensing station beside the articulated robot."""
    PIPETTE = "/api/laboratory/physics/robot/pipette"
    """Run the robot demonstration with two measured reagent doses."""
    WORKFLOW = "/api/laboratory/physics/robot/workflow"
    """Prepare or resume the complete four-sample laboratory workflow."""
    CENTRIFUGE = "/api/laboratory/physics/robot/centrifuge"
    """Operate the centrifuge beside the articulated robot."""
    REFRIGERATOR = "/api/laboratory/physics/robot/refrigerator"
    """Operate the sample cabinet while the robot program is inactive."""
    MANUAL = "/api/laboratory/physics/robot/manual"
    """Read operating rules while the robot program is paused or inactive."""
    SCALE_TARE = "/api/laboratory/physics/robot/scale/tare"
    """Zero the laboratory scale while the robot program is paused or idle."""
    RELEASE = "/api/laboratory/physics/robot/release"
    """Remove the manual manipulation force and pause the robot reference."""
    RESET = "/api/laboratory/physics/robot/reset"
    """Restore the robot, objects and program."""
    STOP = "/api/laboratory/physics/robot/stop"
    """Terminate the robot contact worker."""
    RUN = "/api/laboratory/physics/robot/run"
    """Execute the saved native joint targets through physical actuators."""
    MIX = "/api/laboratory/physics/robot/mix"
    """Pour two source liquids, swirl the receiver and place it in A2."""
    PAUSE = "/api/laboratory/physics/robot/pause"
    """Pause advancement of the robot program."""


class RobotField(StrEnum):
    """Robot-specific scene and snapshot fields."""

    ROBOT = "robot"
    """Robot description or measured program status."""
    ROBOTS = "robots"
    """Optional robot catalog in the recorded scene."""
    MODELS = "models"
    """Models rendered in the laboratory."""
    URDF = "urdf"
    """Robot description relative to the generated bundle."""
    SOURCE = "source"
    """Origin of the motor setpoints."""
    MOVABLE_JOINTS = "movableJoints"
    """Visual joints updated from measured physical coordinates."""
    POSE = "pose"
    """Recorded model pose with XYZW orientation."""
    LINKS = "links"
    """Number of links in the renderer model."""


class ProgramSimulation(ContactSimulation, Protocol):
    """A contact engine that also advances a bounded robot program."""

    def run_program(self) -> None:
        """Start or resume the reference motor targets."""

    def run_mixing(self) -> None:
        """Start or resume the fixed two-liquid mixing recipe."""

    def run_pipetting(self) -> None:
        """Start or resume the stationary dispensing recipe."""

    def pause_program(self) -> None:
        """Hold the current robot program stage."""

    def run_workflow(self, prepare: bool) -> None:
        """Prepare fresh demo samples or resume the current processing workflow."""


def create_robot_physics(
    bundle_directory: Path, recording_directory: Path
) -> ProgramSimulation:
    """Load the robot engine and controller only when their scene is requested."""
    robot_module = importlib.import_module("cramera.laboratory_physics_robot")
    model = robot_module.LaboratoryRobotPhysics(
        bundle_directory=bundle_directory,
        robot_directory=recording_directory,
        robot_parameters=robot_module.RobotPhysicsParameters(mobile_base=True),
    )
    return importlib.import_module(
        "cramera.laboratory_physics_program"
    ).LaboratoryProgramPhysics(physics=model, recording_directory=recording_directory)


# %% isolated world and presentation
@dataclass
class LaboratoryRobotSession(LaboratoryPhysicsSession):
    """Serialize physical robot execution and independent manual object manipulation."""

    physics: ProgramSimulation | None = field(default=None, init=False)
    """The contact engine and its robot program."""
    recording_directory: Path | None = field(default=None, init=False)
    """Resolved native PR2 reference bundle."""

    RECORDING_SCENE: ClassVar[str] = "precision_lab_pr2"
    """Source of robot visuals and the native joint reference."""
    SCENE_NAME: ClassVar[str] = "precision_lab_pr2_physics"
    """Generated scene containing both robot and laboratory contacts."""
    VIEWER_URL: ClassVar[str] = f"/index.html?scene={SCENE_NAME}&layout=scene&offline=1"
    """Viewer isolated from the native robot live bridge."""
    ROBOT_ASSETS: ClassVar[str] = "robot"
    """Subdirectory preserving robot mesh-relative paths."""
    REFERENCE_SOURCE: ClassVar[str] = "recorded_cram_joint_targets"
    """Explicit provenance of the motor commands."""

    def _create_physics(self, directory: Path) -> ProgramSimulation:
        """Construct the robot contact world from the existing native reference."""
        self.recording_directory = paths.resolve_scene_directory(self.RECORDING_SCENE)
        if self.recording_directory is None:
            raise FileNotFoundError(
                "Run the native PR2 laboratory task first to generate the motion reference."
            )
        return create_robot_physics(
            bundle_directory=directory, recording_directory=self.recording_directory
        )

    def _write_scene(self, source_directory: Path) -> None:
        """Combine authored laboratory visuals with the physical robot's measured pose."""
        super()._write_scene(source_directory)
        robot_directory = self.output_directory / self.ROBOT_ASSETS
        shutil.copytree(
            self.recording_directory,
            robot_directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                *self.EXCLUDED_ASSETS, "trajectory.json", "acceptance_metrics.json"
            ),
        )
        source = json.loads((self.recording_directory / PhysicsFile.SCENE).read_text())
        scene_path = self.output_directory / PhysicsFile.SCENE
        scene = json.loads(scene_path.read_text())
        for item in source[RobotField.MODELS]:
            if not item.get(RobotField.ROBOT):
                continue
            robot = copy.deepcopy(item)
            robot[RobotField.URDF] = (
                Path(self.ROBOT_ASSETS) / robot[RobotField.URDF]
            ).as_posix()
            self._add_mobile_base(robot)
            scene[RobotField.MODELS].append(robot)
        for key in (RobotField.ROBOT, RobotField.ROBOTS):
            if key in source:
                scene[key] = source[key]
        scene[PhysicsField.TASK] = "PR2 · Contact physics"
        scene[PhysicsField.PHYSICS][RobotField.ROBOT] = True
        scene[PhysicsField.VALIDATION][RobotField.SOURCE] = self.REFERENCE_SOURCE
        scene_path.write_text(json.dumps(scene, indent=2))
        trajectory_path = self.output_directory / PhysicsFile.TRAJECTORY
        trajectory = json.loads(trajectory_path.read_text())
        snapshot = self.physics.snapshot()
        frame = dict(snapshot[PhysicsField.FRAMES])
        instrument = snapshot.get(CentrifugeField.CENTRIFUGE)
        if isinstance(instrument, dict):
            frame.update(instrument.get(CentrifugeField.JOINTS, {}))
        refrigerator = snapshot.get(RefrigeratorField.REFRIGERATOR)
        if isinstance(refrigerator, dict):
            frame.update(refrigerator.get(RefrigeratorField.JOINTS, {}))
        trajectory[PhysicsField.FRAMES] = [frame]
        trajectory_path.write_text(json.dumps(trajectory, indent=2))

    def _add_mobile_base(self, robot: dict[str, Any]) -> None:
        """Expose physical base coordinates only in the copied visualization model."""
        base_module = importlib.import_module("cramera.laboratory_robot_base")
        frames = self.physics.snapshot()[PhysicsField.FRAMES]
        if not all(joint in frames for joint in base_module.RobotBaseJoint):
            return
        base_module.RobotBaseDescription(
            self.output_directory / robot[RobotField.URDF],
            tuple(robot[RobotField.POSE][3:7]),
        ).write()
        robot.setdefault(RobotField.MOVABLE_JOINTS, []).extend(
            base_module.RobotBaseJoint
        )
        if RobotField.LINKS in robot:
            robot[RobotField.LINKS] += len(base_module.RobotBaseJoint)

    def _validate_target(self, key: str, position: list[float]) -> None:
        """Allow manual targets only when the robot program is not advancing."""
        self._require_workflow_release()
        super()._validate_target(key, position)
        if (
            self.physics.snapshot()[RobotField.ROBOT][PhysicsField.STATE]
            == PhysicsState.RUNNING
        ):
            raise InvalidPhysicsRequest(
                "Pause the PR2 program before moving objects manually."
            )

    def run_program(self) -> dict[str, Any]:
        """Transfer manipulation control from the cursor to the robot program."""
        with self.lock:
            self._require_running()
            self._require_workflow_release()
            self.physics.run_program()
            return self._status()

    def run_mixing(self) -> dict[str, Any]:
        """Transfer manipulation control to the fixed mixing recipe."""
        with self.lock:
            self._require_running()
            self._require_workflow_release()
            self.physics.run_mixing()
            return self._status()

    def run_pipetting(self) -> dict[str, Any]:
        """Transfer manipulation control to the stationary dispensing recipe."""
        with self.lock:
            self._require_running()
            self._require_workflow_release()
            self.physics.run_pipetting()
            return self._status()

    def run_workflow(self, command: WorkflowCommand) -> dict[str, Any]:
        """Serialize explicit demo preparation and state-preserving resume."""
        with self.lock:
            self._require_running()
            self.physics.run_workflow(prepare=command.prepare)
            return self._status()

    def _require_workflow_release(self) -> None:
        """Preserve workflow ownership during running and paused processing."""
        robot = self.physics.snapshot()[RobotField.ROBOT]
        if robot.get(WorkflowField.WORKFLOW, {}).get(WorkflowField.RESERVED, False):
            raise InvalidPhysicsRequest(
                "Resume or reset the sample workflow before changing its samples or instruments."
            )

    def _validate_pipetting(self, command: PipettingCommand) -> None:
        """Keep emergency pump stopping available during robot execution."""
        if command.action != PipettingAction.STOP:
            self._require_workflow_release()
        super()._validate_pipetting(command)
        if (
            command.action != PipettingAction.STOP
            and self.physics.snapshot()[RobotField.ROBOT][PhysicsField.STATE]
            == PhysicsState.RUNNING
        ):
            raise InvalidPhysicsRequest(
                "Pause the PR2 program before operating the pipetting station."
            )

    def _validate_liquid(self, key: str, volume_ml: float) -> None:
        """Require the robot program to pause before changing liquid quantities."""
        self._require_workflow_release()
        super()._validate_liquid(key, volume_ml)
        if (
            self.physics.snapshot()[RobotField.ROBOT][PhysicsField.STATE]
            == PhysicsState.RUNNING
        ):
            raise InvalidPhysicsRequest("Pause the PR2 program before filling vessels.")

    def _validate_centrifuge(self, command: CentrifugeCommand) -> None:
        """Keep rotor stopping available while the robot owns manipulation."""
        if command.action != CentrifugeAction.STOP:
            self._require_workflow_release()
        super()._validate_centrifuge(command)
        if (
            command.action != CentrifugeAction.STOP
            and self.physics.snapshot()[RobotField.ROBOT][PhysicsField.STATE]
            == PhysicsState.RUNNING
        ):
            raise InvalidPhysicsRequest(
                "Pause the PR2 program before operating the centrifuge."
            )

    def _validate_refrigerator(self, command: RefrigeratorCommand) -> None:
        """Require the robot program to yield before moving the refrigerator door."""
        self._require_workflow_release()
        if (
            self.physics.snapshot()[RobotField.ROBOT][PhysicsField.STATE]
            == PhysicsState.RUNNING
        ):
            raise InvalidPhysicsRequest(
                "Pause the PR2 program before operating the refrigerator."
            )

    def _validate_manual(self) -> None:
        """Preserve the active robot program's accepted operating knowledge."""
        self._require_workflow_release()
        if (
            self.physics.snapshot()[RobotField.ROBOT][PhysicsField.STATE]
            == PhysicsState.RUNNING
        ):
            raise InvalidPhysicsRequest(
                "Pause the PR2 program before reading another manual."
            )

    def _validate_scale(self) -> None:
        """Require the robot program to pause before zeroing the scale."""
        self._require_workflow_release()
        if (
            self.physics.snapshot()[RobotField.ROBOT][PhysicsField.STATE]
            == PhysicsState.RUNNING
        ):
            raise InvalidPhysicsRequest(
                "Pause the PR2 program before taring the scale."
            )

    def pause_program(self) -> dict[str, Any]:
        """Pause the reference while physics continues to calculate contacts."""
        with self.lock:
            self._require_running()
            self.physics.pause_program()
            return self._status()

    def _stop_physics(self) -> None:
        """Pause motor commands and remove manual forces after worker termination."""
        self.physics.pause_program()
        super()._stop_physics()
