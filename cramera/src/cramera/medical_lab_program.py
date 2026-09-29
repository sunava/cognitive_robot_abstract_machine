"""Execute native instrument inspection plans against measured MuJoCo joints."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math

from typing_extensions import Any

from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_plan_world import LaboratoryPlanWorld
from cramera.laboratory_world import LaboratoryWorld
from cramera.medical_lab_command import (
    InvalidMedicalLabCommand,
    MedicalLabAction,
    MedicalLabCommand,
)
from cramera.medical_lab_dynamics import MedicalPhysicsField
from cramera.medical_lab_description import MedicalLabJointKind
from cramera.medical_lab_plan import (
    MedicalLabExecution,
    MedicalLabMotionRequest,
    MedicalLabPlan,
    MedicalLabState,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.shape_collection import ShapeCollection


# %% native plan ownership
class MedicalProgramField(StrEnum):
    """Measured mechanism feedback consumed by the laboratory panel."""

    DEMO = "demo"
    """Native sequence lifecycle and action progress."""
    DEVICES = "devices"
    """Per-resource pose state derived from measured joints."""
    ID = "id"
    """Independent model identifier."""
    LABEL = "label"
    """English resource name."""
    STATE = "state"
    """Measured mechanism pose."""


class MedicalDeviceState(StrEnum):
    """Observable instrument configurations."""

    STATIC = "static"
    """Accessory without actuated joints."""
    REST = "rest"
    """All joints have reached their rest positions."""
    INSPECTION = "inspection"
    """All joints have reached their inspection positions."""
    MOVING = "moving"
    """The measured configuration lies between the defined positions."""


@dataclass
class MedicalLabProgram(LaboratoryProgramPhysics):
    """Share one contact world between legacy robot plans and new mechanism plans."""

    execution: MedicalLabExecution = field(
        default_factory=MedicalLabExecution, init=False
    )
    """Native plan worker that never touches MuJoCo state."""
    active_request: MedicalLabMotionRequest | None = field(default=None, init=False)
    """Current physical action, retained until measured completion."""
    world: World | None = field(default=None, init=False)
    """Lazily built semantic context for the expanded authored environment."""
    elapsed: float = field(default=0.0, init=False)
    """Simulated seconds consumed by the current physical action."""
    deadline: float = field(default=0.0, init=False)
    """Measured motion timeout derived from requested travel and velocity."""
    settled: float = field(default=0.0, init=False)
    """Continuous time inside endpoint and velocity tolerances."""
    settling_duration: float = 0.15
    """Required settled observation period before acknowledging an action."""
    timeout_margin: float = 5.0
    """Additional simulated seconds allowed for physical settling."""

    @property
    def reserved(self) -> bool:
        """Whether an unfinished mechanism plan owns the laboratory controls."""
        return self.execution.state in (MedicalLabState.RUNNING, MedicalLabState.PAUSED)

    def run_devices(self, command: MedicalLabCommand) -> None:
        """Start or resume an authored inspection without inventing physical outcomes."""
        super()._require_workflow_release()
        if self.program.state in (ProgramState.RUNNING, ProgramState.PAUSED):
            raise InvalidMedicalLabCommand(
                "Finish or reset the PR2 program before operating instrument mechanisms"
            )
        if command.action is MedicalLabAction.PAUSE:
            self.execution.pause()
            self.physics.instruments.hold()
            self.active_request = None
            return
        if command.action is MedicalLabAction.RESET:
            self.execution.cancel()
            self.active_request = None
            self.physics.instruments.hold()
            self.physics.instruments.set_targets(
                {
                    name: joint.rest
                    for name, joint in self.physics.instruments.joints.items()
                }
            )
            return
        if (
            self.execution.state is MedicalLabState.PAUSED
            and command.action is MedicalLabAction.RUN
            and command.device == self.execution.device
        ):
            self.execution.resume()
            return
        if self.reserved:
            raise InvalidMedicalLabCommand(
                "Finish or reset the current instrument sequence first"
            )
        recipe = MedicalLabPlan(
            self.physics.instruments.description, command.device, command.action
        )
        recipe.actions()
        self.execution.cancel()
        self.execution = MedicalLabExecution()
        self.active_request = None
        if self.world is None:
            self.world = LaboratoryWorld(
                bundle_directory=self.physics.bundle_directory,
                allow_support_contacts=False,
            ).build()
            for body in self.world.bodies:
                body.visual = ShapeCollection()
        [robot] = self.world.get_semantic_annotations_by_type(PR2)
        self.execution.start(
            recipe, self.world, robot, LaboratoryPlanWorld.capture(self.physics)
        )

    def _advance_instruments(self) -> None:
        """Convert only the currently pending native action into motor demands."""
        request = self.execution.pending()
        if request is None:
            return
        if request is self.active_request:
            return
        permitted = {
            joint.name: joint
            for instrument in self.physics.instruments.description.instruments
            if instrument.id == request.device
            for joint in instrument.joints
        }
        names = [target.name for target in request.targets]
        valid = bool(names) and len(names) == len(set(names))
        valid = valid and all(
            target.name in permitted
            and isinstance(target.velocity, (int, float))
            and not isinstance(target.velocity, bool)
            and math.isfinite(target.velocity)
            and 0 < target.velocity <= permitted[target.name].velocity
            and isinstance(target.position, (int, float))
            and not isinstance(target.position, bool)
            and math.isfinite(target.position)
            and (
                permitted[target.name].kind == MedicalLabJointKind.CONTINUOUS
                or permitted[target.name].lower
                <= target.position
                <= permitted[target.name].upper
            )
            for target in request.targets
        )
        if not valid:
            self.physics.instruments.hold()
            self.execution.complete(
                error="Instrument targets must belong to their resource and respect authored drive speeds"
            )
            return
        self.active_request = request
        targets = {target.name: target.position for target in request.targets}
        self.physics.instruments.set_targets(
            targets, {target.name: target.velocity for target in request.targets}
        )
        measured = self.physics.instruments.positions()
        duration = max(
            abs(target.position - measured[target.name]) / target.velocity
            for target in request.targets
        )
        self.deadline = max(self.timeout_margin * 2, duration * 2 + self.timeout_margin)
        self.elapsed = 0.0
        self.settled = 0.0

    def _require_workflow_release(self) -> None:
        """Preserve mechanism ownership before any inherited robot or manual action."""
        if self.reserved:
            raise InvalidPhysicsTarget(
                "Finish or reset the instrument demonstration before operating the PR2"
            )
        super()._require_workflow_release()

    def run_workflow(self, prepare: bool) -> None:
        """Preserve mechanism ownership for workflow preparation and resumption."""
        if self.reserved:
            raise InvalidPhysicsTarget(
                "Finish or reset the instrument demonstration before operating the PR2"
            )
        super().run_workflow(prepare)

    def _observe_instruments(self) -> None:
        """Acknowledge measured endpoints or propagate a bounded motion failure."""
        if (
            self.active_request is None
            or self.execution.state is not MedicalLabState.RUNNING
        ):
            return
        targets = {
            target.name: target.position for target in self.active_request.targets
        }
        self.elapsed += self.timestep
        self.settled = (
            self.settled + self.timestep
            if self.physics.instruments.reached(targets)
            else 0.0
        )
        if self.settled >= self.settling_duration:
            self.execution.complete(
                measurement=LaboratoryPlanWorld.capture(self.physics)
            )
            self.active_request = None
        elif self.elapsed >= self.deadline:
            device = self.active_request.device
            self.physics.instruments.hold()
            self.execution.complete(
                error=f"{device}: a mechanism did not reach its measured target; inspect contacts before retrying",
                measurement=LaboratoryPlanWorld.capture(self.physics),
            )
            self.active_request = None

    def step(self, count: int = 1) -> None:
        """Interleave native action requests, bounded drives and physical observations."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Physics step count must be a positive integer")
        for _ in range(count):
            self._advance_instruments()
            super().step()
            self._observe_instruments()

    def reset(self) -> None:
        """Cancel native execution before restoring the explicit initial physical state."""
        self.execution.cancel()
        self.execution = MedicalLabExecution()
        self.active_request = None
        super().reset()

    def stop_program(self) -> None:
        """Join both native workers before the contact world is discarded."""
        self.execution.cancel()
        super().stop_program()

    def snapshot(self) -> dict[str, Any]:
        """Publish observed articulation independently of requested endpoints."""
        result = super().snapshot()
        devices = []
        for instrument in self.physics.instruments.description.instruments:
            if not instrument.joints:
                state = MedicalDeviceState.STATIC
            elif self.physics.instruments.reached(
                {joint.name: joint.rest for joint in instrument.joints}
            ):
                state = MedicalDeviceState.REST
            elif self.physics.instruments.reached(
                {joint.name: joint.inspection for joint in instrument.joints}
            ):
                state = MedicalDeviceState.INSPECTION
            else:
                state = MedicalDeviceState.MOVING
            devices.append(
                {
                    MedicalProgramField.ID: instrument.id,
                    MedicalProgramField.LABEL: instrument.label,
                    MedicalProgramField.STATE: state,
                }
            )
        result[MedicalPhysicsField.INSTRUMENTS].update(
            {
                MedicalProgramField.DEMO: self.execution.snapshot(),
                MedicalProgramField.DEVICES: devices,
            }
        )
        return result
