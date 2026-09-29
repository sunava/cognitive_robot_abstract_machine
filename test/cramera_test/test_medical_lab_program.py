"""
Native mechanism sequencing observes physical state and preserves ownership.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import pytest
from typing_extensions import Any, Iterator

import cramera.medical_lab_program as program_module
from cramera.laboratory_physics import InvalidPhysicsTarget
from cramera.laboratory_physics_program import ProgramField, ProgramState
from cramera.laboratory_workflow_command import WorkflowField
from cramera.medical_lab_command import (
    InvalidMedicalLabCommand,
    MedicalLabAction,
    MedicalLabCommand,
)
from cramera.medical_lab_description import MedicalLabDescription
from cramera.medical_lab_dynamics import (
    MedicalLabCollision,
    MedicalLabDrives,
    MedicalPhysicsField,
)
from cramera.medical_lab_plan import (
    MedicalLabJointTarget,
    MedicalLabPlan,
    MedicalLabState,
    MoveInstrumentJointsAction,
)
from cramera.medical_lab_program import (
    MedicalDeviceState,
    MedicalLabProgram,
    MedicalProgramField,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World

from .test_medical_lab_dynamics import instrument_drives


# %% measured contact world and independent robot owner
@dataclass
class MeasuredMechanismWorld:
    """
    Integrate a real bounded hinge while an independent robot remains stationary.
    """

    instruments: MedicalLabDrives
    """Production joint importer and force-limited motor controller."""
    frozen: bool = False
    """
    Whether physical integration is held to exercise motion timeouts.
    """

    @property
    def model(self) -> mujoco.MjModel:
        """
        Compiled model used by detached native-world measurements.
        """
        return self.instruments.model

    @property
    def data(self) -> mujoco.MjData:
        """
        Measured state owned by the stepping thread.
        """
        return self.instruments.data

    @property
    def timestep(self) -> float:
        """
        Physical observation interval in seconds.
        """
        return float(self.model.opt.timestep)

    def step(self) -> None:
        """
        Advance motor demands and authoritative contact integration.
        """
        self.instruments.advance()
        if not self.frozen:
            mujoco.mj_step(self.model, self.data)

    def robot_state(self) -> dict[str, float]:
        """
        No physical robot joints are needed to observe this independent hinge.
        """
        return {}

    def snapshot(self) -> dict[str, Any]:
        """
        Publish measured instrument positions in the production snapshot shape.
        """
        return {
            MedicalPhysicsField.INSTRUMENTS: {
                MedicalPhysicsField.JOINTS: self.instruments.positions()
            }
        }

    def reset(self) -> None:
        """
        Explicitly restore the fixture's initial physical coordinates.
        """
        self.instruments.reset()


@dataclass
class ReservedRobotProgram:
    """
    Represent a separate robot controller with explicit workflow ownership.
    """

    state: ProgramState = ProgramState.IDLE
    """
    Lifecycle state retained by the independent robot controller.
    """

    workflow_reserved: bool = False
    """
    Whether a sample workflow still owns its samples and mechanisms.
    """

    def advance(self) -> None:
        """
        The stationary robot supplies no mechanism demands.
        """

    def observe(self) -> None:
        """
        The fixture has no robot acceptance conditions.
        """

    def snapshot(self) -> dict[str, Any]:
        """
        Publish only the robot state and workflow reservation contract.
        """
        return {
            ProgramField.STATE: self.state,
            WorkflowField.WORKFLOW: {WorkflowField.RESERVED: self.workflow_reserved},
        }

    def reset(self) -> None:
        """
        Release the old program only on explicit world reset.
        """
        self.state = ProgramState.IDLE
        self.workflow_reserved = False

    def pause(self) -> None:
        """
        Retain program ownership while stopping new robot intentions.
        """
        self.state = ProgramState.PAUSED


@dataclass
class IndependentMechanismProgram(MedicalLabProgram):
    """
    Exercise the production adapter without a recorded robot trajectory.
    """

    def __post_init__(self) -> None:
        """
        Provide a stationary owner instead of loading a PR2 reference file.
        """
        self.program = ReservedRobotProgram()


@pytest.fixture
def mechanism_program(
    instrument_drives: MedicalLabDrives,
    cylinder_bot_world: World,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[IndependentMechanismProgram]:
    """
    Use a native robot context with a small real MuJoCo articulation.
    """
    monkeypatch.setattr(program_module, "PR2", MinimalRobot)
    program = IndependentMechanismProgram(
        MeasuredMechanismWorld(instrument_drives), tmp_path
    )
    program.world = cylinder_bot_world
    yield program
    program.stop_program()


def step_until_state(
    program: MedicalLabProgram, state: MedicalLabState, seconds: float = 15
) -> None:
    """
    Advance bounded physical batches and yield to the native action worker.
    """
    for _ in range(int(seconds / program.timestep / 50)):
        program.step(50)
        if program.execution.state is state:
            return
        time.sleep(0)
    assert program.execution.state is state


# %% measured sequencing and pause lifecycle
def test_native_sequence_completes_only_after_both_physical_positions(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    The return action cannot start until the inspection joint is settled.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    joint = instrument.joints[0]
    program.run_devices(MedicalLabCommand(MedicalLabAction.RUN, instrument.id))
    first_request = program.execution.wait_for_request()
    program.step()
    assert program.execution.completed == 0
    assert program.physics.instruments.positions()[joint.name] != pytest.approx(
        joint.inspection
    )
    for _ in range(120):
        program.step(50)
        if program.execution.completed == 1:
            break
        time.sleep(0)
    assert program.execution.completed == 1
    assert program.physics.instruments.positions()[joint.name] == pytest.approx(
        joint.inspection, abs=program.physics.instruments.parameters.angular_tolerance
    )
    second_request = program.execution.wait_for_request()
    assert second_request != first_request
    assert second_request.targets[0].position == joint.rest
    step_until_state(program, MedicalLabState.COMPLETE)
    assert program.execution.completed == 2
    assert program.physics.instruments.reached({joint.name: joint.rest})
    assert (
        program.snapshot()[MedicalPhysicsField.INSTRUMENTS][
            MedicalProgramField.DEVICES
        ][0][MedicalProgramField.STATE]
        is MedicalDeviceState.REST
    )


def test_pause_retains_pending_action_and_resume_reaches_original_target(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    Pausing holds measured demand without teleportation or losing the action.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    joint = instrument.joints[0]
    program.run_devices(MedicalLabCommand(MedicalLabAction.OPEN, instrument.id))
    request = program.execution.wait_for_request()
    program.step(300)
    before = program.physics.instruments.positions()
    program.run_devices(MedicalLabCommand(MedicalLabAction.PAUSE))
    assert program.physics.instruments.positions() == before
    assert program.physics.instruments.targets == before
    assert program.reserved
    program.step(200)
    assert program.execution.completed == 0
    assert program.execution.pending() is None
    program.run_devices(MedicalLabCommand(MedicalLabAction.RUN, instrument.id))
    assert program.execution.pending() is request
    step_until_state(program, MedicalLabState.COMPLETE)
    assert program.physics.instruments.reached({joint.name: joint.inspection})


def test_reset_joins_paused_worker_without_assigning_measured_position(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    Device reset cancels native execution and applies a physical rest demand.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    program.run_devices(MedicalLabCommand(MedicalLabAction.OPEN, instrument.id))
    program.execution.wait_for_request()
    program.step(300)
    program.run_devices(MedicalLabCommand(MedicalLabAction.PAUSE))
    worker = program.execution.worker
    before = program.physics.instruments.positions()
    program.run_devices(MedicalLabCommand(MedicalLabAction.RESET))
    assert not worker.is_alive()
    assert program.physics.instruments.positions() == before
    assert program.physics.instruments.targets == {
        joint.name: joint.rest for joint in instrument.joints
    }
    assert program.execution.state is MedicalLabState.IDLE
    program.step(3000)
    assert program.physics.instruments.reached(program.physics.instruments.targets)


def test_obstructed_action_fails_without_starting_next_action(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    An unachievable measured endpoint produces a bounded native plan failure.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    program.physics.frozen = True
    program.run_devices(MedicalLabCommand(MedicalLabAction.RUN, instrument.id))
    program.execution.wait_for_request()
    step_until_state(program, MedicalLabState.FAILED)
    assert program.execution.completed == 0
    assert (
        program.physics.instruments.targets == program.physics.instruments.positions()
    )
    assert program.execution.error is not None


# %% exclusive control of the shared physical world
@pytest.mark.parametrize("state", [ProgramState.RUNNING, ProgramState.PAUSED])
def test_robot_program_ownership_blocks_new_mechanism_plan(
    mechanism_program: IndependentMechanismProgram, state: ProgramState
) -> None:
    """
    A paused robot may still hold a sample and cannot lose its world ownership.
    """
    program = mechanism_program
    program.program.state = state
    instrument = program.physics.instruments.description.instruments[0]
    with pytest.raises(InvalidMedicalLabCommand):
        program.run_devices(MedicalLabCommand(MedicalLabAction.OPEN, instrument.id))
    assert program.execution.state is MedicalLabState.IDLE


def test_workflow_reservation_blocks_device_reset(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    A separate paused native workflow must retain instrument control.
    """
    program = mechanism_program
    program.program.state = ProgramState.PAUSED
    program.program.workflow_reserved = True
    with pytest.raises(InvalidPhysicsTarget):
        program.run_devices(MedicalLabCommand(MedicalLabAction.RESET))


def test_active_mechanism_plan_blocks_starting_robot_recipe(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    The ownership check is symmetric when a mechanism native plan is active.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    program.run_devices(MedicalLabCommand(MedicalLabAction.OPEN, instrument.id))
    program.execution.wait_for_request()
    with pytest.raises(InvalidPhysicsTarget):
        program.run_mixing()


# %% mechanical mating collision ownership
def test_joint_mating_exclusion_keeps_external_contact_active() -> None:
    """
    Ignoring an articulated mounting seam must retain contacts with loose objects.
    """
    directory = Path(__file__).parent / "dataset" / "medical_lab"
    description = MedicalLabDescription.from_json(
        json.loads((directory / "description.json").read_text())
    )
    source = ElementTree.parse(directory / "hinge.urdf").getroot()
    root = ElementTree.Element("mujoco")
    ElementTree.SubElement(root, "compiler", angle="radian")
    world = ElementTree.SubElement(root, "worldbody")
    base = ElementTree.SubElement(world, "body", name="base")
    ElementTree.SubElement(base, "geom", type="box", size="0.1 0.1 0.1")
    lid = ElementTree.SubElement(base, "body", name="lid")
    ElementTree.SubElement(lid, "geom", type="box", size="0.1 0.1 0.1")
    loose = ElementTree.SubElement(world, "body", name="loose", pos="0.15 0 0")
    ElementTree.SubElement(loose, "freejoint")
    ElementTree.SubElement(loose, "geom", type="box", size="0.1 0.1 0.1")
    importer = MedicalLabCollision(description)
    importer.add_joint(lid, source.find("link[@name='lid']"), source.find("joint"))
    importer.complete(root)
    assert [
        (item.attrib["body1"], item.attrib["body2"])
        for item in root.findall("contact/exclude")
    ] == [("base", "lid")]
    model = mujoco.MjModel.from_xml_string(
        ElementTree.tostring(root, encoding="unicode")
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs = {
        frozenset(
            (
                model.body(model.geom_bodyid[contact.geom1]).name,
                model.body(model.geom_bodyid[contact.geom2]).name,
            )
        )
        for contact in data.contact[: data.ncon]
    }
    assert frozenset(("base", "lid")) not in pairs
    assert frozenset(("lid", "loose")) in pairs


# %% independently authored native actions
@dataclass(frozen=True)
class CustomMechanismPlan(MedicalLabPlan):
    """
    Supply an independent action through the ordinary native execution path.
    """

    requested_action: MoveInstrumentJointsAction = field(kw_only=True)
    """
    Caller-authored motion checked again by the physical owner.
    """

    def actions(self) -> list[MoveInstrumentJointsAction]:
        """
        Construct a fresh action description for each native plan expansion.
        """
        return [replace(self.requested_action)]


def start_custom_action(
    program: IndependentMechanismProgram, action: MoveInstrumentJointsAction
) -> None:
    """
    Launch one external native action without bypassing its physical handshake.
    """
    recipe = CustomMechanismPlan(
        program.physics.instruments.description, action.device, requested_action=action
    )
    [robot] = program.world.get_semantic_annotations_by_type(MinimalRobot)
    program.execution.start(recipe, program.world, robot)
    program.execution.wait_for_request()


@pytest.mark.parametrize("position", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_target_fails_native_plan_without_crashing_physics(
    mechanism_program: IndependentMechanismProgram, position: float
) -> None:
    """
    Externally authored actions cannot inject nonfinite physical coordinates.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    joint = instrument.joints[0]
    target = MedicalLabJointTarget(joint.name, position, joint.velocity)
    start_custom_action(
        program, MoveInstrumentJointsAction(instrument.id, (target,), instrument.label)
    )
    step_until_state(program, MedicalLabState.FAILED)
    assert program.execution.completed == 0
    assert (
        program.physics.instruments.targets == program.physics.instruments.positions()
    )


def test_outside_limit_action_target_fails_native_plan_without_crashing_physics(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    A modular native action must respect the same real joint limits as the UI.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    joint = instrument.joints[0]
    target = MedicalLabJointTarget(joint.name, joint.upper + 1.0, joint.velocity)
    start_custom_action(
        program, MoveInstrumentJointsAction(instrument.id, (target,), instrument.label)
    )
    step_until_state(program, MedicalLabState.FAILED)
    assert program.execution.completed == 0


def test_unknown_action_device_fails_native_plan_without_crashing_physics(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    Unknown resource ownership must be rejected at the physical action boundary.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    joint = instrument.joints[0]
    target = MedicalLabJointTarget(joint.name, joint.inspection, joint.velocity)
    start_custom_action(
        program, MoveInstrumentJointsAction("missing", (target,), instrument.label)
    )
    step_until_state(program, MedicalLabState.FAILED)
    assert program.execution.completed == 0


def test_lower_native_target_speed_limits_motor_demand(
    mechanism_program: IndependentMechanismProgram,
) -> None:
    """
    Caller-requested speed is a real actuator demand bound, not only a timeout hint.
    """
    program = mechanism_program
    instrument = program.physics.instruments.description.instruments[0]
    joint = instrument.joints[0]
    target = MedicalLabJointTarget(joint.name, joint.inspection, joint.velocity / 4)
    start_custom_action(
        program, MoveInstrumentJointsAction(instrument.id, (target,), instrument.label)
    )
    before = program.physics.instruments.demands[joint.name]
    program.step(100)
    assert program.physics.instruments.demands[joint.name] - before == pytest.approx(
        target.velocity * 100 * program.timestep
    )
