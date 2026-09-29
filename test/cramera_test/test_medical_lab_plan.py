"""
Expanded-laboratory mechanism demos remain native, bounded and measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from coraplex.datastructures.enums import TaskStatus
from coraplex.execution_environment import no_execution, simulated_robot_advanced
from cramera.laboratory_plan_world import LaboratoryPlanWorld, LaboratoryPoseMeasurement
from cramera.medical_lab_command import (
    InvalidMedicalLabCommand,
    MedicalLabAction,
    MedicalLabCommand,
)
from cramera.medical_lab_description import (
    InvalidMedicalLabDescription,
    MedicalLabDescription,
    MedicalLabInstrument,
    MedicalLabJoint,
    MedicalLabJointKind,
)
from cramera.medical_lab_plan import (
    MedicalLabContext,
    MedicalLabExecution,
    MedicalLabMotionFailed,
    MedicalLabMotionNode,
    MedicalLabMotionRequest,
    MedicalLabPlan,
    MedicalLabState,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot


# %% declarative model and inert executor
@pytest.fixture
def mechanism_description() -> MedicalLabDescription:
    """
    Describe a moving cover beside an independent non-actuated accessory.
    """
    return MedicalLabDescription(
        instruments=(
            MedicalLabInstrument(
                "tip_rack",
                "Tip rack",
                "preparation",
                (
                    MedicalLabJoint(
                        "tip_rack_lid_joint",
                        MedicalLabJointKind.REVOLUTE,
                        0,
                        1.2,
                        0,
                        1.5,
                        0.4,
                        2.0,
                    ),
                ),
            ),
            MedicalLabInstrument("ph_probe", "pH probe", "analysis", ()),
        )
    )


@dataclass
class RecordedMechanisms:
    """
    Record requested mechanism positions without moving a simulated robot.
    """

    requests: list[MedicalLabMotionRequest] = field(default_factory=list)
    """
    Operations observed during actual native plan execution.
    """

    reject: bool = False
    """
    Whether the physical operation deliberately fails.
    """

    def execute(self, request: MedicalLabMotionRequest) -> None:
        """
        Record a dispatch and optionally reject its physical completion.
        """
        self.requests.append(request)
        if self.reject:
            raise MedicalLabMotionFailed("Measured position not reached")


def test_command_schema_rejects_extra_fields_and_unknown_operations() -> None:
    """
    Remote callers cannot select arbitrary joint values or alter command shape.
    """
    for payload in (
        {"action": "run", "device": "all", "position": 4},
        {"action": "pause", "device": "all"},
        {"action": "open", "device": "all"},
        {"action": "unknown"},
        {"action": "run"},
    ):
        with pytest.raises(InvalidMedicalLabCommand):
            MedicalLabCommand.from_payload(payload)


@pytest.mark.parametrize(
    "action", [MedicalLabAction.RUN, MedicalLabAction.OPEN, MedicalLabAction.CLOSE]
)
def test_device_commands_round_trip(action: MedicalLabAction) -> None:
    """
    Exact supported per-device commands retain their authored selection.
    """
    payload = {"action": action.value, "device": "tip_rack"}
    command = MedicalLabCommand.from_payload(payload)
    assert command.action == action
    assert command.device == payload["device"]


def test_inspection_recipe_skips_static_accessories(
    mechanism_description: MedicalLabDescription,
) -> None:
    """
    A complete inspection visits each moving instrument and returns it to rest.
    """
    recipe = MedicalLabPlan(mechanism_description)
    actions = recipe.actions()
    joint = mechanism_description.instruments[0].joints[0]
    assert len(actions) == 2
    assert [action.device for action in actions] == ["tip_rack", "tip_rack"]
    assert [action.targets[0].position for action in actions] == [
        joint.inspection,
        joint.rest,
    ]


def test_static_accessory_cannot_claim_a_mechanism_demo(
    mechanism_description: MedicalLabDescription,
) -> None:
    """
    Selecting a passive accessory does not report a fabricated actuation.
    """
    with pytest.raises(InvalidMedicalLabCommand):
        MedicalLabPlan(mechanism_description, device="ph_probe").actions()


def test_metadata_cannot_target_another_instrument_or_robot() -> None:
    """
    Only a resource's own prefixed scalar joints may be actuated.
    """
    with pytest.raises(InvalidMedicalLabDescription):
        MedicalLabDescription.from_json(
            {
                "instruments": [
                    {
                        "id": "tip_rack",
                        "label": "Tip rack",
                        "station": "preparation",
                        "joints": [
                            {
                                "name": "pr2/head_pan_joint",
                                "type": "revolute",
                                "rest": 0,
                                "inspection": 1,
                                "lower": 0,
                                "upper": 2,
                                "velocity": 0.2,
                            }
                        ],
                    }
                ]
            }
        )


def test_metadata_rejects_unreachable_inspection_pose() -> None:
    """
    An inspection endpoint must respect the real bounded URDF joint travel.
    """
    with pytest.raises(InvalidMedicalLabDescription):
        MedicalLabDescription.from_json(
            {
                "instruments": [
                    {
                        "id": "tip_rack",
                        "label": "Tip rack",
                        "station": "preparation",
                        "joints": [
                            {
                                "name": "tip_rack_lid_joint",
                                "type": "revolute",
                                "rest": 0,
                                "inspection": 3,
                                "lower": 0,
                                "upper": 2,
                                "velocity": 0.2,
                            }
                        ],
                    }
                ]
            }
        )


def test_plan_expansion_and_no_execution_do_not_actuate(
    cylinder_bot_world, mechanism_description: MedicalLabDescription
) -> None:
    """
    Building and dry-running a recipe must leave every physical actuator alone.
    """
    recorder = RecordedMechanisms()
    context = MedicalLabContext(
        world=cylinder_bot_world,
        robot=cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
        executor=recorder,
    )
    plan = MedicalLabPlan(mechanism_description).build(context)
    plan.root.notify()
    plan.root.parse()
    assert recorder.requests == []
    assert (
        len(
            [
                node
                for node in plan.root.descendants
                if isinstance(node, MedicalLabMotionNode)
            ]
        )
        == 2
    )
    with replace(no_execution):
        plan.perform()
    assert recorder.requests == []


def test_native_actions_execute_in_order(
    cylinder_bot_world, mechanism_description: MedicalLabDescription
) -> None:
    """
    Measured native operations follow their explicit sequential recipe.
    """
    recorder = RecordedMechanisms()
    context = MedicalLabContext(
        world=cylinder_bot_world,
        robot=cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
        executor=recorder,
    )
    recipe = MedicalLabPlan(mechanism_description)
    plan = recipe.build(context)
    with replace(simulated_robot_advanced):
        plan.perform()
    assert recorder.requests == [action.request for action in recipe.actions()]
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_execution_handshake_waits_for_physical_acknowledgment(
    cylinder_bot_world, mechanism_description: MedicalLabDescription
) -> None:
    """
    A second native action cannot start before the first measured completion.
    """
    execution = MedicalLabExecution()
    recipe = MedicalLabPlan(mechanism_description)
    execution.start(
        recipe,
        cylinder_bot_world,
        cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
    )
    try:
        request = execution.wait_for_request(timeout=3)
        assert request == recipe.actions()[0].request
        assert execution.completed == 0
        assert execution.pending() == request
        execution.complete()
        assert execution.wait_for_request(timeout=3) == recipe.actions()[1].request
        assert execution.completed == 1
        execution.complete()
        execution.worker.join(timeout=3)
        assert not execution.worker.is_alive()
        assert execution.snapshot()["state"] == MedicalLabState.COMPLETE
        assert execution.completed == len(recipe.actions())
    finally:
        execution.cancel()


def test_physical_failure_prevents_following_action(
    cylinder_bot_world, mechanism_description: MedicalLabDescription
) -> None:
    """
    A reported physical failure fails the native plan instead of advancing it.
    """
    execution = MedicalLabExecution()
    recipe = MedicalLabPlan(mechanism_description)
    execution.start(
        recipe,
        cylinder_bot_world,
        cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
    )
    try:
        assert execution.wait_for_request(timeout=3) is not None
        execution.complete(error="Joint obstructed")
        execution.worker.join(timeout=3)
        assert not execution.worker.is_alive()
        assert execution.pending() is None
        assert execution.completed == 0
        assert execution.snapshot()["state"] == MedicalLabState.FAILED
        assert "Joint obstructed" in execution.snapshot()["error"]
    finally:
        execution.cancel()


def test_paused_native_action_retains_measured_acknowledgment(
    cylinder_bot_world, mechanism_description: MedicalLabDescription
) -> None:
    """
    Pausing cannot acknowledge a first action and silently start the next one.
    """
    execution = MedicalLabExecution()
    recipe = MedicalLabPlan(mechanism_description)
    execution.start(
        recipe,
        cylinder_bot_world,
        cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
    )
    try:
        first = execution.wait_for_request(timeout=3)
        execution.pause()
        assert execution.pending() is None
        execution.complete()
        with execution.condition:
            advanced = execution.condition.wait_for(
                lambda: execution.completed > 0, timeout=0.1
            )
        assert not advanced
        assert execution.active_request == first
        execution.resume()
        assert execution.wait_for_request(timeout=3) == recipe.actions()[1].request
        execution.complete()
        execution.worker.join(timeout=3)
        assert execution.state == MedicalLabState.COMPLETE
    finally:
        execution.cancel()


def test_reset_joins_a_paused_native_plan(
    cylinder_bot_world, mechanism_description: MedicalLabDescription
) -> None:
    """
    Reset releases the waiting action before a simulator may restore its joints.
    """
    execution = MedicalLabExecution()
    execution.start(
        MedicalLabPlan(mechanism_description),
        cylinder_bot_world,
        cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0],
    )
    assert execution.wait_for_request(timeout=3) is not None
    execution.pause()
    execution.cancel()
    assert not execution.worker.is_alive()
    assert execution.state == MedicalLabState.IDLE
    assert execution.completed == 0
    assert execution.pending() is None


def test_acknowledged_measurement_updates_the_next_action_world(
    cylinder_bot_world, mechanism_description: MedicalLabDescription
) -> None:
    """
    The next action sees detached measured state instead of assumed joint targets.
    """
    execution = MedicalLabExecution()
    robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]
    observation = LaboratoryPlanWorld(
        1.0,
        (),
        (
            LaboratoryPoseMeasurement(
                str(robot.root.name), (0.13, 0.07, 0), (0, 0, 0, 1)
            ),
        ),
    )
    execution.start(MedicalLabPlan(mechanism_description), cylinder_bot_world, robot)
    try:
        assert execution.wait_for_request(timeout=3) is not None
        execution.complete(measurement=observation)
        assert execution.wait_for_request(timeout=3) is not None
        assert np.asarray(
            robot.root.global_pose.to_position().to_np()[:3]
        ) == pytest.approx(observation.poses[0].position)
        execution.complete()
        execution.worker.join(timeout=3)
        assert execution.state == MedicalLabState.COMPLETE
    finally:
        execution.cancel()
