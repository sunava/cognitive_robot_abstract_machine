"""
A plan's demo against the CRAM workspace: building the plan's world and native actions,
performing it, and staging its world for the viewer.
"""

from __future__ import annotations

import json
import math
import signal

import pytest

from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from cramera.locations import DataLocations
from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import TaskFailed
from cramera_bridge.geometry import NumericPose
from cramera_bridge.plans.command import STOP_SIGNALS, PlanDemoCommand, StagedWorld
from cramera_bridge.plans.demo import (
    MEASUREMENT_SEPARATOR,
    PLACEMENT_ERROR,
    PLACEMENT_TOLERANCE,
    PlannedTaskDemo,
)
from cramera_bridge.plans.steps import StepActions
from cramera_bridge.plans.surfaces import SurfaceTargets
from cramera_bridge.plans.world import PlannedGraspable
from cramera_bridge.exceptions import AnnotationNotInWorld, SurfaceNotInWorld
from cramera_protocol.bundle.acceptance import AcceptanceMetrics
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.plan_builder import (
    DemoDocument,
    DemoOption,
    PlanDocument,
    PlanStep,
    StepConstraint,
)
from cramera_protocol.viewer import EnvironmentVariable
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.world_description.geometry import Scale

from ..dataset.bundles import DATASET

PLANS = DATASET / "plans"
"""
The plans and the environment bundle of the plan tests.
"""

PLAN_FILE = PLANS / "pick_and_place.json"
"""
The plan the tests run: one PR2 picks the milk up and puts it down, another stands.
"""

TABLE_SIZE = Scale(1.0, 0.6, 0.75)
"""
The extent of the table the surface tests put the milk on, in metres.
"""

LOOK_SWITCH = "look_at_operation_site"
"""
The switch the plan builder's rule for "look where it operates" sets, which a pick-up
lacks.
"""

TABLE_POSITION = (2.0, 0.0)
"""
Where the table stands on the floor, in metres.
"""


@pytest.fixture
def document() -> DemoDocument:
    """
    :return: The test plan on an empty floor.
    """
    return DemoDocument(
        plan=PlanDocument.from_json(json.loads(PLAN_FILE.read_text("utf-8")))
    )


@pytest.fixture
def locations(tmp_path, monkeypatch) -> DataLocations:
    """
    :return: Data locations below the test's directory, which demos read from the
        environment.
    """
    monkeypatch.setenv(EnvironmentVariable.DATA, str(tmp_path / "data"))
    return DataLocations(data=tmp_path / "data", shared_scenes=None)


@pytest.fixture
def demo(document, locations) -> PlannedTaskDemo:
    """
    :return: The test plan's demo.
    """
    return PlannedTaskDemo.of_document(document, port=0)


@pytest.fixture
def built(demo):
    """
    :return: The test plan's world and the context of its performing robot.
    """
    world = demo.build_world()
    return world, demo.build_context(world)


# %% world and actions


def test_every_robot_stands_under_its_namespace(document, built):
    world, context = built
    robots = world.get_semantic_annotations_by_type(PR2)
    assert sorted(robot.root.name.prefix for robot in robots) == [
        robot.identifier for robot in document.plan.robots
    ]
    assert context.robot.root.name.prefix == document.plan.selected_robot


def test_the_objects_start_where_the_plan_puts_them(document, demo, built):
    world, _ = built
    milk = document.plan.objects[0]
    position = NumericPose.of_entity(world, demo.bodies[milk.identifier]).position
    assert position == pytest.approx(milk.pose.to_json()[:3])


@pytest.fixture
def actions(demo, built) -> StepActions:
    """
    :return: Builds the native actions of the test plan's steps.
    """
    _, context = built
    return StepActions(context, demo.bodies, demo.discovered_actions)


def test_the_steps_become_their_native_actions(demo, built, actions):
    assert [type(actions.action(step)) for step in demo.steps()] == [
        ParkArmsAction,
        MoveTorsoAction,
        PickUpAction,
        PlaceAction,
        SetGripperAction,
        SetGripperAction,
    ]
    _, context = built
    assert actions.action(demo.steps()[0]).arms == context.robot.all_arms


def test_a_pick_up_takes_the_grasp_of_its_object(demo, built, actions):
    _, context = built
    pick_up = actions.action(demo.steps()[2])
    assert isinstance(pick_up.grasp, GraspCandidate)
    assert pick_up.grasp.graspable is PlannedGraspable.of(demo.bodies["milk"])
    assert pick_up.arm is context.robot.left_arm


def test_a_place_puts_down_the_object_it_names(demo, actions):
    place = actions.action(demo.steps()[3])
    assert place.object_designator is PlannedGraspable.of(demo.bodies["milk"])


def test_a_gripper_step_opens_and_closes_the_gripper(demo, built, actions):
    _, context = built
    close, open_ = (actions.action(step) for step in demo.steps()[4:])
    assert (close.motion, open_.motion) == (GripperState.CLOSE, GripperState.OPEN)
    assert close.gripper is context.robot.left_arm.end_effector


def test_a_switch_the_action_lacks_is_left_out(demo, actions):
    pick_up = demo.steps()[2]
    switch = LOOK_SWITCH
    pick_up.constraints = [StepConstraint(text="Look", switch=switch)]
    assert not hasattr(actions.action(pick_up), switch)


def test_a_navigation_drives_to_its_target(document, actions):
    navigate = document.plan.robots[1].steps[0]
    action = actions.action(navigate)
    assert isinstance(action, NavigateAction)
    assert action.target_location.position.to_np()[:3] == pytest.approx(
        navigate.values["target_location"][:3]
    )


def test_an_annotation_the_world_lacks_is_refused(actions):
    cutting = PlanStep(
        identifier="step_9",
        action="CuttingAction",
        values={"arm": "LEFT", "tool": "knife", "object_to_cut": "milk"},
    )
    with pytest.raises(AnnotationNotInWorld):
        actions.action(cutting)


def test_a_place_on_a_surface_the_world_lacks_offers_no_spot(actions):
    place = PlanStep(
        identifier="step_9",
        action="PlaceAction",
        values={
            "object_designator": "milk",
            "target_location": {"placement": "surface", "surfaceType": "Table"},
        },
    )
    with pytest.raises(SurfaceNotInWorld):
        actions.action(place)


def test_a_surface_the_world_lacks_offers_no_spot(demo, built):
    world, _ = built
    targets = SurfaceTargets(
        world=world, body=demo.bodies["milk"], surface_type="Table"
    )
    with pytest.raises(SurfaceNotInWorld):
        targets.first()


def test_a_spot_on_a_surface_rests_the_object_on_it(demo, built):
    world, _ = built
    table = BodySpecification.box("table", TABLE_SIZE).spawn(
        world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            *TABLE_POSITION, TABLE_SIZE.z / 2
        ),
    )
    with world.modify_world():
        world.add_semantic_annotation(Table(root=table))
    targets = SurfaceTargets(
        world=world, body=demo.bodies["milk"], surface_type="Table"
    )
    position = targets.first().position.to_np()
    assert position[2] > TABLE_SIZE.z + targets.bottom_offset() - PLACEMENT_TOLERANCE
    assert abs(position[0] - TABLE_POSITION[0]) <= TABLE_SIZE.x / 2


# %% naming and measuring


def test_a_plan_on_an_empty_floor_is_recorded_under_its_name(document, demo):
    assert demo.scene_name() == document.plan.name


def test_a_taken_name_gets_the_next_number(demo, locations):
    (locations.local_scenes / demo.name).mkdir(parents=True)
    assert demo.scene_name() == f"{demo.name}_2"


def test_an_environment_bundle_names_the_recording(document, locations):
    document.environment_directory = str(PLANS / "lab")
    demo = PlannedTaskDemo.of_document(document, port=0)
    assert demo.scene_name() == "lab_pr2_pr2_pick_and_place"
    assert demo.environment_name == "Test laboratory"


def test_an_object_far_from_its_pose_fails_the_task(demo, built):
    world, context = built
    milk = demo.bodies["milk"]
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        5.0, 5.0, 0.0, reference_frame=world.root
    )
    with pytest.raises(TaskFailed):
        demo.measure(context)


# %% running


def test_the_plan_is_performed_and_its_recording_saved(document, locations):
    saved = PlanDemoCommand(document).main([DemoOption.PORT, "0"])
    bundle = locations.local_scenes / saved.scene
    metrics = DocumentFile(bundle / BundleFile.ACCEPTANCE_METRICS).read(
        AcceptanceMetrics
    )
    error = metrics.measurements[
        MEASUREMENT_SEPARATOR.join((PLACEMENT_ERROR, "step_4"))
    ]
    assert saved.scene == document.plan.name
    assert (bundle / BundleFile.SCENE).is_file()
    assert math.isfinite(error) and error <= PLACEMENT_TOLERANCE


def test_a_stopped_stage_restores_the_signal_handlers(demo):
    handlers = {number: signal.getsignal(number) for number in STOP_SIGNALS}
    staged = StagedWorld(demo)
    staged.stopped.set()
    staged.run()
    assert {number: signal.getsignal(number) for number in STOP_SIGNALS} == handlers
