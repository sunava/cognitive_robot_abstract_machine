"""
Check independent MuJoCo contents driven by a native box trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from coraplex.plans.plan_node import ActionNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from cramera.workshop_contents import InvalidContentsCoupling, WorkshopContents
from cramera.workshop_physics import WorkshopPhysics
from cramera.workshop_world import WorkshopWorld
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstTrueNode,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF

from .test_workshop_bundle import workshop_directory
from .test_workshop_world import workshop_bundle
from .test_workshop_physics import place_initial_body

# %% real independent native and contact worlds


@pytest.fixture
def contents(workshop_bundle: Path) -> WorkshopContents:
    """
    Create matching native and MuJoCo models with independent screws.
    """
    specification = WorkshopWorld(bundle_directory=workshop_bundle)
    world = specification.create_world()
    physics = WorkshopPhysics(bundle_directory=workshop_bundle)
    return WorkshopContents(
        world=world,
        physics=physics,
        box=world.get_body_by_name("screw_box"),
        contents=tuple(world.get_body_by_name(key) for key in physics.geometry.screws),
        control_dt=QPControllerConfig(target_frequency=20).control_dt,
    )


def relative_native_pose(contents: WorkshopContents, key: str) -> np.ndarray:
    """
    Read an independent part in the currently controlled native box frame.
    """
    return (
        np.linalg.inv(contents.box.global_pose.to_np())
        @ contents.world.get_body_by_name(key).global_pose.to_np()
    )


def move_box(
    contents: WorkshopContents, position: np.ndarray, rotation: Rotation
) -> None:
    """
    Supply the externally controlled box trajectory without touching its contents.
    """
    contents.box.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            *position, *rotation.as_quat(), reference_frame=contents.world.root
        )
    )


def test_tilted_contents_follow_contact_motion_in_the_native_box_frame(
    contents: WorkshopContents,
) -> None:
    """
    Transfer relative slip while preserving the native box and unrelated objects.
    """
    unrelated = {
        key: contents.world.get_body_by_name(key).global_pose.to_np().copy()
        for key in ("driver", "nail_box", "workshop_assembly")
    }
    contents.start()
    initial_box = contents.box.global_pose.to_np().copy()
    initial_part = relative_native_pose(contents, contents.physics.geometry.screws[0])
    for index in range(150):
        progress = min((index + 1) / 100, 1.0)
        position = initial_box[:3, 3] + np.asarray([0.0, -0.42, 0.17]) * progress
        move_box(contents, position, Rotation.identity())
        contents.advance(0.02)
    for index in range(125):
        rotation = Rotation.from_euler("x", 0.48 * min((index + 1) / 70, 1.0))
        move_box(contents, position, rotation)
        native_box = contents.box.global_pose.to_np().copy()
        contents.advance(0.02)
        np.testing.assert_array_equal(contents.box.global_pose.to_np(), native_box)
    relative = relative_native_pose(contents, contents.physics.geometry.screws[0])
    assert np.linalg.norm(relative[:3, 3] - initial_part[:3, 3]) > 0.01
    for body in contents.contents:
        assert isinstance(body.parent_connection, Connection6DoF)
        assert body.parent_connection.parent is contents.box
        physics_box = contents.physics.pose(str(contents.box.name))
        physics_body = contents.physics.pose(str(body.name))
        measured_relative = (
            Rotation.from_quat(physics_box[3:])
            .inv()
            .apply(physics_body[:3] - physics_box[:3])
        )
        np.testing.assert_allclose(
            relative_native_pose(contents, str(body.name))[:3, 3],
            measured_relative,
            atol=1e-9,
        )
    for key, before in unrelated.items():
        np.testing.assert_array_equal(
            contents.world.get_body_by_name(key).global_pose.to_np(), before
        )
    assert contents.measurements.maximum_box_position_lag > 0.001
    assert contents.measurements.ticks == 275
    assert contents.measurements.model == "one-way-relative-contact-contents"


def test_stopping_leaves_free_contents_for_later_native_pickup(
    contents: WorkshopContents,
) -> None:
    """
    Complete contact updates at the retrieval boundary and never overwrite a grasp.
    """
    contents.start()
    contents.advance(0.05)
    contents.stop(settle_seconds=0.1)
    before = {
        str(body.name): body.global_pose.to_np().copy() for body in contents.contents
    }
    integration_time = contents.physics.data.time
    with contents.world.modify_world():
        contents.world.move_branch_with_fixed_connection(
            contents.contents[0], contents.box
        )
    contents.advance(0.05)
    assert not contents.active
    assert contents.physics.held is None
    assert contents.physics.data.time == integration_time
    for body in contents.contents:
        np.testing.assert_array_equal(body.global_pose.to_np(), before[str(body.name)])


def test_attached_contents_are_rejected_before_any_native_write(
    contents: WorkshopContents,
) -> None:
    """
    Refuse the old containment attachment model instead of moving a parent branch.
    """
    with contents.world.modify_world():
        contents.world.move_branch_with_fixed_connection(
            contents.contents[0], contents.box
        )
    before = contents.world.state._data.copy()
    with pytest.raises(InvalidContentsCoupling):
        contents.start()
    np.testing.assert_array_equal(contents.world.state._data, before)
    assert contents.physics.data.time == 0.0


# %% plan lifecycle and controller integration time
@dataclass
class CarrierAction(ActionDescription):
    """
    Identify one retrieval action without executing a robot controller.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        Prevent this lifecycle-only fixture from becoming an executable plan.
        """
        raise NotImplementedError


def test_only_the_selected_action_activates_and_stops_contact_updates(
    contents: WorkshopContents,
) -> None:
    """
    Match action identity and integrate precisely one native control period.
    """
    selected = CarrierAction()
    contents.action = selected
    unrelated = ActionNode(designator=CarrierAction())
    observed = ActionNode(designator=selected)
    contents.on_start(unrelated)
    assert not contents.active
    contents.start()
    contents.on_start(observed)
    before = contents.physics.data.time
    contents.on_motion_tick(MotionStatechart())
    assert contents.physics.data.time - before == pytest.approx(contents.control_dt)
    contents.on_end(unrelated)
    assert contents.active
    contents.on_end(observed)
    assert not contents.active


@pytest.mark.parametrize("frequency", [20, 50])
def test_actual_statechart_uses_the_supplied_executor_period(
    contents: WorkshopContents, frequency: float
) -> None:
    """
    Follow an actual compiled statechart without assuming it owns its context.
    """
    context = MotionStatechartContext(
        world=contents.world,
        qp_controller_config=QPControllerConfig(target_frequency=frequency),
    )
    statechart = MotionStatechart()
    motion = ConstTrueNode()
    statechart.add_nodes([motion, EndMotion.when_true(motion)])
    executor = Executor(context=context)
    contents.control_dt = context.qp_controller_config.control_dt
    contents.start()
    executor.compile(statechart)
    before = contents.physics.data.time
    executor.tick()
    contents.on_motion_tick(executor.motion_statechart)
    assert contents.physics.data.time - before == pytest.approx(
        context.qp_controller_config.control_dt
    )
    assert contents.measurements.ticks == 1


def test_loose_contents_belong_to_the_carried_branch_only_during_retrieval(
    contents: WorkshopContents,
) -> None:
    """
    Include free contents in the carrier footprint and preserve poses on release.
    """
    for body in contents.contents:
        assert body.parent_connection.parent is contents.world.root
    contents.start()
    descendants = contents.world.compute_child_kinematic_structure_entities(
        contents.box
    )
    for body in contents.contents:
        assert body in descendants
        assert body.parent_connection.parent is contents.box
        assert isinstance(body.parent_connection, Connection6DoF)
    contents.advance(contents.control_dt)
    before_release = {
        str(body.name): body.global_pose.to_np().copy() for body in contents.contents
    }
    contents.stop(settle_seconds=0.0)
    for body in contents.contents:
        assert body.parent_connection.parent is contents.world.root
        assert isinstance(body.parent_connection, Connection6DoF)
        np.testing.assert_allclose(
            body.global_pose.to_np(), before_release[str(body.name)], atol=1e-12
        )


def test_escaped_screw_rejects_retrieval_completion(contents: WorkshopContents) -> None:
    """
    Prevent a successful recording when a loose screw leaves the carrier cavity.
    """
    contents.start()
    screw = contents.contents[0]
    escaped = contents.physics.pose(str(screw.name))
    escaped[0] += 0.5
    place_initial_body(contents.physics, str(screw.name), escaped.tolist())
    with pytest.raises(InvalidContentsCoupling):
        contents.stop(settle_seconds=0.0)
    assert contents.measurements.escaped_contents == (str(screw.name),)
    assert not contents.measurements.retained_contents


def drawer_controller(contents: WorkshopContents) -> Executor:
    """
    Compile a real Cartesian goal on an actuated workshop drawer.
    """
    drawer = contents.world.get_body_by_name("workshop_drawer_1")
    target = drawer.global_pose.to_np().copy()
    target[1, 3] -= 0.04
    goal = CartesianPose(
        root_link=contents.world.root,
        tip_link=drawer,
        goal_pose=HomogeneousTransformationMatrix(
            target, reference_frame=contents.world.root
        ).to_pose(),
        translation_threshold=0.0005,
        orientation_threshold=0.003,
    )
    chart = MotionStatechart()
    chart.add_nodes([goal, EndMotion.when_true(goal)])
    executor = Executor(
        MotionStatechartContext(
            world=contents.world,
            qp_controller_config=QPControllerConfig(
                target_frequency=1.0 / contents.control_dt
            ),
        )
    )
    executor.compile(chart)
    return executor


def test_unprepared_action_cannot_mutate_a_compiled_cartesian_world(
    contents: WorkshopContents,
) -> None:
    """
    Reject late carrier parenting before it invalidates an actual goal binding.
    """
    action = CarrierAction()
    contents.action = action
    executor = drawer_controller(contents)
    before = contents.world.state._data.copy()
    with pytest.raises(InvalidContentsCoupling):
        contents.on_start(ActionNode(designator=action))
    np.testing.assert_array_equal(contents.world.state._data, before)
    assert not contents.active
    assert contents.physics.data.time == 0.0
    executor.tick_until_end(timeout=200)
    assert executor.motion_statechart.is_end_motion()


def test_prepared_contents_keep_cartesian_observations_live(
    contents: WorkshopContents,
) -> None:
    """
    Advance independent contacts while a real controller reaches its pose goal.
    """
    action = CarrierAction()
    contents.action = action
    contents.start()
    executor = drawer_controller(contents)
    contents.on_start(ActionNode(designator=action))
    for _ in range(200):
        executor.tick()
        contents.on_motion_tick(executor.motion_statechart)
        if executor.motion_statechart.is_end_motion():
            break
    assert executor.motion_statechart.is_end_motion()
    drawer = contents.world.get_connection_by_name("workshop_drawer_1_joint")
    assert drawer.position == pytest.approx(0.04, abs=0.0005)
    assert contents.measurements.ticks > 1
