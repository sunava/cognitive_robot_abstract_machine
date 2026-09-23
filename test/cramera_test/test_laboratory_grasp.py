"""
Laboratory grasps resolve the finger stop after reaching the glass.
"""

from __future__ import annotations

from pathlib import Path

from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from cramera.laboratory_demo import LaboratoryDemo
from cramera.laboratory_grasp import ContactGripperMotion
from cramera.laboratory_world import LaboratoryWorld
from giskardpy.motion_statechart.goals.templates import Sequence
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_world import laboratory_bundle


# %% execution stage and closing policy
def test_closing_chart_defers_contact_resolution() -> None:
    """
    Compiling a plan must not evaluate contact before the reach has executed.
    """
    closing = ContactGripperMotion(
        motion=GripperState.CLOSE,
        gripper=Arms.LEFT,
        object_designator=Body(name=PrefixedName("glass")),
    )
    chart = closing.motion_chart
    assert isinstance(chart, Sequence)
    assert chart.nodes == []


def test_laboratory_closure_is_resolved_after_the_reach(
    laboratory_bundle: Path,
) -> None:
    """
    The close stage must wait for the reached object pose before planning fingers.
    """
    demonstration = LaboratoryDemo(
        laboratory=LaboratoryWorld(bundle_directory=laboratory_bundle)
    )
    context = demonstration.build_context(demonstration.build_simulated_world())
    plan = demonstration.build_plan(context)
    pickup = plan.root.children[0].designator
    closing = pickup.create_grasp_motion()
    assert closing.motion is GripperState.CLOSE
    assert closing.gripper is Arms.LEFT
    assert isinstance(closing, MoveGripperMotion)
    assert closing.requires_individual_execution is True
    assert closing.object_designator is context.world.get_body_by_name(
        demonstration.laboratory.source_body_name
    )


def test_laboratory_closing_uses_a_finite_speed_and_precise_stop(
    laboratory_bundle: Path,
) -> None:
    """
    The glass-specific close stage receives the configured speed and precision.
    """
    demonstration = LaboratoryDemo(
        laboratory=LaboratoryWorld(bundle_directory=laboratory_bundle)
    )
    context = demonstration.build_context(demonstration.build_simulated_world())
    plan = demonstration.build_plan(context)
    pickup = plan.root.children[0].designator
    closing = pickup.create_grasp_motion()
    assert closing.finger_velocity == demonstration.finger_closing_velocity
    assert closing.joint_position_threshold == demonstration.finger_position_threshold
    assert closing.contact_clearance == demonstration.finger_contact_clearance
    assert closing.tolerate_stall is False
