"""
Exercise the reusable laboratory example without a viewer or a server.
"""

from pathlib import Path

import numpy as np
import pytest

from cramera.examples.laboratory_world import ExampleRobot, LaboratoryExample
from cramera.laboratory_world import LaboratoryBody, LaboratorySlot
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_world import laboratory_bundle


# %% reusable example execution
def test_example_defaults_to_robot_free_world(laboratory_bundle: Path) -> None:
    """
    A headless laboratory exposes real bodies and frame-bound slot targets.
    """
    result = LaboratoryExample(bundle_directory=laboratory_bundle).load()

    assert result.context is None
    assert result.world.get_semantic_annotations_by_type(AbstractRobot) == []
    source = result.world.get_body_by_name(LaboratoryBody.CLEAR_TUBE)
    source_pose = result.laboratory.slot_pose(LaboratorySlot.A1, world=result.world)
    assert source_pose.reference_frame is result.world.root
    np.testing.assert_allclose(source.global_pose.to_np(), source_pose.to_np())


@pytest.mark.parametrize(
    ("choice", "annotation_type"),
    [(ExampleRobot.PR2, PR2), (ExampleRobot.HSRB, HSRB)],
)
def test_robot_example_creates_cram_context(
    laboratory_bundle: Path,
    choice: ExampleRobot,
    annotation_type: type[AbstractRobot],
) -> None:
    """
    Both supported robot examples bind a native CRAM context to their world.
    """
    example = LaboratoryExample(robot=choice, bundle_directory=laboratory_bundle)
    result = example.load()

    [robot] = result.world.get_semantic_annotations_by_type(annotation_type)
    assert result.context.world is result.world
    assert result.context.robot is robot
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), example.odom_T_robot_start.to_np()
    )
