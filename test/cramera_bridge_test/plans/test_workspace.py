"""
The installed CRAM workspace as the catalog reads it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cramera_bridge.plans.arms import Arms
from cramera_bridge.plans.workspace import (
    STANDING_HEIGHTS,
    PlanWorkspace,
    RobotAnnotation,
    RobotAnnotations,
    RobotDescription,
    SurfaceAnnotations,
)
from cramera_bridge.exceptions import (
    RobotDescriptionUnavailable,
    UnknownWorkspaceClass,
)
from cramera_bridge.names import ANNOTATION_NAMES, ENUM_NAMES, ROBOT_NAMES
from cramera_bridge.scenes.multi_robot import RobotPlacement, RobotScene
from cramera_protocol.bundle.files import MeshFormat
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.mixins import HasCaseAsRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table

from ..dataset.bundles import BUNDLED_CUP_URDF


@pytest.fixture(scope="module")
def workspace() -> PlanWorkspace:
    """
    :return: The workspace this interpreter imports.
    """
    return PlanWorkspace()


def test_a_robot_with_two_arms_and_a_torso_offers_every_choice():
    assert RobotAnnotation(PR2).choices() == {
        ENUM_NAMES.identifier_of(Arms): [
            ENUM_NAMES.member_identifier(Arms, arm) for arm in Arms
        ],
        ENUM_NAMES.identifier_of(TorsoState): [
            ENUM_NAMES.member_identifier(TorsoState, state) for state in TorsoState
        ],
    }


def test_a_robot_without_a_torso_offers_no_torso_state():
    assert RobotAnnotation(Tracy).choices()[ENUM_NAMES.identifier_of(TorsoState)] == []


def test_a_robot_with_one_arm_offers_the_choice_coraplex_resolves_to_it():
    assert RobotAnnotation(Stretch).arms() == [
        ENUM_NAMES.member_identifier(Arms, Arms.BOTH)
    ]


def test_robots_without_a_model_description_are_not_offered():
    names = [annotation.__name__ for annotation in RobotAnnotations().annotations()]
    assert PR2.__name__ in names
    assert MinimalRobot.__name__ not in names
    assert names == sorted(names)


def test_the_catalog_robots_are_the_offered_annotations(workspace):
    assert [robot.name for robot in workspace.robots()] == [
        ROBOT_NAMES.identifier_of(annotation)
        for annotation in RobotAnnotations().annotations()
    ]
    assert [robot.label for robot in workspace.robots()] == [
        annotation.__name__ for annotation in RobotAnnotations().annotations()
    ]


def test_the_object_models_are_the_meshes_beside_coraplex(workspace):
    identifiers = [model.identifier for model in workspace.objects()]
    assert "milk.stl" in identifiers
    assert not any(
        identifier.endswith(MeshFormat.MATERIAL_LIBRARY) for identifier in identifiers
    )
    assert (workspace.object_directory() / "milk.stl").is_file()


def test_the_catalog_robots_name_their_choices(workspace):
    pr2 = next(
        robot
        for robot in workspace.robots()
        if robot.name == ROBOT_NAMES.identifier_of(PR2)
    )
    assert pr2.choices == RobotAnnotation(PR2).choices()


def test_a_robot_whose_root_is_its_pelvis_stands_that_high(workspace):
    heights = {robot.name: robot.standing_height for robot in workspace.robots()}
    assert heights[ROBOT_NAMES.identifier_of(UnitreeG1)] == 0.7923
    assert heights[ROBOT_NAMES.identifier_of(PR2)] == 0


def test_the_unitree_g1_at_its_standing_height_rests_on_the_floor():
    placement = RobotPlacement(
        namespace="g1",
        label="G1",
        robot_type=UnitreeG1,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=STANDING_HEIGHTS[ROBOT_NAMES.identifier_of(UnitreeG1)]
        ),
    )
    world = RobotScene([placement], "g1").build_world()
    robot = placement.resolve(world)
    lowest = min(
        body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .min_z
        for body in world.get_kinematic_structure_entities_of_branch(robot.root)
        if body.collision
    )
    assert lowest == pytest.approx(0, abs=1e-3)


def test_the_catalog_lists_the_discovered_actions(workspace):
    assert workspace.action_catalog() == workspace.discovered_actions.catalog()


def test_the_surfaces_are_concrete_annotations_without_the_mixins(workspace):
    surfaces = workspace.surface_types()
    assert ANNOTATION_NAMES.identifier_of(Table) in surfaces
    assert HasCaseAsRootBody.__name__ not in surfaces
    assert SurfaceAnnotations().named(ANNOTATION_NAMES.identifier_of(Table)) is Table
    assert SurfaceAnnotations().named(Table.__name__) is Table


def test_an_unknown_class_name_is_refused():
    with pytest.raises(UnknownWorkspaceClass):
        RobotAnnotations().named("R2D2")


# %% robot descriptions


class DescribedRobot:
    """
    Stands in for a robot annotation naming a description; a subclass of the workspace's
    robot base would join the catalog of every later test.
    """

    description = BUNDLED_CUP_URDF.as_uri()

    @classmethod
    def get_ros_file_path(cls) -> str:
        return cls.description


def described_by(description: str) -> type:
    """
    :param description: The reference the robot names.
    :return: A robot annotation naming it.
    """
    return type(
        DescribedRobot.__name__, (DescribedRobot,), {"description": description}
    )


def test_a_robots_description_is_written_with_its_meshes(tmp_path):
    robot = described_by(BUNDLED_CUP_URDF.as_uri())
    urdf = RobotDescription(robot).write(tmp_path)
    identifier = ROBOT_NAMES.identifier_of(robot)
    assert urdf == tmp_path / f"{identifier}.urdf"
    assert [path.name for path in (tmp_path / "meshes").rglob("*.obj")] == ["cup.obj"]


def test_a_description_that_is_not_installed_is_unavailable(tmp_path):
    robot = described_by("package://no_such_description/urdf/robot.urdf")
    with pytest.raises(RobotDescriptionUnavailable):
        RobotDescription(robot).write(tmp_path)


def test_a_description_that_is_no_urdf_is_unavailable(tmp_path):
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text("<mujoco/>", encoding="utf-8")
    with pytest.raises(RobotDescriptionUnavailable):
        RobotDescription(described_by(mjcf.as_uri())).write(tmp_path / "out")


def test_a_xacro_that_does_not_expand_is_unavailable(tmp_path):
    xacro = tmp_path / "robot.urdf.xacro"
    xacro.write_text(
        '<robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro">'
        '<xacro:include filename="$(find no_such_description)/robot.xacro"/></robot>',
        encoding="utf-8",
    )
    with pytest.raises(RobotDescriptionUnavailable):
        RobotDescription(described_by(xacro.as_uri())).write(tmp_path / "out")


def test_the_workspace_writes_a_robots_description_by_its_identifier(
    workspace, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        PR2, "get_ros_file_path", classmethod(lambda cls: BUNDLED_CUP_URDF.as_uri())
    )
    urdf = workspace.write_robot_description(ROBOT_NAMES.identifier_of(PR2), tmp_path)
    assert urdf == tmp_path / f"{ROBOT_NAMES.identifier_of(PR2)}.urdf"
