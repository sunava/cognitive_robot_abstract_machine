"""
The models and free objects of a world written into a bundle.
"""

from __future__ import annotations

import pytest

from cramera_bridge.exceptions import MissingMeshFile
from cramera_bridge.export.world_models import (
    ENVIRONMENT_MODEL_NAME,
    ObjectFiles,
    WorldModels,
)
from cramera_bridge.geometry import NumericPose, PublishedColor, PublishedScale
from cramera_bridge.robots import RobotParts, RobotRoster
from cramera_bridge.world_objects import WorldObjects
from cramera_protocol.bundle.files import BundleFile

from ..dataset.worlds import CUP_MESH

MESH_DIRECTORY = "test"
"""
The folder below ``meshes`` the tests write meshes to.
"""

# %% models


def test_free_objects_are_left_out_of_the_environment(kitchen, tmp_path):
    objects = WorldObjects(kitchen.world).objects()
    bundle = WorldModels(kitchen.world, tmp_path, MESH_DIRECTORY).write(objects, None)
    [environment] = bundle.models
    assert environment.name == ENVIRONMENT_MODEL_NAME
    assert environment.robot is False
    assert environment.links == len(kitchen.world.bodies) - len(objects) + 1
    assert (tmp_path / environment.urdf).is_file()
    assert bundle.robot is None


@pytest.fixture(scope="module")
def robot_bundle(robot_world, tmp_path_factory):
    directory = tmp_path_factory.mktemp("robot_bundle")
    models = WorldModels(robot_world.world, directory, MESH_DIRECTORY)
    return models.write([], robot_world.robot)


def test_a_robot_model_is_identified_and_posed(robot_world, robot_bundle):
    [robot_model] = [model for model in robot_bundle.models if model.robot]
    assert robot_model.identifier == RobotRoster.identifier(robot_world.robot)
    assert (
        robot_model.pose
        == NumericPose.of_entity(robot_world.world, robot_world.robot.root).published()
    )
    assert "pr2/torso_lift_joint" in robot_model.movable_joints


def test_the_selected_robot_carries_its_parts(robot_world, robot_bundle):
    parts = RobotParts(robot_world.world, robot_world.robot).descriptions()
    assert robot_bundle.robot.part_annotations == parts
    assert robot_bundle.robot.base_body == str(robot_world.robot.root.name)
    assert robot_bundle.robot.parts == {part.name: part.links for part in parts}


def test_the_robot_alone_leaves_no_environment(robot_bundle):
    assert all(model.robot for model in robot_bundle.models)


# %% objects


def spawn_of(world, body):
    return NumericPose.of_entity(world, body).published()


def test_a_single_box_is_written_as_a_box(kitchen, tmp_path):
    description = ObjectFiles(tmp_path).describe(
        kitchen.table, "table", spawn_of(kitchen.world, kitchen.table)
    )
    shape = kitchen.table.visual.shapes[0]
    assert description.box == PublishedScale(shape.scale).published()
    assert description.color == PublishedColor(shape.color).hexadecimal()
    assert description.mesh is None
    assert description.height == shape.scale.z


def test_a_single_mesh_is_copied_with_its_material_library(kitchen, tmp_path):
    key = str(kitchen.cup.id)
    description = ObjectFiles(tmp_path).describe(
        kitchen.cup, key, spawn_of(kitchen.world, kitchen.cup)
    )
    assert description.key == key
    assert description.identifier == str(kitchen.cup.name)
    assert description.mesh == f"{BundleFile.OBJECT_MESHES}/{key}/{CUP_MESH.name}"
    assert (tmp_path / description.mesh).read_bytes() == CUP_MESH.read_bytes()
    assert description.material_library == f"{BundleFile.OBJECT_MESHES}/{key}/cup.mtl"


def test_a_missing_mesh_file_is_an_error(kitchen, tmp_path):
    [mesh] = kitchen.cup.visual.shapes
    mesh.filename = str(tmp_path / "absent" / CUP_MESH.name)
    with pytest.raises(MissingMeshFile):
        ObjectFiles(tmp_path / "bundle").describe(
            kitchen.cup, str(kitchen.cup.id), spawn_of(kitchen.world, kitchen.cup)
        )


def test_other_shapes_become_one_combined_mesh(kitchen, tmp_path):
    description = ObjectFiles(tmp_path).describe(
        kitchen.milk, "milk", spawn_of(kitchen.world, kitchen.milk)
    )
    cylinder = kitchen.milk.visual.shapes[0]
    assert description.mesh.endswith(".obj")
    assert (tmp_path / description.mesh).is_file()
    assert description.height == pytest.approx(cylinder.height)
    assert description.color == PublishedColor(cylinder.color).hexadecimal()


def test_a_body_without_shapes_gets_no_geometry_and_no_colour(kitchen, tmp_path):
    spawn = spawn_of(kitchen.world, kitchen.frame)
    description = ObjectFiles(tmp_path).describe(kitchen.frame, "frame", spawn)
    assert description.spawn == spawn
    assert (description.mesh, description.box, description.color) == (None,) * 3
    assert not tmp_path.joinpath(BundleFile.OBJECT_MESHES).exists()
