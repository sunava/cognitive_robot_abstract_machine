"""
URDF documents written from world bodies and copied from URDF files.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import pytest

from cramera_bridge.exceptions import (
    PathOutsideBundle,
    UnrepresentableJointLimits,
    UrdfSourceNotFound,
)
from cramera_bridge.export.assets import BundledAssets
from cramera_bridge.export.urdf import (
    ROOT_LINK,
    UrdfAttribute,
    UrdfBundle,
    UrdfDocument,
    UrdfElement,
    UrdfJointType,
)
from cramera_protocol.bundle.files import BundleFile
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from ..dataset.bundles import BUNDLED_CUP_URDF
from ..dataset.worlds import CUP_MESH

MESH_DIRECTORY = "test"
"""
The folder below ``meshes`` the tests write meshes to.
"""


def document(world, directory) -> UrdfDocument:
    return UrdfDocument(
        world=world,
        assets=BundledAssets(bundle_root=directory),
        mesh_directory=MESH_DIRECTORY,
    )


def joints_of(path):
    root = ElementTree.parse(path).getroot()
    return {
        joint.attrib[UrdfAttribute.NAME]: joint
        for joint in root.iter(UrdfElement.JOINT)
    }


def links_of(path):
    root = ElementTree.parse(path).getroot()
    return {
        link.attrib[UrdfAttribute.NAME]: link for link in root.iter(UrdfElement.LINK)
    }


# %% world bodies


def test_a_kitchen_round_trips_through_its_urdf(kitchen, tmp_path):
    bodies = kitchen.world.bodies_topologically_sorted
    report = document(kitchen.world, tmp_path).write(bodies, "kitchen")
    parsed = URDFParser.from_file(str(tmp_path / report.urdf), prefix="").parse()
    assert {str(body.name) for body in parsed.bodies} == {ROOT_LINK} | {
        str(body.name) for body in bodies
    }
    assert report.links == [ROOT_LINK] + [str(body.name) for body in bodies]


def test_free_bodies_become_floating_joints_with_streamed_poses(kitchen, tmp_path):
    report = document(kitchen.world, tmp_path).write(
        kitchen.world.bodies_topologically_sorted, "kitchen"
    )
    joints = joints_of(tmp_path / report.urdf)
    milk = str(kitchen.milk_connection.name)
    assert joints[milk].attrib[UrdfAttribute.TYPE] == UrdfJointType.FLOATING
    assert milk in report.posed_joints
    assert report.movable_joints == []


def test_a_body_without_shapes_gets_no_geometry(kitchen, tmp_path):
    report = document(kitchen.world, tmp_path).write(
        kitchen.world.bodies_topologically_sorted, "kitchen"
    )
    frame = links_of(tmp_path / report.urdf)[str(kitchen.frame.name)]
    assert list(frame) == []


def test_visuals_carry_the_native_colour(kitchen, tmp_path):
    report = document(kitchen.world, tmp_path).write([kitchen.milk], "milk")
    milk = links_of(tmp_path / report.urdf)[str(kitchen.milk.name)]
    rgba = milk.find("visual/material/color").attrib[UrdfAttribute.RGBA]
    assert [float(value) for value in rgba.split()] == list(
        kitchen.milk.visual.shapes[0].color.to_rgba()
    )


def test_a_mesh_is_copied_with_its_material_library(kitchen, tmp_path):
    report = document(kitchen.world, tmp_path).write([kitchen.cup], "cup")
    mesh = links_of(tmp_path / report.urdf)[str(kitchen.cup.name)].find(
        "visual/geometry/mesh"
    )
    bundled = tmp_path / mesh.attrib[UrdfAttribute.FILENAME]
    assert bundled.read_bytes() == CUP_MESH.read_bytes()
    assert (bundled.parent / "cup.mtl").is_file()
    assert report.missing == []


def test_a_body_whose_parent_is_left_out_is_fixed_at_its_world_pose(kitchen, tmp_path):
    report = document(kitchen.world, tmp_path).write([kitchen.milk], "milk")
    graft = joints_of(tmp_path / report.urdf)[f"{ROOT_LINK}_to_{kitchen.milk.name}"]
    assert graft.attrib[UrdfAttribute.TYPE] == UrdfJointType.FIXED
    assert (
        graft.find(UrdfElement.ORIGIN).attrib[UrdfAttribute.TRANSLATION]
        == "1.0 0.1 0.9"
    )


def test_bodies_sharing_a_name_get_distinct_links(kitchen, tmp_path):
    twin = Body(name=PrefixedName("milk", "kitchen"))
    with kitchen.world.modify_world():
        kitchen.world.add_connection(
            type(kitchen.table.parent_connection)(parent=kitchen.table, child=twin)
        )
    report = document(kitchen.world, tmp_path).write([kitchen.milk, twin], "milks")
    assert len(set(report.links)) == 3
    assert report.links[2] == f"{twin.name}__{twin.id.hex}"


def test_a_prismatic_joint_without_limits_is_refused(kitchen, tmp_path):
    slide = Body(name=PrefixedName("slide", "kitchen"))
    with kitchen.world.modify_world():
        kitchen.world.add_connection(
            PrismaticConnection.create_with_dofs(
                world=kitchen.world,
                parent=kitchen.table,
                child=slide,
                axis=Vector3.X(reference_frame=kitchen.table),
            )
        )
    with pytest.raises(UnrepresentableJointLimits):
        document(kitchen.world, tmp_path).write([kitchen.table, slide], "slide")


def test_an_unlimited_revolute_joint_is_continuous(kitchen, tmp_path):
    wheel = Body(name=PrefixedName("wheel", "kitchen"))
    with kitchen.world.modify_world():
        connection = RevoluteConnection.create_with_dofs(
            world=kitchen.world,
            parent=kitchen.table,
            child=wheel,
            axis=Vector3.Z(reference_frame=kitchen.table),
        )
        kitchen.world.add_connection(connection)
    report = document(kitchen.world, tmp_path).write([kitchen.table, wheel], "wheel")
    joint = joints_of(tmp_path / report.urdf)[str(connection.name)]
    assert joint.attrib[UrdfAttribute.TYPE] == UrdfJointType.CONTINUOUS
    assert joint.find(UrdfElement.LIMIT) is None
    assert report.movable_joints == [str(connection.name)]


def test_a_robot_keeps_its_joints_and_limits(robot_world, tmp_path):
    world = robot_world.world
    report = document(world, tmp_path).write(
        world.bodies_topologically_sorted, "pr2", identity_root=robot_world.robot.root
    )
    parsed = URDFParser.from_file(str(tmp_path / report.urdf), prefix="").parse()
    original = {
        str(connection.name): type(connection).__name__
        for connection in world.connections
    }
    written = {
        str(connection.name): type(connection).__name__
        for connection in parsed.connections
        if str(connection.name) in original
    }
    assert written == original
    assert "pr2/torso_lift_joint" in report.movable_joints


def test_a_document_cannot_be_written_outside_its_bundle(kitchen, tmp_path):
    with pytest.raises(PathOutsideBundle):
        document(kitchen.world, tmp_path / "bundle").write([kitchen.milk], "../escape")


# %% existing URDF files


def test_an_existing_urdf_gets_its_meshes_and_reports_missing_ones(tmp_path):
    report = UrdfBundle(tmp_path, MESH_DIRECTORY).write(BUNDLED_CUP_URDF, "cup")
    meshes = [
        mesh.attrib[UrdfAttribute.FILENAME]
        for mesh in ElementTree.parse(report.urdf).getroot().iter(UrdfElement.MESH)
    ]
    assert meshes[0].startswith(f"{BundleFile.MESHES}/{MESH_DIRECTORY}/")
    assert (tmp_path / meshes[0]).read_bytes() == CUP_MESH.read_bytes()
    assert report.missing == ["package://no_such_cramera_package/meshes/lid.stl"]
    assert report.links == ["base", "lid"]
    assert report.movable_joints == ["lid_hinge"]


def test_a_missing_urdf_is_reported(tmp_path):
    with pytest.raises(UrdfSourceNotFound):
        UrdfBundle(tmp_path, MESH_DIRECTORY).write(tmp_path / "none.urdf", "none")
