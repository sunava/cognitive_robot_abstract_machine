"""
Environment bundles built into worlds: meshes found beside their URDF, and the
annotations CRAM's world reasoner infers for an environment without prims.
"""

from __future__ import annotations

import shutil

from cramera_bridge.export.urdf_environment import UrdfEnvironmentBundle
from cramera_bridge.plans.workspace import PlanWorkspace
from cramera_bridge.scenes.bundle_world import BundleEnvironment
from cramera_bridge.scenes.inferred_annotations import InferredAnnotations
from cramera_protocol.live_messages import SurfaceDescription
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.world_description.geometry import Mesh

from ..dataset.bundles import DRAWER_CABINET_URDF
from ..dataset.worlds import CUP_MESH

MESH_URDF = """<robot name="shelf">
  <link name="shelf"><visual><geometry><mesh filename="meshes/cup.obj"/></geometry>
  </visual></link>
</robot>"""
"""
A URDF naming its mesh relative to itself.
"""


def test_a_bundled_urdf_finds_its_meshes_wherever_the_process_runs(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    (source / "meshes").mkdir(parents=True)
    shutil.copyfile(CUP_MESH, source / "meshes" / "cup.obj")
    (source / "shelf.urdf").write_text(MESH_URDF, encoding="utf-8")
    loaded = UrdfEnvironmentBundle(tmp_path / "scenes").write(
        source / "shelf.urdf", set()
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    world = BundleEnvironment(tmp_path / "scenes" / loaded.entry.name).create_world()
    [mesh] = [
        shape
        for body in world.bodies
        for shape in body.visual.shapes
        if isinstance(shape, Mesh)
    ]
    assert mesh.filename.startswith(str(tmp_path / "scenes" / loaded.entry.name))


def environment_world(tmp_path):
    """
    :return: The world of the drawer cabinet loaded as an environment.
    """
    loaded = UrdfEnvironmentBundle(tmp_path).write(DRAWER_CABINET_URDF, set())
    return BundleEnvironment(tmp_path / loaded.entry.name).create_world()


def test_an_environment_without_prims_gets_annotations_named_after_their_bodies(
    tmp_path,
):
    world = environment_world(tmp_path)
    assert [
        str(handle.name) for handle in world.get_semantic_annotations_by_type(Handle)
    ] == ["handle_cab1_drawer_top"]
    assert [
        str(drawer.name) for drawer in world.get_semantic_annotations_by_type(Drawer)
    ] == ["cabinet1_drawer_top"]


def test_an_annotation_rooted_outside_the_environment_is_removed():
    world = URDFParser.from_file(str(DRAWER_CABINET_URDF)).parse()
    environment = [
        body for body in world.bodies if body.name.name != "handle_cab1_drawer_top"
    ]
    kept = InferredAnnotations(world).add(environment)
    assert world.get_semantic_annotations_by_type(Handle) == []
    assert {str(annotation.name) for annotation in kept} >= {"cabinet1_drawer_top"}


def test_the_workspace_lists_an_environments_annotations(tmp_path):
    loaded = UrdfEnvironmentBundle(tmp_path).write(DRAWER_CABINET_URDF, set())
    listed = PlanWorkspace().environment_annotations(tmp_path / loaded.entry.name)
    assert SurfaceDescription(type_name="handle", name="handle_cab1_drawer_top") in (
        listed.annotations
    )
    assert listed.annotations == sorted(
        listed.annotations, key=lambda entry: (entry.type_name, entry.name)
    )
