"""
A plan's objects taken from an object library: their URDFs merged into the plan's world
as free, graspable objects, and offered in the workspace's catalog.
"""

from __future__ import annotations

import pytest

from cramera_bridge.geometry import NumericPose
from cramera_bridge.plans.objects import ObjectLibraries, ObjectLibrary
from cramera_bridge.plans.workspace import PlanWorkspace
from cramera_bridge.plans.world import PlannedGraspable, PlannedObjects
from cramera_protocol.geometry import Pose
from cramera_protocol.plan_builder import PlannedObject
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from .test_objects import CUP, LIBRARY

LIBRARIES = ObjectLibraries([ObjectLibrary(LIBRARY)])
"""
The stand-in library alone.
"""


@pytest.fixture
def world() -> World:
    """
    :return: A world of nothing but its root.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("root")))
    return world


def test_a_library_model_joins_the_world_free_and_graspable_where_the_plan_puts_it(
    world, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    planned = PlannedObject(
        identifier="cup", model=CUP, pose=Pose.from_json([1.0, 0.5, 0.8, 0, 0, 0, 1])
    )
    (body,) = PlannedObjects([planned], libraries=LIBRARIES).spawn(world).values()
    assert body.name.prefix == "cup_"
    assert isinstance(body.parent_connection, Connection6DoF)
    assert NumericPose.of_entity(world, body).position == pytest.approx((1.0, 0.5, 0.8))
    assert PlannedGraspable.of(body).root is body
    lid = world.get_kinematic_structure_entity_by_name(PrefixedName("cup_lid", "cup_"))
    assert lid.parent_connection.parent is body


def test_the_catalog_offers_the_library_models_after_the_meshes(tmp_path):
    (tmp_path / "milk.stl").write_text("solid milk\nendsolid milk\n", encoding="utf-8")
    workspace = PlanWorkspace(directory=tmp_path, libraries=LIBRARIES)
    assert [model.identifier for model in workspace.objects()][:2] == ["milk.stl", CUP]
    assert workspace.object_roots() == {LIBRARY.name: LIBRARY}
