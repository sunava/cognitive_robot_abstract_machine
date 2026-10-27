"""
The laboratory as a CRAM world.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramera_protocol.laboratory.cabinet import CabinetStructure
from cramera_bridge.laboratory.world import LaboratoryEnvironment, LaboratoryHandle
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle


@pytest.fixture(scope="module")
def environment(laboratory_bundle) -> LaboratoryEnvironment:
    """
    :return: The shipped bench as a CRAM environment.
    """
    return LaboratoryEnvironment(laboratory_bundle)


@pytest.fixture(scope="module")
def world(environment):
    """
    :return: A world holding only the laboratory.
    """
    return environment.create_world()


def test_every_free_object_stands_at_its_spawn_with_collision_shapes(
    environment, world
):
    for description in environment.contents.scene.objects:
        body = world.get_body_by_name(description.key)
        np.testing.assert_allclose(
            body.global_pose.to_np()[:3, 3],
            [description.spawn.x, description.spawn.y, description.spawn.z],
        )
        assert body.collision.shapes


def test_every_handle_is_annotated(world):
    handles = {
        handle.root.name.name
        for handle in world.get_semantic_annotations_by_type(Handle)
    }
    assert handles == {handle.value for handle in LaboratoryHandle}


def test_the_closed_cabinet_has_collision_but_no_visual(world):
    cabinet = world.get_body_by_name(CabinetStructure().name)
    assert len(cabinet.collision.shapes) == len(CabinetStructure().panels())
    assert not cabinet.visual.shapes


def test_a_slot_pose_is_the_bench_layout_in_the_world_root(environment, world):
    slot = environment.bench.rack_slots[-1]
    pose = environment.slot_pose(slot.identifier, world)
    assert pose.reference_frame is world.root
    np.testing.assert_allclose(
        pose.to_np()[:3, 3], [slot.pose.x, slot.pose.y, slot.pose.z]
    )


def test_the_vessel_dimensions_come_from_the_semantics(environment):
    assert 0 < environment.vessels.outer_radius < environment.vessels.height
    assert environment.vessels.grasp_height < environment.vessels.height


def test_support_contacts_are_allowed_only_on_request(environment):
    plain = environment.create_world()
    allowing = LaboratoryEnvironment(
        environment.directory, allow_support_contacts=True
    ).create_world()
    assert len(allowing.collision_manager.ignore_collision_rules) == (
        len(plain.collision_manager.ignore_collision_rules) + 1
    )


def test_a_robot_joins_the_laboratory(environment):
    world = environment.create_world(
        robots=[RobotSpecification(semantic_annotation_type=PR2)]
    )
    assert world.get_semantic_annotations_by_type(PR2)
    assert world.get_body_by_name(environment.bench.tubes[0].key)
