import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "krrood" / "src"))

pytest.importorskip("rustworkx")

from krrood.entity_query_language.factories import entity, variable

from pycram.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import (
    on_supporting_surface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3

from pycram.robot_plans.actions.composite.cleanup import query_bowls_on_support


def _parse_bowl_world():
    return STLParser(
        os.path.join(
            Path(__file__).resolve().parents[2],
            "pycram",
            "resources",
            "objects",
            "bowl.stl",
        )
    ).parse()


def _test_world():
    world = setup_world()

    with world.modify_world():
        table_body = world.get_body_by_name("table_area_main")

        table_annotation = Table(root=table_body, name=PrefixedName("cleanup_table"))
        world.add_semantic_annotation(table_annotation)
        table_annotation.calculate_supporting_surface()

    surface = table_annotation.supporting_surface
    surface_bbox = surface.area.as_bounding_box_collection_in_frame(
        surface
    ).bounding_box()
    x_center = 0.5 * (surface_bbox.min_x + surface_bbox.max_x)
    y_center = 0.5 * (surface_bbox.min_y + surface_bbox.max_y)
    bowl_world = _parse_bowl_world()
    bowl_world.root.name.name = "cleanup_bowl_1"
    bowl_bbox = bowl_world.root.collision.as_bounding_box_collection_in_frame(
        bowl_world.root
    ).bounding_box()
    bowl_center_x = 0.5 * (bowl_bbox.min_x + bowl_bbox.max_x)
    bowl_center_y = 0.5 * (bowl_bbox.min_y + bowl_bbox.max_y)

    local_point = Point3(
        x=float(x_center - bowl_center_x),
        y=float(y_center - bowl_center_y),
        z=float(surface_bbox.max_z - bowl_bbox.min_z),
        reference_frame=surface,
    )
    world_point = world.transform(local_point, world.root)
    bowl_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(world_point.x),
        y=float(world_point.y),
        z=float(world_point.z),
        reference_frame=world.root,
    )
    world.merge_world_at_pose(bowl_world, bowl_pose)

    with world.modify_world():
        world.add_semantic_annotation(
            Bowl(
                root=world.get_body_by_name("cleanup_bowl_1"),
                name=PrefixedName("cleanup_bowl_annotation_1"),
            )
        )

    world.update_forward_kinematics()
    return world


def test_on_supporting_surface_predicate():
    world = _test_world()
    table = world.get_semantic_annotations_by_type(Table)[0]
    bowl = world.get_semantic_annotations_by_type(Bowl)[0]

    assert on_supporting_surface(bowl, table)


def test_eql_query_for_bowls_on_support():
    world = _test_world()
    table = world.get_semantic_annotations_by_type(Table)[0]

    bowls = query_bowls_on_support(world, table.root)

    assert [item.root.name.name for item in bowls] == ["cleanup_bowl_1"]

    bowl = variable(Bowl, domain=world.get_semantic_annotations_by_type(Bowl))
    table_var = variable(Table, domain=[table])
    query = entity(bowl).where(on_supporting_surface(bowl, table_var))

    assert [item.root.name.name for item in query.evaluate()] == ["cleanup_bowl_1"]
