"""
Unit tests for :func:`cramera.body_geometry.measure_body` and
:func:`~cramera.body_geometry.rounded_scale`.

Every shape is attached to a real, single-body ``World`` rather than a duck-typed mimic:
``ShapeCollection.scale`` transforms each shape's bounding box through its reference
frame, which requires that frame to belong to an actual world.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Scale,
    Sphere,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Optional

from krrood.symbolic_math.symbolic_math import SymbolicMathType

from cramera.body_geometry import measure_body, rounded_pose, rounded_scale


# %% fixtures
def _body_with_shapes(
    visual: Optional[ShapeCollection] = None,
    collision: Optional[ShapeCollection] = None,
) -> Body:
    """
    A real ``Body``, registered in its own single-body ``World``.

    ``ShapeCollection.scale`` reads each shape's reference frame back to a world via
    :meth:`Body.__post_init__`'s wiring, so a bare, unregistered ``Body`` is not enough.
    """
    body = Body(
        name=PrefixedName("object"),
        visual=visual if visual is not None else ShapeCollection(),
        collision=collision if collision is not None else ShapeCollection(),
    )
    world = World()
    with world.modify_world():
        world.add_body(body)
    return body


# %% measure_body
def test_of_measures_a_box():
    body = _body_with_shapes(
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.2, 0.3, 0.4))])
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.2, 0.3, 0.4])


def test_of_measures_a_mesh():
    """
    A unit mesh scaled by ``Scale(0.2, 0.3, 0.4)`` measures to that scale.

    True whether read directly off the ``Mesh.scale`` field (the pre-fix path) or from
    the scaled geometry's own bounding box (the post-fix path) - both agree here.
    """
    body = _body_with_shapes(
        collision=ShapeCollection(
            shapes=[Mesh.box(extents=(1.0, 1.0, 1.0), scale=Scale(0.2, 0.3, 0.4))]
        )
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.2, 0.3, 0.4])


def test_of_measures_a_sphere():
    """
    A sphere has no ``.scale`` attribute; before the fix this reported ``None``.
    """
    body = _body_with_shapes(collision=ShapeCollection(shapes=[Sphere(radius=0.5)]))
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([1.0, 1.0, 1.0])


def test_of_measures_a_cylinder():
    """
    A cylinder has no ``.scale`` attribute; before the fix this reported ``None``.
    """
    body = _body_with_shapes(
        collision=ShapeCollection(shapes=[Cylinder(width=0.4, height=0.6)])
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.4, 0.4, 0.6])


def test_of_returns_none_when_the_body_has_no_shapes():
    body = _body_with_shapes()
    assert measure_body(body) is None


def test_of_prefers_visual_over_collision():
    body = _body_with_shapes(
        visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.9, 0.9, 0.9))]),
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.1, 0.1, 0.1])


# %% rounded_scale
def test_rounded_scale_rounds_each_axis():
    assert rounded_scale(Scale(x=0.123456, y=1.0, z=2.987654), 3) == [0.123, 1.0, 2.988]


# %% rounded_pose
def test_rounded_pose_reports_position_then_quaternion():
    """
    The pose is published in the ``[x, y, z, qx, qy, qz, qw]`` order the viewer reads,
    which is the order semantic_digital_twin's own conversion produces.
    """
    body = _body_with_shapes()
    assert rounded_pose(body, 5) == pytest.approx(
        body.global_pose.to_position_quaternion_list()
    )


def test_rounded_pose_rounds_every_value():
    world = World()
    root = Body(name=PrefixedName("world"))
    body = Body(name=PrefixedName("object"))
    with world.modify_world():
        degrees_of_freedom = {
            component: DegreeOfFreedom(name=PrefixedName(component))
            for component in ("x", "y", "z", "qx", "qy", "qz", "qw")
        }
        for degree_of_freedom in degrees_of_freedom.values():
            world.add_degree_of_freedom(degree_of_freedom)
        world.add_connection(
            Connection6DoF(parent=root, child=body, **degrees_of_freedom)
        )
        world.state[degrees_of_freedom["qw"].id].position = 1.0
        world.state[degrees_of_freedom["x"].id].position = 0.123456789

    assert rounded_pose(body, 3)[0] == 0.123


def _freely_posed_body() -> Body:
    """
    A body on a 6-DOF connection, moved to a pose with a nontrivial rotation.

    Rotated a quarter turn around z (``qz = qw = sqrt(0.5)``) and translated, so a
    pose conversion that mixes up axes, order or handedness cannot pass.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    body = Body(name=PrefixedName("object"))
    with world.modify_world():
        degrees_of_freedom = {
            component: DegreeOfFreedom(name=PrefixedName(component))
            for component in ("x", "y", "z", "qx", "qy", "qz", "qw")
        }
        for degree_of_freedom in degrees_of_freedom.values():
            world.add_degree_of_freedom(degree_of_freedom)
        world.add_connection(
            Connection6DoF(parent=root, child=body, **degrees_of_freedom)
        )
        world.state[degrees_of_freedom["x"].id].position = 0.4
        world.state[degrees_of_freedom["y"].id].position = -0.2
        world.state[degrees_of_freedom["z"].id].position = 0.9
        world.state[degrees_of_freedom["qz"].id].position = 0.5**0.5
        world.state[degrees_of_freedom["qw"].id].position = 0.5**0.5
    return body


def test_rounded_pose_matches_the_symbolic_conversion_for_a_rotated_body():
    body = _freely_posed_body()
    assert rounded_pose(body, 5) == pytest.approx(
        [round(value, 5) for value in body.global_pose.to_position_quaternion_list()]
    )


def test_rounded_pose_constructs_no_symbolic_expressions(monkeypatch):
    """
    The live bridge publishes poses from the physics thread's state-change callback, and
    CasADi expression construction is not thread-safe, so the pose must be derived
    purely numerically.
    """
    body = _freely_posed_body()
    constructed = []
    original_post_init = SymbolicMathType.__post_init__

    def counting_post_init(self) -> None:
        constructed.append(type(self).__name__)
        original_post_init(self)

    monkeypatch.setattr(SymbolicMathType, "__post_init__", counting_post_init)
    rounded_pose(body)
    assert constructed == []
