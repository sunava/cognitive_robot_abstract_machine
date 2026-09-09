"""
Tests of which bodies the viewer streams instead of loading with the scene.
"""

from __future__ import annotations

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.soft_connections import (
    PiecewiseConstantCurvatureConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.world_entity import Body

from cramera.streamed_bodies import (
    hangs_off_a_non_urdf_joint,
    is_streamed,
    is_urdf_joint,
    names_a_mesh_file,
)

SEGMENT_LENGTH = 0.1
"""
Arc length of the continuum segment the fixtures build.
"""


def continuum_world() -> World:
    """
    A world whose root carries a rigid shelf and a continuum segment, and where a plate
    hangs rigidly off that segment.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="world"))
    shelf = Body(name=PrefixedName("shelf", prefix="world"))
    segment = Body(name=PrefixedName("segment", prefix="trunk"))
    plate = Body(name=PrefixedName("plate", prefix="trunk"))
    limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(position=-10.0, velocity=-10.0),
        upper=DerivativeMap(position=10.0, velocity=10.0),
    )
    curvature = DegreeOfFreedom(name=PrefixedName("kappa"), limits=limits)
    bending_plane = DegreeOfFreedom(name=PrefixedName("phi"), limits=limits)
    with world.modify_world():
        world.add_body(root)
        world.add_degree_of_freedom(curvature)
        world.add_degree_of_freedom(bending_plane)
        world.add_connection(FixedConnection(parent=root, child=shelf))
        world.add_connection(
            PiecewiseConstantCurvatureConnection(
                parent=root,
                child=segment,
                kappa_dof_id=curvature.id,
                phi_dof_id=bending_plane.id,
                segment_length=SEGMENT_LENGTH,
            )
        )
        world.add_connection(FixedConnection(parent=segment, child=plate))
    return world


# %% what a URDF joint can express


def test_a_fixed_connection_is_a_urdf_joint():
    world = continuum_world()
    shelf = world.get_body_by_name(PrefixedName("shelf", prefix="world"))

    assert is_urdf_joint(shelf.parent_connection)


def test_a_continuum_connection_is_no_urdf_joint():
    world = continuum_world()
    segment = world.get_body_by_name(PrefixedName("segment", prefix="trunk"))

    assert not is_urdf_joint(segment.parent_connection)


# %% what that makes of the bodies below it


def test_a_rigidly_placed_body_is_loaded_with_the_scene():
    world = continuum_world()
    shelf = world.get_body_by_name(PrefixedName("shelf", prefix="world"))

    assert not hangs_off_a_non_urdf_joint(shelf)
    assert not is_streamed(shelf)


def test_a_continuum_segment_is_streamed():
    world = continuum_world()
    segment = world.get_body_by_name(PrefixedName("segment", prefix="trunk"))

    assert is_streamed(segment)


def test_a_body_rigidly_attached_below_a_continuum_segment_is_streamed():
    """
    Its own connection is a URDF joint, but the segment it hangs off is not, so a bundle
    could only nail it down where it happened to be.
    """
    world = continuum_world()
    plate = world.get_body_by_name(PrefixedName("plate", prefix="trunk"))

    assert is_urdf_joint(plate.parent_connection)
    assert is_streamed(plate)


# %% the demo's own objects


def test_a_body_named_after_a_mesh_file_is_streamed():
    world = World()
    parcel = Body(name=PrefixedName("parcel.obj", prefix="world"))
    with world.modify_world():
        world.add_body(parcel)

    assert names_a_mesh_file(parcel)
    assert is_streamed(parcel)
