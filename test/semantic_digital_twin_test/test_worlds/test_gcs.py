import plotly.graph_objects as go
import pytest
from dataclasses import dataclass

from random_events.interval import SimpleInterval
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GraphOfConvexSetsFixture:
    """
    Data class for Graph of Convex Sets test fixture.
    """

    world: World
    graph_of_convex_sets: GraphOfConvexSets


@pytest.fixture
def graph_of_convex_sets_unit_box() -> GraphOfConvexSetsFixture:
    """
    Create a GraphOfConvexSets for navigation around a unit box.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body())

    graph_of_convex_sets = GraphOfConvexSets(world)

    obstacle = BoundingBox(0, 0, 0, 1, 1, 1, world.root.global_pose)

    z_lim = SimpleInterval.from_data(0.45, 0.55)
    x_lim = SimpleInterval.from_data(-2, 3)
    y_lim = SimpleInterval.from_data(-2, 3)
    limiting_event = SimpleEvent.from_data(
        {
            SpatialVariables.x.value: x_lim,
            SpatialVariables.y.value: y_lim,
            SpatialVariables.z.value: z_lim,
        }
    )
    obstacles = BoundingBoxCollection.from_event(
        world.root,
        ~obstacle.simple_event.as_composite_set() & limiting_event.as_composite_set(),
    )
    for bounding_box in obstacles:
        graph_of_convex_sets.add_node(bounding_box)

    graph_of_convex_sets.calculate_connectivity()
    return GraphOfConvexSetsFixture(world, graph_of_convex_sets)


def test_reachability(graph_of_convex_sets_unit_box: GraphOfConvexSetsFixture):
    """
    Verify if a path can be found around the unit box.
    """
    start_point = Point3(
        -1, -1, 0.5, reference_frame=graph_of_convex_sets_unit_box.world.root
    )
    target_point = Point3(
        2, 2, 0.5, reference_frame=graph_of_convex_sets_unit_box.world.root
    )

    path = graph_of_convex_sets_unit_box.graph_of_convex_sets.path_from_to(
        start_point, target_point
    )
    assert len(path) == 4


def test_plot(graph_of_convex_sets_unit_box: GraphOfConvexSetsFixture):
    """
    Verify if the free and occupied space can be plotted.
    """
    free_space_plot = go.Figure(
        graph_of_convex_sets_unit_box.graph_of_convex_sets.plot_free_space()
    )
    assert free_space_plot is not None
    occupied_space_plot = go.Figure(
        graph_of_convex_sets_unit_box.graph_of_convex_sets.plot_occupied_space()
    )
    assert occupied_space_plot is not None


def test_from_world(table_world: World):
    """
    Verify the generation of a connectivity graph from a world.
    """
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=-5,
                max_x=-2,
                min_y=-1,
                max_y=2,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=table_world.root
                ),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = GraphOfConvexSets.free_space_from_world(
        table_world, search_space=search_space
    )
    assert graph_of_convex_sets is not None
    assert len(graph_of_convex_sets.graph.nodes()) > 0
    assert len(graph_of_convex_sets.graph.edges()) > 0

    start = Point3(-4.5, -0.5, 0.4, reference_frame=table_world.root)
    target = Point3(-2.5, 1.5, 0.9, reference_frame=table_world.root)

    path = graph_of_convex_sets.path_from_to(start, target)

    assert path is not None
    assert len(path) > 1

    with pytest.raises(PointOccupiedError):
        start_occupied = Point3(-10, -10, -10, reference_frame=table_world.root)
        target_occupied = Point3(10, 10, 10, reference_frame=table_world.root)
        graph_of_convex_sets.path_from_to(start_occupied, target_occupied)


def test_navigation_map_from_world(table_world: World):
    """
    Verify the generation of a navigation map from a world.
    """
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=-5,
                max_x=-2,
                min_y=-1,
                max_y=2,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=table_world.root
                ),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = GraphOfConvexSets.navigation_map_from_world(
        table_world, search_space=search_space
    )
    assert len(graph_of_convex_sets.graph.nodes()) > 0
    assert len(graph_of_convex_sets.graph.edges()) > 0
