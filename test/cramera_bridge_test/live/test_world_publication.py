"""
The world's state, objects, meshes, statechart and transforms as the publisher serves
them.
"""

from __future__ import annotations

import json
from http import HTTPStatus

import pytest

from cramera.exceptions import PathOutsideDirectory
from cramera.http.messages import HttpMethod
from cramera_bridge.exceptions import MissingMeshFile, NoWorldAttached, UnknownMesh
from cramera_bridge.geometry import NumericPose
from cramera_bridge.live.activity import SimulationActivity
from cramera_bridge.live.objects import ServedMesh, ShapePublication
from cramera_bridge.live.publisher import LivePublisher
from cramera_bridge.live.statechart import ChartStructure
from cramera_bridge.live.transforms import (
    MOVING_SECONDS,
    SETTLED_SECONDS,
    TransformTracker,
)
from cramera_protocol.live import LiveRoute
from cramera_protocol.live_messages import (
    LiveStatechart,
    ShapeKind,
    TransformFreshness,
    TransformWriter,
)
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Mesh

from ..dataset.live_worlds import HeightChart
from ..dataset.worlds import CUP_MESH
from .conftest import live_request

# %% state


def test_the_state_keys_objects_by_body_identity(publisher, kitchen):
    state = publisher.current_state()
    assert (
        state.objects[str(kitchen.milk.id)]
        == NumericPose.of_entity(kitchen.world, kitchen.milk).published()
    )
    assert state.base is None
    assert state.sequence_number > 0


def test_a_new_state_counts_up(publisher):
    before = publisher.current_state().sequence_number
    publisher.observe_state_change()
    assert publisher.current_state().sequence_number == before + 1


def test_nothing_is_published_before_a_world_is_attached(locations):
    publisher = LivePublisher(locations=locations)
    publisher.observe_state_change()
    assert publisher.info().running is False
    assert publisher.live_scene().scene is None
    with pytest.raises(NoWorldAttached):
        publisher.surfaces()


# %% objects


def test_objects_carry_their_native_shapes(publisher, kitchen):
    catalog = {entry.identifier: entry for entry in publisher.object_catalog().objects}
    milk = catalog[str(kitchen.milk.name)]
    cup = catalog[str(kitchen.cup.name)]
    assert [shape.kind for shape in milk.shapes] == [ShapeKind.CYLINDER]
    assert milk.shapes[0].opacity == kitchen.milk.visual.shapes[0].color.A
    assert cup.shapes[0].kind is ShapeKind.MESH
    assert cup.shapes[0].file_format == CUP_MESH.suffix.lstrip(".")


def test_a_body_without_shapes_has_no_geometry_and_no_colour(publisher, kitchen):
    catalog = {entry.identifier: entry for entry in publisher.object_catalog().objects}
    assert catalog[str(kitchen.frame.name)].shapes == []


def test_a_mesh_and_its_material_library_are_served(answerer, publisher, kitchen):
    cup = next(
        entry
        for entry in publisher.object_catalog().objects
        if entry.identifier == str(kitchen.cup.name)
    )
    mesh = answerer.answer(
        live_request(
            HttpMethod.GET, LiveRoute.MESH, query=cup.shapes[0].mesh.split("?")[1]
        )
    )
    library = answerer.answer(
        live_request(
            HttpMethod.GET,
            LiveRoute.MESH,
            query=cup.shapes[0].material_library.split("?")[1],
        )
    )
    assert mesh.body == CUP_MESH.read_bytes()
    assert library.body == CUP_MESH.with_suffix(".mtl").read_bytes()


def test_no_file_outside_the_mesh_directory_is_served(publisher, kitchen):
    key = next(iter(publisher.objects.meshes))
    with pytest.raises(PathOutsideDirectory):
        publisher.mesh_file(key, "../worlds.py")
    with pytest.raises(UnknownMesh):
        publisher.mesh_file("unknown", None)


def test_a_mesh_without_its_file_is_an_error(tmp_path):
    publication = ShapePublication(Mesh(filename=str(tmp_path / "gone.obj")))
    with pytest.raises(MissingMeshFile):
        publication.description("key")


def test_a_mesh_of_another_format_has_no_material_library(tmp_path):
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"solid part\nendsolid part\n")
    assert ServedMesh(mesh).material_library() is None


# %% statechart


def test_the_statechart_shows_its_last_native_state(publisher, kitchen):
    chart = HeightChart.build(kitchen.world, kitchen.milk)
    chart.executor.tick()
    publisher.chart_title = "observe"
    publisher.observe_motion_tick(chart.chart)
    statechart = publisher.current_statechart()
    monitor = chart.chart.nodes[0]
    assert statechart.title == "observe"
    assert statechart.nodes[0].identifier == ChartStructure.identifier(monitor.index)
    assert statechart.nodes[0].life_cycle == LifeCycleValues.RUNNING.name
    assert statechart.nodes[0].observation == ObservationStateValues.TRUE.name


def test_an_unchanged_statechart_is_not_republished(publisher, kitchen):
    chart = HeightChart.build(kitchen.world, kitchen.milk)
    assert publisher.statechart.observe(chart.chart, "observe") is True
    assert publisher.statechart.observe(chart.chart, "observe") is False


def test_no_statechart_before_a_motion(publisher):
    assert publisher.current_statechart() == LiveStatechart()


# %% transforms


def test_a_moved_connection_reads_as_moving_then_settles(kitchen):
    name = str(kitchen.milk_connection.name)
    tracker = TransformTracker()
    tracker.observe(kitchen.world, 0.0)
    kitchen.milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.0, 0.0, 0.9, reference_frame=kitchen.root
    )
    tracker.observe(kitchen.world, 10.0)
    moving = {entry.name: entry for entry in tracker.graph(10.0).connections}
    later = {
        entry.name: entry
        for entry in tracker.graph(10.0 + MOVING_SECONDS + 0.1).connections
    }
    stale = {
        entry.name: entry
        for entry in tracker.graph(10.0 + SETTLED_SECONDS + 1).connections
    }
    assert moving[name].freshness is TransformFreshness.MOVING
    assert moving[name].writer is TransformWriter.DEMO
    assert later[name].freshness is TransformFreshness.SETTLED
    assert stale[name].freshness is TransformFreshness.STALE


def test_a_fixed_connection_is_static(publisher, kitchen):
    graph = publisher.transform_graph()
    table = next(
        entry for entry in graph.connections if entry.child == str(kitchen.table.name)
    )
    assert table.freshness is TransformFreshness.STATIC


def test_the_transform_route_answers_the_graph(answerer):
    answer = answerer.answer(live_request(HttpMethod.GET, LiveRoute.TRANSFORMS))
    assert answer.status is HTTPStatus.OK
    assert json.loads(answer.body)["connections"]


# %% activity


def test_the_simulation_is_busy_right_after_a_tick():
    times = iter([1.0, 1.1, 2.0])
    activity = SimulationActivity(clock=lambda: next(times), idle_seconds=0.5)
    assert activity.is_busy() is False
    activity.tick()
    assert activity.is_busy() is True
    assert activity.is_busy() is False
