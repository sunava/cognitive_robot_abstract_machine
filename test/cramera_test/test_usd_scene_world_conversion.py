"""
A stage converted by ``cramera-import`` loads into a CRAM world with its geometry where
the stage puts it and its annotated prims labelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest

pxr = pytest.importorskip("pxr")

from pxr import Usd, UsdGeom

from cramera.live.bridge import Bridge
from cramera.onboard.usd_scene.bundle import BundleKey
from cramera.onboard.usd_scene.bundle_layout import BundleFile
from cramera.onboard.usd_scene.command_line import ImportOption, main
from cramera.onboard.usd_scene.stage_reader import PrimExclusion, StageReader
from cramera.scene_presentation import PresentationField
from cramera.usd_scene_world import UsdSceneEnvironment
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from .test_usd_scene_world import (
    DATASET,
    environment_link_names,
    labels_by_path,
    read_json,
)

STAGE = DATASET / "annotated_lab.usda"
"""
The annotated stage converted by every test of this module.
"""


# %% converted bundle
@pytest.fixture
def converted_bundle(tmp_path: Path) -> Path:
    """
    The annotated stage converted into a scene bundle.
    """
    main([str(STAGE), ImportOption.NAME, STAGE.stem, ImportOption.OUT, str(tmp_path)])
    return tmp_path / STAGE.stem


@pytest.fixture
def converted_world(converted_bundle: Path) -> World:
    """
    A world holding only the converted stage.
    """
    return UsdSceneEnvironment(bundle_directory=converted_bundle).create_world()


# %% world
def test_every_converted_link_is_a_body(
    converted_bundle: Path, converted_world: World
) -> None:
    """
    The world holds its root and every link the converter wrote.
    """
    assert {body.name.name for body in converted_world.bodies} == (
        environment_link_names(converted_bundle) | {converted_world.root.name.name}
    )


def test_converted_movable_joints_are_the_degrees_of_freedom(
    converted_bundle: Path, converted_world: World
) -> None:
    """
    The stage's revolute and prismatic joints can be actuated in the world.
    """
    [model] = read_json(converted_bundle, BundleFile.SCENE)[PresentationField.MODELS]
    assert {
        str(connection.name)
        for connection in converted_world.connections
        if isinstance(connection, ActiveConnection1DOF)
    } == set(model[BundleKey.MOVABLE_JOINTS])


@pytest.mark.parametrize(
    ("link", "prim_path"),
    [
        ("Storage__Fridge", "/World/Storage/Fridge/Shell"),
        ("Storage__Fridge__Door", "/World/Storage/Fridge/Door"),
        ("Stations__L09_PlateHotel__Gate", "/World/Stations/L09_PlateHotel/Gate"),
        ("Storage__Shelf", "/World/Storage/Shelf"),
    ],
)
def test_glb_visuals_lie_where_the_stage_puts_the_geometry(
    converted_world: World, link: str, prim_path: str
) -> None:
    """
    A link's GLB, loaded in its body frame, covers the box the stage bounds its prims by.
    """
    stage = Usd.Stage.Open(str(STAGE))
    stage_bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(stage.GetPrimAtPath(prim_path))
        .ComputeAlignedRange()
    )
    body = converted_world.get_body_by_name(link)
    bounds = body.visual.as_bounding_box_collection_in_frame(
        converted_world.root
    ).bounding_box()
    numpy.testing.assert_allclose(
        [
            bounds.min_x,
            bounds.min_y,
            bounds.min_z,
            bounds.max_x,
            bounds.max_y,
            bounds.max_z,
        ],
        [*stage_bounds.GetMin(), *stage_bounds.GetMax()],
        atol=1e-5,
    )


# %% semantics
def test_converted_prims_label_the_bodies_of_their_links(
    converted_bundle: Path, converted_world: World
) -> None:
    """
    Every prim the converter found semantics on labels its link's body.
    """
    prims = read_json(converted_bundle, BundleFile.SEMANTICS)[BundleKey.PRIMS]
    assert {
        path: label.root.name.name
        for path, label in labels_by_path(converted_world).items()
    } == {path: entry[BundleKey.LINK] for path, entry in prims.items()}


def test_converted_scene_answers_eql(
    converted_bundle: Path, converted_world: World
) -> None:
    """
    A capability authored on the stage finds the station body through the live bridge.
    """
    bridge = Bridge()
    bridge.register_query_source(
        UsdSceneEnvironment(bundle_directory=converted_bundle).query_source(
            converted_world
        )
    )
    result = bridge.run_query(
        "an(entity(prim.root).where(contains(prim.capabilities, 'retrieve_plate')))"
    )
    assert result.ok
    assert [row["__entity__"] for row in result.rows] == ["Stations__L09_PlateHotel"]


# %% environment without robots
def test_world_of_a_stage_with_robots_holds_only_its_environment(
    tmp_path: Path,
) -> None:
    """
    A stage converted by default loads without the bodies of the robots it marks.
    """
    stage = DATASET / "robot_markers.usda"
    main([str(stage), ImportOption.NAME, stage.stem, ImportOption.OUT, str(tmp_path)])
    world = UsdSceneEnvironment(bundle_directory=tmp_path / stage.stem).create_world()
    robots = [excluded.path for excluded in StageReader(stage).read().report.excluded]
    with_robots = StageReader(stage, exclusion=PrimExclusion(keep_robots=True)).read()
    robot_links = {
        link.name
        for link in with_robots.links
        if any(link.path.startswith(robot) for robot in robots)
    }
    bodies = {body.name.name for body in world.bodies}
    assert bodies == environment_link_names(tmp_path / stage.stem) | {
        world.root.name.name
    }
    assert robot_links == {
        "Robots__Rover__Chassis",
        "Robots__Rover__Chassis__Wheel",
        "Robots__Walker",
    }
    assert bodies & robot_links == set()
