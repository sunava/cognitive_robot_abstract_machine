"""
Recorded carriers keep one authored visual subtree and independent pose tracks.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import pytest

from cramera.body_geometry import rounded_pose
from cramera.live.robot_models import RobotModels
from cramera.onboard.demo import Recorder, SceneBuilder
from cramera.onboard.world_to_urdf import UrdfDocument
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from .test_onboard import recording


# %% parsed carrier and unrelated loose object
@pytest.fixture(params=["", "laboratory"])
def recorded_carrier(
    request: pytest.FixtureRequest, cylinder_bot_world: World, tmp_path: Path
) -> SceneBuilder:
    """
    Record an actual laboratory source alongside an independently spawned sample.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "cramera/src/cramera/resources/precision_lab/environment.urdf"
    )
    laboratory = URDFParser.from_file(str(source), prefix=request.param).parse()
    world = cylinder_bot_world
    robot_source = UrdfDocument.of_bodies(
        bodies=list(world.bodies_topologically_sorted),
        name="fixture_robot",
        output_directory=str(tmp_path / "robot"),
        mesh_subdirectory="robot",
    )
    world.merge_world(laboratory)
    robot = RobotModels(world).robots()[0]
    loose = Body(
        name=PrefixedName("unclaimed_sample"),
        visual=ShapeCollection([Box(scale=Scale(0.01, 0.01, 0.02))]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=world.root, child=loose)
        )
    recorder = Recorder()
    recorder.world, recorder.robot = world, robot
    bodies = {
        recorder.object_key(body): body for body in recorder.free_floating_bodies()
    }
    first = {key: rounded_pose(body) for key, body in bodies.items()}
    last = {key: list(pose) for key, pose in first.items()}
    last["laboratory_rack"][:3] = [-0.52, -0.4, 0.8]
    recorder = recording([first, last], [rounded_pose(robot.root)] * 2)
    recorder.world, recorder.robot = world, robot
    recorder._bodies = bodies
    recorder.urdf_sources = [str(source), robot_source.urdf]
    return SceneBuilder(recorder, "carrier", str(tmp_path / "bundle"), 1)


def test_parsed_carrier_has_one_visual_and_preserves_its_child_frame(
    recorded_carrier: SceneBuilder,
) -> None:
    """
    A source-described rack is animated through its URDF without a duplicate mesh.
    """
    scene = recorded_carrier.build()
    assert [item["key"] for item in scene["objects"]] == ["unclaimed_sample"]
    output = Path(recorded_carrier.output_directory)
    model = next(item for item in scene["models"] if item["urdf"] == "environment.urdf")
    description = ElementTree.parse(output / model["urdf"])
    assert len(description.findall("link[@name='laboratory_rack']/visual")) == 1
    handle = description.find("joint[@name='laboratory_rack_handle_joint']")
    assert handle.attrib["type"] == "fixed"
    assert handle.find("parent").attrib["link"] == "laboratory_rack"
    trajectory = json.loads((output / "trajectory.json").read_text())
    name = str(recorded_carrier.recorder._bodies["laboratory_rack"].name)
    assert trajectory["bodyPoses"] == [
        {name: frame["laboratory_rack"]}
        for frame in recorded_carrier.recorder.object_frames
    ]
    assert trajectory["objects"] == recorded_carrier.recorder.object_frames


def test_authored_body_mapping_is_available_to_live_updates(
    recorded_carrier: SceneBuilder,
) -> None:
    """
    The scene maps semantic object keys to source-owned model links for live poses.
    """
    scene = recorded_carrier.build()
    body = recorded_carrier.recorder._bodies["laboratory_rack"]
    assert scene["modelBodies"] == {"laboratory_rack": str(body.name)}
