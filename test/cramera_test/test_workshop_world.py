"""
Native workshop object freedom and handle frame propagation.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cramera.workshop_world import WorkshopWorld
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.connections import Connection6DoF

from .test_workshop_bundle import workshop_directory

# %% portable visual fixture


@pytest.fixture
def workshop_bundle(workshop_directory: Path) -> Path:
    """
    Supply small visual meshes without requiring the Blender authoring runtime.
    """
    for description in workshop_directory.glob("*.urdf"):
        for visual in ElementTree.parse(description).findall(".//visual/geometry/mesh"):
            destination = workshop_directory / visual.attrib["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                trimesh.creation.box().export(destination)
    return workshop_directory


# %% semantic articulation


def test_free_parts_and_handle_annotations_follow_manifest(
    workshop_bundle: Path,
) -> None:
    """
    Load all loose bodies independently and keep semantic grasp frames on drawers.
    """
    specification = WorkshopWorld(bundle_directory=workshop_bundle)
    world = specification.create_world()
    scene = json.loads((workshop_bundle / "scene.json").read_text())
    handles = {
        str(annotation.root.name)
        for annotation in world.get_semantic_annotations_by_type(Handle)
    }
    assert handles == {drawer["handle"] for drawer in scene["workshop"]["drawers"]}
    for item in scene["objects"]:
        body = world.get_body_by_name(item["key"])
        assert isinstance(body.parent_connection, Connection6DoF)
        np.testing.assert_allclose(body.global_pose.to_np()[:3, 3], item["spawn"][:3])
    drawer = scene["workshop"]["drawers"][0]
    handle = world.get_body_by_name(drawer["handle"])
    before = handle.global_pose.to_np()[:3, 3].copy()
    slide = world.get_connection_by_name(drawer["joint"])
    slide.position = drawer["open"]
    np.testing.assert_allclose(
        handle.global_pose.to_np()[:3, 3], before + [0, -drawer["open"], 0]
    )
