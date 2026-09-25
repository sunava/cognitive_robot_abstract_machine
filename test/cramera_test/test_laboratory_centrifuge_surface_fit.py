"""
Centrifuge markings and controls remain attached to their supporting surfaces.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest
import trimesh


# %% Packaged instrument geometry
class CentrifugeSurface(StrEnum):
    """
    Authored surfaces involved in the front control assembly.
    """

    SHELL = "Centrifuge chamber shell"
    BEZEL = "Centrifuge control bezel"
    DISPLAY = "Centrifuge display"
    IDENTITY = "Centrifuge engraved identity"


@pytest.fixture
def centrifuge_surfaces() -> dict[str, trimesh.Trimesh]:
    """
    Read the shipped device so fitting checks need no Blender installation.
    """
    path = (
        Path(__file__).parents[2]
        / "cramera/src/cramera/resources/aicor_medical_lab/assets/centrifuge_base.glb"
    )
    scene = trimesh.load(path, force="scene", process=False, merge_primitives=True)
    return {name: scene.geometry[scene.graph[name][1]] for name in CentrifugeSurface}


def surface_distances(support: trimesh.Trimesh, vertices: np.ndarray) -> np.ndarray:
    """
    Measure mounting clearances without an optional spatial-index dependency.
    """
    return np.concatenate(
        [
            trimesh.proximity.closest_point_naive(support, chunk)[1]
            for chunk in np.array_split(vertices, max(1, len(vertices) // 32))
        ]
    )


# %% Surface fitting regressions
def test_centrifuge_identity_follows_the_curved_housing(
    centrifuge_surfaces: dict[str, trimesh.Trimesh],
) -> None:
    """
    Printed lettering stays within half a millimetre of the curved enamel.
    """
    distances = surface_distances(
        centrifuge_surfaces[CentrifugeSurface.SHELL],
        centrifuge_surfaces[CentrifugeSurface.IDENTITY].vertices,
    )
    assert distances.max() < 0.0005


def test_control_enclosure_rear_matches_the_round_housing(
    centrifuge_surfaces: dict[str, trimesh.Trimesh],
) -> None:
    """
    The rear mounting surface seats against the shell across its full width.
    """
    bezel = centrifuge_surfaces[CentrifugeSurface.BEZEL]
    rear_faces = bezel.faces[bezel.face_normals[:, 1] > 0.5]
    vertices = bezel.vertices[np.unique(rear_faces)]
    distances = surface_distances(
        centrifuge_surfaces[CentrifugeSurface.SHELL], vertices
    )
    assert distances.max() < 0.0005


def test_display_is_recessed_into_its_control_enclosure(
    centrifuge_surfaces: dict[str, trimesh.Trimesh],
) -> None:
    """
    Screen glass meets the bezel face instead of forming a detached plate.
    """
    bezel = centrifuge_surfaces[CentrifugeSurface.BEZEL]
    display = centrifuge_surfaces[CentrifugeSurface.DISPLAY]
    bezel_front = bezel.bounds[0, 1]
    display_front = display.bounds[0, 1]
    display_back = display.bounds[1, 1]
    assert 0 <= bezel_front - display_front < 0.0005
    assert display_back > bezel_front
