"""
Dispenser artwork and liquid feeds fit their supporting instrument surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cramera.laboratory_pipetting import PipettingSource

from .test_laboratory_branding import BrandedAsset, LogoSurface


# %% dispenser mesh placement
class DispenserSurface(StrEnum):
    """
    Named supporting surfaces of the authored dispenser.
    """

    COVER = "Pipetting head cover"
    LABEL = "Pipetting instrument label"
    INLET = "head inlet coupling"
    TUBE = "feed tube"


@dataclass(frozen=True)
class DispenserMeshes:
    """
    Authored meshes transformed into the common instrument frame.
    """

    scene: trimesh.Scene
    """
    Packaged scene containing the feed lines and their supporting housing.
    """

    def mesh(self, name: str) -> trimesh.Trimesh:
        """
        Resolve an authored object with its scene transform applied.
        """
        transform, geometry = self.scene.graph[name]
        result = self.scene.geometry[geometry].copy()
        result.apply_transform(transform)
        return result

    def channel_mesh(
        self, channel: PipettingSource, part: DispenserSurface
    ) -> trimesh.Trimesh:
        """
        Resolve the corresponding part of an independent liquid feed.
        """
        return self.mesh(f"Pipetting {channel} {part}")


@pytest.fixture
def dispenser_meshes() -> DispenserMeshes:
    """
    Load the shipped geometry without requiring Blender during tests.
    """
    asset = (
        Path(__file__).parents[2]
        / "cramera/src/cramera/resources/precision_lab/assets"
        / f"{BrandedAsset.PIPETTING}.glb"
    )
    return DispenserMeshes(trimesh.load(asset, merge_primitives=True))


def test_dispenser_logo_is_above_the_instrument_label(
    dispenser_meshes: DispenserMeshes,
) -> None:
    """
    The upper housing carries the brand above the instrument description.
    """
    logo = dispenser_meshes.mesh(LogoSurface.DISPENSER).bounds
    label = dispenser_meshes.mesh(DispenserSurface.LABEL).bounds
    assert logo[0, 2] > label[1, 2]


@pytest.mark.parametrize("channel", [PipettingSource.AMBER, PipettingSource.TEAL])
def test_feed_inlet_is_seated_in_the_head_cover(
    dispenser_meshes: DispenserMeshes, channel: PipettingSource
) -> None:
    """
    Each inlet straddles the top cover within its horizontal footprint.
    """
    cover = dispenser_meshes.mesh(DispenserSurface.COVER).bounds
    inlet = dispenser_meshes.channel_mesh(channel, DispenserSurface.INLET).bounds
    assert cover[0, 2] < inlet[0, 2] < cover[1, 2] < inlet[1, 2]
    assert np.all(inlet[0, :2] > cover[0, :2])
    assert np.all(inlet[1, :2] < cover[1, :2])


@pytest.mark.parametrize("channel", [PipettingSource.AMBER, PipettingSource.TEAL])
def test_feed_tube_enters_its_head_inlet(
    dispenser_meshes: DispenserMeshes, channel: PipettingSource
) -> None:
    """
    The tube penetrates the interior of its fitting instead of floating above.
    """
    inlet = dispenser_meshes.channel_mesh(channel, DispenserSurface.INLET).bounds
    vertices = dispenser_meshes.channel_mesh(channel, DispenserSurface.TUBE).vertices
    center = inlet.mean(axis=0)
    radius = min(inlet[1, :2] - inlet[0, :2]) / 2
    central_height = (inlet[1, 2] - inlet[0, 2]) / 4
    distance = np.linalg.norm(vertices[:, :2] - center[:2], axis=1)
    assert np.any(
        (distance < radius)
        & (vertices[:, 2] > center[2] - central_height)
        & (vertices[:, 2] < center[2] + central_height)
    )
