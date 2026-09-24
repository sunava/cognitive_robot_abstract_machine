"""Bottle labels and their lettering follow the exported vessel surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest
import trimesh

# %% authored bottle surfaces


class BottleAsset(StrEnum):
    """Exported models containing labelled reagent bottles."""

    DISPENSER = "pipetting_station.glb"
    """Two transparent reagent reservoirs above the dispensing instrument."""
    ROOM = "environment.glb"
    """Opaque buffer bottles placed around the laboratory."""


@dataclass(frozen=True)
class BottleLabel:
    """Named meshes belonging to one bottle and its printed paper label."""

    asset: BottleAsset
    """Model containing all surfaces for this bottle."""
    bottle: str
    """Outer vessel mesh used as the physical attachment surface."""
    paper: str
    """Thin backing carrying the printed text."""
    lettering: tuple[str, str]
    """Printed reagent name and second identification line."""


@dataclass(frozen=True)
class LabelSurfaces:
    """Transformed geometry for evaluating adhesion to a physical vessel."""

    bottle: trimesh.Trimesh
    """Complete bottle surface in model coordinates."""
    paper: trimesh.Trimesh
    """Label surface in the same coordinate frame."""
    lettering: tuple[trimesh.Trimesh, trimesh.Trimesh]
    """Text geometry in the same coordinate frame."""

    def radial_clearance(
        self, surface: trimesh.Trimesh, points: np.ndarray
    ) -> np.ndarray:
        """Measure outward clearance from the first physical surface on each ray."""
        center = self.bottle.bounds.mean(axis=0)
        directions = points - center
        directions[:, 2] = 0
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        origins = points + directions * np.linalg.norm(self.bottle.extents)
        locations, indices, _ = surface.ray.intersects_location(
            origins, -directions, multiple_hits=False
        )
        assert len(indices) == len(points), "Every sample must lie over its support."
        return np.einsum("ij,ij->i", points[indices] - locations, directions[indices])


def transformed_mesh(scene: trimesh.Scene, name: str) -> trimesh.Trimesh:
    """Resolve a named glTF node and retain its complete model transform."""
    transform, geometry = scene.graph[name]
    mesh = scene.geometry[geometry].copy()
    mesh.apply_transform(transform)
    return mesh


@pytest.fixture(scope="module")
def bottle_models() -> dict[BottleAsset, trimesh.Scene]:
    """Load the actual packaged visuals without merging independently named parts."""
    directory = (
        Path(__file__).resolve().parents[2]
        / "cramera/src/cramera/resources/precision_lab/assets"
    )
    return {
        asset: trimesh.load(
            directory / asset,
            force="scene",
            process=False,
            merge_primitives=True,
        )
        for asset in BottleAsset
    }


@pytest.fixture(
    params=[
        *(
            BottleLabel(
                BottleAsset.DISPENSER,
                f"Pipetting {channel} bottle",
                f"Pipetting {channel} bottle label",
                (
                    f"Pipetting {channel} label text",
                    f"Pipetting {channel} identification",
                ),
            )
            for channel in ("amber", "teal")
        ),
        *(
            BottleLabel(
                BottleAsset.ROOM,
                f"Reagent bottle{suffix}",
                f"Bottle label{suffix}",
                (f"Bottle lettering{suffix}", f"Bottle specification{suffix}"),
            )
            for suffix in ("", ".001", ".002", ".003", ".004")
        ),
    ],
    ids=lambda item: item.bottle,
)
def label_surfaces(
    request: pytest.FixtureRequest, bottle_models: dict[BottleAsset, trimesh.Scene]
) -> LabelSurfaces:
    """Pair each label with its own bottle rather than an ideal cylinder."""
    definition = request.param
    scene = bottle_models[definition.asset]
    return LabelSurfaces(
        transformed_mesh(scene, definition.bottle),
        transformed_mesh(scene, definition.paper),
        tuple(transformed_mesh(scene, name) for name in definition.lettering),
    )


# %% adhesion across the complete label


def test_paper_follows_bottle_surface_across_its_full_width(
    label_surfaces: LabelSurfaces,
) -> None:
    """Paper edges and face interiors stay outside and within thin-label clearance."""
    samples = np.vstack(
        (label_surfaces.paper.vertices, label_surfaces.paper.triangles_center)
    )
    clearance = label_surfaces.radial_clearance(label_surfaces.bottle, samples)
    maximum_paper_clearance = 0.00035
    export_tolerance = 0.000003
    assert clearance.min() >= -export_tolerance
    assert clearance.max() <= maximum_paper_clearance


def test_lettering_stays_printed_on_paper_across_each_character(
    label_surfaces: LabelSurfaces,
) -> None:
    """Both text lines follow the paper without floating or passing beneath it."""
    maximum_ink_clearance = 0.00015
    export_tolerance = 0.000003
    for lettering in label_surfaces.lettering:
        samples = np.vstack((lettering.vertices, lettering.triangles_center))
        clearance = label_surfaces.radial_clearance(label_surfaces.paper, samples)
        assert clearance.min() >= -export_tolerance
        assert clearance.max() <= maximum_ink_clearance
