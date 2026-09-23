"""
Check the hollow geometry before it is passed to Blender.
"""

import sys
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from math import isclose
from pathlib import Path
from types import ModuleType


# %% Asset module
def load_geometry() -> ModuleType:
    """
    Import the dependency-free geometry module from the asset tools.
    """
    path = (
        Path(__file__).parents[2] / "cramera" / "tools" / "laboratory" / "geometry.py"
    )
    specification = spec_from_file_location("laboratory_geometry", path)
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


# %% Hollow glass
class TestHollowGlass:
    """
    The glass wall encloses material without closing the mouth.
    """

    def test_profile_has_inner_and_outer_walls(self) -> None:
        """
        The straight tube wall keeps its declared thickness.
        """
        dimensions = load_geometry().TubeDimensions()
        profile = dimensions.profile()
        assert max(point.height for point in profile) == dimensions.height
        assert max(point.radius for point in profile) == dimensions.outer_radius
        assert any(
            point.radius == dimensions.inner_radius
            and point.height == dimensions.bottom_radius
            for point in profile
        )
        assert profile[0].radius == profile[-1].radius == 0
        assert isclose(
            profile[-1].height - profile[0].height, dimensions.wall_thickness
        )

    def test_surface_has_two_faces_per_edge(self) -> None:
        """
        Every glass material edge belongs to a closed manifold surface.
        """
        geometry = load_geometry()
        dimensions = geometry.TubeDimensions()
        mesh = geometry.LatheMesh.from_profile(dimensions.profile(), 64)
        edge_counts = Counter()
        for face in mesh.faces:
            for start, end in zip(face, (*face[1:], face[0])):
                edge_counts[tuple(sorted((start, end)))] += 1
        assert set(edge_counts.values()) == {2}

    def test_no_face_spans_the_tube_mouth(self) -> None:
        """
        Faces near the rim never enter the hollow tube opening.
        """
        geometry = load_geometry()
        dimensions = geometry.TubeDimensions()
        mesh = geometry.LatheMesh.from_profile(dimensions.profile(), 64)
        mouth_height = dimensions.height - dimensions.wall_thickness
        for face in mesh.faces:
            points = [mesh.vertices[index] for index in face]
            if min(point[2] for point in points) >= mouth_height:
                assert (
                    min((point[0] ** 2 + point[1] ** 2) ** 0.5 for point in points)
                    >= dimensions.inner_radius - 1e-10
                )
