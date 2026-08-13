"""
Generate the wind-turbine rotor blade mesh used by ``wind_turbine_assembly.urdf``.

Lofts a circular blade root into a cambered airfoil that tapers and untwists
towards the tip, producing ``meshes/turbine_blade.stl`` next to this script. The
blade lies along +x with its root flange at the origin.

Run once (the STL is checked in)::

    python generate_turbine_blade_mesh.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import trimesh

# %% blade dimensions

BLADE_LENGTH = 4.0
"""
Root-to-tip length of the blade, in meters.
"""

ROOT_RADIUS = 0.16
"""
Radius of the circular root flange, in meters.
"""

MAXIMUM_CHORD = 0.58
"""
The widest chord of the blade, reached a quarter of the way to the tip, in meters.
"""

TIP_CHORD = 0.09
"""
Chord right before the tip closes, in meters.
"""

MAXIMUM_CHORD_STATION = 0.28
"""
Fraction of the blade length at which the chord is widest.
"""

ROOT_BLEND_END = 0.18
"""
Fraction of the blade length over which the circular root blends into the airfoil.
"""

ROOT_TWIST = math.radians(16.0)
"""
Twist of the airfoil at the root end of the blend, in radians; fades to zero at the tip.
"""

THICKNESS_RATIO_ROOT = 0.30
"""
Airfoil thickness as a fraction of the chord where the root blend ends.
"""

THICKNESS_RATIO_TIP = 0.10
"""
Airfoil thickness as a fraction of the chord at the tip.
"""

CAMBER_RATIO = 0.035
"""
Maximum camber of the airfoil as a fraction of the chord.
"""

PIVOT_CHORD_FRACTION = 0.35
"""
Chord fraction the profile is centered on, so the blade twists about its spar line.
"""

STATION_COUNT = 48
"""
Number of cross sections along the blade.
"""

RING_POINT_COUNT = 64
"""
Number of points around each cross section.
"""

# %% cross-section profiles


def airfoil_half_thickness(chord_position: np.ndarray, ratio: float) -> np.ndarray:
    """
    Half thickness of a NACA four-digit airfoil along the chord.

    :param chord_position: Positions along the chord in ``[0, 1]``.
    :param ratio: Thickness as a fraction of the chord.
    :return: Half thickness at each position, as a fraction of the chord.
    """
    return (
        5.0
        * ratio
        * (
            0.2969 * np.sqrt(chord_position)
            - 0.1260 * chord_position
            - 0.3516 * chord_position**2
            + 0.2843 * chord_position**3
            - 0.1015 * chord_position**4
        )
    )


@dataclass
class BladeStation:
    """
    One cross section of the blade.
    """

    span_fraction: float
    """
    Position along the blade as a fraction of :data:`BLADE_LENGTH`.
    """

    @property
    def chord(self) -> float:
        """
        Chord length at this station, widening to the maximum and tapering to the tip.
        """
        t = self.span_fraction
        if t <= MAXIMUM_CHORD_STATION:
            blend = t / MAXIMUM_CHORD_STATION
            return 2 * ROOT_RADIUS + (MAXIMUM_CHORD - 2 * ROOT_RADIUS) * math.sin(
                blend * math.pi / 2
            )
        taper = (t - MAXIMUM_CHORD_STATION) / (1.0 - MAXIMUM_CHORD_STATION)
        return MAXIMUM_CHORD + (TIP_CHORD - MAXIMUM_CHORD) * taper**0.9

    @property
    def thickness_ratio(self) -> float:
        """
        Airfoil thickness ratio at this station, thinning towards the tip.
        """
        return (
            THICKNESS_RATIO_ROOT
            + (THICKNESS_RATIO_TIP - THICKNESS_RATIO_ROOT) * self.span_fraction**0.8
        )

    @property
    def twist(self) -> float:
        """
        Twist angle at this station, in radians, fading from the root to the tip.
        """
        return ROOT_TWIST * (1.0 - self.span_fraction) ** 1.6

    @property
    def circle_weight(self) -> float:
        """
        How circular this station still is: 1 at the root flange, 0 past the blend.
        """
        if self.span_fraction >= ROOT_BLEND_END:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * self.span_fraction / ROOT_BLEND_END))

    def ring(self) -> np.ndarray:
        """
        The cross section as ``(RING_POINT_COUNT, 3)`` vertices in blade coordinates.
        """
        angles = np.linspace(0.0, 2.0 * math.pi, RING_POINT_COUNT, endpoint=False)
        chord_position = 0.5 * (1.0 + np.cos(angles))
        half_thickness = airfoil_half_thickness(chord_position, self.thickness_ratio)
        camber = CAMBER_RATIO * np.sin(math.pi * chord_position)
        sign = np.where(np.sin(angles) >= 0.0, 1.0, -1.0)
        airfoil_y = -(chord_position - PIVOT_CHORD_FRACTION) * self.chord
        airfoil_z = (camber + sign * half_thickness) * self.chord
        circle_y = ROOT_RADIUS * np.cos(angles)
        circle_z = ROOT_RADIUS * np.sin(angles)
        weight = self.circle_weight
        y = weight * circle_y + (1.0 - weight) * airfoil_y
        z = weight * circle_z + (1.0 - weight) * airfoil_z
        twisted_y = y * math.cos(self.twist) - z * math.sin(self.twist)
        twisted_z = y * math.sin(self.twist) + z * math.cos(self.twist)
        x = np.full_like(y, self.span_fraction * BLADE_LENGTH)
        return np.column_stack([x, twisted_y, twisted_z])


# %% lofting


def loft_blade() -> trimesh.Trimesh:
    """
    :return: The closed blade surface lofted through all stations.
    """
    stations = [
        BladeStation(span_fraction=t) for t in np.linspace(0.0, 1.0, STATION_COUNT)
    ]
    rings = [station.ring() for station in stations]
    vertices = np.concatenate(rings)
    faces = []
    for station_index in range(STATION_COUNT - 1):
        base = station_index * RING_POINT_COUNT
        for point_index in range(RING_POINT_COUNT):
            next_point = (point_index + 1) % RING_POINT_COUNT
            a = base + point_index
            b = base + next_point
            c = a + RING_POINT_COUNT
            d = b + RING_POINT_COUNT
            faces.append([a, b, c])
            faces.append([b, d, c])
    root_center = len(vertices)
    tip_center = root_center + 1
    vertices = np.vstack([vertices, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
    for point_index in range(RING_POINT_COUNT):
        next_point = (point_index + 1) % RING_POINT_COUNT
        faces.append([root_center, next_point, point_index])
        tip_base = (STATION_COUNT - 1) * RING_POINT_COUNT
        faces.append([tip_center, tip_base + point_index, tip_base + next_point])
    blade = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    blade.fix_normals()
    return blade


def main() -> None:
    """
    Write ``meshes/turbine_blade.stl`` next to this script.
    """
    blade = loft_blade()
    output = os.path.join(os.path.dirname(__file__), "meshes", "turbine_blade.stl")
    blade.export(output)
    print(
        "wrote %s: %d vertices, %d faces, watertight=%s"
        % (output, len(blade.vertices), len(blade.faces), blade.is_watertight)
    )


if __name__ == "__main__":
    main()
