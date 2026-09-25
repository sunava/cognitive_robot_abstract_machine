"""
Metric placement contract for the AICOR woodworking workshop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

# %% authored names


class Asset(StrEnum):
    """
    Stable portable filenames and scene identity.
    """

    SCENE_NAME = "aicor_workshop"
    SCENE = "scene.json"
    TRAJECTORY = "trajectory.json"
    SEMANTICS = "semantics.json"
    ENVIRONMENT = "environment.urdf"
    ENVIRONMENT_MESH = "assets/environment.glb"
    DRAWER_MESH = "assets/drawer.glb"
    ASSEMBLY_MESH = "assets/assembly.glb"
    ROOT = "workshop_environment"
    ASSEMBLY = "workshop_assembly"
    SOCKET = "workshop_socket"


class PartKind(StrEnum):
    """
    Independently manipulable fastener types.
    """

    SCREW = "screw"
    NAIL = "nail"
    BIN = "bin"


@dataclass(frozen=True)
class Geometry:
    """
    Shared metre dimensions of authored workshop assets.
    """

    bench_height: float = 0.92
    """
    Upper worktop surface in world coordinates.
    """

    bench_thickness: float = 0.065
    """
    Solid wooden worktop thickness.
    """

    drawer_size: tuple[float, float, float] = (0.48, 0.56, 0.14)
    """
    Drawer dimensions before the handle and front are added.
    """

    drawer_travel: float = 0.40
    """
    Maximum frontward drawer extension.
    """

    drawer_handle: tuple[float, float, float] = (0.0, -0.318, 0.085)
    """
    Handle grasp frame relative to the drawer bottom centre.
    """

    bin_size: tuple[float, float, float] = (0.28, 0.32, 0.075)
    """
    Open box dimensions.
    """

    bin_wall: float = 0.006
    """
    Box floor and wall thickness.
    """

    screw_length: float = 0.11
    """
    Screw shaft length from its tip frame.
    """

    screw_radius: float = 0.012
    """
    Bounding shaft radius including the coarse thread.
    """

    screw_head_radius: float = 0.026
    """
    Hexagonal head circumradius.
    """

    screw_head_thickness: float = 0.018
    """
    Head thickness above the shaft.
    """

    nail_length: float = 0.075
    """
    Nail shaft length from its tip frame.
    """

    nail_radius: float = 0.0035
    """
    Nail shaft radius.
    """

    nail_head_radius: float = 0.009
    """
    Nail head radius.
    """

    nail_head_thickness: float = 0.006
    """
    Nail head thickness.
    """

    assembly_size: tuple[float, float, float] = (0.25, 0.20, 0.075)
    """
    Wooden assembly block dimensions.
    """

    socket_radius: float = 0.014
    """
    Open pilot-hole radius, exceeding the screw shaft radius.
    """

    assembly_origin: tuple[float, float, float] = (0.28, -0.10, 0.92)
    """
    Assembly block bottom centre in world coordinates.
    """

    socket_height: float = 0.042
    """
    Final screw-tip height relative to the block bottom.
    """

    def drawers(self) -> list[dict]:
        """
        Expose independent joints, moving handle frames and manual limits.
        """
        return [
            {
                "id": f"drawer_{index}",
                "label": f"{label} drawer",
                "joint": f"workshop_drawer_{index}_joint",
                "link": f"workshop_drawer_{index}",
                "handle": f"workshop_drawer_{index}_handle",
                "origin": [-0.78, 0.0, height],
                "open": self.drawer_travel,
            }
            for index, (label, height) in enumerate(
                (("Lower", 0.30), ("Middle", 0.49), ("Upper", 0.68)), start=1
            )
        ]

    def bins(self) -> list[dict]:
        """
        Expose independently movable sorting boxes.
        """
        return [
            {
                "key": "screw_box",
                "label": "Screw box",
                "home": [-0.25, -0.10, self.bench_height, 0, 0, 0, 1],
            },
            {
                "key": "nail_box",
                "label": "Nail box",
                "home": [-0.72, -0.10, self.bench_height, 0, 0, 0, 1],
            },
        ]

    def parts(self) -> list[dict]:
        """
        Place ten horizontal fasteners with clearance between their heads.
        """
        quaternion = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
        parts = []
        for kind, bin_item, radius in (
            (PartKind.SCREW, self.bins()[0], self.screw_head_radius),
            (PartKind.NAIL, self.bins()[1], self.nail_head_radius),
        ):
            for index, (column, row) in enumerate(
                ((0, 0), (0, -1), (0, 1), (1, -1), (1, 1)), start=1
            ):
                x = bin_item["home"][0] - 0.128 + column * 0.130
                y = bin_item["home"][1] + row * 0.12
                z = self.bench_height + self.bin_wall + radius + 0.001
                parts.append(
                    {
                        "key": f"{kind}_{index}",
                        "label": f"{kind.title()} {index}",
                        "kind": kind,
                        "home": [x, y, z, *quaternion],
                    }
                )
        return parts

    def metadata(self) -> dict:
        """
        Publish the geometric interaction contract for the viewer and plans.
        """
        origin = list(self.assembly_origin)
        socket = [origin[0], origin[1], origin[2] + self.socket_height, 0, 0, 0, 1]
        return {
            "drawers": self.drawers(),
            "parts": self.parts(),
            "bins": self.bins(),
            "assembly": {"pose": [*origin, 0, 0, 0, 1], "socketPose": socket},
            "liftHeight": 1.30,
            "bounds": {"min": [-1.05, -0.36], "max": [1.05, 0.10]},
            "cameras": {
                "overview": {"position": [2.4, -3.0, 2.15], "target": [0, 0.1, 1.0]},
                "bench": {"position": [1.3, -1.7, 1.8], "target": [-0.05, -0.05, 1.0]},
                "drawers": {
                    "position": [-1.8, -1.6, 1.0],
                    "target": [-0.78, -0.1, 0.5],
                },
            },
        }
