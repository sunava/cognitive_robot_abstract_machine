"""
Export a reusable AICOR woodworking world with articulated storage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laboratory"))
from bundle import UrdfDescription
from spec import Asset, Geometry, PartKind
from rights import LaboratoryAssetRights

# %% portable resource authoring


@dataclass
class WorkshopBundle:
    """
    Generate metric descriptions around separately authored Blender visuals.
    """

    output: Path
    """Destination directory for the standalone world bundle."""
    geometry: Geometry = field(default_factory=Geometry)
    """
    Dimensions shared by manual controls and collision descriptions.
    """

    def environment(self) -> None:
        """
        Keep each drawer and handle in its own articulated link frame.
        """
        model = UrdfDescription.named(Asset.SCENE_NAME)
        room = model.link(Asset.ROOT, Asset.ENVIRONMENT_MESH)
        model.box(
            room,
            "worktop",
            (0, 0, self.geometry.bench_height - self.geometry.bench_thickness / 2),
            (2.2, 0.8, self.geometry.bench_thickness),
        )
        ground = model.link("workshop_floor")
        model.box(ground, "floor", (0, -0.05, -0.065), (4.7, 4.4, 0.13))
        model.joint("workshop_floor_joint", Asset.ROOT, "workshop_floor", (0, 0, 0))
        model.box(room, "rear_wall", (0, 1.00, 1.40), (4.7, 0.12, 2.8))
        model.box(room, "left_wall", (-2.31, 0.14, 1.40), (0.12, 1.75, 2.8))
        for x in (-0.95, 0.95):
            for y in (-0.30, 0.30):
                model.box(room, "bench_leg", (x, y, 0.433), (0.065, 0.065, 0.855))
        for x in (-1.044, -0.516):
            model.box(room, "cabinet_side", (x, 0.009, 0.497), (0.030, 0.64, 0.69))
        model.box(room, "cabinet_rear", (-0.78, 0.315, 0.497), (0.54, 0.025, 0.69))
        model.box(room, "cabinet_plinth", (-0.78, 0, 0.17), (0.55, 0.64, 0.034))
        model.box(room, "lower_wood_shelf", (0.36, 0, 0.183), (1.06, 0.57, 0.028))
        for y in (-0.305, 0.305):
            model.box(room, "bench_apron", (0, y, 0.81), (1.94, 0.046, 0.075))
        for x in (1.218, 1.742):
            for y in (-0.13, 0.43):
                model.box(room, "shelf_upright", (x, y, 0.825), (0.036, 0.036, 1.65))
        for z in (0.13, 0.53, 0.96, 1.55):
            model.box(room, "shelf_board", (1.48, 0.15, z), (0.55, 0.60, 0.025))
        for drawer in self.geometry.drawers():
            link = model.link(drawer["link"], Asset.DRAWER_MESH)
            self.open_box(model, link, self.geometry.drawer_size, 0.012)
            model.box(link, "front", (0, -0.285, 0.076), (0.48, 0.020, 0.148))
            model.inertia(link, 2.0, (0, 0, 0.07), self.geometry.drawer_size)
            model.joint(
                drawer["joint"],
                Asset.ROOT,
                drawer["link"],
                tuple(drawer["origin"]),
                self.geometry.drawer_travel,
            )
            handle = model.link(drawer["handle"])
            model.box(handle, "grasp_bar", (0, 0, 0), (0.158, 0.016, 0.021))
            model.joint(
                f"{drawer['handle']}_joint",
                drawer["link"],
                drawer["handle"],
                self.geometry.drawer_handle,
            )
        assembly = model.link(Asset.ASSEMBLY, Asset.ASSEMBLY_MESH)
        self.socket_collisions(model, assembly)
        model.joint(
            "workshop_assembly_joint",
            Asset.ROOT,
            Asset.ASSEMBLY,
            self.geometry.assembly_origin,
        )
        model.link(Asset.SOCKET)
        model.joint(
            "workshop_socket_joint",
            Asset.ASSEMBLY,
            Asset.SOCKET,
            (0, 0, self.geometry.socket_height),
        )
        model.write(self.output / Asset.ENVIRONMENT)

    def socket_collisions(self, model: UrdfDescription, link) -> None:
        """
        Approximate a circular pilot hole using a clear sixteen-sided bore.
        """
        width, depth, height = self.geometry.assembly_size
        radius = self.geometry.socket_radius
        # Horizontal strips leave a conservative circular opening through the block.
        divisions = 24
        for index in range(divisions):
            low = -depth / 2 + index * depth / divisions
            high = low + depth / divisions
            closest = 0.0 if low <= 0 <= high else min(abs(low), abs(high))
            opening = math.sqrt(max(0.0, radius * radius - closest * closest))
            if opening == 0.0:
                model.box(
                    link,
                    f"wood_{index}",
                    (0, (low + high) / 2, height / 2),
                    (width, depth / divisions, height),
                )
                continue
            segment_width = width / 2 - opening
            for side in (-1, 1):
                center_x = side * (opening + segment_width / 2)
                model.box(
                    link,
                    f"wood_{index}_{side}",
                    (center_x, (low + high) / 2, height / 2),
                    (segment_width, depth / divisions, height),
                )

    @staticmethod
    def open_box(
        model: UrdfDescription, link, size: tuple[float, float, float], wall: float
    ) -> None:
        """
        Describe a hollow box using a floor and four walls.
        """
        width, depth, height = size
        model.box(link, "floor", (0, 0, wall / 2), (width, depth, wall))
        for side in (-1, 1):
            model.box(
                link,
                f"side_{side}",
                (side * (width - wall) / 2, 0, height / 2),
                (wall, depth, height),
            )
            model.box(
                link,
                f"end_{side}",
                (0, side * (depth - wall) / 2, height / 2),
                (width - 2 * wall, wall, height),
            )

    def object(self, key: str, kind: PartKind) -> None:
        """
        Write one freely movable fastener or sorting box.
        """
        model = UrdfDescription.named(key)
        link = model.link(key, f"assets/{kind}.glb")
        geometry = self.geometry
        if kind is PartKind.BIN:
            self.open_box(model, link, geometry.bin_size, geometry.bin_wall)
            model.inertia(link, 0.20, (0, 0, 0.025), geometry.bin_size)
        else:
            length, radius, head_radius, head_thickness, mass = (
                (
                    geometry.screw_length,
                    geometry.screw_radius,
                    geometry.screw_head_radius,
                    geometry.screw_head_thickness,
                    0.095,
                )
                if kind is PartKind.SCREW
                else (
                    geometry.nail_length,
                    geometry.nail_radius,
                    geometry.nail_head_radius,
                    geometry.nail_head_thickness,
                    0.010,
                )
            )
            model.cylinder(link, "shaft", radius, length, length / 2)
            model.cylinder(
                link, "head", head_radius, head_thickness, length + head_thickness / 2
            )
            model.inertia(
                link,
                mass,
                (0, 0, length * 0.6),
                (2 * head_radius, 2 * head_radius, length + head_thickness),
            )
        model.write(self.output / f"{key}.urdf")

    def write(self) -> None:
        """
        Create URDF, interaction metadata and a one-frame reset trajectory.
        """
        self.output.mkdir(parents=True, exist_ok=True)
        self.environment()
        metadata = self.geometry.metadata()
        objects = []
        for item in [*metadata["parts"], *metadata["bins"]]:
            key = item["key"]
            kind = PartKind(item.get("kind", PartKind.BIN))
            self.object(key, kind)
            objects.append(
                {
                    "id": key,
                    "key": key,
                    "mesh": f"assets/{kind}.glb",
                    "urdf": f"{key}.urdf",
                    "spawn": item["home"],
                    "preserveMaterials": True,
                }
            )
        joints = [drawer["joint"] for drawer in metadata["drawers"]]
        scene = {
            "name": Asset.SCENE_NAME,
            "environmentName": "AICOR Wood Workshop",
            "task": "Woodworking · Articulated storage and loose fasteners",
            "rendering": {"ambientOcclusion": False, "exposure": 0.72},
            "framesPerSecond": 1,
            "trajectory": Asset.TRAJECTORY,
            "camera": metadata["cameras"]["overview"],
            "models": [
                {
                    "name": "workshop",
                    "urdf": Asset.ENVIRONMENT,
                    "prefix": "",
                    "robot": False,
                    "links": 10,
                    "movableJoints": joints,
                    "preserveMaterials": True,
                }
            ],
            "objects": objects,
            "workshop": metadata,
        }
        trajectory = {
            "framesPerSecond": 1,
            "frames": [{joint: 0.0 for joint in joints}],
            "objects": [{item["key"]: item["spawn"] for item in objects}],
        }
        semantics = {
            "name": Asset.SCENE_NAME,
            "units": "metres",
            "upAxis": "Z",
            "handles": [drawer["handle"] for drawer in metadata["drawers"]],
            "socket": {"frame": Asset.SOCKET, "radius": self.geometry.socket_radius},
            "workshop": metadata,
        }
        for filename, content in (
            (Asset.SCENE, scene),
            (Asset.TRAJECTORY, trajectory),
            (Asset.SEMANTICS, semantics),
        ):
            (self.output / filename).write_text(json.dumps(content, indent=2) + "\n")
        LaboratoryAssetRights(source=Path(__file__).with_name("asset_rights")).install(
            self.output
        )


def main() -> None:
    """
    Export the workshop bundle at a requested portable location.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    WorkshopBundle(options.output).write()


if __name__ == "__main__":
    main()
