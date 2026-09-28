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
    DRIVER = "driver"
    DRIVER_TIP = "driver_tip"
    DRIVER_GRASP = "driver_grasp"


class PartKind(StrEnum):
    """
    Independently manipulable workshop object types.
    """

    SCREW = "screw"
    NAIL = "nail"
    BIN = "bin"
    ASSEMBLY = "assembly"
    DRIVER = "driver"
    STORAGE_BOX = "storage_box"


class StructureKind(StrEnum):
    """
    Shared physical components of the bench and articulated cabinet.
    """

    LEG = "bench_leg"
    FOOT = "bench_foot"
    BRACE = "bench_lower_brace"
    LOWER_SHELF = "bench_wood_shelf"
    APRON = "bench_apron"
    CABINET_SIDE = "cabinet_side"
    CABINET_REAR = "cabinet_rear"
    PLINTH = "cabinet_plinth"
    CROSSRAIL = "cabinet_crossrail"
    SLIDE = "cabinet_slide"
    SHELF_STOCK = "shelf_stock"
    DRIVER_CRADLE = "driver_cradle"
    ASSEMBLY_JIG = "assembly_jig"


@dataclass(frozen=True)
class StructureBox:
    """
    A measured support shared by the visible model and collision description.
    """

    kind: StructureKind
    """
    Component role determining its visible finish.
    """

    centre: tuple[float, float, float]
    """
    World-coordinate centre in metres.
    """

    size: tuple[float, float, float]
    """
    Axis-aligned full dimensions in metres.
    """


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

    bin_handle: tuple[float, float, float] = (0.0, -0.204, 0.055)
    """
    Front carrying-bar grasp frame relative to the fastener box bottom centre.
    """

    bin_handle_size: tuple[float, float, float] = (0.070, 0.014, 0.018)
    """
    Carrying-bar dimensions allowing either a horizontal or vertical pinch.
    """

    bin_handle_mount: tuple[float, float, float] = (0.029, 0.021, 0.0)
    """
    One carrying-bar mount centre relative to its grasp frame, mirrored across X.
    """

    bin_handle_mount_size: tuple[float, float, float] = (0.012, 0.042, 0.018)
    """
    Connecting bracket dimensions between the carrying bar and front wall.
    """

    screw_box_origin: tuple[float, float, float] = (1.48, 0.10, 0.974)
    """
    Loaded screw box bottom centre on the right shelf, above its board surface.
    """

    screw_box_bench: tuple[float, float, float] = (-0.25, -0.22, 0.92)
    """
    Front worktop destination keeping the loaded box reachable from the clear aisle.
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

    assembly_size: tuple[float, float, float] = (0.25, 0.20, 0.14)
    """
    Wooden assembly block dimensions.
    """

    socket_radius: float = 0.014
    """
    Open pilot-hole radius, exceeding the screw shaft radius.
    """

    assembly_origin: tuple[float, float, float] = (0.24, 0.10, 1.11)
    """
    Upright assembly block rear-face centre in world coordinates.
    """

    socket_height: float = 0.10
    """
    Initial insertion height of the screw tip relative to the block bottom.
    """

    storage_box_size: tuple[float, float, float] = (0.24, 0.44, 0.23)
    """
    Dimensions of the independently carried shelf containers.
    """

    storage_box_wall: float = 0.010
    """
    Shelf container wall and floor thickness.
    """

    storage_box_handle: tuple[float, float, float] = (0.0, -0.246, 0.175)
    """
    Front carrying bar relative to the shelf container bottom centre.
    """

    screw_thread_pitch: float = 0.006
    """
    Axial advance per revolution of the coarse screw thread.
    """

    screw_slot_size: tuple[float, float, float] = (0.034, 0.006, 0.006)
    """
    Open recess length, width and depth in the screw head.
    """

    driver_blade_size: tuple[float, float, float] = (0.030, 0.004, 0.026)
    """
    Flat blade width, thickness and tip-to-shaft length.
    """

    driver_grasp: tuple[float, float, float] = (0.0, 0.0, 0.165)
    """
    Grasp centre inside the screwdriver handle, relative to its bit tip.
    """

    driver_origin: tuple[float, float, float] = (0.40, -0.30, 1.04)
    """
    Horizontal screwdriver tip on an elevated worktop cradle.
    """

    driver_shaft_radius: float = 0.0045
    """
    Cylindrical steel shaft radius used by its cradle and collision body.
    """

    driver_handle_radius: float = 0.024
    """
    Cylindrical handgrip radius used by its cradle and collision body.
    """

    def structure(self) -> list[StructureBox]:
        """
        Keep every support outside the complete swept volume of each drawer.
        """
        parts = []
        for x in (-1.075, 1.075):
            for y in (-0.30, 0.30):
                parts.append(
                    StructureBox(
                        StructureKind.LEG, (x, y, 0.4375), (0.045, 0.045, 0.835)
                    )
                )
                parts.append(
                    StructureBox(
                        StructureKind.FOOT, (x, y, 0.013), (0.065, 0.065, 0.026)
                    )
                )
            parts.append(
                StructureBox(StructureKind.BRACE, (x, 0, 0.15), (0.045, 0.60, 0.046))
            )
        parts.append(
            StructureBox(
                StructureKind.LOWER_SHELF, (0.27125, 0, 0.183), (1.5625, 0.57, 0.028)
            )
        )
        for y in (-0.24, 0.24):
            parts.append(
                StructureBox(
                    StructureKind.BRACE, (0.275, y, 0.151), (1.57, 0.035, 0.035)
                )
            )
        parts.extend(
            (
                StructureBox(
                    StructureKind.APRON, (0, 0.37, 0.81), (2.105, 0.046, 0.075)
                ),
                StructureBox(
                    StructureKind.APRON, (0.27875, -0.34, 0.81), (1.5475, 0.046, 0.075)
                ),
                StructureBox(
                    StructureKind.CABINET_REAR,
                    (-0.78, 0.315, 0.497),
                    (0.54, 0.025, 0.69),
                ),
                StructureBox(
                    StructureKind.PLINTH, (-0.78, 0, 0.17), (0.54, 0.64, 0.034)
                ),
            )
        )
        for x in (-1.038, -0.522):
            parts.append(
                StructureBox(
                    StructureKind.CABINET_SIDE, (x, 0.009, 0.497), (0.024, 0.64, 0.69)
                )
            )
        for z in (0.29, 0.48, 0.67, 0.845):
            parts.append(
                StructureBox(
                    StructureKind.CROSSRAIL, (-0.78, -0.304, z), (0.50, 0.023, 0.018)
                )
            )
        for z in (0.31, 0.50, 0.69):
            for x in (-1.023, -0.537):
                parts.append(
                    StructureBox(
                        StructureKind.SLIDE, (x, 0.015, z), (0.004, 0.50, 0.018)
                    )
                )
        return parts

    def shelf_stock(self) -> list[StructureBox]:
        """
        Store timber above the fastener shelf with matching visible and solid bounds.
        """
        return [
            StructureBox(
                StructureKind.SHELF_STOCK,
                (1.45, 0.15, 1.584 + index * 0.042),
                (0.43, 0.42, 0.042),
            )
            for index in range(4)
        ] + [
            StructureBox(
                StructureKind.SHELF_STOCK,
                (1.715, 0.22, 1.803),
                (0.045, 0.28, 0.48),
            )
        ]

    def driver_cradle(self) -> list[StructureBox]:
        """
        Support the screwdriver outside its finger grasp with a shaft groove.
        """
        x, y, z = self.driver_origin
        base_top = self.bench_height + 0.01
        parts = [
            StructureBox(
                StructureKind.DRIVER_CRADLE,
                (x + 0.15, y, self.bench_height + 0.005),
                (0.24, 0.08, 0.01),
            )
        ]
        for offset, radius in (
            (0.055, self.driver_shaft_radius),
            (0.217, self.driver_handle_radius),
        ):
            top = z - radius
            parts.append(
                StructureBox(
                    StructureKind.DRIVER_CRADLE,
                    (x + offset, y, (base_top + top) / 2),
                    (0.012, 0.012, top - base_top),
                )
            )
        for side in (-1, 1):
            parts.append(
                StructureBox(
                    StructureKind.DRIVER_CRADLE,
                    (x + 0.055, y + side * 0.0105, z),
                    (0.012, 0.004, 0.010),
                )
            )
        return parts

    def assembly(self) -> dict:
        """
        Describe the movable wooden workpiece and its attached pilot frame.
        """
        orientation = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
        home = [*self.assembly_origin, *orientation]
        local = [0, 0, self.socket_height, 0, 0, 0, 1]
        socket = [
            self.assembly_origin[0],
            self.assembly_origin[1] - self.socket_height,
            self.assembly_origin[2],
            *orientation,
        ]
        return {
            "key": Asset.ASSEMBLY,
            "label": "Wooden workpiece",
            "kind": PartKind.ASSEMBLY,
            "home": home,
            "pose": home,
            "socketPose": socket,
            "localSocket": local,
            "socketFrame": Asset.SOCKET,
            "localMouth": [0, 0, self.assembly_size[2]],
            "maximumDepth": self.screw_length,
        }

    def assembly_jig(self) -> list[StructureBox]:
        """
        Support the aisle-facing pilot block with a pedestal, rear brace and side stops.
        """
        x, y, z = self.assembly_origin
        width, depth, height = self.assembly_size
        floor = z - depth / 2
        center_y = y - height / 2
        shim_height = floor - self.bench_height
        parts = [
            StructureBox(
                StructureKind.ASSEMBLY_JIG,
                (x, center_y, self.bench_height + shim_height / 2),
                (width + 0.04, height + 0.04, shim_height),
            ),
            StructureBox(
                StructureKind.ASSEMBLY_JIG,
                (x, y + 0.010, floor + depth / 2),
                (width + 0.04, 0.02, depth),
            ),
        ]
        for side in (-1, 1):
            parts.append(
                StructureBox(
                    StructureKind.ASSEMBLY_JIG,
                    (x + side * (width / 2 + 0.007), center_y, floor + 0.0125),
                    (0.012, height + 0.02, 0.025),
                )
            )
        return parts

    def driver(self) -> dict:
        """
        Publish the screwdriver bit, handle and compatible blade dimensions.
        """
        return {
            "key": Asset.DRIVER,
            "label": "Screwdriver",
            "kind": PartKind.DRIVER,
            "home": [*self.driver_origin, 0, math.sqrt(0.5), 0, math.sqrt(0.5)],
            "bitTip": [0, 0, 0],
            "axis": [0, 0, -1],
            "grasp": list(self.driver_grasp),
            "bladeSize": list(self.driver_blade_size),
            "bladeDirection": [1, 0, 0],
            "bitFrame": Asset.DRIVER_TIP,
            "graspFrame": Asset.DRIVER_GRASP,
        }

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
        fasteners = [
            {
                "key": "screw_box",
                "label": "Screw box",
                "home": [*self.screw_box_origin, 0, 0, 0, 1],
                "benchPose": [*self.screw_box_bench, 0, 0, 0, 1],
                "handle": "screw_box_handle",
            },
            {
                "key": "nail_box",
                "label": "Nail box",
                "home": [-0.72, -0.10, self.bench_height, 0, 0, 0, 1],
                "handle": "nail_box_handle",
            },
        ]

        storage = [
            {
                "key": f"storage_box_{index}",
                "label": f"Storage box {index}",
                "kind": PartKind.STORAGE_BOX,
                "home": [x, 0.124, z, 0, 0, 0, 1],
                "handle": f"storage_box_{index}_handle",
            }
            for index, (x, z) in enumerate(
                ((1.343, 0.144), (1.617, 0.144), (1.343, 0.544), (1.617, 0.544)),
                start=1,
            )
        ]
        return fasteners + storage

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
                z = bin_item["home"][2] + self.bin_wall + radius + 0.001
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
        return {
            "drawers": self.drawers(),
            "parts": self.parts(),
            "bins": self.bins(),
            "assembly": self.assembly(),
            "driver": self.driver(),
            "screw": {
                "headTop": self.screw_length + self.screw_head_thickness,
                "slotSize": list(self.screw_slot_size),
                "threadPitch": self.screw_thread_pitch,
            },
            "liftHeight": 1.30,
            "bounds": {"min": [-1.15, -0.60], "max": [1.85, 0.55]},
            "cameras": {
                "overview": {"position": [2.4, -3.0, 2.15], "target": [0, 0.1, 1.0]},
                "bench": {"position": [1.3, -1.7, 1.8], "target": [-0.05, -0.05, 1.0]},
                "drawers": {
                    "position": [-1.8, -1.6, 1.0],
                    "target": [-0.78, -0.1, 0.5],
                },
            },
        }
