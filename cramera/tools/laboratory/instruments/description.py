"""Write portable URDFs, conservative contact geometry and interaction metadata."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from spec import DeviceSpec, JointKind, ShapeKind, ShapeSpec

# %% frame calculations


def rotation_matrix(angles: tuple[float, float, float]) -> np.ndarray:
    """Return the URDF roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = angles
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def rotation_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Recover one roll-pitch-yaw representation of a rotation."""
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1, 1)))
    if abs(math.cos(pitch)) < 1e-8:
        return (0.0, pitch, math.atan2(-matrix[0, 1], matrix[1, 1]))
    return (
        math.atan2(matrix[2, 1], matrix[2, 2]),
        pitch,
        math.atan2(matrix[1, 0], matrix[0, 0]),
    )


def numbers(values: Iterable[float]) -> str:
    """Serialize finite metric coordinates without losing small handle clearances."""
    return " ".join(f"{float(value):.9g}" for value in values)


@dataclass(frozen=True)
class ContactPrimitive:
    """One convex primitive in a rigid body's local frame."""

    kind: str
    """URDF geometry element name."""
    center: tuple[float, float, float]
    """Primitive center in metres."""
    rotation: tuple[float, float, float]
    """Local roll-pitch-yaw orientation."""
    dimensions: tuple[float, ...]
    """Box extents, cylinder radius and depth, or sphere radius."""

    @classmethod
    def from_shape(cls, shape: ShapeSpec) -> list[ContactPrimitive]:
        """Approximate only authored solid walls, preserving open insertion bores."""
        if not shape.collision:
            return []
        if shape.kind in (ShapeKind.BOX, ShapeKind.CYLINDER, ShapeKind.SPHERE):
            return [
                cls(shape.kind.value, shape.center, shape.rotation, shape.dimensions)
            ]
        if shape.kind == ShapeKind.CONE:
            bottom, top, height = shape.dimensions
            return [
                cls(
                    "cylinder", shape.center, shape.rotation, (max(bottom, top), height)
                )
            ]
        if shape.kind == ShapeKind.RING:
            outer, inner, height = shape.dimensions
            count = max(
                24, min(64, math.ceil(math.pi * math.sqrt(outer / (outer - inner))))
            )
            radius = (outer + inner) / 2
            orientation = rotation_matrix(shape.rotation)
            result = []
            for index in range(count):
                angle = math.tau * index / count
                center = np.asarray(shape.center) + orientation @ np.array(
                    (radius * math.cos(angle), radius * math.sin(angle), 0)
                )
                rotation = rotation_angles(orientation @ rotation_matrix((0, 0, angle)))
                result.append(
                    cls(
                        "box",
                        tuple(center),
                        rotation,
                        (outer - inner, 2 * radius * math.tan(math.pi / count), height),
                    )
                )
            return result
        return []

    def half_extents(self) -> np.ndarray:
        """Return a conservative local bounding box for inertia estimation."""
        if self.kind == "box":
            return np.asarray(self.dimensions) / 2
        if self.kind == "sphere":
            return np.repeat(self.dimensions[0], 3)
        radius, height = self.dimensions
        return np.array((radius, radius, height / 2))


# %% portable descriptions


@dataclass
class InstrumentDescription:
    """Serialize an authored device without depending on Blender or CRAMERA."""

    model: DeviceSpec
    """Metric articulated device definition."""

    def link_name(self, local_name: str) -> str:
        """Namespace every link and target within the independent device."""
        return f"{self.model.identifier}_{local_name}"

    def write(self, directory: Path) -> None:
        """Write the complete local frame tree and its manipulation metadata."""
        self.model.validate()
        directory.mkdir(parents=True, exist_ok=True)
        root = ElementTree.Element("robot", name=self.model.identifier)
        moving = []
        for part in self.model.parts:
            name = self.link_name(part.name)
            link = ElementTree.SubElement(root, "link", name=name)
            if part.shapes:
                visual = ElementTree.SubElement(link, "visual")
                geometry = ElementTree.SubElement(visual, "geometry")
                ElementTree.SubElement(
                    geometry, "mesh", filename=f"assets/{part.name}.glb"
                )
            contacts = [
                primitive
                for shape in part.shapes
                for primitive in ContactPrimitive.from_shape(shape)
            ]
            for index, contact in enumerate(contacts):
                collision = ElementTree.SubElement(
                    link, "collision", name=f"{name}_contact_{index}"
                )
                ElementTree.SubElement(
                    collision,
                    "origin",
                    xyz=numbers(contact.center),
                    rpy=numbers(contact.rotation),
                )
                geometry = ElementTree.SubElement(collision, "geometry")
                if contact.kind == "box":
                    ElementTree.SubElement(
                        geometry, "box", size=numbers(contact.dimensions)
                    )
                elif contact.kind == "sphere":
                    ElementTree.SubElement(
                        geometry, "sphere", radius=numbers(contact.dimensions)
                    )
                else:
                    ElementTree.SubElement(
                        geometry,
                        "cylinder",
                        radius=str(contact.dimensions[0]),
                        length=str(contact.dimensions[1]),
                    )
            solids = [
                primitive
                for shape in part.shapes
                for primitive in ContactPrimitive.from_shape(
                    replace(shape, collision=True)
                )
            ]
            if solids:
                lower = np.min(
                    [
                        np.asarray(item.center)
                        - abs(rotation_matrix(item.rotation)) @ item.half_extents()
                        for item in solids
                    ],
                    axis=0,
                )
                upper = np.max(
                    [
                        np.asarray(item.center)
                        + abs(rotation_matrix(item.rotation)) @ item.half_extents()
                        for item in solids
                    ],
                    axis=0,
                )
                dimensions = np.maximum(upper - lower, 0.001)
                inertia = part.mass / 12 * (sum(dimensions**2) - dimensions**2)
                inertial = ElementTree.SubElement(link, "inertial")
                ElementTree.SubElement(
                    inertial, "origin", xyz=numbers((upper + lower) / 2)
                )
                ElementTree.SubElement(inertial, "mass", value=str(part.mass))
                ElementTree.SubElement(
                    inertial,
                    "inertia",
                    ixx=str(inertia[0]),
                    iyy=str(inertia[1]),
                    izz=str(inertia[2]),
                    ixy="0",
                    ixz="0",
                    iyz="0",
                )
            if part.parent is not None:
                joint = self._joint(
                    root,
                    name,
                    self.link_name(part.parent),
                    part.joint,
                    part.origin,
                    part.rotation,
                )
                if part.joint != JointKind.FIXED:
                    ElementTree.SubElement(joint, "axis", xyz=numbers(part.axis))
                    limits = {"effort": "20", "velocity": "1"}
                    if part.joint != JointKind.CONTINUOUS:
                        limits.update(lower=str(part.lower), upper=str(part.upper))
                    ElementTree.SubElement(joint, "limit", **limits)
                    ElementTree.SubElement(
                        joint, "dynamics", damping="0.1", friction="0.02"
                    )
                    moving.append(
                        {
                            "name": f"{name}_joint",
                            "type": part.joint.value,
                            "lower": part.lower,
                            "upper": part.upper,
                            "inspectionPosition": part.open_value,
                        }
                    )
        for frame in self.model.frames:
            name = self.link_name(frame.name)
            ElementTree.SubElement(root, "link", name=name)
            self._joint(
                root,
                name,
                self.link_name(frame.parent),
                JointKind.FIXED,
                frame.position,
                frame.rotation,
            )
        ElementTree.indent(root)
        ElementTree.ElementTree(root).write(
            directory / "model.urdf", encoding="utf-8", xml_declaration=True
        )
        metadata = {
            "device": self.model.identifier,
            "title": self.model.title,
            "description": self.model.description,
            "rootLink": self.link_name(self.model.parts[0].name),
            "units": "metres",
            "upAxis": "Z",
            "frontAxis": "-Y",
            "joints": moving,
            "frames": [
                {
                    "name": self.link_name(frame.name),
                    "parent": self.link_name(frame.parent),
                    "position": frame.position,
                    "rotation": frame.rotation,
                    "role": frame.role,
                }
                for frame in self.model.frames
            ],
            "accessories": self.model.accessories,
            "blender": "model.blend",
            "previews": {"closed": "preview.png", "open": "preview_open.png"},
            "modelScope": "Original articulated concept model. Instrument controls and measurements require a separate controller.",
            "inertialModel": "Nominal masses with compound-bounds box inertia; not calibrated hardware parameters.",
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (directory / "definition.json").write_text(
            json.dumps(asdict(self.model), indent=2) + "\n"
        )
        lines = [
            f"# {self.model.title}",
            "",
            self.model.description,
            "",
            "Original AICOR concept model. Principal author and design direction: Vanessa Hassouna.",
            "",
            "`model.blend` is editable in Blender 4.0 or newer. Its timeline shows closed/rest at frame 1,",
            "an inspection pose at frame 60, and closed/rest again at frame 120.",
            "`model.urdf` loads independently using only local `assets/*.glb` files.",
            "Copy this entire folder, including its notices, for an authorized reuse.",
            "",
            "Coordinates are metres, Z up, front toward negative Y. Use a unique URDF parser prefix for each instance.",
            "Collision sleeves retain open bores; casing contacts use convex primitives.",
            "Masses/inertias are approximate model parameters, not manufacturer specifications.",
            "",
            "## Moving parts",
            "",
            "| Joint | Motion | Inspection position |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| `{item['name']}` | {item['type']} | {item['inspectionPosition']:.4g} |"
            for item in moving
        ]
        lines += [
            "",
            "## Interaction frames",
            "",
            "| Frame | Attached to | Purpose |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| `{self.link_name(frame.name)}` | `{self.link_name(frame.parent)}` | {frame.role} |"
            for frame in self.model.frames
        ]
        lines += [
            "",
            "The URDF supplies geometry, articulation and frames. It does not itself execute dosing,",
            "mixing, heating, optical measurements, printing or biological processes.",
            "",
        ]
        (directory / "README.md").write_text("\n".join(lines))

    @staticmethod
    def _joint(
        root: ElementTree.Element,
        child: str,
        parent: str,
        kind: JointKind,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ) -> ElementTree.Element:
        """Attach a named body or target to its owning frame."""
        joint = ElementTree.SubElement(
            root, "joint", name=f"{child}_joint", type=kind.value
        )
        ElementTree.SubElement(joint, "parent", link=parent)
        ElementTree.SubElement(joint, "child", link=child)
        ElementTree.SubElement(
            joint, "origin", xyz=numbers(position), rpy=numbers(rotation)
        )
        return joint
