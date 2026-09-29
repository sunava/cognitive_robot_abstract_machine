"""Validate portable instrument URDFs and real glTF geometry without Blender.

Run with ``python validate_packages.py --root RESOURCE_DIRECTORY``. Select an
individual completed package with ``--device IDENTIFIER`` while others build.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import trimesh
from PIL import Image

from catalog import build_devices
from description import rotation_matrix
from spec import DeviceSpec, JointKind, ShapeKind


# %% Validation results
class InvalidPackage(ValueError):
    """A generated device cannot load independently with its described articulation."""


@dataclass(frozen=True)
class MeshSummary:
    """Measured surfaces decoded from a part's exported mesh."""

    path: str
    """Package-local mesh filename."""
    triangles: int
    """Number of actually decoded triangle faces."""
    lower: list[float]
    """Minimum local coordinates measured from the glTF scene."""
    upper: list[float]
    """Maximum local coordinates measured from the glTF scene."""


@dataclass(frozen=True)
class PackageReport:
    """Verified package contents and the measured exported geometry."""

    device: str
    """Stable instrument identifier."""
    links: int
    """Number of connected physical and interaction frames."""
    moving_joints: int
    """Number of bounded or continuous movable connections."""
    interaction_frames: int
    """Number of correctly attached semantic targets."""
    meshes: list[MeshSummary]
    """Decoded surfaces for all physical bodies."""


# %% Independent package checks
@dataclass(frozen=True)
class PackageValidator:
    """Inspect serialized assets independently of Blender's authoring state."""

    directory: Path
    """Self-contained folder holding one instrument."""

    def require(self, condition: bool, message: str) -> None:
        """Report a broken package invariant with the affected resource name."""
        if not condition:
            raise InvalidPackage(f"{self.directory.name}: {message}")

    def local_file(self, reference: str) -> Path:
        """Resolve a dependency only when it remains inside the copied package."""
        self.require(
            bool(reference) and not urlsplit(reference).scheme,
            f"nonlocal reference {reference!r}",
        )
        path = Path(reference)
        self.require(
            not path.is_absolute() and ".." not in path.parts,
            f"escaping reference {reference!r}",
        )
        resolved = (self.directory / path).resolve()
        self.require(
            resolved.is_relative_to(self.directory.resolve()),
            f"external symlink {reference!r}",
        )
        self.require(resolved.is_file(), f"missing file {reference!r}")
        return resolved

    def vector(self, text: str, size: int = 3) -> np.ndarray:
        """Read an explicitly finite spatial vector of the required dimension."""
        values = np.asarray([float(value) for value in text.split()])
        self.require(
            values.shape == (size,) and np.isfinite(values).all(),
            "nonfinite or malformed coordinates",
        )
        return values

    def validate_definition(self, model: DeviceSpec) -> None:
        """Require regeneration when packaged geometry differs from its current source."""
        exported = json.loads(self.local_file("definition.json").read_text())
        authored = json.loads(json.dumps(asdict(model)))
        self.require(
            exported == authored,
            "generated definition differs from current model source; rebuild this resource",
        )

    def validate_structure(self) -> None:
        """Verify the actual URDF tree, joint ranges, inertias and declared frames."""
        root = ElementTree.parse(self.local_file("model.urdf")).getroot()
        metadata = json.loads(self.local_file("metadata.json").read_text())
        definition = json.loads(self.local_file("definition.json").read_text())
        self.require(root.tag == "robot", "URDF root is not a robot")
        self.require(
            root.get("name")
            == metadata["device"]
            == definition["identifier"]
            == self.directory.name,
            "inconsistent resource identity",
        )
        self.require(
            metadata["units"] == "metres"
            and metadata["upAxis"] == "Z"
            and metadata["frontAxis"] == "-Y",
            "unsupported authored coordinate convention",
        )
        links = root.findall("link")
        joints = root.findall("joint")
        link_names = {link.get("name") for link in links}
        joint_names = {joint.get("name") for joint in joints}
        self.require(
            len(link_names) == len(links) and None not in link_names,
            "duplicate or unnamed links",
        )
        self.require(
            len(joint_names) == len(joints) and None not in joint_names,
            "duplicate or unnamed joints",
        )
        parents: dict[str, str] = {}
        children: dict[str, list[str]] = {name: [] for name in link_names}
        for joint in joints:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            self.require(
                parent in link_names and child in link_names,
                "joint references missing link",
            )
            self.require(
                child not in parents and parent != child,
                "multiply parented or self-referencing link",
            )
            parents[child] = parent
            children[parent].append(child)
            self._validate_joint(joint)
        roots = link_names - set(parents)
        self.require(
            roots == {metadata["rootLink"]},
            "root frame mismatch or disconnected forest",
        )
        pending = list(roots)
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            self.require(current not in visited, "cyclic URDF graph")
            visited.add(current)
            pending.extend(children[current])
        self.require(visited == link_names, "disconnected URDF cycle or body")
        self._validate_joint_metadata(joints, metadata["joints"])
        self._validate_frames(joints, metadata["frames"], definition)
        self._validate_bodies(root, definition)
        for link in links:
            inertial = link.find("inertial")
            if inertial is not None:
                mass = float(inertial.find("mass").get("value"))
                self.require(math.isfinite(mass) and mass > 0, "invalid body mass")
                attributes = inertial.find("inertia").attrib
                inertia = np.array(
                    [
                        [
                            float(
                                attributes[f"i{row}{column}"]
                                if f"i{row}{column}" in attributes
                                else attributes[f"i{column}{row}"]
                            )
                            for column in "xyz"
                        ]
                        for row in "xyz"
                    ]
                )
                self.require(
                    np.isfinite(inertia).all()
                    and (np.linalg.eigvalsh(inertia) > 0).all(),
                    "invalid inertia tensor",
                )
            for collision in link.findall("collision"):
                for origin in collision.findall("origin"):
                    self.vector(origin.get("xyz", "0 0 0"))
                    self.vector(origin.get("rpy", "0 0 0"))
                geometry = collision.find("geometry")
                self.require(
                    geometry is not None and len(geometry) == 1,
                    "missing collision geometry",
                )
                primitive = list(geometry)[0]
                if primitive.tag == "box":
                    dimensions = self.vector(primitive.get("size"))
                elif primitive.tag == "cylinder":
                    dimensions = np.asarray(
                        [float(primitive.get("radius")), float(primitive.get("length"))]
                    )
                elif primitive.tag == "sphere":
                    dimensions = np.asarray([float(primitive.get("radius"))])
                else:
                    raise InvalidPackage(
                        f"{self.directory.name}: unexpected collision primitive {primitive.tag}"
                    )
                self.require(
                    np.isfinite(dimensions).all() and (dimensions > 0).all(),
                    "nonpositive collision geometry",
                )

    def _validate_joint(self, joint: ElementTree.Element) -> None:
        """Check articulation units, directions and finite positive actuator limits."""
        kind = joint.get("type")
        self.require(
            kind in {member.value for member in JointKind}, "unsupported joint type"
        )
        origin = joint.find("origin")
        self.require(origin is not None, "missing joint origin")
        self.vector(origin.get("xyz", "0 0 0"))
        self.vector(origin.get("rpy", "0 0 0"))
        if kind == JointKind.FIXED:
            return
        axis = joint.find("axis")
        self.require(axis is not None, "missing moving-joint axis")
        self.require(
            np.isclose(np.linalg.norm(self.vector(axis.get("xyz"))), 1),
            "joint axis is not a unit vector",
        )
        limit = joint.find("limit")
        self.require(limit is not None, "missing actuator limits")
        for field in ("effort", "velocity"):
            value = float(limit.get(field))
            self.require(math.isfinite(value) and value > 0, "invalid actuator limit")
        if kind != JointKind.CONTINUOUS:
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            self.require(
                math.isfinite(lower) and math.isfinite(upper) and lower < upper,
                "invalid bounded joint interval",
            )

    def _validate_joint_metadata(
        self, joints: list[ElementTree.Element], records: list[dict]
    ) -> None:
        """Ensure inspection poses and limits agree with the loadable URDF."""
        moving = {
            joint.get("name"): joint
            for joint in joints
            if joint.get("type") != JointKind.FIXED
        }
        self.require(
            set(moving) == {record["name"] for record in records}
            and len(moving) == len(records),
            "moving joint metadata mismatch",
        )
        for record in records:
            joint = moving[record["name"]]
            self.require(
                joint.get("type") == record["type"], "joint kind metadata mismatch"
            )
            value = float(record["inspectionPosition"])
            self.require(math.isfinite(value), "nonfinite inspection pose")
            if joint.get("type") != JointKind.CONTINUOUS:
                limit = joint.find("limit")
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                self.require(
                    lower == record["lower"] and upper == record["upper"],
                    "joint bound metadata mismatch",
                )
                self.require(
                    lower <= value <= upper, "inspection pose outside joint limits"
                )

    def _validate_frames(
        self, joints: list[ElementTree.Element], records: list[dict], definition: dict
    ) -> None:
        """Verify every metadata target follows its physical body with the authored pose."""
        connections = {joint.find("child").get("link"): joint for joint in joints}
        prefix = definition["identifier"] + "_"
        self.require(
            {record["name"] for record in records}
            == {prefix + frame["name"] for frame in definition["frames"]},
            "missing or extra interaction frames",
        )
        self.require(
            len(records) == len({record["name"] for record in records}),
            "duplicate interaction frames",
        )
        for record in records:
            self.require(
                record["name"] in connections and bool(record["role"]),
                "unconnected or undescribed target",
            )
            joint = connections[record["name"]]
            self.require(
                joint.get("type") == JointKind.FIXED
                and joint.find("parent").get("link") == record["parent"],
                "interaction frame attached to wrong body",
            )
            origin = joint.find("origin")
            self.require(
                np.allclose(
                    self.vector(origin.get("xyz")),
                    record["position"],
                    rtol=0,
                    atol=1e-8,
                ),
                "interaction frame position mismatch",
            )
            self.require(
                np.allclose(
                    self.vector(origin.get("rpy")),
                    record["rotation"],
                    rtol=0,
                    atol=1e-8,
                ),
                "interaction frame rotation mismatch",
            )

    def _validate_bodies(self, root: ElementTree.Element, definition: dict) -> None:
        """Compare serialized physical attachments with the authored body definition."""
        links = {link.get("name"): link for link in root.findall("link")}
        connections = {
            joint.find("child").get("link"): joint for joint in root.findall("joint")
        }
        prefix = definition["identifier"] + "_"
        for part in definition["parts"]:
            name = prefix + part["name"]
            self.require(name in links, "missing authored body")
            if part["shapes"]:
                self.require(
                    len(links[name].findall("visual/geometry/mesh")) == 1,
                    "physical body has no unique mesh",
                )
            if any(shape["collision"] for shape in part["shapes"]):
                self.require(
                    bool(links[name].findall("collision")),
                    "physical body has no contact geometry",
                )
            if part["parent"] is not None:
                joint = connections[name]
                self.require(
                    joint.get("type") == part["joint"]
                    and joint.find("parent").get("link") == prefix + part["parent"],
                    "authored body hierarchy changed",
                )
                origin = joint.find("origin")
                self.require(
                    np.allclose(
                        self.vector(origin.get("xyz")),
                        part["origin"],
                        rtol=0,
                        atol=1e-8,
                    ),
                    "body origin changed",
                )
                self.require(
                    np.allclose(
                        self.vector(origin.get("rpy")),
                        part["rotation"],
                        rtol=0,
                        atol=1e-8,
                    ),
                    "body orientation changed",
                )

    def validate_meshes(self) -> list[MeshSummary]:
        """Decode all actual meshes and check local bounds against authored dimensions."""
        root = ElementTree.parse(self.local_file("model.urdf")).getroot()
        definition = json.loads(self.local_file("definition.json").read_text())
        parts = {
            definition["identifier"] + "_" + part["name"]: part
            for part in definition["parts"]
        }
        summaries = []
        for link in root.findall("link"):
            for mesh_reference in link.findall("visual/geometry/mesh"):
                reference = mesh_reference.get("filename")
                path = self.local_file(reference)
                self._validate_glb_dependencies(path)
                scene = trimesh.load_scene(path, process=False)
                self.require(bool(scene.geometry), f"empty mesh {reference}")
                triangles = 0
                for mesh in scene.geometry.values():
                    self.require(
                        isinstance(mesh, trimesh.Trimesh),
                        f"non-triangle geometry {reference}",
                    )
                    self.require(
                        mesh.vertices.size > 0 and np.isfinite(mesh.vertices).all(),
                        f"nonfinite or empty positions {reference}",
                    )
                    self.require(
                        mesh.faces.size > 0
                        and mesh.faces.min() >= 0
                        and mesh.faces.max() < len(mesh.vertices),
                        f"invalid triangle indices {reference}",
                    )
                    triangles += len(mesh.faces)
                bounds = np.asarray(scene.bounds)
                self.require(
                    bounds.shape == (2, 3)
                    and np.isfinite(bounds).all()
                    and (bounds[1] > bounds[0]).all(),
                    f"nonpositive or nonfinite mesh bounds {reference}",
                )
                self._compare_authored_bounds(
                    parts[link.get("name")], bounds, reference
                )
                summaries.append(
                    MeshSummary(
                        reference, triangles, bounds[0].tolist(), bounds[1].tolist()
                    )
                )
        self.require(bool(summaries), "no physical meshes")
        return summaries

    def _validate_glb_dependencies(self, path: Path) -> None:
        """Reject malformed containers and any mesh dependency outside the package."""
        content = path.read_bytes()
        self.require(len(content) >= 20, f"truncated glTF file {path.name}")
        magic, version, length = struct.unpack_from("<4sII", content)
        self.require(
            magic == b"glTF" and version == 2 and length == len(content),
            f"invalid glTF header {path.name}",
        )
        chunk_length, chunk_type = struct.unpack_from("<II", content, 12)
        self.require(
            chunk_type == 0x4E4F534A and 20 + chunk_length <= len(content),
            f"missing glTF metadata {path.name}",
        )
        document = json.loads(content[20 : 20 + chunk_length])
        for dependency in document.get("buffers", []) + document.get("images", []):
            if "uri" in dependency and not dependency["uri"].startswith("data:"):
                relative = path.relative_to(self.directory).parent / dependency["uri"]
                self.local_file(str(relative))

    def _compare_authored_bounds(
        self, part: dict, actual: np.ndarray, reference: str
    ) -> None:
        """Detect wrong units or baked parent transforms without depending on tessellation."""
        bounds = []
        marking_size = 0.0
        for shape in part["shapes"]:
            kind = shape["kind"]
            dimensions = shape["dimensions"]
            center = np.asarray(shape["center"])
            if kind in (ShapeKind.LABEL, ShapeKind.LOGO):
                marking_size = max(marking_size, dimensions[0])
                continue
            if kind == ShapeKind.BOX:
                half = np.asarray(dimensions) / 2
            elif kind == ShapeKind.SPHERE:
                half = np.repeat(dimensions[0], 3)
            elif kind == ShapeKind.CONE:
                radius = max(dimensions[:2])
                half = np.array((radius, radius, dimensions[2] / 2))
            elif kind == ShapeKind.RING:
                half = np.array((dimensions[0], dimensions[0], dimensions[2] / 2))
            elif kind == ShapeKind.CYLINDER:
                half = np.array((dimensions[0], dimensions[0], dimensions[1] / 2))
            elif kind == ShapeKind.TUBE:
                points = np.asarray(shape["points"])
                bounds.extend(
                    (
                        points.min(axis=0) - dimensions[0],
                        points.max(axis=0) + dimensions[0],
                    )
                )
                continue
            else:
                raise InvalidPackage(
                    f"{self.directory.name}: unknown authored shape {kind}"
                )
            half = np.abs(rotation_matrix(shape["rotation"])) @ half
            bounds.extend((center - half, center + half))
        if not bounds:
            return
        expected = np.array((np.min(bounds, axis=0), np.max(bounds, axis=0)))
        tolerance = max(
            0.002, np.max(expected[1] - expected[0]) * 0.06, marking_size * 0.3
        )
        self.require(
            np.allclose(actual, expected, rtol=0, atol=tolerance),
            f"mesh coordinate or unit mismatch {reference}: {actual.tolist()} versus authored {expected.tolist()}",
        )

    def validate(self) -> PackageReport:
        """Validate a complete distributable package, including previews and notices."""
        self.validate_structure()
        meshes = self.validate_meshes()
        metadata = json.loads(self.local_file("metadata.json").read_text())
        blend = self.local_file(metadata["blender"])
        with blend.open("rb") as stream:
            header = stream.read(12)
        self.require(
            len(header) == 12
            and header[:7] == b"BLENDER"
            and header[7:8] in (b"_", b"-")
            and header[8:9] in (b"v", b"V"),
            "unrecognized Blender file header",
        )
        for preview in metadata["previews"].values():
            path = self.local_file(preview)
            with Image.open(path) as image:
                self.require(min(image.size) >= 64, "undersized preview")
                image.verify()
        for notice in ("README.md", "LICENSE.txt", "NOTICE.md", "rights.json"):
            self.require(
                self.local_file(notice).stat().st_size > 0, f"empty notice {notice}"
            )
        root = ElementTree.parse(self.local_file("model.urdf")).getroot()
        return PackageReport(
            metadata["device"],
            len(root.findall("link")),
            len(metadata["joints"]),
            len(metadata["frames"]),
            meshes,
        )


# %% Command line


def main() -> None:
    """Validate all catalogue resources or selected completed instruments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", action="append")
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    known = {model.identifier: model for model in build_devices()}
    identifiers = set(options.device or known)
    if identifiers - set(known):
        raise InvalidPackage(
            f"Unknown device identifiers: {sorted(identifiers - set(known))}"
        )
    reports = []
    for name in sorted(identifiers):
        validator = PackageValidator(options.root / name)
        validator.validate_definition(known[name])
        reports.append(validator.validate())
    content = json.dumps([asdict(report) for report in reports], indent=2) + "\n"
    if options.report:
        options.report.write_text(content)
    print(
        json.dumps(
            {
                "validated": len(reports),
                "devices": [report.device for report in reports],
                "meshes": sum(len(report.meshes) for report in reports),
                "triangles": sum(
                    mesh.triangles for report in reports for mesh in report.meshes
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
