"""
Compose a self-contained expanded laboratory while preserving calibrated legacy poses.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "instruments"))

from description import InstrumentDescription
from device_resources import GlbField, StaticGlb
from large_layout import InstrumentPlacement, LargeLaboratoryLayout
from rights import LaboratoryAssetRights


# %% Portable world composition
class NavigationField(StrEnum):
    """Authored planar support geometry retained as physical contact surfaces."""

    NAVIGATION = "navigation"
    """World navigation metadata."""
    SUPPORT_GEOMETRIES = "supportGeometries"
    """Exact compiled geometry names that support the mobile base."""
    ROOM_FLOOR = "room__medical_lab_room_shell_room__medical_lab_room_shell_contact_0"
    """Floor box of the expanded room, whose upper surface lies at zero height."""


@dataclass(frozen=True)
class LargeLaboratoryBundle:
    """
    Assemble portable instrument subtrees into the existing single-environment world.
    """

    source: Path
    """
    Existing calibrated laboratory resource directory.
    """

    devices: Path
    """
    Independent instrument resource catalogue.
    """

    output: Path
    """
    New self-contained world directory; never the existing laboratory.
    """

    layout: LargeLaboratoryLayout = LargeLaboratoryLayout()
    """
    Metric room and task-area specification.
    """

    def build(self) -> None:
        """
        Copy the preserved bench, compose new bodies and write shared scene metadata.
        """
        if self.source.resolve() == self.output.resolve():
            raise ValueError(
                "The expanded world must not overwrite its source laboratory"
            )
        shutil.copytree(self.source, self.output, dirs_exist_ok=True)
        self._remove_small_room()
        room_directory = self.output / "room"
        InstrumentDescription(self.layout.room_definition()).write(room_directory)
        room_metadata = json.loads((room_directory / "metadata.json").read_text())
        room_metadata["previews"] = {}
        (room_directory / "metadata.json").write_text(
            json.dumps(room_metadata, indent=2) + "\n"
        )
        LaboratoryAssetRights().install(room_directory)
        root = ElementTree.parse(self.output / "environment.urdf").getroot()
        legacy_joint_tails = {
            joint.get("name"): joint.tail for joint in root.findall("joint")
        }
        root.set("name", "aicor_medical_lab_large")
        self._append_model(
            root,
            room_directory / "model.urdf",
            "room",
            "room__",
            "laboratory_root",
            (0, 0, 0),
        )
        instruments = []
        for placement in self.layout.placements():
            resource = self.devices / placement.identifier
            destination = self.output / "devices" / placement.identifier
            shutil.copytree(resource, destination, dirs_exist_ok=True)
            self._append_model(
                root,
                destination / "model.urdf",
                f"devices/{placement.identifier}",
                placement.prefix,
                placement.attachment_parent,
                placement.attachment_position or placement.position,
            )
            instruments.append(self._instrument_metadata(placement, resource))
        ElementTree.indent(root)
        for joint in root.findall("joint"):
            if joint.get("name") in legacy_joint_tails:
                joint.tail = legacy_joint_tails[joint.get("name")]
        ElementTree.ElementTree(root).write(
            self.output / "environment.urdf", encoding="utf-8", xml_declaration=True
        )
        scene = json.loads((self.source / "scene.json").read_text())
        scene["name"] = "aicor_medical_lab_large"
        scene["environmentName"] = "AICOR Medical Laboratory"
        scene["task"] = "Expanded medical laboratory · Modular instruments"
        scene["camera"] = {"position": [2.80, -5.2, 4.0], "target": [0, 0.9, 0.95]}
        scene["laboratory"]["cameras"]["overview"] = copy.deepcopy(scene["camera"])
        scene[NavigationField.NAVIGATION] = {
            NavigationField.SUPPORT_GEOMETRIES: [NavigationField.ROOM_FLOOR]
        }
        scene["medicalLab"] = {
            "title": "AICOR Medical Laboratory",
            "instruments": instruments,
            "bounds": {"min": [-3.4, -2.7, 0.04], "max": [3.4, 3.1, 2.5]},
            "contactExclusions": [
                {
                    "first": "ph_meter__ph_meter_base",
                    "second": "ph_meter__ph_meter_probe_arm",
                }
            ],
            "stations": [
                {
                    "id": station.identifier,
                    "label": station.label,
                    "camera": {
                        "position": list(station.position),
                        "target": list(station.target),
                    },
                }
                for station in self.layout.stations()
            ],
            "room": {
                "bounds": [-3.5, 3.5, -2.8, 3.2],
                "rearAisleWidth": self.layout.rear_bench_front
                - self.layout.original_bench_back,
                "workSurfaceHeight": self.layout.work_height,
            },
        }
        model = next(model for model in scene["models"] if not model["robot"])
        model["name"] = "AICOR Medical Laboratory"
        model["links"] = len(root.findall("link"))
        model["movableJoints"] = [
            joint.get("name")
            for joint in root.findall("joint")
            if joint.get("type") != "fixed"
        ]
        model["preserveMaterials"] = True
        self._write_json("scene.json", scene)
        trajectory = json.loads((self.source / "trajectory.json").read_text())
        for frame in trajectory["frames"]:
            frame.update(
                {
                    joint["name"]: joint["rest"]
                    for instrument in instruments
                    for joint in instrument["joints"]
                }
            )
        self._write_json("trajectory.json", trajectory)
        semantics = json.loads((self.source / "semantics.json").read_text())
        semantics["medicalLab"] = scene["medicalLab"]
        self._write_json("semantics.json", semantics)
        LaboratoryAssetRights().install(self.output)
        (self.output / "README.md").write_text(
            "# AICOR Medical Laboratory — expanded world\n\n"
            "This seven-by-six-metre room preserves the original sample-preparation bench, its five instruments, calibrated object poses and existing robot task frames.\n\n"
            "The rear workbench contains fifteen separate new instrument/accessory resources. The clear rear aisle is 1.75 metres wide. All new links are namespaced; the independent model packages remain in `devices/`.\n\n"
            "`environment.urdf` is the composed CRAM environment. `scene.json` and `semantics.json` provide `medicalLab` station cameras, instrument poses, interaction frames and joint capabilities. `room/model.blend` edits the new architecture; `laboratory_large.blend`, when rendered, contains the assembled scene.\n\n"
            "The pH meter's nested support bearing uses one explicit base/arm self-contact exclusion. Its joint represents the bearing motion; its mating visual surfaces are not intended as a detailed bearing contact simulation. External contacts remain enabled.\n\n"
            "Instrument models provide geometry and articulated motion. Biological, chemical, thermal and optical outcomes require a separate controller or simulation. See each instrument package for its detailed scope and licence.\n"
        )

    def _write_json(self, filename: str, data: dict) -> None:
        """
        Write finite, readable scene metadata.
        """
        (self.output / filename).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n"
        )

    def _remove_small_room(self) -> None:
        """
        Keep original bench/device surfaces while removing the compact architectural
        shell.
        """
        source = StaticGlb.read(self.source / "assets/environment.glb")
        architecture = (
            "Floor",
            "Back wall",
            "Left wall",
            "Wall panel seam",
            "Wall skirting",
            "Shelf",
            "Laboratory subtitle",
            "Storage box",
            "Storage label",
            "Ceiling task light",
            "Task light diffuser",
            "Light suspension",
            "Wall AICOR logo",
        )
        keep = []
        for node in source.document[GlbField.NODES]:
            if node.get(GlbField.NAME, "").startswith(architecture):
                continue
            mesh = source.document[GlbField.MESHES][node[GlbField.MESH]]
            accessors = [
                source.document[GlbField.ACCESSORS][
                    primitive[GlbField.ATTRIBUTES][GlbField.POSITION]
                ]
                for primitive in mesh[GlbField.PRIMITIVES]
            ]
            lower = np.min(
                [accessor[GlbField.MINIMUM] for accessor in accessors], axis=0
            )
            upper = np.max(
                [accessor[GlbField.MAXIMUM] for accessor in accessors], axis=0
            )
            center = (lower + upper) / 2 + np.asarray(
                node.get(GlbField.TRANSLATION, (0, 0, 0))
            )
            if center[1] > 0.65:
                continue
            keep.append(node)
        source.document[GlbField.NODES] = keep
        trimmed = source.extract(tuple(node[GlbField.NAME] for node in keep), (0, 0, 0))
        trimmed.write(self.output / "assets/environment.glb")

    @staticmethod
    def _append_model(
        root: ElementTree.Element,
        description: Path,
        local_directory: str,
        prefix: str,
        parent: str,
        position: tuple[float, float, float],
    ) -> None:
        """
        Preserve one URDF subtree and all local meshes beneath a fixed placement joint.
        """
        model = ElementTree.parse(description).getroot()
        link_names = {link.get("name") for link in model.findall("link")}
        children = {joint.find("child").get("link") for joint in model.findall("joint")}
        [model_root] = link_names - children
        existing = {link.get("name") for link in root.findall("link")}
        if parent not in existing or existing & {prefix + name for name in link_names}:
            raise ValueError(f"Invalid or duplicate placement for {prefix}")
        for link in model.findall("link"):
            link.set("name", prefix + link.get("name"))
            for collision in link.findall("collision"):
                collision.set("name", prefix + collision.get("name"))
            for mesh in link.findall(".//mesh"):
                mesh.set("filename", str(Path(local_directory) / mesh.get("filename")))
            root.append(link)
        for joint in model.findall("joint"):
            joint.set("name", prefix + joint.get("name"))
            for element in (joint.find("parent"), joint.find("child")):
                element.set("link", prefix + element.get("link"))
            mimic = joint.find("mimic")
            if mimic is not None:
                mimic.set("joint", prefix + mimic.get("joint"))
            root.append(joint)
        placement = ElementTree.SubElement(
            root, "joint", name=prefix + "placement_joint", type="fixed"
        )
        ElementTree.SubElement(placement, "parent", link=parent)
        ElementTree.SubElement(placement, "child", link=prefix + model_root)
        ElementTree.SubElement(
            placement, "origin", xyz=" ".join(map(str, position)), rpy="0 0 0"
        )

    def _instrument_metadata(
        self, placement: InstrumentPlacement, resource: Path
    ) -> dict:
        """
        Publish exactly the capabilities and force limits in the portable URDF.
        """
        metadata = json.loads((resource / "metadata.json").read_text())
        description = ElementTree.parse(resource / "model.urdf").getroot()
        joints = {joint.get("name"): joint for joint in description.findall("joint")}
        records = []
        for item in metadata["joints"]:
            joint = joints[item["name"]]
            limits = joint.find("limit")
            continuous = item["type"] == "continuous"
            records.append(
                {
                    "name": placement.prefix + item["name"],
                    "type": item["type"],
                    "rest": 0.0,
                    "inspection": item["inspectionPosition"],
                    "lower": -math.pi if continuous else float(limits.get("lower")),
                    "upper": math.pi if continuous else float(limits.get("upper")),
                    "velocity": float(limits.get("velocity")),
                    "effort": float(limits.get("effort")),
                }
            )
        return {
            "id": placement.identifier,
            "label": metadata["title"],
            "station": placement.station,
            "joints": records,
            "pose": placement.pose,
            "root": placement.prefix + metadata["rootLink"],
            "resource": f"devices/{placement.identifier}/model.urdf",
            "frames": [
                {
                    **frame,
                    "name": placement.prefix + frame["name"],
                    "parent": placement.prefix + frame["parent"],
                }
                for frame in metadata["frames"]
            ],
        }


# %% Command line


def main() -> None:
    """
    Compose a new resource folder using previously generated room geometry.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    LargeLaboratoryBundle(options.source, options.devices, options.output).build()
    print(json.dumps({"world": str(options.output), "instruments": 15}))


if __name__ == "__main__":
    main()
