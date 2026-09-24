"""
Articulated cold-storage geometry and supported sample targets.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from .test_laboratory_bundle import laboratory_directory

# %% independently articulated cold storage


def test_door_clears_fixed_cabinet_throughout_its_sweep(
    laboratory_directory: Path,
) -> None:
    """
    Every moving door primitive remains outside the fixed collision shells.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    refrigerator = scene["laboratory"]["refrigerator"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    fixed = description.findall(f"link[@name='{refrigerator['body']}']/collision")
    moving = description.findall(f"link[@name='{refrigerator['doorBody']}']/collision")
    for step in range(26):
        angle = -refrigerator["doorOpen"] * step / 25
        for door in moving:
            local = [
                float(value) for value in door.find("origin").attrib["xyz"].split()
            ]
            door_size = [
                float(value)
                for value in door.find("geometry/box").attrib["size"].split()
            ]
            center = [
                refrigerator["doorOrigin"][0]
                + math.cos(angle) * local[0]
                - math.sin(angle) * local[1],
                refrigerator["doorOrigin"][1]
                + math.sin(angle) * local[0]
                + math.cos(angle) * local[1],
                refrigerator["doorOrigin"][2] + local[2],
            ]
            for body in fixed:
                body_center = [
                    float(value) for value in body.find("origin").attrib["xyz"].split()
                ]
                body_size = [
                    float(value)
                    for value in body.find("geometry/box").attrib["size"].split()
                ]
                if (
                    abs(center[2] - body_center[2])
                    >= (door_size[2] + body_size[2]) / 2 - 1e-8
                ):
                    continue
                yaw = float(body.find("origin").attrib["rpy"].split()[2])
                axes = [
                    (math.cos(rotation), math.sin(rotation))
                    for rotation in (angle, angle + math.pi / 2, yaw, yaw + math.pi / 2)
                ]
                separated = False
                for axis in axes:
                    distance = abs(
                        sum(
                            (center[index] - body_center[index]) * axis[index]
                            for index in range(2)
                        )
                    )
                    radii = []
                    for size, rotation in ((door_size, angle), (body_size, yaw)):
                        radii.append(
                            sum(
                                size[index]
                                / 2
                                * abs(
                                    axis[0] * math.cos(rotation + index * math.pi / 2)
                                    + axis[1] * math.sin(rotation + index * math.pi / 2)
                                )
                                for index in range(2)
                            )
                        )
                    if distance >= sum(radii) - 1e-8:
                        separated = True
                        break
                assert separated, (door.attrib["name"], body.attrib["name"], angle)


def test_door_opens_outward_from_the_front_left_hinge(
    laboratory_directory: Path,
) -> None:
    """
    The bounded hinge swings the right door edge towards the viewer.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    refrigerator = scene["laboratory"]["refrigerator"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    joint = description.find(f"joint[@name='{refrigerator['doorJoint']}']")
    assert joint.attrib["type"] == "revolute"
    assert joint.find("parent").attrib["link"] == refrigerator["body"]
    assert joint.find("child").attrib["link"] == refrigerator["doorBody"]
    assert joint.find("axis").attrib["xyz"] == "0 0 -1"
    assert float(joint.find("limit").attrib["upper"]) == refrigerator["doorOpen"]
    assert float(joint.find("limit").attrib["lower"]) == 0
    assert [
        float(value) for value in joint.find("origin").attrib["xyz"].split()
    ] == refrigerator["doorOrigin"]
    assert refrigerator["doorOrigin"][0] < 0
    assert refrigerator["doorOrigin"][1] < 0
    assert -math.sin(refrigerator["doorOpen"]) < 0
    assert refrigerator["doorJoint"] in scene["models"][0]["movableJoints"]
    trajectory = json.loads((laboratory_directory / "trajectory.json").read_text())
    assert trajectory["frames"][0][refrigerator["doorJoint"]] == 0


def test_six_samples_have_open_supported_shelves(laboratory_directory: Path) -> None:
    """
    Each complete upright tube clears the cabinet, other shelf, and collar.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    refrigerator = scene["laboratory"]["refrigerator"]
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    collisions = description.findall(f"link[@name='{refrigerator['body']}']/collision")
    slots = refrigerator["slots"]
    assert [slot["id"] for slot in slots] == [f"R{index + 1}" for index in range(6)]
    assert len({slot["position"][2] for slot in slots}) == 2
    for slot in slots:
        target = slot["localPosition"]
        assert slot["position"] == pytest.approx(
            [refrigerator["origin"][axis] + target[axis] for axis in range(3)]
        )
        support_count = 0
        for collision in collisions:
            center = [
                float(value) for value in collision.find("origin").attrib["xyz"].split()
            ]
            size = [
                float(value)
                for value in collision.find("geometry/box").attrib["size"].split()
            ]
            yaw = float(collision.find("origin").attrib["rpy"].split()[2])
            delta_x, delta_y = target[0] - center[0], target[1] - center[1]
            local = (
                math.cos(yaw) * delta_x + math.sin(yaw) * delta_y,
                -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y,
            )
            if center[2] + size[2] / 2 <= target[2] + 1e-9:
                if abs(center[2] + size[2] / 2 - target[2]) < 1e-8 and all(
                    abs(local[axis]) < size[axis] / 2 for axis in range(2)
                ):
                    support_count += 1
                continue
            if center[2] - size[2] / 2 >= target[2] + semantics["tube"]["height"]:
                continue
            distance = sum(
                max(abs(local[axis]) - size[axis] / 2, 0) ** 2 for axis in range(2)
            )
            assert distance > semantics["tube"]["outerRadius"] ** 2
        assert support_count == 1


def test_closed_door_has_real_glass_and_frame_collisions(
    laboratory_directory: Path,
) -> None:
    """
    A closed glass door obstructs the otherwise open shelf entrance.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    refrigerator = scene["laboratory"]["refrigerator"]
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    door = description.find(f"link[@name='{refrigerator['doorBody']}']")
    assert door.find("inertial/mass") is not None
    glass = door.find("collision[@name='refrigerator_door_glass']")
    center = [
        float(value) + refrigerator["doorOrigin"][axis]
        for axis, value in enumerate(glass.find("origin").attrib["xyz"].split())
    ]
    size = [float(value) for value in glass.find("geometry/box").attrib["size"].split()]
    for slot in refrigerator["slots"]:
        assert abs(slot["localPosition"][0] - center[0]) < size[0] / 2
        assert (
            center[2] - size[2] / 2 < slot["localPosition"][2] < center[2] + size[2] / 2
        )
        assert center[1] < slot["localPosition"][1]
    assert len(door.findall("collision")) >= 5


def test_cabinet_clears_bench_and_preserves_portable_meshes(
    laboratory_directory: Path,
) -> None:
    """
    The floor-standing resource leaves a gap beside the laboratory bench.
    """
    scene = json.loads((laboratory_directory / "scene.json").read_text())
    refrigerator = scene["laboratory"]["refrigerator"]
    semantics = json.loads((laboratory_directory / "semantics.json").read_text())
    assert (
        refrigerator["origin"][0] + refrigerator["dimensions"][0] / 2
        < semantics["workSurface"]["bounds"][0]
    )
    assert refrigerator["origin"][2] == 0
    assert semantics["refrigerator"] == refrigerator
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    for body, asset in (("body", "asset"), ("doorBody", "doorAsset")):
        link = description.find(f"link[@name='{refrigerator[body]}']")
        assert (
            link.find("visual/geometry/mesh").attrib["filename"] == refrigerator[asset]
        )
