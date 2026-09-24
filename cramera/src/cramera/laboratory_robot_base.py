"""
Expose physical planar base motion through ordinary renderer joints.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cramera.laboratory_physics_robot import (
    LaboratoryRobotPhysics,
    RobotBaseJoint,
    RobotPhysicsParameters,
)


# %% recording articulation
@dataclass(frozen=True)
class RobotBaseDescription:
    """
    Extend a copied robot description with world-aligned planar coordinates.
    """

    path: Path
    """
    Copied URDF whose original root becomes the planar chain's child.
    """

    rotation: tuple[float, float, float, float]
    """
    Original model orientation in XYZW quaternion order.
    """

    def write(self) -> None:
        """
        Add the three physical base coordinates once, keeping meshes unchanged.
        """
        tree = ElementTree.parse(self.path)
        root = tree.getroot()
        if root.find(f"joint[@name='{RobotBaseJoint.HORIZONTAL}']") is not None:
            return
        children = {
            joint.find("child").attrib["link"] for joint in root.findall("joint")
        }
        [original] = [
            link.attrib["name"]
            for link in root.findall("link")
            if link.attrib["name"] not in children
        ]
        rotation = Rotation.from_quat(self.rotation)
        parameters = RobotPhysicsParameters()
        previous = "pr2/mobile_origin"
        ElementTree.SubElement(root, "link", name=previous)
        for index, name in enumerate(RobotBaseJoint):
            child = (
                original
                if name == RobotBaseJoint.HEADING
                else name.replace("_joint", "_link")
            )
            if child != original:
                ElementTree.SubElement(root, "link", name=child)
            heading = name == RobotBaseJoint.HEADING
            joint = ElementTree.SubElement(
                root, "joint", name=name, type="revolute" if heading else "prismatic"
            )
            ElementTree.SubElement(joint, "parent", link=previous)
            ElementTree.SubElement(joint, "child", link=child)
            ElementTree.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ElementTree.SubElement(
                joint,
                "axis",
                xyz=" ".join(map(str, rotation.inv().apply(np.eye(3)[index]))),
            )
            limit = math.pi if heading else 2.0
            ElementTree.SubElement(
                joint,
                "limit",
                lower=str(-limit),
                upper=str(limit),
                effort=str(
                    parameters.base_torque if heading else parameters.base_force
                ),
                velocity="0.2",
            )
            previous = child
        ElementTree.indent(tree)
        tree.write(self.path, encoding="utf-8", xml_declaration=True)


# %% swept robot clearance
@dataclass
class UnsafeBasePath(ValueError):
    """
    A proposed planar movement intersects an observed laboratory obstacle.
    """

    robot_geometry: str
    """
    Robot collision surface entering the obstacle.
    """

    obstacle_geometry: str
    """
    Laboratory collision surface blocking the planned movement.
    """

    def __str__(self) -> str:
        """
        Identify the two colliders that prohibit the commanded travel.
        """
        return f"Unsafe base path: {self.robot_geometry} would hit {self.obstacle_geometry}"


@dataclass
class LaboratoryBasePath:
    """
    Check swept measured robot geometry before translating the base.
    """

    physics: LaboratoryRobotPhysics
    """
    Authoritative measured model and state, left unchanged by checking.
    """

    held_object: str | None = None
    """
    A grasped object whose measured geometry travels with the robot.
    """

    clearance: float = 0.008
    """
    Required separation in meters from every non-ground environment collider.
    """

    sampling_distance: float = 0.008
    """
    Maximum displacement between intermediate clearance checks in meters.
    """

    scratch: mujoco.MjData = field(init=False)
    """
    Independent kinematic state used solely for swept collision queries.
    """

    moving_geometry: np.ndarray = field(init=False)
    """
    Robot and optional carried-object collision indices.
    """

    obstacle_geometry: np.ndarray = field(init=False)
    """
    Environment and loose-object collision indices, excluding the floor.
    """

    def __post_init__(self) -> None:
        """
        Index colliders without mutating authoritative positions or geometry.
        """
        model = self.physics.model
        self.scratch = mujoco.MjData(model)
        held_body = model.body(self.held_object).id if self.held_object else None
        self.moving_geometry = np.array(
            [
                index
                for index in range(model.ngeom)
                if model.body(model.geom_bodyid[index]).name.startswith("pr2/")
                or model.geom_bodyid[index] == held_body
            ],
            dtype=int,
        )
        moving = set(self.moving_geometry)
        self.obstacle_geometry = np.array(
            [
                index
                for index in range(model.ngeom)
                if index not in moving
                and model.geom_type[index] != mujoco.mjtGeom.mjGEOM_PLANE
                and (model.geom_contype[index] or model.geom_conaffinity[index])
            ],
            dtype=int,
        )

    def validate(self, targets: dict[str, float]) -> None:
        """
        Reject a path whose intermediate robot geometry enters an obstacle.
        """
        obstruction = self.obstruction(targets)
        if obstruction is not None:
            raise obstruction

    def obstruction(self, targets: dict[str, float]) -> UnsafeBasePath | None:
        """
        Identify the first blocked sampled pose, or accept a clear path.
        """
        model = self.physics.model
        measured = self.physics.robot_state()
        differences = {
            name: value - measured[name]
            for name, value in targets.items()
            if name in RobotBaseJoint
        }
        translation = np.array(
            [
                differences.get(RobotBaseJoint.HORIZONTAL, 0.0),
                differences.get(RobotBaseJoint.DEPTH, 0.0),
            ]
        )
        base = self.physics.data.xpos[model.body("robot_mount").id]
        radius = np.max(
            np.linalg.norm(
                self.physics.data.geom_xpos[self.moving_geometry, :2] - base[:2], axis=1
            )
            + model.geom_rbound[self.moving_geometry]
        )
        distance = np.linalg.norm(translation) + radius * abs(
            differences.get(RobotBaseJoint.HEADING, 0.0)
        )
        if distance < np.finfo(float).eps:
            return None
        samples = max(1, math.ceil(distance / self.sampling_distance))
        for amount in np.linspace(0.0, 1.0, samples + 1):
            self.scratch.qpos[:] = self.physics.data.qpos
            for name, difference in differences.items():
                self.scratch.qpos[self.physics.robot_joints[name].position_index] += (
                    amount * difference
                )
            if self.held_object is not None:
                self._move_held_object(amount, differences)
            mujoco.mj_kinematics(model, self.scratch)
            obstruction = self._check_clearance()
            if obstruction is not None:
                return obstruction
        return None

    def _move_held_object(self, amount: float, differences: dict[str, float]) -> None:
        """
        Translate and rotate the carried collision hull with the sampled base pose.
        """
        model = self.physics.model
        body = model.body(self.held_object).id
        coordinate = model.jnt_qposadr[model.body_jntadr[body]]
        base = self.physics.data.xpos[model.body("robot_mount").id]
        heading = Rotation.from_euler(
            "z", amount * differences.get(RobotBaseJoint.HEADING, 0.0)
        )
        origin = self.scratch.qpos[coordinate : coordinate + 3]
        translation = amount * np.array(
            [
                differences.get(RobotBaseJoint.HORIZONTAL, 0.0),
                differences.get(RobotBaseJoint.DEPTH, 0.0),
                0.0,
            ]
        )
        origin[:] = base + heading.apply(origin - base) + translation
        orientation = Rotation.from_quat(
            self.scratch.qpos[coordinate + 3 : coordinate + 7], scalar_first=True
        )
        self.scratch.qpos[coordinate + 3 : coordinate + 7] = (
            heading * orientation
        ).as_quat(scalar_first=True)

    def _check_clearance(self) -> UnsafeBasePath | None:
        """
        Use exact convex distance only for collider pairs with nearby bounds.
        """
        model = self.physics.model
        moving = self.moving_geometry
        obstacles = self.obstacle_geometry
        centers = (
            self.scratch.geom_xpos[moving, None] - self.scratch.geom_xpos[obstacles]
        )
        bounds = (
            model.geom_rbound[moving, None]
            + model.geom_rbound[obstacles]
            + self.clearance
        )
        candidates = np.argwhere(np.linalg.norm(centers, axis=2) < bounds)
        for moving_index, obstacle_index in candidates:
            first, second = int(moving[moving_index]), int(obstacles[obstacle_index])
            distance = mujoco.mj_geomDistance(
                model, self.scratch, first, second, self.clearance, None
            )
            if distance < self.clearance:
                return UnsafeBasePath(model.geom(first).name, model.geom(second).name)
        return None
