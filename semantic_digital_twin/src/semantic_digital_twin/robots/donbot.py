from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
    Arm,
    Base,
    Camera,
    FieldOfView,
    Finger,
    Neck,
    ParallelGripper,
    Torso,
)
from semantic_digital_twin.robots.robot_mixins import HasArms, HasNeck
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
)


@dataclass(eq=False)
class Donbot(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Donbot Robot.
    """

    @property
    def arm(self) -> Arm:
        return self.arms[0]

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="donbot", prefix=world.name),
            root=world.get_body_by_name("base_footprint"),
            _world=world,
        )

    def _setup_semantic_annotations(self):
        gripper_thumb = Finger(
            name=PrefixedName("gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_gripper_left_link"),
            tip=self._world.get_body_by_name("gripper_finger_right_link"),
            _world=self._world,
        )

        gripper_finger = Finger(
            name=PrefixedName("gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_gripper_left_link"),
            tip=self._world.get_body_by_name("gripper_finger_left_link"),
            _world=self._world,
        )

        gripper = ParallelGripper(
            name=PrefixedName("gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_base_link"),
            tool_frame=self._world.get_body_by_name("gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
            front_facing_axis=Vector3(0, 0, 1),
            thumb=gripper_thumb,
            finger=gripper_finger,
            _world=self._world,
        )

        arm = Arm(
            name=PrefixedName("arm", prefix=self.name.name),
            root=self._world.get_body_by_name("ur5_base_link"),
            tip=self._world.get_body_by_name("ur5_wrist_3_link"),
            manipulator=gripper,
            _world=self._world,
        )
        self.add_arm(arm)

        camera = Camera(
            name=PrefixedName("camera_link", prefix=self.name.name),
            root=self._world.get_body_by_name("camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.5,
            maximal_height=1.2,
            _world=self._world,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[camera],
            root=self._world.get_body_by_name("ur5_base_link"),
            tip=self._world.get_body_by_name("ur5_base_link"),
            _world=self._world,
        )
        self.add_neck(neck)

        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("base_footprint"),
            tip=self._world.get_body_by_name("ur5_base_link"),
            _world=self._world,
        )
        self.add_torso(torso)

        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("base_link"),
            _world=self._world,
        )
        self.add_base(base)

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "iai_donbot.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05,
                violated_distance=0.0,
                robot=self,
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        chains = [self.arm, self.arm.manipulator, self.torso]
        if self.neck is not None:
            chains.append(self.neck)

        for chain in chains:
            for connection in chain.connections:
                if isinstance(connection, ActiveConnection):
                    connection.has_hardware_interface = True

        if isinstance(self.drive, ActiveConnection):
            self.drive.has_hardware_interface = True

    def _setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [c for c in self.arm.connections if type(c) != FixedConnection],
                    [3.23, -1.51, -1.57, 0.0, 1.57, -1.65],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.arm.add_joint_state(arm_park)

        looking = JointState.from_mapping(
            name=PrefixedName("looking", prefix=self.name.name),
            mapping=dict(
                zip(
                    [c for c in self.neck.connections if type(c) != FixedConnection],
                    [0.0, -0.35, -2.15, -0.7, 1.57, -1.57],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.neck.add_joint_state(looking)

        gripper_joints = [
            self._world.get_connection_by_name("gripper_joint"),
            self._world.get_connection_by_name("gripper_base_gripper_left_joint"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.109, -0.055])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0065, -0.0027])),
            state_type=GripperState.CLOSE,
        )

        self.arm.manipulator.add_joint_state(gripper_close)
        self.arm.manipulator.add_joint_state(gripper_open)

        torso_joint = [self._world.get_connection_by_name("arm_base_mounting_joint")]

        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)
