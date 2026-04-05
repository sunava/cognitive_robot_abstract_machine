from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.robots.robot_mixins import HasNeck, SpecifiesLeftRightArm
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidAllCollisions,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    Neck,
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    AbstractRobot,
    Base,
    HumanoidGripper,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
)


@dataclass(eq=False)
class Justin(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Justin Robot.
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="rollin_justin", prefix=world.name),
            root=world.get_body_by_name("base_footprint"),
            _world=world,
        )

    def _setup_semantic_annotations(self):
        # Create left arm
        left_gripper_thumb = Finger(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("left_1thumb_base"),
            tip=self._world.get_body_by_name("left_1thumb4"),
            _world=self._world,
        )

        left_gripper_tip_finger = Finger(
            name=PrefixedName("left_gripper_tip_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("left_2tip_base"),
            tip=self._world.get_body_by_name("left_2tip4"),
            _world=self._world,
        )

        left_gripper_middle_finger = Finger(
            name=PrefixedName("left_gripper_middle_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("left_3middle_base"),
            tip=self._world.get_body_by_name("left_3middle4"),
            _world=self._world,
        )

        left_gripper_ring_finger = Finger(
            name=PrefixedName("left_gripper_ring_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("left_4ring_base"),
            tip=self._world.get_body_by_name("left_4ring4"),
            _world=self._world,
        )

        left_gripper = HumanoidGripper(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("left_arm7"),
            tool_frame=self._world.get_body_by_name("l_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
            front_facing_axis=Vector3(0, 0, 1),
            thumb=left_gripper_thumb,
            fingers=[
                left_gripper_tip_finger,
                left_gripper_middle_finger,
                left_gripper_ring_finger,
            ],
            _world=self._world,
        )
        left_arm = Arm(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("left_arm1"),
            tip=self._world.get_body_by_name("left_arm7"),
            manipulator=left_gripper,
            _world=self._world,
        )

        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("right_1thumb_base"),
            tip=self._world.get_body_by_name("right_1thumb4"),
            _world=self._world,
        )

        right_gripper_tip_finger = Finger(
            name=PrefixedName("right_gripper_tip_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("right_2tip_base"),
            tip=self._world.get_body_by_name("right_2tip4"),
            _world=self._world,
        )

        right_gripper_middle_finger = Finger(
            name=PrefixedName("right_gripper_middle_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("right_3middle_base"),
            tip=self._world.get_body_by_name("right_3middle4"),
            _world=self._world,
        )

        right_gripper_ring_finger = Finger(
            name=PrefixedName("right_gripper_ring_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("right_4ring_base"),
            tip=self._world.get_body_by_name("right_4ring4"),
            _world=self._world,
        )

        right_gripper = HumanoidGripper(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("right_arm7"),
            tool_frame=self._world.get_body_by_name("r_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.707, 0.707, 0.707, 0.707),
            front_facing_axis=Vector3(0, 0, 1),
            thumb=right_gripper_thumb,
            fingers=[
                right_gripper_tip_finger,
                right_gripper_middle_finger,
                right_gripper_ring_finger,
            ],
            _world=self._world,
        )
        right_arm = Arm(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("right_arm1"),
            tip=self._world.get_body_by_name("right_arm7"),
            manipulator=right_gripper,
            _world=self._world,
        )

        self.add_arm(right_arm)

        # Create camera and neck

        # real camera unknown at the moment of writing (also missing in urdf), so using dummy camera for now
        camera = Camera(
            name=PrefixedName("dummy_camera", prefix=self.name.name),
            root=self._world.get_body_by_name("head2"),
            forward_facing_axis=Vector3(1, 0, 0),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.85,
            _world=self._world,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[camera],
            root=self._world.get_body_by_name("torso4"),
            tip=self._world.get_body_by_name("head2"),
            pitch_body=self._world.get_body_by_name("head1"),
            yaw_body=self._world.get_body_by_name("head2"),
            _world=self._world,
        )
        self.add_neck(neck)

        # Create torso
        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("torso1"),
            tip=self._world.get_body_by_name("torso4"),
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
        """
        Loads the SRDF file for the Justin robot, if it exists.
        """
        # return
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "justin.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.1, violated_distance=0.0, robot=self
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                    robot=self,
                    body_subset=self.left_arm.bodies_with_collision
                    + self.right_arm.bodies_with_collision,
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.2,
                    violated_distance=0.05,
                    robot=self,
                    body_subset={self._world.get_body_by_name("base_link")},
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

        self._world.collision_manager.extend_max_avoided_bodies_rules(
            [
                MaxAvoidedCollisionsOverride(
                    2, bodies={self._world.get_body_by_name("base_link")}
                ),
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("right_arm7")
                        )
                    )
                    | set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("left_arm7")
                        )
                    ),
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(
            lambda: 1.0,
        )
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        controlled_joints = [
            "torso1_joint",
            "torso2_joint",
            "torso3_joint",
            "torso4_joint",
            "head1_joint",
            "head2_joint",
            "left_arm1_joint",
            "left_arm2_joint",
            "left_arm3_joint",
            "left_arm4_joint",
            "left_arm5_joint",
            "left_arm6_joint",
            "left_arm7_joint",
            "right_arm1_joint",
            "right_arm2_joint",
            "right_arm3_joint",
            "right_arm4_joint",
            "right_arm5_joint",
            "right_arm6_joint",
            "right_arm7_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def _setup_joint_states(self):
        def gripper_joint_targets(side: str, closed: bool) -> dict:
            finger_names = ["1thumb", "2tip", "3middle", "4ring"]
            per_finger_targets = (
                [0.523599, 1.50098, 1.76278, 1.76278]
                if closed
                else [0.0, 0.0, 0.0, 0.0]
            )
            joint_names = [
                f"{side}_{finger}{joint_idx}_joint"
                for finger in finger_names
                for joint_idx in range(1, 5)
            ]
            targets = per_finger_targets * len(finger_names)
            return {
                self._world.get_connection_by_name(joint_name): value
                for joint_name, value in zip(joint_names, targets)
            }

        # Create states
        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.left_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [
                        0.0,
                        0.0,
                        0.174533,
                        0.0,
                        0.0,
                        -1.9,
                        0.0,
                        1.0,
                        0.0,
                        -1.0,
                        0.0,
                    ],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.left_arm.add_joint_state(left_arm_park)

        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.right_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [
                        0.0,
                        0.0,
                        0.174533,
                        0.0,
                        0.0,
                        -1.9,
                        0.0,
                        1.0,
                        0.0,
                        -1.0,
                        0.0,
                    ],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.right_arm.add_joint_state(right_arm_park)

        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=gripper_joint_targets("left", closed=False),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=gripper_joint_targets("left", closed=True),
            state_type=GripperState.CLOSE,
        )

        self.left_arm.manipulator.add_joint_state(left_gripper_close)
        self.left_arm.manipulator.add_joint_state(left_gripper_open)

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=gripper_joint_targets("right", closed=False),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=gripper_joint_targets("right", closed=True),
            state_type=GripperState.CLOSE,
        )

        self.right_arm.manipulator.add_joint_state(right_gripper_close)
        self.right_arm.manipulator.add_joint_state(right_gripper_open)

        torso_joints = [
            self._world.get_connection_by_name("torso2_joint"),
            self._world.get_connection_by_name("torso3_joint"),
            self._world.get_connection_by_name("torso4_joint"),
        ]

        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.9, 2.33874, -1.57])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.8, 1.57, -0.77])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [0.0, 0.174533, 0.0])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)
