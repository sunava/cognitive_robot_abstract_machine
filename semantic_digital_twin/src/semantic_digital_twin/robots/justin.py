from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Self, Union

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasNeck,
    HasTorso,
    HasMobileBase,
    HasFingers,
    GenericFinger,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    FieldOfView,
    Finger,
    Neck,
    Torso,
    MobileBase,
    EndEffector,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class JustinFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class JustinLeftThumb(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_1thumb_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_1thumb4"),
        )
        return finger


@dataclass(eq=False)
class JustinLeftIndexFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_2tip_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_2tip4"),
        )
        return finger


@dataclass(eq=False)
class JustinLeftMiddleFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_3middle_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_3middle4"),
        )
        return finger


@dataclass(eq=False)
class JustinLeftRingFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_4ring_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_4ring4"),
        )
        return finger


@dataclass(eq=False)
class JustinRightThumb(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_1thumb_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_1thumb4"),
        )
        return finger


@dataclass(eq=False)
class JustinRightIndexFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_2tip_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_2tip4"),
        )
        return finger


@dataclass(eq=False)
class JustinRightMiddleFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_3middle_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_3middle4"),
        )
        return finger


@dataclass(eq=False)
class JustinRightRingFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_4ring_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_4ring4"),
        )
        return finger


@dataclass(eq=False)
class JustinGripper(EndEffector, HasFingers[GenericFinger], ABC):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0] * len(gripper_joints))),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [1.0] * len(gripper_joints))),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)


@dataclass(eq=False)
class JustinLeftHand(
    JustinGripper[
        Union[
            JustinLeftThumb,
            JustinLeftIndexFinger,
            JustinLeftMiddleFinger,
            JustinLeftRingFinger,
        ]
    ]
):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_arm7"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
        )
        return gripper


@dataclass(eq=False)
class JustinRightHand(
    JustinGripper[
        Union[
            JustinRightThumb,
            JustinRightRingFinger,
            JustinRightIndexFinger,
            JustinRightMiddleFinger,
        ]
    ]
):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_arm7"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.707, 0.707, 0.707, 0.707),
        )
        return gripper


@dataclass(eq=False)
class JustinLeftArm(Arm[JustinLeftHand]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_arm7"),
        )
        return arm


@dataclass(eq=False)
class JustinRightArm(Arm[JustinRightHand]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_arm7"),
        )
        return arm


@dataclass(eq=False)
class JustinCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(robot_root, "head2"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )
        return camera

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class JustinNeck(Neck[JustinCamera]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        neck = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso4"),
            tip=world.get_body_in_branch_by_name(robot_root, "head2"),
        )
        return neck


@dataclass(eq=False)
class JustinTorso(
    Torso, HasLeftRightArm[JustinLeftArm, JustinRightArm], HasNeck[JustinNeck]
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "torso4"),
        )
        return torso


@dataclass(eq=False)
class JustinMobileBase(MobileBase, HasTorso[JustinTorso]):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
        )
        return torso

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class Justin(AbstractRobot, HasMobileBase[JustinMobileBase]):

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_dlr_rollin_justin/urdf/rollin_justin.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"
