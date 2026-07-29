from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from typing_extensions import Optional, Union, List, TYPE_CHECKING

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    Point3,
    Vector3,
    Quaternion,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from coraplex.datastructures.rotations import Rotations
from coraplex.datastructures.enums import (
    AxisIdentifier,
    ApproachDirection,
    VerticalAlignment,
    Arms,
)
from coraplex.tf_transformations import quaternion_multiply
from coraplex.utils import translate_pose_along_local_axis

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import EndEffector


@dataclass
class GraspDescription:
    """
    Describes a grasp configuration for a end_effector the description consists of the
    approach direction (the side from which to grasp e.g. FRONT, LEFT, etc and the
    vertical alignment (TOP, BOTTOM).

    The approach direction is interpreted relative to the robot: *FRONT* always denotes
    the side of the target that faces the robot, independent of the target's own
    orientation. It is resolved onto the target's frame via
    :meth:`resolve_approach_direction` when poses are calculated.
    """

    approach_direction: ApproachDirection
    """
    The robot-relative direction from which the body should be grasped. These are the
    four directions in the x-y plane (FRONT, BACK, LEFT, RIGHT), where FRONT is the side
    of the body facing the robot.
    """

    vertical_alignment: VerticalAlignment
    """
    The alignment of the gripper with the body in the z-axis (TOP, BOTTOM).
    """

    end_effector: EndEffector
    """
    The end_effector that is used to grasp the body.
    """

    rotate_gripper: bool = False
    """
    Rotate the gripper by 90 degrees.
    """

    manipulation_offset: float = 0.05
    """
    The offset between the center of the pose in the grasp sequence.
    """

    @classmethod
    def robot_direction_in_pose_frame(
        cls, end_effector: EndEffector, pose: Pose, robot_pose: Optional[Pose] = None
    ) -> Vector3:
        """
        The direction from the pose origin towards the root of the robot owning the
        end_effector, expressed in the pose's frame.

        :param end_effector: The end_effector whose robot defines the direction.
        :param pose: The pose whose frame the direction is expressed in.
        :param robot_pose: An explicit robot root pose to measure from, for example a
            standing pose the robot will navigate to. Defaults to the robot's current
            root pose.
        :return: The direction towards the robot in the pose's frame.
        """
        world = end_effector._world
        map_T_pose = world.transform(pose.to_homogeneous_matrix(), world.root).to_pose()

        if robot_pose is None:
            map_T_robot = end_effector._robot.root.global_pose
        else:
            robot_world = robot_pose.reference_frame._world
            map_T_robot = robot_world.transform(
                robot_pose.to_homogeneous_matrix(), robot_world.root
            ).to_pose()

        map_V_towards_robot = map_T_robot.to_position() - map_T_pose.to_position()
        pose_R_map = map_T_pose.to_rotation_matrix().inverse()

        return pose_R_map @ map_V_towards_robot

    def robot_facing_direction(
        self, target_pose: Pose, robot_pose: Optional[Pose] = None
    ) -> ApproachDirection:
        """
        The approach direction, expressed in the frame of the target pose, whose face
        points towards the robot.

        :param target_pose: The pose of the grasp target.
        :param robot_pose: An explicit robot root pose to measure from. Defaults to the
            robot's current root pose.
        :return: The object-frame direction facing the robot.
        """
        direction_towards_robot = self.robot_direction_in_pose_frame(
            self.end_effector, target_pose, robot_pose
        )
        side_vector = Vector3(
            direction_towards_robot.x, direction_towards_robot.y, np.nan
        )
        return self.calculate_closest_faces(side_vector)[0]

    def resolve_approach_direction(
        self, target_pose: Pose, robot_pose: Optional[Pose] = None
    ) -> ApproachDirection:
        """
        Translate the robot-relative approach direction into the frame of the target
        pose.

        *FRONT* always resolves to the side of the target that faces the robot.

        :param target_pose: The pose of the grasp target.
        :param robot_pose: An explicit robot root pose to resolve from, for example a
            standing pose the robot will navigate to. Defaults to the robot's current
            root pose.
        :return: The approach direction in the target's frame.
        """
        return self.approach_direction.to_object_frame(
            self.robot_facing_direction(target_pose, robot_pose)
        )

    def pose_sequence(
        self,
        target_T_grasp_pose: Pose,
        body: Body = None,
        reverse: bool = False,
        robot_pose: Optional[Pose] = None,
    ) -> List[Pose]:
        """
        Calculates the pose sequence to grasp something at the pose if the body is given
        its geometry is also taken into account. The pose sequence consists of 3 poses:
        one in front of the body (taking body geometry into account), one at the center
        of the body, and the last one above the body to lift it.

        :param target_T_grasp_pose: The pose of the grasp in the target frame.
        :param body: The body of the grasp.
        :param reverse: If the sequence should be reversed.
        :param robot_pose: An explicit robot root pose to resolve the approach direction
            from. Defaults to the robot's current root pose.
        :return: The pose sequence.
        """
        target = target_T_grasp_pose.reference_frame

        world = target._world

        resolved_approach_direction = self.resolve_approach_direction(
            target_T_grasp_pose, robot_pose
        )
        grasp_pose_R_gripper_goal = self.grasp_orientation(resolved_approach_direction)

        # if we just did target_T_grasp_pose @ grasp_pose_R_gripper_goal we would also rotate the translation in the
        # global frame, which we dont want here. Thus we just multiply the rotations, and take the translation as is
        target_R_gripper_goal = (
            target_T_grasp_pose.to_rotation_matrix()
            @ grasp_pose_R_gripper_goal.to_rotation_matrix()
        )
        target_T_gripper_goal: Pose = Pose(
            position=target_T_grasp_pose.to_position(),
            orientation=target_R_gripper_goal.to_quaternion(),
            reference_frame=target,
        )

        if body:
            bb_in_frame = body.collision.as_bounding_box_collection_in_frame(
                body
            ).bounding_box()

            approach_axis = np.array(resolved_approach_direction.axis.value, dtype=bool)

            # Pre-pose calculation
            offset = (
                np.array(bb_in_frame.dimensions)[approach_axis] / 2
                + self.manipulation_offset
            )[0]
        else:
            offset = 0

        target_T_gripper_goal_copy = deepcopy(target_T_gripper_goal)
        pre_pose = translate_pose_along_local_axis(
            target_T_gripper_goal_copy, self.manipulation_axis(), -offset
        )

        target_T_gripper_goal_copy = deepcopy(target_T_gripper_goal)

        # Lift pose calculation. We want the lift pose to be moved along the global z-axis, but the final pose should be in the target frame.
        map_T_grasp = world.transform(
            target_T_grasp_pose.to_homogeneous_matrix(), world.root
        )
        grasp_T_lift = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=self.manipulation_offset
        )

        # the grasp pose, not adjusted for the gripper orientation, used to calculate the lift pose
        map_T_lift = (map_T_grasp @ grasp_T_lift).to_position()

        # the result is transformed to the target frame
        target_P_lift = world.transform(map_T_lift, target)

        # the lift pose is adjusted for the gripper orientation, but without rotating the point we want to grasp
        lift_pose = Pose(
            target_P_lift, target_T_gripper_goal.to_quaternion(), reference_frame=target
        )

        sequence = [pre_pose, target_T_gripper_goal_copy, lift_pose]

        if reverse:
            sequence.reverse()
        return sequence

    def grasp_pose_sequence(self, body: Body):
        """
        Calculates the pose sequence to grasp the body.

        The sequence is 3 poses, one in front of the body (taking body geometry into
        account), one at the center of the body, and the last one above the body to lift
        it.

        :param body: The body of the grasp.
        :return: The pose sequence.
        """
        return self.pose_sequence(Pose(reference_frame=body), body)

    def place_pose_sequence(self, pose: Pose) -> List[Pose]:
        """
        Calculates the pose sequence to place a body at the given pose.

        Assumes that the end_effector is holding a body which is being placed.

        :param pose: The pose at which the body in the end_effector should be placed
        :return: The pose sequence.
        """
        body = self.end_effector.tool_frame.child_kinematic_structure_entities[0]
        return self.pose_sequence(pose, body, reverse=True)

    def manipulation_axis(self) -> List[float]:
        """
        Axis of the end_effector that is manipulating the body.

        Translates the x-axis of the global frame to how the end_effector is rotated.

        :returns: The axis of the end_effector that is manipulating the body.
        """
        return self.calculate_end_effector_axis(AxisIdentifier.X)

    def lift_axis(self) -> List[float]:
        """
        Axis of the end_effector that is lifting the body.

        Translates the z-axis of the global frame to how the end_effector is rotated.

        :returns: The axis of the end_effector that is lifting the body.
        """
        return self.calculate_end_effector_axis(AxisIdentifier.Z)

    def calculate_end_effector_axis(self, axis: AxisIdentifier) -> List[float]:
        """
        Calculates the corresponding axis of the end_effector for a given axis of the
        body.

        :param axis: The axis of the body as a list of [x, y, z] indices.
        :return: The corresponding axis of the end_effector as a list of [x, y, z]
            values.
        """
        axis_list = axis.value
        front_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            *axis_list, reference_frame=self.end_effector._world.root
        )

        grasp_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
            0,
            0,
            0,
            *self.end_effector.front_facing_orientation.to_np(),
            reference_frame=self.end_effector._world.root,
        )
        world = self.end_effector._world

        front_global = world.transform(front_pose, world.root)

        grasp_global = world.transform(grasp_pose, world.root)

        t = grasp_global.inverse().to_np() @ front_global.to_np()

        return t[:3, 3].astype(float).tolist()

    def grasp_orientation(
        self, approach_direction: Optional[ApproachDirection] = None
    ) -> Quaternion:
        """
        The orientation of the grasp in the frame of the grasp target.

        Takes into account the approach direction and vertical alignment.

        :param approach_direction: The approach direction expressed in the target's
            frame, usually obtained via :meth:`resolve_approach_direction`. Falls back
            to the raw approach direction of this description.
        """
        rotation = Rotations.SIDE_ROTATIONS[
            approach_direction or self.approach_direction
        ]
        rotation = quaternion_multiply(
            rotation, Rotations.VERTICAL_ROTATIONS[self.vertical_alignment]
        )
        rotation = quaternion_multiply(
            rotation, Rotations.HORIZONTAL_ROTATIONS[self.rotate_gripper]
        )

        orientation = quaternion_multiply(
            rotation, self.end_effector.front_facing_orientation.to_np()
        )

        norm = math.sqrt(sum(comp**2 for comp in orientation))
        orientation = [comp / norm for comp in orientation]

        return Quaternion(*orientation)

    def edge_offset(
        self, body: Body, approach_direction: Optional[ApproachDirection] = None
    ) -> float:
        """
        The offset between the center of the body and its edge in the direction of the
        approach axis.

        :param body: The body to calculate the edge offset for.
        :param approach_direction: The approach direction expressed in the body's frame.
            Resolved from the robot-relative approach direction if omitted.
        :return: The edge offset.
        """
        approach_direction = approach_direction or self.resolve_approach_direction(
            Pose(reference_frame=body)
        )
        rim_direction_index = approach_direction.axis.value.index(1)

        rim_offset = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .dimensions[rim_direction_index]
            / 2
        )
        return rim_offset

    def grasp_pose(self, body: Body, grasp_edge: bool = False) -> Pose:
        """
        The pose for the given end_effector to grasp the body in the frame of the body.

        :param body: The body to grasp.
        :param grasp_edge: Indicates if the pose should be for the edge of the body or
            the center.
        :return: The pose of the body in the body frame.
        """
        approach_direction = self.resolve_approach_direction(Pose(reference_frame=body))
        edge_offset = -self.edge_offset(body, approach_direction) if grasp_edge else 0
        orientation = self.grasp_orientation(approach_direction)
        grasp_pose = Pose(Point3(edge_offset, 0, 0), orientation, reference_frame=body)

        return grasp_pose

    @classmethod
    def calculate_grasp_descriptions(
        cls,
        end_effector: EndEffector,
        pose: Pose,
        grasp_alignment: Optional[PreferredGraspAlignment] = None,
    ) -> List[GraspDescription]:
        """
        This method determines the possible grasp configurations (approach axis and
        vertical alignment) of the body, taking into account the bodies orientation,
        position, and whether the gripper should be rotated by 90°.

        The approach directions of the returned descriptions are robot-relative: the
        best-aligned side of the body resolves to *FRONT*.

        :param end_effector: The end_effector to use.
        :param grasp_alignment: An optional PreferredGraspAlignment object that
            specifies preferred grasp axis,
        :param pose: The pose of the object to be grasped.
        :return: A sorted list of GraspDescription instances representing all grasp
            permutations.
        """
        if grasp_alignment:
            side_axis = grasp_alignment.preferred_axis
            vertical = grasp_alignment.with_vertical_alignment
            rotated_gripper = grasp_alignment.with_rotated_gripper
        else:
            side_axis, vertical, rotated_gripper = (
                AxisIdentifier.Undefined,
                False,
                False,
            )

        object_V_robot = cls.robot_direction_in_pose_frame(end_effector, pose)

        vector_side = Vector3(object_V_robot.x, object_V_robot.y, np.nan)
        robot_facing_direction = GraspDescription.calculate_closest_faces(vector_side)[
            0
        ]
        side_faces = [
            face.to_robot_relative(robot_facing_direction)
            for face in GraspDescription.calculate_closest_faces(vector_side, side_axis)
        ]

        vector_vertical = Vector3(np.nan, np.nan, object_V_robot.z)
        if vertical:
            vertical_faces = GraspDescription.calculate_closest_faces(vector_vertical)
        else:
            vertical_faces = [VerticalAlignment.NoAlignment]

        grasp_configs = [
            GraspDescription(
                approach_direction=side,
                vertical_alignment=top_face,
                rotate_gripper=rotated_gripper,
                end_effector=end_effector,
            )
            for top_face in vertical_faces
            for side in side_faces
        ]

        return grasp_configs

    @staticmethod
    def calculate_closest_faces(
        pose_to_robot_vector: Vector3,
        specified_grasp_axis: AxisIdentifier = AxisIdentifier.Undefined,
    ) -> Union[
        Tuple[ApproachDirection, ApproachDirection],
        Tuple[VerticalAlignment, VerticalAlignment],
    ]:
        """
        Determines the faces of the object based on the input vector.

        If `specified_grasp_axis` is None, it calculates the primary and secondary faces
        based on the vector's magnitude determining which sides of the object are most
        aligned with the robot. This will either be the x, y plane for side faces or the
        z axis for top/bottom faces. If `specified_grasp_axis` is provided, it only
        considers the specified axis and calculates the faces aligned with that axis.

        :param pose_to_robot_vector: A 3D vector representing one of the robot's axes in
            the pose's frame, with irrelevant components set to np.nan.
        :param specified_grasp_axis: Specifies a specific axis (e.g., X, Y, Z) to focus
            on.
        :return: A tuple of two Grasp enums representing the primary and secondary
            faces.
        """
        all_axes = [AxisIdentifier.X, AxisIdentifier.Y, AxisIdentifier.Z]

        if not specified_grasp_axis == AxisIdentifier.Undefined:
            valid_axes = [specified_grasp_axis]
        else:
            valid_axes = [
                axis
                for axis in all_axes
                if not np.isnan(pose_to_robot_vector.to_list()[axis.value.index(1)])
            ]

        object_to_robot_vector = np.array(pose_to_robot_vector.to_list()) + 1e-9
        sorted_axes = sorted(
            valid_axes,
            key=lambda axis: abs(object_to_robot_vector[axis.value.index(1)]),
            reverse=True,
        )

        primary_axis: AxisIdentifier = sorted_axes[0]
        primary_sign = int(np.sign(object_to_robot_vector[primary_axis.value.index(1)]))

        primary_axis_class = (
            VerticalAlignment if primary_axis == AxisIdentifier.Z else ApproachDirection
        )
        primary_face = primary_axis_class.from_axis_direction(
            primary_axis, primary_sign
        )

        if len(sorted_axes) > 1:
            secondary_axis: AxisIdentifier = sorted_axes[1]
            secondary_sign = int(
                np.sign(object_to_robot_vector[secondary_axis.value.index(1)])
            )
        else:
            secondary_axis: AxisIdentifier = primary_axis
            secondary_sign = -primary_sign

        secondary_axis_class = (
            VerticalAlignment
            if secondary_axis == AxisIdentifier.Z
            else ApproachDirection
        )
        secondary_face = secondary_axis_class.from_axis_direction(
            secondary_axis, secondary_sign
        )

        return primary_face, secondary_face

    def __hash__(self):
        return id(self)


@dataclass
class PreferredGraspAlignment:
    """
    Description of the preferred grasp alignment for an object.
    """

    preferred_axis: Optional[AxisIdentifier]
    """
    The preferred axis, X, Y, or Z, for grasping the object, or None if not specified.
    """

    with_vertical_alignment: bool
    """
    Indicates if the object should be grasped with a vertical alignment.
    """

    with_rotated_gripper: bool
    """
    Indicates if the gripper should be rotated by 90° around X.
    """


@dataclass(eq=False, init=False)
class GraspPose(Pose):
    """
    A pose from which a grasp can be performed along with the respective arm and grasp
    description.
    """

    arm: Optional[Arms] = None
    """
    Arm corresponding to the grasp pose.
    """

    grasp_description: Optional[GraspDescription] = None
    """
    Grasp description corresponding to the grasp pose.
    """

    def __init__(
        self,
        position: Optional[Point3] = None,
        orientation: Optional[Quaternion] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
        arm: Optional[Arms] = None,
        grasp_description: Optional[GraspDescription] = None,
    ):
        super().__init__(position, orientation, reference_frame)
        self.arm = arm
        self.grasp_description = grasp_description

    @classmethod
    def from_pose(
        cls,
        pose: Pose,
        arm: Arms,
        grasp_description: GraspDescription,
    ) -> GraspPose:
        return cls(
            position=pose.to_position(),
            orientation=pose.to_quaternion(),
            reference_frame=pose.reference_frame,
            arm=arm,
            grasp_description=grasp_description,
        )
