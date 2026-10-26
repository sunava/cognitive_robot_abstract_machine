"""
The arms a step names: the builder's own choice of a robot's arm, offered for every
field that takes an arm, several arms or an arm's end effector, and resolved into the
robot's parts when the step's action is built.

coraplex names an arm by the robot part itself, which exists only once a robot stands in
a world; a plan names it by side, so that the same plan reads for every robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from typing_extensions import Any, List, get_args, get_origin

from cramera_bridge.exceptions import ArmNotOnRobot
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm, EndEffector


class Arms(Enum):
    """
    The arms a step can name; the names are those plans are saved with.
    """

    LEFT = auto()
    """
    The left arm of a robot with a left and a right arm.
    """

    RIGHT = auto()
    """
    The right arm of a robot with a left and a right arm.
    """

    BOTH = auto()
    """
    Every arm of the robot, or the only one of a robot with one arm.
    """


ARM_PARTS = (Arm, EndEffector)
"""
The robot parts a field taking an arm names: the arm or its end effector.
"""


@dataclass(frozen=True)
class ArmField:
    """
    The type of a field, asked whether it takes an arm.
    """

    annotation: Any
    """
    A resolved field type.
    """

    def part(self) -> Any:
        """
        :return: The robot part the field takes one or several of, else None.
        """
        if get_origin(self.annotation) in (list, List):
            (member,) = get_args(self.annotation) or (None,)
            return member if self.is_arm_part(member) else None
        return self.annotation if self.is_arm_part(self.annotation) else None

    def takes_several(self) -> bool:
        """
        :return: Whether the field takes a list of arms.
        """
        return get_origin(self.annotation) in (list, List)

    @staticmethod
    def is_arm_part(candidate: Any) -> bool:
        """
        :param candidate: A type.
        :return: Whether it is an arm or an end effector.
        """
        return isinstance(candidate, type) and issubclass(candidate, ARM_PARTS)


@dataclass(frozen=True)
class RobotArms:
    """
    Resolves the arms a step names into the parts of the robot performing it.
    """

    robot: AbstractRobot
    """
    The robot.
    """

    def arms(self, choice: Arms) -> List[Arm]:
        """
        :param choice: The arms a step names.
        :return: The robot's arms the choice stands for.
        :raises ArmNotOnRobot: When the robot, or a part of it, names no arm on the
            side.
        """
        if choice is Arms.BOTH:
            return list(self.robot.all_arms)
        arm = (
            self.robot.get_left_arm_if_specified()
            if choice is Arms.LEFT
            else self.robot.get_right_arm_if_specified()
        )
        if arm is None:
            raise ArmNotOnRobot(choice.name, type(self.robot).__name__)
        return [arm]

    def value(self, choice: Arms, annotation: Any) -> Any:
        """
        :param choice: The arms a step names.
        :param annotation: The type of the field taking them.
        :return: The field's value: the arms for a field taking several, else the one
            arm, or its end effector for a field taking an end effector.
        :raises ArmNotOnRobot: When the choice names several arms for a field taking
            one.
        """
        field_type = ArmField(annotation)
        arms = self.arms(choice)
        if field_type.takes_several():
            return arms
        if len(arms) != 1:
            raise ArmNotOnRobot(choice.name, type(self.robot).__name__)
        (arm,) = arms
        if issubclass(field_type.part(), EndEffector):
            return arm.end_effector
        return arm
