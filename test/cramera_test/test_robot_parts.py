"""
Tests for reading a world's sem_dt robot-part annotations and publishing them.
"""

from dataclasses import dataclass, field

from typing_extensions import Any, List, Optional

from cramera.robot_parts import (
    ArmSide,
    RobotPartAnnotation,
    RobotPartRole,
    model_identity,
    robot_base_link,
    robot_bases,
    robot_model_names,
    robot_prefix,
)

# %% mimics standing in for the sem_dt annotations of a world


@dataclass
class NamedBody:
    """
    A world body carrying a model-prefixed name.
    """

    name: str


@dataclass
class PartWithBodies:
    """
    A robot part exposing the bodies whose link names get published.
    """

    bodies: List[Any] = field(default_factory=list)


@dataclass
class EndEffectorPart(PartWithBodies):
    """
    An end effector attached to an arm.
    """


@dataclass
class ArmPart(PartWithBodies):
    """
    An arm, optionally carrying an end effector.
    """

    end_effector: Optional[EndEffectorPart] = None


@dataclass
class TwoArmedRobot:
    """
    A robot naming which of its arms is the left and which is the right one.
    """

    left: ArmPart
    right: ArmPart

    def get_arms(self) -> List[ArmPart]:
        return [self.left, self.right]

    def get_left_arm_if_specified(self) -> ArmPart:
        return self.left

    def get_right_arm_if_specified(self) -> ArmPart:
        return self.right


@dataclass
class OneArmedRobot:
    """
    A robot that specifies neither a left nor a right arm.
    """

    arm: ArmPart

    root: NamedBody = field(default_factory=lambda: NamedBody("robot/base_link"))
    """
    The robot's root body, read for the base link name.
    """

    def get_arms(self) -> List[ArmPart]:
        return [self.arm]

    def get_left_arm_if_specified(self) -> None:
        return None

    def get_right_arm_if_specified(self) -> None:
        return None


# %% link names


class TestLinkNames:
    def test_the_model_prefix_is_stripped(self):
        part = PartWithBodies(bodies=[NamedBody("pr2/l_wrist_link")])
        assert RobotPartAnnotation.link_names(part) == ["l_wrist_link"]

    def test_an_unprefixed_name_is_kept(self):
        part = PartWithBodies(bodies=[NamedBody("l_wrist_link")])
        assert RobotPartAnnotation.link_names(part) == ["l_wrist_link"]

    def test_a_part_without_bodies_has_no_links(self):
        assert RobotPartAnnotation.link_names(PartWithBodies()) == []


# %% reading the annotations off a robot


class TestDescribeRobotParts:
    def test_each_arm_is_published_with_its_end_effector(self):
        """
        An arm and the end effector it carries are published as two annotations, the end
        effector naming the arm it is attached to.
        """
        gripper = EndEffectorPart(bodies=[NamedBody("pr2/l_gripper_link")])
        arm = ArmPart(
            bodies=[NamedBody("pr2/l_upper_arm_link"), NamedBody("pr2/l_gripper_link")],
            end_effector=gripper,
        )
        robot = OneArmedRobot(arm=arm)

        assert RobotPartAnnotation.of_robot(robot) == [
            RobotPartAnnotation(
                name="ArmPart",
                role=RobotPartRole.ARM,
                side=None,
                links=["l_upper_arm_link"],
                robot="robot",
            ),
            RobotPartAnnotation(
                name="EndEffectorPart",
                role=RobotPartRole.END_EFFECTOR,
                side=None,
                links=["l_gripper_link"],
                attached_to="ArmPart",
                robot="robot",
            ),
        ]

    def test_every_annotation_names_the_robot_it_was_read_off(self):
        """
        A world holding several robots publishes one flat list of part annotations; the
        prefix each one carries is what says whose arm it is.
        """
        robot = OneArmedRobot(
            arm=ArmPart(bodies=[NamedBody("uMe/arm_link")]),
            root=NamedBody("uMe/base_link"),
        )

        [annotation] = RobotPartAnnotation.of_robot(robot)

        assert annotation.robot == "uMe"

    def test_a_robot_whose_bodies_carry_no_prefix_names_none(self):
        robot = OneArmedRobot(
            arm=ArmPart(bodies=[NamedBody("arm_link")]), root=NamedBody("base_link")
        )

        [annotation] = RobotPartAnnotation.of_robot(robot)

        assert annotation.robot is None

    def test_the_side_comes_from_the_robots_own_left_right_annotation(self):
        """
        Which arm is the left one is what the robot annotation says, not what the part
        or link names happen to spell.
        """
        robot = TwoArmedRobot(
            left=ArmPart(bodies=[NamedBody("pr2/first_link")]),
            right=ArmPart(bodies=[NamedBody("pr2/second_link")]),
        )
        sides = [annotation.side for annotation in RobotPartAnnotation.of_robot(robot)]
        assert sides == [ArmSide.LEFT, ArmSide.RIGHT]

    def test_an_arm_of_a_robot_without_a_left_and_a_right_arm_has_no_side(self):
        robot = OneArmedRobot(arm=ArmPart(bodies=[NamedBody("stretch/arm_link")]))
        [annotation] = RobotPartAnnotation.of_robot(robot)
        assert annotation.side is None


# %% the published shape


class TestRobotPartAnnotationPayload:
    def test_a_payload_round_trips(self):
        annotation = RobotPartAnnotation(
            name="PR2LeftGripper",
            role=RobotPartRole.END_EFFECTOR,
            side=ArmSide.LEFT,
            links=["l_gripper_link"],
            attached_to="PR2LeftArm",
        )
        assert RobotPartAnnotation.from_payload(annotation.to_payload()) == annotation

    def test_the_payload_names_the_side_in_lower_case(self):
        annotation = RobotPartAnnotation(
            name="PR2LeftArm", role=RobotPartRole.ARM, side=ArmSide.LEFT
        )
        assert annotation.to_payload() == {
            "name": "PR2LeftArm",
            "role": "arm",
            "side": "left",
            "links": [],
            "attachedTo": None,
            "robot": None,
        }

    def test_a_sideless_payload_round_trips(self):
        annotation = RobotPartAnnotation(
            name="StretchArm", role=RobotPartRole.ARM, side=None
        )
        assert RobotPartAnnotation.from_payload(annotation.to_payload()) == annotation


# %% naming a world's robots


class TestRobotIdentity:
    """
    Two robots of the same class carry the same link names and the same class name; only
    the prefix their bodies are spawned under tells them apart.
    """

    def test_the_prefix_is_the_one_the_root_body_is_named_under(self):
        robot = OneArmedRobot(arm=ArmPart(), root=NamedBody("walker_s2/base_link"))

        assert robot_prefix(robot) == "walker_s2"
        assert robot_base_link(robot) == "base_link"

    def test_an_unprefixed_robot_has_no_prefix(self):
        robot = OneArmedRobot(arm=ArmPart(), root=NamedBody("base_link"))

        assert robot_prefix(robot) == ""
        assert robot_base_link(robot) == "base_link"

    def test_every_robot_is_named_after_its_class(self):
        robots = [
            OneArmedRobot(arm=ArmPart(), root=NamedBody("a/base_link")),
            TwoArmedRobot(left=ArmPart(), right=ArmPart()),
        ]

        assert robot_model_names(robots) == ["onearmedrobot", "twoarmedrobot"]

    def test_robots_of_one_class_are_numbered_apart(self):
        """
        Two robots of a class would write over each other's URDF, and the viewer would
        have no way to tell their models apart.
        """
        robots = [
            OneArmedRobot(arm=ArmPart(), root=NamedBody("pr2_1/base_link")),
            OneArmedRobot(arm=ArmPart(), root=NamedBody("pr2_2/base_link")),
            OneArmedRobot(arm=ArmPart(), root=NamedBody("pr2_3/base_link")),
        ]

        assert robot_model_names(robots) == [
            "onearmedrobot",
            "onearmedrobot_2",
            "onearmedrobot_3",
        ]

    def test_the_bases_of_a_world_are_keyed_by_prefix(self):
        robots = [
            OneArmedRobot(arm=ArmPart(), root=NamedBody("pr2_1/base_link")),
            OneArmedRobot(arm=ArmPart(), root=NamedBody("uMe/pelvis")),
        ]

        assert robot_bases(robots) == {"pr2_1": "base_link", "uMe": "pelvis"}


# %% identifying a model within a world
class TestModelIdentity:
    """
    Telling a model's role (robot or environment) and world-instance prefix apart from
    its link names alone, shared by onboarding and live model serving.
    """

    def test_a_model_named_under_a_robots_prefix_is_that_robot(self):
        prefix, is_robot = model_identity(
            links=["base_link", "arm_link"],
            world_body_names=["pr2_1/base_link", "pr2_1/arm_link"],
            robot_bases={"pr2_1": "base_link"},
            probe_link_count=12,
        )

        assert is_robot is True
        assert prefix == "pr2_1"

    def test_each_robot_of_a_world_gets_its_own_model_recognized(self):
        """
        Two robots of the same class carry the same link names, so only their prefixes
        tell their models apart -- and both of them are a robot.
        """
        robot_bases = {"pr2_1": "base_link", "pr2_2": "base_link"}
        world_body_names = ["pr2_1/base_link", "pr2_2/base_link", "lab_1/table"]

        identities = [
            model_identity(
                links=["base_link"],
                world_body_names=world_body_names,
                robot_bases=robot_bases,
                probe_link_count=12,
            ),
            model_identity(
                links=["table"],
                world_body_names=world_body_names,
                robot_bases=robot_bases,
                probe_link_count=12,
            ),
        ]

        assert identities == [("pr2_1", True), ("lab_1", False)]

    def test_a_model_named_under_no_robots_prefix_is_an_environment_model(self):
        prefix, is_robot = model_identity(
            links=["table", "lid"],
            world_body_names=["lab_1/table", "lab_1/lid"],
            robot_bases={"pr2_1": "base_link"},
            probe_link_count=12,
        )

        assert is_robot is False
        assert prefix == "lab_1"

    def test_an_unprefixed_world_has_no_prefix(self):
        prefix, is_robot = model_identity(
            links=["table"],
            world_body_names=["table"],
            robot_bases={"": "base_link"},
            probe_link_count=12,
        )

        assert prefix == ""
        assert is_robot is False

    def test_an_unprefixed_world_falls_back_to_the_robots_base_link(self):
        """
        Without prefixes there is nothing to match a robot's model on, so the model
        holding a robot's base link is that robot's.
        """
        prefix, is_robot = model_identity(
            links=["base_link", "arm_link"],
            world_body_names=["base_link", "arm_link"],
            robot_bases={"": "base_link"},
            probe_link_count=12,
        )

        assert prefix == ""
        assert is_robot is True

    def test_only_the_first_probe_link_count_links_are_checked_for_a_prefix(self):
        prefix, _ = model_identity(
            links=["a", "b", "c"],
            world_body_names=["lab_1/c"],
            robot_bases={"pr2_1": "base_link"},
            probe_link_count=2,
        )

        assert prefix == ""

    def test_no_bound_robot_means_nothing_is_the_robot(self):
        _, is_robot = model_identity(
            links=["base_link"],
            world_body_names=["base_link"],
            robot_bases={},
            probe_link_count=12,
        )

        assert is_robot is False
