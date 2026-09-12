"""
The action calls the Plan Builder writes into a generated demo.

The generated file hands these actions positional arguments, so their order is part of
the contract between the two packages: reordering a parameter here silently changes what
the generated demo does, and has to fail here rather than in somebody's demo run.
"""

import inspect

import pytest
from typing_extensions import ClassVar, List

from cramera.paths import WEB_ROOT

pytest.importorskip("coraplex", reason="coraplex not installed")

from coraplex.datastructures.grasp import GraspDescription  # noqa: E402
from coraplex.robot_plans.actions.core.pick_up import PickUpAction  # noqa: E402
from coraplex.robot_plans.actions.core.placing import PlaceAction  # noqa: E402
from semantic_digital_twin.robots.robot_parts import MobileBase  # noqa: E402
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase  # noqa: E402


def leading_parameters(action, count: int) -> List[str]:
    """
    The first parameters of an action, in the order a caller passes them positionally.

    :param action: The action class whose signature is read.
    :param count: How many leading parameters to name.
    """
    return list(inspect.signature(action).parameters)[:count]


# %% what a generated Pick / Place step calls


class TestGeneratedActionCalls:
    def test_pick_up_takes_the_object_then_the_arm_then_the_grasp(self):
        """
        Generated as ``PickUpAction(_pick_<id>, Arms.<arm>, _grasp_<id>)``.
        """
        assert leading_parameters(PickUpAction, 3) == [
            "object_designator",
            "arm",
            "grasp_description",
        ]

    def test_place_takes_the_object_then_the_target_then_the_arm(self):
        """
        Generated as ``PlaceAction(<body>, <target pose>, Arms.<arm>)``.
        """
        assert leading_parameters(PlaceAction, 3) == [
            "object_designator",
            "target_location",
            "arm",
        ]

    def test_the_default_grasp_takes_the_end_effector_the_pose_and_the_body(self):
        """
        Generated as ``GraspDescription.robot_relative_default(<end effector>, <pose>,
        <body>)``, leaving the side to approach from to the robot's reach.
        """
        assert leading_parameters(GraspDescription.robot_relative_default, 3) == [
            "end_effector",
            "pose",
            "body",
        ]


# %% what a generated demo writes to keep the base still


class TestGeneratedBaseControl:
    """
    A generated demo pins whole-body control by assigning the mobile base's own field,
    guarded by the mixin, so both have to be there to assign.
    """

    def test_the_mobile_base_carries_the_setting_that_is_assigned(self):
        """
        Generated as ``robot.mobile_base.full_body_controlled = ...``.
        """
        assert "full_body_controlled" in inspect.signature(MobileBase).parameters

    def test_the_guard_the_assignment_is_wrapped_in_exists(self):
        """
        Generated as ``if isinstance(robot, HasMobileBase):``, so a robot without a base
        is left alone.
        """
        assert isinstance(HasMobileBase, type)

    def test_a_robot_without_a_mobile_base_is_not_one(self):
        """
        The guard has to actually exclude such a robot, or the assignment raises.
        """
        from semantic_digital_twin.robots.tracy import Tracy

        assert not issubclass(Tracy, HasMobileBase)


# %% what the page offers


class TestPlanBuilderPalette:
    """
    The blocks the page has to offer for a plan that picks and places without a
    transport.
    """

    STEP_KINDS: ClassVar[List[str]] = ["transport", "pick", "place"]
    """
    Step kinds acting on a placed object, as ``core/plan_steps.js`` lists them.
    """

    def test_every_object_step_kind_is_an_offered_block(self):
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        for kind in self.STEP_KINDS:
            assert "\n    %s: { name: " % kind in page_script, kind

    def test_the_step_kinds_are_the_ones_the_shared_module_names(self):
        """
        The page reads the kinds from ``core/plan_steps.js``; a kind added to one and
        not the other is a block whose object is never spawned.
        """
        module = (WEB_ROOT / "core" / "plan_steps.js").read_text(encoding="utf-8")

        assert (
            "const ACTS_ON_AN_OBJECT = ['%s']" % "', '".join(self.STEP_KINDS) in module
        )

    def test_the_page_loads_every_shared_module_it_reads(self):
        page = (WEB_ROOT / "plan_builder.html").read_text(encoding="utf-8")

        for module in ("plan_steps", "base_control", "execution_environment"):
            assert '<script src="core/%s.js">' % module in page, module

    def test_the_page_offers_the_base_control_choice(self):
        page = (WEB_ROOT / "plan_builder.html").read_text(encoding="utf-8")

        assert 'id="pb-base"' in page


# %% a robot that is not parsed from a description


class TestRobotsWithoutADescriptionFile:
    """
    The generated demo spawns every robot through a ``RobotSpecification``, which asks
    the robot type for itself rather than parsing its file, so a robot built from
    measurements can be offered next to the URDF ones.
    """

    def test_a_robot_type_creates_itself_from_its_description(self):
        """
        Generated as ``RobotSpecification(semantic_annotation_type=<Cls>, ...)``, which
        calls this.
        """
        from semantic_digital_twin.robots.robot_parts import AbstractRobot

        assert callable(getattr(AbstractRobot, "from_description", None))

    def test_the_page_spells_out_the_import_of_an_external_robot(self):
        """
        A robot outside ``semantic_digital_twin.robots`` carries its own import line.
        """
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        assert (
            "import: 'from siemens_external_robots.robots.continuum_robot import ContinuumRobot'"
            in page_script
        )
        offered = page_script.split("const WORKING_ROBOTS = ")[1].split("]")[0]
        assert "'ContinuumRobot'" in offered  # offered in the dropdown

    def test_the_page_narrows_the_palette_to_what_the_robot_can_do(self):
        """
        A robot with no base and no torso offers Pick and Place only.
        """
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        assert "steps: ['pick', 'place']" in page_script
        assert "Object.keys(BLOCKS).filter(isOffered)" in page_script

    def test_the_generated_world_is_built_from_a_specification(self):
        """
        Generated as ``WorldSpecification.from_urdf(...)`` or ``from_gazebo(...)`` with
        ``robots=[robot]``; the robot's own description is never parsed by the demo.
        """
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        assert "URDFParser.from_file(' + R.cls" not in page_script
        assert "WorldSpecification.from_gazebo(" in page_script
        assert "WorldSpecification.from_urdf(" in page_script

    def test_the_page_offers_the_warehouse_of_the_g1_demo(self):
        page = (WEB_ROOT / "plan_builder.html").read_text(encoding="utf-8")

        assert "no_roof_small_warehouse.world" in page


# %% the humanoids of the siemens_external_robots package


class TestTheExternalHumanoids:
    """
    The Walker S2 and the uMe are parsed from their own ROS packages and drive as a
    whole, so they are listed like the continuum robot, with an import line of their
    own, and are lifted onto the floor after spawning, since their roots sit in the
    pelvis and the trunk.
    """

    ROBOTS: ClassVar[List[str]] = ["WalkerS2", "UMe"]
    MODULES: ClassVar[List[str]] = ["walker_s2", "ume"]

    def test_the_page_spells_out_the_import_of_each_humanoid(self):
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        for robot, module in zip(self.ROBOTS, self.MODULES):
            assert (
                "import: 'from siemens_external_robots.robots.%s import %s'"
                % (module, robot)
                in page_script
            ), robot

    def test_each_humanoid_is_offered_in_the_dropdown(self):
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        for robot in self.ROBOTS:
            assert (
                "'%s'" % robot
                in page_script.split("const WORKING_ROBOTS = ")[1].split("]")[0]
            ), robot

    def test_a_humanoid_offers_everything_but_a_torso_move(self):
        """
        Neither robot's torso has a raised or lowered state: the Walker S2's waist
        bends, the uMe's trunk is rigid.
        """
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        assert page_script.count(
            "steps: ['park_arms', 'navigate', 'transport', 'pick', 'place']"
        ) == len(self.ROBOTS)

    def test_both_output_styles_stand_the_robot_on_the_floor(self):
        """
        Generated as ``standing = max(0.0, -world.height_of_lowest_collision_point_of_branch(...))``
        and a lifted odom, in the flat script and in the demonstration class alike.
        """
        from semantic_digital_twin.world import World

        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        assert "standingLines(R.cls, 'robot_xy'" in page_script
        assert "standingLines('self.used_robot', 'ROBOT_XY'" in page_script
        assert callable(getattr(World, "height_of_lowest_collision_point_of_branch"))

    def test_a_robot_that_cannot_stand_still_chooses_its_own_base_control(self):
        """
        The uMe's four-joint arms need the base to drive while reaching, so selecting it
        switches the base-control choice to the robot's own setting rather than leaving
        the page's default, which pins the base still and fails every pick.
        """
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")
        base_control = (WEB_ROOT / "core" / "base_control.js").read_text(
            encoding="utf-8"
        )

        assert "baseControl: 'robot_default'" in page_script
        assert "name: 'robot_default'" in base_control
        assert "$('pb-base').value = robotInfo().baseControl" in page_script
