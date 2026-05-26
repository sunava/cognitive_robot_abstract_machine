from dataclasses import dataclass

from krrood.entity_query_language.factories import underspecified, variable_from

from pycram.datastructures.dataclasses import Context
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.base import ActionDescription
from pycram.testing import setup_world
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class DemoNavigateAction(ActionDescription):
    """
    Small PyCRAM action designator for demonstrating underspecified resolution.

    This is intentionally not the real NavigateAction: the real one starts the
    full motion stack. Here we only want to demonstrate that the plan resolves an
    underspecified action automatically when plan.perform() is called.
    """

    target_location: Pose
    keep_joint_states: bool = True

    def execute(self) -> None:
        position = self.target_location.to_position().to_np()
        print(
            "Executing resolved DemoNavigateAction at "
            f"x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}"
        )


def make_demo_context() -> tuple:
    """
    Standalone replacement for the mutable_model_world pytest fixture.

    Do not import pytest fixtures in demos. Fixtures are only meant to be called
    by pytest. A normal script should create its own world, robot view, and
    context explicitly.
    """

    world = setup_world()
    robot = PR2.from_world(world)
    context = Context(world=world, robot=robot)
    return world, robot, context


def main() -> None:
    world, _robot, context = make_demo_context()

    navigate_action = underspecified(DemoNavigateAction)(
        target_location=variable_from(
            [
                Pose.from_xyz_quaternion(1, 0, 0, reference_frame=world.root),
                Pose.from_xyz_quaternion(2, 0, 0, reference_frame=world.root),
            ]
        ),
        keep_joint_states=True,
    )

    plan = execute_single(action_like=navigate_action, context=context).plan

    plan.perform()

    resolved_action = plan.root.children[0].designator
    resolved_position = resolved_action.target_location.to_position().to_np()

    print("Underspecified DemoNavigateAction was resolved automatically.")
    print(
        "Selected target pose: "
        f"x={resolved_position[0]:.2f}, "
        f"y={resolved_position[1]:.2f}, "
        f"z={resolved_position[2]:.2f}"
    )


if __name__ == "__main__":
    main()
