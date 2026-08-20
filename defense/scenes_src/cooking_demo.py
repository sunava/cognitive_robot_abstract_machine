"""
Cooking-task demo recorded for the defense deck: one PR2, one apartment kitchen, one
continuous session in which the robot cuts bread, pours from a cup into a bowl, and
mixes the bowl's contents.

The point of the recording is that all three actions are the *same* OAAT structure with
different bindings — so they run in one world, on one robot, without the plan being
rewritten between them.

Recorded with::

    CRAMERA_SCENES=defense/scenes cramera-onboard defense/scenes_src/cooking_demo.py \
        --name pr2_cooking
"""

from __future__ import annotations

import logging

from experiments.tool_based_actions.simple_demo.demo_world import (
    BASE_POSITION_XYZ,
    BOWL_COLOR,
    BREAD_COLOR,
    CUP_COLOR,
    CUT_MOUNT,
    MIX_MOUNT,
    POUR_MOUNT,
    add_cutting_board,
    parse_object,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Bread,
    CuttingKnife,
    PouringCup,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, CuttingTechnique
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import attach_tool, setup_world, start_visualization
from coraplex.view_manager import ViewManager

logger = logging.getLogger(__name__)

# %% counter layout

BREAD_POSITION_XYZ = (2.40, 2.48, 1.00)
"""
Where the bread rests on the kitchen counter, left of the bowl from the robot's view.
"""

BOWL_POSITION_XYZ = (2.40, 2.20, 1.00)
"""
Where the bowl stands on the kitchen counter.
"""

KNIFE_PARKING_XYZ = (2.28, 2.70, 1.03)
"""
Where the knife is set down once the cutting is finished, so the gripper is free for the
cup.
"""

CUP_PARKING_XYZ = (2.30, 1.82, 1.00)
"""
Where the cup waits on the counter before and after the pour.
"""

WHISK_PARKING_XYZ = (2.30, 1.66, 1.00)
"""
Where the whisk waits on the counter before the mixing starts.
"""

CUTS_ALONG_LOCAL_X = 3
"""
Number of slices taken off the bread.
"""

SLICE_THICKNESS = 0.03
"""
Slice thickness in meters, bound as the cutting action's spacing parameter.
"""


# %% tool handling


def reattach(world: World, body: Body, parent: KinematicStructureEntity, **offset) -> None:
    """
    Re-parent a body with a fixed connection, so a tool can move between the counter and
    a gripper without being re-parsed.

    Every tool exists from the first recorded tick — the recorder binds the object set
    once at startup, so a tool merged mid-run would stay invisible in the recording.

    :param world: The world the body lives in.
    :param body: The body to re-parent.
    :param parent: The body's new parent — the world root to set it down, a gripper's
        tool frame to pick it up.
    :param offset: Keyword arguments of the offset transform against the new parent.
    """
    with world.modify_world():
        world.remove_connection(body.parent_connection)
        world.add_connection(
            FixedConnection(
                parent=parent,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=parent, **offset
                ),
            )
        )


def park_tool(world: World, tool: Body, position_xyz: tuple[float, float, float]) -> None:
    """
    Set a held tool down on the counter, freeing the gripper that held it.

    :param world: The world the tool lives in.
    :param tool: The tool body currently attached to a gripper.
    :param position_xyz: Where on the counter the tool is placed.
    """
    x, y, z = position_xyz
    reattach(world, tool, world.root, x=x, y=y, z=z)


def take_tool(world: World, robot: PR2, arm: Arms, tool: Body, mount: dict) -> None:
    """
    Pick a tool up off the counter into the arm's gripper.

    :param world: The world the tool lives in.
    :param robot: The robot taking the tool.
    :param arm: The arm that takes it.
    :param tool: The tool body resting on the counter.
    :param mount: The tool's mount transform on the gripper's tool frame.
    """
    reattach(world, tool, ViewManager.get_end_effector_view(arm, robot).tool_frame, **mount)


# %% world construction


def merge_on_counter(
    world: World, tool_world: World, position_xyz: tuple[float, float, float]
) -> Body:
    """
    Merge a parsed tool into the world, resting on the kitchen counter.

    :param world: The world to merge into.
    :param tool_world: The parsed tool.
    :param position_xyz: Where on the counter the tool rests.
    :return: The tool's root body inside ``world``.
    """
    tool_name = tool_world.root.name.name
    with world.modify_world():
        world.merge_world_at_pose(
            tool_world,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                *position_xyz, reference_frame=world.root
            ),
        )
    return world.get_body_by_name(tool_name)


def build_world() -> tuple[World, PR2, Context, Body, Body, Body, Body, Body]:
    """
    Build the apartment kitchen with bread on a cutting board, a bowl on the counter and
    the knife already held in the right gripper.

    :return: The world, the robot, the plan context, and the bread, bowl, knife, cup and
        whisk bodies.
    """
    world = setup_world(PR2)

    bread_world = parse_object("bread.stl", color=BREAD_COLOR)
    board = add_cutting_board(world, bread_world, BREAD_POSITION_XYZ)
    board_top_x, board_top_y, board_top_z = BREAD_POSITION_XYZ
    with world.modify_world():
        world.merge_world_at_pose(
            bread_world,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                board_top_x,
                board_top_y,
                board_top_z + 0.02,
                reference_frame=world.root,
            ),
        )
        world.merge_world_at_pose(
            parse_object("bowl.stl", color=BOWL_COLOR),
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                *BOWL_POSITION_XYZ, reference_frame=world.root
            ),
        )
    logger.debug("cutting board added as %s", board.name)
    start_visualization(world)

    robot = PR2.from_world(world)
    context = Context(world=world, robot=robot, _debug=False, ros_node=None)
    context.evaluate_conditions = False

    knife = attach_tool(
        world, robot, Arms.RIGHT, parse_object("big-knife.stl"), CUT_MOUNT
    )
    cup = merge_on_counter(
        world, parse_object("jeroen_cup.stl", color=CUP_COLOR), CUP_PARKING_XYZ
    )
    whisk = merge_on_counter(world, parse_object("whisk.stl"), WHISK_PARKING_XYZ)
    bread = world.get_body_by_name("bread.stl")
    bowl = world.get_body_by_name("bowl.stl")
    with world.modify_world():
        world.add_semantic_annotations(
            [
                Bread(root=bread),
                Bowl(root=bowl),
                CuttingKnife(root=knife),
                PouringCup(root=cup),
                Whisk(root=whisk),
            ]
        )
    return world, robot, context, bread, bowl, knife, cup, whisk


# %% the cooking task


def main() -> None:
    """
    Run cut, pour and mix as three plans in one continuous session, swapping the tool in
    the right gripper between them.
    """
    world, robot, context, bread, bowl, knife, cup, whisk = build_world()
    counter_pose = Pose.from_xyz_rpy(*BASE_POSITION_XYZ, reference_frame=world.root)

    separation_and_division = sequential(
        [
            SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            NavigateAction(counter_pose),
            CuttingAction(
                object_to_cut=bread,
                arm=Arms.RIGHT,
                tool=CuttingKnife(root=knife),
                technique=CuttingTechnique.SLICE,
                number_of_cuts_on_local_x_axis=CUTS_ALONG_LOCAL_X,
                slice_thickness=SLICE_THICKNESS,
            ),
            ParkArmsAction(Arms.RIGHT),
        ],
        context=context,
    ).plan
    with simulated_robot:
        logger.info("cutting the bread")
        separation_and_division.perform()

    park_tool(world, knife, KNIFE_PARKING_XYZ)
    take_tool(world, robot, Arms.RIGHT, cup, POUR_MOUNT)

    material_transfer = sequential(
        [
            PouringAction(
                target_container=bowl,
                source_container=PouringCup(root=cup),
                arm=Arms.RIGHT,
            ),
            ParkArmsAction(Arms.RIGHT),
        ],
        context=context,
    ).plan
    with simulated_robot:
        logger.info("pouring into the bowl")
        material_transfer.perform()

    park_tool(world, cup, CUP_PARKING_XYZ)
    take_tool(world, robot, Arms.RIGHT, whisk, MIX_MOUNT)

    aggregation_and_mixing = sequential(
        [
            MixingAction(container=bowl, arm=Arms.RIGHT, tool=Whisk(root=whisk)),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    ).plan
    with simulated_robot:
        logger.info("mixing the bowl")
        aggregation_and_mixing.perform()


if __name__ == "__main__":
    main()
