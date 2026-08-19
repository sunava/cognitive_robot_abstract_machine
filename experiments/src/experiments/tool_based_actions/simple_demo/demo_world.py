"""
Shared world and tool setup for the tool-based action demos (cutting, pouring, mixing,
wiping).
"""

import math
import os

from typing_extensions import Optional

import coraplex
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager

OBJECTS_DIRECTORY = os.path.join(
    os.path.dirname(coraplex.__file__), "..", "..", "resources", "objects"
)

TARGET_POSITION_XYZ = (2.4, 2.2, 1.0)
"""
Position of the manipulated object on the apartment kitchen counter.
"""

BASE_POSITION_XYZ = (1.85, 2.2, 0.0)
"""
Base position in front of the kitchen counter, facing the target.
"""

CUT_MOUNT = {"z": 0.08, "pitch": -math.pi / 2}
"""
Knife mount transform on the right gripper's tool frame.
"""

MIX_MOUNT = {"z": -0.08, "pitch": math.pi / 2}
"""
Whisk mount transform on the right gripper's tool frame.
"""

POUR_MOUNT = {"z": -0.08}
"""
Cup mount transform on the right gripper's tool frame.
"""

WIPE_MOUNT = {"pitch": math.pi / 2}
"""
Sponge mount transform on the right gripper's tool frame.
"""

BREAD_COLOR = Color(0.55, 0.35, 0.17)
"""
Brownish color for the bread.
"""

CUP_COLOR = Color(0.3, 0.5, 0.9)
"""
Blueish color for the cup.
"""

BOWL_COLOR = Color(0.8, 0.2, 0.2)
"""
Redish color for the bowl.
"""

SPONGE_COLOR = Color(0.95, 0.85, 0.3)
"""
Yellowish color for the sponge.
"""

CUTTING_BOARD_COLOR = Color(0.82, 0.65, 0.42)
"""
Light wood color for the cutting board.
"""

CUTTING_BOARD_THICKNESS = 0.02
"""
Thickness of the primitive-box cutting board placed under a cut object.
"""

CUTTING_BOARD_MARGIN = 0.08
"""
How far the cutting board's footprint extends past the cut object's footprint on each
side.
"""


# %% robot placement


def standing_pose_at(world: World, robot: AbstractRobot) -> Pose:
    """
    :param world: The apartment world the robot stands in.
    :param robot: The robot to place.
    :return: The pose in front of the kitchen counter, at the height the robot's root
        already stands at, so a robot whose root is its pelvis keeps its feet on the
        floor instead of sinking into it.
    """
    root_height = float(world.compute_forward_kinematics(world.root, robot.root).z)
    return Pose.from_xyz_rpy(
        *BASE_POSITION_XYZ[:2], root_height, reference_frame=world.root
    )


# %% demo objects


def parse_object(stl_file_name: str, color: Optional[Color] = None) -> World:
    """
    :param stl_file_name: Name of the mesh file in the demo resources.
    :param color: Color to dye the mesh's visual shapes with. Keeps the mesh's own
        appearance if None.
    :return: A world containing the mesh from the demo resources.
    """
    object_world = STLParser(os.path.join(OBJECTS_DIRECTORY, stl_file_name)).parse()
    if color is not None:
        object_world.root.visual.dye_shapes(color)
    return object_world


def add_cutting_board(
    world: World, cut_object: World, cut_object_position_xyz: tuple
) -> Body:
    """
    Add a flat board to ``world``, directly under where ``cut_object`` will be placed.

    The board's footprint is sized to ``cut_object``'s own footprint plus a margin on
    each side. ``cut_object_position_xyz`` is the placement ``cut_object`` would need to
    rest directly on the counter without a board — its origin is not generally its
    bottom face, so the board's bottom is derived from ``cut_object``'s own local
    geometry rather than from ``cut_object_position_xyz`` directly, putting the board's
    bottom exactly where ``cut_object``'s bottom would otherwise have been. Its top is
    one board thickness higher. Callers should therefore merge ``cut_object`` one board
    thickness higher than ``cut_object_position_xyz``, so it rests on top of the board
    instead of inside it.

    :param world: The world to add the board to.
    :param cut_object: The not-yet-merged world containing the object to be cut, used
        to size the board's footprint and to locate its own bottom face.
    :param cut_object_position_xyz: The x, y, z position ``cut_object``'s origin would
        occupy resting directly on the counter, without a board.
    :return: The cutting board body inside ``world``.
    """
    footprint = cut_object.root.collision.as_bounding_box_collection_in_frame(
        cut_object.root
    ).bounding_box()
    board_width = footprint.max_x - footprint.min_x + 2 * CUTTING_BOARD_MARGIN
    board_depth = footprint.max_y - footprint.min_y + 2 * CUTTING_BOARD_MARGIN
    board_scale = Scale(board_width, board_depth, CUTTING_BOARD_THICKNESS)
    x, y, object_origin_z = cut_object_position_xyz
    counter_z = object_origin_z + footprint.min_z
    board_body = Body(
        name=PrefixedName("cutting_board"),
        collision=ShapeCollection([Box(scale=board_scale)]),
        visual=ShapeCollection([Box(scale=board_scale, color=CUTTING_BOARD_COLOR)]),
    )
    connection = FixedConnection(
        parent=world.root,
        child=board_body,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            x,
            y,
            counter_z + CUTTING_BOARD_THICKNESS / 2,
            reference_frame=world.root,
        ),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(board_body)
        world.add_connection(connection)
    return board_body


def attach_sponge(world: World, robot: AbstractRobot, arm: Arms) -> Body:
    """
    Attach a primitive box sponge to the arm's tool frame.

    :return: The sponge body inside ``world``.
    """
    sponge_body = Body(
        name=PrefixedName("sponge"),
        collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
        visual=ShapeCollection(
            [Box(scale=Scale(0.05, 0.05, 0.05), color=SPONGE_COLOR)]
        ),
    )
    tool_frame = ViewManager.get_end_effector_view(arm, robot).tool_frame
    connection = FixedConnection(
        parent=tool_frame,
        child=sponge_body,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=tool_frame, **WIPE_MOUNT
        ),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(sponge_body)
        world.add_connection(connection)
    return sponge_body
