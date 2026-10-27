"""
A coraplex demo for the onboarding tests: it shows a world through coraplex's world
visualization and slides a box until it touches a table.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from coraplex.visualization import WorldVisualization
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

TABLE_HEIGHT = 0.75
"""
The height of the table's top, in metres.
"""

BOX_SIZE = 0.1
"""
The edge length of the box, in metres.
"""

BOX_HEIGHTS = (1.2, 1.0, 0.9, TABLE_HEIGHT + BOX_SIZE / 2)
"""
The heights the box is moved to, the last one resting on the table.
"""

FAILURE_ARGUMENT = "--fail"
"""
Makes the demo exit with :data:`FAILURE_STATUS` after its visualization started.
"""

FAILURE_STATUS = 3
"""
The exit status of a demo asked to fail.
"""

# %% world


@dataclass
class SlidingBoxWorld:
    """
    A table fixed to the root and a free box above it.
    """

    world: World
    """
    The world.
    """

    table: Body
    """
    The table.
    """

    box: Body
    """
    The box.
    """

    box_connection: Connection6DoF
    """
    The connection holding the box.
    """

    @classmethod
    def build(cls) -> SlidingBoxWorld:
        """
        :return: A new world with the box above the table.
        """
        world = World()
        root = Body(name=PrefixedName("map"))
        table = Body(
            name=PrefixedName("table", "demo"),
            visual=ShapeCollection(
                shapes=[Box(scale=Scale(1.0, 0.6, 0.05), color=Color(0.6, 0.4, 0.2))]
            ),
            collision=ShapeCollection(shapes=[Box(scale=Scale(1.0, 0.6, 0.05))]),
        )
        box = Body(
            name=PrefixedName("box", "demo"),
            visual=ShapeCollection(
                shapes=[Box(scale=Scale(BOX_SIZE, BOX_SIZE, BOX_SIZE))]
            ),
            collision=ShapeCollection(
                shapes=[Box(scale=Scale(BOX_SIZE, BOX_SIZE, BOX_SIZE))]
            ),
        )
        with world.modify_world():
            world.add_kinematic_structure_entity(root)
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=table,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        0.0, 0.0, TABLE_HEIGHT - 0.025
                    ),
                )
            )
            box_connection = Connection6DoF.create_with_dofs(
                parent=root, child=box, world=world
            )
            world.add_connection(box_connection)
        built = cls(world=world, table=table, box=box, box_connection=box_connection)
        built.move_box(BOX_HEIGHTS[0])
        return built

    def move_box(self, height: float) -> None:
        """
        :param height: The height the box's centre is moved to, above the table's centre.
        """
        self.box_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            0.0, 0.0, height, reference_frame=self.world.root
        )


# %% demo


def main() -> None:
    """
    Show the world with the backend the environment selects and lower the box onto the
    table.

    :raises SystemExit: With :data:`FAILURE_STATUS` when the demo is asked to fail.
    """
    scene = SlidingBoxWorld.build()
    visualization = WorldVisualization.from_environment(scene.world).start()
    if FAILURE_ARGUMENT in sys.argv[1:]:
        sys.exit(FAILURE_STATUS)
    for height in BOX_HEIGHTS:
        scene.move_box(height)
    visualization.stop()


if __name__ == "__main__":
    main()
