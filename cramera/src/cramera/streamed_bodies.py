"""
Which of a world's bodies the viewer streams instead of loading once.

A scene bundle is a URDF, so it can only hold a body whose place in the world a URDF
joint can express. Everything else has to reach the viewer as an object whose pose is
published every tick.

Deliberately free of :mod:`coraplex`, because :mod:`cramera.live.bridge` reads this and
has to stay importable outside a demo environment.
"""

from __future__ import annotations

from semantic_digital_twin.world_description.connections import (
    Connection,
    Connection6DoF,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from typing_extensions import Tuple, Type

from cramera.mesh_format import MeshFormat

URDF_JOINT_CONNECTIONS: Tuple[Type[Connection], ...] = (
    FixedConnection,
    RevoluteConnection,
    PrismaticConnection,
    Connection6DoF,
    OmniDrive,
)
"""
The connections a URDF joint can express.

:class:`~cramera.onboard.world_to_urdf.UrdfDocument` decides which joint each of them
becomes; what is listed here is only whether one can be written at all.
"""


def is_urdf_joint(connection: Connection) -> bool:
    """
    Whether a URDF joint can express this connection.

    :param connection: The connection to check.
    """
    return isinstance(connection, URDF_JOINT_CONNECTIONS)


def hangs_off_a_non_urdf_joint(entity: KinematicStructureEntity) -> bool:
    """
    Whether anything between the world root and this entity is not a URDF joint.

    A continuum section is the case this exists for: its curvature is no joint a URDF
    can name, so the bundle would have to nail every body below it down at the pose it
    held when the bundle was written, and the robot would stand frozen while it bends.

    :param entity: The body or region to check.
    """
    while (connection := entity.parent_connection) is not None:
        if not is_urdf_joint(connection):
            return True
        entity = connection.parent
    return False


def names_a_mesh_file(body: Body) -> bool:
    """
    Whether a body is named after a mesh file, which is how a demo's own objects are
    told apart from the scene they stand in.

    :param body: The body to check.
    """
    return MeshFormat.of_path(str(body.name).split("/")[-1]) is not None


def is_streamed(body: Body) -> bool:
    """
    Whether the viewer receives this body's pose every tick instead of loading it with
    the scene.

    True for a demo's own objects, which spawn, move and disappear mid-run, and for
    anything a URDF joint cannot hold in place.

    :param body: The body to check.
    """
    return names_a_mesh_file(body) or hangs_off_a_non_urdf_joint(body)
