"""
Native poses, scales and colours turned into the plain numbers the protocol writes.

Conversion happens here, at the boundary: everything before it works with the native
CRAM types, everything after it with protocol values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING, Tuple

from cramera_protocol.geometry import Pose, Vector3
from cramera_protocol.live_messages import Quaternion

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.geometry import Color, Scale
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

POSE_PRECISION = 5
"""
Decimal places of a published coordinate; finer differences are numerical noise.
"""

SIZE_PRECISION = 4
"""
Decimal places of a published size or radius.
"""

COLOR_CHANNEL_MAXIMUM = 255
"""
The largest value of one channel of a CSS hexadecimal colour.
"""

# %% poses


@dataclass(frozen=True)
class NumericPose:
    """
    A pose copied into plain numbers, so later changes of the world do not reach it.
    """

    position: Tuple[float, float, float]
    """
    The translation in metres.
    """

    quaternion: Tuple[float, float, float, float]
    """
    The orientation in ``x, y, z, w`` order.
    """

    @classmethod
    def of_matrix(cls, matrix: numpy.ndarray) -> NumericPose:
        """
        :param matrix: A homogeneous transformation as a 4x4 array.
        :return: Its translation and orientation.
        """
        return cls(
            position=tuple(float(value) for value in matrix[:3, 3]),
            quaternion=tuple(
                float(value) for value in Rotation.from_matrix(matrix[:3, :3]).as_quat()
            ),
        )

    @classmethod
    def of_entity(cls, world: World, entity: KinematicStructureEntity) -> NumericPose:
        """
        Read an entity's world pose numerically, without building an expression.

        :param world: The world holding the entity.
        :param entity: A body or region of the world.
        :return: Its pose relative to the world's root.
        """
        return cls.of_matrix(world.compute_forward_kinematics_np(world.root, entity))

    def published(self, precision: int = POSE_PRECISION) -> Pose:
        """
        :param precision: Decimal places kept in each coordinate.
        :return: The pose as the protocol writes it.
        """
        return Pose(*(round(value, precision) for value in self.coordinates()))

    def published_position(self, precision: int = POSE_PRECISION) -> Vector3:
        """
        :param precision: Decimal places kept in each coordinate.
        :return: The translation as the protocol writes it.
        """
        return Vector3(*(round(value, precision) for value in self.position))

    def published_orientation(self, precision: int = POSE_PRECISION) -> Quaternion:
        """
        :param precision: Decimal places kept in each component.
        :return: The orientation as the protocol writes it.
        """
        return Quaternion(*(round(value, precision) for value in self.quaternion))

    def coordinates(self) -> Tuple[float, ...]:
        """
        :return: The position followed by the quaternion.
        """
        return (*self.position, *self.quaternion)

    @property
    def label(self) -> str:
        """
        The pose as a short text for answers.
        """
        return "(%.2f, %.2f, %.2f) q(%.2f, %.2f, %.2f, %.2f)" % self.coordinates()


# %% sizes


@dataclass(frozen=True)
class PublishedScale:
    """
    A native scale as the protocol writes it.
    """

    scale: Scale
    """
    The native scale.
    """

    def published(self, precision: int = SIZE_PRECISION) -> Vector3:
        """
        :param precision: Decimal places kept on each axis.
        :return: The scale's axes.
        """
        return Vector3(
            round(float(self.scale.x), precision),
            round(float(self.scale.y), precision),
            round(float(self.scale.z), precision),
        )


# %% colours


@dataclass(frozen=True)
class PublishedColor:
    """
    A native colour as the protocol writes it.
    """

    color: Color
    """
    The native colour, whose channels lie between 0 and 1.
    """

    def hexadecimal(self) -> str:
        """
        :return: The colour as a CSS hexadecimal colour without its opacity.
        """
        channels = (self.color.R, self.color.G, self.color.B)
        return "#" + "".join(
            "%02x" % round(self.clamped(channel) * COLOR_CHANNEL_MAXIMUM)
            for channel in channels
        )

    def opacity(self) -> float:
        """
        :return: The colour's opacity between 0 and 1.
        """
        return self.clamped(self.color.A)

    @staticmethod
    def clamped(channel: float) -> float:
        """
        :param channel: One channel of a colour.
        :return: The channel held between 0 and 1.
        """
        return min(max(float(channel), 0.0), 1.0)
