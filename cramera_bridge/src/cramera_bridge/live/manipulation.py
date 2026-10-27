"""
Changes a person makes to the live world through the viewer: moving free objects and
setting joints.

Requests are checked on the HTTP thread and applied on the thread that owns the world:
the simulation thread while a plan moves the robot, else the requesting thread, since
nothing else writes the world then.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING, Dict, List

from cramera_bridge.exceptions import ObjectNotMovable, UnknownJoint, UnknownObject
from cramera_bridge.geometry import NumericPose
from cramera_bridge.names import ANNOTATION_NAMES
from cramera_bridge.live.transforms import TransformTracker
from cramera_bridge.live.world_model import WorldModelView
from cramera_protocol.geometry import Pose
from cramera_protocol.live_messages import (
    JointRequest,
    LiveState,
    MoveRequest,
    ObjectPoses,
    SurfaceDescription,
    Surfaces,
)
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

# %% one move


@dataclass(frozen=True)
class ObjectMove:
    """
    Writes a requested world pose into the connection holding an object.
    """

    request: MoveRequest
    """
    The request.
    """

    body: Body
    """
    The object.
    """

    def target(self, world: World) -> NumericPose:
        """
        :param world: The world.
        :return: The requested world pose; a request without an orientation keeps the
            object's current one.
        """
        if self.request.quaternion is not None:
            orientation = tuple(self.request.quaternion.to_json())
        else:
            orientation = NumericPose.of_entity(world, self.body).quaternion
        return NumericPose(
            position=tuple(self.request.position.to_json()), quaternion=orientation
        )

    def apply(self, world: World) -> None:
        """
        Set the object's connection so the object stands at the requested world pose.

        :param world: The world.
        """
        connection = self.body.parent_connection
        target = self.target(world)
        world_T_object = numpy.eye(4)
        world_T_object[:3, :3] = Rotation.from_quat(target.quaternion).as_matrix()
        world_T_object[:3, 3] = target.position
        world_T_parent = world.compute_forward_kinematics_np(
            world.root, connection.parent
        )
        local = NumericPose.of_matrix(numpy.linalg.inv(world_T_parent) @ world_T_object)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            *local.coordinates(),
            reference_frame=connection.parent,
            child_frame=self.body,
        )


# %% queue


@dataclass
class WorldEdits:
    """
    The moves and joint values waiting to be written into the world.
    """

    moves: List[MoveRequest] = field(default_factory=list)
    """
    Object moves not yet applied.
    """

    joints: List[JointRequest] = field(default_factory=list)
    """
    Joint values not yet applied.
    """

    targets: Dict[str, Pose] = field(default_factory=dict)
    """
    The pose each object was last moved to, applied or not, by object key.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every field.
    """

    def queue_move(self, request: MoveRequest, model: WorldModelView) -> None:
        """
        :param request: A move.
        :param model: The world's current model.
        :raises UnknownObject: When no free object has the key.
        :raises ObjectNotMovable: When the object is not held by six degrees of
            freedom.
        """
        body = self.movable(request.object_key, model)
        target = ObjectMove(request, body).target(model.world).published()
        with self.lock:
            self.moves.append(request)
            self.targets[request.object_key] = target

    @staticmethod
    def movable(key: str, model: WorldModelView) -> Body:
        """
        :param key: An object key.
        :param model: The world's current model.
        :return: The object, held by six degrees of freedom.
        :raises UnknownObject: When no free object has the key.
        :raises ObjectNotMovable: When the object is held otherwise.
        """
        body = model.object_with_key(key)
        if body is None:
            raise UnknownObject(key)
        if not isinstance(body.parent_connection, Connection6DoF):
            raise ObjectNotMovable(key, type(body.parent_connection).__name__)
        return body

    def queue_joint(self, request: JointRequest, model: WorldModelView) -> None:
        """
        :param request: A joint value.
        :param model: The world's current model.
        :raises UnknownJoint: When the world has no such movable joint.
        """
        if model.joint_named(request.joint) is None:
            raise UnknownJoint(request.joint)
        with self.lock:
            self.joints.append(request)

    def apply(self, model: WorldModelView, transforms: TransformTracker) -> bool:
        """
        Write every waiting change into the world; changes whose object or joint left
        the world meanwhile are dropped.

        :param model: The world's current model.
        :param transforms: Notes which connections a person changed.
        :return: Whether anything was written.
        """
        with self.lock:
            moves, self.moves = self.moves, []
            joints, self.joints = self.joints, []
        written = False
        for move in moves:
            body = model.object_with_key(move.object_key)
            if body is None or not isinstance(body.parent_connection, Connection6DoF):
                continue
            ObjectMove(move, body).apply(model.world)
            transforms.note_viewer_write(str(body.parent_connection.name))
            written = True
        for request in joints:
            joint = model.joint_named(request.joint)
            if joint is None:
                continue
            limits = joint.dof.limits
            position = request.position
            if limits.lower.position is not None:
                position = max(limits.lower.position, position)
            if limits.upper.position is not None:
                position = min(limits.upper.position, position)
            joint.position = position
            transforms.note_viewer_write(str(joint.name))
            written = True
        return written

    def captured(self, state: LiveState) -> ObjectPoses:
        """
        :param state: The newest state snapshot.
        :return: Every object's pose, where a move a person made counts even before the
            world applied it.
        """
        with self.lock:
            targets = dict(self.targets)
        return ObjectPoses(objects={**state.objects, **targets})


# %% surfaces


@dataclass(frozen=True)
class WorldSurfaces:
    """
    The annotated surfaces of a world objects can be placed on or in.
    """

    world: World
    """
    The world.
    """

    def surfaces(self) -> Surfaces:
        """
        :return: Every annotation with a supporting surface, containers included.
        """
        return Surfaces(
            surfaces=[
                SurfaceDescription(
                    type_name=ANNOTATION_NAMES.identifier_of(type(annotation)),
                    name=str(annotation.name),
                )
                for annotation in self.world.get_semantic_annotations_by_type(
                    HasSupportingSurface
                )
            ]
        )
