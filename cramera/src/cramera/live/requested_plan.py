"""
The plan the Plan Builder asks a running scene to perform.

The builder posts its steps rather than the Python it also generates: the viewer's
bridge listens on every interface, so a scene that ran posted source would run whatever
anyone on the network sent it. Reading a fixed set of steps keeps what a request can ask
for bounded, and keeps the failure — a step naming an arm the robot has not got — at the
door instead of inside a motion.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from typing_extensions import Any, Dict, List, Tuple, Type

from cramera.live.placement_surface import placement_surface_type


class StepType(StrEnum):
    """
    A step the Plan Builder can put in a plan.
    """

    PARK_ARMS = "park_arms"
    MOVE_TORSO = "move_torso"
    NAVIGATE = "navigate"
    TRANSPORT = "transport"


class TargetMode(StrEnum):
    """
    How a transport step says where the object goes.
    """

    POSE = "pose"
    SEMANTIC = "semantic"


class MalformedPlanRequest(Exception):
    """
    Raised when a posted plan cannot be read.
    """


class SurfaceNotInWorld(Exception):
    """
    Raised when the surface a transport places on is not in the running world.
    """


class NoFreeSpotOnSurface(Exception):
    """
    Raised when a surface has no free place left for the object being transported.
    """


# %% reading values off the wire


def _number(parameters: Dict[str, Any], key: str) -> float:
    """
    One finite coordinate of a step.

    :param parameters: The step's parameters.
    :param key: The parameter to read.
    :raises MalformedPlanRequest: If the value is missing or not a finite number.
    """
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedPlanRequest("%r must be a number, got %r" % (key, value))
    if not math.isfinite(value):
        raise MalformedPlanRequest("%r must be finite, got %r" % (key, value))
    return float(value)


def _member(enumeration: Type, parameters: Dict[str, Any], key: str) -> Any:
    """
    The enum member a step names.

    :param enumeration: The enumeration the name has to belong to.
    :param parameters: The step's parameters.
    :param key: The parameter holding the member's name.
    :raises MalformedPlanRequest: If the name is not one of the enumeration's members.
    """
    name = parameters.get(key)
    if name not in enumeration.__members__:
        raise MalformedPlanRequest(
            "%r must be one of %s, got %r" % (key, list(enumeration.__members__), name)
        )
    return enumeration[name]


@dataclass(frozen=True)
class LevelPose:
    """
    A pose given as a place and a heading, with no roll or pitch — how the builder's
    scene lets one be set.
    """

    x: float
    """
    Position along the world's x axis, in metres.
    """

    y: float
    """
    Position along the world's y axis, in metres.
    """

    z: float
    """
    Height above the world's origin, in metres.
    """

    yaw: float
    """
    Heading about the vertical axis, in radians.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> LevelPose:
        """
        Read a pose off a step's parameters.

        :param parameters: The step's parameters.
        :raises MalformedPlanRequest: If a coordinate is missing or unusable.
        """
        return cls(
            x=_number(parameters, "x"),
            y=_number(parameters, "y"),
            z=_number(parameters, "z"),
            yaw=_number(parameters, "yaw"),
        )

    def pose(self, world: World) -> Pose:
        """
        The pose itself, in the world's frame.

        :param world: The world the pose is expressed in.
        """
        return Pose.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw, reference_frame=world.root
        )


# %% where a transport puts the object


class TransportTarget(ABC):
    """
    Where a transport step puts the object it carries.
    """

    @abstractmethod
    def pose(self, world: World, transported: Body) -> Pose:
        """
        The pose to place the object at.

        :param world: The running world.
        :param transported: The body being carried, which a surface sizes its free spot
            for.
        """


@dataclass(frozen=True)
class PoseTarget(TransportTarget):
    """
    A place the user pointed at in the scene.
    """

    where: LevelPose
    """
    The pose the object is put down at.
    """

    def pose(self, world: World, transported: Body) -> Pose:
        return self.where.pose(world)


@dataclass(frozen=True)
class SurfaceTarget(TransportTarget):
    """
    A surface to put the object on, with the free spot picked when the plan runs.
    """

    surface_type: Type[SemanticAnnotation]
    """
    The kind of surface to place on.
    """

    surface_name: str
    """
    The name of the surface's root body, or empty to take the first one of its kind.
    """

    def surface(self, world: World) -> SemanticAnnotation:
        """
        The surface in the running world this target means.

        :param world: The running world.
        :raises SurfaceNotInWorld: If the world holds no such surface.
        """
        found: List[SemanticAnnotation] = world.get_semantic_annotations_by_type(
            self.surface_type
        )
        if self.surface_name:
            found = [
                surface
                for surface in found
                if str(surface.root.name) == self.surface_name
                or str(surface.root.name).split("/")[-1] == self.surface_name
            ]
        if not found:
            raise SurfaceNotInWorld(
                "no %s named %r in this world"
                % (self.surface_type.__name__, self.surface_name or "<any>")
            )
        return found[0]

    def pose(self, world: World, transported: Body) -> Pose:
        surface = self.surface(world)
        # The sampler sizes the free spot from an annotation's root body; a mesh the
        # builder placed carries no annotation, so the spot is sized generically.
        points = surface.sample_points_from_surface()
        if not points:
            raise NoFreeSpotOnSurface(
                "no free place for %s on %s — the surface is full or too small"
                % (transported.name, self.surface_type.__name__)
            )
        return Pose(points[0], reference_frame=points[0].reference_frame)


# %% the steps themselves


@dataclass(frozen=True)
class PlanStep(ABC):
    """
    One step of a plan the builder asks for.
    """

    @classmethod
    @abstractmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> PlanStep:
        """
        Read the step off its posted parameters.

        :param parameters: The step's parameters.
        :raises MalformedPlanRequest: If a parameter is missing or unusable.
        """

    @abstractmethod
    def action(self, world: World) -> ActionDescription:
        """
        The coraplex action carrying this step out.

        :param world: The running world the step's bodies and poses are resolved in.
        """


@dataclass(frozen=True)
class ParkArms(PlanStep):
    """
    Bring an arm back to its parked pose.
    """

    arm: Arms
    """
    The arm to park.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> ParkArms:
        return cls(arm=_member(Arms, parameters, "arm"))

    def action(self, world: World) -> ActionDescription:
        return ParkArmsAction(self.arm)


@dataclass(frozen=True)
class MoveTorso(PlanStep):
    """
    Raise or lower the torso.
    """

    torso_state: TorsoState
    """
    The height the torso is moved to.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> MoveTorso:
        return cls(torso_state=_member(TorsoState, parameters, "torso"))

    def action(self, world: World) -> ActionDescription:
        return MoveTorsoAction(self.torso_state)


@dataclass(frozen=True)
class Navigate(PlanStep):
    """
    Drive the robot's base somewhere.
    """

    target: LevelPose
    """
    Where the robot drives to, and which way it ends up facing.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Navigate:
        return cls(target=LevelPose.from_parameters(parameters))

    def action(self, world: World) -> ActionDescription:
        return NavigateAction(self.target.pose(world))


@dataclass(frozen=True)
class Transport(PlanStep):
    """
    Carry an object somewhere else.
    """

    object_name: str
    """
    The name of the body being carried, which is its mesh file name.
    """

    arm: Arms
    """
    The arm that picks the object up and puts it down.
    """

    target: TransportTarget
    """
    Where the object ends up.
    """

    look_at_operation_site: bool
    """
    Whether the robot looks at the object before picking it up and at the target before
    placing it.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Transport:
        object_name = parameters.get("object")
        if not isinstance(object_name, str) or not object_name:
            raise MalformedPlanRequest("a transport step must name the object to carry")
        return cls(
            object_name=object_name,
            arm=_member(Arms, parameters, "arm"),
            target=cls._target(parameters),
            look_at_operation_site=bool(
                parameters.get("look_at_operation_site", False)
            ),
        )

    @staticmethod
    def _target(parameters: Dict[str, Any]) -> TransportTarget:
        """
        Where this transport puts the object, in whichever way it was given.

        :param parameters: The step's parameters.
        :raises MalformedPlanRequest: If the target mode is not one the builder offers.
        :raises cramera.live.placement_surface.UnknownPlacementSurface: If a named
            surface is not one a plan can place on.
        """
        mode = parameters.get("targetMode", TargetMode.POSE.value)
        if mode == TargetMode.POSE:
            return PoseTarget(where=LevelPose.from_parameters(parameters))
        if mode == TargetMode.SEMANTIC:
            return SurfaceTarget(
                surface_type=placement_surface_type(
                    str(parameters.get("surfaceType", ""))
                ),
                surface_name=str(parameters.get("surfaceName", "")),
            )
        raise MalformedPlanRequest(
            "'targetMode' must be one of %s, got %r"
            % ([mode.value for mode in TargetMode], mode)
        )

    def action(self, world: World) -> ActionDescription:
        transported = world.get_body_by_name(self.object_name)
        return TransportAction(
            transported,
            self.target.pose(world, transported),
            self.arm,
            look_at_operation_site=self.look_at_operation_site,
        )


STEP_TYPES: Dict[StepType, Type[PlanStep]] = {
    StepType.PARK_ARMS: ParkArms,
    StepType.MOVE_TORSO: MoveTorso,
    StepType.NAVIGATE: Navigate,
    StepType.TRANSPORT: Transport,
}
"""
The step each type in a posted plan is read as.
"""


@dataclass(frozen=True)
class RequestedPlan:
    """
    A plan the Plan Builder asked a running scene to perform.
    """

    steps: Tuple[PlanStep, ...]
    """
    The steps to perform, in order.
    """

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> RequestedPlan:
        """
        Build a request from a decoded ``POST /run`` body.

        :param payload: The decoded JSON body.
        :raises MalformedPlanRequest: If the plan or one of its steps is unusable.
        """
        posted = payload.get("steps")
        if not isinstance(posted, list) or not posted:
            raise MalformedPlanRequest("'steps' must be a non-empty list")
        return cls(steps=tuple(cls._step(entry) for entry in posted))

    @staticmethod
    def _step(entry: Any) -> PlanStep:
        """
        One step of a posted plan.

        :param entry: The step's entry in the payload.
        :raises MalformedPlanRequest: If the entry names no known step type.
        """
        if not isinstance(entry, dict):
            raise MalformedPlanRequest("every step must be an object")
        named = entry.get("type")
        if named not in STEP_TYPES:
            raise MalformedPlanRequest(
                "'type' must be one of %s, got %r"
                % ([step.value for step in StepType], named)
            )
        parameters = entry.get("params") or {}
        if not isinstance(parameters, dict):
            raise MalformedPlanRequest("'params' must be an object")
        return STEP_TYPES[StepType(named)].from_parameters(parameters)

    def plan(self, context: Context) -> PlanNode:
        """
        The coraplex plan performing these steps against a running scene.

        :param context: The running scene's context, whose world the steps resolve in.
        """
        actions = [step.action(context.world) for step in self.steps]
        return sequential(actions, context=context).plan
