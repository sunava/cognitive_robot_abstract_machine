"""
The debug markers of the CRAM processes, kept the way RViz keeps them and posed in the
world for the viewer.

Messages are read by their fields only, so this module needs no ROS.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum

import numpy
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from cramera_bridge.geometry import POSE_PRECISION, NumericPose, PublishedColor
from cramera_protocol.geometry import Vector3
from cramera_protocol.live_messages import MarkerDescription, MarkerKind, MarkerOverlay
from semantic_digital_twin.world_description.geometry import Color

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

OPACITY_PRECISION = 3
"""
Decimal places of a published marker opacity.
"""

WORLD_FRAME_NAMES = frozenset({"", "map", "world"})
"""
Frame names that mean the world itself, besides the name of the world's root.
"""


class MarkerAction(IntEnum):
    """
    The actions of a ``visualization_msgs/Marker``.
    """

    ADD = 0
    """
    Adds or replaces the marker.
    """

    DELETE = 2
    """
    Removes the marker.
    """

    DELETE_ALL = 3
    """
    Removes every marker of the topic.
    """


class MarkerType(IntEnum):
    """
    The marker types of ``visualization_msgs/Marker`` the viewer draws; mesh and
    triangle-list markers are not drawn.
    """

    ARROW = 0
    """
    An arrow.
    """

    CUBE = 1
    """
    A box.
    """

    SPHERE = 2
    """
    A sphere.
    """

    CYLINDER = 3
    """
    A cylinder.
    """

    LINE_STRIP = 4
    """
    Connected line segments.
    """

    LINE_LIST = 5
    """
    Separate line segments.
    """

    CUBE_LIST = 6
    """
    One box per point.
    """

    SPHERE_LIST = 7
    """
    One sphere per point.
    """

    POINTS = 8
    """
    One dot per point.
    """

    TEXT = 9
    """
    A text that faces the camera.
    """

    @property
    def kind(self) -> MarkerKind:
        """
        The primitive the viewer draws for this type.
        """
        return MarkerKind[self.name]


# %% one marker


@dataclass(frozen=True)
class MarkerEntry:
    """
    One marker in its own frame, as the last message described it.
    """

    namespace: str
    """
    The marker's namespace.
    """

    identifier: int
    """
    The marker's identity within its namespace.
    """

    kind: MarkerKind
    """
    The primitive it draws.
    """

    frame: str
    """
    The frame its pose is given in.
    """

    pose: Tuple[float, ...]
    """
    Its pose in that frame as position and quaternion.
    """

    scale: Vector3
    """
    Its scale.
    """

    color: str
    """
    Its colour as a CSS hexadecimal colour.
    """

    opacity: float
    """
    Its opacity.
    """

    points: Tuple[Vector3, ...] = ()
    """
    Its points relative to its pose.
    """

    text: str = ""
    """
    Its text.
    """

    @classmethod
    def of_message(cls, marker: Any) -> Optional[MarkerEntry]:
        """
        :param marker: A ``visualization_msgs/Marker`` message.
        :return: The marker, or None for a type the viewer does not draw.
        """
        marker_type = int(marker.type)
        if marker_type not in {member.value for member in MarkerType}:
            return None
        position, orientation = marker.pose.position, marker.pose.orientation
        color = ColorChannels(marker.color)
        return cls(
            namespace=str(marker.ns),
            identifier=int(marker.id),
            kind=MarkerType(marker_type).kind,
            frame=str(marker.header.frame_id),
            pose=(
                float(position.x),
                float(position.y),
                float(position.z),
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
            scale=PointMessage(marker.scale).vector(),
            color=color.hexadecimal(),
            opacity=color.opacity(),
            points=tuple(PointMessage(point).vector() for point in marker.points),
            text=str(marker.text),
        )


@dataclass(frozen=True)
class PointMessage:
    """
    A message with ``x``, ``y`` and ``z`` fields.
    """

    message: Any
    """
    The message.
    """

    def vector(self) -> Vector3:
        """
        :return: The rounded coordinates.
        """
        return Vector3(
            round(float(self.message.x), POSE_PRECISION),
            round(float(self.message.y), POSE_PRECISION),
            round(float(self.message.z), POSE_PRECISION),
        )


@dataclass(frozen=True)
class ColorChannels:
    """
    A ``std_msgs/ColorRGBA`` message.
    """

    message: Any
    """
    The message.
    """

    def published(self) -> PublishedColor:
        """
        :return: The colour as the geometry boundary publishes it.
        """
        return PublishedColor(
            Color(self.message.r, self.message.g, self.message.b, self.message.a)
        )

    def hexadecimal(self) -> str:
        """
        :return: The colour as a CSS hexadecimal colour.
        """
        return self.published().hexadecimal()

    def opacity(self) -> float:
        """
        :return: The colour's opacity.
        """
        return round(self.published().opacity(), OPACITY_PRECISION)


# %% one topic


@dataclass
class MarkerStore:
    """
    The markers of one topic, kept the way RViz keeps them.
    """

    entries: Dict[Tuple[str, int], MarkerEntry] = field(default_factory=dict)
    """
    The markers by namespace and identity.
    """

    revision: int = 0
    """
    Grows whenever the markers change.
    """

    def observe(self, markers: List[Any]) -> bool:
        """
        Apply one ``MarkerArray``'s markers.

        :param markers: The markers.
        :return: Whether the markers changed.
        """
        changed = False
        for marker in markers:
            changed = self.apply(marker) or changed
        if changed:
            self.revision += 1
        return changed

    def apply(self, marker: Any) -> bool:
        """
        :param marker: One marker message.
        :return: Whether it changed the markers.
        """
        action = int(marker.action)
        key = (str(marker.ns), int(marker.id))
        if action == MarkerAction.DELETE_ALL:
            changed = bool(self.entries)
            self.entries.clear()
            return changed
        if action == MarkerAction.DELETE:
            return self.entries.pop(key, None) is not None
        if action != MarkerAction.ADD:
            return False
        entry = MarkerEntry.of_message(marker)
        if entry is None or self.entries.get(key) == entry:
            return False
        self.entries[key] = entry
        return True


# %% every topic


@dataclass
class MarkerBoard:
    """
    The markers of every watched topic; a ROS thread delivers messages, the simulation
    thread poses them in the world, HTTP threads read the overlay.
    """

    stores: Dict[str, MarkerStore] = field(default_factory=dict)
    """
    The markers per topic.
    """

    overlay: MarkerOverlay = field(default_factory=MarkerOverlay)
    """
    The newest overlay.
    """

    published_revisions: Optional[Tuple[Tuple[str, int], ...]] = None
    """
    The topic revisions :attr:`overlay` was built from.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every field.
    """

    def observe(self, topic: str, markers: List[Any]) -> None:
        """
        :param topic: The topic a ``MarkerArray`` arrived on.
        :param markers: Its markers.
        """
        with self.lock:
            self.stores.setdefault(topic, MarkerStore()).observe(markers)

    def drop(self, topic: str) -> None:
        """
        Forget a topic's markers, as removing an RViz display does.

        :param topic: The topic no longer watched.
        """
        with self.lock:
            self.stores.pop(topic, None)

    def revisions(self) -> Tuple[Tuple[str, int], ...]:
        """
        :return: The revision of every topic; the caller holds :attr:`lock`.
        """
        return tuple(
            (topic, self.stores[topic].revision) for topic in sorted(self.stores)
        )

    def refresh(self, world: World) -> bool:
        """
        Pose the markers in the world if any topic changed; markers whose namespace
        names a body or region of the world draw what the scene already shows and are
        left out, as are markers in a frame the world cannot resolve.

        :param world: The world marker frames are resolved in.
        :return: Whether the overlay changed.
        """
        with self.lock:
            revisions = self.revisions()
            if revisions == self.published_revisions:
                return False
            entries = [
                (topic, entry)
                for topic in sorted(self.stores)
                for entry in self.stores[topic].entries.values()
            ]
        frames = MarkerFrames(world)
        world_entities = frames.entity_names()
        described = [
            frames.description(topic, entry)
            for topic, entry in entries
            if entry.namespace not in world_entities
        ]
        markers = [marker for marker in described if marker is not None]
        with self.lock:
            self.overlay = MarkerOverlay(
                version=self.overlay.version + 1, markers=markers
            )
            self.published_revisions = revisions
        return True

    def current(self) -> MarkerOverlay:
        """
        :return: The newest overlay.
        """
        with self.lock:
            return self.overlay

    def version(self) -> int:
        """
        :return: The version of the newest overlay.
        """
        with self.lock:
            return self.overlay.version


@dataclass(frozen=True)
class MarkerFrames:
    """
    Poses markers given in a frame of the world.
    """

    world: World
    """
    The world.
    """

    def entity_names(self) -> frozenset:
        """
        :return: The names of the world's bodies and regions.
        """
        return frozenset(str(body.name) for body in self.world.bodies) | frozenset(
            str(region.name) for region in self.world.regions
        )

    def frame_matrix(self, frame: str) -> Optional[numpy.ndarray]:
        """
        :param frame: A marker's frame.
        :return: The frame's pose in the world, the world itself for the root, or None
            for a frame no body or several bodies are named, where the marker cannot be
            posed.
        """
        root = self.world.root
        if frame in WORLD_FRAME_NAMES or frame in (str(root.name), root.name.name):
            return numpy.eye(4)
        candidates = [
            body
            for body in self.world.bodies
            if frame in (str(body.name), body.name.name)
        ]
        if len(candidates) != 1:
            return None
        return self.world.compute_forward_kinematics_np(root, candidates[0])

    def description(
        self, topic: str, entry: MarkerEntry
    ) -> Optional[MarkerDescription]:
        """
        :param topic: The topic the marker arrived on.
        :param entry: The marker.
        :return: The marker posed in the world, or None when its frame cannot be
            resolved.
        """
        frame_matrix = self.frame_matrix(entry.frame)
        if frame_matrix is None:
            return None
        local = numpy.eye(4)
        local[:3, :3] = Rotation.from_quat(entry.pose[3:]).as_matrix()
        local[:3, 3] = entry.pose[:3]
        pose = NumericPose.of_matrix(frame_matrix @ local)
        return MarkerDescription(
            topic=topic,
            namespace=entry.namespace,
            identifier=entry.identifier,
            kind=entry.kind,
            pose=pose.published(),
            scale=entry.scale,
            color=entry.color,
            opacity=entry.opacity,
            points=list(entry.points),
            text=entry.text,
        )
