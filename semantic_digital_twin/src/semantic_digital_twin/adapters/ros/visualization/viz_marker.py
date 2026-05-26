from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union
from uuid import UUID

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import MarkerArray, Marker

from semantic_digital_twin.adapters.ros.msg_converter import SemDTToRos2Converter
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from pycram.robot_plans.actions.composite.thesis_math.world_utils import body_local_aabb

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....world import World


class ShapeSource(Enum):
    """
    Enum to specify which shapes to use for visualization.
    """

    VISUAL_ONLY = "visual_only"
    """
    The shapes to use for visualization are visual shapes only.
    """

    COLLISION_ONLY = "collision_only"
    """
    The shapes to use for visualization are collision shapes only.
    """

    VISUAL_WITH_COLLISION_BACKUP = "visual_with_collision_backup"
    """
    The shapes to use for visualization are visual shapes, but if there are no visual shapes, use collision shapes as a backup.
    """


@dataclass(eq=False)
class VizMarkerPublisher(ModelChangeCallback):
    """
    Publishes the world model as a visualization marker.
    .. warning:: Relies on the tf tree to correctly position the markers.
        Use TFPublisher to publish the tf tree.
    .. warning:: To see something in Rviz you must:
        1. add a MarkerArray plugin,
        2. set the current topic name,
        3. set DurabilityPolicy.TRANSIENT_LOCAL,
        4. make sure that the fixed frame is the tf root.
    """

    node: Node = field(kw_only=True)
    """
    The ROS2 node that will be used to publish the visualization marker.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the Visualization Marker should be published.
    """

    shape_source: ShapeSource = field(
        kw_only=True, default=ShapeSource.VISUAL_WITH_COLLISION_BACKUP
    )
    """
    Which shapes to use for each body
    """

    alpha: float = field(kw_only=True, default=1.0)
    """
    Marker transparency in [0.0, 1.0]. 0.0 is fully transparent.
    """

    highlight_entities: List[Union[Body, SemanticAnnotation]] = field(
        kw_only=True, default_factory=list
    )
    """
    Entities that should be highlighted in RViz.
    """

    highlight_color: Color = field(kw_only=True, default_factory=Color.LIGHT_BLUE)
    """
    Color used to highlight entities.
    """

    highlight_text: Optional[str] = field(kw_only=True, default=None)
    """
    Text that should be displayed next to highlighted entities.
    """

    highlight_outline: bool = field(kw_only=True, default=False)
    """
    If True, highlighted entities will be shown as an outline (bounding box).
    """

    highlight_alpha: float = field(kw_only=True, default=0.5)
    """
    Transparency for highlighted entities.
    """

    markers: MarkerArray = field(init=False, default_factory=MarkerArray)
    """Maker message to be published."""
    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """QoS profile for the publisher."""

    def _format_query_lines(self, text: str, max_chars: int = 40) -> list[str]:
        """Auto-wraps and shortens query text for RViz display."""
        result = []
        for raw_line in text.strip().split("\n"):
            line = raw_line.strip().rstrip(",")
            if not line:
                continue

            # Clean up known verbose patterns
            line = line.replace(" == True", " ✓").replace(" == False", " ✗")

            # If line still exceeds max_chars, try to wrap it at logical points
            if len(line) <= max_chars:
                result.append(line)
            else:
                import textwrap

                # Wrap the line into multiple lines if too long
                wrapped = textwrap.wrap(line, width=max_chars, break_long_words=True)
                result.extend(wrapped)
        return result

    def __post_init__(self):
        super().__post_init__()

        self.pub = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        time.sleep(0.2)
        self.notify()
        time.sleep(0.2)

    def with_tf_publisher(self):
        """
        Launches a tf publisher in conjunction with the VizMarkerPublisher.
        """
        TFPublisher(_world=self._world, node=self.node)

    def _select_shapes(self, body):
        if self.shape_source is ShapeSource.VISUAL_ONLY:
            return body.visual.shapes
        if self.shape_source is ShapeSource.COLLISION_ONLY:
            return body.collision.shapes
        if self.shape_source is ShapeSource.VISUAL_WITH_COLLISION_BACKUP:
            return body.visual.shapes if body.visual.shapes else body.collision.shapes
        raise ValueError(f"Unsupported shape_source: {self.shape_source!r}")

    def _notify(self, **kwargs):
        self.markers = MarkerArray()
        highlighted_bodies = []
        for entity in self.highlight_entities:
            if isinstance(entity, Body):
                highlighted_bodies.append(entity)
            elif isinstance(entity, SemanticAnnotation):
                highlighted_bodies.extend(entity.bodies)

        for body in self._world.bodies:
            shapes = self._select_shapes(body)
            if not shapes:
                continue
            marker_ns = str(body.name)
            is_highlighted = body in highlighted_bodies

            if is_highlighted and self.highlight_outline:
                # Create a bounding box outline instead of showing the shapes
                mins, maxs = body_local_aabb(body)
                marker = Marker()
                marker.header.frame_id = str(body.name)
                marker.ns = marker_ns + "_outline"
                marker.id = 0
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.scale.x = 0.005  # Line width
                marker.color.r = self.highlight_color.R
                marker.color.g = self.highlight_color.G
                marker.color.b = self.highlight_color.B
                marker.color.a = 1.0
                marker.frame_locked = True

                # Define the 12 edges of the bounding box
                p1 = [mins[0], mins[1], mins[2]]
                p2 = [maxs[0], mins[1], mins[2]]
                p3 = [maxs[0], maxs[1], mins[2]]
                p4 = [mins[0], maxs[1], mins[2]]
                p5 = [mins[0], mins[1], maxs[2]]
                p6 = [maxs[0], mins[1], maxs[2]]
                p7 = [maxs[0], maxs[1], maxs[2]]
                p8 = [mins[0], maxs[1], maxs[2]]

                edges = [
                    (p1, p2),
                    (p2, p3),
                    (p3, p4),
                    (p4, p1),  # Bottom
                    (p5, p6),
                    (p6, p7),
                    (p7, p8),
                    (p8, p5),  # Top
                    (p1, p5),
                    (p2, p6),
                    (p3, p7),
                    (p4, p8),  # Verticals
                ]

                from geometry_msgs.msg import Point

                for start, end in edges:
                    marker.points.append(
                        Point(x=float(start[0]), y=float(start[1]), z=float(start[2]))
                    )
                    marker.points.append(
                        Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))
                    )

                self.markers.markers.append(marker)
                continue

            for i, shape in enumerate(shapes):
                marker = SemDTToRos2Converter.convert(shape)
                if is_highlighted:
                    marker.color.r = self.highlight_color.R
                    marker.color.g = self.highlight_color.G
                    marker.color.b = self.highlight_color.B
                    marker.color.a = self.highlight_alpha
                elif not marker.mesh_use_embedded_materials:
                    marker.color.a = self.alpha
                marker.frame_locked = True
                marker.id = i
                marker.ns = marker_ns
                self.markers.markers.append(marker)

        self.pub.publish(self.markers)
