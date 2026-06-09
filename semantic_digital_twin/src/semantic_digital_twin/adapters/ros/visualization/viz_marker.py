from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from pycram.robot_plans.actions.composite.utils.rviz import _norm_topic, _next_marker_group
from semantic_digital_twin.adapters.ros.msg_converter import SemDTToRos2Converter
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.callbacks.callback import ModelChangeCallback

from typing import TYPE_CHECKING

from geometry_msgs.msg import Point
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

    alpha: float = field(kw_only=True, default=0.5)
    """
    Marker transparency in [0.0, 1.0]. 0.0 is fully transparent.
    """

    markers: MarkerArray = field(init=False, default_factory=MarkerArray)
    """Maker message to be published."""
    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """QoS profile for the publisher."""

    _tf_publisher: Optional[TFPublisher] = field(init=False, default=None)

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
        self._tf_publisher = TFPublisher(_world=self._world, node=self.node)

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
        for body in self._world.bodies:
            shapes = self._select_shapes(body)
            self._add_markers_for_shapes(shapes, str(body.name))

        for region in self._world.regions:
            self._add_markers_for_shapes(region.area.shapes, str(region.name))

        self.pub.publish(self.markers)

    def _add_markers_for_shapes(self, shapes, marker_ns):
        if not shapes:
            return
        for i, shape in enumerate(shapes):
            marker = SemDTToRos2Converter.convert(shape)
            if not marker.mesh_use_embedded_materials:
                marker.color.a *= self.alpha
            marker.frame_locked = True
            marker.id = i
            marker.ns = marker_ns
            self.markers.markers.append(marker)
def _color(r, g, b, a=1.0):
    """Create a ColorRGBA message."""
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c

def _heat_color(value, alpha=0.85):
    """Map a normalized scalar in [0, 1] to a readable heatmap color."""
    v = float(np.clip(value, 0.0, 1.0))
    if v <= 0.25:
        t = v / 0.25
        return _color(0.10, 0.25 + 0.55 * t, 0.95, alpha)
    if v <= 0.50:
        t = (v - 0.25) / 0.25
        return _color(0.10, 0.80 + 0.15 * t, 0.95 - 0.55 * t, alpha)
    if v <= 0.75:
        t = (v - 0.50) / 0.25
        return _color(0.10 + 0.90 * t, 0.95, 0.40 - 0.30 * t, alpha)
    t = (v - 0.75) / 0.25
    return _color(1.00, 0.95 - 0.70 * t, 0.10, alpha)


def _quaternion_to_rotation_matrix(quaternion):
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=float)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )

def _pose_to_position_and_orientation_lists(pose):
    """Extract xyz and xyzw lists from pose-like objects with either old or new APIs."""
    if pose is None:
        raise ValueError("pose must not be None")

    position = getattr(pose, "position", None)
    if position is None and hasattr(pose, "to_position"):
        position = pose.to_position()

    orientation = getattr(pose, "orientation", None)
    if orientation is None and hasattr(pose, "to_quaternion"):
        orientation = pose.to_quaternion()

    if position is None or orientation is None:
        raise AttributeError(
            "pose must provide position/orientation attributes or "
            "to_position()/to_quaternion() methods"
        )
    if hasattr(position, "to_list"):
        position_values = position.to_list()
    else:
        position_values = [
            getattr(position, axis)
            for axis in ("x", "y", "z")
            if hasattr(position, axis)
        ]

    if hasattr(orientation, "to_list"):
        orientation_values = orientation.to_list()
    else:
        orientation_values = [
            getattr(orientation, axis)
            for axis in ("x", "y", "z", "w")
            if hasattr(orientation, axis)
        ]

    position_values = np.asarray(position_values, dtype=float).reshape(-1)
    orientation_values = np.asarray(orientation_values, dtype=float).reshape(-1)

    if position_values.size == 4:
        position_values = position_values[:3]
    elif position_values.size != 3:
        raise ValueError(
            f"pose position must contain 3 values (xyz) or 4 homogeneous values, got {position_values.size}"
        )

    if orientation_values.size != 4:
        raise ValueError(
            f"pose orientation must contain 4 quaternion values (xyzw), got {orientation_values.size}"
        )

    return position_values.tolist(), orientation_values.tolist()


def _transform_point(position, orientation, local_point):
    """Apply a pose defined by position and quaternion to a local 3D point."""
    position_vec = np.asarray(position, dtype=float).reshape(-1)
    if position_vec.size == 4:
        position_vec = position_vec[:3]
    elif position_vec.size != 3:
        raise ValueError(
            f"position must contain 3 values (xyz) or 4 homogeneous values, got {position_vec.size}"
        )

    local_point_vec = np.asarray(local_point, dtype=float).reshape(-1)
    if local_point_vec.size != 3:
        raise ValueError(
            f"local_point must contain exactly 3 values, got {local_point_vec.size}"
        )

    world_point = (
        position_vec + _quaternion_to_rotation_matrix(orientation) @ local_point_vec
    )
    point = Point()
    point.x = float(world_point[0])
    point.y = float(world_point[1])
    point.z = float(world_point[2])
    return point

_MARKER_GROUP_COUNTER = 0
def _next_marker_group(prefix="rviz"):
    global _MARKER_GROUP_COUNTER
    _MARKER_GROUP_COUNTER += 1
    return f"{prefix}_{_MARKER_GROUP_COUNTER}"

class CostmapHeatmapRviz:
    def __init__(
        self,
        costmap,
        *,
        node,
        topic="costmap_heatmap",
        frame_id="map",
        marker_ns=None,
        marker_type=Marker.CUBE_LIST,
        z_offset=0.02,
        z_scale=0.08,
        xy_scale=None,
        cell_height=0.003,
        alpha=0.85,
        republish_hz=2.0,
        min_value=1e-9,
        min_normalized_value=0.0,
        sample_stride=1,
    ):
        """
        Publish a costmap as a colored RViz heatmap using a list marker.
        """
        if node is None:
            raise ValueError("node must be provided when publishing a costmap heatmap")

        self.costmap = costmap
        self.node = node
        self.topic = _norm_topic(topic)
        self.frame_id = str(frame_id)
        self.marker_ns = (
            str(marker_ns)
            if marker_ns is not None
            else _next_marker_group("costmap_heat")
        )
        self.marker_type = int(marker_type)
        self.z_offset = float(z_offset)
        self.z_scale = float(z_scale)
        self.xy_scale = xy_scale
        self.cell_height = float(cell_height)
        self.alpha = float(alpha)
        self.min_value = float(min_value)
        self.min_normalized_value = float(min_normalized_value)
        self.sample_stride = max(int(sample_stride), 1)
        self.pub = self.node.create_publisher(MarkerArray, self.topic, 10)

        self.node.get_logger().info(
            f"Publishing costmap heatmap on {self.topic} in frame '{self.frame_id}'"
        )

        self.publish_once()

        self._timer = None
        if republish_hz is not None and float(republish_hz) > 0.0:
            self._timer = self.node.create_timer(
                1.0 / float(republish_hz), self.publish_once
            )

    def set_costmap(self, costmap):
        self.costmap = costmap

    def _costmap_to_points_and_colors(self):
        if self.costmap is None:
            return [], []

        map_data = np.asarray(self.costmap.map, dtype=float)
        valid_mask = np.isfinite(map_data) & (map_data > self.min_value)
        if not np.any(valid_mask):
            return [], []

        values = map_data[valid_mask]
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        denom = v_max - v_min

        origin_position, origin_orientation = _pose_to_position_and_orientation_lists(
            self.costmap.origin
        )
        corner_offset = np.array(
            [
                -self.costmap.height * self.costmap.resolution / 2.0,
                -self.costmap.width * self.costmap.resolution / 2.0,
                self.z_offset,
            ],
            dtype=float,
        )

        points = []
        colors = []
        for row, col in np.argwhere(valid_mask):
            if (row % self.sample_stride) != 0 or (col % self.sample_stride) != 0:
                continue
            normalized = 1.0 if denom <= 0.0 else (map_data[row, col] - v_min) / denom
            if normalized < self.min_normalized_value:
                continue

            local_point = corner_offset + np.array(
                [
                    (float(row) + 0.5) * self.costmap.resolution,
                    (float(col) + 0.5) * self.costmap.resolution,
                    normalized * self.z_scale,
                ],
                dtype=float,
            )
            points.append(
                _transform_point(origin_position, origin_orientation, local_point)
            )

            colors.append(_heat_color(normalized, alpha=self.alpha))

        return points, colors

    def publish_once(self):
        now = self.node.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = now
        marker.ns = self.marker_ns
        marker.id = 0
        marker.type = self.marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        default_xy_scale = float(self.costmap.resolution) if self.costmap else 0.01
        xy_scale = (
            float(self.xy_scale) if self.xy_scale is not None else default_xy_scale
        )
        marker.scale.x = xy_scale
        marker.scale.y = xy_scale
        marker.scale.z = self.cell_height

        marker.points, marker.colors = self._costmap_to_points_and_colors()

        arr = MarkerArray()
        arr.markers.append(marker)
        self.pub.publish(arr)
