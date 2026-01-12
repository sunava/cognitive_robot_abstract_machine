import atexit
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import rclpy.node

from .. import logger
from ..callbacks.callback import StateChangeCallback

try:
    from builtin_interfaces.msg import Duration, Time
    from geometry_msgs.msg import Vector3, Point, Quaternion, Pose
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Vector3, Point, PoseStamped, Quaternion, Pose
except ImportError as e:
    logger.warning(
        f"Could not import ros messages, viz marker will not be available: {e}"
    )

from scipy.spatial.transform import Rotation

from ..world_description.geometry import (
    FileMesh,
    Box,
    Sphere,
    Cylinder,
    TriangleMesh,
)
from ..world import World


@dataclass
class VizMarkerPublisher(StateChangeCallback):
    """
    Publishes an Array of visualization marker which represent the situation in the World
    """

    node: rclpy.node.Node
    """
    The ROS2 node that will be used to publish the visualization marker.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the Visualization Marker should be published.
    """

    reference_frame: str = "map"
    """
    The reference frame of the visualization marker.
    """

    throttle_state_updates: int = 15
    """
    Only published every n-th state update.
    """

    opacity: float = 0.4
    """
    The opacity of the visualization marker.
    """

    def __post_init__(self):
        """
        Initializes the publisher and registers the callback to the world.
        """
        super().__post_init__()
        self.pub = self.node.create_publisher(MarkerArray, self.topic_name, 10)
        time.sleep(0.2)
        self.notify()

    def _notify(self):
        """
        Publishes the Marker Array on world changes.
        """
        if self.world.state.version % self.throttle_state_updates != 0:
            return
        marker_array = self._make_marker_array()
        self.pub.publish(marker_array)

    def _make_marker_array(self) -> MarkerArray:
        """
        Creates the Marker Array to be published. There is one Marker for link for each object in the Array, each Object
        creates a name space in the visualization Marker. The type of Visualization Marker is decided by the collision
        tag of the URDF.

        :return: An Array of Visualization Marker
        """
        marker_array = MarkerArray()
        for body in self.world.bodies:
            for i, collision in enumerate(body.collision):
                msg = Marker()
                msg.header.frame_id = self.reference_frame
                msg.ns = body.name.name
                msg.id = i
                msg.action = Marker.ADD
                msg.pose = self.transform_to_pose(
                    (
                        self.world.compute_forward_kinematics(self.world.root, body)
                        @ collision.origin
                    ).to_np()
                )

                msg.color = ColorRGBA(
                    r=float(collision.color.R),
                    g=float(collision.color.G),
                    b=float(collision.color.B),
                    a=float(collision.color.A) * self.opacity,
                )
                msg.lifetime = Duration(sec=0)

                if isinstance(collision, FileMesh):
                    msg.type = Marker.MESH_RESOURCE
                    msg.mesh_resource = "file://" + collision.filename
                    msg.scale = Vector3(
                        x=float(collision.scale.x),
                        y=float(collision.scale.y),
                        z=float(collision.scale.z),
                    )
                    msg.mesh_use_embedded_materials = True
                elif isinstance(collision, TriangleMesh):
                    f = collision.file
                    msg.type = Marker.MESH_RESOURCE
                    msg.mesh_resource = "file://" + f.name
                    msg.scale = Vector3(
                        x=float(collision.scale.x),
                        y=float(collision.scale.y),
                        z=float(collision.scale.z),
                    )
                    msg.mesh_use_embedded_materials = True
                elif isinstance(collision, Cylinder):
                    msg.type = Marker.CYLINDER
                    msg.scale = Vector3(
                        x=float(collision.width),
                        y=float(collision.width),
                        z=float(collision.height),
                    )
                elif isinstance(collision, Box):
                    msg.type = Marker.CUBE
                    msg.scale = Vector3(
                        x=float(collision.scale.x),
                        y=float(collision.scale.y),
                        z=float(collision.scale.z),
                    )
                elif isinstance(collision, Sphere):
                    msg.type = Marker.SPHERE
                    msg.scale = Vector3(
                        x=float(collision.radius * 2),
                        y=float(collision.radius * 2),
                        z=float(collision.radius * 2),
                    )

                marker_array.markers.append(msg)
        return marker_array

    @staticmethod
    def transform_to_pose(transform: np.ndarray) -> Pose:
        """
        Converts a 4x4 transformation matrix to a PoseStamped message.

        :param transform: The transformation matrix to convert.
        :return: A PoseStamped message.
        """
        pose = Pose()
        pose.position = Point(**dict(zip(["x", "y", "z"], transform[:3, 3])))
        pose.orientation = Quaternion(
            **dict(
                zip(
                    ["x", "y", "z", "w"],
                    Rotation.from_matrix(transform[:3, :3]).as_quat(),
                )
            )
        )
        return pose


class TrajLinePublisher:
    def __init__(self, world, node):
        self.world = world
        self.pub = node.create_publisher(MarkerArray, "/semworld/primitive_traj", 10)

    def _to_ros_pose(self, p):
        # PyCRAM PoseStamped → world → numeric → ROS PoseStamped
        T = self.world.transform(p.to_spatial_type(), self.world.root)

        pos = T.to_position().to_np()
        quat = T.to_quaternion().to_np()

        msg = PoseStamped()
        msg.header.frame_id = "map"

        if hasattr(p.header, "stamp"):
            msg.header.stamp = self._to_ros_time(p.header.stamp)

        msg.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        msg.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
        )
        return msg

    def _to_ros_time(self, dt):
        t = Time()
        sec = int(dt.timestamp())
        nsec = int((dt.timestamp() - sec) * 1_000_000_000)
        t.sec = sec
        t.nanosec = nsec
        return t

    def publish(self, poses, ns: str, marker_id: int = 0):
        poses = [self._to_ros_pose(p) for p in poses]
        if not poses:
            return

        m = Marker()
        m.header.frame_id = poses[0].header.frame_id
        m.ns = ns
        m.id = marker_id
        m.action = Marker.ADD
        m.type = Marker.LINE_STRIP
        m.lifetime = Duration(sec=0)
        m.scale = Vector3(x=0.004, y=0.0, z=0.0)
        m.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)

        m.points = [
            Point(x=p.pose.position.x, y=p.pose.position.y, z=p.pose.position.z)
            for p in poses
        ]

        arr = MarkerArray()
        arr.markers.append(m)
        self.pub.publish(arr)
