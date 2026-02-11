import numpy as np

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def _norm_topic(topic: str) -> str:
    """Ensure the topic name starts with a slash."""
    t = str(topic)
    return t if t.startswith("/") else "/" + t


def _color(r, g, b, a=1.0):
    """Create a ColorRGBA message."""
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def _as_points(P):
    """Convert an array of points into a list of ROS Point messages."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    pts = []
    for i in range(P.shape[0]):
        p = Point()
        p.x = float(P[i, 0])
        p.y = float(P[i, 1])
        p.z = float(P[i, 2])
        pts.append(p)
    return pts


def _phase_color(k, a=1.0):
    """Pick a color from a fixed palette for a phase index."""
    palette = [
        (1.0, 0.2, 0.2),
        (0.2, 1.0, 0.2),
        (0.2, 0.2, 1.0),
        (1.0, 0.8, 0.2),
        (0.8, 0.2, 1.0),
        (0.2, 1.0, 1.0),
    ]
    r, g, b = palette[int(k) % len(palette)]
    return _color(r, g, b, a)


class PhaseSequenceRviz:
    def __init__(
        self,
        P,
        phase_id=None,
        frame_id="map",
        topic="phase_sequence_markers",
        node=None,
        line_width=0.01,
        alpha=1.0,
        republish_hz=2.0,
    ):
        """Publish a phase sequence as RViz line markers."""
        self.P = np.asarray(P, dtype=float).reshape(-1, 3)
        self.phase_id = (
            None if phase_id is None else np.asarray(phase_id, dtype=int).reshape(-1)
        )
        self.frame_id = str(frame_id)
        self.topic = _norm_topic(topic)
        self.line_width = float(line_width)
        self.alpha = float(alpha)

        if node is None:
            raise ValueError(
                "node must be provided when using PhaseSequenceRviz directly in your system"
            )

        self.node = node
        self.pub = self.node.create_publisher(MarkerArray, self.topic, 10)

        self.node.get_logger().info(
            f"Publishing MarkerArray on {self.topic} in frame '{self.frame_id}'"
        )

        self.publish_once()

        self._timer = None
        if republish_hz is not None and float(republish_hz) > 0.0:
            period = 1.0 / float(republish_hz)
            self._timer = self.node.create_timer(period, self.publish_once)

    def publish_once(self):
        """Publish the current sequence once."""
        now = self.node.get_clock().now().to_msg()
        arr = MarkerArray()

        if self.phase_id is None:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = "phase_seq"
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = self.line_width
            m.color = _color(1.0, 1.0, 1.0, self.alpha)
            m.points = _as_points(self.P)
            arr.markers.append(m)
        else:
            start = 0
            mid = 0
            while start < self.P.shape[0]:
                pid = int(self.phase_id[start])
                end = start + 1
                while end < self.P.shape[0] and int(self.phase_id[end]) == pid:
                    end += 1

                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = now
                m.ns = "phase_seq_phase"
                m.id = mid
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = self.line_width
                m.color = _phase_color(pid, a=self.alpha)
                m.points = _as_points(self.P[start:end, :])
                arr.markers.append(m)

                mid += 1
                start = end

        self.pub.publish(arr)
