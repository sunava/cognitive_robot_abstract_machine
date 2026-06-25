from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

from ... import MoveTCPWaypointsAlignedMotion

from ....datastructures.enums import (
    Arms,
    CuttingTechnique,
    MixingPattern,
    MovementType,
    WipingTechnique,
)
from ....plans.factories import sequential
from ....robot_plans.actions.base import ActionDescription
from ....robot_plans.motions.gripper import MoveToolCenterPointMotion
from ....view_manager import ViewManager

DEFAULT_SAMPLE_DT = 0.01
APPROACH_Z_EXTRA_CLEARANCE_M = 0.02


def _lerp(a, b, tau):
    """Linear interpolation from a to b for tau in [0, 1]."""
    return a + (b - a) * float(tau)


# ---------------------------------------------------------------------------
# Self-contained motion math (previously the thesis_math package).
# ---------------------------------------------------------------------------


def _shape_scale_xyz(shape):
    """Extract an (x,y,z) scale array from a shape if present."""
    scale = getattr(shape, "scale", None)
    if scale is None:
        return None
    if hasattr(scale, "x") and hasattr(scale, "y") and hasattr(scale, "z"):
        return np.array([scale.x, scale.y, scale.z], dtype=float)
    try:
        arr = np.asarray(scale, dtype=float).reshape(3)
    except Exception:
        return None
    return arr


def body_local_aabb(body, use_visual=False, apply_shape_scale=False):
    """Compute the local AABB for a body."""
    geom = body.visual if use_visual else body.collision
    bbc = geom.as_bounding_box_collection_in_frame(body)
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for bb in bbc.bounding_boxes:
        mins = np.minimum(mins, [bb.min_x, bb.min_y, bb.min_z])
        maxs = np.maximum(maxs, [bb.max_x, bb.max_y, bb.max_z])
    if apply_shape_scale:
        max_scale = np.ones(3, dtype=float)
        for shape in getattr(geom, "shapes", []):
            scale = _shape_scale_xyz(shape)
            if scale is not None:
                max_scale = np.maximum(max_scale, np.abs(scale))
        if not np.allclose(max_scale, 1.0):
            mins = mins * max_scale
            maxs = maxs * max_scale
    return mins, maxs


class MotionSegment:
    def __init__(self, name, duration_s, local_curve):
        """Define a local motion curve over a fixed duration."""
        self.name = str(name)
        self.duration_s = float(duration_s)
        self.local_curve = local_curve

    def sample(self, frame, dt, t0=0.0):
        n = max(2, int(np.ceil(self.duration_s / float(dt))) + 1)
        times = np.linspace(t0, t0 + self.duration_s, n)
        tau = (times - t0) / self.duration_s

        F = np.asarray(frame, dtype=float)
        R = F[:3, :3]
        t = F[:3, 3]

        q = np.array([self.local_curve(float(u)) for u in tau], dtype=float).reshape(
            -1, 3
        )
        pts = q @ R.T + t
        return times, pts


class MotionSequence:
    def __init__(self, phases):
        """Store an ordered list of phases."""
        self.phases = list(phases)

    @property
    def duration_s(self):
        """Total duration across all phases."""
        return float(sum(p.duration_s for p in self.phases))

    def sample(self, frame, dt, t0=0.0):
        """Sample all phases into one concatenated sequence."""
        all_t, all_p, all_id = [], [], []
        t = float(t0)

        for k, ph in enumerate(self.phases):
            tt, pp = ph.sample(frame.to_np(), dt=dt, t0=t)
            if all_t:
                tt = tt[1:]
                pp = pp[1:]

            all_t.append(tt)
            all_p.append(pp)
            all_id.append(np.full(len(tt), k, dtype=int))
            t += ph.duration_s

        return np.concatenate(all_t), np.vstack(all_p), np.concatenate(all_id)


def ramp(tau, tau_end, d_max):
    """Linear ramp from 0 to d_max over tau_end."""
    if tau <= 0.0:
        return 0.0
    if tau >= tau_end:
        return float(d_max)
    return float(d_max) * (tau / tau_end)


def planar_spiral_xy(tau, r0, r1, cycles):
    """Planar spiral in XY with linearly growing radius."""
    r = r0 + (r1 - r0) * tau
    ang = 2.0 * np.pi * cycles * tau
    return np.array([r * np.cos(ang), r * np.sin(ang), 0.0], dtype=float)


def planar_sweep_x(tau, length, cycles):
    """Sinusoidal sweep along the X axis."""
    s = float(length) * np.sin(2.0 * np.pi * float(cycles) * tau)
    return np.array([s, 0.0, 0.0], dtype=float)


def planar_raster_xy(tau, width, height, lanes):
    """Raster scan covering a rectangle in XY."""
    w = float(width)
    h = float(height)
    n = max(2, int(lanes))
    u = float(np.clip(tau, 0.0, 1.0)) * n
    lane = int(np.floor(u))
    if lane >= n:
        lane = n - 1
    local_t = u - lane

    x0 = -0.5 * w
    x1 = 0.5 * w
    if (lane % 2) == 0:
        x = x0 + (x1 - x0) * local_t
    else:
        x = x1 - (x1 - x0) * local_t

    y = -0.5 * h + (h * lane / float(n - 1))
    return np.array([x, y, 0.0], dtype=float)


@dataclass(frozen=True)
class ShearProfile:
    depth_max: float
    depth_ramp_end: float
    shear_amp: float
    shear_cycles: float


@dataclass(frozen=True)
class ShearXYProfile:
    shear_amp: float
    shear_cycles: float


def oscillatory_shear_local_profiled(tau, prof: ShearProfile):
    """Oscillatory shear with a monotone depth profile."""
    d = ramp(tau, tau_end=prof.depth_ramp_end, d_max=prof.depth_max)
    s = float(prof.shear_amp) * np.sin(2.0 * np.pi * float(prof.shear_cycles) * tau)
    return np.array([s, 0.0, -d], dtype=float)


def oscillatory_shear_xy_profiled(tau, prof: ShearXYProfile):
    """Oscillatory shear in XY plane with no depth change."""
    ang = 2.0 * np.pi * float(prof.shear_cycles) * tau
    s = float(prof.shear_amp)
    return np.array([s * np.sin(ang), s * np.cos(ang), 0.0], dtype=float)


def clamp_to_cylinder_xy(q_local, radius, z_min, z_max, margin=0.0):
    """Clamp a point to a vertical cylinder in XY and Z bounds."""
    q = np.asarray(q_local, dtype=float).reshape(3)
    r = float(radius) - float(margin)
    xy = q[:2]
    r_xy = np.linalg.norm(xy)
    if r_xy > r and r_xy > 1e-9:
        xy = (r / r_xy) * xy
    z = np.clip(q[2], float(z_min) + float(margin), float(z_max) - float(margin))
    return np.array([xy[0], xy[1], z], dtype=float)


def make_constrained_curve(local_curve, constraint_fn):
    """Wrap a curve so it respects a constraint function."""
    return lambda tau: constraint_fn(local_curve(float(tau)))


def duration_scale_from_body(
    body,
    reference_size=0.10,
    use_visual_aabb=False,
    apply_shape_scale=False,
):
    """Compute a scaling factor from the body's AABB size."""
    mins, maxs = body_local_aabb(
        body, use_visual=use_visual_aabb, apply_shape_scale=apply_shape_scale
    )
    diag = float(np.linalg.norm(maxs - mins))
    ref = float(reference_size)
    if ref <= 0.0:
        raise ValueError("reference_size must be positive")
    return max(diag, 1e-6) / ref


def _cross_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if len(pts) <= 1:
        return pts
    pts = np.unique(np.round(pts, decimals=8), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _point_in_convex_polygon_xy(point_xy: np.ndarray, hull_xy: np.ndarray) -> bool:
    hull = np.asarray(hull_xy, dtype=float).reshape(-1, 2)
    point = np.asarray(point_xy, dtype=float).reshape(2)
    if len(hull) < 3:
        return True
    eps = 1e-9
    prev_sign = 0.0
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        cross = _cross_2d(a, b, point)
        if abs(cross) <= eps:
            continue
        sign = np.sign(cross)
        if prev_sign == 0.0:
            prev_sign = sign
        elif sign != prev_sign:
            return False
    return True


def _project_point_to_segment_xy(
    point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray
) -> np.ndarray:
    p = np.asarray(point_xy, dtype=float).reshape(2)
    a = np.asarray(a_xy, dtype=float).reshape(2)
    b = np.asarray(b_xy, dtype=float).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy()
    t = float(np.dot(p - a, ab) / denom)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


def constrain_to_convex_hull_xy(q_local: np.ndarray, hull_xy: np.ndarray) -> np.ndarray:
    q = np.asarray(q_local, dtype=float).reshape(3)
    if len(hull_xy) < 3 or _point_in_convex_polygon_xy(q[:2], hull_xy):
        return q

    best_xy = None
    best_dist = float("inf")
    for i in range(len(hull_xy)):
        a = hull_xy[i]
        b = hull_xy[(i + 1) % len(hull_xy)]
        candidate_xy = _project_point_to_segment_xy(q[:2], a, b)
        dist = float(np.linalg.norm(q[:2] - candidate_xy))
        if dist < best_dist:
            best_dist = dist
            best_xy = candidate_xy

    return np.array([best_xy[0], best_xy[1], q[2]], dtype=float)


def top_surface_hull_xy(surface_body, upward_threshold=0.6):
    mesh = getattr(surface_body, "combined_mesh", None)
    if mesh is None or getattr(mesh, "is_empty", True):
        return None

    normals = np.asarray(mesh.face_normals, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(normals) == 0 or len(faces) == 0 or len(vertices) == 0:
        return None

    upward_mask = normals[:, 2] > float(upward_threshold)
    if not upward_mask.any():
        return None

    upward_faces = faces[upward_mask]
    top_vertices = vertices[np.unique(upward_faces.reshape(-1))]
    if len(top_vertices) < 3:
        return None

    return _convex_hull_xy(top_vertices[:, :2])


@dataclass
class GeneralizedActionPlan(ActionDescription):
    """
    Base class for tool-based motion sequences over a container.
    """

    #: Reference object size (m) used to scale motion duration to object size.
    REFERENCE_SIZE_M = 0.10

    arm: Arms = None
    """
    Arm used for the motion.
    """

    tool: HasRootBody = None
    """
    Tool body used to estimate the tip offset.
    """

    pointer_stride: Optional[int] = 1
    """
    Keep every Nth waypoint for execution (testing downsampling).
    """

    def execute(self) -> None:
        sampled = self._sample_motion()
        self.robot.full_body_controlled = True
        waypoints = self._build_waypoints(sampled)
        self._last_waypoints = waypoints

        try:
            self.add_subplan(sequential([self._build_motion(waypoints)])).perform()
        except Exception as exc:
            if self._accept_execution_failure_as_success(waypoints, exc):
                return
            raise

    def _accept_execution_failure_as_success(self, waypoints, exc: Exception) -> bool:
        return False

    def _sample_motion(self):
        return self._sample_points()

    def _duration_scale(self, body, *, use_visual_aabb=False):
        """Scale a motion's duration to the size of the given body."""
        return duration_scale_from_body(
            body,
            reference_size=self.REFERENCE_SIZE_M,
            use_visual_aabb=use_visual_aabb,
            apply_shape_scale=True,
        )

    def _build_waypoints(self, sampled):
        _, points, _ = sampled
        stride = max(1, int(self.pointer_stride))
        return self._to_waypoints(points, stride)

    def _build_motion(self, waypoints):
        alignment_target = self._resolve_alignment_target()

        alignment_pairs = (
            self.tool.tool_alignment(alignment_target)
            if (self.tool is not None and alignment_target is not None)
            else []
        )
        try:
            tip = self.tool.get_tool_frame()
        except Exception:
            tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame

        return MoveTCPWaypointsAlignedMotion(
            waypoints,
            self.arm,
            allow_gripper_collision=True,
            alignment_pairs=alignment_pairs,
            tip=tip,
        )

    def _sample_points(self):
        raise NotImplementedError

    def _to_waypoints(self, points: np.ndarray, stride: int) -> list[Point3]:
        waypoints = [
            Point3(x=p[0], y=p[1], z=p[2], reference_frame=self.world.root)
            for p in points
        ][::stride]
        if not waypoints:
            raise ValueError("No waypoints left after applying pointer_stride.")
        return waypoints

    def _resolve_alignment_target(self):
        if hasattr(self, "surface_body") and self.surface_body is not None:
            return self.surface_body
        if hasattr(self, "container") and self.container is not None:
            return self.container
        if hasattr(self, "target_pose") and self.target_pose is not None:
            return self.target_pose
        return None


@dataclass
class MixingAction(GeneralizedActionPlan):
    """
    Execute a mixing motion sequence around a container.
    """

    motion_timeout_ticks = 300

    container: Body = None
    """
    The container (e.g., bowl) to operate in.
    """

    mix_duration_s: float = 0.0
    """
    Total mixing time in seconds for a continuous connected stir loop.
    If <= 0, the default short pattern is used.
    """

    def build_container_sequence(self):
        """Build a spiral/stir sequence sized to the bowl-like container."""
        pattern = (
            MixingPattern.STIR
            if float(self.mix_duration_s) > 0.0
            else MixingPattern.SPIRAL
        )
        mix_duration_s = (
            self.mix_duration_s if float(self.mix_duration_s) > 0.0 else None
        )

        mins, maxs = body_local_aabb(
            self.container, use_visual=False, apply_shape_scale=True
        )
        size_x = maxs[0] - mins[0]
        size_y = maxs[1] - mins[1]
        radius_xy = 0.5 * min(size_x, size_y)
        z_min, z_max = mins[2], maxs[2]

        center_y = 0.5 * (mins[1] + maxs[1])
        surface_margin = 0.005
        start_offset = np.array(
            [0.0, center_y, maxs[2] + APPROACH_Z_EXTRA_CLEARANCE_M - surface_margin],
            dtype=float,
        )
        duration_scale = self._duration_scale(self.container)
        spiral_r1 = 0.75 * radius_xy

        # Keep points inside a bowl-shaped vertical cylinder.
        def _bowl_constraint(q_local):
            return clamp_to_cylinder_xy(
                q_local, radius=radius_xy, z_min=z_min, z_max=z_max, margin=0.005
            )

        # Offset the local curve and apply the bowl constraint.
        def _with_offset(curve):
            return make_constrained_curve(
                lambda tau: curve(tau) + start_offset, _bowl_constraint
            )

        phase_spiral_container = MotionSegment(
            name="planar_spiral_bowl",
            duration_s=2.0 * duration_scale,
            local_curve=_with_offset(
                lambda tau: planar_spiral_xy(tau, r0=0.00, r1=spiral_r1, cycles=2.0)
            ),
        )

        stir_amp = max(0.005, 0.55 * radius_xy)
        stir_base_duration = max(1.0, 2.0 * duration_scale)
        if mix_duration_s is not None and float(mix_duration_s) > 0.0:
            total_duration = float(mix_duration_s)
        else:
            total_duration = stir_base_duration
        stir_loops = max(1, int(np.ceil(total_duration / stir_base_duration)))

        phase_stir_container = MotionSegment(
            name="continuous_stir_bowl",
            duration_s=total_duration,
            local_curve=_with_offset(
                lambda tau: oscillatory_shear_xy_profiled(
                    tau,
                    ShearXYProfile(
                        shear_amp=stir_amp,
                        shear_cycles=stir_loops,
                    ),
                )
            ),
        )

        if pattern == MixingPattern.SPIRAL:
            return MotionSequence([phase_spiral_container])
        if pattern == MixingPattern.STIR:
            return MotionSequence([phase_stir_container])

        raise ValueError(f"Unknown pattern '{pattern}'")

    def _sample_points(self):
        seq = self.build_container_sequence()
        return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)


@dataclass(kw_only=True)
class WipingAction(GeneralizedActionPlan):
    """
    Execute a planar wiping motion around a target pose.
    """

    motion_timeout_ticks = 300

    final_waypoint_success_tolerance_m: float = 0.08
    """
    Accept a timeout as successful if the tool is already this close to the final wipe waypoint.
    """

    container: Optional[Body] = None
    """
    Optional alias for surface_body (backward compatibility).
    """
    target_pose: Optional[Pose] = None
    """
    Center pose for the wiping patch.
    """
    technique: WipingTechnique = WipingTechnique.WIPE
    """
    Surface-contact technique variant.
    """
    length: float = 0.20
    """
    Sweep length for the wiping motion.
    """

    cycles: float = 1.0
    """
    Number of sweep cycles.
    """
    _resolved_surface_body: Optional[Body] = field(default=None, init=False, repr=False)
    _resolved_alignment_target: Optional[Body | Pose] = field(
        default=None, init=False, repr=False
    )

    def _target_pose_to_spatial(self):
        if self.target_pose is None:
            return None
        if getattr(self.target_pose, "frame_id", None) is None:
            self.target_pose.frame_id = self.world.root
        return self.target_pose

    def _resolve_surface_body(self) -> Body:
        if self._resolved_surface_body is not None:
            return self._resolved_surface_body

        if self.container is None:
            if self.target_pose is None:
                raise ValueError(
                    "WipingAction requires either container or target_pose."
                )
            resolved = getattr(self.target_pose, "frame_id", None)
        else:
            resolved = self.container
        self._resolved_surface_body = resolved
        return resolved

    def _resolve_alignment_target(self):
        if self._resolved_alignment_target is not None:
            return self._resolved_alignment_target
        if self.container is None and self.target_pose is not None:
            self._resolved_alignment_target = self._resolve_pose_alignment_target()
            return self._resolved_alignment_target
        return self._resolve_surface_body()

    def _resolve_pose_alignment_target(self):
        return self.target_pose

    def build_surface_sequence(self):
        """Build a planar wiping/spreading sequence over the surface body."""
        technique = self.technique

        mins, maxs = body_local_aabb(
            self.container, use_visual=True, apply_shape_scale=True
        )
        size_x = maxs[0] - mins[0]
        size_y = maxs[1] - mins[1]
        radius_xy = 0.45 * min(size_x, size_y)

        center_x = 0.5 * (mins[0] + maxs[0])
        center_y = 0.5 * (mins[1] + maxs[1])
        surface_margin = 0.005
        start_offset = np.array(
            [
                center_x,
                center_y,
                maxs[2] + APPROACH_Z_EXTRA_CLEARANCE_M - surface_margin,
            ],
            dtype=float,
        )
        duration_scale = self._duration_scale(self.container, use_visual_aabb=True)
        spiral_r1 = 0.9 * radius_xy
        shear_amp = 0.35 * radius_xy
        raster_width = 0.9 * size_x
        raster_height = 0.9 * size_y
        lane_spacing = max(0.02, 0.25 * float(self.REFERENCE_SIZE_M))
        raster_lanes = max(4, int(np.ceil(raster_height / max(lane_spacing, 1e-6))))

        hull_xy = top_surface_hull_xy(self.container)

        def _with_hull_constraint(curve):
            if hull_xy is None:
                return lambda tau: curve(tau) + start_offset
            return make_constrained_curve(
                lambda tau: curve(tau) + start_offset,
                lambda q_local: constrain_to_convex_hull_xy(q_local, hull_xy),
            )

        phase_spiral_surface = MotionSegment(
            name="planar_spiral_surface",
            duration_s=2.0 * duration_scale,
            local_curve=_with_hull_constraint(
                lambda tau: planar_spiral_xy(tau, r0=0.00, r1=spiral_r1, cycles=2.0)
            ),
        )

        phase_shear_surface = MotionSegment(
            name="oscillatory_shear_surface",
            duration_s=1.5 * duration_scale,
            local_curve=_with_hull_constraint(
                lambda tau: oscillatory_shear_xy_profiled(
                    tau,
                    ShearXYProfile(
                        shear_amp=shear_amp,
                        shear_cycles=5.0,
                    ),
                )
            ),
        )

        phase_raster_surface = MotionSegment(
            name="planar_raster_surface",
            duration_s=2.0 * duration_scale,
            local_curve=_with_hull_constraint(
                lambda tau: planar_raster_xy(
                    tau,
                    width=raster_width,
                    height=raster_height,
                    lanes=raster_lanes,
                )
            ),
        )

        if technique == WipingTechnique.WIPE:
            return MotionSequence([phase_spiral_surface])
        if technique == WipingTechnique.SHEAR:
            return MotionSequence([phase_shear_surface])
        if technique == WipingTechnique.SPREAD:
            return MotionSequence([phase_raster_surface])

        raise ValueError(f"Unknown surface technique '{technique}'")

    def build_patch_sequence(self):
        if self.technique == WipingTechnique.SPREAD:
            segment = MotionSegment(
                name="spread_patch",
                duration_s=2.0,
                local_curve=lambda tau: planar_sweep_x(
                    tau, length=float(self.length), cycles=max(1.0, float(self.cycles))
                ),
            )
        else:
            segment = MotionSegment(
                name="wipe_patch",
                duration_s=2.0,
                local_curve=lambda tau: planar_spiral_xy(
                    tau, r0=0.00, r1=0.12, cycles=2.5
                ),
            )
        return MotionSequence([segment])

    def _sample_points(self):
        if self.container is not None:
            seq = self.build_surface_sequence()
            return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)

        if self.target_pose is None:
            raise ValueError(
                "WipingAction requires either container or target_pose for sampling."
            )

        t_pose = self._target_pose_to_spatial()
        seq = self.build_patch_sequence()
        return seq.sample(frame=t_pose, dt=DEFAULT_SAMPLE_DT)

    def _accept_execution_failure_as_success(self, waypoints, exc: Exception) -> bool:
        if not self._is_timeout_failure(exc) or not waypoints:
            return False

        sponge = getattr(getattr(self, "tool", None), "root", None)
        if sponge is None:
            return False

        try:
            tip_point = self.world.transform(
                sponge.global_pose.to_position(), self.world.root
            )
            tip_xyz = np.asarray(tip_point.to_np(), dtype=float).reshape(-1)[:3]
            goal_point = self.world.transform(waypoints[-1], self.world.root)
            goal_xyz = np.asarray(goal_point.to_np(), dtype=float).reshape(-1)[:3]
            distance = float(np.linalg.norm(tip_xyz - goal_xyz))
        except Exception:
            return False

        return distance <= float(self.final_waypoint_success_tolerance_m)

    def _accept_motion_timeout_as_success(self, exc: Exception) -> bool:
        waypoints = getattr(self, "_last_waypoints", [])
        return self._accept_execution_failure_as_success(waypoints, exc)

    @staticmethod
    def _is_timeout_failure(exc: Exception) -> bool:
        current = exc
        while current is not None:
            if isinstance(current, TimeoutError):
                return True
            msg = str(current)
            if (
                "Timeout reached while waiting for end of motion" in msg
                or "Motion stalled while waiting for end of motion" in msg
                or "Hard timeout reached while waiting for end of motion" in msg
            ):
                return True
            current = current.__cause__ or current.__context__
        return False


@dataclass
class CuttingAction(GeneralizedActionPlan):
    """
    Execute a cutting motion sequence on a food object.
    """

    motion_timeout_ticks = 100

    container: Body = None
    """
    The object to cut.
    """
    technique: CuttingTechnique = CuttingTechnique.SAW
    """
    Cutting trajectory variant.
    """

    slice_thickness: float = 0.03
    """
    Target slice thickness used to place the cut anchor.
    """

    num_cuts_x: int = 1
    """
    Number of repeated cut passes distributed across local X.
    """

    def build_cutting_sequence(self):
        """Build a cutting sequence sized to the food object."""
        technique = self.technique

        mins, maxs = body_local_aabb(
            self.container, use_visual=True, apply_shape_scale=True
        )
        size_x, size_y, size_z = maxs - mins
        duration_scale = self._duration_scale(self.container)

        margin_x = min(0.01, 0.15 * size_x)
        margin_y = min(0.01, 0.10 * size_y)
        z_clearance = max(0.01, 0.25 * size_z) + APPROACH_Z_EXTRA_CLEARANCE_M
        z_top = maxs[2] + z_clearance
        z_cut = mins[2] + max(0.003, 0.05 * size_z)
        center_y = 0.5 * (mins[1] + maxs[1])

        usable_x = max(0.0, size_x - 2.0 * margin_x)
        requested_thickness = max(float(self.slice_thickness), 1e-4)
        x_anchor = mins[0] + margin_x + min(0.5 * requested_thickness, 0.5 * usable_x)
        x_max_anchor = (
            maxs[0] - margin_x - min(0.5 * requested_thickness, 0.5 * usable_x)
        )

        n_cuts = max(1, int(self.num_cuts_x))
        if technique == CuttingTechnique.HALVE:
            x_anchors = [0.5 * (mins[0] + maxs[0])]
            z_cut = 0.5 * (mins[2] + maxs[2])
        elif n_cuts == 1:
            x_anchors = [x_anchor]
        elif x_max_anchor <= x_anchor:
            x_anchors = [x_anchor] * n_cuts
        else:
            x_anchors = np.linspace(x_anchor, x_max_anchor, n_cuts).tolist()

        y_min = mins[1] + margin_y
        y_max = maxs[1] - margin_y
        saw_cycles = max(2.0, round(size_y / max(self.REFERENCE_SIZE_M, 1e-6)))
        shear_depth_max = maxs[2] - z_cut
        shear_amp = 0.5 * max(y_max - y_min, 0.0)

        # Straight vertical move at a fixed x (approach / descend / retract).
        def vertical_segment(name, x_val, z_from, z_to, duration_s):
            return MotionSegment(
                name=name,
                duration_s=duration_s,
                local_curve=lambda tau, x_val=x_val, z_from=z_from, z_to=z_to: np.array(
                    [x_val, center_y, _lerp(z_from, z_to, tau)], dtype=float
                ),
            )

        # Back-and-forth sawing in the y/z plane at a fixed x.
        def saw_segment(name, x_val):
            profile = ShearProfile(
                depth_max=shear_depth_max,
                depth_ramp_end=0.7,
                shear_amp=shear_amp,
                shear_cycles=saw_cycles,
            )

            def curve(tau, x_val=x_val):
                q = oscillatory_shear_local_profiled(tau, profile)
                return np.array(
                    [x_val, center_y + q[0], maxs[2] + q[2]], dtype=float
                )

            return MotionSegment(
                name=name, duration_s=5.0 * duration_scale, local_curve=curve
            )

        if technique in (CuttingTechnique.SLICE, CuttingTechnique.HALVE):

            def cut_segment(idx, x_val):
                return vertical_segment(
                    f"cut_descend_x{idx}", x_val, maxs[2], z_cut, 1.0 * duration_scale
                )

        elif technique == CuttingTechnique.SAW:

            def cut_segment(idx, x_val):
                return saw_segment(f"oscillatory_shear_x{idx}", x_val)

        else:
            raise ValueError(f"Unknown cutting technique '{technique}'")

        phases = []
        for idx, x_val in enumerate(x_anchors):
            phases.append(
                vertical_segment(
                    f"cut_approach_x{idx}", x_val, z_top, maxs[2], 0.8 * duration_scale
                )
            )
            phases.append(cut_segment(idx, x_val))
            phases.append(
                vertical_segment(
                    f"cut_retract_x{idx}", x_val, z_cut, z_top, 0.8 * duration_scale
                )
            )

        return MotionSequence(phases)

    def _sample_points(self):
        seq = self.build_cutting_sequence()
        return seq.sample(frame=self.container.global_pose, dt=DEFAULT_SAMPLE_DT)


@dataclass(kw_only=True)
class SimplePouringAction(ActionDescription):
    """
    Park the arms of the robot.
    """

    object_designator: Body = None
    """
    The object to pick up
    """

    source_object_designator: Body = None
    """
    The object to pick up
    """

    arm: Arms = None
    """
    Physical robot arm used for motion execution.
    """

    pour_side: Optional[Arms] = None
    """
    Robot-relative side of the bowl to pour from. Defaults to the physical arm.
    This lets one-arm robots use their available arm while still trying the
    right-side or left-side pouring geometry.
    """

    nav: Pose = None
    """
    Entry from the enum for which arm should be parked.
    """

    pour_side_offset_m: float = 0.10
    """
    Lateral TCP offset from the bowl center in the robot-relative left/right direction.
    """

    pour_approach_offset_m: float = 0.0
    """
    Extra offset away from the bowl along the robot-to-bowl approach direction.
    """

    pour_height_m: float = 0.13
    """
    TCP height above the bowl pose for the pre-pour pose.
    """

    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    tilt_degrees: float = 0.0

    def _held_object_height_m(self) -> float:
        held_body = getattr(self.source_object_designator, "root", None)
        if held_body is None:
            held_body = self.source_object_designator
        try:
            mins, maxs = body_local_aabb(
                held_body,
                use_visual=True,
                apply_shape_scale=True,
            )
            height = float(maxs[2] - mins[2])
            if height > 0.0:
                return height
        except Exception:
            pass
        return 0.0

    def _effective_pour_side(self) -> Arms:
        return self.pour_side or self.arm

    def execute(self) -> None:
        pour_side = self._effective_pour_side()

        from scipy.spatial.transform import Rotation as R

        object_pose = self.object_designator.global_pose
        robot_pose = self.robot.root.global_pose
        bowl_x = float(object_pose.x)
        bowl_y = float(object_pose.y)
        robot_x = float(robot_pose.x)
        robot_y = float(robot_pose.y)

        # Unit vector from robot toward bowl
        approach_x = bowl_x - robot_x
        approach_y = bowl_y - robot_y
        approach_norm = math.hypot(approach_x, approach_y)
        if approach_norm < 1e-6:
            approach_x = 1.0
            approach_y = 0.0
        else:
            approach_x /= approach_norm
            approach_y /= approach_norm

        # Snap approach direction to nearest bowl axis so we never aim at a corner
        bowl_quat = [float(x) for x in object_pose.to_quaternion().to_np()]
        bowl_rot = R.from_quat(bowl_quat)
        bowl_x_axis = bowl_rot.apply([1, 0, 0])
        bowl_y_axis = bowl_rot.apply([0, 1, 0])
        approach_vec = np.array([approach_x, approach_y, 0.0])
        candidates = [bowl_x_axis, -bowl_x_axis, bowl_y_axis, -bowl_y_axis]
        dots = [np.dot(approach_vec, c) for c in candidates]
        best = candidates[int(np.argmax(dots))]
        approach_x = float(best[0])
        approach_y = float(best[1])
        approach_norm = math.hypot(approach_x, approach_y)
        if approach_norm > 1e-6:
            approach_x /= approach_norm
            approach_y /= approach_norm

        robot_right_x = approach_y
        robot_right_y = -approach_x
        side_sign = 1.0 if pour_side == Arms.RIGHT else -1.0
        side_x = side_sign * robot_right_x
        side_y = side_sign * robot_right_y

        side_offset = float(self.pour_side_offset_m) + (
            0.7 * self._held_object_height_m()
        )
        approach_offset = float(self.pour_approach_offset_m)  # tilt_reach not in XY

        target_x = bowl_x + side_x * side_offset - approach_x * approach_offset
        target_y = bowl_y + side_y * side_offset - approach_y * approach_offset
        self.offset_x = float(target_x - bowl_x)
        self.offset_y = float(target_y - bowl_y)
        self.offset_z = float(self.pour_height_m)

        yaw_to_bowl = math.atan2(
            bowl_y - target_y,
            bowl_x - target_x,
        )

        # Tilt angle — direction depends on arm
        angle = 1.85 if pour_side == Arms.RIGHT else -1.85
        self.tilt_degrees = math.degrees(float(angle))

        # Raise Z so cup opening lands at correct height after tilting
        pour_z = float(object_pose.z) + float(self.pour_height_m)

        # Build orientation from yaw_to_bowl, not robot_quat,
        # so roll-tilt is always correct relative to the bowl regardless of robot pose
        base_rot = R.from_euler("z", yaw_to_bowl)
        qx, qy, qz, qw = base_rot.as_quat()  # scipy convention: x,y,z,w

        new_pose = Pose.from_xyz_quaternion(
            pos_x=target_x,
            pos_y=target_y,
            pos_z=pour_z,
            quat_x=qx,
            quat_y=qy,
            quat_z=qz,
            quat_w=qw,
            reference_frame=self.world.root,
        )

        # Left arm: rotate orientation 180° around world Z,
        # position stays untouched
        if pour_side == Arms.LEFT:
            new_mat = new_pose.to_homogeneous_matrix()
            z180_mat = Pose.from_xyz_rpy(yaw=math.pi).to_homogeneous_matrix()
            new_mat[:3, :3] = z180_mat[:3, :3] @ new_mat[:3, :3]
            new_pose = new_mat.to_pose()

        rot = Pose.from_xyz_rpy(pitch=angle)
        rot_new_pose = new_pose.to_homogeneous_matrix() @ rot.to_homogeneous_matrix()

        self.robot.full_body_controlled = True
        self.add_subplan(
            sequential(
                [
                    MoveToolCenterPointMotion(
                        new_pose,
                        self.arm,
                        allow_gripper_collision=True,
                        movement_type=MovementType.CARTESIAN,
                    ),
                    MoveToolCenterPointMotion(
                        rot_new_pose.to_pose(),
                        self.arm,
                        allow_gripper_collision=True,
                        movement_type=MovementType.CARTESIAN,
                    ),
                ],
            )
        ).perform()
