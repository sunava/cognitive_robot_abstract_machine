import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from suturo_resources.suturo_map import load_environment

from demos.pycram_hsrb_demo.simulation_setup import (
    setup_hsrb_suturo,
)
from demos.thesis.simulation_setup import add_box, BoxSpec
from demos.thesis_new.frame_provider import WorldTransformFrameProvider
from demos.thesis_new.rviz import PhaseSequenceRviz
from giskardpy.motion_statechart.context import ExecutionContext
from pycram.datastructures.enums import TorsoState, Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
from pycram.robot_plans import ParkArmsActionDescription, MoveTorsoActionDescription
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

result = setup_hsrb_suturo(load_environment=load_environment, with_viz=True)
world: World
context: ExecutionContext
viz: VizMarkerPublisher

world, context, viz = result.world, result.context, result.viz

if viz is None:
    raise RuntimeError("viz is None although with_viz=True")

with world.modify_world():
    box = add_box(
        world, BoxSpec(name="muh_box", scale_xyz=(0.3, 0.1, 0.1)), tf_frame="/map"
    )
    world.add_connection(
        FixedConnection(
            parent=world.root,
            child=box,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.9, y=1.0, z=0.95
            ),
        )
    )


def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero-length vector")
    return v / n


def aligned_plane_frame(origin, normal, tangent_hint):
    p = np.asarray(origin, dtype=float).reshape(3)
    z = unit(normal)

    th = np.asarray(tangent_hint, dtype=float).reshape(3)
    x = th - (th @ z) * z
    x = unit(x)

    y = unit(np.cross(z, x))
    R = np.column_stack([x, y, z])
    return R, p


class Pose:
    def __init__(self, R=None, p=None):
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)
        self.p = np.zeros(3) if p is None else np.asarray(p, dtype=float).reshape(3)

    def transform_point(self, q_local):
        q_local = np.asarray(q_local, dtype=float).reshape(3)
        return self.p + self.R @ q_local


class FrameProvider:
    def get_pose(self) -> Pose:
        raise NotImplementedError


class FixedFrameProvider(FrameProvider):
    def __init__(self, pose: Pose):
        self._pose = pose

    def get_pose(self) -> Pose:
        return self._pose


class Phase:
    def __init__(self, name, duration_s, local_curve):
        self.name = str(name)
        self.duration_s = float(duration_s)
        self.local_curve = local_curve

    def sample(self, frame_provider: FrameProvider, dt: float, t0: float = 0.0):
        F = frame_provider.get_pose()

        n = max(2, int(np.ceil(self.duration_s / float(dt))) + 1)
        times = np.linspace(t0, t0 + self.duration_s, n)

        tau = (times - t0) / self.duration_s
        pts = np.empty((n, 3), dtype=float)
        for i, u in enumerate(tau):
            pts[i] = F.transform_point(self.local_curve(float(u)))

        return times, pts


class PhaseSequence:
    def __init__(self, phases):
        self.phases = list(phases)

    @property
    def duration_s(self):
        return float(sum(p.duration_s for p in self.phases))

    def sample(self, frame_provider: FrameProvider, dt: float, t0: float = 0.0):
        all_t, all_p, all_id = [], [], []
        t = float(t0)

        for k, ph in enumerate(self.phases):
            tt, pp = ph.sample(frame_provider, dt=dt, t0=t)
            if all_t:
                tt = tt[1:]
                pp = pp[1:]

            all_t.append(tt)
            all_p.append(pp)
            all_id.append(np.full(len(tt), k, dtype=int))
            t += ph.duration_s

        return np.concatenate(all_t), np.vstack(all_p), np.concatenate(all_id)


def ramp(tau, tau_end, d_max):
    if tau <= 0.0:
        return 0.0
    if tau >= tau_end:
        return float(d_max)
    return float(d_max) * (tau / tau_end)


def planar_spiral_xy(tau, r0, r1, cycles):
    r = r0 + (r1 - r0) * tau
    ang = 2.0 * np.pi * cycles * tau
    return np.array([r * np.cos(ang), r * np.sin(ang), 0.0], dtype=float)


def planar_sweep_x(tau, length, cycles):
    s = float(length) * np.sin(2.0 * np.pi * float(cycles) * tau)
    return np.array([s, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class ShearProfile:
    depth_max: float
    depth_ramp_end: float
    shear_amp: float
    shear_cycles: float


@dataclass(frozen=True)
class SpiralProfile:
    r0: float
    r1: float
    cycles: float


@dataclass(frozen=True)
class SweepProfile:
    length: float
    cycles: float


def oscillatory_shear_local_profiled(tau, prof: ShearProfile):
    d = ramp(tau, tau_end=prof.depth_ramp_end, d_max=prof.depth_max)
    s = float(prof.shear_amp) * np.sin(2.0 * np.pi * float(prof.shear_cycles) * tau)
    return np.array([s, 0.0, -d], dtype=float)


def sample_local_curve(local_curve, taus):
    pts = np.empty((len(taus), 3), dtype=float)
    for i, u in enumerate(taus):
        pts[i] = local_curve(float(u))
    return pts


taus = np.linspace(0.0, 1.0, 900)

shear_profiles = [
    (
        "set A: small amp, high freq",
        ShearProfile(
            depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.006, shear_cycles=9.0
        ),
    ),
    (
        "set B: medium amp, medium freq",
        ShearProfile(
            depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.012, shear_cycles=5.0
        ),
    ),
    (
        "set C: large amp, low freq",
        ShearProfile(
            depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.020, shear_cycles=2.5
        ),
    ),
]

spiral_profiles = [
    ("set A: tight spiral", SpiralProfile(r0=0.00, r1=0.10, cycles=5.0)),
    ("set B: medium spiral", SpiralProfile(r0=0.00, r1=0.14, cycles=3.0)),
    ("set C: wide spiral", SpiralProfile(r0=0.00, r1=0.18, cycles=1.8)),
]

sweep_profiles = [
    ("set A: short sweep, high freq", SweepProfile(length=0.06, cycles=4.0)),
    ("set B: medium sweep, medium freq", SweepProfile(length=0.10, cycles=2.0)),
    ("set C: long sweep, low freq", SweepProfile(length=0.14, cycles=1.2)),
]

phase_spiral = Phase(
    name="planar_spiral",
    duration_s=2.0,
    local_curve=lambda tau: planar_spiral_xy(tau, r0=0.00, r1=0.12, cycles=2.5),
)

phase_shear = Phase(
    name="oscillatory_shear",
    duration_s=1.5,
    local_curve=lambda tau: oscillatory_shear_local_profiled(
        tau,
        ShearProfile(
            depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.012, shear_cycles=5.0
        ),
    ),
)

phase_sweep = Phase(
    name="planar_sweep",
    duration_s=1.5,
    local_curve=lambda tau: planar_sweep_x(tau, length=0.10, cycles=2.0),
)

seq = PhaseSequence([phase_spiral, phase_shear, phase_sweep])

R_A, p_A = aligned_plane_frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
R_B, p_B = aligned_plane_frame([0.35, 0.10, 0.20], [0.3, 0.4, 0.85], [0.0, 1.0, 0.0])

prov_A = FixedFrameProvider(Pose(R_A, p_A))
prov_B = FixedFrameProvider(Pose(R_B, p_B))

dt = 0.01
tA, PA, idA = seq.sample(prov_A, dt=dt)
tB, PB, idB = seq.sample(prov_B, dt=dt)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(PA[:, 0], PA[:, 1], PA[:, 2], label="sequence + frame A")
ax.plot(PB[:, 0], PB[:, 1], PB[:, 2], label="sequence + frame B")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("PhaseSequence reused under two different frame alignments")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tA, PA[:, 2])
ax.set_xlabel("t [s]")
ax.set_ylabel("z [m]")
ax.set_title("Sequence timing visible in z(t) for frame A")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.step(tA, idA, where="post")
ax.set_xlabel("t [s]")
ax.set_ylabel("phase index")
ax.set_title("Phase index over time (frame A)")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
for label, prof in shear_profiles:
    curve = lambda tau, prof=prof: oscillatory_shear_local_profiled(tau, prof)
    P = sample_local_curve(curve, taus)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], label=label)
ax.set_xlabel("x (shear)")
ax.set_ylabel("y")
ax.set_zlabel("z (monotone)")
ax.set_title("Same local phase: oscillatory shear (parameter variation)")
ax.legend()
plt.show()

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
for label, prof in shear_profiles:
    curve = lambda tau, prof=prof: oscillatory_shear_local_profiled(tau, prof)
    P = sample_local_curve(curve, taus)
    ax.plot(taus, P[:, 0], label=label)
ax.set_xlabel("tau")
ax.set_ylabel("x")
ax.set_title("Shear component x(tau) under different parameter sets")
ax.legend()
plt.show()

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
for label, prof in shear_profiles:
    curve = lambda tau, prof=prof: oscillatory_shear_local_profiled(tau, prof)
    P = sample_local_curve(curve, taus)
    ax.plot(taus, P[:, 2], label=label)
ax.set_xlabel("tau")
ax.set_ylabel("z")
ax.set_title("Monotone displacement z(tau) under different parameter sets")
ax.legend()
plt.show()

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
for label, prof in spiral_profiles:
    curve = lambda tau, prof=prof: planar_spiral_xy(tau, prof.r0, prof.r1, prof.cycles)
    P = sample_local_curve(curve, taus)
    ax.plot(P[:, 0], P[:, 1], label=label)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Same local phase: planar spiral (parameter variation)")
ax.set_aspect("equal", adjustable="box")
ax.legend()
plt.show()

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
for label, prof in sweep_profiles:
    curve = lambda tau, prof=prof: planar_sweep_x(tau, prof.length, prof.cycles)
    P = sample_local_curve(curve, taus)
    ax.plot(taus, P[:, 0], label=label)
ax.set_xlabel("tau")
ax.set_ylabel("x")
ax.set_title("Same local phase: planar sweep x(tau) (parameter variation)")
ax.legend()
plt.show()


plan = SequentialPlan(
    context,
    ParkArmsActionDescription(Arms.BOTH),
    MoveTorsoActionDescription(TorsoState.HIGH),
)
with simulated_robot:
    plan.perform()


# prov_table = WorldTransformFrameProvider(
#     world=world,
#     source_frame=world.get_body_by_name("muh_box"),
#     root_frame=world.root,
#     make_identity_spatial=make_identity_pose_stamped,
# )

# F_table_in_map = prov_table.get_pose()
# t, P, phase_id = seq.sample(prov_table, dt=0.01)


# Example usage (you fill in objects):
# test_world_transform_identity(world, PoseStamped, cookingTable_body, world.root)
def _to_np_matrix(x):
    if hasattr(x, "to_np"):
        return np.asarray(x.to_np(), dtype=float)
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=float)
    if hasattr(x, "full"):
        return np.asarray(x.full(), dtype=float)
    return np.asarray(x, dtype=float)


def Rp_from_spatial(spatial_pose_or_T):
    if hasattr(spatial_pose_or_T, "to_homogeneous_matrix"):
        T = spatial_pose_or_T.to_homogeneous_matrix()
    else:
        T = spatial_pose_or_T

    if hasattr(T, "casadi_sx"):
        T_np = _to_np_matrix(T.casadi_sx)
    else:
        T_np = _to_np_matrix(T)

    T_np = T_np.reshape(4, 4)
    R = T_np[:3, :3]
    p = T_np[:3, 3]
    return R, p


rf = world.get_body_by_name("muh_box").global_pose
R, p = Rp_from_spatial(rf.evaluate())
F = Pose(R=R, p=p)
print("p =", F.p)
print("R =\n", F.R)
print("det(R) =", np.linalg.det(F.R))
print("R^T R =\n", F.R.T @ F.R)

print("origin_world =", F.transform_point([0.0, 0.0, 0.0]))
print(
    "x_axis_world =",
    F.transform_point([0.1, 0.0, 0.0]) - F.transform_point([0.0, 0.0, 0.0]),
)
print(
    "z_axis_world =",
    F.transform_point([0.0, 0.0, 0.1]) - F.transform_point([0.0, 0.0, 0.0]),
)
prov = FixedFrameProvider(F)
dt = 0.01
tA, PA, idA = seq.sample(prov, dt=dt)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(PA[:, 0], PA[:, 1], PA[:, 2], label="sequence in world (provider=F)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("PhaseSequence under world frame F")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tA, PA[:, 2])
ax.set_xlabel("t [s]")
ax.set_ylabel("z [m]")
ax.set_title("z(t)")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.step(tA, idA, where="post")
ax.set_xlabel("t [s]")
ax.set_ylabel("phase index")
ax.set_title("phase index over time")
plt.show()


dt = 0.01
prov = FixedFrameProvider(F)
tA, PA, idA = seq.sample(prov, dt=dt)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    PA[:, 0], PA[:, 1], PA[:, 2], label="sequence in muh_box frame (mapped to root)"
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("PhaseSequence in world (using extracted F)")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tA, PA[:, 2])
ax.set_xlabel("t [s]")
ax.set_ylabel("z [m]")
ax.set_title("z(t) for sampled sequence")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.step(tA, idA, where="post")
ax.set_xlabel("t [s]")
ax.set_ylabel("phase index")
ax.set_title("phase segmentation over time")
plt.show()


frame_id = world.root
rv = PhaseSequenceRviz(
    PA, idA, frame_id="map", topic="phase_sequence_markers", node=viz.node
)
rv.publish_once()
