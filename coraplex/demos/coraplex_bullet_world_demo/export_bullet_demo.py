"""Run the coraplex bullet-world demo (PR2 in the apartment kitchen, transport
milk/bowl/spoon) and capture a per-tick trajectory for browser playback.

Mirrors export_tracy_trajectory.py: hook the giskardpy Executor.tick, snapshot
every movable connection's scalar position plus the moved objects' world poses,
run the plan, and dump a raw JSON. Progress is printed verbosely and flushed so
a background run is observable.

    ~/.virtualenvs/action-cram/bin/python export_bullet_demo.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bullet_demo_raw.json")


def log(*a):
    print(*a, flush=True)


t0 = time.time()
log("[%.1fs] importing stack..." % (time.time() - t0))

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl, Spoon, Drawer, Handle,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from giskardpy.executor import Executor

# --- force the base to DRIVE (with collision avoidance) instead of teleporting.
# In SIMULATED mode MoveMotion normally emits SetOdometry (instant base-pose
# set). Override it to emit a driven CartesianPose on the base so giskardpy
# plans a real, collision-free path that we can capture per tick.
from coraplex.robot_plans.motions.navigation import MoveMotion
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose

def _driven_move_chart(self):
    return CartesianPose(
        root_link=self.world.root,
        tip_link=self.robot.root,
        goal_pose=self.target,
    )
MoveMotion._motion_chart = property(_driven_move_chart)

# Collision avoidance ONLY for navigation: enabling it globally aborts the picks
# (the arms deliberately reach into cabinets). So wrap the motion-chart build and
# switch collision avoidance on only for executables that contain a MoveMotion —
# the base then drives around furniture, while pick/place stay unconstrained.
from coraplex.plans.executables import GiskardExecutable
_orig_msc = GiskardExecutable.motion_state_chart.fget
def _msc_navonly(self):
    try:
        has_nav = any(isinstance(m, MoveMotion) for m in self.motion_mappings.values())
    except Exception:
        has_nav = False
    prev = GiskardExecutable.collision_avoidance
    GiskardExecutable.collision_avoidance = bool(has_nav)
    try:
        return _orig_msc(self)
    finally:
        GiskardExecutable.collision_avoidance = prev
GiskardExecutable.motion_state_chart = property(_msc_navonly)

log("[%.1fs] building world..." % (time.time() - t0))
world = setup_world()

spoon = STLParser(os.path.join(HERE, "..", "..", "resources", "objects", "spoon.stl")).parse()
bowl = STLParser(os.path.join(HERE, "..", "..", "resources", "objects", "bowl.stl")).parse()

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(2.4, 2.2, 1, reference_frame=world.root),
    )
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"),
        child=spoon.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(-0.05, -0.05, 0),
    )
    world.merge_world(spoon, connection)

log("[%.1fs] world built. PR2 + reasoning..." % (time.time() - t0))
pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=None)

with world.modify_world():
    WorldReasoner(world).reason()
    world.add_semantic_annotations([
        Bowl(root=world.get_body_by_name("bowl.stl")),
        Spoon(root=world.get_body_by_name("spoon.stl")),
    ])
    world.add_semantic_annotation_recursively(
        Drawer(
            root=world.get_body_by_name("cabinet10_drawer_top"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
        )
    )
context.evaluate_conditions = False
log("[%.1fs] reasoning done." % (time.time() - t0))


# ---- what to capture --------------------------------------------------------
def movable_connections(w):
    out = []
    for c in getattr(w, "connections", None) or []:
        if hasattr(c, "position"):
            try:
                float(c.position)
                out.append(c)
            except Exception:
                pass
    return out

conns = movable_connections(world)
log("[%.1fs] %d movable connections" % (time.time() - t0, len(conns)))

# objects to track + the PR2 base (OmniDrive has no scalar .position, so its
# world pose must be captured explicitly or the robot stays at the origin)
OBJECTS = ["milk.stl", "bowl.stl", "spoon.stl"]
BASE_BODY = "base_footprint"
obj_bodies = {}
for n in OBJECTS + [BASE_BODY]:
    try:
        obj_bodies[n] = world.get_body_by_name(n)
    except Exception as e:
        log("  (no body %s: %s)" % (n, e))


def pose_xyzq(body):
    p = body.global_pose
    t = p.to_position().to_np().flatten()
    q = p.to_quaternion().to_np().flatten()   # x,y,z,w
    return [round(float(v), 5) for v in (t[0], t[1], t[2], q[0], q[1], q[2], q[3])]


frames = []          # list of {conn_name: pos}
obj_frames = []      # list of {obj_name: [x,y,z, qx,qy,qz,qw]}
_last = [time.time()]


def snap():
    fr = {}
    for i, c in enumerate(conns):
        try:
            fr[str(getattr(c, "name", i))] = round(float(c.position), 5)
        except Exception:
            pass
    frames.append(fr)
    of = {}
    for n, b in obj_bodies.items():
        try:
            of[n] = pose_xyzq(b)
        except Exception:
            pass
    obj_frames.append(of)
    if len(frames) % 200 == 0:
        log("    ...%d frames (%.1fs)" % (len(frames), time.time() - t0))


orig_tick = Executor.tick
def rec_tick(self, *a, **k):
    r = orig_tick(self, *a, **k)
    snap()
    return r
Executor.tick = rec_tick
log("[%.1fs] hooked Executor.tick" % (time.time() - t0))

# ---- the plan (same as demo.py) with per-action segment markers -------------
segments = []
ACTIONS = [
    ("park",        ParkArmsAction(Arms.BOTH)),
    ("torso_up",    MoveTorsoAction(TorsoState.HIGH)),
    ("transport_milk", TransportAction(
        world.get_body_by_name("milk.stl"),
        Pose.from_xyz_rpy(4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root), Arms.LEFT)),
    ("transport_bowl", TransportAction(
        world.get_body_by_name("bowl.stl"),
        Pose.from_xyz_rpy(5, 3.3, 0.75, yaw=1.57, reference_frame=world.root), Arms.LEFT)),
    ("transport_spoon", TransportAction(
        world.get_body_by_name("spoon.stl"),
        Pose.from_xyz_rpy(5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root), Arms.LEFT,
        GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, pr2.left_arm.end_effector))),
]

log("[%.1fs] performing plan (driven base, no global collision avoidance)..." % (time.time() - t0))
with simulated_robot:
    for name, action in ACTIONS:
        start = len(frames)
        ta = time.time()
        try:
            sequential([action], context=context).plan.perform()
            log("[%.1fs] %-16s done  (+%d frames, %.1fs)" %
                (time.time() - t0, name, len(frames) - start, time.time() - ta))
        except Exception as e:
            import traceback
            log("[%.1fs] %-16s FAILED: %s" % (time.time() - t0, name, e))
            traceback.print_exc()
        segments.append({"step": name, "start": start, "end": len(frames)})

log("[%.1fs] TOTAL %d frames" % (time.time() - t0, len(frames)))
json.dump({"frames": frames, "obj_frames": obj_frames, "segments": segments,
           "objects": OBJECTS, "n_conns": len(conns)},
          open(OUT, "w"))
log("wrote %s" % OUT)
