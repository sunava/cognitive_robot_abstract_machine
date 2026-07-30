"""
The live-viz bridge state: what a running demo publishes to the viewer.

This module is free of HTTP and of hook installation — it holds a :class:`Bridge`
instance whose snapshot methods run on the *simulation* thread (see
:mod:`cram_viz.live.hooks` for why that matters) and whose ``get_*`` accessors hand
finished, plain-dict snapshots to the HTTP layer.

Node status is where the plan and the statechart differ: coraplex only performs the plan
root (``Plan.perform`` → ``root.perform``); ``ActionNode.notify`` expands its children
but never performs them, so every inner ``PlanNode`` keeps status ``CREATED`` for the
whole run. The real per-step progress lives in the giskardpy motion statechart's life
cycle. ``GiskardExecutable.motion_mappings`` (a ``{MotionNode: Task}`` dict) is the
bridge between the two — the life cycle of each motion node's task is read and
propagated up the plan tree; those statuses are flagged ``derived``.

.. note:: :class:`TaskStatus` mirrors the vocabulary of coraplex's own
   ``TaskStatus`` enum (``coraplex.datastructures.enums.TaskStatus``) but is a
   distinct, local definition: this module must stay importable without coraplex
   installed (see :mod:`cram_viz.live`), so it cannot import that enum directly.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.parse
from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum
from pathlib import Path

from typing_extensions import (
    Any,
    Literal,
    Protocol,
    TYPE_CHECKING,
    TypedDict,
    runtime_checkable,
)

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh

if TYPE_CHECKING:
    from coraplex.plans.executables import GiskardExecutable
    from coraplex.plans.plan import Plan
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


class MoveRequestError(Exception):
    """
    Raised when a viewer-submitted ``/move`` payload is malformed.
    """


class TaskStatus(str, Enum):
    """
    Status vocabulary shared by plan nodes and statechart nodes.

    Members are also valid plain strings (``TaskStatus.SUCCEEDED == "SUCCEEDED"``), so
    the wire format sent to the viewer is unaffected by using this enum internally.
    """

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSE = "PAUSE"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@runtime_checkable
class _DesignatorNodeLike(Protocol):
    """
    Structural type for plan nodes that carry a designator.

    Only coraplex's ``DesignatorNode`` subclasses (``ActionNode``, ``MotionNode``, ...)
    do; the bare ``PlanNode`` base does not.
    """

    designator: Any


@runtime_checkable
class _NamedFieldValue(Protocol):
    """
    Structural type for a designator field value that names the entity it refers to.
    """

    name: Any


#: same palette the onboarder / viewer use, so a live object keeps its colour
PALETTE = [
    "#f3f0ea",
    "#cf5b3a",
    "#b8bcc4",
    "#e7c26a",
    "#7fb069",
    "#5b8cff",
    "#c98bdb",
    "#ff9d6b",
    "#6bd0c0",
    "#d0c86b",
]

#: giskardpy ``LifeCycleValues`` ordinal → its name, for the statechart view
LIFE_CYCLE_NAME = {0: "NOT_STARTED", 1: "RUNNING", 2: "PAUSED", 3: "DONE", 4: "FAILED"}

#: giskardpy ``LifeCycleValues`` ordinal → :class:`TaskStatus`, so the viewer styles
#: plan nodes and statechart nodes with one status palette
LIFE_CYCLE_TO_STATUS = {
    0: TaskStatus.CREATED,
    1: TaskStatus.RUNNING,
    2: TaskStatus.PAUSE,
    3: TaskStatus.SUCCEEDED,
    4: TaskStatus.FAILED,
}

#: for bottom-up aggregation in the plan tree: the higher rank wins
STATUS_RANK = {
    TaskStatus.CREATED: 0,
    TaskStatus.SUCCEEDED: 1,
    TaskStatus.PAUSE: 2,
    TaskStatus.RUNNING: 3,
    TaskStatus.INTERRUPTED: 4,
    TaskStatus.FAILED: 5,
}

#: how long a world binding stays fresh before the bridge re-discovers bodies
REBIND_INTERVAL_SECONDS = 3.0


class MeshObjectEntry(TypedDict):
    """
    Geometry-catalog entry for an object whose mesh file the bridge can serve.
    """

    key: str
    """
    Object body key (its mesh-file basename), e.g. ``"milk.stl"``.
    """

    id: str
    """
    Short display id, the mesh key without its extension.
    """

    kind: Literal["mesh"]
    """
    Discriminant against :class:`BoxObjectEntry`.
    """

    mesh: str
    """
    URL the viewer fetches the mesh bytes from (the ``/mesh`` endpoint).
    """

    format: str
    """
    Mesh file extension (``"stl"``, ``"obj"``, ``"dae"``), lowercased.
    """

    color: str
    """
    Fallback colour, used until the mesh's own material loads.
    """


class BoxObjectEntry(TypedDict):
    """
    Geometry-catalog entry for an object with no known mesh, drawn as a box.
    """

    key: str
    """
    Object body key.
    """

    id: str
    """
    Short display id, the key without its extension.
    """

    kind: Literal["box"]
    """
    Discriminant against :class:`MeshObjectEntry`.
    """

    size: list[float]
    """
    Bounding-box size in metres, ``[x, y, z]``.
    """

    color: str
    """
    Box colour.
    """


#: one geometry-catalog entry, keyed by :attr:`MeshObjectEntry.kind` / :attr:`BoxObjectEntry.kind`
ObjectMetaEntry = MeshObjectEntry | BoxObjectEntry


class WorldStateSnapshot(TypedDict):
    """
    The world snapshot published by :meth:`Bridge.snapshot`.
    """

    seq: int
    """
    Monotonic snapshot counter.
    """

    frames: dict[str, float]
    """
    Actuated joint name → position, radians or metres.
    """

    base: list[float] | None
    """
    Robot base pose ``[x, y, z, qx, qy, qz, qw]``, or ``None`` if there is no robot.
    """

    objects: dict[str, list[float]]
    """
    Object key → pose ``[x, y, z, qx, qy, qz, qw]``.
    """


class PlanNodeEntry(TypedDict):
    """
    One serialized plan node, as produced by :meth:`Bridge._serialize_plan_node`.
    """

    id: str
    """
    Node id, stable for the lifetime of the plan node object.
    """

    parent: str | None
    """
    Parent node's id, or ``None`` for the root.
    """

    kind: str
    """
    Plan-node class name.
    """

    label: str
    """
    Designator class name, or :attr:`kind` if the node has no designator.
    """

    status: str
    """
    Current :class:`TaskStatus` value.
    """

    derived: bool
    """
    Whether :attr:`status` was derived from the motion statechart.
    """


class PlanSnapshot(TypedDict):
    """
    The plan-tree snapshot published by :meth:`Bridge.snapshot_plan`.
    """

    signature: str
    """
    Structure signature: node-id order, unaffected by status changes.
    """

    nodes: list[PlanNodeEntry]
    """
    Every plan node, in depth-first order.
    """


#: One serialized statechart node, as produced by
#: :meth:`Bridge._serialize_chart_structure`. Built with the functional ``TypedDict``
#: form because its wire key ``"class"`` is a Python keyword and cannot name a field in
#: the class-body form used elsewhere in this module.
#:
#: :key id: Node id, stable for the lifetime of the statechart.
#: :key name: Statechart node name.
#: :key class: Statechart node class name.
#: :key parent: Parent node's id, or ``None`` for a root node.
#: :key life: Current life-cycle name (see :data:`LIFE_CYCLE_NAME`).
#: :key observation: Current observation name (``"TRUE"``, ``"FALSE"`` or ``"UNKNOWN"``).
ChartNodeEntry = TypedDict(
    "ChartNodeEntry",
    {
        "id": str,
        "name": str,
        "class": str,
        "parent": str | None,
        "life": str,
        "observation": str,
    },
)


#: One transition edge between two statechart nodes. Built with the functional
#: ``TypedDict`` form because its wire key ``"from"`` is a Python keyword.
#:
#: :key from: Source node id.
#: :key to: Target node id.
#: :key kind: Transition kind name.
ChartEdgeEntry = TypedDict("ChartEdgeEntry", {"from": str, "to": str, "kind": str})


class ChartSnapshot(TypedDict):
    """
    The motion-statechart snapshot published by :meth:`Bridge.observe_chart`.
    """

    signature: str
    """
    Structure signature.
    """

    title: str
    """
    Name of the action whose motion group is executing.
    """

    nodes: list[ChartNodeEntry]
    """
    Every statechart node.
    """

    edges: list[ChartEdgeEntry]
    """
    Every transition edge.
    """


@dataclass(frozen=True)
class MoveRequest:
    """
    One validated object move submitted by the viewer.
    """

    object_key: str
    """
    Key of the object body to move, matching a :attr:`Bridge._bodies` entry.
    """

    position: list[float]
    """
    Target position ``[x, y, z]`` in the world frame.
    """

    quaternion: list[float] | None
    """
    Target orientation ``[x, y, z, w]``, or ``None`` to keep the current orientation.
    """

    final: bool
    """
    Whether this is the drag's final placement (as opposed to a live-drag update).
    """

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MoveRequest:
        """
        Validate and coerce a raw HTTP ``/move`` payload.

        :raises MoveRequestError: if a required field is missing or malformed.
        """
        object_key = payload.get("object")
        if not isinstance(object_key, str) or not object_key:
            raise MoveRequestError("move payload must have a non-empty 'object' key")
        position = cls._as_vector(payload.get("position"), 3, "position")
        quaternion = payload.get("quaternion")
        if quaternion is not None:
            quaternion = cls._as_vector(quaternion, 4, "quaternion")
        return cls(
            object_key=object_key,
            position=position,
            quaternion=quaternion,
            final=bool(payload.get("final", False)),
        )

    @staticmethod
    def _as_vector(value: Any, length: int, field_name: str) -> list[float]:
        """
        Coerce a payload field into a fixed-length list of floats.
        """
        if not isinstance(value, list) or len(value) != length:
            raise MoveRequestError(
                "move payload's '%s' must be a %d-element list" % (field_name, length)
            )
        try:
            return [float(component) for component in value]
        except (TypeError, ValueError) as error:
            raise MoveRequestError(
                "move payload's '%s' must contain numbers" % field_name
            ) from error


@dataclass
class MoveQueue:
    """
    Thread-safe hand-off of viewer-submitted moves to the simulation thread.
    """

    _pending: list[MoveRequest] = field(default_factory=list)
    """
    Moves queued by HTTP threads, drained by :meth:`drain`.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`_pending`.
    """

    def push(self, move: MoveRequest) -> None:
        """
        Queue a move (called on an HTTP thread).
        """
        with self._lock:
            self._pending.append(move)

    def drain(self) -> list[MoveRequest]:
        """
        Take and clear every queued move (called on the simulation thread).
        """
        with self._lock:
            pending, self._pending = self._pending, []
        return pending


def _pose_as_position_quaternion(body: Body) -> list[float]:
    """
    Return a body's world pose as ``[x, y, z, qx, qy, qz, qw]``.
    """
    pose = body.global_pose
    translation = pose.to_position().to_np().flatten()
    quaternion = pose.to_quaternion().to_np().flatten()
    return [round(float(value), 5) for value in (*translation[:3], *quaternion[:4])]


@dataclass
class Bridge:
    """
    Shared state between the running demo and the viewer.

    All world reads and writes happen on the simulation thread (the tick hook); the HTTP
    handlers only ever read the finished snapshot dicts under :attr:`_lock`.
    """

    world: World | None = None
    """
    The executing world, captured by the tick hook on its first call.
    """

    robot: AbstractRobot | None = None
    """
    The robot annotation of :attr:`world`, re-discovered on every bind.
    """

    seq: int = 0
    """
    Monotonic snapshot counter so the viewer can skip unchanged states.
    """

    state: WorldStateSnapshot = field(
        default_factory=lambda: WorldStateSnapshot(
            seq=0, frames={}, base=None, objects={}
        )
    )
    """
    The newest world snapshot in the trajectory-frame format.
    """

    object_meta: list[ObjectMetaEntry] = field(default_factory=list)
    """
    Geometry catalog for the viewer: one entry per loose object.
    """

    plan_state: PlanSnapshot = field(
        default_factory=lambda: PlanSnapshot(signature="", nodes=[])
    )
    """
    The newest plan-tree snapshot (see :meth:`snapshot_plan`).
    """

    chart_state: ChartSnapshot = field(
        default_factory=lambda: ChartSnapshot(
            signature="", title="", nodes=[], edges=[]
        )
    )
    """
    The newest motion-statechart snapshot (see :meth:`observe_chart`).
    """

    _connections: list[Any] = field(default_factory=list)
    """
    Actuated world connections whose positions are published as frames.
    """

    _bodies: dict[str, Body] = field(default_factory=dict)
    """
    Published bodies by mesh key; ``__base__`` is the robot root.
    """

    _last_bind: float = 0.0
    """
    Timestamp of the last world discovery (see ``REBIND_INTERVAL_SECONDS``).
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every snapshot dict that the HTTP layer reads.
    """

    _moves: MoveQueue = field(default_factory=MoveQueue)
    """
    Object moves queued by the viewer, applied on the simulation thread.
    """

    _mesh_files: dict[str, str] = field(default_factory=dict)
    """
    Mesh basename (lowercase) → absolute path, filled by the mesh hook.
    """

    _mesh_serve: dict[str, str] = field(default_factory=dict)
    """
    Object key → absolute mesh path served via the ``/mesh`` endpoint.
    """

    _plan: Plan | None = None
    """
    The coraplex plan captured by the ``Plan.perform`` hook.
    """

    _chart: MotionStatechart | None = None
    """
    The motion statechart the executor is currently ticking.
    """

    _chart_structure: dict[str, Any] | None = None
    """
    Serialized structure of :attr:`_chart`, rebuilt when it changes.
    """

    _chart_title: str = ""
    """
    Name of the action whose motion group is executing.
    """

    _motion_tasks: dict[int, Any] = field(default_factory=dict)
    """
    ``id(MotionNode)`` → giskard task of the live motion group.
    """

    _frozen: dict[int, TaskStatus] = field(default_factory=dict)
    """
    ``id(PlanNode)`` → final status pinned when its motion group ended.
    """

    _ticks: int = 0
    """
    Tick counter used to throttle the plan snapshot.
    """

    _last_life_cycle: list[int] | None = None
    """
    Life-cycle vector of the last published chart snapshot.
    """

    # %% viewer -> world -------------------------------------------------------
    def queue_move(self, move: MoveRequest) -> None:
        """
        Queue a validated object move from the viewer (called on an HTTP thread).
        """
        self._moves.push(move)

    def apply_moves(self) -> None:
        """
        Apply queued object moves to the world.

        Called from the tick hook — the simulation thread is the only place that may
        write to the world.
        """
        moves = self._moves.drain()
        if not moves or self.world is None:
            return
        for move in moves:
            body = self._bodies.get(move.object_key)
            if body is None:
                continue
            self._apply_move(move, body)

    def _apply_move(self, move: MoveRequest, body: Body) -> None:
        """
        Write one viewer move into the world.

        Only free-floating (:class:`Connection6DoF`) objects are draggable.
        Objects rigidly fixed to furniture — e.g. a spoon on a drawer that
        must ride along when the drawer opens — keep their ``FixedConnection``
        and are left untouched (a fixed connection has no settable origin).

        .. note:: The object is deliberately *not* re-parented here. That is a
           structural change of the kinematic tree (``modify_world`` + forward
           kinematics recompile) and running it inside the tick hook while a
           giskard goal is live hangs the executor. Re-parenting must happen
           as its own plan step between motions; the plain pose write already
           makes ``body.global_pose`` correct, which is all the plan's
           navigate/pick reachability needs.
        """
        connection = body.parent_connection
        if not isinstance(connection, Connection6DoF):
            logger.info(
                "%s is fixed (%s) — not draggable, skipping",
                move.object_key,
                type(connection).__name__,
            )
            return
        position = move.position
        orientation = (
            RotationMatrix.from_quaternion(
                Quaternion(
                    w=move.quaternion[3],
                    x=move.quaternion[0],
                    y=move.quaternion[1],
                    z=move.quaternion[2],
                )
            )
            if move.quaternion
            else body.global_pose.to_rotation_matrix()
        )
        world_T_object = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            Point3(x=position[0], y=position[1], z=position[2]),
            orientation,
            reference_frame=self.world.root,
        )
        # ``origin`` is parent-relative; express the target in the parent
        # frame (a no-op while the parent is the world root)
        parent_T_object = (
            connection.parent.global_pose.to_homogeneous_matrix().inverse()
            @ world_T_object
        )
        connection.origin = parent_T_object
        logger.info(
            "moved %s -> world (%.3f, %.3f, %.3f) [final=%s]",
            move.object_key,
            position[0],
            position[1],
            position[2],
            move.final,
        )

    # %% world discovery -------------------------------------------------------
    def _bind(self) -> None:
        """
        Discover the robot, joints and loose objects of the current world.

        Re-run periodically because demos modify their world (objects get spawned and
        removed mid-run).
        """
        world = self.world
        if world is None:
            return
        self._last_bind = time.time()
        robots = world.get_semantic_annotations_by_type(AbstractRobot)
        self.robot = robots[0] if robots else None
        self._connections = self._actuated_connections(world)
        bodies: dict[str, Body] = {}
        if self.robot is not None:
            bodies["__base__"] = self.robot.root
        try:
            # loose objects by convention: bodies named like mesh files
            for body in world.bodies:
                basename = str(body.name).split("/")[-1]
                if basename.lower().endswith((".stl", ".obj", ".dae")):
                    bodies[basename] = body
        except RuntimeError as error:
            # the world is mid-modification (a body is being spawned or
            # removed, mutating the kinematic structure this scan walks):
            # keep the previous catalog instead of publishing an empty one —
            # the viewer would otherwise hide every object it already shows
            logger.debug("body scan skipped this bind: %s", error)
            for key, body in self._bodies.items():
                bodies.setdefault(key, body)
        self._bodies = bodies
        self._build_object_meta(bodies)

    @staticmethod
    def _actuated_connections(world: World) -> list[ActiveConnection1DOF]:
        """
        All 1-DOF connections — the joints published as trajectory frames.
        """
        return [
            connection
            for connection in world.connections or []
            if isinstance(connection, ActiveConnection1DOF)
        ]

    def _build_object_meta(self, bodies: dict[str, Body]) -> None:
        """
        Rebuild the geometry catalog the viewer spawns live objects from.

        Each object gets either a mesh URL (served by the bridge) or a box size, so
        objects the viewer does not know yet can appear mid-run.
        """
        catalog: list[ObjectMetaEntry] = []
        serve: dict[str, str] = {}
        for index, (key, body) in enumerate(
            item for item in bodies.items() if item[0] != "__base__"
        ):
            color = PALETTE[index % len(PALETTE)]
            object_id = Path(key).stem
            mesh_path = self._mesh_files.get(key.lower())
            if mesh_path and Path(mesh_path).is_file():
                serve[key] = mesh_path
                mesh_entry: MeshObjectEntry = {
                    "key": key,
                    "id": object_id,
                    "kind": "mesh",
                    "mesh": "/mesh?key=" + urllib.parse.quote(key),
                    "format": Path(key).suffix.lstrip(".").lower(),
                    "color": color,
                }
                catalog.append(mesh_entry)
            else:
                box_entry: BoxObjectEntry = {
                    "key": key,
                    "id": object_id,
                    "kind": "box",
                    "size": self._box_size(body) or [0.06, 0.06, 0.12],
                    "color": color,
                }
                catalog.append(box_entry)
        self._mesh_serve = serve
        with self._lock:
            self.object_meta = catalog

    @staticmethod
    def _box_size(body: Body) -> list[float] | None:
        """
        Best-effort bounding-box size of a body's visual shape in metres.
        """
        for shape_collection in (body.visual, body.collision):
            for shape in shape_collection.shapes:
                if isinstance(shape, (Box, Mesh)):
                    scale = shape.scale
                    return [
                        round(float(scale.x), 4),
                        round(float(scale.y), 4),
                        round(float(scale.z), 4),
                    ]
        return None

    # %% world snapshot ---------------------------------------------------------
    def snapshot(self) -> None:
        """
        Publish the world's joints, base pose and object poses.

        Runs on the simulation thread; rebinds the world periodically so mid-run spawns
        show up.
        """
        if self.world is None:
            return
        if time.time() - self._last_bind > REBIND_INTERVAL_SECONDS:
            self._bind()
        frames = {
            str(connection.name): round(float(connection.position), 5)
            for connection in self._connections
        }
        base_pose: list[float] | None = None
        object_poses: dict[str, list[float]] = {}
        for name, body in self._bodies.items():
            if name == "__base__":
                base_pose = _pose_as_position_quaternion(body)
            else:
                object_poses[name] = _pose_as_position_quaternion(body)
        with self._lock:
            self.seq += 1
            self.state = {
                "seq": self.seq,
                "frames": frames,
                "base": base_pose,
                "objects": object_poses,
            }

    def get_state(self) -> WorldStateSnapshot:
        """
        The newest world snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return dict(self.state)

    # %% plan tree ----------------------------------------------------------------
    def bind_motion_group(self, executable: GiskardExecutable) -> None:
        """
        Remember which plan motion node maps to which statechart task.

        Called when a ``GiskardExecutable`` is about to run, so the plan tree can show
        live per-step progress.
        """
        for node, task in (executable.motion_mappings or {}).items():
            self._motion_tasks[id(node)] = task
        self._chart_title = self._motion_group_title(executable)

    @staticmethod
    def _motion_group_title(executable: GiskardExecutable) -> str:
        """
        Name of the action the motion group belongs to, or ``''``.
        """
        for node in executable.motion_mappings or {}:
            action_node = node.parent_action_node
            if action_node is not None and action_node.designator is not None:
                return type(action_node.designator).__name__
        return ""

    def freeze_motion_group(
        self, executable: GiskardExecutable, status: TaskStatus
    ) -> None:
        """
        Pin the final status of a finished motion group.

        Reading the tasks' life cycle afterwards is not reliable — the executor cleans
        its nodes up.
        """
        for node in executable.motion_mappings or {}:
            self._frozen[id(node)] = status
            self._motion_tasks.pop(id(node), None)
        for condition in (
            executable.pre_condition_node,
            executable.post_condition_node,
        ):
            if condition is not None:
                self._frozen[id(condition)] = status

    def _live_motion_status(self, node: Any) -> TaskStatus | None:
        """
        Status of one plan node from the statechart, or None.
        """
        task = self._motion_tasks.get(id(node))
        if task is not None:
            return LIFE_CYCLE_TO_STATUS.get(int(task.life_cycle_state))
        return self._frozen.get(id(node))

    def snapshot_plan(self) -> None:
        """
        Serialize the plan tree with per-node execution status.

        A node's own status wins while it says something; otherwise the live statechart
        status is used, else the aggregate of the children (running/failed bubble up; a
        parent whose children are only partly done reads as running, not succeeded).
        """
        plan = self._plan
        if plan is None:
            return
        try:
            root = plan.root
        except ValueError:
            # ``Plan.root`` unpacks "exactly one parentless node"; the plan is
            # mid-mutation and momentarily has zero or more than one — next tick
            return
        nodes: list[PlanNodeEntry] = []
        order: list[str] = []
        self._serialize_plan_node(root, None, nodes, order)
        with self._lock:
            self.plan_state = {"signature": "|".join(order), "nodes": nodes}

    def _serialize_plan_node(
        self,
        node: Any,
        parent_id: str | None,
        nodes: list[PlanNodeEntry],
        order: list[str],
    ) -> TaskStatus:
        """
        Serialize one plan node and its subtree; returns the node's status.
        """
        node_id = "p%d" % id(node)
        designator = node.designator if isinstance(node, _DesignatorNodeLike) else None
        own_status = TaskStatus(node.status.name)
        entry: PlanNodeEntry = {
            "id": node_id,
            "parent": parent_id,
            "kind": type(node).__name__,
            "label": (
                type(designator).__name__
                if designator is not None
                else type(node).__name__
            ),
            "status": own_status,
            "derived": False,
        }
        self._add_designator_metadata(entry, designator)
        nodes.append(entry)
        order.append(node_id)

        child_best, children, done = TaskStatus.CREATED, 0, 0
        for child in node.children:
            child_status = self._serialize_plan_node(child, node_id, nodes, order)
            child_best = self._max_status(child_best, child_status)
            children += 1
            if child_status == TaskStatus.SUCCEEDED:
                done += 1
        if own_status == TaskStatus.CREATED:
            if child_best == TaskStatus.SUCCEEDED and done < children:
                child_best = TaskStatus.RUNNING
            derived = self._live_motion_status(node) or (
                child_best if child_best != TaskStatus.CREATED else None
            )
            if derived:
                entry["status"] = derived
                entry["derived"] = True
        return entry["status"]

    def _add_designator_metadata(
        self, entry: PlanNodeEntry, designator: Any | None
    ) -> None:
        """
        Add arm and target-object info from a node's designator, if any.
        """
        if designator is None:
            return
        arm = self._designator_arm(designator)
        if arm is not None:
            entry["arm"] = str(arm)
        target = self._designator_target(designator)
        if target:
            entry["target"] = target

    @staticmethod
    def _designator_arm(designator: Any) -> Any | None:
        """
        A designator's arm field, whichever name it declares.

        Coraplex designator subclasses are heterogeneous here: some declare
        ``arm`` (single-arm actions), others ``arms`` (bimanual actions).
        """
        field_names = {declared.name for declared in dataclass_fields(designator)}
        if "arm" in field_names:
            return designator.arm
        if "arms" in field_names:
            return designator.arms
        return None

    @staticmethod
    def _max_status(first: TaskStatus, second: TaskStatus) -> TaskStatus:
        """
        The higher-ranked of two statuses (see ``STATUS_RANK``).
        """
        if STATUS_RANK.get(first, 0) >= STATUS_RANK.get(second, 0):
            return first
        return second

    def _designator_target(self, designator: Any) -> str | None:
        """
        Name of the published object a designator refers to, if any.
        """
        known = set(self._bodies)
        for value in vars(designator).values():
            if not isinstance(value, _NamedFieldValue):
                continue
            basename = str(value.name).split("/")[-1]
            if basename in known:
                return basename
        return None

    def get_plan(self) -> PlanSnapshot:
        """
        The newest plan snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return dict(self.plan_state)

    # %% motion statechart -----------------------------------------------------------
    def observe_chart(self, chart: MotionStatechart | None) -> None:
        """
        Publish the executing statechart's structure and node states.

        Called from the tick hook. The structure is re-serialized only when the executor
        compiled a new chart; the life-cycle / observation vectors are cheap and
        refreshed whenever they change.
        """
        if chart is None:
            return
        if chart is not self._chart or self._chart_structure is None:
            self._chart = chart
            self._chart_structure = self._serialize_chart_structure(chart)
            self._last_life_cycle = None
        structure = self._chart_structure
        life_cycle = [
            int(chart.life_cycle_state.data[index]) for index in structure["indices"]
        ]
        observations = [
            float(chart.observation_state.data[index]) for index in structure["indices"]
        ]
        if life_cycle == self._last_life_cycle:
            return
        self._last_life_cycle = life_cycle
        nodes = []
        for position, node in enumerate(structure["nodes"]):
            entry = dict(node)
            entry["life"] = LIFE_CYCLE_NAME.get(
                life_cycle[position], str(life_cycle[position])
            )
            entry["observation"] = self._observation_name(observations[position])
            nodes.append(entry)
        with self._lock:
            self.chart_state = {
                "signature": structure["signature"],
                "title": self._chart_title,
                "nodes": nodes,
                "edges": structure["edges"],
            }

    @staticmethod
    def _observation_name(observation: float) -> str:
        """
        Trinary observation value → name (0 false, 0.5 unknown, 1 true).
        """
        if observation >= 0.75:
            return "TRUE"
        if observation <= 0.25:
            return "FALSE"
        return "UNKNOWN"

    @staticmethod
    def _serialize_chart_structure(chart: MotionStatechart) -> dict[str, Any]:
        """
        Nodes and transition edges of a statechart, as plain dicts.

        A statechart always has at least one node once it exists, so this never needs to
        report "no structure" — :meth:`observe_chart` already skips ``None`` charts.
        """
        nodes: list[dict[str, Any]] = []
        indices: list[int] = []
        for node in chart.nodes:
            parent_index = node.parent_node_index
            nodes.append(
                {
                    "id": "s%d" % node.index,
                    "name": node.name,
                    "class": type(node).__name__,
                    "parent": (
                        ("s%d" % parent_index) if parent_index is not None else None
                    ),
                }
            )
            indices.append(node.index)
        edges = []
        for source, target, transition in chart.rx_graph.edge_index_map().values():
            edges.append(
                {
                    "from": "s%d" % chart.rx_graph.get_node_data(source).index,
                    "to": "s%d" % chart.rx_graph.get_node_data(target).index,
                    "kind": transition.kind.name,
                }
            )
        signature = "|".join(node["id"] + ":" + node["name"] for node in nodes)
        return {
            "nodes": nodes,
            "edges": edges,
            "indices": indices,
            "signature": signature,
        }

    def get_chart(self) -> ChartSnapshot:
        """
        The newest statechart snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return dict(self.chart_state)

    # %% http support -----------------------------------------------------------------
    def get_objects(self) -> list[ObjectMetaEntry]:
        """
        The newest geometry catalog (safe to call from HTTP threads).
        """
        with self._lock:
            return list(self.object_meta)

    def get_info(self) -> dict[str, Any]:
        """
        Summary of the bridge's attachment state (safe to call from HTTP threads).
        """
        with self._lock:
            return {
                "running": self.world is not None,
                "robot": type(self.robot).__name__ if self.robot else None,
                "objects": [key for key in self._bodies if key != "__base__"],
                "movable": True,
                "plan": bool(self.plan_state.get("nodes")),
                "chart": bool(self.chart_state.get("nodes")),
                "seq": self.seq,
            }

    def get_mesh_path(self, key: str) -> str | None:
        """
        Absolute path of the mesh file served for ``key``, or ``None``.
        """
        return self._mesh_serve.get(key)
