#!/usr/bin/env python3
"""
demo.py — turn a coraplex demo into a self-contained web-viewer scene.

Runs the demo file *unmodified* under instrumentation and emits a scene bundle:

    static/scenes/<name>/
        scene.json         models, robot parts, objects, segments, targets
        trajectory.json    per-tick joints + robot base + object world poses
        <model>.urdf       package:// resolved & rewritten
        meshes/...         all meshes + textures the scene needs

What the hooks capture while the demo runs:
  - every package:// asset resolution and every URDF/STL the world loads
  - per-tick positions of all movable connections (giskardpy Executor.tick)
  - world pose of the robot base and of every loose (STL) object
  - one segment per executed plan ActionNode, with nesting depth
  - the robot's semantic annotation: base body, arms, end-effector link sets

Usage (interpreter needs the CRAM stack, e.g. the action-cram venv):
    cram-viz-onboard \
        ~/actions_cram/cognitive_robot_abstract_machine/coraplex/demos/coraplex_bullet_world_demo/demo.py \
        --name pr2_kitchen
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import runpy
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Any

from cram_viz import paths
from cram_viz.onboard.bundle_urdf import BundleReport, bundle_urdf

if TYPE_CHECKING:
    from coraplex.plans.designator import Designator
    from coraplex.plans.executables import Executable
    from coraplex.plans.plan_node import ActionNode, PlanNode
    from giskardpy.executor import Executor
    from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm, EndEffector
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
    from semantic_digital_twin.world_description.world_entity import Body


logger = logging.getLogger(__name__)


@dataclass
class Timer:
    """
    Wall-clock stopwatch used to prefix progress lines with elapsed time.
    """

    start: float
    """
    The ``time.time()`` value this timer was created at.
    """

    def log(self, *parts: object) -> None:
        """
        Log one progress line prefixed with the elapsed time since start.
        """
        logger.info(
            "[%6.1fs] %s", time.time() - self.start, " ".join(str(part) for part in parts)
        )


@dataclass(frozen=True)
class ObjectPalette:
    """
    Cycling color palette assigned to scene objects, by spawn order.
    """

    colors: tuple[str, ...] = (
        "#f3f0ea",
        "#cf5b3a",
        "#b8bcc4",
        "#e7c26a",
        "#7fb069",
        "#5b8cff",
    )
    """
    Hex colors cycled through in spawn order.
    """

    def color_for(self, index: int) -> str:
        """
        The color assigned to the object at this spawn index.
        """
        return self.colors[index % len(self.colors)]


PALETTE = ObjectPalette()


# %% recorder
@dataclass
class Recorder:
    """
    Records one demo run: assets, per-tick motion and the executed plan.

    .. note:: The CRAM-stack imports inside the ``install_*`` hook methods are
       intentionally local: importing coraplex/giskardpy/semantic_digital_twin
       at module level would make this module require the full CRAM stack
       just to import it, which would block unit-testing its pure helper
       functions (:func:`moved`, :func:`derive_segments`, ...) in isolation.
       This is one of the documented exceptions to the imports-at-top rule.
    """

    timer: Timer
    """
    Stopwatch used to prefix this recorder's progress lines.
    """

    resolutions: dict[str, str] = field(default_factory=dict)
    """
    Every ``uri -> resolved path`` mapping seen while the demo ran.
    """

    urdf_sources: list[str] = field(default_factory=list)
    """
    URDF/xacro files the world was built from.
    """

    stl_sources: list[str] = field(default_factory=list)
    """
    Loose object mesh files the world loaded.
    """

    frames: list[dict[str, float]] = field(default_factory=list)
    """
    One ``{connection_name: position}`` dict per recorded tick.
    """

    base_frames: list[list[float] | None] = field(default_factory=list)
    """
    One ``[x, y, z, qx, qy, qz, qw]`` (or ``None``) per recorded tick, for the robot
    base.
    """

    obj_frames: list[dict[str, list[float]]] = field(default_factory=list)
    """
    One ``{object_name: pose7}`` dict per recorded tick.
    """

    actions: list[dict[str, Any]] = field(default_factory=list)
    """
    One ``{action, arm, target}`` dict per parsed plan action, in plan order.
    """

    plan_nodes: list[ActionNode] = field(default_factory=list)
    """
    The actual ``ActionNode`` objects parsed while the plan ran.
    """

    world: World | None = None
    """
    The world the demo built, bound on the first recorded tick.
    """

    robot: AbstractRobot | None = None
    """
    The robot semantic annotation found in the world, if any.
    """

    control_dt: float | None = None
    """
    The controller's simulated seconds per tick.
    """

    _conns: list[ActiveConnection1DOF] | None = None
    """
    The world's actuated (1-DOF) connections, bound on the first tick.
    """

    _bodies: dict[str, Body] | None = None
    """
    ``{object_name: body}``, including ``"__base__"`` for the robot root.
    """

    # %% asset hooks
    def install_asset_hooks(self) -> None:
        """
        Record every asset resolution so the bundler can copy the files.
        """
        from semantic_digital_twin.adapters.mesh import STLParser
        from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
        from semantic_digital_twin.adapters.urdf import URDFParser

        recorder = self

        original_resolve = PackageUriResolver.resolve

        def _resolve(self, uri: str) -> str:
            """
            Resolve as usual, but remember the uri -> path mapping.
            """
            resolved_path = original_resolve(self, uri)
            recorder.resolutions[uri] = resolved_path
            return resolved_path

        PackageUriResolver.resolve = _resolve

        original_from_file = URDFParser.from_file.__func__

        def _from_file(cls, file_path: str, **kwargs: Any) -> URDFParser:
            """
            Parse as usual, but remember this URDF/xacro source file.
            """
            if file_path not in recorder.urdf_sources:
                recorder.urdf_sources.append(file_path)
            return original_from_file(cls, file_path, **kwargs)

        URDFParser.from_file = classmethod(_from_file)

        original_stl_init = STLParser.__init__

        def _stl_init(self, file_path: str, *args: Any, **kwargs: Any) -> None:
            """
            Init as usual, but remember this loose object's mesh file.
            """
            if file_path not in recorder.stl_sources:
                recorder.stl_sources.append(file_path)
            return original_stl_init(self, file_path, *args, **kwargs)

        STLParser.__init__ = _stl_init

    # %% trajectory hook
    def install_tick_hook(self) -> None:
        """
        Wrap Executor.tick so every simulation step is snapshotted.
        """
        from giskardpy.executor import Executor

        recorder = self
        original_tick = Executor.tick

        def _tick(self, *args: Any, **kwargs: Any) -> None:
            """
            Run the real tick, then record its resulting world state.
            """
            result = original_tick(self, *args, **kwargs)
            recorder._snap(self)
            return result

        Executor.tick = _tick

    def _lazy_bind(self, executor: Executor) -> None:
        """
        Bind to the executor's world and locate the robot + recordable objects the first
        time a tick fires (the world doesn't exist any earlier).
        """
        from semantic_digital_twin.exceptions import WorldEntityNotFoundError
        from semantic_digital_twin.robots.robot_parts import AbstractRobot
        from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

        self.world = executor.context.world
        self.control_dt = executor.context.qp_controller_config.control_dt

        robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
        self.robot = robots[0] if robots else None
        bodies: dict[str, Body] = {}
        if self.robot is not None:
            bodies["__base__"] = self.robot.root
        for stl_source in self.stl_sources:
            name = os.path.basename(stl_source)
            try:
                bodies[name] = self.world.get_body_by_name(name)
            except WorldEntityNotFoundError:
                continue
        self._bodies = bodies
        self._conns = [
            connection
            for connection in self.world.connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
        self.timer.log(
            "bound: robot=%s, %d movable connections, objects=%s"
            % (
                type(self.robot).__name__ if self.robot else None,
                len(self._conns),
                [key for key in self._bodies if key != "__base__"],
            )
        )

    @staticmethod
    def _pose7(body: Body) -> list[float]:
        """
        A body's world pose as [x, y, z, qx, qy, qz, qw].
        """
        pose = body.global_pose
        position = pose.to_position().to_np().flatten()
        orientation = pose.to_quaternion().to_np().flatten()
        return [
            round(float(value), 5)
            for value in (position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3])
        ]

    def _snap(self, executor: Executor) -> None:
        """
        Append one frame: every movable connection's position, the robot base pose and
        every tracked object's pose.
        """
        if self._conns is None:
            self._lazy_bind(executor)
        frame = {
            str(connection.name): round(float(connection.position), 5)
            for connection in self._conns
        }
        self.frames.append(frame)
        self.base_frames.append(
            self._pose7(self._bodies["__base__"]) if "__base__" in self._bodies else None
        )
        object_frame = {
            name: self._pose7(body)
            for name, body in self._bodies.items()
            if name != "__base__"
        }
        self.obj_frames.append(object_frame)
        if len(self.frames) % 2000 == 0:
            self.timer.log("... %d frames" % len(self.frames))

    # %% action metadata hook
    # Plans compile all actions into merged motion statecharts, so there is no
    # per-action call boundary at execution time. ActionNode.parse *does* fire
    # once per action (in plan order) — we record the action class, its arm and
    # its target object there, and later derive the segment timing from the
    # recorded data (object attach/detach + first base motion).
    def _target_of(self, designator: Designator) -> str | None:
        """
        The recorded object a designator refers to, by matching any of its attributes'
        names against the demo's known mesh basenames.
        """
        basenames = {os.path.basename(source) for source in self.stl_sources}
        for value in vars(designator).values():
            if not hasattr(value, "name"):
                continue
            candidate = str(value.name).split("/")[-1]
            if candidate in basenames:
                return candidate
        return None

    def install_segment_hook(self) -> None:
        """
        Wrap ActionNode.parse to record each action's class, arm and target as it is
        parsed (in plan order).
        """
        from coraplex.plans.plan_node import ActionNode

        recorder = self
        original_parse = ActionNode.parse

        def _parse(node: ActionNode, *args: Any, **kwargs: Any) -> Executable:
            """
            Record this action's designator before letting it parse normally.
            """
            designator = node.designator
            recorder.actions.append(
                {
                    "action": type(designator).__name__,
                    "arm": _arm_of(designator),
                    "target": recorder._target_of(designator),
                }
            )
            recorder.plan_nodes.append(node)
            recorder.timer.log(
                "action parsed:",
                recorder.actions[-1]["action"],
                "->",
                recorder.actions[-1]["target"] or "-",
            )
            return original_parse(node, *args, **kwargs)

        ActionNode.parse = _parse

    # %% the executed plan tree, serialized from the real PlanNode graph
    def serialize_plans(self, max_nodes: int = 400) -> list[dict[str, Any]]:
        """
        The executed plan tree(s) (deduped by root), as nested dicts capped at max_nodes
        total.
        """
        from coraplex.plans.plan_node import DesignatorNode

        roots: list[PlanNode] = []
        seen: set[int] = set()
        for node in self.plan_nodes:
            root = node
            while root.parent is not None:
                root = root.parent
            if id(root) not in seen:
                seen.add(id(root))
                roots.append(root)
        remaining = [max_nodes]

        def _serialize(node: PlanNode) -> dict[str, Any] | None:
            """
            One PlanNode and its children as a dict, or None past max_nodes.
            """
            if remaining[0] <= 0:
                return None
            remaining[0] -= 1
            designator = node.designator if isinstance(node, DesignatorNode) else None
            entry = {
                "kind": type(node).__name__,
                "label": type(designator).__name__ if designator is not None else type(node).__name__,
                "status": node.status.name,
            }
            if designator is not None:
                target = self._target_of(designator)
                if target:
                    entry["target"] = target
                arm = _arm_of(designator)
                if arm is not None:
                    entry["arm"] = arm
            children = [_serialize(child) for child in node.children]
            entry["children"] = [child for child in children if child]
            return entry

        return [tree for tree in (_serialize(root) for root in roots) if tree]


def _arm_of(designator: Designator) -> str | None:
    """
    The arm-side field of a designator that has one ("arm" or "arms"), if any.
    """
    if hasattr(designator, "arm"):
        return str(designator.arm)
    if hasattr(designator, "arms"):
        return str(designator.arms)
    return None


# %% post-process
def moved(a: Sequence[float], b: Sequence[float], eps: float = 0.02) -> bool:
    """
    Whether pose b is more than eps away from pose a (planar distance + |dz|).
    """
    return math.hypot(a[0] - b[0], a[1] - b[1]) + abs(a[2] - b[2]) > eps


def object_windows(recorder: Recorder) -> list[dict[str, Any]]:
    """
    attach..detach window (raw frames) per object that travelled overall.
    """
    object_frames = recorder.obj_frames
    frame_count = len(object_frames)
    windows = []
    for name in object_frames[0]:
        first_pose, last_pose = object_frames[0].get(name), object_frames[frame_count - 1].get(name)
        if not first_pose or not last_pose or not moved(first_pose, last_pose, 0.03):
            continue
        attach = next(
            (
                index
                for index in range(frame_count)
                if name in object_frames[index] and moved(object_frames[index][name], first_pose)
            ),
            frame_count - 1,
        )
        detach = (
            next(
                (
                    index
                    for index in range(frame_count - 1, -1, -1)
                    if name in object_frames[index] and moved(object_frames[index][name], last_pose)
                ),
                0,
            )
            + 1
        )
        if attach < detach:
            windows.append(
                {"object": name, "attach": attach, "detach": detach, "place": last_pose[:3]}
            )
    windows.sort(key=lambda window: window["attach"])
    return windows


def first_base_motion(recorder: Recorder, before: int) -> int:
    """
    First raw frame (< before) at which the robot base left its spawn.
    """
    spawn = recorder.base_frames[0]
    for index in range(min(before, len(recorder.base_frames))):
        base = recorder.base_frames[index]
        if base and spawn and math.hypot(base[0] - spawn[0], base[1] - spawn[1]) > 0.05:
            return index
    return before


def derive_segments(recorder: Recorder) -> list[dict[str, Any]]:
    """Segments = data-driven windows, labelled from the parsed action list."""
    frame_count = len(recorder.frames)
    windows = object_windows(recorder)
    manipulation_actions = [action for action in recorder.actions if action.get("target")]
    lead_actions = []
    for action in recorder.actions:
        if action.get("target"):
            break
        lead_actions.append(action)

    segments = []
    previous_end = 0
    if windows:
        lead_end = first_base_motion(recorder, windows[0]["attach"])
        if lead_end > 10:
            label = (
                lead_actions[0]["action"].replace("Action", "").lower()
                if len(lead_actions) == 1
                else "prepare"
            )
            segments.append(
                {
                    "step": label,
                    "action": ",".join(action["action"] for action in lead_actions) or None,
                    "arm": None,
                    "start": 0,
                    "end": lead_end,
                }
            )
            previous_end = lead_end
    remaining_actions = list(manipulation_actions)
    for window_index, window in enumerate(windows):
        action = next(
            (candidate for candidate in remaining_actions if candidate["target"] == window["object"]),
            None,
        )
        if action:
            remaining_actions.remove(action)
        object_id = os.path.splitext(window["object"])[0]
        verb = action["action"].replace("Action", "").lower() if action else "move"
        next_attach = windows[window_index + 1]["attach"] if window_index + 1 < len(windows) else frame_count - 1
        end = (
            min(next_attach, window["detach"] + max(10, (next_attach - window["detach"]) // 2))
            if window_index + 1 < len(windows)
            else frame_count - 1
        )
        segments.append(
            {
                "step": "%s_%s" % (verb, object_id),
                "action": action["action"] if action else None,
                "arm": action["arm"] if action else None,
                "start": previous_end,
                "end": end,
                "picks": object_id,
                "attach": window["attach"],
                "detach": window["detach"],
                "place": window["place"],
            }
        )
        previous_end = end
    if not segments:
        label = (
            recorder.actions[0]["action"].replace("Action", "").lower()
            if len(recorder.actions) == 1
            else "plan"
        )
        segments.append(
            {"step": label, "action": None, "arm": None, "start": 0, "end": frame_count - 1}
        )
    return segments


def link_set(part: Arm | EndEffector) -> list[str]:
    """
    A robot part's link names, stripped of their model-name prefix.
    """
    names = []
    for body in part.bodies:
        name = str(body.name)
        names.append(name.split("/", 1)[1] if "/" in name else name)
    return names


@dataclass
class RobotDescription:
    """
    The robot's semantic description as written into scene.json.
    """

    name: str
    """
    The robot type's class name, lowercased.
    """

    prefix: str
    """
    The robot's name prefix in the composed world (before the ``/``).
    """

    base_body: str
    """
    The robot root body's name, without its prefix.
    """

    parts: dict[str, list[str]] = field(default_factory=dict)
    """
    ``{part_class_name: link_names}`` for every arm and end-effector.
    """

def _build_robot_description(robot: AbstractRobot, prefix: str, base_body: str) -> RobotDescription:
    """
    Describe a robot's arms and end-effectors as named link groups.
    """
    parts: dict[str, list[str]] = {}
    for arm in robot.get_arms():
        arm_links = link_set(arm)
        end_effector = arm.end_effector
        end_effector_links = link_set(end_effector) if end_effector is not None else []
        parts[type(arm).__name__] = sorted(set(arm_links) - set(end_effector_links))
        if end_effector is not None:
            parts[type(end_effector).__name__] = sorted(set(end_effector_links))
    return RobotDescription(
        name=type(robot).__name__.lower(), prefix=prefix, base_body=base_body, parts=parts
    )


def _build_segments(recorder: Recorder, nearest: Any) -> list[dict[str, Any]]:
    """
    Derive segments from the recording and remap their frame indices onto the
    downsampled trajectory.
    """
    segments = []
    for segment in derive_segments(recorder):
        remapped = dict(segment)
        remapped["start"] = nearest(segment["start"])
        remapped["end"] = nearest(segment["end"])
        if "attach" in remapped:
            remapped["attach"] = nearest(segment["attach"])
            remapped["detach"] = nearest(segment["detach"])
        segments.append(remapped)
    seen_labels: dict[str, int] = {}
    for segment in segments:
        seen_labels[segment["step"]] = seen_labels.get(segment["step"], 0) + 1
        if seen_labels[segment["step"]] > 1:
            segment["step"] = "%s_%d" % (segment["step"], seen_labels[segment["step"]])
    return segments


def _bundle_scene_objects(recorder: Recorder, out_dir: str) -> list[dict[str, Any]]:
    """
    Copy every loose object's mesh into the bundle and describe its spawn pose.
    """
    objects = []
    for index, source in enumerate(recorder.stl_sources):
        mesh = os.path.basename(source)
        if mesh not in recorder.obj_frames[0]:
            continue
        destination = os.path.join(out_dir, "meshes", "objects", mesh)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        objects.append(
            {
                "id": os.path.splitext(mesh)[0],
                "key": mesh,
                "mesh": "meshes/objects/" + mesh,
                "spawn": recorder.obj_frames[0][mesh],
                "color": PALETTE.color_for(index),
            }
        )
    return objects


def _place_target_and_drag_bounds(
    segments: list[dict[str, Any]], objects: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    The averaged place location (with drag bounds) and the objects' overall drag bounds,
    for the viewer's interactive controls.
    """
    places = [segment["place"] for segment in segments if segment.get("place")]
    place_target = None
    if places:
        center_x = sum(place[0] for place in places) / len(places)
        center_y = sum(place[1] for place in places) / len(places)
        center_z = min(place[2] for place in places)
        place_target = {
            "pos": [round(center_x, 3), round(center_y, 3)],
            "z": round(center_z - 0.02, 3),
            "bounds": {
                "minX": round(center_x - 0.55, 2),
                "maxX": round(center_x + 0.55, 2),
                "minY": round(center_y - 0.55, 2),
                "maxY": round(center_y + 0.65, 2),
            },
        }
    drag_bounds = None
    if objects:
        spawn_x = [obj["spawn"][0] for obj in objects]
        spawn_y = [obj["spawn"][1] for obj in objects]
        drag_bounds = {
            "minX": round(min(spawn_x) - 0.35, 2),
            "maxX": round(max(spawn_x) + 0.35, 2),
            "minY": round(min(spawn_y) - 0.6, 2),
            "maxY": round(max(spawn_y) + 0.6, 2),
        }
    return place_target, drag_bounds


def _bundle_scene_models(
    recorder: Recorder, out_dir: str, base_body: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Bundle every URDF/xacro the world was built from, and report which one is the robot
    by checking for its base body among the bundled links.
    """
    world_body_names = [str(body.name) for body in recorder.world.bodies]
    models = []
    missing: list[str] = []
    for source in recorder.urdf_sources:
        base_name = os.path.splitext(os.path.basename(source))[0]
        report: BundleReport = bundle_urdf(source, base_name, out_dir, hints=recorder.resolutions)
        missing += report.missing
        # find this model's prefix in the composed world via one of its links
        model_prefix = ""
        for link_name in report.links[:12]:
            hit = next((name for name in world_body_names if name.endswith("/" + link_name)), None)
            if hit:
                model_prefix = hit.split("/", 1)[0]
                break
        is_robot = base_body in report.links
        models.append(
            {
                "name": base_name,
                "urdf": "%s.urdf" % base_name,
                "prefix": model_prefix,
                "robot": is_robot,
                "links": len(report.links),
                "movableJoints": report.movable_joints,
            }
        )
        recorder.timer.log(
            "bundled %-28s prefix=%-12s robot=%s meshes=%d missing=%d"
            % (base_name, model_prefix or "-", is_robot, report.meshes_copied, len(report.missing))
        )
    return models, missing


def build_scene(recorder: Recorder, name: str, out_dir: str, step: int) -> dict[str, Any]:
    """
    Downsample the recording to every step-th frame (always keeping the last) and
    assemble scene.json + trajectory.json from it.
    """
    frame_count = len(recorder.frames)
    downsample_indices = list(range(0, frame_count, step))
    if downsample_indices and downsample_indices[-1] != frame_count - 1:
        downsample_indices.append(frame_count - 1)
    remap = {original: downsampled for downsampled, original in enumerate(downsample_indices)}

    def nearest(index: int) -> int:
        """
        The downsampled index closest to raw frame index.
        """
        return remap.get(index, remap[min(remap, key=lambda original: abs(original - index))])

    frames = [recorder.frames[index] for index in downsample_indices]
    base = [recorder.base_frames[index] for index in downsample_indices]
    objects_by_frame = [recorder.obj_frames[index] for index in downsample_indices]

    raw_fps = 1.0 / recorder.control_dt if recorder.control_dt else 50.0
    fps = max(10, round(raw_fps / step))

    robot = recorder.robot
    root_name = str(robot.root.name)
    prefix = root_name.split("/", 1)[0] if "/" in root_name else ""
    base_body = root_name.split("/", 1)[1] if "/" in root_name else root_name

    robot_description = _build_robot_description(robot, prefix, base_body)
    segments = _build_segments(recorder, nearest)
    objects = _bundle_scene_objects(recorder, out_dir)
    place_target, drag_bounds = _place_target_and_drag_bounds(segments, objects)
    models, missing = _bundle_scene_models(recorder, out_dir, base_body)

    scene = {
        "name": name,
        "fps": fps,
        "trajectory": "trajectory.json",
        "models": models,
        "robot": {
            "name": robot_description.name,
            "prefix": robot_description.prefix,
            "baseBody": robot_description.base_body,
            "parts": robot_description.parts,
        },
        "objects": objects,
        "segments": segments,
        "actions": recorder.actions,
        "planTrees": recorder.serialize_plans(),
        "placeTarget": place_target,
        "dragBounds": drag_bounds,
        "missingAssets": sorted(set(missing)),
    }
    with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as scene_file:
        json.dump(scene, scene_file, indent=1)
    with open(os.path.join(out_dir, "trajectory.json"), "w", encoding="utf-8") as trajectory_file:
        json.dump(
            {"fps": fps, "frames": frames, "base": base, "objects": objects_by_frame},
            trajectory_file,
        )
    return scene


# %% main
def main() -> None:
    """
    ``cram-viz-onboard`` — record one demo run into a scene bundle.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argument_parser.add_argument("demo", help="path to the coraplex demo .py file")
    argument_parser.add_argument("--name", required=True, help="scene name (output folder)")
    argument_parser.add_argument(
        "--out",
        default=str(paths.scenes_dir()),
        help="scenes directory (default: CRAM_VIZ_SCENES or ~/.cram_viz/scenes)",
    )
    argument_parser.add_argument("--step", type=int, default=0, help="downsample step (0 = auto)")
    arguments = argument_parser.parse_args()

    try:
        import coraplex  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "The CRAM stack is not importable — run under the action-cram venv:\n"
            "  the workspace venv (uv sync), then: cram-viz-onboard ..."
        )

    timer = Timer(start=time.time())
    recorder = Recorder(timer=timer)
    recorder.install_asset_hooks()
    recorder.install_tick_hook()
    recorder.install_segment_hook()

    demo = os.path.abspath(arguments.demo)
    timer.log("running demo:", demo)
    sys.path.insert(0, os.path.dirname(demo))
    # make repo-level helper packages (e.g. test.conftest) importable — the
    # demos rely on pytest's rootdir behaviour for that
    directory = os.path.dirname(demo)
    while directory != os.path.dirname(directory):
        if os.path.isdir(os.path.join(directory, "coraplex")) and os.path.isdir(
            os.path.join(directory, "test")
        ):
            sys.path.insert(0, directory)
            timer.log("repo root on sys.path:", directory)
            break
        directory = os.path.dirname(directory)
    runpy.run_path(demo, run_name="__main__")
    timer.log(
        "demo finished: %d raw frames, %d actions"
        % (len(recorder.frames), len(recorder.actions))
    )

    if not recorder.frames:
        sys.exit("No frames captured — did the demo perform a plan?")
    if recorder.robot is None:
        sys.exit("No AbstractRobot semantic annotation found in the world.")

    step = arguments.step or max(1, len(recorder.frames) // 1500)
    out_dir = os.path.join(arguments.out, arguments.name)
    os.makedirs(out_dir, exist_ok=True)
    scene = build_scene(recorder, arguments.name, out_dir, step)

    # maintain the scene index the viewer reads
    index_path = os.path.join(arguments.out, "index.json")
    try:
        with open(index_path, encoding="utf-8") as index_file:
            index = json.load(index_file)
    except FileNotFoundError:
        index = {"default": arguments.name, "scenes": []}
    if arguments.name not in index["scenes"]:
        index["scenes"].append(arguments.name)
    index.setdefault("default", arguments.name)
    with open(index_path, "w", encoding="utf-8") as index_file:
        json.dump(index, index_file, indent=1)

    timer.log("scene '%s' written to %s" % (arguments.name, out_dir))
    timer.log(
        "  models:  %s"
        % ", ".join(
            "%s%s" % (model["name"], " (robot)" if model["robot"] else "")
            for model in scene["models"]
        )
    )
    timer.log("  objects: %s" % ", ".join(obj["id"] for obj in scene["objects"]))
    timer.log("  segments: %s" % " → ".join(segment["step"] for segment in scene["segments"]))
    if scene["missingAssets"]:
        timer.log("  warning — %d missing assets:" % len(scene["missingAssets"]))
        for missing_asset in scene["missingAssets"][:10]:
            timer.log("   ", missing_asset)
    sys.stdout.flush()
    os._exit(0)  # don't hang on non-daemon ROS/viz threads the demo started


if __name__ == "__main__":
    main()
