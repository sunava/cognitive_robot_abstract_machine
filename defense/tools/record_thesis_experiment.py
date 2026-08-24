"""
Record one ``thesis_new`` experiment run as a cramera scene bundle the defense deck replays.

Runs a ``pycram/demos/thesis_new`` runner unchanged and attaches cramera's live bridge to
the world the runner builds, so the run is captured frame by frame and saved as a named
scene under :func:`cramera.paths.local_scenes_directory`.

The thesis branch carries an older ``semantic_digital_twin`` than cramera expects and has
no ``coraplex`` installed, so two adapters are applied before any cramera import: a module
alias for the robot-part annotations, and a state/model callback pair replacing
``coraplex``-dependent :class:`cramera.live.visualization.LiveVisualization`.

Usage, from the repository holding the thesis branch inside its virtualenv::

    source /opt/ros/jazzy/setup.bash
    PYTHONPATH=/home/vee/thesis/viz_pr/cramera/src:pycram:pycram/demos \\
      python record_thesis_experiment.py --task cut --robot pr2 --environment apartment \\
                                         --seed 42 --objects 2 --name thesis_cut_pr2_apartment
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import types

SETUP_FUNCTION_PATTERN = re.compile(r"^(setup|sample)_random_\w+$")
"""
Name of the per-action world builders in the thesis_new runners; their return value
carries the world the bridge attaches to. Cutting and mixing build a world, wiping
samples poses, and both spell it differently.
"""

ROBOT_PARTS_MODULE = "semantic_digital_twin.robots.robot_parts"
"""
Module cramera imports the robot-part annotations from; the thesis branch spells it
``semantic_digital_twin.robots.abstract_robot``.
"""

TARGET_COLLECTOR = "collect_named_targets"
"""
Name of the runner helper listing the spawned objects to work on, patched to cap how many
of them one recording covers.
"""


def alias_robot_parts_module() -> None:
    """
    Expose the thesis branch's robot annotations under the module name cramera imports.
    """
    annotations = importlib.import_module("semantic_digital_twin.robots.abstract_robot")
    alias = types.ModuleType(ROBOT_PARTS_MODULE)
    alias.AbstractRobot = annotations.AbstractRobot
    alias.AbstractRobotPart = annotations.SemanticRobotAnnotation
    for name in dir(annotations):
        if not name.startswith("_") and not hasattr(alias, name):
            setattr(alias, name, getattr(annotations, name))
    sys.modules[ROBOT_PARTS_MODULE] = alias


def add_pose_list_accessor() -> None:
    """
    Give the thesis branch's poses and transforms the flat accessor cramera reads them
    through, ``[x, y, z, qx, qy, qz, qw]``.
    """
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
        Pose,
    )

    def to_position_quaternion_list(self):
        translation = self.to_position().to_np().flatten()
        quaternion = self.to_quaternion().to_np().flatten()
        return [float(value) for value in (*translation[:3], *quaternion[:4])]

    for spatial_type in (Pose, HomogeneousTransformationMatrix):
        if not hasattr(spatial_type, "to_position_quaternion_list"):
            spatial_type.to_position_quaternion_list = to_position_quaternion_list


def add_cylinder_radius() -> None:
    """
    Give the thesis branch's cylinders the ``radius`` accessor the bundle writer reads,
    derived from the diameter it stores in ``width``.
    """
    from semantic_digital_twin.world_description.geometry import Cylinder

    if not hasattr(Cylinder, "radius"):
        Cylinder.radius = property(lambda self: self.width / 2)


def add_robot_part_accessors() -> None:
    """
    Give the thesis branch's robot annotations the accessors the bundle writer reads the
    arm/end-effector layout through: ``get_arms``, the optional left/right arm getters,
    and ``Arm.end_effector`` for what this branch calls the manipulator.
    """
    from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm

    def get_arms(self):
        return list(self.arms) if hasattr(self, "arms") else []

    def specified_arm(side: str):
        def getter(self):
            return getattr(self, side, None) if hasattr(type(self), side) else None

        return getter

    if not hasattr(AbstractRobot, "get_arms"):
        AbstractRobot.get_arms = get_arms
    for method_name, attribute in (
        ("get_left_arm_if_specified", "left_arm"),
        ("get_right_arm_if_specified", "right_arm"),
    ):
        if not hasattr(AbstractRobot, method_name):
            setattr(AbstractRobot, method_name, specified_arm(attribute))
    if not hasattr(Arm, "end_effector"):
        Arm.end_effector = property(lambda self: self.manipulator)


def alias_joint_type_enum() -> None:
    """
    Provide the one ``coraplex`` symbol the bundle writer imports, so bundling works in
    an environment without coraplex installed.
    """
    from enum import Enum

    class JointType(Enum):
        """
        Readable joint types, mirroring ``coraplex.datastructures.enums.JointType``.
        """

        REVOLUTE = 0
        PRISMATIC = 1
        SPHERICAL = 2
        PLANAR = 3
        FIXED = 4
        UNKNOWN = 5
        CONTINUOUS = 6
        FLOATING = 7

    package = types.ModuleType("coraplex")
    datastructures = types.ModuleType("coraplex.datastructures")
    enums = types.ModuleType("coraplex.datastructures.enums")
    enums.JointType = JointType
    datastructures.enums = enums
    package.datastructures = datastructures
    sys.modules.update({
        "coraplex": package,
        "coraplex.datastructures": datastructures,
        "coraplex.datastructures.enums": enums,
    })


def add_gaze_constraint() -> None:
    """
    Keep the robot's camera pointed at the object while the tool moves.

    Adds a :class:`giskardpy.motion_statechart.tasks.pointing.Pointing` task alongside
    the Cartesian trajectory of every aligned TCP motion, so the head tracks the target
    throughout the motion instead of only before it, as a sequential
    ``LookingMotion`` would.

    Applied only to recordings: the experiment's own motion charts are left untouched, so
    the numbers reported in the thesis stay reproducible.
    """
    from giskardpy.motion_statechart.goals.templates import Parallel
    from giskardpy.motion_statechart.tasks.pointing import Pointing
    from pycram.robot_plans.motions.gripper import MoveTCPWaypointsAlignedMotion

    original = MoveTCPWaypointsAlignedMotion._motion_chart.fget

    def motion_chart_with_gaze(self):
        chart = original(self)
        camera = self.robot.get_default_camera()
        if camera is None or not self.waypoints:
            return chart
        camera.forward_facing_axis.reference_frame = camera.root
        gaze = Pointing(
            root_link=self.robot.torso.root if self.robot.torso is not None else self.robot.root,
            tip_link=camera.root,
            goal_point=self.waypoints[len(self.waypoints) // 2],
            pointing_axis=camera.forward_facing_axis,
            name="LookAtTarget",
        )
        return Parallel([chart, gaze])

    MoveTCPWaypointsAlignedMotion._motion_chart = property(motion_chart_with_gaze)
    print("[record] gaze constraint added to aligned TCP motions", flush=True)


CUTTING_BOARD_MESH = ("pycram_object_gap_demo", "board.stl")
"""
Mesh spawned under each cut object, from the demo resources.
"""

BOARD_NAME_PREFIX = "cutting_board_"
"""
Prefix marking a spawned board, so boards do not get boards of their own.
"""

SPAWN_HEIGHT_OFFSET_M = 0.05
"""
Height above the supporting surface the demo spawns its objects at, mirroring the
``z_offset`` default of the layout in ``spawn_random_breads``. The board's origin sits on
its own underside, so lowering it by this offset rests it on the surface.
"""


def add_cutting_boards() -> None:
    """
    Spawn a cutting board resting on the surface under every object the demo places.

    The object itself keeps the pose the experiment gave it, so the cut geometry is
    unchanged.

    Applied only to recordings: the boards are extra collision geometry near the cut, so
    leaving the experiment's own scene untouched keeps its numbers reproducible.

    ..warning:: The knife descends to a depth measured from the object, so a board can be
        hit on deep cuts.
    """
    from semantic_digital_twin.world_description.geometry import Color

    from thesis_new.src import spawn_random_breads as spawner

    original = spawner._spawn_object_at_local_pose

    def spawn_on_board(*, object_name, z_local, **keywords):
        if object_name.startswith(BOARD_NAME_PREFIX):
            return original(object_name=object_name, z_local=z_local, **keywords)
        board_keywords = dict(keywords)
        board_keywords.update(
            scale=1.0,
            mesh_parts=CUTTING_BOARD_MESH,
            color=Color(R=0.55, G=0.38, B=0.22),
        )
        original(
            object_name=f"{BOARD_NAME_PREFIX}{object_name}",
            z_local=z_local - SPAWN_HEIGHT_OFFSET_M,
            **board_keywords,
        )
        return original(object_name=object_name, z_local=z_local, **keywords)

    spawner._spawn_object_at_local_pose = spawn_on_board
    print("[record] cutting boards spawned under every placed object", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="cut | mix | pour | wipe")
    parser.add_argument("--single", action="store_true",
                        help="run the single-object demo: one action on one spawned object")
    parser.add_argument("--object", default=None, help="single-object demo: what to act on")
    parser.add_argument("--technique", default=None, help="single-object demo: technique to use")
    parser.add_argument("--spawn-yaw", type=float, default=None,
                        help="single-object demo: yaw the object is spawned at, in radians")
    parser.add_argument("--robot", default=None)
    parser.add_argument("--environment", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", required=True, help="scene name to save the recording under")
    parser.add_argument("--target-surfaces", default=None,
                        help="comma-separated surface name parts the targets must sit on")
    parser.add_argument("--vertical-targets", action="store_true",
                        help="wipe only targets on vertical surfaces, e.g. cabinet fronts")
    parser.add_argument("--look-at", action="store_true",
                        help="keep the robot's camera pointed at the target during the motion")
    parser.add_argument("--cutting-boards", action="store_true",
                        help="spawn a cutting board under every object the cutting demo places")
    parser.add_argument("--objects", type=int, default=0,
                        help="cap on how many spawned objects the run covers (0 = all)")
    return parser.parse_args()


def build_state_recorder(bridge):
    """
    Callbacks feeding the world's own state and model changes into the bridge.

    Replaces :class:`cramera.live.visualization.LiveVisualization`, which cannot be
    imported without ``coraplex``. The thesis branch's callbacks dispatch through
    ``_notify`` rather than the newer ``on_state_change``/``on_model_change``.

    :param bridge: The bridge snapshots are published to.
    :return: The state-change and model-change callback pair, already attached.
    """
    from dataclasses import dataclass, field

    from semantic_digital_twin.callbacks.callback import (
        ModelChangeCallback,
        StateChangeCallback,
    )

    @dataclass(eq=False)
    class RecordingStateSync(StateChangeCallback):
        """Publishes a world snapshot to the bridge on every state change."""

        bridge: object = field(kw_only=True)

        def _notify(self, **kwargs) -> None:
            self.bridge.snapshot()
            if self.bridge.recording is not None:
                self.bridge.recording.append(self.bridge.state, self.bridge.running_step())

    @dataclass(eq=False)
    class RecordingModelSync(ModelChangeCallback):
        """Refreshes the bridge's geometry catalogs when the world model changes."""

        bridge: object = field(kw_only=True)

        def _notify(self, **kwargs) -> None:
            self.bridge.observe_model_change()

    return (
        RecordingStateSync(_world=bridge.world, bridge=bridge),
        RecordingModelSync(_world=bridge.world, bridge=bridge),
    )


def start_recording(world) -> None:
    """
    Attach the bridge to a world and begin capturing its state changes.
    """
    from cramera.live.bridge import BRIDGE
    from cramera.live.recording import Recording

    BRIDGE.attach(world)
    BRIDGE.recording = Recording()
    BRIDGE.recording.start()
    BRIDGE.snapshot()
    build_state_recorder(BRIDGE)
    print("[record] recording started", flush=True)


def patch_world_builders(runner_module) -> None:
    """
    Wrap the runner module's world builders so recording starts on the world they return.
    """
    builders = [n for n in dir(runner_module) if SETUP_FUNCTION_PATTERN.match(n)]
    if not builders:
        sys.exit(f"no world builder found in {runner_module.__name__}")
    for builder_name in builders:
        original = getattr(runner_module, builder_name)

        def wrapped(*arguments, _original=original, **keywords):
            result = _original(*arguments, **keywords)
            start_recording(result[0] if isinstance(result, tuple) else result)
            return result

        setattr(runner_module, builder_name, wrapped)
    print(f"[record] patched world builders: {', '.join(builders)}", flush=True)


def patch_target_surfaces(runner_module, wanted: str) -> None:
    """
    Keep only the targets whose supporting surface matches one of the given name parts,
    picking which run of cabinets or counters a recording works on.
    """
    parts = [part.strip() for part in wanted.split(",") if part.strip()]

    for builder_name in [n for n in dir(runner_module) if SETUP_FUNCTION_PATTERN.match(n)]:
        original = getattr(runner_module, builder_name)

        def keep_surfaces(*arguments, _original=original, **keywords):
            result = _original(*arguments, **keywords)
            if not isinstance(result, tuple) or len(result) < 2 or not isinstance(result[1], list):
                return result
            kept = [
                target for target in result[1]
                if any(part in target.get("surface_name", "") for part in parts)
            ]
            print(f"[record] {len(kept)} of {len(result[1])} targets on {wanted}", flush=True)
            if not kept:
                sys.exit(f"[record] no target on any of: {wanted}")
            return (result[0], kept) + result[2:]

        setattr(runner_module, builder_name, keep_surfaces)


def patch_vertical_targets(runner_module) -> None:
    """
    Keep only the wipe targets that sit on a vertical surface, judged by the runner's own
    predicate, so a recording shows wiping a cabinet front rather than a countertop.
    """
    is_vertical = runner_module._is_vertical_wipe_pose

    for builder_name in [n for n in dir(runner_module) if SETUP_FUNCTION_PATTERN.match(n)]:
        original = getattr(runner_module, builder_name)

        def keep_vertical(*arguments, _original=original, **keywords):
            result = _original(*arguments, **keywords)
            if not isinstance(result, tuple) or len(result) < 2 or not isinstance(result[1], list):
                return result
            vertical = [t for t in result[1] if is_vertical(t["world_pose"])]
            print(f"[record] {len(vertical)} of {len(result[1])} targets are vertical", flush=True)
            if not vertical:
                sys.exit("[record] no vertical target in this environment and seed")
            return (result[0], vertical) + result[2:]

        setattr(runner_module, builder_name, keep_vertical)


def patch_target_cap(runner_module, limit: int) -> None:
    """
    Cap how many objects the run works on, keeping the recording short.

    Runners that look their targets up in the world are capped at that lookup; a runner
    whose world builder hands the targets back is capped on that return value instead.
    """
    if limit <= 0:
        return

    def announce(targets):
        print(f"[record] {len(targets)} targets available, recording the first {limit}", flush=True)
        return targets[:limit]

    if hasattr(runner_module, TARGET_COLLECTOR):
        original = getattr(runner_module, TARGET_COLLECTOR)

        def capped(*arguments, **keywords):
            return announce(original(*arguments, **keywords))

        setattr(runner_module, TARGET_COLLECTOR, capped)
        return

    for builder_name in [n for n in dir(runner_module) if SETUP_FUNCTION_PATTERN.match(n)]:
        original = getattr(runner_module, builder_name)

        def cap_returned_targets(*arguments, _original=original, **keywords):
            result = _original(*arguments, **keywords)
            if not isinstance(result, tuple) or len(result) < 2 or not isinstance(result[1], list):
                return result
            return (result[0], announce(result[1])) + result[2:]

        setattr(runner_module, builder_name, cap_returned_targets)


def show_collision_only_bodies() -> None:
    """
    Give every body that carries collision geometry but no visual geometry a visual copy
    of it, so the bundle shows it.

    The demos build their tools as collision-only boxes — a sponge is invisible in the
    exported bundle otherwise, because the exporter writes visual shapes.
    """
    from copy import deepcopy

    from cramera.live.bridge import BRIDGE
    from semantic_digital_twin.world_description.shape_collection import ShapeCollection

    made_visible = []
    for body in BRIDGE.world.bodies:
        if len(body.visual) or not len(body.collision):
            continue
        body.visual = ShapeCollection(
            [deepcopy(shape) for shape in body.collision], reference_frame=body
        )
        made_visible.append(str(body.name))
    if made_visible:
        print(f"[record] made visible: {', '.join(made_visible)}", flush=True)


def save_bundle(name: str) -> None:
    """
    Write the captured frames to disk and promote them to a permanently saved scene.
    """
    from cramera.live.bridge import BRIDGE
    from cramera.live.recording_bundle import finalize_recording
    from cramera.live.recording_storage import save_recording_bundle

    if BRIDGE.recording is None:
        sys.exit("[record] nothing was recorded — the world builder was never called")
    show_collision_only_bodies()
    finalize_recording(BRIDGE, BRIDGE.recording)
    print(f"[record] saved scene: {save_recording_bundle(name)}", flush=True)


def export_demo_utilities() -> None:
    """
    Re-export the demo helpers on the ``utils`` package the single-object demo imports them
    from; the package's own ``__init__`` is empty, so that import fails otherwise.
    """
    from thesis_new.src.utils import demo_utils

    package = importlib.import_module("demos.thesis_new.src.utils")
    for name in dir(demo_utils):
        if not name.startswith("_") and not hasattr(package, name):
            setattr(package, name, getattr(demo_utils, name))


def record_single_object_demo(arguments) -> None:
    """
    Run the single-object demo, which performs one action on one spawned object.

    It builds its world through ``world_setup.setup_thesis_world`` at call time rather
    than through a runner-module attribute, so recording is started there.
    """
    export_demo_utilities()
    from demos.thesis_single_object import single_object_cut_demo
    from thesis_new.src import world_setup

    run_single_object_demo = single_object_cut_demo.run_single_object_demo
    original = single_object_cut_demo.setup_thesis_world

    def build_and_record(*call_arguments, **keywords):
        world = original(*call_arguments, **keywords)
        start_recording(world)
        return world

    #: the demo binds the builder into its own namespace at import time
    single_object_cut_demo.setup_thesis_world = build_and_record
    try:
        keywords = {}
        if arguments.object is not None:
            keywords["object_kind"] = arguments.object
        if arguments.spawn_yaw is not None:
            keywords["spawn_yaw"] = arguments.spawn_yaw
        run_single_object_demo(
            action=arguments.task,
            technique=arguments.technique,
            robot_name=world_setup.resolve_robot_name(arguments.robot),
            environment_name=arguments.environment,
            **keywords,
        )
    except Exception as failure:
        # boundary guard: this demo performs a single attempt and lets a stalled motion
        # escape, unlike the batch runners; the frames recorded up to that point are
        # still worth bundling
        print(f"[record] the demo failed: {type(failure).__name__}: {failure}", flush=True)


def main() -> None:
    arguments = parse_arguments()
    alias_robot_parts_module()
    add_pose_list_accessor()
    alias_joint_type_enum()
    add_cylinder_radius()
    add_robot_part_accessors()
    from thesis_new.src.demo_runners import get_thesis_demo_runner
    from thesis_new.src.world_setup import resolve_robot_name

    if arguments.look_at:
        add_gaze_constraint()
    if arguments.cutting_boards:
        add_cutting_boards()
    if arguments.single:
        record_single_object_demo(arguments)
        save_bundle(arguments.name)
        return
    runner = get_thesis_demo_runner(arguments.task)
    runner_module = sys.modules[runner.__module__]
    patch_world_builders(runner_module)
    if arguments.vertical_targets:
        patch_vertical_targets(runner_module)
    if arguments.target_surfaces:
        patch_target_surfaces(runner_module, arguments.target_surfaces)
    patch_target_cap(runner_module, arguments.objects)   # caps whatever the filters left
    runner(
        seed=arguments.seed,
        robot_name=resolve_robot_name(arguments.robot),
        environment_name=arguments.environment,
    )
    save_bundle(arguments.name)


if __name__ == "__main__":
    main()
