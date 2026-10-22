"""
Playback of recorded body motion: joint values recovered from body world poses, the
robot as a model of its own, moving free bodies as objects, recorded steps as segments,
and the forward kinematics check of all of it.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path

import numpy
import pytest
from scipy.spatial.transform import Rotation

pxr = pytest.importorskip("pxr")

from pxr import Usd, UsdGeom

from cramera.onboard.usd_scene.animation import (
    BodyTrace,
    LinkTree,
    RecordedRun,
    ResidualColumn,
    ResidualKind,
    ResidualTolerance,
    RobotNaming,
    SceneAnimator,
    Segment,
    TraceArray,
)
from cramera.onboard.usd_scene.bundle import BundleKey
from cramera.onboard.usd_scene.bundle_layout import BundleFile
from cramera.onboard.usd_scene.command_line import ImportOption, main
from cramera.onboard.usd_scene.exceptions import (
    RobotRootNotFound,
    TraceIncomplete,
    TraceNotFound,
    VariantNotFound,
)
from cramera.onboard.usd_scene.robot_description import UrdfAttribute, UrdfTag
from cramera.onboard.usd_scene.scene_description import (
    ImportedJoint,
    ImportedLink,
    JointKind,
    Pose,
    PoseTrack,
    SceneDescription,
)
from cramera.onboard.usd_scene.stage_reader import StageReader, VariantSelection
from cramera.scene_presentation import PresentationField

DATASET = Path(__file__).parent / "dataset" / "usd_scene"
"""
Small stages, one per behaviour.
"""

ARM = DATASET / "two_joint_arm.usda"
"""
The animated two-joint arm whose joints exist only in its physics variant.
"""

TRACE = DATASET / "two_joint_arm_trace.npz"
"""
The arm's body world poses at 2 Hz, in the twin recorder's format.
"""

RECORD = DATASET / "two_joint_arm_record.json"
"""
The recorded steps of the arm's run, on the recorder's clock.
"""

PHYSICS = VariantSelection("/World/Arm", "Physics", "physics")
"""
The variant holding the arm's rigid bodies and joints.
"""

SHOULDER = "/World/Arm/Joints/Shoulder"
SLIDE = "/World/Arm/Joints/Slide"
PUCK = "/World/Puck"
LOWER = "/World/Arm/Base/Upper/Lower"


# %% helpers


def joint_name(description: SceneDescription, path: str) -> str:
    """
    The converted name of the joint authored at a path.

    :param description: The converted stage.
    :param path: The joint prim's path.
    """
    return next(joint.name for joint in description.joints if joint.path == path)


def link_name(description: SceneDescription, path: str) -> str:
    """
    The converted name of the link standing for a prim.

    :param description: The converted stage.
    :param path: The prim's path.
    """
    return next(link.name for link in description.links if link.path == path)


def authored(stage: Usd.Stage, path: str, operation: str, seconds: float) -> object:
    """
    The value an xform operation of a prim takes at a time.

    :param stage: The fixture stage.
    :param path: The prim's path.
    :param operation: The xform operation's attribute name.
    :param seconds: The time in seconds.
    """
    return (
        stage.GetPrimAtPath(path)
        .GetAttribute(operation)
        .Get(Usd.TimeCode(seconds * stage.GetTimeCodesPerSecond()))
    )


def world_pose(stage: Usd.Stage, path: str, seconds: float) -> Pose:
    """
    The rigid world pose of a prim at a time, as the stage composes it.

    :param stage: The fixture stage.
    :param path: The prim's path.
    :param seconds: The time in seconds.
    """
    cache = UsdGeom.XformCache(Usd.TimeCode(seconds * stage.GetTimeCodesPerSecond()))
    matrix = numpy.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
    return Pose.rigid_part(matrix)


def convert(scenes: Path, name: str, *options: str) -> Path:
    """
    Run ``cramera-import`` on the arm with its physics variant selected.

    :param scenes: The scenes directory.
    :param name: The bundle name.
    :param options: Further command line options.
    """
    main(
        [
            str(ARM),
            ImportOption.NAME,
            name,
            ImportOption.OUT,
            str(scenes),
            ImportOption.VARIANT,
            PHYSICS.prim_path,
            PHYSICS.variant_set,
            PHYSICS.variant,
            *options,
        ]
    )
    return scenes / name


def load(bundle: Path, file: BundleFile) -> dict:
    """
    One JSON file of a bundle.

    :param bundle: The bundle directory.
    :param file: The file.
    """
    return json.loads((bundle / file).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stage() -> Usd.Stage:
    return Usd.Stage.Open(str(ARM))


@pytest.fixture(scope="module")
def description() -> SceneDescription:
    return StageReader(ARM, variant_selections=[PHYSICS]).read()


@pytest.fixture(scope="module")
def trace() -> BodyTrace:
    return BodyTrace.from_file(TRACE)


@pytest.fixture(scope="module")
def trace_times(trace) -> numpy.ndarray:
    return numpy.arange(trace.frame_count) / trace.frames_per_second


@pytest.fixture(scope="module")
def animation(description, trace):
    return SceneAnimator(description, trace).animate()


@pytest.fixture(scope="module")
def robot_animation(description, trace):
    return SceneAnimator(description, trace, robot_path="/World/Arm").animate()


# %% variant selection


def test_session_variant_brings_the_joints_without_touching_the_file():
    before = ARM.read_bytes()
    plain = StageReader(ARM).read()
    selected = StageReader(ARM, variant_selections=[PHYSICS]).read()
    assert plain.movable_joints == []
    assert [joint.path for joint in selected.movable_joints] == [SHOULDER, SLIDE]
    assert ARM.read_bytes() == before


def test_disabled_joints_of_the_variant_still_drive_their_bodies(description):
    assert all(not joint.enabled for joint in description.movable_joints)


def test_unknown_variant_is_refused():
    with pytest.raises(VariantNotFound):
        StageReader(ARM, variant_selections=[replace(PHYSICS, variant="rigid")]).read()


# %% trace file


def test_trace_poses_match_the_stage_with_scalar_first_quaternions(
    trace, trace_times, stage
):
    for frame, seconds in enumerate(trace_times):
        assert (
            trace.poses[PUCK]
            .pose_at(frame)
            .is_close(world_pose(stage, PUCK, seconds), 1e-9)
        )


def test_resampling_keeps_the_nearest_frames(trace):
    halved = trace.resampled(trace.frames_per_second / 2)
    assert halved.frames_per_second == trace.frames_per_second / 2
    assert halved.frame_count == 4
    for frame in range(halved.frame_count):
        assert (
            halved.poses[LOWER]
            .pose_at(frame)
            .is_close(trace.poses[LOWER].pose_at(2 * frame), 0.0)
        )


def test_missing_trace_file_is_refused(tmp_path):
    with pytest.raises(TraceNotFound):
        BodyTrace.from_file(tmp_path / "missing.npz")


def test_trace_without_orientations_is_refused(tmp_path):
    recorded = numpy.load(TRACE)
    partial = tmp_path / "partial.npz"
    numpy.savez(
        partial,
        **{
            name: recorded[name]
            for name in recorded.files
            if name != TraceArray.QUATERNIONS
        },
    )
    with pytest.raises(TraceIncomplete) as refusal:
        BodyTrace.from_file(partial)
    assert refusal.value.missing == [TraceArray.QUATERNIONS]


# %% joint recovery


def test_joint_values_follow_the_authored_motion(
    animation, description, stage, trace_times
):
    shoulder = animation.joint_values[joint_name(description, SHOULDER)]
    slide = animation.joint_values[joint_name(description, SLIDE)]
    upper = "/World/Arm/Base/Upper"
    assert shoulder == pytest.approx(
        [
            numpy.radians(authored(stage, upper, "xformOp:rotateZ", seconds))
            for seconds in trace_times
        ],
        abs=1e-6,
    )
    assert slide == pytest.approx(
        [
            authored(stage, LOWER, "xformOp:translate", seconds)[0]
            - authored(stage, LOWER, "xformOp:translate", 0.0)[0]
            for seconds in trace_times
        ],
        abs=1e-6,
    )


def test_angle_past_half_a_turn_stays_within_the_joint_limits(animation, description):
    joint = next(joint for joint in description.joints if joint.path == SHOULDER)
    values = animation.joint_values[joint.name]
    assert values.min() >= joint.lower
    assert values.max() <= joint.upper
    assert values[-1] == pytest.approx(numpy.radians(200.0), abs=1e-6)


def test_angle_of_a_joint_turning_more_than_once_has_no_jumps():
    turned = numpy.radians(numpy.arange(0.0, 351.0, 50.0))
    hinge = ImportedJoint(
        name="Hinge",
        path="/World/Hinge",
        kind=JointKind.REVOLUTE,
        parent="Frame",
        child="Leaf",
        origin=Pose(),
        axis=numpy.array([0.0, 0.0, 1.0]),
        lower=numpy.radians(-400.0),
        upper=numpy.radians(400.0),
    )
    relative = PoseTrack.of_parts(
        Rotation.from_rotvec(numpy.outer(turned, hinge.axis)).as_matrix(),
        numpy.zeros((len(turned), 3)),
    )
    assert hinge.values_of(relative) == pytest.approx(turned)


def test_motion_off_the_axis_is_reported_as_joint_residual(description, trace):
    displaced = trace.poses[LOWER].matrices.copy()
    displaced[2, 2, 3] += 0.002
    bent = replace(trace, poses={**trace.poses, LOWER: PoseTrack(displaced)})
    animation = SceneAnimator(description, bent).animate()
    residual = next(
        residual for residual in animation.joint_residuals if residual.path == SLIDE
    )
    assert residual.largest_distance == pytest.approx(0.002, abs=1e-9)
    assert residual.frame == 2


# %% replay


def test_replay_reproduces_every_traced_body(animation, trace):
    assert sorted(residual.path for residual in animation.body_residuals) == sorted(
        trace.poses
    )
    for residual in animation.body_residuals:
        assert residual.largest_distance < 1e-6
        assert residual.largest_angle < 1e-6


def test_moving_free_body_becomes_an_object_with_its_traced_pose(
    animation, description, trace
):
    (moving,) = animation.objects
    assert moving.link.path == PUCK
    assert moving.key == "Puck"
    for frame in range(trace.frame_count):
        assert moving.poses.pose_at(frame).is_close(
            trace.poses[PUCK].pose_at(frame) @ moving.link.prim_to_link, 1e-9
        )
    environment = animation.models[0]
    assert link_name(description, PUCK) not in [link.name for link in environment.links]


def test_resting_free_body_stays_in_the_environment(animation, description):
    environment = animation.models[0]
    assert link_name(description, "/World/Crate") in [
        link.name for link in environment.links
    ]


def test_tolerance_flags_a_body_the_replay_misses(description, trace):
    displaced = trace.poses[LOWER].matrices.copy()
    displaced[2, 2, 3] += 0.002
    bent = replace(trace, poses={**trace.poses, LOWER: PoseTrack(displaced)})
    animation = SceneAnimator(description, bent, robot_path="/World/Arm").animate()
    tolerance = ResidualTolerance()
    beyond = [
        residual.path
        for residual in animation.body_residuals
        if not residual.within(tolerance)
    ]
    assert beyond == [LOWER]


def test_environment_body_the_replay_misses_falls_back_to_an_object(description, trace):
    displaced = trace.poses[LOWER].matrices.copy()
    displaced[2, 2, 3] += 0.002
    bent = replace(trace, poses={**trace.poses, LOWER: PoseTrack(displaced)})
    animation = SceneAnimator(description, bent).animate()
    assert sorted(moving.link.path for moving in animation.objects) == [LOWER, PUCK]
    assert all(
        residual.within(ResidualTolerance()) for residual in animation.body_residuals
    )


# %% robot


def test_robot_is_a_model_of_its_own_named_after_its_prims(
    robot_animation, description
):
    environment, robot = robot_animation.models
    assert (robot.name, robot.prefix, robot.robot) == ("arm", "arm", True)
    assert [link.name for link in robot.links] == ["Arm", "Base", "Upper", "Lower"]
    assert [joint.name for joint in robot.joints] == ["RootJoint", "Shoulder", "Slide"]
    robot_paths = {link.path for link in robot.links}
    assert not robot_paths & {link.path for link in environment.links}


def test_robot_joint_values_are_keyed_by_the_robot_prefix(
    robot_animation, animation, description
):
    assert robot_animation.joint_values["arm/Shoulder"] == pytest.approx(
        animation.joint_values[joint_name(description, SHOULDER)]
    )
    assert robot_animation.joint_values["arm/Slide"] == pytest.approx(
        animation.joint_values[joint_name(description, SLIDE)]
    )


def test_robot_base_follows_its_root_prim(robot_animation, stage, trace_times):
    for frame, seconds in enumerate(trace_times):
        assert robot_animation.base.pose_at(frame).is_close(
            world_pose(stage, "/World/Arm", seconds), 1e-9
        )


def test_robot_parts_group_links_below_their_common_prim(robot_animation):
    robot = robot_animation.models[1]
    assert robot.parts == {"arm": ["Arm"], "Base": ["Base", "Upper", "Lower"]}


def test_robot_parts_follow_the_common_prim_prefix_only():
    links = [
        ImportedLink("Arm", "/World/Robot/Arm/Link", Pose()),
        ImportedLink("Hand", "/World/Robot/Hand/Link", Pose()),
    ]
    tree = LinkTree(SceneDescription(ARM, root_link="Arm", links=links))
    naming = RobotNaming(tree, ["Arm", "Hand"])
    assert naming.parts("robot") == {"Arm": ["Link"], "Hand": ["Link_2"]}


def test_robot_replay_reproduces_every_traced_body(robot_animation):
    for residual in robot_animation.body_residuals:
        assert residual.largest_distance < 1e-6
        assert residual.largest_angle < 1e-6


def test_robot_prim_without_one_root_link_is_refused(description, trace):
    with pytest.raises(RobotRootNotFound):
        SceneAnimator(description, trace, robot_path="/World/Missing").animate()


# %% segments


def test_steps_become_segments_on_the_trace_clock():
    run = RecordedRun.from_file(RECORD)
    assert run.segments(2.0, 7) == [
        Segment("turn shoulder", 0, 2),
        Segment("extend slide", 2, 5),
        Segment("slide extended", 5, 5),
        Segment("settle", 5, 6),
    ]


# %% bundle


class TestAnimatedBundle:
    """
    The bundle of the arm converted with its trace, robot and recorded steps.
    """

    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory) -> Path:
        return convert(
            tmp_path_factory.mktemp("scenes"),
            "arm",
            ImportOption.TRACE,
            str(TRACE),
            ImportOption.FRAMES_PER_SECOND,
            "2",
            ImportOption.ROBOT,
            "/World/Arm",
            ImportOption.RECORD,
            str(RECORD),
        )

    def test_scene_declares_environment_and_robot_models(self, bundle):
        models = load(bundle, BundleFile.SCENE)[PresentationField.MODELS]
        assert [
            (
                model[BundleKey.NAME],
                model[PresentationField.ROBOT],
                model[BundleKey.PREFIX],
            )
            for model in models
        ] == [("environment", False, ""), ("arm", True, "arm")]
        assert models[1][BundleKey.MOVABLE_JOINTS] == ["Shoulder", "Slide"]
        assert all(model[PresentationField.PRESERVE_MATERIALS] for model in models)

    def test_scene_describes_the_robot(self, bundle):
        robot = load(bundle, BundleFile.SCENE)[PresentationField.ROBOT]
        assert robot[BundleKey.NAME] == "arm"
        assert robot[BundleKey.PREFIX] == "arm"
        assert robot[BundleKey.BASE_BODY] == "Arm"
        assert robot[BundleKey.PARTS] == {
            "arm": ["Arm"],
            "Base": ["Base", "Upper", "Lower"],
        }

    def test_robot_urdf_holds_its_links_with_existing_meshes(self, bundle):
        urdf = (
            bundle
            / load(bundle, BundleFile.SCENE)[PresentationField.MODELS][1][
                BundleKey.URDF
            ]
        )
        robot = ElementTree.parse(urdf).getroot()
        assert [link.get(UrdfAttribute.NAME) for link in robot.iter(UrdfTag.LINK)] == [
            "Arm",
            "Base",
            "Upper",
            "Lower",
        ]
        files = [mesh.get(UrdfAttribute.FILENAME) for mesh in robot.iter(UrdfTag.MESH)]
        assert files and all((bundle / file).is_file() for file in files)

    def test_moving_free_body_is_an_object_with_its_own_mesh(self, bundle):
        (entry,) = load(bundle, BundleFile.SCENE)[PresentationField.OBJECTS]
        assert entry[BundleKey.ID] == entry[PresentationField.KEY] == "Puck"
        assert (bundle / entry[BundleKey.MESH]).is_file()
        assert entry[PresentationField.PRESERVE_MATERIALS] is True
        first = load(bundle, BundleFile.TRAJECTORY)[PresentationField.OBJECTS][0][
            "Puck"
        ]
        assert entry[BundleKey.SPAWN] == first

    def test_trajectory_tracks_have_one_entry_per_frame(self, bundle):
        trajectory = load(bundle, BundleFile.TRAJECTORY)
        assert trajectory[BundleKey.FRAMES_PER_SECOND] == 2.0
        assert (
            len(trajectory[BundleKey.FRAMES])
            == len(trajectory[BundleKey.BASE])
            == len(trajectory[PresentationField.OBJECTS])
            == BodyTrace.from_file(TRACE).frame_count
        )
        assert set(trajectory[BundleKey.FRAMES][0]) == {"arm/Shoulder", "arm/Slide"}

    def test_steps_are_the_scene_segments(self, bundle):
        segments = load(bundle, BundleFile.SCENE)[BundleKey.SEGMENTS]
        expected = RecordedRun.from_file(RECORD).segments(2.0, 7)
        assert [
            (segment[BundleKey.STEP], segment[BundleKey.START], segment[BundleKey.END])
            for segment in segments
        ] == [(segment.step, segment.start, segment.end) for segment in expected]

    def test_semantics_name_the_model_and_link_of_robot_and_object_prims(self, bundle):
        prims = load(bundle, BundleFile.SEMANTICS)[BundleKey.PRIMS]
        upper = prims["/World/Arm/Base/Upper"]
        puck = prims[PUCK]
        assert (upper[BundleKey.MODEL], upper[BundleKey.LINK]) == ("arm", "Upper")
        assert (puck[BundleKey.MODEL], puck[BundleKey.LINK]) == ("Puck", "Puck")

    def test_residuals_list_every_traced_body_and_movable_joint(self, bundle):
        with (bundle / BundleFile.RESIDUALS).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        bodies = [
            row[ResidualColumn.PATH]
            for row in rows
            if row[ResidualColumn.KIND] == ResidualKind.BODY
        ]
        joints = [
            row[ResidualColumn.PATH]
            for row in rows
            if row[ResidualColumn.KIND] == ResidualKind.JOINT
        ]
        assert sorted(bodies) == sorted(BodyTrace.from_file(TRACE).poses)
        assert joints == [SHOULDER, SLIDE]
        assert all(row[ResidualColumn.WITHIN_TOLERANCE] == "True" for row in rows)

    def test_report_summarises_the_animation(self, bundle):
        animation = load(bundle, BundleFile.REPORT)[BundleKey.ANIMATION]
        assert animation[BundleKey.FRAME_COUNT] == 7
        assert animation[BundleKey.SOURCE] == str(TRACE)
        assert animation[BundleKey.BEYOND_TOLERANCE] == []
        assert animation[PresentationField.OBJECTS] == ["Puck"]


def test_time_samples_animate_the_stage_without_a_trace(tmp_path, stage):
    bundle = convert(tmp_path, "arm", ImportOption.FRAMES_PER_SECOND, "1")
    trajectory = load(bundle, BundleFile.TRAJECTORY)
    description = StageReader(ARM, variant_selections=[PHYSICS]).read()
    shoulder = joint_name(description, SHOULDER)
    assert [frame[shoulder] for frame in trajectory[BundleKey.FRAMES]] == pytest.approx(
        [
            numpy.radians(
                authored(stage, "/World/Arm/Base/Upper", "xformOp:rotateZ", seconds)
            )
            for seconds in range(4)
        ],
        abs=1e-6,
    )
    assert load(bundle, BundleFile.REPORT)[BundleKey.ANIMATION][
        BundleKey.SOURCE
    ] == str(ARM)


def test_frame_rate_defaults_to_thirty(tmp_path):
    bundle = convert(tmp_path, "arm")
    trajectory = load(bundle, BundleFile.TRAJECTORY)
    assert trajectory[BundleKey.FRAMES_PER_SECOND] == 30.0
    assert len(trajectory[BundleKey.FRAMES]) == 3 * 30 + 1
