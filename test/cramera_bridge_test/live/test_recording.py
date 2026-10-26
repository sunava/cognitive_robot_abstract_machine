"""
Recording a run, writing it as a bundle, and saving, trimming or discarding it.
"""

from __future__ import annotations

import pytest
from typing_extensions import List

from cramera_bridge.documents import DocumentFile
from cramera_bridge.exceptions import (
    EmptyRecording,
    InvalidFrameRange,
    NoRecording,
    SceneNameTaken,
    SharedScenesUnavailable,
)
from cramera_bridge.live.recording import FrameRange
from cramera_bridge.live.recording_storage import RecordingStorage
from cramera_bridge.robots import RobotRoster
from cramera_protocol.bundle.files import BundleFile
from cramera_protocol.bundle.index import (
    InvalidSceneName,
    ReservedSceneName,
    SceneIndex,
)
from cramera_protocol.bundle.files import ScenesDirectoryFile
from cramera_protocol.bundle.scene import PlanNode, PlanNodeKind, SceneDescription
from cramera_protocol.bundle.statecharts import Statecharts
from cramera_protocol.bundle.trajectory import Trajectory
from cramera_protocol.live_messages import (
    ConstraintActivation,
    ConstraintMonitorType,
    ConstraintRequest,
    MonitorSpecification,
    PlanNodeReference,
    RecordingSaveRequest,
    RecordingState,
    SceneDestination,
)
from cramera.locations import DataLocations
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from ..dataset.live_worlds import HeightChart

FRAME_COUNT = 6
"""
How many frames the recorded run has besides the one taken when it was attached.
"""

# %% helpers


def carry_milk(kitchen, publisher) -> None:
    """
    Move the milk away and record a frame per step, with a statechart while moving.
    """
    chart = HeightChart.build(kitchen.world, kitchen.milk)
    for step in range(FRAME_COUNT):
        kitchen.milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            1.0 + 0.1 * step, 0.1, 0.9, reference_frame=kitchen.root
        )
        publisher.observe_motion_tick(chart.chart)
        publisher.motion_running = True
        publisher.observe_state_change()


def bundle_files(locations: DataLocations):
    """
    :return: The trajectory and scene of the ``__recording__`` bundle.
    """
    bundle = locations.local_scenes / ReservedSceneName.RECORDING
    return (
        DocumentFile(bundle / BundleFile.TRAJECTORY).read(Trajectory),
        DocumentFile(bundle / BundleFile.SCENE).read(SceneDescription),
    )


# %% recording


def test_recorded_frames_are_copies(publisher, kitchen):
    carry_milk(kitchen, publisher)
    first = publisher.recording.frames[0].objects
    kitchen.milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        5.0, 5.0, 5.0, reference_frame=kitchen.root
    )
    publisher.observe_state_change()
    assert publisher.recording.frames[0].objects == first


def test_finishing_writes_the_recording_bundle(publisher, kitchen, locations):
    carry_milk(kitchen, publisher)
    status = publisher.finish_recording()
    trajectory, scene = bundle_files(locations)
    assert status.state is RecordingState.FINALIZED
    assert status.scene_name == ReservedSceneName.RECORDING
    assert trajectory.frame_count() == status.frame_count
    assert [entry.key for entry in scene.objects] == [
        str(body.id) for body in publisher.model.objects
    ]
    assert scene.segments[0].start == 0
    assert any(segment.picks == str(kitchen.milk.id) for segment in scene.segments)


def test_the_statechart_of_each_frame_is_recorded(publisher, kitchen, locations):
    carry_milk(kitchen, publisher)
    publisher.finish_recording()
    statecharts = DocumentFile(
        locations.local_scenes / ReservedSceneName.RECORDING / BundleFile.STATECHARTS
    ).read(Statecharts)
    assert len(statecharts.charts) == 1
    assert len(statecharts.frames) == publisher.recording.frame_count()


def test_finishing_twice_reports_the_same_bundle(publisher, kitchen):
    carry_milk(kitchen, publisher)
    assert publisher.finish_recording() == publisher.finish_recording()


def test_an_empty_recording_is_not_written(publisher):
    publisher.recording.start()
    with pytest.raises(EmptyRecording):
        publisher.finish_recording()


def test_nothing_recorded_cannot_be_finished(publisher):
    publisher.recording.discard()
    with pytest.raises(NoRecording):
        publisher.finish_recording()


# %% saving


def test_a_saved_recording_is_trimmed_and_indexed(publisher, kitchen, locations):
    carry_milk(kitchen, publisher)
    publisher.finish_recording()
    saved = publisher.save_recording(
        RecordingSaveRequest(name="carry", first_frame=2, last_frame=4, task="carry")
    )
    bundle = locations.local_scenes / saved.scene
    trajectory = DocumentFile(bundle / BundleFile.TRAJECTORY).read(Trajectory)
    scene = DocumentFile(bundle / BundleFile.SCENE).read(SceneDescription)
    index = DocumentFile(locations.local_scenes / ScenesDirectoryFile.INDEX).read(
        SceneIndex
    )
    assert trajectory.frame_count() == FrameRange(2, 4).length()
    assert scene.name == saved.scene
    assert scene.task == "carry"
    assert scene.objects[0].spawn == trajectory.objects[0][scene.objects[0].key]
    assert [entry.name for entry in index.scenes] == [saved.scene]
    assert publisher.recording_status().state is RecordingState.IDLE
    assert not (locations.local_scenes / ReservedSceneName.RECORDING).exists()


def test_a_save_to_the_shared_directory(publisher, kitchen, locations):
    carry_milk(kitchen, publisher)
    publisher.finish_recording()
    publisher.save_recording(
        RecordingSaveRequest(name="shared_run", destination=SceneDestination.SHARED)
    )
    assert (locations.shared_scenes / "shared_run" / BundleFile.SCENE).is_file()


@pytest.mark.parametrize(
    "request_document, error",
    [
        (RecordingSaveRequest(name="no spaces", first_frame=1), InvalidSceneName),
        (RecordingSaveRequest(name="taken", first_frame=1), SceneNameTaken),
        (
            RecordingSaveRequest(name="late", first_frame=1, last_frame=99),
            InvalidFrameRange,
        ),
    ],
)
def test_a_refused_save_leaves_the_recording_untouched(
    publisher, kitchen, locations, request_document, error
):
    carry_milk(kitchen, publisher)
    publisher.finish_recording()
    (locations.local_scenes / "taken").mkdir(parents=True)
    before = bundle_files(locations)
    with pytest.raises(error):
        publisher.save_recording(request_document)
    assert bundle_files(locations) == before
    assert publisher.recording_status().state is RecordingState.FINALIZED


def test_a_shared_save_without_a_shared_directory_is_refused(kitchen, tmp_path):
    storage = RecordingStorage(DataLocations(data=tmp_path, shared_scenes=None))
    with pytest.raises(SharedScenesUnavailable):
        storage.save(
            RecordingSaveRequest(name="run", destination=SceneDestination.SHARED)
        )


def test_saving_before_finishing_is_refused(publisher, kitchen):
    carry_milk(kitchen, publisher)
    with pytest.raises(NoRecording):
        publisher.save_recording(RecordingSaveRequest(name="early"))


def test_discarding_drops_frames_and_bundle(publisher, kitchen, locations):
    carry_milk(kitchen, publisher)
    publisher.finish_recording()
    publisher.discard_recording()
    assert publisher.recording_status().frame_count == 0
    assert not (locations.local_scenes / ReservedSceneName.RECORDING).exists()


# %% plan steps


def run_the_small_plan(kitchen, publisher, small_plan) -> FrameRange:
    """
    Record the small plan's action before, while and after it runs, one carry each.

    :return: The frames recorded while the action ran.
    """
    publisher.begin_plan(small_plan.plan)
    carry_milk(kitchen, publisher)
    small_plan.action.status = LifeCycleValues.RUNNING
    first = publisher.recording.frame_count()
    carry_milk(kitchen, publisher)
    running = FrameRange(first, publisher.recording.frame_count() - 1)
    small_plan.action.status = LifeCycleValues.SUCCEEDED
    carry_milk(kitchen, publisher)
    return running


def saved_plan_root(publisher, locations, kept: FrameRange) -> PlanNode:
    """
    Save the finished recording's kept frames.

    :return: The root of the saved bundle's plan.
    """
    saved = publisher.save_recording(
        RecordingSaveRequest(name="steps", first_frame=kept.first, last_frame=kept.last)
    )
    scene = DocumentFile(locations.local_scenes / saved.scene / BundleFile.SCENE).read(
        SceneDescription
    )
    return scene.plan_trees[0]


def test_a_recorded_step_names_the_frames_its_action_ran_in(
    publisher, kitchen, small_plan, locations
):
    running = run_the_small_plan(kitchen, publisher, small_plan)
    publisher.finish_recording()
    root = bundle_files(locations)[1].plan_trees[0]
    action = root.children[0]
    assert action.kind == PlanNodeKind.ACTION
    assert (action.start, action.end) == (running.first, running.last)
    assert (root.start, root.end) == (running.first, running.last)


def test_a_recorded_step_keeps_its_parameters_and_constraints(
    publisher, kitchen, small_plan, locations
):
    publisher.begin_plan(small_plan.plan)
    publisher.attach_constraint(
        ConstraintRequest(
            target_plan_node=PlanNodeReference(
                identifier=publisher.plan.active_plan.identifier(small_plan.action)
            ),
            monitor=MonitorSpecification(
                monitor_type=ConstraintMonitorType.HEIGHT,
                parameters={"tip_link": kitchen.milk.name.name, "lower_limit": 0.5},
            ),
            text="keep the milk up",
            apply=ConstraintActivation.IMMEDIATE,
        )
    )
    live = publisher.refresh_plan().nodes[1].constraints[0]
    small_plan.action.status = LifeCycleValues.RUNNING
    carry_milk(kitchen, publisher)
    small_plan.action.status = LifeCycleValues.SUCCEEDED
    carry_milk(kitchen, publisher)
    publisher.finish_recording()
    root = bundle_files(locations)[1].plan_trees[0]
    action = root.children[0]
    assert action.parameters == {"torso_state": TorsoState.HIGH.name}
    assert [(each.text, each.monitor) for each in action.constraints] == [
        (live.text, live.monitor)
    ]
    assert (root.parameters, root.constraints) == ({}, [])


def test_a_node_that_never_was_the_running_action_has_no_frames(
    publisher, kitchen, small_plan, locations
):
    run_the_small_plan(kitchen, publisher, small_plan)
    publisher.finish_recording()
    motion = bundle_files(locations)[1].plan_trees[0].children[0].children[0]
    assert (motion.start, motion.end) == (None, None)


def test_a_saved_stretch_counts_step_frames_from_its_first_frame(
    publisher, kitchen, small_plan, locations
):
    running = run_the_small_plan(kitchen, publisher, small_plan)
    publisher.finish_recording()
    kept = FrameRange(running.first + 1, running.last + 2)
    action = saved_plan_root(publisher, locations, kept).children[0]
    assert (action.start, action.end) == (0, running.last - kept.first)


def test_a_step_outside_the_saved_stretch_has_no_frames(
    publisher, kitchen, small_plan, locations
):
    running = run_the_small_plan(kitchen, publisher, small_plan)
    publisher.finish_recording()
    kept = FrameRange(running.last + 1, running.last + 2)
    action = saved_plan_root(publisher, locations, kept).children[0]
    assert (action.start, action.end) == (None, None)


# %% plans of several robots


def all_nodes(nodes: List[PlanNode]) -> List[PlanNode]:
    """
    :return: The nodes and their descendants, parents before their children.
    """
    return [each for node in nodes for each in [node] + all_nodes(node.children)]


def test_a_recording_keeps_every_plan_with_the_robot_that_performed_it(
    publisher, kitchen, robot_plans, two_robots, locations
):
    for plan in robot_plans:
        run_the_small_plan(kitchen, publisher, plan)
    publisher.finish_recording()
    roots = bundle_files(locations)[1].plan_trees
    assert [root.robot for root in roots] == [
        RobotRoster.label(robot) for robot in two_robots.robots
    ]
    assert [root.children[0].robot for root in roots] == [None, None]


def test_each_plan_names_the_frames_its_own_steps_ran_in(
    publisher, kitchen, robot_plans, locations
):
    running = [run_the_small_plan(kitchen, publisher, plan) for plan in robot_plans]
    publisher.finish_recording()
    actions = [root.children[0] for root in bundle_files(locations)[1].plan_trees]
    assert [(action.start, action.end) for action in actions] == [
        (frames.first, frames.last) for frames in running
    ]


def test_the_nodes_of_two_plans_have_distinct_identities(
    publisher, kitchen, robot_plans, locations
):
    for plan in robot_plans:
        run_the_small_plan(kitchen, publisher, plan)
    publisher.finish_recording()
    identities = [
        node.identifier for node in all_nodes(bundle_files(locations)[1].plan_trees)
    ]
    assert len(set(identities)) == len(identities)
    assert len(identities) == sum(len(plan.plan.all_nodes) for plan in robot_plans)


def test_a_plan_begun_again_is_recorded_once(publisher, kitchen, small_plan, locations):
    run_the_small_plan(kitchen, publisher, small_plan)
    running = run_the_small_plan(kitchen, publisher, small_plan)
    publisher.finish_recording()
    [root] = bundle_files(locations)[1].plan_trees
    assert root.children[0].end == running.last


def test_a_plan_without_a_context_names_no_robot(
    publisher, kitchen, small_plan, locations
):
    run_the_small_plan(kitchen, publisher, small_plan)
    publisher.finish_recording()
    assert bundle_files(locations)[1].plan_trees[0].robot is None
