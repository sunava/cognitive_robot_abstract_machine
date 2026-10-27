"""
The live publisher: what a running CRAM world offers the viewer.

The simulation thread reports the world's state and model changes, the plan's progress
and every motion tick; each component takes its snapshot then. HTTP threads only read
those snapshots, and queue the changes a person makes, which are written on the thread
that owns the world.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import distributions
from pathlib import Path

from typing_extensions import TYPE_CHECKING, Callable, List, Optional

from cramera.locations import DataLocations
from cramera_bridge.exceptions import (
    EmptyRecording,
    MarkersUnavailable,
    NoRecording,
    NoWorldAttached,
    RobotSelectionBusy,
    TeleoperationUnavailable,
    UnknownPerceptionImage,
)
from cramera_bridge.live.activity import SimulationActivity
from cramera_bridge.live.bundles import LiveBundleWriter, WorldSignature
from cramera_bridge.live.constraints import (
    AttachedConstraint,
    ConstraintBoard,
    ConstraintMonitorFactory,
)
from cramera_bridge.live.manipulation import WorldEdits, WorldSurfaces
from cramera_bridge.live.marker_topics import MarkerTopicSource
from cramera_bridge.live.markers import MarkerBoard
from cramera_bridge.live.objects import ObjectCatalogPublisher
from cramera_bridge.live.perception_images import TakenImage
from cramera_bridge.live.plan import PlanPublisher
from cramera_bridge.live.recording import Recording
from cramera_bridge.live.recording_storage import RecordingStorage
from cramera_bridge.live.statechart import StatechartPublisher
from cramera_bridge.live.teleoperation import TeleoperationDriver
from cramera_bridge.live.transforms import TransformTracker
from cramera_bridge.live.world_model import WorldModelView
from cramera_bridge.live.world_state import WorldStatePublisher
from cramera_bridge.questions.live import WorldKnowledge, WorldQuestionDesk
from cramera_bridge.robots import RobotParts, RobotRoster
from cramera_protocol.live import Capability, LiveInfo
from cramera_protocol.live_messages import (
    Acknowledgement,
    ConstraintRequest,
    JointRequest,
    LivePlan,
    LiveScene,
    LiveState,
    LiveStatechart,
    MarkerOverlay,
    MarkerTopicRequest,
    MarkerTopics,
    MoveRequest,
    ObjectCatalog,
    ObjectPoses,
    RecordingSaveRequest,
    RecordingState,
    RecordingStatus,
    RobotRoster as RobotRosterDocument,
    RobotSelection,
    SavedScene,
    Surfaces,
    TeleoperationRequest,
    TransformGraph,
)
from cramera_protocol.questions import (
    Members,
    PresetCatalog,
    QueryAnswer,
    QueryRequest,
    QueryScope,
    QuestionMatch,
    QuestionRequest,
    Vocabulary,
)
from cramera_bridge.questions.desk import QuestionDesk
from cramera_bridge.questions.workspace_classes import WorkspaceClassIndex
from coraplex.plans.plan_node import MotionNode, PlanNode
from semantic_digital_twin.robots.robot_parts import AbstractRobot

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

DISTRIBUTION = "cramera-bridge"
"""
The distribution this package is installed as.
"""

SOURCE_VERSION = "1.0.0"
"""
The package's version when it runs from its source tree without being installed.
"""

PLAN_REFRESH_TICKS = 5
"""
How many motion ticks pass between two plan snapshots; walking the plan costs more than
a tick and its statuses change far less often.
"""

# %% publisher


@dataclass
class LivePublisher:
    """
    Composes the live components around one world.
    """

    locations: DataLocations
    """
    Where bundles are written.
    """

    class_index: WorkspaceClassIndex = field(default_factory=WorkspaceClassIndex)
    """
    The workspace classes a query may name.
    """

    marker_source: Optional[MarkerTopicSource] = None
    """
    Delivers ROS markers, when ROS is available.
    """

    clock: Callable[[], float] = time.monotonic
    """
    The monotonic clock transforms are aged with.
    """

    world: Optional[World] = field(default=None, init=False)
    """
    The attached world.
    """

    model: Optional[WorldModelView] = field(default=None, init=False)
    """
    The attached world's current model.
    """

    selected_robot: Optional[AbstractRobot] = field(default=None, init=False)
    """
    The robot the live controls act on.
    """

    world_lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    """
    Held while the world is read or written by the publisher; taken before any lock of a
    component.
    """

    activity: SimulationActivity = field(default_factory=SimulationActivity, init=False)
    """
    Whether a plan moves the world.
    """

    state: WorldStatePublisher = field(default_factory=WorldStatePublisher, init=False)
    """
    The state snapshots.
    """

    objects: ObjectCatalogPublisher = field(
        default_factory=ObjectCatalogPublisher, init=False
    )
    """
    The free objects and their meshes.
    """

    plan: PlanPublisher = field(default_factory=PlanPublisher, init=False)
    """
    The plan snapshots.
    """

    statechart: StatechartPublisher = field(
        default_factory=StatechartPublisher, init=False
    )
    """
    The statechart snapshots.
    """

    chart_title: str = field(default="", init=False)
    """
    The action whose motions the ticked statechart executes.
    """

    motion_running: bool = field(default=False, init=False)
    """
    Whether a motion runs, so a recorded frame shows the statechart.
    """

    ticks: int = field(default=0, init=False)
    """
    How many motion ticks happened.
    """

    transforms: TransformTracker = field(default_factory=TransformTracker, init=False)
    """
    When each connection changed.
    """

    markers: MarkerBoard = field(default_factory=MarkerBoard, init=False)
    """
    The debug markers.
    """

    edits: WorldEdits = field(default_factory=WorldEdits, init=False)
    """
    The changes a person made that wait to be written.
    """

    constraints: ConstraintBoard = field(default_factory=ConstraintBoard, init=False)
    """
    The constraints attached to plan nodes.
    """

    recording: Recording = field(default_factory=Recording, init=False)
    """
    The recording of the run.
    """

    teleoperation: Optional[TeleoperationDriver] = field(default=None, init=False)
    """
    The hand-control driver, once a person moved a hand.
    """

    desk: Optional[QuestionDesk] = field(default=None, init=False)
    """
    Answers questions about the world, once one is attached.
    """

    bundle_writer: LiveBundleWriter = field(init=False)
    """
    Writes the live and recording bundles.
    """

    def __post_init__(self) -> None:
        self.bundle_writer = LiveBundleWriter(self.locations.local_scenes)

    # %% simulation thread

    def attach(self, world: World) -> None:
        """
        Start publishing a world and recording its run.

        :param world: The world.
        """
        with self.world_lock:
            self.world = world
            self.model = None
            self.selected_robot = None
            self.desk = WorldQuestionDesk(
                world, self.current_objects, self.class_index
            ).desk()
            self.recording.start()
            self.observe_model_change()

    def current_objects(self) -> List[Body]:
        """
        :return: The free objects of the current model.
        """
        model = self.model
        return [] if model is None else list(model.objects)

    def observe_model_change(self) -> None:
        """
        Read the world's model again, keeping every object published before.
        """
        with self.world_lock:
            world = self.require_world()
            tracked = (
                [] if self.model is None else [body.id for body in self.model.objects]
            )
            self.model = WorldModelView.of_world(world, tracked)
            if self.selected_robot not in self.model.robots:
                self.selected_robot = (
                    self.model.robots[0] if self.model.robots else None
                )
            self.objects.refresh(self.model)
            self.desk.presets = WorldKnowledge(world, self.current_objects).presets()
            self.observe_state_change()

    def observe_state_change(self) -> None:
        """
        Take a state snapshot and record it.
        """
        with self.world_lock:
            if self.model is None:
                return
            self.markers.refresh(self.model.world)
            snapshot = self.state.refresh(
                self.model, self.selected_robot, self.markers.version()
            )
            self.transforms.observe(self.model.world, self.clock())
            self.recording.append(
                snapshot,
                self.plan.running(),
                self.statechart.current() if self.motion_running else None,
            )

    def begin_plan(self, plan: Plan) -> None:
        """
        Publish a plan that is about to perform, keeping the plans performed before for
        the recording; its robot becomes the selected one.

        :param plan: The plan.
        """
        with self.world_lock:
            if plan.context is not None and self.model is not None:
                robot = plan.context.robot
                if robot in self.model.robots:
                    self.select(robot)
            self.plan.begin(plan)
            self.constraints.clear()
            self.refresh_plan()

    def observe_node_started(self, node: PlanNode) -> None:
        """
        :param node: The plan node that started.
        """
        with self.world_lock:
            if isinstance(node, MotionNode):
                action = node.parent_action_node
                self.chart_title = (
                    type(action.designator).__name__ if action is not None else ""
                )
                self.motion_running = True
                self.constraints.note_motion_started()
            self.plan.observe_started(node)
            self.refresh_plan()

    def observe_node_ended(self, node: PlanNode) -> None:
        """
        :param node: The plan node that ended.
        """
        with self.world_lock:
            if isinstance(node, MotionNode):
                self.motion_running = False
            object_keys = set() if self.model is None else set(self.model.object_keys())
            self.plan.observe_ended(node, object_keys)
            self.refresh_plan()

    def observe_motion_tick(self, chart: MotionStatechart) -> None:
        """
        Write the waiting changes, evaluate the constraints and publish the statechart.

        :param chart: The statechart the executor ticks.
        """
        with self.world_lock:
            self.activity.tick()
            world = self.require_world()
            if self.edits.apply(self.model, self.transforms):
                world.notify_state_change()
            self.constraints.tick(world)
            self.statechart.observe(chart, self.chart_title)
            self.ticks += 1
            if self.ticks % PLAN_REFRESH_TICKS == 0:
                self.refresh_plan()

    def refresh_plan(self) -> LivePlan:
        """
        :return: A new plan snapshot.
        """
        object_keys = set() if self.model is None else set(self.model.object_keys())
        return self.plan.refresh(object_keys, self.constraints.observations)

    def require_world(self) -> World:
        """
        :return: The attached world.
        :raises NoWorldAttached: Before a world was attached.
        """
        if self.world is None:
            raise NoWorldAttached()
        return self.world

    def require_model(self) -> WorldModelView:
        """
        :return: The attached world's current model.
        :raises NoWorldAttached: Before a world was attached.
        """
        if self.model is None:
            raise NoWorldAttached()
        return self.model

    # %% information

    def capabilities(self) -> List[Capability]:
        """
        :return: What the publisher offers now.
        """
        offered = [
            Capability.PLAN,
            Capability.STATECHART,
            Capability.TRANSFORMS,
            Capability.QUESTIONS,
            Capability.RECORDING,
            Capability.CONSTRAINTS,
            Capability.OBJECT_PLACEMENT,
        ]
        if self.marker_source is not None:
            offered.append(Capability.MARKERS)
        model = self.model
        if model is not None and len(model.robots) > 1:
            offered.append(Capability.ROBOT_SELECTION)
        if (
            self.selected_robot is not None
            and RobotParts(model.world, self.selected_robot).arms()
        ):
            offered.append(Capability.TELEOPERATION)
        return offered

    def info(self) -> LiveInfo:
        """
        :return: What the publisher says about itself.
        """
        with self.world_lock:
            return LiveInfo(
                publisher=f"{DISTRIBUTION} {PublisherVersion().text()}",
                capabilities=self.capabilities(),
                running=self.world is not None,
                bundle_signature=(
                    None
                    if self.model is None
                    else WorldSignature(self.model, self.selected_robot).digest()
                ),
            )

    # %% snapshots

    def current_state(self) -> LiveState:
        """
        :return: The newest state snapshot.
        """
        return self.state.state()

    def current_plan(self) -> LivePlan:
        """
        :return: The newest plan snapshot.
        """
        return self.plan.current()

    def current_statechart(self) -> LiveStatechart:
        """
        :return: The newest statechart snapshot, empty before a chart was ticked.
        """
        return self.statechart.current() or LiveStatechart()

    def transform_graph(self) -> TransformGraph:
        """
        :return: Every connection with how recently it changed.
        """
        return self.transforms.graph(self.clock())

    def object_catalog(self) -> ObjectCatalog:
        """
        :return: The free objects with their shapes.
        """
        return self.objects.current()

    def perception_image(self, key: str) -> TakenImage:
        """
        :param key: The key of an image a perception step took.
        :return: The image.
        :raises UnknownPerceptionImage: When no step took an image under the key.
        """
        image = self.plan.images.image(key)
        if image is None:
            raise UnknownPerceptionImage(key)
        return image

    def mesh_file(self, key: str, side: Optional[str]) -> Path:
        """
        :param key: The key of a served mesh.
        :param side: A file beside the mesh, or None for the mesh.
        :return: The file.
        """
        return self.objects.mesh_file(key, side)

    def marker_overlay(self) -> MarkerOverlay:
        """
        :return: The debug markers posed in the world.
        """
        return self.markers.current()

    def marker_topics(self) -> MarkerTopics:
        """
        :return: The marker topics read and on offer; none without ROS.
        """
        if self.marker_source is None:
            return MarkerTopics(ros=False)
        return self.marker_source.topics()

    def change_marker_topic(self, request: MarkerTopicRequest) -> MarkerTopics:
        """
        :param request: The topic to read or stop reading.
        :return: The topics read and on offer afterwards.
        :raises MarkersUnavailable: Without ROS.
        """
        if self.marker_source is None:
            raise MarkersUnavailable()
        if request.subscribed:
            self.marker_source.subscribe(request.topic)
        else:
            self.marker_source.unsubscribe(request.topic)
        return self.marker_topics()

    def captured_objects(self) -> ObjectPoses:
        """
        :return: Every object's pose including moves not yet written.
        """
        return self.edits.captured(self.state.state())

    def surfaces(self) -> Surfaces:
        """
        :return: The annotated surfaces of the world.
        """
        with self.world_lock:
            return WorldSurfaces(self.require_world()).surfaces()

    def live_scene(self) -> LiveScene:
        """
        :return: The bundle showing the world's current model, written when the model
            changed; no bundle before a world is attached.
        """
        with self.world_lock:
            if self.model is None:
                return LiveScene()
            return LiveScene(
                scene=self.bundle_writer.live_scene(self.model, self.selected_robot)
            )

    # %% robots

    def robots(self) -> RobotRosterDocument:
        """
        :return: The robots and the selected one.
        """
        with self.world_lock:
            if self.model is None:
                return RobotRosterDocument()
            roster = RobotRoster(self.model.world)
            return RobotRosterDocument(
                robots=[roster.instance(robot) for robot in self.model.robots],
                active_identifier=(
                    None
                    if self.selected_robot is None
                    else RobotRoster.identifier(self.selected_robot)
                ),
            )

    def select_robot(self, selection: RobotSelection) -> RobotRosterDocument:
        """
        :param selection: The robot to select.
        :return: The robots and the selected one afterwards.
        :raises RobotSelectionBusy: While the plan runs or is paused.
        :raises UnknownRobot: When no robot has the identity.
        """
        if self.plan.is_executing():
            raise RobotSelectionBusy()
        with self.world_lock:
            self.select(RobotRoster(self.require_world()).named(selection.identifier))
        return self.robots()

    def select(self, robot: AbstractRobot) -> None:
        """
        Select a robot, stopping hand control of the previous one.

        :param robot: A robot of the world.
        """
        if robot is self.selected_robot:
            return
        self.stop_teleoperation()
        self.teleoperation = None
        self.selected_robot = robot
        self.observe_state_change()

    # %% questions

    def require_desk(self) -> QuestionDesk:
        """
        :return: The desk answering questions about the world.
        :raises NoWorldAttached: Before a world was attached.
        """
        if self.desk is None:
            raise NoWorldAttached()
        return self.desk

    def presets(self) -> PresetCatalog:
        """
        :return: The ready-made questions about the world.
        """
        with self.world_lock:
            return self.require_desk().preset_catalog()

    def vocabulary(self, scope: QueryScope) -> Vocabulary:
        """
        :param scope: A body of knowledge.
        :return: The names a query of it may use.
        """
        return self.require_desk().vocabulary(scope)

    def members(self, scope: QueryScope, name: str) -> Members:
        """
        :param scope: A body of knowledge.
        :param name: A name a query may use.
        :return: The members following the name's dot.
        """
        return self.require_desk().members(scope, name)

    def answer_query(self, request: QueryRequest) -> QueryAnswer:
        """
        Answer a query while the world is held still.

        :param request: The query.
        :return: The answer.
        """
        with self.world_lock:
            return self.require_desk().answer_query(request)

    def answer_question(self, request: QuestionRequest) -> QuestionMatch:
        """
        :param request: A plain-language question.
        :return: The ready-made question it matches.
        """
        with self.world_lock:
            return self.require_desk().answer_question(request)

    # %% changing the world

    def move(self, request: MoveRequest) -> Acknowledgement:
        """
        Queue an object move; while no plan moves the world it is written at once.

        :param request: The move.
        :return: The acknowledgement.
        """
        with self.world_lock:
            self.edits.queue_move(request, self.require_model())
        self.write_edits_while_idle()
        return Acknowledgement()

    def set_joint(self, request: JointRequest) -> Acknowledgement:
        """
        Queue a joint value; while no plan moves the world it is written at once.

        :param request: The joint value.
        :return: The acknowledgement.
        """
        with self.world_lock:
            self.edits.queue_joint(request, self.require_model())
        self.write_edits_while_idle()
        return Acknowledgement()

    def write_edits_while_idle(self) -> None:
        """
        Write waiting changes on the calling thread unless a plan moves the world, in
        which case the next motion tick writes them.
        """
        if self.activity.is_busy():
            return
        with self.world_lock:
            if self.edits.apply(self.require_model(), self.transforms):
                self.require_world().notify_state_change()

    def attach_constraint(self, request: ConstraintRequest) -> LivePlan:
        """
        Attach a constraint monitor to a plan node.

        :param request: The constraint.
        :return: The plan with the new constraint.
        :raises UnknownPlanNode: When the plan has no such node.
        """
        identifier = request.target_plan_node.identifier
        self.plan.published_node(identifier)
        with self.world_lock:
            monitor = ConstraintMonitorFactory(self.require_world()).monitor(request)
        self.constraints.attach(identifier, AttachedConstraint(request, monitor))
        return self.plan.with_constraints(self.constraints.observations)

    def teleoperate(self, request: TeleoperationRequest) -> Acknowledgement:
        """
        :param request: A hand target of the selected robot.
        :return: The acknowledgement.
        :raises TeleoperationUnavailable: Without a selected robot.
        """
        with self.world_lock:
            if self.selected_robot is None:
                raise TeleoperationUnavailable("no robot is selected")
            if self.teleoperation is None:
                self.teleoperation = TeleoperationDriver(
                    self.require_world(),
                    self.selected_robot,
                    self.activity,
                    self.world_lock,
                )
            driver = self.teleoperation
        driver.submit(request)
        return Acknowledgement()

    def stop_teleoperation(self) -> Acknowledgement:
        """
        Stop hand control; the arms keep their poses.

        :return: The acknowledgement.
        """
        if self.teleoperation is not None:
            self.teleoperation.stop()
        return Acknowledgement()

    # %% recording

    def recording_status(self) -> RecordingStatus:
        """
        :return: The recording's state.
        """
        return self.recording.status()

    def finish_recording(self) -> RecordingStatus:
        """
        End the recording and write it as the ``__recording__`` bundle; finishing it
        again reports the same bundle.

        :return: The recording's state, naming the bundle.
        :raises NoRecording: When nothing was recorded.
        :raises EmptyRecording: When the recording has no frames.
        """
        frames = self.recording.finish()
        if self.recording.status().scene_name is not None:
            return self.recording.status()
        if not frames:
            raise EmptyRecording()
        with self.world_lock:
            scene = self.bundle_writer.recording(
                self.require_model(),
                self.selected_robot,
                frames,
                self.plan.performed_plans(),
            )
        self.recording.written(scene)
        return self.recording.status()

    def save_recording(self, request: RecordingSaveRequest) -> SavedScene:
        """
        Save the finished recording; a refused save changes nothing.

        :param request: The save request.
        :return: The saved bundle.
        :raises NoRecording: While no finished recording waits to be saved.
        """
        if self.recording.status().state is not RecordingState.FINALIZED:
            raise NoRecording(RecordingState.FINALIZED)
        saved = RecordingStorage(self.locations).save(request)
        self.recording.discard()
        return saved

    def discard_recording(self) -> Acknowledgement:
        """
        Drop the recording and its unsaved bundle.

        :return: The acknowledgement.
        """
        self.recording.discard()
        RecordingStorage(self.locations).discard()
        return Acknowledgement()


# %% version


@dataclass(frozen=True)
class PublisherVersion:
    """
    The version the publisher reports.
    """

    def text(self) -> str:
        """
        :return: The installed distribution's version, or the source tree's version
            when the package runs uninstalled.
        """
        installed = {
            distribution.metadata["Name"]: distribution.version
            for distribution in distributions()
        }
        return installed.get(DISTRIBUTION, SOURCE_VERSION)
